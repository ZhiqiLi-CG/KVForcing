"""KV Forcing: a one-pass autoregressive training rollout whose KV memory is
differentiable end to end.

Each generated block runs two forwards that carry gradient:

  noisy   the block at its sampled training timestep, attending to the pages of
          every earlier block. Its prediction is what the loss sees.
  write   the block's (noised) prediction at the context timestep, attending to
          the same pages. It returns this block's K/V as new pages.

Pages are immutable (wan/modules/causal_model.py, PersistentKVStore). Nothing
is detached, so block i's loss reaches block j < i through the page chain; and
because a page never changes, every block forward can be activation-
checkpointed and recomputed at backward time against exactly the history it saw
at forward time. A KV cache that is overwritten in place cannot offer that,
which is why one-pass Self Forcing has to stop the gradient at the cache
(pipeline/self_forcing_training.py) and why its differentiable variant keeps
every activation alive.

The rollout itself (exit-step draw, schedule, context noising, what is written)
is the one Self Forcing runs.
"""
from typing import Optional

import torch
import torch.distributed as dist

from wan.modules.causal_model import PersistentKVStore


class KVForcingTrainingPipeline:
    def __init__(
        self,
        denoising_step_list,
        scheduler,
        generator,
        num_frame_per_block: int = 1,
        last_step_only: bool = False,
        num_max_frames: int = 21,
        context_noise: int = 0,
        per_rank_exit_step: bool = True,
        match_context: bool = True,
        cache_mode: str = "exit",
        gradient_checkpointing: bool = True,
        checkpoint_group: int = 1,
        **kwargs,
    ):
        if cache_mode not in ("exit", "final_clean"):
            raise ValueError(f"cache_mode must be 'exit' or 'final_clean'. Got {cache_mode!r}.")
        if num_frame_per_block != 1:
            raise ValueError(
                "KV Forcing is implemented and verified for num_frame_per_block=1; got "
                f"{num_frame_per_block}. A block is written as one page, so intra-block "
                "visibility for chunkwise models is not defined here.")
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]
        self.scheduler = scheduler
        self.generator = generator
        self.num_frame_per_block = num_frame_per_block
        self.num_max_frames = num_max_frames
        self.context_noise = context_noise
        self.last_step_only = last_step_only
        self.per_rank_exit_step = per_rank_exit_step
        self.match_context = match_context
        self.cache_mode = cache_mode
        self.gradient_checkpointing = gradient_checkpointing
        # blocks per activation-checkpoint region; larger keeps fewer block
        # outputs alive between forward and backward (see CausalWanModel._persistent_blocks)
        self.checkpoint_group = int(checkpoint_group)

    @property
    def _model(self):
        m = self.generator.model
        return getattr(m, "module", m)   # unwrap FSDP if the model is its own unit

    # Both forwards go through the generator wrapper, never through the model
    # directly: under FSDP the wrapper is the root module, and its forward is
    # where the root's own parameters are all-gathered and cast.
    def _noisy(self, latent, timestep, cond, hist, start):
        _, x0 = self.generator(
            noisy_image_or_video=latent, conditional_dict=cond, timestep=timestep,
            persistent_mode="noisy", persistent_history=hist, persistent_start=start)
        return x0

    def _write(self, latent, timestep, cond, hist, start):
        return self.generator(
            noisy_image_or_video=latent, conditional_dict=cond, timestep=timestep,
            persistent_mode="write", persistent_history=hist, persistent_start=start)

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                indices = torch.randint(low=0, high=num_denoising_steps, size=(num_blocks,), device=device)
                if self.last_step_only:
                    indices = torch.ones_like(indices) * (num_denoising_steps - 1)
            else:
                indices = torch.empty(num_blocks, dtype=torch.long, device=device)
            dist.broadcast(indices, src=0)
            return indices.tolist()
        indices = torch.randint(low=0, high=num_denoising_steps, size=(num_blocks,), device=device)
        if self.last_step_only:
            indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        return indices.tolist()

    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        clean_image_or_video: torch.Tensor = None,
        initial_latent: Optional[torch.Tensor] = None,
        return_sim_step: bool = False,
        **conditional_dict,
    ):
        assert initial_latent is None, "KV Forcing training assumes text-to-video inputs."
        batch_size, num_frames = noise.shape[:2]
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block
        device = noise.device
        model = self._model

        # The persistent forwards attend to every page, i.e. to the whole
        # history. That equals the model's (sink, window) attention only while
        # sink + window covers the clip, so anything longer is refused.
        local, sink = getattr(model, "local_attn_size", -1), getattr(model, "sink_size", 0)
        if local != -1 and num_frames > local + sink:
            raise ValueError(
                f"KV Forcing attends to all history, which matches local_attn_size={local} + "
                f"sink_size={sink} only up to {local + sink} frames; got {num_frames}.")

        num_denoising_steps = len(self.denoising_step_list)
        if self.cache_mode == "exit" and self.per_rank_exit_step:
            idx = int(torch.randint(low=0, high=num_denoising_steps, size=(1,), device=device).item())
            if self.last_step_only:
                idx = num_denoising_steps - 1
        else:
            idx = int(self.generate_and_sync_list(1, num_denoising_steps, device)[0])

        model.persistent_block_checkpointing = self.gradient_checkpointing
        model.persistent_checkpoint_group = self.checkpoint_group
        store = PersistentKVStore(len(model.blocks))
        outputs = []
        start = 0
        for _ in range(num_blocks):
            n = self.num_frame_per_block
            hist = store.snapshot_all_layers()

            # The rollout, exactly as at inference. The exit-step forward is
            # the one the loss sees, so it is the only one run with gradient
            # (when the caller allows it): its prediction is both the output
            # and, detached, the rollout's x0 at that step. Under no_grad
            # (DMD's critic step) the rollout alone gives the output.
            noisy_input = noise[:, start:start + n]
            x0_exit = None
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones([batch_size, n], device=device, dtype=torch.int64) * current_timestep
                if index == idx:
                    out = self._noisy(noisy_input, timestep, conditional_dict, hist, start)
                    outputs.append(out)
                    x0 = x0_exit = out.detach()
                    if self.cache_mode == "exit" and not self.per_rank_exit_step:
                        break
                else:
                    with torch.no_grad():
                        x0 = self._noisy(noisy_input, timestep, conditional_dict, hist, start)
                if index < num_denoising_steps - 1:
                    with torch.no_grad():
                        noisy_input = self.scheduler.add_noise(
                            x0.flatten(0, 1), torch.randn_like(x0.flatten(0, 1)),
                            self.denoising_step_list[index + 1] * torch.ones(
                                [batch_size * n], device=device, dtype=torch.long),
                        ).unflatten(0, x0.shape[:2])
            with torch.no_grad():
                x0_cache = x0 if self.cache_mode == "final_clean" else x0_exit
                context_timestep = torch.ones_like(timestep) * self.context_noise
                x_ctx = self.scheduler.add_noise(
                    x0_cache.flatten(0, 1), torch.randn_like(x0_cache.flatten(0, 1)),
                    context_timestep.flatten(0, 1).long(),
                ).unflatten(0, x0_cache.shape[:2])

            # The memory write that becomes this block's pages.
            if self.match_context:
                write_x, write_t = x_ctx, context_timestep
            else:
                write_x, write_t = x0_cache, torch.zeros_like(context_timestep)
            store.append_all_layers(self._write(write_x, write_t, conditional_dict, hist, start))
            start += n

        output = torch.cat(outputs, dim=1)

        # The timestep window the sampled exit step corresponds to, for DMD.
        ts = self.scheduler.timesteps.to(device)
        steps = self.denoising_step_list

        def nearest(t):
            return 1000 - torch.argmin((ts - t.to(device)).abs(), dim=0).item()
        denoised_timestep_from = nearest(steps[idx])
        denoised_timestep_to = 0 if idx == num_denoising_steps - 1 else nearest(steps[idx + 1])

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, idx + 1
        return output, denoised_timestep_from, denoised_timestep_to
