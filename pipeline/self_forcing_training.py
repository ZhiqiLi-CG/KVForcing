"""One-pass Self Forcing training rollouts, with and without the stop-gradient
on the KV cache. Both are baselines for KV Forcing (pipeline/kv_forcing_training.py)
and share its rollout exactly: same exit-step draw, same schedule, same context
noising, same cache writes.

  self_forcing        the gradient is taken at the exit step of each block; the
                      cache write is stop-gradiented, so the loss never reaches
                      how a generated frame entered the KV cache.
  self_forcing_no_sg  the cache write keeps its graph, so the loss backpropagates
                      through the in-place KV writes into earlier blocks. This is
                      full BPTT through a mutable cache: its memory grows with the
                      rollout because nothing can be recomputed against a buffer
                      that later frames overwrite -- the problem KV Forcing solves
                      with immutable pages.
"""
from typing import List

import torch
import torch.distributed as dist

from utils.scheduler import SchedulerInterface
from utils.wan_wrapper import WanDiffusionWrapper


class SelfForcingTrainingPipeline:
    GRAD_MODES = ("self_forcing", "self_forcing_no_sg")

    def __init__(
        self,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: WanDiffusionWrapper,
        num_frame_per_block: int = 1,
        last_step_only: bool = False,
        num_max_frames: int = 21,
        context_noise: int = 0,
        grad_mode: str = "self_forcing",
        per_rank_exit_step: bool = True,
        cache_mode: str = "exit",
        **kwargs,
    ):
        if grad_mode not in self.GRAD_MODES:
            raise ValueError(f"grad_mode must be one of {self.GRAD_MODES}. Got {grad_mode!r}.")
        if cache_mode not in ("exit", "final_clean"):
            raise ValueError(f"cache_mode must be 'exit' or 'final_clean'. Got {cache_mode!r}.")
        if grad_mode == "self_forcing_no_sg" and cache_mode == "final_clean":
            raise ValueError(
                "self_forcing_no_sg needs cache_mode='exit': with 'final_clean' the cached "
                "prediction comes from the last denoising step, which runs under no_grad "
                "whenever the exit step is not the last one, so the cache write would carry "
                "no graph and the variant would silently degenerate into self_forcing.")
        self.grad_mode = grad_mode
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]
        # Wan specific hyperparameters
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.kv_cache1 = None
        self.crossattn_cache = None
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length
        self.per_rank_exit_step = per_rank_exit_step
        self.cache_mode = cache_mode

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

    def _sample_exit_step(self, device):
        num_denoising_steps = len(self.denoising_step_list)
        if self.cache_mode == "exit" and self.per_rank_exit_step:
            return int(torch.randint(low=0, high=num_denoising_steps, size=(1,), device=device).item())
        return int(self.generate_and_sync_list(1, num_denoising_steps, device=device)[0])

    def _denoised_timestep_range(self, idx, device):
        """The timestep window the sampled exit step corresponds to, for DMD."""
        ts = self.scheduler.timesteps.to(device)
        steps = self.denoising_step_list

        def nearest(t):
            return 1000 - torch.argmin((ts - t.to(device)).abs(), dim=0).item()
        if idx == len(steps) - 1:
            return nearest(steps[idx]), 0
        return nearest(steps[idx]), nearest(steps[idx + 1])

    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        clean_image_or_video: torch.Tensor = None,
        initial_latent=None,
        return_sim_step: bool = False,
        **conditional_dict,
    ):
        assert initial_latent is None, "Self Forcing training assumes text-to-video inputs."
        stop_gradient = self.grad_mode == "self_forcing"
        # Respect the caller's grad mode: critic_loss runs the generator inside
        # torch.no_grad(), and set_grad_enabled(True) would override that.
        grad_allowed = torch.is_grad_enabled()
        batch_size, num_frames = noise.shape[:2]
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block
        device, dtype = noise.device, noise.dtype

        self._initialize_kv_cache(batch_size=batch_size, dtype=dtype, device=device)
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=dtype, device=device)

        num_denoising_steps = len(self.denoising_step_list)
        idx = self._sample_exit_step(device)
        output = torch.zeros_like(noise)

        current_start_frame = 0
        for _ in range(num_blocks):
            n = self.num_frame_per_block
            noisy_input = noise[:, current_start_frame:current_start_frame + n]
            x0_exit = None

            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones([batch_size, n], device=device, dtype=torch.int64) * current_timestep
                # The exit step is the only differentiated denoising forward.
                with torch.set_grad_enabled(grad_allowed and index == idx):
                    _, x0 = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                if index == idx:
                    x0_exit = x0
                    if self.cache_mode == "exit" and not self.per_rank_exit_step:
                        break
                if index < num_denoising_steps - 1:
                    # Past the exit step the schedule must not extend the graph.
                    x0_next = x0.detach()
                    noisy_input = self.scheduler.add_noise(
                        x0_next.flatten(0, 1),
                        torch.randn_like(x0_next.flatten(0, 1)),
                        self.denoising_step_list[index + 1] * torch.ones(
                            [batch_size * n], device=device, dtype=torch.long),
                    ).unflatten(0, x0_next.shape[:2])

            x0_cache = x0 if self.cache_mode == "final_clean" else x0_exit
            output[:, current_start_frame:current_start_frame + n] = x0_exit

            # The KV write: stop-gradiented (self_forcing) or differentiable
            # (self_forcing_no_sg).
            x0_ctx = x0_cache.detach() if stop_gradient else x0_cache
            if not stop_gradient and torch.is_grad_enabled():
                assert x0_ctx.requires_grad, (
                    "self_forcing_no_sg lost the cache gradient: the context fed to the KV "
                    "write carries no graph, so this run is indistinguishable from self_forcing.")
            context_timestep = torch.ones_like(timestep) * self.context_noise
            x_ctx = self.scheduler.add_noise(
                x0_ctx.flatten(0, 1),
                torch.randn_like(x0_ctx.flatten(0, 1)),
                context_timestep.flatten(0, 1).long(),
            ).unflatten(0, x0_ctx.shape[:2])
            with torch.set_grad_enabled(grad_allowed and not stop_gradient):
                self.generator(
                    noisy_image_or_video=x_ctx,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    memory_lora_write=True,   # the context write is the memory write
                )
            if stop_gradient:
                # The exit-step forward wrote its K/V into the cache in place and
                # with grad. Left attached, that would chain every block's graph
                # into the next -- full BPTT, not Self Forcing's frozen cache.
                # detach() shares storage, so this costs no memory.
                for entry in self.kv_cache1:
                    entry["k"] = entry["k"].detach()
                    entry["v"] = entry["v"].detach()
            current_start_frame += n

        denoised_timestep_from, denoised_timestep_to = self._denoised_timestep_range(idx, device)
        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, idx + 1
        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        self.kv_cache1 = [{
            "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        } for _ in range(self.num_transformer_blocks)]

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        self.crossattn_cache = [{
            "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
            "is_init": False,
        } for _ in range(self.num_transformer_blocks)]
