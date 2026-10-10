import gc
import json
import logging
import random
import shutil
import numpy as np
from utils.dataset import cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
from model import DMD
import torch
import wandb
import time
import os


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=False
        )

        # ======== ours ========
        # The frozen teacher (real_score) may run in bf16 while everything that is
        # trained stays fp32: it has no gradients, so no accumulation precision is
        # at stake, and it is by far the largest model. It casts its own latents
        # and text embeddings (WanDiffusionWrapper.input_dtype) and keeps the
        # timestep in fp32: FSDP's input cast would round the timestep to bf16
        # (795.78 -> 796) for the teacher only, so the DMD gradient would compare
        # the critic and the teacher at two different timesteps.
        teacher_bf16 = bool(getattr(config, "real_score_bf16", False)) and not config.mixed_precision
        if teacher_bf16:
            self.model.real_score.to(torch.bfloat16)
            self.model.real_score.input_dtype = torch.bfloat16
        # ==== ours end ====
        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            # ======== ours ========
            mixed_precision=config.mixed_precision or teacher_bf16,
            cast_root_forward_inputs=not teacher_bf16,
            # ==== ours end ====
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        # ======== ours ========
        # Memory-LoRA adapters start at zero and need their own lr
        # (lr_memory_lora) to move; models without them take a single group.
        # fsdp_wrap uses use_orig_params=True, so parameter names survive.
        gen_params = [p for p in self.model.generator.parameters() if p.requires_grad]
        lora_params = [p for n, p in self.model.generator.named_parameters()
                       if p.requires_grad and "memory_lora" in n]
        if lora_params:
            lora_ids = {id(p) for p in lora_params}
            param_groups = [{"params": [p for p in gen_params if id(p) not in lora_ids], "lr": config.lr},
                            {"params": lora_params, "lr": float(getattr(config, "lr_memory_lora", config.lr))}]
            if self.is_main_process:
                print(f"generator optimizer: {len(param_groups[0]['params'])} backbone tensors at lr={config.lr}, "
                      f"{len(lora_params)} memory-adapter tensors at lr={param_groups[1]['lr']}", flush=True)
        else:
            param_groups = [{"params": gen_params, "lr": config.lr}]
        # ==== ours end ====
        self.generator_optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
                fixed = {}
                for k, v in state_dict.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "generator_ema" in state_dict:
                gen_sd = state_dict["generator_ema"]
                fixed = {}
                for k, v in gen_sd.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )

        ##############################################################################################################

        # ======== ours ========
        # Full-state resume for preemptible partitions. With resume_every > 0 every rank
        # writes, every resume_every steps, all that the next step depends on (its shards
        # of the generator and the critic, both optimizers, the EMA, the RNG streams and
        # the number of batches drawn) to resume_dir, and a restart continues from the
        # newest complete state. With resume_every == 0 (default) nothing here runs.
        self.resume_every = max(0, int(getattr(config, "resume_every", 0) or 0))
        self.resume_dir = str(getattr(config, "resume_dir", "") or "") or os.path.join(str(config.logdir), "resume")
        self._batches_consumed = 0
        if self.resume_every > 0:
            self._load_resume_state()
        # ==== ours end ====

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None
        self.step_time_log_interval = max(
            0, int(getattr(config, "step_time_log_interval", 0) or 0))

    # ======== ours ========
    def _next_batch(self):
        self._batches_consumed += 1
        return next(self.dataloader)

    @torch.no_grad()
    def save_resume_state(self):
        """Write this rank's part of the training state after `self.step` steps.

        Parameters and optimizer state are saved as the rank's own FSDP shards
        (use_orig_params=True keeps the original parameters as views of them), so a
        state can only be loaded with the same number of ranks and wrap policy.
        The new state is written next to the previous one and swapped in only when
        every rank has finished, so an interruption at any point leaves a complete state."""
        tmp, old = self.resume_dir + ".tmp", self.resume_dir + ".old"
        if self.is_main_process:
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp, exist_ok=True)
        dist.barrier()
        rank = dist.get_rank()
        state = {
            "step": self.step, "world_size": self.world_size, "rank": rank,
            "batches_consumed": self._batches_consumed,
            "generator": {n: p.detach().cpu().clone() for n, p in self.model.generator.named_parameters()},
            "critic": {n: p.detach().cpu().clone() for n, p in self.model.fake_score.named_parameters()},
            "generator_optimizer": self.generator_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "generator_ema": None if self.generator_ema is None else self.generator_ema.state_dict(),
            "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(self.device),
                    "python": random.getstate(), "numpy": np.random.get_state()},
        }
        torch.save(state, os.path.join(tmp, f"rank{rank}.pt"))
        del state
        dist.barrier()
        if self.is_main_process:
            with open(os.path.join(tmp, "COMPLETE"), "w") as f:
                json.dump({"step": self.step, "world_size": self.world_size}, f)
            shutil.rmtree(old, ignore_errors=True)
            if os.path.isdir(self.resume_dir):
                os.replace(self.resume_dir, old)
            os.replace(tmp, self.resume_dir)
            print(f"[resume] state after step {self.step} saved to {self.resume_dir}", flush=True)
        dist.barrier()

    def _load_resume_state(self):
        """Continue from the newest complete state in resume_dir, if there is one."""
        found = []
        for d in (self.resume_dir, self.resume_dir + ".tmp", self.resume_dir + ".old"):
            marker = os.path.join(d, "COMPLETE")
            if os.path.exists(marker):
                with open(marker) as f:
                    found.append((int(json.load(f)["step"]), d))
        if not found:
            if self.is_main_process:
                print(f"[resume] no saved state in {self.resume_dir}: starting from step 0", flush=True)
            return
        step, d = max(found)
        rank = dist.get_rank()
        state = torch.load(os.path.join(d, f"rank{rank}.pt"), map_location="cpu", weights_only=False)
        assert state["step"] == step and state["rank"] == rank and state["world_size"] == self.world_size, (
            f"[resume] {d}: saved for step {state['step']} rank {state['rank']} of {state['world_size']}, "
            f"this is rank {rank} of {self.world_size}")
        for key, module in (("generator", self.model.generator), ("critic", self.model.fake_score)):
            saved = state[key]
            assert [n for n, _ in module.named_parameters()] == list(saved), f"[resume] parameter names of the {key} differ"
            for n, p in module.named_parameters():
                assert p.shape == saved[n].shape, f"[resume] {key} {n}: shard {tuple(p.shape)} vs saved {tuple(saved[n].shape)}"
                p.data.copy_(saved[n].to(device=p.device, dtype=p.dtype))
        self.generator_optimizer.load_state_dict(state["generator_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if state["generator_ema"] is not None:
            if self.generator_ema is None:
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
            self.generator_ema.load_state_dict(state["generator_ema"])
        self.step = int(state["step"])
        # same prompts as the uninterrupted run: the sampler's order does not depend on the RNG restored below
        for _ in range(int(state["batches_consumed"])):
            next(self.dataloader)
        self._batches_consumed = int(state["batches_consumed"])
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"], self.device)
        del state
        if self.is_main_process:
            print(f"[resume] continuing after step {self.step} from {d} "
                  f"({self._batches_consumed} batches already drawn)", flush=True)
    # ==== ours end ====

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator_ema": self.generator_ema.full_state_dict(self.model.generator),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def save_critic(self):
        print("Start gathering distributed model states...")
        
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        
        state_dict = critic_state_dict

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))
            
    def fwdbwd_one_step(self, batch, train_generator, clean_latent=None):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            # clean_latent = None #original code here
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            # clean_latent = None #original code here
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            # Re-DMD needs the prompt strings for the reward model; without reward_dmd the call is unchanged
            reward_kwargs = {"text_prompts": text_prompts} if getattr(self.config, "reward_dmd", False) else {}
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None,
                **reward_kwargs
            )
           
            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict


    def train(self):
        start_step = self.step
        # Total training steps from config (max_steps); -1 or absent = unlimited.
        max_steps = getattr(self.config, "max_steps", -1)
        step_time_interval = self.step_time_log_interval
        step_time_start_step = self.step
        if step_time_interval > 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            step_time_start = time.time()
        else:
            step_time_start = None

        while True:
            # Skip the generator update on the very first step so the critic
            # (fake_score) gets warmed up before the generator first moves.
            TRAIN_GENERATOR = self.step > 0 and self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                
                batch = self._next_batch()
                generator_log_dict = self.fwdbwd_one_step(batch, True)

                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)
                
                
                

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            batch = self._next_batch()
            critic_log_dict = self.fwdbwd_one_step(batch, False)
                
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1
            step_time_report = None
            if step_time_interval > 0 and (self.step - step_time_start_step) >= step_time_interval:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                current_time = time.time()
                timed_steps = self.step - step_time_start_step
                elapsed_time = current_time - step_time_start
                seconds_per_step = elapsed_time / timed_steps
                step_time_report = (timed_steps, elapsed_time, seconds_per_step)

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                log_str = "  ".join(f"{k}: {v:.4f}" for k, v in wandb_loss_dict.items())
                print(f"[step {self.step}] {log_str}", flush=True)
                # ======== ours ========
                if self.step % self.config.log_iters == 0 and torch.cuda.is_available():
                    print("[mem] step %d: max allocated %.1f GiB, max reserved %.1f GiB (this rank)" % (
                        self.step, torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30), flush=True)
                # ==== ours end ====

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            # ======== ours ========
            if self.resume_every > 0 and self.step % self.resume_every == 0 and not (max_steps > 0 and self.step >= max_steps):
                self.save_resume_state()
            # ==== ours end ====

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            if step_time_report is not None:
                timed_steps, elapsed_time, seconds_per_step = step_time_report
                if self.is_main_process:
                    print(
                        f"[timing] step {self.step}: last {timed_steps} steps "
                        f"{elapsed_time:.2f}s total, {seconds_per_step:.2f}s/step",
                        flush=True
                    )
                    if not self.disable_wandb:
                        wandb.log(
                            {
                                "step_time/last_steps_sec": elapsed_time,
                                "step_time/sec_per_step": seconds_per_step,
                                "step_time/interval_steps": timed_steps,
                            },
                            step=self.step
                        )
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                step_time_start = time.time()
                step_time_start_step = self.step

            # Stop after reaching max_steps; checkpoint the final step if the
            # periodic save above did not already land on it.
            if max_steps > 0 and self.step >= max_steps:
                if (not self.config.no_save) and (self.step % self.config.log_iters != 0):
                    torch.cuda.empty_cache()
                    self.save()
                    torch.cuda.empty_cache()
                if self.is_main_process:
                    print(f"[step {self.step}] reached max_steps={max_steps}, stopping.", flush=True)
                break
