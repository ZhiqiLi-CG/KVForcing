# KV Forcing: the algorithm and where it lives in the code

## The problem

An autoregressive video diffusion model generates a clip frame by frame and keeps the frames
it has produced in a KV cache. Training it on its own rollouts (Self Forcing) takes the
gradient at one denoising step per frame, but the KV cache is written in place and the write
is stop-gradiented, so the loss of a later frame never reaches the way an earlier frame was
written into memory. Two-pass methods recover that path with a second, teacher-forced
parallel forward over the whole clip. KV Forcing recovers it in one pass.

## The idea

Each generated frame writes its K/V as **immutable pages** (one `(K, V)` per layer) instead of
into a cache buffer. Pages carry the autograd graph, so frame *i*'s loss reaches every earlier
frame through the chain `loss_i -> pages_{<i} -> writer_{<i} -> pages_{<j} -> ... -> parameters`.
Because a page never changes, every per-frame forward can be activation-checkpointed and
recomputed at backward time against exactly the history it saw at forward time; a cache that
later frames overwrite could not offer that, which is why one-pass Self Forcing had to cut the
gradient at the cache and why its differentiable variant keeps every activation alive.

## One training step of the generator

Frames T = 21, denoising steps S = 4 (`[1000, 750, 500, 250]`, warped).

1. Draw the exit step `idx` (per rank by default, or on rank 0 and broadcast); `train_t = steps[idx]`.
2. For each frame *i*: take a snapshot of the history pages (tuples of references; later appends
   cannot change it).
3. **Rollout under `no_grad`, exactly as at inference**: denoise from noise over the S steps,
   reading the pages; remember `noisy_at_t` (the input at the exit step) and `x0_exit`;
   `x_ctx = add_noise(x0_exit, context_timestep)`.
4. **Two forwards that carry gradient**: the *noisy* forward of `noisy_at_t` at `train_t`
   (its prediction is what the loss sees) and the *write* forward of `x_ctx` at the context
   timestep, which returns this frame's 30 layers of `(K, V)` pages and appends them to the store.
5. Concatenate the 21 predictions for the DMD loss.

The rollout (exit-step draw, schedule, context noising, what is written) is the Self Forcing
rollout. Inference is unchanged and uses the ordinary KV cache.

## Code map

Every insertion into an upstream file is delimited by `# ======== ours ========` / `# ==== ours end ====`.

| where | what |
|---|---|
| `wan/modules/causal_model.py` `PersistentKVStore` | per-layer page lists; `append_all_layers`, `snapshot_all_layers` |
| `wan/modules/causal_model.py` `_PageCat` | `torch.cat` over the history pages whose backward hands each page a contiguous gradient (cat's own backward returns views of one whole-history buffer, which a page's `.grad` would pin) |
| `CausalWanSelfAttention.forward_persistent` | q/k/v projections, RoPE at the absolute frame position, `_PageCat(history, own)`, flash-attn; returns `(k_rope, v)` as the page when `write_page` |
| `CausalWanAttentionBlock.forward_persistent` | the block against pages instead of a cache; cross-attention recomputed (no mutable state inside a checkpoint region) |
| `CausalWanModel.forward` dispatch, `forward_noisy_frame`, `forward_clean_writer` | reached through `__call__` (FSDP's forward hook gathers and casts the parameters there) |
| `CausalWanModel._persistent_blocks` / `_run_blocks` | activation checkpointing in groups of `persistent_checkpoint_group` blocks inside the FSDP unit |
| `utils/wan_wrapper.py` `persistent_mode` branch | "noisy" shares the flow-to-x0 conversion with the cache path; "write" returns the pages |
| `pipeline/kv_forcing_training.py` | the step above; `checkpoint_group`, `match_context`, `cache_mode`, `per_rank_exit_step`, `last_step_only` |
| `pipeline/self_forcing_training.py` | the two one-pass baselines (`self_forcing`, `self_forcing_no_sg`): the same rollout, differing only in the stop-gradient on the KV write |
| `model/base.py` | `grad_mode` dispatch |

## Memory-LoRA (`model_kwargs.memory_lora: true`)

`MemoryLoRA` (rank 64, `up` zero-initialised) is added to the K/V projections **before `norm_k`**,
and only for the memory write: the page write in KV Forcing, the `memory_write=True` cache
refresh (training baselines and inference), and the clean half of a teacher-forced forward.
Generation keeps `W`. The adapters get their own learning rate (`lr_memory_lora`);
`scripts/make_lora_init.py` writes zero adapters into an init checkpoint.

## Memory

Peak activations at T = 21 on one H100, generator only, bf16: 20.5 GiB with one checkpoint
region per block, **17.3 GiB with `kv_forcing_checkpoint_group: 5`** (the default), and the
fwd+bwd time drops from 25.3 s to 15.9 s; the numerics are identical. What remains at the
peak is the pages (5.6 GiB) and their gradients (5.3 GiB), which are the method itself, plus the
kept group boundaries and backward temporaries. The teacher-forced parallel forward of a
two-pass method peaks at 40 GiB on its first step (block-mask construction) and about 21.8 GiB
afterwards.

## Precision

The shipped configs train in fp32 (`mixed_precision: false`): weights, activations and gradient
accumulation. In the one-pass every parameter receives 42 backward contributions per step
(21 frames x noisy + write); summing them in a bf16 `.grad` adds roughly 40% relative error
over a single-contribution two-pass gradient (measured on a small model: 6.3e-3 vs 4.4e-3
against fp32; 4.1e-3 with fp32 accumulation). The frozen 14B teacher may run in bf16
(`real_score_bf16: true`): it has no gradients. flash-attn itself always computes in bf16.

## Limits

* `num_frame_per_block` must be 1.
* The persistent forwards attend to all history, which equals sink + window attention only
  while `local_attn_size + sink_size` covers the clip (18 + 3 = 21 by default); longer clips are refused.
