# KV Forcing

**One-pass autoregressive video distillation with a differentiable KV memory.**

Autoregressive video diffusion models generate a clip frame by frame and keep
the frames they have already produced in a KV cache. When such a model is
trained on its own rollouts (Self Forcing), the gradient of a later frame's
loss cannot reach the memory the earlier frames wrote, because the cache is
overwritten in place and the write is stop-gradiented. KV Forcing removes that
cut without a second pass:

* each generated frame writes its K/V as **immutable pages** instead of into a
  cache buffer, so the pages can carry a graph and a later frame's loss
  backpropagates through them into every earlier frame;
* because a page never changes, every per-frame forward can be
  **activation-checkpointed** and recomputed at backward time against exactly
  the history it saw at forward time, so the memory cost stays that of one
  frame's activations plus the pages.

The rollout itself is the ordinary Self Forcing rollout: same exit-step
sampling, same denoising schedule, same context noising. Inference is
unchanged and uses the ordinary KV cache.

## What is in this repository

| path | content |
|---|---|
| `pipeline/kv_forcing_training.py` | the KV Forcing training rollout |
| `pipeline/self_forcing_training.py` | the two one-pass baselines: Self Forcing (stop-gradient KV write) and Self Forcing without the stop-gradient (full BPTT through a mutable cache) |
| `wan/modules/causal_model.py` | the causal Wan backbone; the page store and the two persistent forwards are the blocks marked `# ======== ours ========` |
| `utils/wan_wrapper.py`, `model/base.py` | the `persistent_mode` entry point and the `grad_mode` dispatch, likewise marked |
| `configs/*_framewise.yaml` | one config per method; they differ only in `grad_mode` (and the Memory-LoRA flags) |

Everything else is the Causal-Forcing / Self-Forcing training and inference
code this repository builds on, including the sink + sliding-window attention
and the streaming long-video inference. Every insertion into an upstream file
is delimited by `# ======== ours ========` / `# ==== ours end ====`.

## Installation

```bash
conda create -n kv_forcing python=3.10 -y
conda activate kv_forcing
pip install torch torchvision
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

## Weights

```bash
bash scripts/download_weights.sh
```

downloads the Wan2.1 1.3B and 14B base models, the Causal-Forcing framewise
initializations (`checkpoints/init/framewise/causal_cd.pt` is the default
starting point in `configs/`) and the training prompt list.

## Training

```bash
# KV Forcing
NUM_GPUS=8 bash scripts/train.sh configs/kv_forcing_framewise.yaml logs/kv_forcing
# baselines
NUM_GPUS=8 bash scripts/train.sh configs/self_forcing_framewise.yaml       logs/self_forcing
NUM_GPUS=8 bash scripts/train.sh configs/self_forcing_no_sg_framewise.yaml logs/self_forcing_no_sg
```

`scripts/train.sh` runs `train.py` under `torchrun`; run the same command on
every node for multi-node training (see the script for the rendezvous
options). Set `seed` to a nonzero value in both configs for a controlled A/B:
`seed: 0` draws a random seed.

The configs train in fp32 (`mixed_precision: false`); flash-attn itself always runs in bf16.
`kv_forcing_checkpoint_group` sets how many blocks share one activation-checkpoint region
(default 5: at 21 frames the one-pass peaks at 17.3 GiB of activations on one GPU versus
20.5 GiB with a region per block, and runs faster; the numerics are identical).

The configs are identical apart from `grad_mode`. `local_attn_size`
and `sink_size` must together cover the training clip (18 + 3 = 21 frames):
the KV Forcing pages are attended to in full, which equals sink + window
attention only while the two cover the clip, and the pipeline refuses longer
clips.

## Memory-LoRA

`configs/kv_forcing_lora_framewise.yaml` trains KV Forcing with a separate
projection for the memory write: the K/V projections used when a frame is
written into memory become `W + dW`, with `dW` a zero-initialised rank-64
adapter (`MemoryLoRA` in `wan/modules/causal_model.py`), while generation keeps
`W`. The adapter is applied inside the K/V projection, before `norm_k`, at the
page write during training and at the KV-cache refresh at inference, and it
gets its own learning rate (`lr_memory_lora`). Build the init checkpoint that
carries the (zero) adapters first:

```bash
python scripts/make_lora_init.py checkpoints/init/framewise/causal_cd.pt \
                                 checkpoints/init/framewise/causal_cd_lora_init.pt
NUM_GPUS=8 bash scripts/train.sh configs/kv_forcing_lora_framewise.yaml logs/kv_forcing_lora
```

## Inference

```bash
# 5 s (21 latent frames)
NUM_OUTPUT_FRAMES=21 bash scripts/infer.sh logs/kv_forcing/checkpoint_model_001500/model.pt prompts/test_prompt.txt
# 60 s / 240 s streaming with a sink + FIFO KV cache
NUM_OUTPUT_FRAMES=241 bash scripts/infer.sh logs/kv_forcing/checkpoint_model_001500/model.pt
NUM_OUTPUT_FRAMES=963 bash scripts/infer.sh logs/kv_forcing/checkpoint_model_001500/model.pt
```

`scripts/eval/infer_long_chunked.py` decodes very long clips in chunks so the
VAE does not run out of memory. `scripts/eval/sample_for_vbench.sh` and
`scripts/eval/eval_vbench.sh` sample and score with the official VBench /
VBench-Long evaluators (cloned separately, see the scripts).

### Evaluating the model the way it was trained

Training attends to the whole 21-frame clip (`local_attn_size 18` + `sink_size 3`).
Three things keep sampled videos consistent with that:

* **Inference mode.** `scripts/infer.sh` and every suite of
  `scripts/eval/sample_for_vbench.sh` pass `--long_video` with a 21-frame window
  (sink 4 + FIFO 16 + current 1); for a 21-frame clip that is exactly the
  training rollout. Calling `inference.py` without `--long_video` is equivalent
  up to 21 frames. (Until 2026-10-09 it was not: the plain mode sized its KV
  cache to `local_attn_size` alone, 18 frames including the 3 sink frames, so
  latent frames 19-21 of every 5 s clip lost 1-3 frames of history. The cache
  now holds sink + window.)
* **Prompts.** The model has only seen long, LLM-extended prompts. Short
  benchmark prompts must be extended: `inference.py --extended_prompt_path`
  generates from the extended prompt and names the file after the short one.
  `prompts/vbench/` has the VBench prompts in both forms.
* **A test.** `scripts/eval/check_train_eval_consistency.py` rolls the same
  noise through the inference pipeline (both modes) and through the training
  rollout and compares frame by frame in exact arithmetic. Run it after any
  change to the attention, the KV cache or the pipelines:

  ```bash
  python scripts/eval/check_train_eval_consistency.py --config_path CFG --checkpoint_path CKPT \
      --mode fp64 --device cpu --hw 12x20        # minutes on a CPU node; drop --device/--hw on a GPU
  ```

Two differences between training and inference are inherited from Self-Forcing
and left as they are: training adds a little noise (sigma about 0.005) to a
frame before writing it to memory, and FSDP mixed precision rounds the float
timestep to bf16 in training (937.5 becomes 936).

## Acknowledgements

This code builds on [Causal-Forcing](https://github.com/thu-ml/Causal-Forcing),
[Self-Forcing](https://github.com/guandeh17/Self-Forcing) and
[Wan2.1](https://github.com/Wan-Video/Wan2.1).

## License

See [LICENSE](LICENSE).
