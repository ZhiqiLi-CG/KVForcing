# Run configs of the slurm-2 decoupled Memory-LoRA write experiment (2026-10-09)

- `kvf_v1fuse_lora_decoupled_8x1.yaml`: the exact config of the trained run (8 GPUs x batch 1, 1500 steps, seed 1, V1 fuse tier).
- `kvf_v1fuse_lora_decoupled_4x2.yaml`: the 4 GPU x batch 2 version it was derived from (not trained: 4 GPUs did not fit).
- `generator_ckpt` and `data_path` are absolute slurm-2 paths: change them on another cluster.
