# Run configs of the slurm-2 Memory-LoRA distance-gate experiments (2026-10-09/10)

- `kvf_v1fuse_lora_gate_<gate>_8x1.yaml`: the exact configs of the trained runs (8 GPUs x batch 1, 1500 steps, seed 1,
  V1 fuse tier, gate lr 3e-3). `..._mix_glr1.2e-2_8x1.yaml`: the mix gate with gate lr 1.2e-2 (x4).
- `kvf_v1fuse_lora_4x2_gate_<gate>.yaml`: the 4 GPU x batch 2 versions they were derived from (not trained: 4 GPUs did not fit).
- `generator_ckpt` and `data_path` are absolute slurm-2 paths: change them on another cluster.
