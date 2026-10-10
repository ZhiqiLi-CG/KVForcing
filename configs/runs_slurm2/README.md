# Run configs of the slurm-2 4 GPU x batch 2 runs (seed 1, 1500 steps); see docs/TASK_slurm2_20261010.md
kvf_v0_lora_4x2 (V0), kvf_v1fuse_lora_4x2 (V1), kvf_v2hpage_lora_4x2 (V2), kvf_v0_nolora_4x2, sf_nomemgrad_4x2,
kvf_v1fuse_lora_lr3e-5_4x2. The replicates *_rep2/_rep3 use these files unchanged.
generator_ckpt and data_path are absolute slurm-2 paths: change them on another cluster.
