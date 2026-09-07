"""Write zero-valued Memory-LoRA adapters into an init checkpoint.

The trainer loads generator_ckpt with strict=True and no pretrained checkpoint
carries the adapters, so they are materialised here: `down` as the model's own
reset_parameters would draw it (seeded per layer, reproducible), `up` as zeros,
so dW == 0 and the memory and generation projections start identical.

usage: python scripts/make_lora_init.py checkpoints/init/framewise/causal_cd.pt \
                                        checkpoints/init/framewise/causal_cd_lora_init.pt
"""
import re
import sys

import torch

RANK = 64
src, dst = sys.argv[1], sys.argv[2]
blob = torch.load(src, map_location="cpu")
key = "generator" if "generator" in blob else "generator_ema"
sd = blob[key]
sample = next(k for k in sd if ".self_attn.k.weight" in k)
prefix = sample.split("blocks.")[0]
dim = sd[sample].shape[0]
layers = sorted({re.search(r"blocks\.(\d+)\.", k).group(1) for k in sd if ".self_attn.k.weight" in k}, key=int)
added = 0
for i in layers:
    for j, which in enumerate(("k", "v")):
        base = f"{prefix}blocks.{i}.self_attn.{which}_memory_lora"
        if base + ".up.weight" in sd:
            continue
        g = torch.Generator().manual_seed(2 * int(i) + j)
        sd[base + ".down.weight"] = torch.randn(RANK, dim, generator=g) / RANK
        sd[base + ".up.weight"] = torch.zeros(dim, RANK)
        added += 2
torch.save({key: sd}, dst)
print(f"{len(layers)} layers, dim {dim}, rank {RANK}: added {added} adapter tensors -> {dst}")
