"""Generate very long videos without the VAE decode running out of memory.

The diffusion side already scales: the streaming KV cache is capped at
kv_cache_max_frames, so a 240s clip costs the same per step as a 60s one. What
does not scale is the decode -- 963 latent frames become 3849 pixel frames, and
holding those as float32 at 480x832x3 needs ~18.4 GiB in one tensor, which is
where 240s dies while 60s (961 frames, ~4.6 GiB) survives.

So decode in chunks and move each chunk to CPU as it is produced, leaving only
one chunk on the GPU. The VAE's cached_decode keeps its feature cache across
calls, so chunking does not introduce seams -- the cache is cleared once before
the sequence, never between chunks.

The chunked decode is installed onto the pipeline's VAE at runtime rather than
edited into the shared wrapper, so every existing path keeps the decode it
always had and short-video results stay exactly comparable.

Usage mirrors inference.py:
  torchrun --standalone --nproc_per_node=8 scripts/eval/infer_long_chunked.py \
      --config_path CFG --checkpoint_path CKPT --data_path prompts/test_prompt.txt \
      --output_folder OUT --num_output_frames 963 [--use_ema]
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.io import write_video

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.causal_inference import CausalInferencePipeline  # noqa: E402
from utils.dataset import TextDataset  # noqa: E402
from utils.misc import set_seed  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--config_path", required=True)
p.add_argument("--checkpoint_path", required=True)
p.add_argument("--data_path", required=True)
p.add_argument("--output_folder", required=True)
p.add_argument("--num_output_frames", type=int, default=963)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--use_ema", action="store_true")
p.add_argument("--decode_chunk", type=int, default=24,
               help="latent frames decoded per chunk; each becomes 4x pixel frames")
p.add_argument("--kv_cache_max_frames", type=int, default=21)
p.add_argument("--kv_cache_sink", type=int, default=4)
p.add_argument("--kv_cache_train_frames", type=int, default=21)
p.add_argument("--kv_cache_position_mode", type=str, default="top_aligned")
args = p.parse_args()

rank = int(os.environ.get("RANK", 0))
world = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if world > 1:
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                         OmegaConf.load(args.config_path))
pipeline = CausalInferencePipeline(config, device=device)

blob = torch.load(args.checkpoint_path, map_location="cpu")
sd = blob["generator_ema" if args.use_ema else "generator"]
try:
    pipeline.generator.load_state_dict(sd)
except RuntimeError:
    fixed = {k.replace("model._fsdp_wrapped_module.", "model.", 1): v for k, v in sd.items()}
    pipeline.generator.load_state_dict(fixed, strict=False)

pipeline = pipeline.to(dtype=torch.bfloat16)
pipeline.text_encoder.to(device)
pipeline.generator.to(device)
pipeline.vae.to(device)

vae = pipeline.vae


def chunked_decode_to_pixel(latent, use_cache=False):
    """Same contract as WanVAEWrapper.decode_to_pixel, bounded in GPU memory."""
    zs = latent.permute(0, 2, 1, 3, 4)                    # [B, C, T, H, W]
    dtype = latent.dtype
    scale = [vae.mean.to(device=latent.device, dtype=dtype),
             1.0 / vae.std.to(device=latent.device, dtype=dtype)]
    # Cleared ONCE: cached_decode carries temporal context in this cache, so
    # clearing between chunks is exactly what would create seams.
    vae.model.clear_cache()
    parts = []
    for i in range(0, zs.shape[2], args.decode_chunk):
        out = vae.model.cached_decode(zs[:, :, i:i + args.decode_chunk], scale)
        parts.append(out.float().clamp_(-1, 1).cpu())
        del out
        torch.cuda.empty_cache()
    vae.model.clear_cache()
    return torch.cat(parts, dim=2).permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]


vae.decode_to_pixel = chunked_decode_to_pixel

os.makedirs(args.output_folder, exist_ok=True)
dataset = TextDataset(prompt_path=args.data_path)
print(f"[rank {rank}] {len(dataset)} prompts, {args.num_output_frames} latent frames "
      f"(~{(4 * args.num_output_frames - 3) // 16}s), decode chunk {args.decode_chunk}")

for idx in range(len(dataset)):
    if idx % world != rank:
        continue
    # TextDataset stores "prompts" as a plain string, so indexing it would take
    # a single character rather than the prompt.
    item = dataset[idx]
    prompt = item["prompts"] if isinstance(item, dict) else item
    if isinstance(prompt, (list, tuple)):
        prompt = prompt[0]
    out_path = os.path.join(args.output_folder, f"{prompt[:100]}.mp4")
    if os.path.exists(out_path):
        print(f"[rank {rank}] skip {idx}")
        continue

    set_seed(args.seed + idx)
    noise = torch.randn([1, args.num_output_frames, 16, 60, 104],
                        device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        video, _ = pipeline.inference(
            noise=noise, text_prompts=[prompt], return_latents=True,
            long_video=True,
            kv_cache_max_frames=args.kv_cache_max_frames,
            kv_cache_sink=args.kv_cache_sink,
            kv_cache_train_frames=args.kv_cache_train_frames,
            kv_cache_position_mode=args.kv_cache_position_mode,
        )
    video = 255.0 * rearrange(video, "b t c h w -> b t h w c")
    tmp = f"{out_path}.rank{local_rank}.tmp.mp4"
    write_video(tmp, video[0], fps=16)
    os.replace(tmp, out_path)
    print(f"[rank {rank}] wrote {idx}: {video.shape[1]} frames")

print(f"[rank {rank}] done")
