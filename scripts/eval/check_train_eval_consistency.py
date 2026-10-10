#!/usr/bin/env python3
"""Does inference run the model the way training ran it?

For one checkpoint and its training config the script rolls out the SAME latent frames from the SAME noise with

  A   pipeline/causal_inference.py, plain mode            (inference.py without --long_video)
  B   pipeline/causal_inference.py, --long_video, window 21 = sink 4 + FIFO 16 + current 1 (--kv_cache_max_frames /
      --kv_cache_sink; chunkwise SGF: 12 = 3 + 6 + 3)
                                                           (scripts/infer.sh, scripts/eval/sample_for_vbench.sh)
  C   the TRAINING rollout the trainer builds for this config (built by this checkout's own
      _initialize_inference_pipeline of the model class the trainer instantiates, model/dmd.py DMD, so every branch
      tests its own pipeline), with the exit step fixed to the last
      denoising step, so that the frame it outputs and the frame it writes to memory are the final ones as at
      inference, and with the context noise it adds before the memory write switched off
  D   C with that context noise left on, exactly as trained

and compares the latents frame by frame. All rollouts draw their re-noising noise from one bank keyed by
(frame, step), so a difference is a difference of the computation, not of the random stream.

How to read the result. The autoregressive rollout is chaotic: two computations that differ by one rounding error
drift apart by roughly an order of magnitude per frame, and the 0.5 % context noise of D alone changes later frames
by tens of percent. So the size of a difference after 21 frames says nothing. What separates rounding from a
structural difference (attention window, RoPE position, which forward carries the Memory-LoRA, schedule) is how the
difference STARTS: rounding starts at the rounding level of the arithmetic and grows smoothly, a structural
difference starts at percent level in the first frame it affects. The verdict (fp64 mode) is therefore:
A vs C and B vs C must start below 1e-9 and never jump by more than 1e4 from one frame to the next.

Modes
  fp64   float64 weights and a dense float64 attention with an exact RMS norm: rounding is ~1e-14. This is the
         verdict. Needs no GPU kernels, so it also runs on a CPU node (--device cpu).
  bf16   the precision and attention kernels actually used. Sizes are reported, not judged (bf16 rounding is ~1e-3
         and is amplified the same way). Also reports the effect of FSDP rounding the timestep to bf16 in training
         and the training rollout with gradient enabled. On a CPU this mode substitutes float32 + dense attention,
         which only exercises the code path.

--hw HxW rolls out latents of another size (tokens per frame (H/2)*(W/2) instead of 1560; the hard-coded 1560-token
constants are overridden on the objects). The frames, attention windows, positions, LoRA routing and schedules are
those of the real size, so it answers the structural question in minutes on a CPU node:
    python scripts/eval/check_train_eval_consistency.py --config_path CFG --checkpoint_path CKPT \\
        --mode fp64 --device cpu --hw 12x20
Full size, float64, needs one GPU with about 60 GB (a CPU node takes about 7 hours per rollout).

Also checked: the checkpoint covers every key of the inference model (inference.py falls back to a non-strict
load, which would hide a missing Memory-LoRA), and the inference and training denoising schedules are equal.

Prints VERDICT PASS / FAIL; exit code 0 / 1. History: 2026-10-09, written when the plain mode was found to cache
3 frames too few for local_attn_size 18 + sink_size 3 (fixed in pipeline/causal_inference.py)."""
import argparse
import gc
import json
import os
import sys
import time

import torch

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--config_path", required=True)
ap.add_argument("--checkpoint_path", required=True)
ap.add_argument("--mode", choices=["fp64", "bf16"], default="fp64")
ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
ap.add_argument("--hw", default="", help="latent HxW instead of the config's (e.g. 12x20)")
ap.add_argument("--frames", type=int, default=0, help="latent frames instead of the config's (quick smoke test)")
ap.add_argument("--prompt_path", default="prompts/test_prompt.txt")
ap.add_argument("--prompt_index", type=int, default=0)
ap.add_argument("--use_ema", action="store_true", help="default: whichever of generator / generator_ema the file has")
ap.add_argument("--out", default="", help="write the full result (per-frame differences) to this JSON file")
ap.add_argument("--kv_cache_max_frames", type=int, default=21, help="window of the --long_video rollout B (framewise: 4+16+1)")
ap.add_argument("--kv_cache_sink", type=int, default=4)
ap.add_argument("--inference_only", action="store_true",
                help="compare the two inference modes only (A vs B); for checkouts whose training rollout this script cannot drive")
args = ap.parse_args()

CPU = args.device == "cpu"
if CPU:
    # the repository imports GPU-only things at import time; none of them is used on this path
    import types
    torch.cuda.get_device_name = lambda *a, **k: "cpu"
    torch.cuda.current_device = lambda *a, **k: "cpu"          # default argument in wan/modules/t5.py
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.empty_cache = lambda *a, **k: None
    _m = types.ModuleType("demo_utils.memory")
    _m.gpu = torch.device("cpu")
    _m.get_cuda_free_memory_gb = lambda *a, **k: 0.0
    _m.DynamicSwapInstaller = type("DynamicSwapInstaller", (), {})
    _m.move_model_to_device_with_memory_preservation = lambda *a, **k: None
    _p = types.ModuleType("demo_utils")
    _p.__path__, _p.memory = [], _m
    sys.modules["demo_utils"], sys.modules["demo_utils.memory"] = _p, _m

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFG, CK = os.path.abspath(args.config_path), os.path.abspath(args.checkpoint_path)
OUT = os.path.abspath(args.out) if args.out else ""
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from omegaconf import OmegaConf  # noqa: E402

from pipeline import CausalInferencePipeline  # noqa: E402

t_start = time.time()
torch.set_grad_enabled(False)
device = torch.device(args.device)
config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(CFG))
R = dict(config=CFG, checkpoint=CK, code=ROOT, mode=args.mode, device=args.device, hw=args.hw or None,
         frames=args.frames or None, host=os.uname().nodename, torch=torch.__version__,
         grad_mode=str(config.grad_mode), model_kwargs=OmegaConf.to_container(config.model_kwargs), checks={}, compare={})
fails, notes = [], []


def say(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ dense attention backend (exact arithmetic)
def _dense_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None, q_scale=None,
                     causal=False, window_size=(-1, -1), deterministic=False, dtype=None, version=None,
                     fa_version=None):
    """wan.modules.attention.attention / flash_attention as one dense softmax attention in the input dtype."""
    assert q_lens is None and dropout_p == 0 and not causal and tuple(window_size) == (-1, -1) and q_scale is None
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    qh, kh, vh = q.transpose(1, 2), k.to(q.dtype).transpose(1, 2), v.to(q.dtype).transpose(1, 2)
    s = (qh @ kh.transpose(-1, -2)) * scale
    if k_lens is not None:
        idx = torch.arange(k.shape[1], device=k.device)
        s = s.masked_fill(idx[None, None, None, :] >= k_lens.to(k.device)[:, None, None, None], float("-inf"))
    return (torch.softmax(s, dim=-1) @ vh).transpose(1, 2)


def install_dense_backend(exact_norm):
    """Replace every flash-attn entry point of the wan.* modules (call after they are imported)."""
    import wan.modules.attention as A
    patched = []
    for name, m in list(sys.modules.items()):
        if m is None or not (name == "wan" or name.startswith("wan.")):
            continue
        for attr in ("attention", "flash_attention"):
            if callable(getattr(m, attr, None)) and not isinstance(getattr(m, attr), type(sys)):
                setattr(m, attr, _dense_attention)
                patched.append("%s.%s" % (name, attr))
        if callable(getattr(m, "_attn_dtype", None)):          # pages / cached text K,V stay in the input dtype
            setattr(m, "_attn_dtype", lambda dtype: dtype)
            patched.append("%s._attn_dtype" % name)
    A.FLASH_ATTN_2_AVAILABLE = False
    A.FLASH_ATTN_3_AVAILABLE = False
    if exact_norm:                                             # upstream's RMS norm computes in float32
        from wan.modules.model import WanRMSNorm
        WanRMSNorm.forward = lambda self, x: self._norm(x) * self.weight
        patched.append("WanRMSNorm.forward")
    return patched


# ------------------------------------------------------------------ model + checkpoint, loaded as inference.py does
random_prompt = CPU and bool(args.frames)                      # smoke test: skip the 20 GB text encoder
pipeline = (CausalInferencePipeline(config, device=device, text_encoder=torch.nn.Identity()) if random_prompt
            else CausalInferencePipeline(config, device=device))
gen = pipeline.generator
blob = torch.load(CK, map_location="cpu")
key = "generator_ema" if (args.use_ema or "generator" not in blob) else "generator"
gen_sd = blob[key]
try:
    gen.load_state_dict(gen_sd)
    load = dict(strict_ok=True, missing=[], unexpected=[])
except RuntimeError as e:                                       # inference.py's fallback
    gen_sd = {(k.replace("model._fsdp_wrapped_module.", "model.", 1) if k.startswith("model._fsdp_wrapped_module.") else k): v
              for k, v in gen_sd.items()}
    res = gen.load_state_dict(gen_sd, strict=False)
    load = dict(strict_ok=False, error=str(e)[:300], missing=list(res.missing_keys), unexpected=list(res.unexpected_keys))
load.update(state_key=key, ckpt_keys=len(gen_sd), model_keys=len(gen.state_dict()))
lora_up = [float(v.double().norm()) for k, v in gen_sd.items() if "memory_lora" in k and k.endswith("up.weight")]
load.update(memory_lora_up_tensors=len(lora_up), memory_lora_up_norm_min=min(lora_up) if lora_up else None)
R["checks"]["checkpoint_load"] = load
say("LOAD key=%s strict_ok=%s ckpt_keys=%d model_keys=%d missing=%d unexpected=%d | memory-lora up tensors %d (min norm %s)"
    % (key, load["strict_ok"], load["ckpt_keys"], load["model_keys"], len(load["missing"]), len(load["unexpected"]),
       len(lora_up), load["memory_lora_up_norm_min"]))
if load["missing"] or load["unexpected"]:
    fails.append("checkpoint does not cover the inference model: missing %s unexpected %s"
                 % (load["missing"][:4], load["unexpected"][:4]))
want_lora = bool(config.model_kwargs.get("memory_lora", False))
if want_lora and len(lora_up) != 2 * len(gen.model.blocks):
    fails.append("config has memory_lora but the checkpoint has %d Memory-LoRA up tensors" % len(lora_up))
del blob
gc.collect()

# ------------------------------------------------------------------ prompt embedding
prompt = [line.rstrip() for line in open(args.prompt_path, encoding="utf-8")][args.prompt_index]
if random_prompt:
    emb = torch.randn([1, 512, 4096], generator=torch.Generator().manual_seed(7), dtype=torch.float32)
else:
    pipeline.text_encoder.to(device=device, dtype=torch.float32 if CPU else torch.bfloat16)
    emb = pipeline.text_encoder(text_prompts=[prompt])["prompt_embeds"].detach()
    pipeline.text_encoder.to("cpu")


class _Cond(torch.nn.Module):
    def __init__(self, cond):
        super().__init__()
        self.cond = cond

    def forward(self, text_prompts):
        return self.cond


if args.mode == "fp64":
    R["backend"], WD = install_dense_backend(exact_norm=True), torch.float64
elif CPU:
    R["backend"], WD = install_dense_backend(exact_norm=False), torch.float32
else:
    R["backend"], WD = "native (flash-attn)", torch.bfloat16
gen.to(device=device, dtype=WD)
cond = {"prompt_embeds": emb.to(device=device, dtype=WD)}
pipeline.text_encoder = _Cond(cond)
pipeline.vae.decode_to_pixel = lambda latent, use_cache=False: torch.zeros(1, device=latent.device)   # latents only
gc.collect()
torch.cuda.empty_cache()

shape = list(config.image_or_video_shape)
if args.frames:
    shape[1] = args.frames
F_, FS = int(shape[1]), 1560
NFPB = int(getattr(config, "num_frame_per_block", 1))
if args.hw:
    shape[3], shape[4] = (int(v) for v in args.hw.split("x"))
    assert shape[3] % 2 == 0 and shape[4] % 2 == 0
    FS = (shape[3] // 2) * (shape[4] // 2)
    pipeline.frame_seq_length = FS
    for m in gen.modules():
        if hasattr(m, "max_attention_size") and hasattr(m, "local_attn_size"):
            m.max_attention_size = (21 * FS) if m.local_attn_size == -1 else m.local_attn_size * FS
    say("reduced geometry: latent %dx%d, %d tokens per frame" % (shape[3], shape[4], FS))
noise = torch.randn(shape, generator=torch.Generator().manual_seed(1234), dtype=torch.float64).to(device=device, dtype=WD)

# schedules: the inference pipeline's list against the trainer's (model/base.py)
sched = gen.get_scheduler()
steps = torch.tensor(list(config.denoising_step_list), dtype=torch.long, device=device)
if config.warp_denoising_step:
    timesteps = torch.cat((sched.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).to(device)
    steps = timesteps[1000 - steps]
same_steps = bool(torch.equal(pipeline.denoising_step_list.double().cpu(), steps.double().cpu()))
ctx_t = int(config.context_noise)
sig = sched.sigmas.to(device)[torch.argmin((sched.timesteps.to(device) - ctx_t).abs())].item()
R["checks"]["schedule"] = dict(inference=[float(x) for x in pipeline.denoising_step_list], training=[float(x) for x in steps],
                               identical=same_steps, sigma_of_training_context_noise=sig,
                               local_attn_size=int(gen.model.local_attn_size), sink_size=int(gen.model.sink_size))
say("SCHEDULE inference %s training %s identical=%s | training context sigma %.5f | local_attn_size %d sink_size %d"
    % ([round(float(x), 3) for x in pipeline.denoising_step_list], [round(float(x), 3) for x in steps], same_steps, sig,
       gen.model.local_attn_size, gen.model.sink_size))
if not same_steps:
    fails.append("inference and training denoising schedules differ")


# ------------------------------------------------------------------ one noise bank for every rollout
class Bank:
    """torch.randn_like replacement: call number -> (block, slot). Slots 0..S-2 are the re-noising draws after the
    denoising steps, the last slot of a training rollout is its context-noise draw."""

    def __init__(self, period):
        self.period, self.calls = period, 0

    def __call__(self, like, **kw):
        frame, slot = divmod(self.calls, self.period)
        self.calls += 1
        assert tuple(like.shape[1:]) == (shape[2], shape[3], shape[4]) and like.shape[0] == shape[0] * NFPB, tuple(like.shape)
        g = torch.Generator().manual_seed(770000 + 16 * frame + slot)
        return torch.randn(tuple(like.shape), generator=g, dtype=torch.float64).to(device=like.device, dtype=like.dtype)


_randn_like = torch.randn_like
S = len(steps)


def rollout(fn, period):
    bank = Bank(period)
    torch.randn_like = bank
    try:
        t0 = time.time()
        out = fn()
        torch.cuda.synchronize()
    finally:
        torch.randn_like = _randn_like
    assert bank.calls == period * (F_ // NFPB), "random draws: %d, expected %d" % (bank.calls, period * (F_ // NFPB))
    return out.detach().to("cpu", torch.float64), time.time() - t0


def run_inference(long_video, sink=None):
    sink = args.kv_cache_sink if sink is None else sink
    def fn():
        return pipeline.inference(noise=noise, text_prompts=[prompt], return_latents=True, long_video=long_video,
                                  kv_cache_max_frames=args.kv_cache_max_frames, kv_cache_sink=sink, kv_cache_train_frames=21,
                                  kv_cache_position_mode="top_aligned")[1]
    out, dt = rollout(fn, S - 1)
    frames = pipeline.kv_cache1[0]["k"].shape[1] // pipeline.frame_seq_length
    pipeline.kv_cache1 = pipeline.crossattn_cache = None
    gc.collect()
    torch.cuda.empty_cache()
    return out, dt, frames


def make_train_pipe():
    """The training rollout exactly as this checkout's trainer builds it (DMD._initialize_inference_pipeline, inherited
    from SelfForcingModel, reads only these attributes), then fixed to exit at the last denoising step."""
    import types
    from model.dmd import DMD as Model                    # the model class of distribution_loss: dmd (trainer/distillation.py)
    ns = types.SimpleNamespace(args=config, denoising_step_list=steps, denoising_step_list_first_chunk=None,
                               scheduler=sched, generator=gen, num_frame_per_block=int(config.num_frame_per_block),
                               num_training_frames=int(getattr(config, "num_training_frames", 21)))
    if hasattr(Model, "_kv_forcing_page_mode"):
        ns._kv_forcing_page_mode = types.MethodType(Model._kv_forcing_page_mode, ns)
    Model._initialize_inference_pipeline(ns)
    pipe = ns.inference_pipeline
    if args.hw:
        pipe.frame_seq_length = FS
        pipe.kv_cache_size = int(getattr(config, "num_training_frames", 21)) * FS
    # exit at the last denoising step. Older one-pass pipelines read last_step_only only on the shared-draw path
    # (per_rank_exit_step False), newer ones on both; with a single process both paths give the last step.
    pipe.last_step_only = True
    if hasattr(pipe, "per_rank_exit_step"):
        pipe.per_rank_exit_step = False
    last = len(pipe.denoising_step_list) - 1
    pipe._sample_exit_step = lambda device: last          # the Self Forcing pipeline ignores last_step_only per rank
    R["training_pipeline"] = type(pipe).__name__
    return pipe


_add_noise = sched.add_noise


def run_training(context_noise_on, with_grad=False):
    pipe = make_train_pipe()
    if not context_noise_on:
        def add_noise(original_samples, noise_, timestep):
            if bool((timestep == ctx_t).all()):
                return original_samples.type_as(noise_)          # the memory write sees the prediction itself, as at inference
            return _add_noise(original_samples, noise_, timestep)
        sched.add_noise = add_noise

    def fn():
        if with_grad:
            gen.model.requires_grad_(True)
            gen.enable_gradient_checkpointing()
            with torch.enable_grad():
                res = pipe.inference_with_trajectory(noise=noise, **cond)
        else:
            res = pipe.inference_with_trajectory(noise=noise, **cond)
        # (output, denoised_timestep_from, denoised_timestep_to): "to" is 0 only when the exit step is the last one
        assert int(res[2]) == 0, "the training rollout did not exit at the last denoising step (to=%s)" % (res[2],)
        return res[0]
    try:
        out, dt = rollout(fn, S)
    finally:
        sched.add_noise = _add_noise
        gen.model.requires_grad_(False)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return out, dt


def compare(label, x, y):
    rel = ((x - y).flatten(2).norm(dim=2)[0] / y.flatten(2).norm(dim=2)[0]).tolist()
    first = next((i for i, v in enumerate(rel) if v > 0), None)
    # growth is judged only where it lands at a visible size: one differing bit spreading to 1e-12 in the next
    # frame is still rounding, a step from 1e-7 to percent level is not
    jump = max([rel[i] / rel[i - 1] for i in range((first or 0) + 1, len(rel)) if rel[i - 1] > 0 and rel[i] >= 1e-3] or [1.0])
    c = dict(per_frame_rel=rel, max_rel=max(rel), first_nonzero_frame=first, onset=rel[first] if first is not None else 0.0,
             largest_growth_factor=jump, max_abs=float((x - y).abs().max()))
    c["rounding_only"] = bool(c["onset"] < 1e-9 and jump < 1e4)
    R["compare"][label] = c
    tag = ("rounding only" if c["rounding_only"] else "NOT rounding") if args.mode == "fp64" else "reported only"
    say("COMPARE %-32s first non-zero frame %-4s onset %.2e | largest growth x%-9.1f | max over frames %.2e | %s"
        % (label, first, c["onset"], jump, c["max_rel"], tag))
    return c


say("=== grad_mode %s | mode %s | %d frames | %s ===" % (config.grad_mode, args.mode, F_, args.device))
A, tA, fA = run_inference(False)
B, tB, fB = run_inference(True)
R["cache_frames"], R["seconds"] = dict(plain=fA, long_video=fB), dict(A=tA, B=tB)
if args.inference_only:
    say("KV cache frames: plain %d, long_video %d | seconds A %.0f B %.0f" % (fA, fB, tA, tB))
    c_ab = compare("A (plain) vs B (long_video)", A, B)
    notes.append("inference only: the training rollout was not compared")
    if args.mode == "fp64" and not c_ab["rounding_only"]:
        fails.append("plain inference differs from --long_video inference beyond rounding: starts at %.2e in frame %s"
                     % (c_ab["onset"], c_ab["first_nonzero_frame"]))
else:
    C, tC = run_training(False)
    D, tD = run_training(True)
    R["seconds"].update(C=tC, D=tD)
    say("KV cache frames: plain %d, long_video %d | training pipeline %s | seconds A %.0f B %.0f C %.0f D %.0f"
        % (fA, fB, R["training_pipeline"], tA, tB, tC, tD))
    c_ac = compare("A (plain) vs C (training)", A, C)
    c_bc = compare("B (long_video) vs C (training)", B, C)
    compare("A (plain) vs B (long_video)", A, B)
    compare("D (context noise on) vs C", D, C)

if args.inference_only:
    pass
elif args.mode == "fp64":
    if not c_bc["rounding_only"]:
        fails.append("--long_video inference is not the training rollout: difference starts at %.2e in frame %s, growth x%.1f"
                     % (c_bc["onset"], c_bc["first_nonzero_frame"], c_bc["largest_growth_factor"]))
    if not c_ac["rounding_only"]:
        fails.append("plain inference is not the training rollout: difference starts at %.2e in frame %s, growth x%.1f"
                     % (c_ac["onset"], c_ac["first_nonzero_frame"], c_ac["largest_growth_factor"]))
else:
    notes.append("bf16: sizes are reported, not judged; run --mode fp64 for the verdict")
    # FSDP mixed precision casts the float timestep to bf16 in training; inference feeds the exact value
    _fwd = gen.forward

    def fwd(*a, **k):
        if torch.is_tensor(k.get("timestep")) and k["timestep"].is_floating_point():
            k["timestep"] = k["timestep"].to(torch.bfloat16)
        return _fwd(*a, **k)
    gen.forward = fwd
    try:
        Bt, _, _ = run_inference(True)
    finally:
        del gen.forward
    compare("B with bf16-rounded timestep vs B", Bt, B)
    try:
        Cg, _ = run_training(False, with_grad=True)       # noqa: F841
        compare("C with gradient enabled vs C", Cg, C)
    except Exception as e:                                          # reported, not a verdict
        notes.append("training rollout with gradient failed: %r" % (e,))

R["fails"], R["notes"], R["seconds_total"] = fails, notes, time.time() - t_start
R["verdict"] = "FAIL" if fails else "PASS"
# slurm-2 2026-10-09: with SAVE_LATENTS=1 also keep the rollouts themselves (A plain, B long_video, C training, D context
# noise), so that two code versions can be compared element by element, not only through their A/B/C summaries.
if OUT and os.environ.get("SAVE_LATENTS") == "1":
    torch.save({k: globals()[k].detach().cpu() for k in ("A", "B", "C", "D") if k in globals()}, OUT[:-5] + "_latents.pt")
if OUT:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(R, f, indent=1, default=str)
for n in notes:
    say("NOTE", n)
for x in fails:
    say("FAIL", x)
say("VERDICT %s (%s, %.0f s)" % (R["verdict"], args.mode, R["seconds_total"]))
sys.exit(1 if fails else 0)
