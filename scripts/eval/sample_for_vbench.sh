#!/usr/bin/env bash
# Sample videos for OFFICIAL VBench evaluation.
#
# VBench's standard mode locates videos by filename: it looks for
# "<full prompt>-<i>.mp4" with i in 0..4 (see VBench/vbench/__init__.py,
# build_full_info_json). So generation must use --vbench_naming (full,
# untruncated prompt) and --num_samples 5. The default naming
# "<prompt[:100]>.mp4" collides for 21 of the 946 official prompts and is
# invisible to VBench.
#
# Two things keep the sampled videos consistent with how the model was trained
# (scripts/eval/check_train_eval_consistency.py verifies the first):
#   * every suite runs the inference mode scripts/infer.sh runs: --long_video
#     with a 21-frame window (sink + FIFO + current). For a 21-frame clip that
#     is exactly the training rollout's attention.
#   * the short suites generate from the EXTENDED prompts (the model has only
#     seen long, LLM-written prompts) and name the files after the official
#     short prompts, as the official Self-Forcing protocol does
#     (prompts/vbench/README.md). EXTENDED_PROMPTS=none samples from the short
#     prompts instead.
#
# Usage:
#   bash scripts/eval/sample_for_vbench.sh <suite> <checkpoint.pt> [num_gpus]
#
#   suite    short         -> 944 official VBench prompts, 5 samples each, 5s videos
#            short_flicker -> the 75 temporal_flickering prompts, 25 samples each
#                             (VBench's static filter needs them; run it after
#                             `short` into the SAME folder, samples 0-4 are kept)
#            long60        -> 128 MovieGen prompts, 1 sample each, 60s videos
#            long240       -> 128 MovieGen prompts, 1 sample each, 240s videos
#
# Example:
#   bash scripts/eval/sample_for_vbench.sh long60 logs/kv_forcing/checkpoint_model_001500/model.pt 8
#   DRY_RUN=1 bash scripts/eval/sample_for_vbench.sh short <ckpt>     # print the command only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"

SUITE="${1:?suite must be short, short_flicker, long60 or long240}"
CKPT="${2:?path to checkpoint .pt}"
NUM_GPUS="${3:-${NUM_GPUS:-1}}"

CONFIG="${CONFIG:-configs/kv_forcing_framewise.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/vbench}"

# Wan VAE maps L latent frames to 4*(L-1)+1 pixel frames; at 16 fps that is
# (4L-3)/16 seconds. 21 -> 5s (the trained window), 241 -> 60s, 963 -> 240s.
case "$SUITE" in
    short|short_flicker)
        if [[ "$SUITE" == "short" ]]; then
            PROMPTS="${PROMPTS:-prompts/vbench/all_dimension_unique.txt}"
            EXTENDED_PROMPTS="${EXTENDED_PROMPTS:-prompts/vbench/all_dimension_unique_extended.txt}"
            NUM_SAMPLES="${NUM_SAMPLES:-5}"
            SEED="${SEED:-0}"
        else
            PROMPTS="${PROMPTS:-prompts/vbench/temporal_flickering.txt}"
            EXTENDED_PROMPTS="${EXTENDED_PROMPTS:-prompts/vbench/temporal_flickering_extended.txt}"
            NUM_SAMPLES="${NUM_SAMPLES:-25}"
            # The seed of a video is SEED + prompt index * NUM_SAMPLES + sample
            # index. `short` uses 0..4719; another base here, so that samples
            # 5-24 of a prompt can never repeat the seed of its samples 0-4.
            SEED="${SEED:-1000000}"
        fi
        NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-21}"
        # Standard mode matches videos by full-prompt filename, so names must
        # not be truncated. The official prompts top out at 210 chars, which
        # still fits the 255-byte filename limit.
        NAMING_ARGS=(--vbench_naming)
        OUTPUT_SUITE=short
        ;;
    long60|long240)
        PROMPTS="${PROMPTS:-prompts/moviegen_128.txt}"
        # The MovieGen prompts in prompts/moviegen_128.txt are already the
        # extended ones.
        EXTENDED_PROMPTS="${EXTENDED_PROMPTS:-none}"
        NUM_SAMPLES="${NUM_SAMPLES:-1}"
        SEED="${SEED:-0}"
        # MovieGen prompts run to ~850 chars, past the 255-byte filename limit,
        # so full-prompt naming is impossible here. VBench-Long's custom mode
        # scores only prompt-free dimensions (consistency / motion / quality),
        # so the default "<prompt[:100]>.mp4" name is sufficient -- it just has
        # to be unique, which is verified below.
        NAMING_ARGS=()
        [[ "$SUITE" == "long60" ]] && NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-241}" \
                                   || NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-963}"
        OUTPUT_SUITE="$SUITE"
        ;;
    *)
        echo "ERROR: suite must be short, short_flicker, long60 or long240 (got '$SUITE')" >&2
        exit 1
        ;;
esac

# Streaming KV-cache geometry, same defaults as scripts/infer.sh, for EVERY
# suite. Without --long_video the cache of the plain mode used to be 3 frames
# too small for this model (local_attn_size 18 + sink_size 3), which cut the
# history of the last 3 latent frames of every 5s video; the plain mode is
# fixed now (pipeline/causal_inference.py), and the suites use one mode anyway.
KV_SINK="${KV_CACHE_SINK:-4}"; KV_FIFO="${KV_CACHE_FIFO_FRAMES:-16}"; KV_CUR="${KV_CACHE_CURRENT_FRAMES:-1}"
LONG_ARGS=(
    --long_video
    --kv_cache_max_frames $((KV_SINK + KV_FIFO + KV_CUR))
    --kv_cache_sink "$KV_SINK"
    --kv_cache_train_frames 21
    --kv_cache_position_mode top_aligned
)

EXT_ARGS=()
if [[ "$EXTENDED_PROMPTS" != "none" ]]; then
    [[ -f "$EXTENDED_PROMPTS" ]] || { echo "ERROR: extended prompt file not found: $EXTENDED_PROMPTS" >&2; exit 1; }
    EXT_ARGS=(--extended_prompt_path "$EXTENDED_PROMPTS")
fi

[[ -f "$CONFIG" ]]  || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
[[ -f "$CKPT" ]]    || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }
[[ -f "$PROMPTS" ]] || { echo "ERROR: prompt file not found: $PROMPTS" >&2; exit 1; }

# A silent filename collision would make VBench score one prompt twice and miss
# another, so refuse to start unless every output name is distinct and legal,
# and unless the extended prompts line up with the short ones.
python3 - "$PROMPTS" "${#NAMING_ARGS[@]}" "$EXTENDED_PROMPTS" <<'PY'
import sys
# parsed exactly as utils/dataset.py TextDataset does
prompts = [l.rstrip() for l in open(sys.argv[1], encoding='utf-8')]
full_naming = int(sys.argv[2]) > 0
if any(not p for p in prompts):
    sys.exit("ERROR: the prompt file has an empty line.")
name_of = (lambda p: p) if full_naming else (lambda p: p[:100])
# Two DIFFERENT prompts must never share an output name (one would be scored
# twice and the other dropped), and with extended prompts the SAME prompt must
# not appear twice either: its two lines may carry different extended versions,
# and which one the video shows would then depend on which GPU got there first.
by_name = {}
for p in prompts:
    by_name.setdefault(name_of(p), []).append(p)
clashes = {n: ps for n, ps in by_name.items() if len(set(ps)) > 1}
if clashes:
    for n, ps in list(clashes.items())[:3]:
        print(f"  {n!r} <- {len(set(ps))} distinct prompts", file=sys.stderr)
    sys.exit(f"ERROR: {len(clashes)} output names are shared by different prompts.")
too_long = [n for n in by_name if len(n.encode()) + len('-24.mp4') > 255]
if too_long:
    sys.exit(f"ERROR: {len(too_long)} filenames exceed the 255-byte limit.")
if sys.argv[3] != "none":
    ext = [l.rstrip() for l in open(sys.argv[3], encoding='utf-8')]
    if len(ext) != len(prompts) or any(not e for e in ext):
        sys.exit(f"ERROR: {len(ext)} extended prompts for {len(prompts)} prompts (or an empty line).")
    repeated = [n for n, ps in by_name.items() if len(ps) > 1]
    if repeated:
        sys.exit(f"ERROR: {len(repeated)} prompts are listed more than once; with extended prompts each "
                 "must appear once (see prompts/vbench/README.md).")
print(f"filename check OK: {len(prompts)} prompts -> {len(by_name)} unique names")
PY

OUTPUT_FOLDER="${OUTPUT_FOLDER:-${OUTPUT_ROOT}/${OUTPUT_SUITE}/videos}"

USE_EMA="${USE_EMA:-1}"
EMA_ARGS=()
[[ "$USE_EMA" == "1" ]] && EMA_ARGS=(--use_ema)

export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"

ARGS=(
    inference.py
    --config_path "$CONFIG"
    --checkpoint_path "$CKPT"
    --data_path "$PROMPTS"
    "${EXT_ARGS[@]}"
    --output_folder "$OUTPUT_FOLDER"
    --num_output_frames "$NUM_OUTPUT_FRAMES"
    --num_samples "$NUM_SAMPLES"
    "${NAMING_ARGS[@]}"
    --seed "$SEED"
    "${LONG_ARGS[@]}"
    "${EMA_ARGS[@]}"
)

cat <<INFO
==========================================
Sampling videos for official VBench
Suite:          $SUITE
Config:         $CONFIG
Checkpoint:     $CKPT
Prompts:        $PROMPTS ($(grep -c . "$PROMPTS") prompts)
Generated from: $([[ "$EXTENDED_PROMPTS" == "none" ]] && echo "the same prompts" || echo "$EXTENDED_PROMPTS")
Samples/prompt: $NUM_SAMPLES  (seed base $SEED)
Latent frames:  $NUM_OUTPUT_FRAMES  (~$(python3 -c "print(f'{(4*$NUM_OUTPUT_FRAMES-3)/16:.0f}')")s at 16 fps)
KV cache:       sink=$KV_SINK, fifo=$KV_FIFO, current=$KV_CUR
GPUs:           $NUM_GPUS
Output:         $OUTPUT_FOLDER
==========================================
INFO

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY RUN, would execute (%s GPU):\n  ' "$NUM_GPUS"; printf '%q ' "${ARGS[@]}"; echo
    exit 0
fi
mkdir -p "$OUTPUT_FOLDER"

if (( NUM_GPUS > 1 )); then
    unset RANK WORLD_SIZE MASTER_ADDR MASTER_PORT NODE_RANK LOCAL_WORLD_SIZE GROUP_RANK
    torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" "${ARGS[@]}"
else
    python "${ARGS[@]}"
fi

echo "[done] $(find "$OUTPUT_FOLDER" -maxdepth 1 -name '*.mp4' | wc -l) videos in $OUTPUT_FOLDER"
