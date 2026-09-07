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
# Usage:
#   bash scripts/eval/sample_for_vbench.sh <suite> <checkpoint.pt> [num_gpus]
#
#   suite    short   -> 946 official VBench prompts, 5 samples each, 5s videos
#            long60  -> 128 MovieGen prompts, 1 sample each, 60s videos
#            long240 -> 128 MovieGen prompts, 1 sample each, 240s videos
#
# Example:
#   bash scripts/eval/sample_for_vbench.sh long60 logs/kv_forcing/checkpoint_model_001500/model.pt 8
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"

SUITE="${1:?suite must be short, long60 or long240}"
CKPT="${2:?path to checkpoint .pt}"
NUM_GPUS="${3:-${NUM_GPUS:-1}}"

CONFIG="${CONFIG:-configs/kv_forcing_framewise.yaml}"
SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/vbench}"

# Wan VAE maps L latent frames to 4*(L-1)+1 pixel frames; at 16 fps that is
# (4L-3)/16 seconds. 21 -> 5s (the trained window), 241 -> 60s, 963 -> 240s.
case "$SUITE" in
    short)
        PROMPTS="${PROMPTS:-VBench/prompts/all_dimension.txt}"
        NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-21}"
        NUM_SAMPLES="${NUM_SAMPLES:-5}"
        # Standard mode matches videos by full-prompt filename, so names must
        # not be truncated. The official prompts top out at 210 chars, which
        # still fits the 255-byte filename limit.
        NAMING_ARGS=(--vbench_naming)
        LONG_ARGS=()
        ;;
    long60|long240)
        PROMPTS="${PROMPTS:-prompts/moviegen_128.txt}"
        NUM_SAMPLES="${NUM_SAMPLES:-1}"
        # MovieGen prompts run to ~850 chars, past the 255-byte filename limit,
        # so full-prompt naming is impossible here. VBench-Long's custom mode
        # scores only prompt-free dimensions (consistency / motion / quality),
        # so the default "<prompt[:100]>.mp4" name is sufficient -- it just has
        # to be unique, which is verified below.
        NAMING_ARGS=()
        [[ "$SUITE" == "long60" ]] && NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-241}" \
                                   || NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-963}"
        # Long-video KV-cache geometry, same defaults as scripts/infer.sh.
        KV_SINK="${KV_CACHE_SINK:-4}"; KV_FIFO="${KV_CACHE_FIFO_FRAMES:-16}"; KV_CUR="${KV_CACHE_CURRENT_FRAMES:-1}"
        LONG_ARGS=(
            --long_video
            --kv_cache_max_frames $((KV_SINK + KV_FIFO + KV_CUR))
            --kv_cache_sink "$KV_SINK"
            --kv_cache_train_frames 21
            --kv_cache_position_mode top_aligned
        )
        ;;
    *)
        echo "ERROR: suite must be short, long60 or long240 (got '$SUITE')" >&2
        exit 1
        ;;
esac

[[ -f "$CONFIG" ]]  || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
[[ -f "$CKPT" ]]    || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }
[[ -f "$PROMPTS" ]] || { echo "ERROR: prompt file not found: $PROMPTS" >&2; exit 1; }

# A silent filename collision would make VBench score one prompt twice and miss
# another, so refuse to start unless every output name is distinct and legal.
python3 - "$PROMPTS" "${#NAMING_ARGS[@]}" <<'PY'
import sys
prompts = [l.strip() for l in open(sys.argv[1], encoding='utf-8') if l.strip()]
full_naming = int(sys.argv[2]) > 0
name_of = (lambda p: p) if full_naming else (lambda p: p[:100])
# A prompt repeated in the list is fine and expected: all_dimension.txt
# concatenates the per-dimension files, so a prompt used by two dimensions
# appears twice and its one video is scored under both. What must never happen
# is two DIFFERENT prompts sharing an output name, which would score one prompt
# twice and drop the other.
by_name = {}
for p in prompts:
    by_name.setdefault(name_of(p), set()).add(p)
clashes = {n: ps for n, ps in by_name.items() if len(ps) > 1}
if clashes:
    for n, ps in list(clashes.items())[:3]:
        print(f"  {n!r} <- {len(ps)} distinct prompts", file=sys.stderr)
    sys.exit(f"ERROR: {len(clashes)} output names are shared by different prompts.")
too_long = [n for n in by_name if len(n.encode()) + len('-0.mp4') > 255]
if too_long:
    sys.exit(f"ERROR: {len(too_long)} filenames exceed the 255-byte limit.")
print(f"filename check OK: {len(prompts)} prompts -> {len(by_name)} unique names")
PY

OUTPUT_FOLDER="${OUTPUT_FOLDER:-${OUTPUT_ROOT}/${SUITE}/videos}"
mkdir -p "$OUTPUT_FOLDER"

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
Samples/prompt: $NUM_SAMPLES
Latent frames:  $NUM_OUTPUT_FRAMES  (~$(python3 -c "print(f'{(4*$NUM_OUTPUT_FRAMES-3)/16:.0f}')")s at 16 fps)
GPUs:           $NUM_GPUS
Output:         $OUTPUT_FOLDER
==========================================
INFO

if (( NUM_GPUS > 1 )); then
    unset RANK WORLD_SIZE MASTER_ADDR MASTER_PORT NODE_RANK LOCAL_WORLD_SIZE GROUP_RANK
    torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" "${ARGS[@]}"
else
    python "${ARGS[@]}"
fi

echo "[done] $(find "$OUTPUT_FOLDER" -maxdepth 1 -name '*.mp4' | wc -l) videos in $OUTPUT_FOLDER"
