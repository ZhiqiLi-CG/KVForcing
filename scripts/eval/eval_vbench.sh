#!/usr/bin/env bash
# Run the OFFICIAL VBench evaluator on sampled videos.
#
# Nothing here reimplements a metric: this calls upstream's own entrypoints
# from a clone of github.com/Vchitect/VBench at VBench/ (not vendored here:
#   git clone https://github.com/Vchitect/VBench.git VBench
# and install it into its own environment), run exactly as its README documents.
#   short  -> VBench/evaluate.py              (standard suite, prompt-matched)
#   custom -> VBench/evaluate.py              (scores whatever videos are present)
#   long   -> VBench/vbench2_beta_long/eval_long.py --mode long_custom_input
#
# "short" only yields a comparable number when the FULL 946-prompt suite has
# been sampled: standard mode walks the official prompt list, warns about every
# video it cannot find, and scores a dimension over zero videos if none of its
# prompts were generated. Use "custom" for partial or ad-hoc video sets.
#
# VBench needs its own environment: it pins transformers==4.33.2 and numpy<2,
# which would break training/inference here, so it lives in a separate venv
# whose interpreter is passed as VBENCH_PYTHON.
#
# Usage:
#   bash scripts/eval/eval_vbench.sh <short|long> <videos_dir> [output_dir] [dimensions...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VBENCH_DIR="$PROJECT_ROOT/VBench"

MODE="${1:?mode must be short, custom or long}"
VIDEOS_DIR="${2:?path to the folder of sampled videos}"
OUTPUT_DIR="${3:-$(dirname "$VIDEOS_DIR")/vbench_results}"
shift 3 2>/dev/null || shift $#

VBENCH_PYTHON="${VBENCH_PYTHON:?set VBENCH_PYTHON to the python of the VBench environment}"
# The evaluator downloads multi-GB checkpoints into this cache.
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-$PROJECT_ROOT/.vbench_cache}"
export TORCH_HOME="${TORCH_HOME:-$VBENCH_CACHE_DIR/torch}"
mkdir -p "$VBENCH_CACHE_DIR" "$OUTPUT_DIR"

[[ -x "$VBENCH_PYTHON" ]] || { echo "ERROR: VBench python not found: $VBENCH_PYTHON" >&2; exit 1; }
[[ -d "$VIDEOS_DIR" ]]    || { echo "ERROR: videos dir not found: $VIDEOS_DIR" >&2; exit 1; }

# VBench's own list of dimensions that work on arbitrary (non-standard-suite)
# videos. The 6 long-video dimensions are the ones reported for 60s/240s.
CUSTOM_DIMENSIONS=(subject_consistency background_consistency motion_smoothness
                   dynamic_degree aesthetic_quality imaging_quality)
STANDARD_DIMENSIONS=(subject_consistency background_consistency temporal_flickering
                     motion_smoothness dynamic_degree aesthetic_quality imaging_quality
                     object_class multiple_objects human_action color spatial_relationship
                     scene appearance_style temporal_style overall_consistency)

cd "$VBENCH_DIR"
export PYTHONPATH="$VBENCH_DIR:${PYTHONPATH:-}"

case "$MODE" in
    short)
        DIMENSIONS=("${@:-${STANDARD_DIMENSIONS[@]}}")
        [[ $# -gt 0 ]] && DIMENSIONS=("$@")
        echo "== official VBench (standard suite) =="
        echo "videos:     $VIDEOS_DIR"
        echo "dimensions: ${DIMENSIONS[*]}"
        echo "output:     $OUTPUT_DIR"
        "$VBENCH_PYTHON" evaluate.py \
            --videos_path "$VIDEOS_DIR" \
            --output_path "$OUTPUT_DIR" \
            --mode vbench_standard \
            --dimension "${DIMENSIONS[@]}"
        ;;
    custom)
        DIMENSIONS=("${CUSTOM_DIMENSIONS[@]}")
        [[ $# -gt 0 ]] && DIMENSIONS=("$@")
        echo "== official VBench (custom_input) =="
        echo "videos:     $VIDEOS_DIR"
        echo "dimensions: ${DIMENSIONS[*]}"
        echo "output:     $OUTPUT_DIR"
        "$VBENCH_PYTHON" evaluate.py \
            --videos_path "$VIDEOS_DIR" \
            --output_path "$OUTPUT_DIR" \
            --mode custom_input \
            --dimension "${DIMENSIONS[@]}"
        ;;
    long)
        DIMENSIONS=("${CUSTOM_DIMENSIONS[@]}")
        [[ $# -gt 0 ]] && DIMENSIONS=("$@")
        echo "== official VBench-Long (long_custom_input) =="
        echo "videos:     $VIDEOS_DIR"
        echo "dimensions: ${DIMENSIONS[*]}"
        echo "output:     $OUTPUT_DIR"
        # One dimension per call with --dev_flag, matching upstream's own
        # vbench2_beta_long/evaluate_long.sh. --num_of_samples_per_prompt
        # defaults to 5 upstream; long videos are sampled once per prompt.
        SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
        for dim in "${DIMENSIONS[@]}"; do
            echo "--- $dim ---"
            "$VBENCH_PYTHON" vbench2_beta_long/eval_long.py \
                --videos_path "$VIDEOS_DIR" \
                --output_path "$OUTPUT_DIR" \
                --mode long_custom_input \
                --dimension "$dim" \
                --num_of_samples_per_prompt "$SAMPLES_PER_PROMPT" \
                --dev_flag
        done
        ;;
    *)
        echo "ERROR: mode must be short, custom or long (got '$MODE')" >&2
        exit 1
        ;;
esac

echo "[done] results in $OUTPUT_DIR"
ls -1 "$OUTPUT_DIR" | head -20
