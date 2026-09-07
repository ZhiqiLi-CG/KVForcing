#!/usr/bin/env bash
# Download the Wan base models, the Causal-Forcing training initializations and the training prompt list.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

HF_REPO="${HF_REPO:-JunhaoZhuang/Self_Gradient_Forcing}"
HF_CLI="${HF_CLI:-hf}"

mkdir -p checkpoints/init/framewise wan_models prompts

"$HF_CLI" download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
"$HF_CLI" download Wan-AI/Wan2.1-T2V-14B --local-dir wan_models/Wan2.1-T2V-14B

# Causal-Forcing initialization checkpoints (framewise). causal_cd.pt is the default in configs/.
"$HF_CLI" download "$HF_REPO" init/framewise/ar_diffusion.pt --local-dir checkpoints
"$HF_CLI" download "$HF_REPO" init/framewise/causal_cd.pt --local-dir checkpoints
"$HF_CLI" download "$HF_REPO" init/framewise/causal_ode.pt --local-dir checkpoints


# Training prompt list.
"$HF_CLI" download "$HF_REPO" vidprom_filtered_extended.txt --local-dir prompts

cat <<INFO

Done.
Framewise training initializations:
  checkpoints/init/framewise/ar_diffusion.pt
  checkpoints/init/framewise/causal_cd.pt
  checkpoints/init/framewise/causal_ode.pt
Training prompt list: prompts/vidprom_filtered_extended.txt
INFO
