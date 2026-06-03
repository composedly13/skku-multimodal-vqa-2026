#!/usr/bin/env bash
# Phase 7 dev셋 평가 러너. WSL conda challenge_env에서 실행.
# 사용: wsl -d Ubuntu-24.04 -u root -- bash -lc "bash /mnt/d/skku-multimodal-vqa-2026/scripts/eval_dev.sh"
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate challenge_env
cd /mnt/d/skku-multimodal-vqa-2026
python -m src.build_dev_set
python -m src.eval_dev "$@"
