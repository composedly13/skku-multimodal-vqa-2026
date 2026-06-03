#!/usr/bin/env bash
# Phase 7 전체추론(7B 원리 프롬프트). WSL conda challenge_env.
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate challenge_env
cd /mnt/d/skku-multimodal-vqa-2026
echo "=== 7B v2+bal (dev best 42/44) ==="
python -m src.phase2_infer --modality text --system-v2 --balance-line \
    --output ./outputs/phase7_v2bal.csv
echo "=== 7B v2 ==="
python -m src.phase2_infer --modality text --system-v2 \
    --output ./outputs/phase7_v2.csv
echo "PHASE7_DONE"
