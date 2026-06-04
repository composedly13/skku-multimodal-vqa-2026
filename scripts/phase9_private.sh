#!/usr/bin/env bash
# Phase 9: private(40%) 최대화 + 0.99517 격차 좁히기.
# 전제: 최종순위=private. best-public 자동선택 가정 → 원리적 개선 후보만 제출.
# 사용 (WSL conda challenge_env, 저장소 루트):
#   bash scripts/phase9_private.sh sc9      # S1: SC n=9 분산축소 (현 0.991 미세개선 후보) ★바로가능
#   bash scripts/phase9_private.sh sanity   # 로드/파싱 8샘플 스모크
# 주의: dev셋은 abstention 보정에 신뢰 불가(Phase 7 교훈). public은 참고신호 1개.
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate challenge_env
cd /mnt/d/skku-multimodal-vqa-2026
MODEL="Qwen/Qwen3-8B-AWQ"
MODE="${1:-sanity}"

case "$MODE" in
  sanity)
    echo "=== Qwen3-8B-AWQ 스모크 8샘플 (v2+balance+SC n9) ==="
    python -m src.phase2_infer --modality text --model "$MODEL" \
        --system-v2 --balance-line --n 9 --temperature 0.7 \
        --max-samples 8 --output ./outputs/_smoke_p9_sc9.csv ;;
  sc9)
    # 현 베스트(0.991, n5)의 SC 샘플수만 9로 올려 경계노이즈 추가정리. 평균화라 private 안전.
    echo "=== Qwen3-8B-AWQ v2+balance+SC(n9) 전체추론 ==="
    python -m src.phase2_infer --modality text --model "$MODEL" \
        --system-v2 --balance-line --n 9 --temperature 0.7 \
        --output ./outputs/phase9_q3_8b_v2bal_sc9.csv ;;
  *) echo "usage: $0 [sanity|sc9]"; exit 1 ;;
esac
echo "PHASE9_${MODE}_DONE"
