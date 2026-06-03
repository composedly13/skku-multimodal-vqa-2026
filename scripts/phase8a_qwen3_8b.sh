#!/usr/bin/env bash
# Phase 8A: Qwen3-8B-AWQ 신세대 모델 (현 스택 호환 확인됨).
# 사용:
#   bash scripts/phase8a_qwen3_8b.sh sanity   # 8샘플 스모크(로드/파싱 확인)
#   bash scripts/phase8a_qwen3_8b.sh best     # phase6 검증 레시피 전체추론(모델 교체 효과 분리) ★1순위
#   bash scripts/phase8a_qwen3_8b.sh v2        # v2 원리 프롬프트 전체추론(강한 모델에서 페이오프 검증)
# 주의: dev셋은 abstention 보정에 신뢰 불가(Phase 7 교훈) → public이 판정.
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate challenge_env
cd /mnt/d/skku-multimodal-vqa-2026
MODEL="Qwen/Qwen3-8B-AWQ"
MODE="${1:-sanity}"

case "$MODE" in
  sanity)
    echo "=== Qwen3-8B-AWQ 스모크 8샘플 (best 레시피) ==="
    python -m src.phase2_infer --modality text --model "$MODEL" \
        --few-shot-v2 --balance-line --n 5 --temperature 0.7 \
        --max-samples 8 --output ./outputs/_smoke_q3_8b.csv ;;
  best)
    # phase6_sc_v2balance(0.97567)와 동일 레시피 = base SYSTEM_PROMPT + balance + few-shot-v2 + SC(n5,t0.7).
    # system-v2 안 씀(7B에서 역효과 확인). 모델 업그레이드 효과만 분리.
    echo "=== Qwen3-8B-AWQ best(phase6 레시피) 전체추론 ==="
    python -m src.phase2_infer --modality text --model "$MODEL" \
        --few-shot-v2 --balance-line --n 5 --temperature 0.7 \
        --output ./outputs/phase8_q3_8b_best.csv ;;
  v2)
    echo "=== Qwen3-8B-AWQ v2 원리 프롬프트 전체추론 ==="
    python -m src.phase2_infer --modality text --model "$MODEL" \
        --system-v2 --balance-line \
        --output ./outputs/phase8_q3_8b_v2bal.csv ;;
  *) echo "usage: $0 [sanity|best|v2]"; exit 1 ;;
esac
echo "PHASE8A_${MODE}_DONE"
