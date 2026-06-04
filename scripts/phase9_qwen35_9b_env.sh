#!/usr/bin/env bash
# Phase 9 (S3) — Qwen3.5-9B 격리 env 구축 + 추론.
# 왜 격리 env: 현 challenge_env(transformers 4.56.2, vllm 0.10.2)는 model_type `qwen3_5` 미지원
#   (DESIGN_LOG Phase 8B에서 ValidationError 확인). challenge_env는 불가변(0.991 재현용) →
#   별도 conda env에 transformers>=4.57 + autoawq를 깔고, vLLM이 아닌 순수 transformers+sdpa로 돌린다
#   (공유노트북 0.99517과 동일 경로). 추론 러너는 src/phase9_vlm_infer.py를 그대로 재사용.
#
# 규칙 적격성(2026-06-04 웹검증): Qwen3.5-9B 가중치 2026-03-02 공개 ≤ 5/31 → rule3 적격.
# rule5 준수: 단일 greedy(do_sample=False) = LLM이 최종답 직접 생성. SC/다수결/조건문 미사용.
# rule6: 기준환경 RTX A6000 48GB → 최종은 bf16 통째 적재(autoawq 불필요). 단 로컬 dev(5080 16GB)는 AWQ로.
#   ⇒ DEV_MODEL(로컬 검증, AWQ ~6GB) / FINAL_MODEL(A6000 제출, bf16 ~18GB) 분리.
#
# ⚠️ 고위험: 버전 핀 실측 미검증. ① env+로드 검증 막히면 즉시 접고 challenge_env 0.98925(합법 안전판) 유지.
#
# 사용 (WSL, 저장소 루트):
#   bash scripts/phase9_qwen35_9b_env.sh setup     # ① 격리 env 생성 + 패키지 설치
#   bash scripts/phase9_qwen35_9b_env.sh verify    # ② qwen3_5 model_type 로드만 검증(추론 X)
#   bash scripts/phase9_qwen35_9b_env.sh smoke      # ③ 8샘플 추론 (DEV_MODEL=AWQ)
#   bash scripts/phase9_qwen35_9b_env.sh full        # ④ 전체 8500 (이미지 포함, DEV_MODEL=AWQ)
# 최종 제출 코드(A6000)에서는 MODEL을 FINAL_MODEL(bf16)로 바꿔 재현.
set -euo pipefail
ENV_NAME="challenge_q35"
DEV_MODEL="QuantTrio/Qwen3.5-9B-AWQ"   # 로컬 5080 16GB 검증용 AWQ 4bit (~6GB). HF 실존 확인 후 사용.
FINAL_MODEL="Qwen/Qwen3.5-9B"          # A6000 48GB 최종 제출용 bf16 (~18GB). rule6 환경 기준.
MODEL="$DEV_MODEL"
REPO="/mnt/d/skku-multimodal-vqa-2026"
source /root/miniconda3/etc/profile.d/conda.sh
MODE="${1:-setup}"

case "$MODE" in
  setup)
    echo "=== ① 격리 env '$ENV_NAME' 생성 ==="
    conda create -y -n "$ENV_NAME" python=3.11
    conda activate "$ENV_NAME"
    # cu128 torch (Blackwell sm_120). 채널 버전은 환경에 맞게 조정.
    pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
    # qwen3_5 지원하는 transformers (>=4.57 또는 5.x). dev 핀이 필요하면 git+ 사용.
    pip install "transformers>=4.57.0" accelerate autoawq qwen-vl-utils pillow pandas tqdm
    echo "SETUP_DONE — 다음: bash $0 verify" ;;
  verify)
    echo "=== ② qwen3_5 로드 검증(설정만, 가중치 미다운로드 가능성) ==="
    conda activate "$ENV_NAME"
    python - <<PY
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("${MODEL}")
print("model_type:", getattr(cfg, "model_type", "?"))
print("OK: config 로드 성공 → qwen3_5 인식됨")
PY
    echo "VERIFY_DONE — 막히면 즉시 중단하고 challenge_env 0.991 유지" ;;
  probe)
    # 가중치 다운로드 없이 추론 스택 준비상태만 점검(transformers/torch CUDA/의존성).
    echo "=== ②.5 추론 스택 probe (다운로드 없음) ==="
    conda activate "$ENV_NAME"
    python - <<PY
import importlib, torch, transformers
print("transformers:", transformers.__version__)
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), "| capability sm_%d%d" % torch.cuda.get_device_capability(0))
    try:
        x = (torch.randn(8,8,device="cuda") @ torch.randn(8,8,device="cuda")).sum().item()
        print("cuda matmul OK:", round(x,3))
    except Exception as e:
        print("cuda matmul FAIL:", e)
for m in ["awq", "qwen_vl_utils", "PIL", "pandas", "accelerate"]:
    try:
        importlib.import_module(m); print(f"import {m}: OK")
    except Exception as e:
        print(f"import {m}: MISSING -> {e}")
PY
    echo "PROBE_DONE" ;;
  smoke)
    # tf5.x는 AWQ 백엔드(autoawq/gptqmodel) 깨짐 → bf16 원본을 bitsandbytes nf4 4bit로 로컬 적재.
    echo "=== ③ 8샘플 스모크 (이미지 포함, bf16원본 4bit 적재) ==="
    conda activate "$ENV_NAME"; cd "$REPO"
    python -m src.phase9_vlm_infer --model "$FINAL_MODEL" --load-4bit \
        --max-samples 8 --output ./outputs/_smoke_q35_9b.csv \
        --dump-raw ./outputs/_smoke_q35_9b_raw.csv ;;
  full)
    echo "=== ④ 전체 8500 추론 (이미지 포함, bf16원본 4bit 적재) ==="
    conda activate "$ENV_NAME"; cd "$REPO"
    python -m src.phase9_vlm_infer --model "$FINAL_MODEL" --load-4bit \
        --output ./outputs/phase9_q35_9b_image.csv \
        --dump-raw ./outputs/phase9_q35_9b_image_raw.csv ;;
  *) echo "usage: $0 [setup|verify|probe|smoke|full]"; exit 1 ;;
esac
