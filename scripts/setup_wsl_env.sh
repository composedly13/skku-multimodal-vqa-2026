#!/usr/bin/env bash
# WSL2(Ubuntu 24.04) + conda 에서 RTX 5080(Blackwell, sm_120) 호환 추론 환경을 구성한다.
# (RTX 5080 + Windows 환경에서 실제 검증된 절차)
#
# 전제:
#   - Windows에 WSL2 + Ubuntu-24.04 설치 완료
#       PowerShell> wsl --install -d Ubuntu-24.04
#   - Windows NVIDIA 드라이버가 WSL CUDA를 제공 (WSL 안에서 nvidia-smi 동작)
#   - Ubuntu 안에 Miniconda 설치 완료
#
# 사용 (WSL Ubuntu 셸에서):
#   bash scripts/setup_wsl_env.sh
set -euo pipefail

ENV_NAME="challenge_env"
PY_VER="3.12"
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# 0) torch.compile(inductor)이 사용하는 C 컴파일러 — 최소 설치 Ubuntu엔 없으므로 설치
if ! command -v gcc >/dev/null 2>&1; then
  echo "[setup] installing build-essential (gcc) ..."
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential
fi

# 1) conda 활성화 (PATH에 없으면 일반 설치 경로 탐색 — 비대화형 셸에서도 동작)
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  for c in "$HOME/miniconda3" "$HOME/anaconda3" /root/miniconda3 /opt/conda; do
    [ -f "$c/etc/profile.d/conda.sh" ] && source "$c/etc/profile.d/conda.sh" && break
  done
fi

# 2) ToS 수락(conda 26+) + 환경 생성
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
if ! conda env list | grep -qE "/${ENV_NAME}\$"; then
  conda create -y -n "${ENV_NAME}" python="${PY_VER}"
fi
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip

# 3) Blackwell 호환 스택 설치 (requirements.txt 안에 cu128 extra-index 포함)
pip install -r "${PROJ_DIR}/requirements.txt"

# 4) 검증
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("arch_list:", torch.cuda.get_arch_list())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0),
          "capability:", torch.cuda.get_device_capability(0))
PY

echo "[setup] DONE — activate with: conda activate ${ENV_NAME}"
echo "[setup] run inference:  cd ${PROJ_DIR} && python -m src.inference --max-samples 4"
