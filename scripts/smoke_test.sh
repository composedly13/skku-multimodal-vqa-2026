#!/usr/bin/env bash
# 스모크 테스트 — 환경이 "지금도" 동작하는지 재판정한다.
#   1) GPU/torch가 RTX 5080(sm_120)을 인식하는지
#   2) vLLM이 LLaVA-OneVision을 로드해 소량 추론 후 제출 형식 CSV를 만드는지
#
# 사용 (WSL Ubuntu 셸, 저장소 루트):
#   bash scripts/smoke_test.sh
#
# 종료 코드 0 = PASS, 그 외 = FAIL.
set -uo pipefail

ENV_NAME="challenge_env"
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
N="${1:-4}"   # 추론 샘플 수 (기본 4)

# conda 활성화 (PATH에 없으면 일반 설치 경로 탐색 — 비대화형 셸에서도 동작)
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  for c in "$HOME/miniconda3" "$HOME/anaconda3" /root/miniconda3 /opt/conda; do
    [ -f "$c/etc/profile.d/conda.sh" ] && source "$c/etc/profile.d/conda.sh" && break
  done
fi
conda activate "${ENV_NAME}"
cd "${PROJ_DIR}"

echo "== [1/3] torch / GPU 점검 =="
python - <<'PY'
import sys, torch
ok = torch.cuda.is_available() and (12, 0) in [torch.cuda.get_device_capability(0)] \
     and "sm_120" in torch.cuda.get_arch_list()
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "| device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
      "| sm_120 in arch:", "sm_120" in torch.cuda.get_arch_list())
sys.exit(0 if ok else 1)
PY
[ $? -eq 0 ] || { echo "FAIL: torch가 RTX 5080(sm_120)을 인식하지 못함"; exit 1; }

echo "== [2/3] 추론 ${N}개 실행 =="
rm -f outputs/baseline_submission.csv
GPU_MODE=gpu0 python -m src.inference --max-samples "${N}" --batch-size "${N}"
[ $? -eq 0 ] || { echo "FAIL: 추론 실행 중 오류"; exit 1; }

echo "== [3/3] 제출 파일 형식 점검 =="
python - "${N}" <<'PY'
import sys, csv, pathlib
n = int(sys.argv[1])
p = pathlib.Path("outputs/baseline_submission.csv")
if not p.exists():
    print("FAIL: 제출 파일 없음"); sys.exit(1)
rows = list(csv.DictReader(p.open()))
assert rows and list(rows[0].keys()) == ["sample_id", "label"], f"헤더 불일치: {rows[:1]}"
assert len(rows) == n, f"행 수 불일치: {len(rows)} != {n}"
assert all(r["label"] in {"0", "1", "2"} for r in rows), "label이 0/1/2가 아님"
print(f"OK: {len(rows)}행, 컬럼 {list(rows[0].keys())}, 예시 {rows[0]}")
PY
[ $? -eq 0 ] || { echo "FAIL: 제출 파일 형식 불일치"; exit 1; }

echo "SMOKE TEST PASS ✅"
