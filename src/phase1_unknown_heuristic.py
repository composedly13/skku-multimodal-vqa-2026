"""Phase 1 — Unknown-우선 휴리스틱 (모델 불필요, 순수 규칙).

각 문항의 보기(answers)에서 "정보 부족/판단 불가" 옵션의 인덱스를 찾아 그대로 label로 출력한다.
BBQ 계열 데이터에서 "근거가 없으면 unknown" 규칙이 정답인지 검증하기 위한 천장 측정용 베이스라인.

사용 (저장소 루트):
    python -m src.phase1_unknown_heuristic
    python -m src.phase1_unknown_heuristic --data-csv ./data/test/test.csv --output ./outputs/phase1_unknown_submission.csv
"""

import argparse
import json
import re
from collections import Counter

import pandas as pd

# "정보 부족 / 판단 불가 / 알 수 없음" 류 옵션 탐지 패턴 (대소문자 무시)
UNKNOWN_RE = re.compile(
    r"cannot|not enough|unknown|undetermin|can.t|determined|"
    r"not answerable|no answer|not known|insufficient|unclear|unidentifi",
    re.IGNORECASE,
)


def find_unknown_index(answers: list[str]) -> int | None:
    """보기 리스트에서 unknown 옵션의 인덱스를 반환. 없거나 모호하면 None."""
    hits = [i for i, opt in enumerate(answers) if UNKNOWN_RE.search(str(opt))]
    return hits[0] if len(hits) == 1 else None


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 unknown-first heuristic baseline")
    p.add_argument("--data-csv", default="./data/test/test.csv")
    p.add_argument("--output", default="./outputs/phase1_unknown_submission.csv")
    p.add_argument(
        "--fallback", type=int, default=0,
        help="unknown 옵션을 못 찾은 문항에 부여할 기본 label (기본 0)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.data_csv)

    labels, idx_dist, fallback_n = [], Counter(), 0
    for raw in df["answers"]:
        answers = json.loads(raw)
        idx = find_unknown_index(answers)
        if idx is None:
            idx = args.fallback
            fallback_n += 1
        idx_dist[idx] += 1
        labels.append(idx)

    out = df[["sample_id"]].copy()
    out["label"] = labels
    out.to_csv(args.output, index=False)

    print(f"[phase1] wrote {args.output}  (rows={len(out)})")
    print(f"[phase1] unknown 인덱스 분포(=label 분포): {dict(sorted(idx_dist.items()))}")
    print(f"[phase1] unknown 미탐지→fallback({args.fallback}) 적용: {fallback_n}건")


if __name__ == "__main__":
    main()
