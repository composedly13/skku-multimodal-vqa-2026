"""Phase 9 (S4) — 다관점 합의 게이팅 병합 (오프라인, GPU 불필요).

원리: 두 독립 관점(예: 텍스트-Qwen3-8B 0.991 + 이미지-Qwen2.5-VL)이 인물에 합의할 때만 인물을 지목.
      합의가 깨지면 unknown으로 보수화. BBQ 오답의 주범은 '과확신 인물지목'이므로, 인물지목을
      관점 간 견고성으로 게이팅하면 정밀도↑ + 7/2 공정성(증거 게이팅) 정합. 단순/단조 변환.

모드:
  intersect-person (기본): anchor(보통 강한 텍스트 모델)를 베이스로,
      anchor가 '인물'을 지목했는데 other가 '동의하지 않으면'(다른 인물 또는 unknown) → unknown으로 강등.
      → 인물지목 = 양관점 합의 필요. anchor의 lone 오답 인물지목을 제거(일부 정답도 손실 가능).
  union-person: 둘 중 하나라도 인물을 지목하고 '서로 충돌 안 하면'(같은 인물 or 한쪽 unknown) 그 인물.
      둘 다 unknown이거나 서로 다른 인물이면 unknown. → 인물 회수 공격적.
  anchor: 변경 없이 anchor 그대로 출력(합의 진단 리포트만).

사용 (저장소 루트, 어디서나 — pandas만 필요):
  python -m src.phase9_agreement \
      --anchor ./outputs/phase8_q3_8b_v2bal_sc.csv \
      --other  ./outputs/phase9_vlm_image.csv \
      --mode intersect-person \
      --output ./outputs/phase9_agree_intersect.csv
"""

import argparse
import json
from collections import Counter

import pandas as pd

from src.phase1_unknown_heuristic import find_unknown_index


def parse_args():
    p = argparse.ArgumentParser(description="Phase 9 multi-view agreement gating merge")
    p.add_argument("--anchor", required=True, help="베이스 제출 CSV(보통 강한 텍스트 모델)")
    p.add_argument("--other", required=True, help="보조 관점 제출 CSV(보통 이미지 모델)")
    p.add_argument("--data-csv", default="./data/test/test.csv")
    p.add_argument("--mode", default="intersect-person",
                   choices=["intersect-person", "union-person", "anchor"])
    p.add_argument("--output", default="./outputs/phase9_agree.csv")
    return p.parse_args()


def main():
    args = parse_args()
    test = pd.read_csv(args.data_csv)[["sample_id", "answers"]]
    a = pd.read_csv(args.anchor).rename(columns={"label": "a"})
    b = pd.read_csv(args.other).rename(columns={"label": "b"})
    df = test.merge(a, on="sample_id").merge(b, on="sample_id")
    if len(df) != len(test):
        raise SystemExit(f"merge 행수 불일치: test={len(test)} merged={len(df)} (sample_id 정렬 확인)")
    df["unk"] = [find_unknown_index(json.loads(x)) for x in df["answers"]]

    def decide(row):
        a_, b_, unk = int(row["a"]), int(row["b"]), int(row["unk"])
        if args.mode == "anchor":
            return a_
        if args.mode == "intersect-person":
            # anchor가 인물 지목 + other가 동의 안 함 → unknown 강등
            if a_ != unk and b_ != a_:
                return unk
            return a_
        # union-person: 한쪽이라도 인물 + 충돌 없음 → 그 인물
        if a_ == b_:
            return a_
        a_person, b_person = a_ != unk, b_ != unk
        if a_person and b_person:   # 서로 다른 인물 → 충돌 → unknown
            return unk
        if a_person:                # other만 unknown → anchor 인물 채택
            return a_
        if b_person:                # anchor만 unknown → other 인물 채택
            return b_
        return unk                  # 둘 다 unknown

    df["label"] = df.apply(decide, axis=1)

    # 진단 리포트
    agree = int((df["a"] == df["b"]).sum())
    changed = int((df["label"] != df["a"]).sum())
    both_person_conflict = int(((df["a"] != df["unk"]) & (df["b"] != df["unk"]) & (df["a"] != df["b"])).sum())
    print(f"[agree] rows={len(df)} mode={args.mode}")
    print(f"[agree] 두 관점 동일: {agree} ({agree/len(df)*100:.1f}%)")
    print(f"[agree] anchor 인물 vs other 다른인물(충돌): {both_person_conflict}")
    print(f"[agree] anchor 대비 변경: {changed} ({changed/len(df)*100:.1f}%)")
    print(f"[agree] anchor 분포: {dict(sorted(Counter(df['a']).items()))}")
    print(f"[agree] 결과 분포: {dict(sorted(Counter(df['label']).items()))}")

    out = df[["sample_id", "label"]].copy()
    out["label"] = out["label"].astype(int)
    out.to_csv(args.output, index=False)
    print(f"[agree] wrote {args.output}")


if __name__ == "__main__":
    main()
