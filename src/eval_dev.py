"""Phase 7 dev셋 평가기 (2026-06-03).

자체 합성 dev셋(data/dev/dev.jsonl)에서 여러 프롬프트 변형을 *한 번의 모델 로드로* 비교한다.
public LB를 만지기 전에 R2/R3/R4 원리가 일반적으로 작동하는지(특히 소거법·역할식별·과교정 차단)
확인하는 용도. 통과한 변형만 test에 제출한다.

실행(WSL conda challenge_env):
  python -m src.eval_dev
  python -m src.eval_dev --max-model-len 4096 --variants base v2 v2+bal v2+fsv2

출력: 변형별 전체 정확도 + 유형별 정확도 표 + 오답 ID 목록.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from src.phase1_unknown_heuristic import find_unknown_index
from src.phase2_infer import (
    ANSWER_SCHEMA,
    FEWSHOT_TURNS_V2,
    SYSTEM_BALANCE,
    SYSTEM_BOOST,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_V2,
    parse_answer_id,
)


def build_user_text(ctx, q, opts):
    return (
        f"Context: {ctx}\n"
        f"Question: {q}\n"
        "Options:\n"
        f"0. {opts[0]}\n"
        f"1. {opts[1]}\n"
        f"2. {opts[2]}\n"
        "Give your answer as JSON {\"reason\": \"...\", \"answer_id\": \"0|1|2\"}."
    )


# 변형 정의: name -> (system_prompt, fewshot_turns)
def make_variants():
    return {
        "base":        (SYSTEM_PROMPT, []),
        "base+boost":  (SYSTEM_PROMPT + SYSTEM_BOOST, []),
        "base+bal":    (SYSTEM_PROMPT + SYSTEM_BALANCE, []),
        "base+fsv2+bal": (SYSTEM_PROMPT + SYSTEM_BALANCE, FEWSHOT_TURNS_V2),  # 현 베스트 구성
        "v2":          (SYSTEM_PROMPT_V2, []),
        "v2+bal":      (SYSTEM_PROMPT_V2 + SYSTEM_BALANCE, []),
        "v2+fsv2":     (SYSTEM_PROMPT_V2, FEWSHOT_TURNS_V2),
        "v2+fsv2+bal": (SYSTEM_PROMPT_V2 + SYSTEM_BALANCE, FEWSHOT_TURNS_V2),
    }


def load_dev(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_variant(llm, sp, items, system_prompt, fewshot_turns):
    convs = []
    for it in items:
        user_text = build_user_text(it["context"], it["question"], it["options"])
        msgs = [{"role": "system", "content": system_prompt}]
        for ex_u, ex_a in fewshot_turns:
            msgs.append({"role": "user", "content": ex_u})
            msgs.append({"role": "assistant", "content": ex_a})
        msgs.append({"role": "user", "content": user_text})
        convs.append(msgs)
    outputs = llm.chat(convs, sp, use_tqdm=False)
    preds = []
    for it, o in zip(items, outputs):
        aid = parse_answer_id(o.outputs[0].text)
        if aid is None:
            ui = find_unknown_index(it["options"])
            aid = ui if ui >= 0 else 0
        preds.append(int(aid))
    return preds


def score(items, preds):
    per_type_tot = defaultdict(int)
    per_type_ok = defaultdict(int)
    wrong = []
    for it, p in zip(items, preds):
        t = it["type"]
        per_type_tot[t] += 1
        if p == it["gold"]:
            per_type_ok[t] += 1
        else:
            wrong.append((it["id"], t, it["gold"], p))
    total = len(items)
    ok = sum(per_type_ok.values())
    return ok, total, per_type_ok, per_type_tot, wrong


def parse_args():
    p = argparse.ArgumentParser(description="Phase 7 dev-set evaluator")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    p.add_argument("--dev", default="data/dev/dev.jsonl")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-mem", type=float, default=0.9)
    p.add_argument("--variants", nargs="*", default=None,
                   help="평가할 변형 이름들(미지정 시 전부). 예: base v2 v2+bal v2+fsv2")
    return p.parse_args()


def main():
    args = parse_args()
    items = load_dev(args.dev)
    print(f"[eval_dev] {len(items)} dev items | "
          f"type 분포: {dict(sorted(Counter(i['type'] for i in items).items()))}")

    all_variants = make_variants()
    names = args.variants or list(all_variants.keys())
    for n in names:
        if n not in all_variants:
            raise SystemExit(f"unknown variant: {n} (available: {list(all_variants)})")

    print(f"[eval_dev] loading {args.model}")
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, seed=42)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=256,
                        guided_decoding=GuidedDecodingParams(json=ANSWER_SCHEMA))

    types = sorted({i["type"] for i in items})
    results = {}
    for name in names:
        sysp, fs = all_variants[name]
        preds = run_variant(llm, sp, items, sysp, fs)
        results[name] = score(items, preds)

    # 표 출력
    col_w = max(len(t) for t in types) + 1
    header = f"{'variant':<16}{'overall':>9}  " + "  ".join(f"{t[:col_w]:>{col_w}}" for t in types)
    print("\n" + header)
    print("-" * len(header))
    for name in names:
        ok, total, ptok, pttot, _ = results[name]
        cells = []
        for t in types:
            cells.append(f"{ptok[t]}/{pttot[t]:<{col_w-2}}")
        acc = f"{ok}/{total} {ok/total*100:.0f}%"
        print(f"{name:<16}{acc:>9}  " + "  ".join(f"{c:>{col_w}}" for c in cells))

    # 오답 상세
    print("\n[오답 상세]")
    for name in names:
        _, _, _, _, wrong = results[name]
        if wrong:
            ws = ", ".join(f"{wid}({t}:gold{g}→{p})" for wid, t, g, p in wrong)
            print(f"  {name}: {ws}")
        else:
            print(f"  {name}: (만점)")


if __name__ == "__main__":
    main()
