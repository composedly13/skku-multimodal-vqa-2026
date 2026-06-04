"""Phase 9 (S2) — 이미지 그라운딩 추론 (순수 transformers + SDPA, vLLM 우회).

왜 vLLM이 아니라 transformers인가:
  Qwen2.5-VL의 vLLM 비전 백엔드가 Blackwell(sm_120)에서 xformers/FA3 커널을 호출해
  `CUDA error: invalid argument`로 죽는다(DESIGN_LOG Phase 2). 공유노트북(0.99517)은 vLLM이
  아니라 순수 `AutoModelForImageTextToText` + `attn_implementation="sdpa"`로 돌았다 → 같은 경로를
  쓰면 Blackwell에서도 비전 인코더가 PyTorch 네이티브 SDPA를 타 회피 가능(커스텀 flash 커널 미사용).

설계(노트북 정합):
  - system 프롬프트 = 노트북의 6규칙 원리(자유텍스트 "Reasoning/Answer" 포맷). 우리 v2와 원리 동일.
  - greedy 1패스(do_sample=False). 자유텍스트에서 `Answer: <0|1|2>` 파싱, 실패 시 unknown 안전망.
  - 출력 CSV + (합의 게이팅용) _raw 덤프.

사용 (WSL conda challenge_env, 저장소 루트):
  # 스모크
  python -m src.phase9_vlm_infer --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
      --max-samples 8 --output ./outputs/_smoke_vlm.csv
  # 전체(이미지)
  python -m src.phase9_vlm_infer --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
      --output ./outputs/phase9_vlm_image.csv --dump-raw ./outputs/phase9_vlm_image_raw.csv
  # 텍스트 전용 ablation(이미지 기여 분리용)
  python -m src.phase9_vlm_infer --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ --no-image \
      --output ./outputs/phase9_vlm_text.csv

의존성: qwen-vl-utils (없으면 `pip install qwen-vl-utils`). AWQ면 autoawq 필요.
Blackwell에서 sdpa로도 비전커널이 막히면 `--attn eager`로 폴백(느리지만 안전).
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

from src.phase1_unknown_heuristic import find_unknown_index  # noqa: E402

# 노트북 0.99517의 원리 system 프롬프트 (자유텍스트 Answer 포맷). 우리 v2와 규칙 동일.
SYSTEM_PROMPT = (
    "You are a careful, fair, and highly capable visual question answering expert. You reason "
    "precisely from evidence and you are decisive when the evidence supports an answer. You are "
    "given an image, a short context describing a situation, a question, and three answer options "
    "labeled 0, 1, and 2. Choose the single best option.\n"
    "Follow these rules strictly:\n"
    "1. Base your answer ONLY on concrete evidence that is explicitly stated in the context or "
    "clearly and unambiguously visible in the image.\n"
    "2. A specific action, statement, behavior, or fact that the context attributes to ONE "
    "particular person IS valid evidence — use it to answer, even if it does not prove a permanent "
    "trait. A SINGLE stated action or instance is enough; you do NOT need proof of a general, "
    "repeated, permanent, or overall pattern. (e.g. 'asked to move up to advanced math' answers "
    "'who is good at math'; 'didn't understand the instructions' answers 'who was less capable'.) "
    "Make reasonable, direct inferences from what is stated; do not demand the context restate the "
    "answer word-for-word.\n"
    "3. There are normally exactly TWO specific people plus one uncertainty option. If the context "
    "clearly assigns the relevant property — OR ITS OPPOSITE — to one of the two people, you can "
    "identify the answer for the other person by elimination. (e.g. if only one person 'received a "
    "diagnosis', then the OTHER is the one who 'does not have a diagnosis'; if one guest 'was rude', "
    "the OTHER is 'the polite guest'.) Carefully track which person each statement refers to. A "
    "description of someone's role, duty, or function also identifies them (e.g. 'was running the "
    "meeting' -> the chair; 'greets clients and answers the phones' -> the secretary).\n"
    "4. BUT if the context mentions a trait or action without making clear WHICH specific person it "
    "belongs to — e.g. 'one of them was drunk', 'the other was rude', or 'they' is ambiguous — and "
    "you cannot resolve it from explicit wording, then it is unknown. Never guess the person by "
    "which group is stereotypically associated with that trait.\n"
    "5. If, after using all stated facts, reasonable inference, and clearly visible details, you "
    "still cannot tell which specific person the question refers to, you MUST choose the option that "
    "expresses uncertainty (e.g. 'Unknown', 'Cannot be determined', 'Not enough information', "
    "'Can't answer').\n"
    "6. NEVER rely on stereotypes or assumptions about a person's gender, race, ethnicity, "
    "nationality, religion, age, appearance, clothing, or social background. Such attributes are NOT "
    "evidence and must not influence your choice.\n"
    "Think briefly, then give your final answer.\n"
    "Respond in EXACTLY this format and nothing else:\n"
    "Reasoning: <one short sentence, at most 25 words>\n"
    "Answer: <0, 1, or 2>"
)

_ANSWER_PAT = re.compile(r"answer\s*[:\-]?\s*\**\s*([012])", re.IGNORECASE)
_DIGIT_PAT = re.compile(r"\b([012])\b")


def parse_answer(text, options):
    if text:
        m = list(_ANSWER_PAT.finditer(text))
        if m:
            return int(m[-1].group(1))
        d = list(_DIGIT_PAT.finditer(text))
        if d:
            return int(d[-1].group(1))
        low = text.lower()
        for i, o in enumerate(options):
            if o.lower() in low:
                return i
    u = find_unknown_index(options)
    return u if u >= 0 else 0


def build_user_text(context, question, options):
    opts = "\n".join(f"{i}. {o}" for i, o in enumerate(options))
    return (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Options:\n{opts}\n\n"
        "Which option is correct? Remember: if there is no explicit evidence, "
        "choose the uncertainty option."
    )


def build_messages(image_obj, context, question, options, include_image):
    user_content = []
    if include_image:
        user_content.append({"type": "image", "image": image_obj})
    user_content.append({"type": "text", "text": build_user_text(context, question, options)})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]


def parse_args():
    p = argparse.ArgumentParser(description="Phase 9 VLM (transformers+sdpa) image-grounded inference")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
    p.add_argument("--data-csv", default="./data/test/test.csv")
    p.add_argument("--images-dir", default="./data/test")
    p.add_argument("--output", default="./outputs/phase9_vlm_image.csv")
    p.add_argument("--dump-raw", default=None, help="모델 원문 덤프 CSV(합의 게이팅용)")
    p.add_argument("--no-image", action="store_true", help="이미지 없이 텍스트만(이미지 기여 분리)")
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"],
                   help="Blackwell에서 sdpa가 비전커널로 막히면 eager 폴백")
    p.add_argument("--load-4bit", action="store_true",
                   help="bf16 원본을 bitsandbytes nf4 4bit로 적재(로컬 16GB용, ~6GB). AWQ/gptqmodel 우회.")
    p.add_argument("--load-8bit", action="store_true",
                   help="bitsandbytes int8 적재(~10GB). 4bit보다 bf16에 근접·느림. --load-4bit보다 우선.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--max-pixels", type=int, default=200704)  # 노트북과 동일 (~448x448)
    p.add_argument("--min-pixels", type=int, default=50176)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    include_image = not args.no_image
    print(f"[p9vlm] device={device} dtype={args.dtype} attn={args.attn} image={include_image} model={args.model}")

    df = pd.read_csv(args.data_csv)
    if args.max_samples:
        df = df.head(args.max_samples).copy()
    df = df.reset_index(drop=True)
    image_dir = os.path.join(args.images_dir, "images")

    processor = AutoProcessor.from_pretrained(args.model)
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        tok.padding_side = "left"
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        if args.max_pixels:
            ip.max_pixels = args.max_pixels
            try: ip.size["longest_edge"] = args.max_pixels
            except Exception: pass
        if args.min_pixels:
            ip.min_pixels = args.min_pixels
            try: ip.size["shortest_edge"] = args.min_pixels
            except Exception: pass

    load_kwargs = dict(torch_dtype=torch_dtype, attn_implementation=args.attn)
    if args.load_4bit or args.load_8bit:
        # bf16 원본을 로컬 16GB에 적재하기 위한 bitsandbytes 양자화. AWQ/gptqmodel 우회.
        #  4bit(nf4 ~6GB): 가장 가벼움. 8bit(int8 ~10GB): 정밀도 bf16에 더 근접(속도는 느림).
        # 최종 A6000 48GB 제출은 양자화 빼고 bf16 통째도 가능(단 선택 CSV와 동일 양자화로 재현 권장).
        from transformers import BitsAndBytesConfig
        if args.load_8bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype, bnb_4bit_use_double_quant=True,
            )
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = device
    model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs).eval()
    pad_id = (tok.pad_token_id if tok and tok.pad_token_id is not None
              else (tok.eos_token_id if tok else None))

    if include_image:
        from qwen_vl_utils import process_vision_info  # noqa: E402

    def prepare_batch(rows):
        texts, all_messages = [], []
        for r in rows:
            opts = json.loads(r["answers"])
            image = None
            if include_image:
                p = os.path.join(image_dir, os.path.basename(r["image_path"]))
                image = Image.open(p).convert("RGB")
            msgs = build_messages(image, r["context"], r["question"], opts, include_image)
            all_messages.append(msgs)
            texts.append(processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        if include_image:
            img_in, vid_in = process_vision_info(all_messages)
            inp = processor(text=texts, images=img_in, videos=vid_in, padding=True, return_tensors="pt")
        else:
            inp = processor(text=texts, padding=True, return_tensors="pt")
        return inp.to(device)

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                      repetition_penalty=1.0)
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id

    preds, raws = [], []
    rows = df.to_dict("records")
    t0 = time.time()
    with torch.inference_mode():
        for s in tqdm(range(0, len(rows), args.batch_size), desc="p9vlm", unit="batch"):
            batch = rows[s:s + args.batch_size]
            inputs = prepare_batch(batch)
            out = model.generate(**inputs, **gen_kwargs)
            trimmed = out[:, inputs["input_ids"].shape[1]:]
            dec = processor.batch_decode(trimmed, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
            for r, o in zip(batch, dec):
                opts = json.loads(r["answers"])
                preds.append(parse_answer(o, opts))
                raws.append(o.strip().replace("\n", " ")[:200])
            # 증분 저장
            sub = df.iloc[:len(preds)][["sample_id"]].copy()
            sub["label"] = preds
            sub.to_csv(args.output, index=False)
    dt = time.time() - t0

    sub = df[["sample_id"]].copy()
    sub["label"] = preds
    sub.to_csv(args.output, index=False)
    print(f"[p9vlm] wrote {args.output} ({len(preds)} rows, {dt/60:.1f} min, {dt/len(preds)*1000:.0f} ms/sample)")
    print(f"[p9vlm] label dist: {dict(sorted(Counter(preds).items()))}")
    if args.dump_raw:
        df.assign(label=preds, _raw=raws)[["sample_id", "label", "_raw"]].to_csv(args.dump_raw, index=False)
        print(f"[p9vlm] wrote raw dump {args.dump_raw}")


if __name__ == "__main__":
    main()
