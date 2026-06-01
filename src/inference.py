"""LLaVA-OneVision 베이스라인 추론 진입점 (베이스라인 노트북 1,6,7,10번 셀).

저장소 루트에서 실행:
    python -m src.inference --data-csv ./data/test/test.csv --images-dir ./data/test
또는 기본 인자로:
    python -m src.inference
"""

import json
import os
import time
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

# GPU 사용 모드: "all"(전체 GPU) 또는 "gpu0"(0번 GPU만).
# CUDA 초기화 이전에 환경변수를 설정해야 하므로 torch/vllm import보다 먼저 처리.
GPU_MODE = os.environ.get("GPU_MODE", "gpu0")
if GPU_MODE == "gpu0":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif GPU_MODE == "all":
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    raise ValueError("GPU_MODE must be 'all' or 'gpu0'")

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.sampling_params import GuidedDecodingParams  # noqa: E402
from vllm.utils import FlexibleArgumentParser  # noqa: E402

from src.model import ReasonAnswer, run_llava_onevision  # noqa: E402
from src.utils import load_image, normalize_answer_id, parse_answers_field  # noqa: E402


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "vision language models for text generation"
    )
    parser.add_argument("--model-type", "-m", type=str, default="llava-onevision")
    parser.add_argument(
        "--seed", type=int, default=42, help="Set the seed when initializing `vllm.LLM`."
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        default="./data/test/test.csv",
        help="Input csv with columns: sample_id,image_path,context,question,answers",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="./data/test",
        help="Directory storing images referenced by image_path",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--output-path",
        type=str,
        default="./outputs/",
        help="File path to store inference outputs CSV.",
    )
    return parser.parse_args()


def main(args):
    modality = "image"

    json_schema = ReasonAnswer.model_json_schema()
    guided_decoding_params_json = GuidedDecodingParams(json=json_schema)

    output_path = os.path.join(args.output_path)
    os.makedirs(output_path, exist_ok=True)

    df = pd.read_csv(args.data_csv)

    if "label" in df.columns:
        print("INFO: input csv has 'label' column, but inference never reads it for prediction.")

    df["model_output"] = None
    df["label"] = None

    if args.max_samples is not None:
        df = df.head(args.max_samples).copy()

    inputs = []
    batch_indices = []
    llm = None

    with tqdm(total=len(df), desc="Inference", unit="sample") as pbar:
        for row_idx, row in df.iterrows():
            image_path = Path(args.images_dir) / str(row["image_path"])

            context = "Context: " + str(row.get("context", ""))
            question = "Question: " + str(row.get("question", ""))

            answers = parse_answers_field(row["answers"])
            options = (
                "Options:\n"
                f"0. {answers[0]}\n"
                f"1. {answers[1]}\n"
                f"2. {answers[2]}\n"
            )

            pre_prompt = (
                "You are an expert Vision Language assistant. "
                "When given an image, a context, a question, and options, "
                "you MUST respond only with a JSON object"
            )

            post_prompt = (
                "Give the output in strict JSON format: "
                "{\n"
                '   "reason": "One short sentence of reasoning.",\n'
                '   "answer_id": "<one of: 0, 1, 2>"\n'
                "}\n"
            )

            rule_prompt = "Do NOT output multiple options."

            prompt_text = (
                pre_prompt + "\n"
                + context + "\n"
                + question + "\n"
                + options + "\n"
                + post_prompt + "\n"
                + rule_prompt
            )

            data = load_image(image_path, img_size=args.img_size)

            if data is None:
                df.at[row_idx, "label"] = "0"
                df.at[row_idx, "model_output"] = ""
                _save_submission(df, output_path)
                pbar.update(1)
                continue

            req_data = run_llava_onevision([prompt_text])

            if llm is None:
                default_limits = {"image": 1, "video": 0, "audio": 0}
                req_data.engine_args.limit_mm_per_prompt = default_limits
                engine_args = asdict(req_data.engine_args) | {"seed": args.seed}
                llm = LLM(**engine_args)

            prompts = req_data.prompts[0]

            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=128,
                stop_token_ids=req_data.stop_token_ids,
                guided_decoding=guided_decoding_params_json,
            )

            inputs.append({
                "prompt": prompts,
                "multi_modal_data": {modality: data},
            })
            batch_indices.append(row_idx)

            is_batch_ready = ((row_idx + 1) % args.batch_size) == 0
            is_last_row = row_idx == df.index[-1]

            if is_batch_ready or is_last_row:
                outputs = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)

                for idx, o in zip(batch_indices, outputs):
                    generated_text = o.outputs[0].text
                    df.at[idx, "model_output"] = generated_text

                    try:
                        json_match_start = generated_text.find("{")
                        json_match_end = generated_text.rfind("}")

                        if json_match_start >= 0 and json_match_end > json_match_start:
                            parsed = json.loads(generated_text[json_match_start:json_match_end + 1])
                        else:
                            parsed = json.loads(generated_text)

                        df.at[idx, "label"] = normalize_answer_id(parsed.get("answer_id"))
                    except Exception:  # noqa: BLE001 - 파싱 실패 시 0으로 처리
                        df.at[idx, "label"] = "0"

                _save_submission(df, output_path)
                pbar.update(len(batch_indices))

                inputs = []
                batch_indices = []


def _save_submission(df, output_path):
    """sample_id,label 형식으로 제출 파일 저장."""
    save_cols = ["sample_id", "label"]
    existing_cols = [c for c in save_cols if c in df.columns]
    df[existing_cols].to_csv(os.path.join(output_path, "baseline_submission.csv"), index=False)


def check_pytorch_gpu():
    try:
        if torch.cuda.is_available():
            print(f"PyTorch can access {torch.cuda.device_count()} GPU(s).")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("PyTorch cannot access any GPUs.")
    except Exception as e:  # noqa: BLE001
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    check_pytorch_gpu()
    args = parse_args()
    _t0 = time.time()
    main(args)
    print(f"total_elapsed_seconds={time.time() - _t0:.2f}")
