"""출력 스키마 및 LLaVA-OneVision 모델 요청 구성 (베이스라인 노트북 3~4번 셀)."""

from typing import Literal, NamedTuple

import torch
from pydantic import BaseModel
from vllm import EngineArgs


class ReasonAnswer(BaseModel):
    """모델이 강제로 따라야 하는 JSON 출력 스키마."""

    reason: str
    answer_id: Literal["0", "1", "2"]


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    stop_token_ids: list[int] | None = None


def run_llava_onevision(questions: list[str]) -> ModelRequestData:
    """질문 리스트로부터 LLaVA-OneVision 프롬프트와 EngineArgs를 구성한다."""

    prompts = [
        f"<|im_start|>user <image>\n{question}<|im_end|> \
        <|im_start|>assistant\n"
        for question in questions
    ]

    engine_args = EngineArgs(
        # 사용할 모델 경로 또는 Hugging Face 모델 ID
        model="llava-hf/llava-onevision-qwen2-0.5b-si-hf",
        # 모델이 처리할 수 있는 최대 입력/출력 토큰 길이. 값이 클수록 긴 문맥 처리가
        # 가능하지만, KV cache로 인해 VRAM 사용량이 증가
        max_model_len=16384,
        # 하나의 프롬프트에서 허용할 멀티모달 입력 개수 제한. image=1은 이미지 1장만 허용
        limit_mm_per_prompt={"image": 1},
        # 모델을 분산 실행할 GPU 개수. 사용 가능한 CUDA GPU 수만큼 tensor parallel 적용
        tensor_parallel_size=torch.cuda.device_count(),
        # vLLM이 사용할 GPU 메모리 비율. 값이 높을수록 KV cache를 크게 잡지만 OOM 위험 증가
        gpu_memory_utilization=0.9,
        # 멀티모달 전처리 결과 캐시 비활성화. 메모리를 보수적으로 관리할 때 유용
        disable_mm_preprocessor_cache=True,
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )
