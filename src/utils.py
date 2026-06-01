"""데이터 파싱 및 이미지 전처리 유틸리티 (베이스라인 노트북 5번 셀)."""

import base64
import json
from io import BytesIO
from pathlib import Path

from PIL import Image


def parse_answers_field(raw: str):
    """answers 컬럼(JSON 문자열)을 리스트로 변환."""
    return json.loads(raw)


def normalize_answer_id(value):
    """0, 1, 2 중 하나의 라벨만 허용합니다."""
    if value is None:
        return "0"
    text = str(value).strip()
    return text if text in {"0", "1", "2"} else "0"


def load_image(image, img_size=512, base_64=False):
    """이미지를 RGB로 열고 가로 기준 img_size로 리사이즈한다.

    실패 시 None을 반환하며, 이 경우 호출부에서 예측을 0으로 처리한다.
    """
    try:
        if isinstance(image, (str, Path)):
            img = Image.open(str(image))
        else:
            img = Image.open(BytesIO(image["bytes"]))
        img = img.convert("RGB")
        width_percent = img_size / float(img.size[0])
        new_height = int((float(img.size[1]) * width_percent))

        img_resized = img.resize((img_size, new_height), Image.LANCZOS)

        if base_64:
            buffered = BytesIO()
            img_resized.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_resized
    except Exception as e:  # noqa: BLE001 - 손상된 이미지는 건너뛰고 0으로 처리
        print(e)
        return None
