# 환경 명세 및 동작 검증 기록

이 문서는 (1) 추론 환경을 **왜 이렇게 구성했는지**, (2) **무엇으로 최종 동작을 판정했는지**를 기록한다.
설치 절차 자체는 [../README.md](../README.md), 정확한 핀은 [../requirements.txt](../requirements.txt) 참고.

## 1. 하드웨어 / 플랫폼

| 항목 | 값 |
| --- | --- |
| OS | Windows 11 + WSL2 (Ubuntu 24.04 LTS) |
| GPU | NVIDIA GeForce RTX 5080 (Blackwell, **compute capability 12.0 / sm_120**), 16GB |
| 드라이버 | 595.97 (WSL CUDA 13.2 제공) |
| Python | 3.12 (conda env `challenge_env`) |

## 2. 왜 베이스라인 그대로 못 쓰는가

대회 베이스라인은 **Linux / RTX 3090 / torch 2.6.0+cu124 / vllm 0.8.3** 기준이다. 이 PC에선 두 가지가 막힌다.

1. **vLLM은 Windows 네이티브 미지원** → WSL2(리눅스) 안에서 실행해야 한다.
2. **RTX 5080은 Blackwell(sm_120)** → torch 2.6/cu124에는 sm_120 커널이 없어
   `CUDA error: no kernel image is available for execution on the device` 발생.
   Blackwell은 **CUDA 12.8+ / torch 2.7+ (cu128)** 가 필요하다.

→ 결론: **WSL2 + conda + cu128 스택으로 업그레이드.**

## 3. 핵심 핀과 그 이유

| 패키지 | 버전 | 이유 |
| --- | --- | --- |
| torch / torchvision / torchaudio | 2.8.0+**cu128** / 0.23.0+cu128 / 2.8.0+cu128 | cu128 빌드여야 `arch_list`에 sm_120 포함 |
| vllm | 0.10.2 | sm_120 지원 + 베이스라인 API(`GuidedDecodingParams` 등) 유지 |
| transformers | **4.56.2** | vllm 0.10.2 요구(≥4.55.2) 충족. **5.x는 `aimv2` config 중복 등록 충돌**로 상한 고정 |
| mistral_common | **1.8.5** | vllm 0.10.2가 `ImageChunk`를 import. **1.9+는 `ImageChunk` 제거**되어 깨짐 |
| xformers | 0.0.32.post1 | torch 2.8/cu128 매칭 빌드 |
| build-essential(gcc) | apt | vLLM의 `torch.compile`(inductor)이 C 컴파일러 필요 |

> CUDA 런타임은 별도 툴킷 설치 없이 **cu128 pip 휠**(`nvidia-*-cu12==12.8.*`)로 제공된다.
> WSL의 `/usr/lib/wsl/lib/libcuda.so`(Windows 드라이버)가 디바이스 드라이버 역할을 한다.

## 4. 설치 중 만난 함정 (해결됨)

- conda 26+ 는 기본 채널 **ToS 수락** 필요 (`conda tos accept ...`).
- `pip install vllm` 가 transformers 5.x / mistral_common 1.11 을 끌어와 충돌 → 위 표대로 상한 고정.
- gcc 없으면 엔진 초기화 후 `Failed to find C compiler` → `build-essential` 설치.

## 5. 최종 동작 판정 (재현 가능)

일회성 주장 대신 **스모크 테스트 스크립트**로 판정한다:

```bash
conda activate challenge_env
cd /mnt/d/skku-multimodal-vqa-2026
bash scripts/smoke_test.sh        # 종료코드 0 = PASS
```

스모크 테스트가 점검하는 것:
1. torch가 RTX 5080을 인식하고 `arch_list`에 `sm_120` 존재
2. vLLM이 LLaVA-OneVision 로드 → 소량 추론 → `outputs/baseline_submission.csv` 생성
3. 제출 CSV 형식(`sample_id,label`, label ∈ {0,1,2}) 일치

### 검증 이력

| 날짜 | 결과 | 비고 |
| --- | --- | --- |
| 2026-06-01 | PASS | `--max-samples 4` 추론 성공. torch 2.8.0+cu128 / vllm 0.10.2, FlashAttention 백엔드. 예측 `0,1,0,0` 생성 확인 |
