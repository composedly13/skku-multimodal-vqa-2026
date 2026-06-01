# 2026 성균관대학교 멀티모달 AI 챌린지

이미지 + 텍스트 기반 멀티모달 시각질의응답(VQA) 대회. 주어진 이미지·문맥·질문과 3개의 보기를 보고 가장 적절한 보기를 선택하며, 근거가 부족한 경우 "정보 부족"을 선택할 수 있어야 한다.

- **대회 링크**: https://dacon.io/competitions/official/236722/overview/description
- **태스크**: 3지선다(0/1/2) 객관식 VQA
- **평가지표**: Accuracy
- **데이터**: 학습 데이터는 형식 이해용 예시 1개만 제공되며, 추가 학습 데이터는 참가자가 직접 구축해야 한다. 평가(test) 데이터 8,500개에 대해 추론한다.

## 디렉터리 구조

```
.
├── data/                       # 대회 제공 데이터
│   ├── sample_submission.csv   # 제출 형식 예시 (sample_id,label)
│   ├── train/
│   │   ├── train.csv           # sample_id,image_path,context,question,answers,label
│   │   └── images/             # 학습 이미지 (예시 1장)
│   └── test/
│       ├── test.csv            # sample_id,image_path,context,question,answers
│       └── images/             # 평가 이미지 8,500장
├── notebooks/
│   └── baseline_llava_onevision.ipynb  # LLaVA-OneVision 기반 베이스라인 추론(노트북)
├── src/                        # 베이스라인 추론 소스 (노트북을 모듈화)
│   ├── inference.py            #   진입점: python -m src.inference
│   ├── model.py                #   출력 스키마 + vLLM 요청 구성
│   └── utils.py                #   파싱/이미지 전처리
├── scripts/
│   └── setup_wsl_env.sh        # WSL2+conda Blackwell 환경 자동 구성
├── requirements.txt            # 검증된 cu128 의존성 핀 (RTX 5080)
├── outputs/                    # 추론 결과 (baseline_submission.csv 생성)
├── models/                     # 모델 가중치/체크포인트
└── docs/                       # 문서
```

## 데이터 형식

| 컬럼 | 설명 |
| --- | --- |
| `sample_id` | 샘플 ID (예: `TEST_0000`) |
| `image_path` | 이미지 상대경로 (예: `./images/test_img_0000.jpg`) |
| `context` | 이미지에 대한 문맥 설명 |
| `question` | 질문 |
| `answers` | 보기 3개의 JSON 배열 (예: `["A", "B", "Not enough information"]`) |
| `label` | 정답 인덱스 `0`/`1`/`2` (train, submission) |

## 환경 구성 (RTX 5080 / Windows)

vLLM은 **Linux 전용**이고, RTX 5080은 **Blackwell(sm_120)** 이라 대회 베이스라인의 핀(torch 2.6/cu124)으로는 동작하지 않는다.
따라서 이 저장소는 **WSL2(Ubuntu 24.04) + conda + Blackwell 호환 스택**으로 구성한다. 아래 절차는 실제 RTX 5080에서 검증됨.

| 구성 | 버전 |
| --- | --- |
| OS | Windows 11 + WSL2 (Ubuntu 24.04) |
| GPU / 드라이버 | RTX 5080 (sm_120) / CUDA 13.2 driver |
| Python | 3.12 (conda `challenge_env`) |
| 핵심 핀 | torch 2.8.0+**cu128**, vllm 0.10.2, transformers **4.56.2**, mistral_common **1.8.5**, xformers 0.0.32.post1 |

> 핀 주의: transformers 5.x(`aimv2` 등록 충돌)와 mistral_common 1.9+(`ImageChunk` 제거)는 vllm 0.10.2와 깨지므로 상한 고정했다.

### 1) WSL2 + Ubuntu 설치 (PowerShell, 최초 1회)

```powershell
wsl --install -d Ubuntu-24.04
```

Windows NVIDIA 드라이버가 WSL CUDA를 제공하므로 WSL 안에 별도 CUDA 툴킷 설치는 불필요하다(런타임은 cu128 pip 휠로 제공). WSL 셸에서 `nvidia-smi`로 GPU가 보이는지 확인한다.

### 2) Miniconda + 환경 구성 (WSL Ubuntu 셸)

```bash
# Miniconda 미설치 시
curl -fsSL -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/miniconda.sh -b -p ~/miniconda3 && ~/miniconda3/bin/conda init bash && exec bash

# 프로젝트 폴더로 이동 (Windows D: 드라이브는 /mnt/d 로 마운트됨)
cd /mnt/d/skku-multimodal-vqa-2026

# 환경 생성 + 의존성 설치 (gcc 설치, conda env, requirements.txt 까지 자동)
bash scripts/setup_wsl_env.sh
```

`scripts/setup_wsl_env.sh`가 `build-essential`(torch.compile용 gcc), `challenge_env`(Python 3.12), `requirements.txt`(cu128 스택)를 설치하고 GPU 인식까지 검증한다.

## 베이스라인 실행

WSL Ubuntu 셸에서 **저장소 루트(`/mnt/d/skku-multimodal-vqa-2026`)** 기준으로 실행한다.

```bash
conda activate challenge_env
cd /mnt/d/skku-multimodal-vqa-2026

# 소스 스크립트로 실행
python -m src.inference                       # 기본: data/test 전체(8,500개) 추론
python -m src.inference --max-samples 4 --batch-size 4   # 동작 확인용 소량 실행
```

기본 인자: `--data-csv ./data/test/test.csv`, `--images-dir ./data/test`, `--output-path ./outputs/`.
`image_path`가 `./images/...` 형식이므로 `--images-dir`에는 split 루트(`data/test`)를 지정한다.

추론이 끝나면 `outputs/baseline_submission.csv` (`sample_id,label`)가 생성되며 이를 제출한다.
파싱 실패 등으로 예측이 결측된 샘플은 일괄 `0`으로 처리된다.

> 노트북([notebooks/baseline_llava_onevision.ipynb](notebooks/baseline_llava_onevision.ipynb))도 동일 환경에서 커널로 등록해 실행할 수 있다:
> `pip install ipykernel && python -m ipykernel install --user --name challenge_env`

## 평가/제출 일정

- 대회 종료: 2026-06-29
- 파이널리스트 코드 제출/검증: 2026-07-02
- 2차 평가에서 코드 검증을 포함하며, 성별·인종·민족에 대한 공정성을 강조한다.
