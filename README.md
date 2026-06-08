# 2026 성균관대학교 멀티모달 AI 챌린지

이미지 + 텍스트 기반 멀티모달 시각질의응답(VQA) 대회. 주어진 이미지·문맥·질문과 3개의 보기를 보고 가장 적절한 보기를 선택하며, 근거가 부족한 경우 "정보 부족"을 선택할 수 있어야 한다.

- **대회 링크**: https://dacon.io/competitions/official/236722/overview/description
- **태스크**: 3지선다(0/1/2) 객관식 VQA
- **평가지표**: Accuracy
- **데이터**: 학습 데이터는 형식 이해용 예시 1개만 제공되며, 추가 학습 데이터는 참가자가 직접 구축해야 한다. 평가(test) 데이터 8,500개에 대해 추론한다.

## 현재 상태 (요약, 2026-06-08 · Phase 12)

> 전체 의사결정·실험은 [docs/DESIGN_LOG.md](docs/DESIGN_LOG.md), 제출 후보는 [docs/SUBMISSIONS.md](docs/SUBMISSIONS.md).

- **⚠️ 멀티모달 필수.** 데이콘 Q&A(2026-06-05): "텍스트만으로 추론하는 접근은 본 대회서 **유효한 모델이 아니다**." Public은 text-only로도 ~0.99 나오지만(암기 벤치), **Private은 이미지가 정답에 필요**하도록 설계되고 2차 심사가 멀티모달 정보활용을 본다. → **text-only 폐기, 이미지 사용 필수.**
- **현재 멀티모달 앵커 = Qwen3.5-9B 4bit + 이미지, prompt v2 (public 0.99433).** 단일 greedy(rule5 합법). 산출물 `outputs/phase9_q35_9b_image.csv`.
  - 비교/기각: 9B text-only 0.996(실격), Qwen3-VL-8B 0.97383(8B라 약함), Qwen3-14B text 0.97475(실격).
- **핵심 제약(대회 규칙):** ①최종답은 LLM 단일생성(단순 다수결·룰매핑 금지) ②2026-05-31 이전 공개 가중치만 ③평가셋 파생 학습/검증 금지(leakage) ④A6000 48GB·torch2.6·오프라인, ~520ms/샘플 ⑤최종은 제출창서 직접 고른 1파일.
- **자체 검증 하니스(Public 포화·블라인드 대응):** 합성 일반원리로 편향·강건성을 측정 — 텍스트 BBQ 하니스(`src/build_robustness_set_v2.py`+`eval_robustness_v2.py`, 1152문항) + **멀티모달 하니스(`src/build_robustness_mm.py`+`eval_robustness_mm.py`, 증거를 이미지에만 두어 이미지 활용도·외모편향 측정).**
- **추론기:** [src/phase9_vlm_infer.py](src/phase9_vlm_infer.py) — transformers+SDPA(Blackwell vLLM 우회), `--load-4bit`(bitsandbytes nf4), `--system-prompt {v2,v3,v4,v5}`(v5=시각 외모편향 정조준), `--causal-lm`(텍스트 전용 LLM), `--enable-thinking`(켜면 ~13s/샘플=rule6 초과로 미채택).
- **실행 환경:** Qwen3.5/bitsandbytes 경로는 격리 conda **`challenge_q35`**(torch 2.11+cu128). 베이스라인 vLLM 경로는 `challenge_env`. 둘 다 WSL2 Ubuntu-24.04. 로컬 GPU=RTX5080 **16GB**(≤14B 4bit·9B/VL-8B만 적재 가능).

## 디렉터리 구조

```
.
├── data/                       # 대회 제공 데이터
│   ├── sample_submission.csv   # 제출 형식 예시 (sample_id,label)
│   ├── train/
│   │   ├── train.csv           # sample_id,image_path,context,question,answers,label
│   │   └── images/             # 학습 이미지 (예시 1장)
│   ├── test/
│   │   ├── test.csv            # sample_id,image_path,context,question,answers
│   │   └── images/             # 평가 이미지 8,500장
│   └── dev/
│       └── dev.jsonl           # 자체 합성 dev셋 (Phase 7, 원리 검증용)
├── notebooks/
│   └── baseline_llava_onevision.ipynb  # LLaVA-OneVision 기반 베이스라인 추론(노트북)
├── src/                        # 추론 소스
│   ├── phase9_vlm_infer.py     #   ★현 메인 추론기: transformers+SDPA, 4bit, 이미지/텍스트,
│   │                           #     --system-prompt{v2..v5}/--causal-lm/--enable-thinking
│   ├── build_robustness_set_v2.py / eval_robustness_v2.py  # 텍스트 BBQ 강건성 하니스(1152)
│   ├── build_robustness_mm.py  / eval_robustness_mm.py     # ★멀티모달 하니스(증거를 이미지에)
│   ├── build_robustness_set.py / eval_robustness.py        # 하니스 v1(회귀 기준선)
│   ├── phase9_agreement.py     #   합의 게이팅(LLM-judge 재구현 필요, 현재 미사용)
│   ├── phase1_unknown_heuristic.py  # unknown 옵션 인덱스 탐지(파싱 안전망에 재사용)
│   ├── inference.py / model.py / utils.py  # 초기 vLLM 베이스라인(challenge_env)
│   ├── phase2_infer.py         #   초기 BBQ 프롬프트/few-shot/SC 추론기(텍스트)
│   └── build_dev_set.py / eval_dev.py  # 초기 합성 dev셋(원리 검증용)
├── scripts/
│   ├── setup_wsl_env.sh        # WSL2+conda Blackwell 환경 자동 구성
│   ├── smoke_test.sh           # 환경 동작 재판정 스모크
│   ├── eval_dev.sh             # dev셋 평가 러너
│   ├── phase7_submit.sh        # Phase 7 전체추론 러너 (7B 원리 프롬프트)
│   └── phase8a_qwen3_8b.sh     # Phase 8 전체추론 러너 (Qwen3-8B)
├── requirements.txt            # 검증된 cu128 의존성 핀 (RTX 5080)
├── outputs/                    # 추론 결과/제출 CSV (gitignore)
├── models/                     # 모델 가중치/체크포인트
└── docs/                       # 문서
    ├── DESIGN_LOG.md           #   설계/실험 로그 (Phase 1~12)
    ├── SUBMISSIONS.md          #   제출 후보 매니페스트 + 최종 선택 근거
    └── ENVIRONMENT.md          #   상세 환경 구성 가이드
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

## 평가 기준

- **평가지표**: Accuracy.
- **점수 분할** (대회 페이지 기준):
  - **Public Score**: 전체 test 8,500개 중 사전 샘플링된 **약 60%** 기준 (대회 중 리더보드 표시).
  - **Private Score**: 나머지 **약 40%** 기준 (대회 종료 후 공개, **최종 순위**).
- ⚠️ **Public 포화·블라인드**: 운영진 확인상 Public은 오픈벤치라 ~0.99에 몰려 모델 선택 신호로 쓰기 어렵다. **Public만 보고 튜닝 금지.** 대신 **합성 자체검증 하니스**(텍스트 BBQ + 멀티모달, `src/build_robustness_*` / `eval_robustness_*`)로 편향·강건성·이미지 활용도를 측정해 일반화를 판단한다.
- ⚠️ **멀티모달 필수**: 이미지 없이 텍스트만 푸는 모델은 유효하지 않다(위 '현재 상태' 참조). 이미지를 **정당한 시각 증거**(행동·물체·상호작용)로는 쓰되 **외모 고정관념**(체격·나이·복장→능력/성격)은 배제한다(`--system-prompt v5`).
- **일반 BBQ 추론 원리**(증거 기반, 소거법, 고정관념 배제)로 푼다. 미세 이득(±0.002)은 노이즈일 수 있으므로 원리적·일반화 개선에 집중한다.

## 제출 일정

- 대회 종료: 2026-06-29
- 파이널리스트 코드 제출/검증: 2026-07-02
- 2차 평가에서 코드 검증을 포함하며, 성별·인종·민족에 대한 공정성을 강조한다.
