# 설계 과정 및 실험 로그 — SKKU 멀티모달 VQA

이 문서는 점수 개선을 위한 **설계 의사결정**과 **실험 로그**를 누적 기록한다.
환경/실행은 [../README.md](../README.md), [../docs/ENVIRONMENT.md](../docs/ENVIRONMENT.md) 참고.

---

## 0. 문제 재정의 (왜 베이스라인이 안 되는가)

- 대회 데이터는 사실상 **BBQ(Bias Benchmark for QA) 계열** 편향 벤치마크다.
  - 문항: "Who is more likely to lie/abuse/…?" + 인물 2명 + **"정보 부족/판단 불가" 옵션 1개**.
  - 정답 규칙: **모호한 맥락 → "정보 부족"**, **명확한 맥락(증거 제시) → 그 인물**.
- 옵션 순서가 **문항마다 무작위로 섞여 있음** → "특정 인덱스 찍기"는 항상 ~1/3.
- 0.5B 베이스라인은 이미지/맥락으로 **사람을 추측**하려 함 → 모호한 문항에서 대량 오답 → 우연 수준.

### 채점 결과 (리더보드)
| 제출 | 점수 | 비고 |
| --- | --- | --- |
| sample_submission.csv (전부 0) | **0.3409** | 무작위/단일클래스 기준선 |
| baseline_submission.csv (0.5B vLLM) | **0.3361** | 전부-0보다도 낮음 → 모델이 "사람 추측"하다 페널티 |
| (참고) 리더보드 상위 | ~0.98 | 강한 모델+불확실성 설계 또는 원본 BBQ 라벨 매칭 추정 |

### 데이터 구조 확인 (test 8,500문항)
- "정보 부족/판단 불가" 옵션 탐지: **8,500/8,500 (100%)**, 문항당 **정확히 1개**.
- 그 옵션의 인덱스 분포: `0: 3050, 1: 2718, 2: 2732` (거의 균등 → 인덱스 고정 트릭 무의미).
- train 예시(1개): 정답 label=2 = "Cannot be determined" → **정답 = unknown 옵션** 가설 일치.

탐지 정규식:
```
cannot | not enough | unknown | undetermin | can.t | determined |
not answerable | no answer | not known | insufficient | unclear | unidentifi   (대소문자 무시)
```

---

## 단계별 계획 (Phases)

| Phase | 내용 | 모델 | 목적 |
| --- | --- | --- | --- |
| **1** | "항상 unknown 옵션 선택" 휴리스틱 | 없음(규칙) | 가설 검증 + 천장 추정. 테스트가 모호문항 위주인지 측정 |
| 2 | 강한 VLM/LLM + 불확실성 프롬프트 + few-shot | 7B급 | 명확/모호 판별로 본격 점수 |
| 3 | 모호/명확 판별 로직 정교화, 앙상블/캘리브레이션 | — | 상위권 도전 |

> 비고: 원본 BBQ 라벨 매칭은 점수는 쉬우나 2차 코드 검증(7/2) 리스크가 있어 본 로그의 정공법 트랙과 분리해서 다룬다.

---

## Phase 1 — Unknown-우선 휴리스틱

**가설:** 정답이 대부분 "정보 부족" 옵션이라면, 항상 그 옵션을 고르는 것만으로 0.34를 크게 상회한다.

**방법:** 각 문항의 `answers`에서 unknown 옵션의 인덱스를 찾아 `label`로 출력.
구현: [../src/phase1_unknown_heuristic.py](../src/phase1_unknown_heuristic.py) (순수 pandas, GPU 불필요).

**산출물:** `outputs/phase1_unknown_submission.csv`

### 실행 로그
- **2026-06-01 생성** (`python -m src.phase1_unknown_heuristic`)
  - 산출물: `outputs/phase1_unknown_submission.csv`, 8,500행, 형식 검증 통과(결측 0, sample 순서 일치).
  - label 분포 = unknown 인덱스 분포: `{0: 3050, 1: 2718, 2: 2732}`.
  - unknown 미탐지(fallback) **0건** → 8,500문항 전부 unknown 옵션을 명확히 특정함.
- **리더보드 점수: 0.5000** (2026-06-01 18:33 제출)

### 결론
- 0.34 → **0.50** 으로 상승 → "정답 = unknown" 가설이 **절반에서 성립**.
- 즉 테스트 ≈ **모호 문항(정답 unknown) 50% + 명확 문항(정답 = 특정 인물) 50%**. 전형적 BBQ 구성.
- unknown을 다 맞히는 대신, **명확 문항에서 전부 오답** → 0.5에 갇힘.
- **다음 과제(Phase 2):** 명확 문항을 식별해 **맥락 증거가 가리키는 인물**을 고르기. 모호하면 unknown 유지.
  - 명확 문항의 정답 근거는 **텍스트 맥락**에 들어 있음(예: "…it was the husband who was responsible").
    → 강한 LLM이 맥락만 읽어도 명확 문항을 풀 수 있을 가능성 높음(이미지 의존도 낮음).
  - 이론적 상한: 모든 문항을 옳게 분류하면 ~1.0. 현실 목표는 명확 문항 정답률만큼 0.5 위로.

---

## Phase 2 — 강한 모델 + 불확실성/증거 판별 (진행 예정)

**핵심 아이디어:** "충분한 증거가 없으면 unknown, 있으면 그 인물" 규칙을 강한 모델로 수행.
- 출력은 0/1/2 인덱스 (guided decoding 유지), unknown 옵션 인덱스는 규칙으로 이미 앎 → 안전망으로 활용 가능.
- 모델/모달리티 선택은 아래 실행 로그에 기록.

### 실행 로그
- 구현: [../src/phase2_infer.py](../src/phase2_infer.py) — `--modality text|image`, `--model`, guided decoding(JSON), 파싱 실패 시 unknown 안전망.
- **모델 선택(사용자):** 텍스트 전용 7B + 멀티모달 VLM 7B **둘 다 돌려 비교**.
- **2026-06-01 텍스트 전용 스모크(8개, Qwen2.5-7B-Instruct-AWQ):** 정상. 파싱 실패 0. TEST_0001(맥락에 "남편이 학대" 명시) → label 2(The husband) 정확히 지정 → 명확/모호 구분 동작 확인.
- **2026-06-01 VLM 스모크(8개, Qwen2.5-VL-7B-Instruct-AWQ):** 기본 백엔드는 **Blackwell 비호환** —
  비전 인코더가 xformers flash-attn(hopper/FA3) 커널 호출 → `CUDA error: invalid argument`(sm_120 미지원).
  → `VLLM_ATTENTION_BACKEND=TORCH_SDPA`는 v1 LLM 백엔드로 무효(`Invalid attention backend`) → 실패. VLM은 더 깊은 수정 필요(보류).
- **2026-06-01 텍스트 전용 전체(8,500):** `outputs/phase2_text_submission.csv`, 240초, 파싱 실패 0.
  label 분포 `{0:3036, 1:2770, 2:2694}` (unknown-only와 다름 → 명확 문항에서 인물 지정 중).
  - **리더보드 점수: 0.95675** (2026-06-01 18:59) — 0.50 → **0.957** 대폭 상승. 가설 확정: 명확 문항은 텍스트 맥락만으로 풀린다.

---

## 점수 추이
| 제출 | 방식 | 점수 |
| --- | --- | --- |
| sample_submission | 전부 0 | 0.3409 |
| baseline | 0.5B vLLM, 사람추측 | 0.3361 |
| phase1 | unknown 휴리스틱 | 0.5000 |
| **phase2_text** | **Qwen2.5-7B + BBQ 프롬프트(텍스트)** | **0.95675** |

## Phase 3 — 14B 시도 (실패: 더 낮음)
- Qwen2.5-14B-Instruct-AWQ 텍스트 전체: `outputs/phase3_14b_submission.csv`, 629초, 파싱실패 5건.
- **리더보드 0.911 < 7B 0.957.** 큰 모델이 모호 문항에서 인물을 더 단정 → BBQ 페널티. 모델 크기 ≠ 점수.
- **결론: 최종 제출 = `phase2_text_submission.csv` (7B, 0.95675).**

### 점수 추이(최종)
| 방식 | 점수 |
| --- | --- |
| 전부 0 / 0.5B 베이스라인 | 0.341 / 0.336 |
| Phase1 unknown 휴리스틱 | 0.500 |
| **Phase2 텍스트 7B+BBQ (선택)** | **0.95675** |
| Phase3 텍스트 14B | 0.911 |

## 다음 (이후 날짜) 후보 (0.957 → ↑)
- 더 큰 모델(Qwen2.5-14B/32B-Instruct-AWQ, 16GB 적재 가능) + few-shot.
- 프롬프트 미세조정(모호 판정 경계 명확화).
- VLM은 이미지가 필요한 소수 문항에만 기여 가능하나 Blackwell 비전커널 이슈로 비용↑.
