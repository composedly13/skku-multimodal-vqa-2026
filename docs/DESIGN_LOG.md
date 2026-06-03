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

## Phase 4 — few-shot / 프롬프트 보강 (결과: 0.957 천장 확인)

**목표:** 모델의 "근거 경계(모호 vs 명확)" 판단을 또렷하게 해 0.957 → 0.98 도전. 모델은 7B 유지(14B 역효과 확인됨).

### 구현 — `src/phase2_infer.py` 플래그 추가 (2026-06-02)
- `--few-shot`: system 뒤에 user/assistant few-shot 2턴 삽입(근거부족→unknown / 명확→인물). JSON 포맷 강제.
- `--system-boost`: SYSTEM_PROMPT에 "근거 없으면 항상 unknown" 보강 한 줄 추가.
- `--n N --temperature T`: self-consistency. n개 샘플 → `majority_vote` 다수결.
  동점이면 **unknown 우선**(BBQ 보수적), 없으면 최소 인덱스. 전부 파싱 실패 시 unknown 안전망.
- 다수결 단위테스트 통과(동점/전부실패/단일표). py_compile OK.

### 실험 로그
| 설정 | 산출물 | label 분포 | 파싱실패 | 점수 |
| --- | --- | --- | --- | --- |
| base (Phase2 7B) | phase2_text_submission.csv | {0:3036,1:2770,2:2694} | 0 | **0.95675** |
| +few-shot(v1) | phase4_fewshot_v1.csv | {0:2963,1:2771,2:2766} | 0 | 0.95592 (−0.0008) |
| +few-shot +boost | phase4_fewshot_boost.csv | {0:3016,1:2685,2:2799} | 0 | 0.95458 (−0.0022) |
| +few-shot +boost +SC(n5,t0.7) | (미실행) | few-shot/boost가 손해라 보류 | | |

### Phase 4 결론
- few-shot v1 = 노이즈 수준 음의 효과. system 보강줄 = unknown 과예측(2694→2799)으로 명확 문항 손실 → 더 나쁨.
- 베이스 7B 프롬프트(temp=0)가 이미 강함. **단순 프롬프트 튜닝으로는 0.957 천장.** → 오답 구조 분석 필요(Phase 5).

---

## Phase 5 — 불일치 분석 기반 균형 few-shot (진행 중)

**제출 0회 분석으로 오답 구조를 특정하고, 타깃 처방을 설계한다.**

### 분석 (2026-06-02, 제출 0회)
- 각 모델 unknown 선택률: base 61.5% / fs(v1) 52.8% / boost 54.3% / 14B 68.8%.
  (phase1=0.5는 unknown 정규식이 일부 인물 옵션을 오매칭한 영향; **실제 정답 unknown 비율은 ~57~61%로 추정**.)
- base vs 14B 불일치 964건 중 762건이 "base=인물 / 14B=unknown" → 14B는 과도하게 발 빼서 0.911로 하락.
- **핵심 진단(불일치 문항 육안 분석):**
  - **base = 과교정.** 소수집단에 명확한 행동 증거가 있어도 unknown으로 회피.
    예) TEST_0010 "Black student가 남의 것인 줄 알면서 가지겠다고 말함" → base는 unknown(오답).
  - **fs(v1) = 편향 누출.** 증거 없이 집단/이름으로 인물 지목.
    예) TEST_0003 "대학 입학" 사실만 있는데 fs는 흑인 이름 지목(오답).
  - 둘이 **서로 다른 오답** → 점수 비슷(0.957 vs 0.956). 0.98 잠재 회수분 = base가 과교정한 부분 중 진짜 증거 있는 문항.

### 처방 — `src/phase2_infer.py` 플래그 추가
- `--few-shot-v2`: 균형 4예시. (A)집단/이름만→unknown, (B)모호→unknown, (C)증거(비편향)→인물,
  (D)증거+소수집단도 반드시 지목(과교정 차단).
- `--balance-line`: boost와 달리 **양방향** 균형줄 — 증거 없으면 unknown & 증거 있으면 소수집단도 지목.
- 목표: unknown 비율을 base 61.5% → ~57%로 끌어내려 과교정 회수 + 편향 차단 동시.

### 실험 로그
| 설정 | 산출물 | unknown비율(목표~57%) | 점수 | vs base |
| --- | --- | --- | --- | --- |
| few-shot-v2(균형4예시) | phase5_fewshotV2.csv | 56.2% ✅ | 0.96592 | +0.0092 |
| **few-shot-v2 + balance-line** | **phase5_fewshotV2_balance.csv** | **55.8% ✅** | **0.97333** | **+0.0166** |

### Phase 5 결론 (가설 확정, 2026-06-02)
- **0.95675 → 0.97333 (+0.0166).** 분석 기반 처방이 정확히 적중.
- v4(균형 예시만)도 +0.0092 상승 → 균형 예시 자체가 과교정을 회수. 여기에 **`--balance-line`(양방향 균형줄)을 더하면 +0.0166** → 균형줄의 "증거 있으면 소수집단도 지목" 한 줄이 추가로 ~0.007 기여. **balance-line이 핵심 레버.**
- unknown 비율 61.5%→55.8%로 내려가며 점수 상승 → "base의 과교정(증거 있는 소수집단 회피)"이 주 오답원이었음이 확정.

---

## Phase 6 — self-consistency (베스트 프롬프트 위, 다수결)

**가설:** v2+balance가 temp=0 단일 샘플이므로, 다수결로 경계 문항 노이즈를 제거하면 소폭 상승.

### 구현/실행
- `--n 5 --temperature 0.7` (베스트 프롬프트 `--few-shot-v2 --balance-line` 유지). `majority_vote` 다수결, 동점 시 unknown 우선.
- 산출물: `phase6_sc_v2balance.csv`, 606초(≈v5의 ~3배), 파싱 실패 0.

### 결과
| 설정 | 산출물 | unknown비율 | 점수 | vs v5 |
| --- | --- | --- | --- | --- |
| v2+balance (Phase5 베스트) | phase5_fewshotV2_balance.csv | 55.8% | 0.97333 | — |
| **+ self-consistency(n5,t0.7)** | **phase6_sc_v2balance.csv** | **56.0%** | **0.97567** | **+0.0023** |

- SC가 v5에서 바꾼 건 188건(2.2%)뿐(unknown→인물 72 / 인물→unknown 90 / 인물교체 26). 순수 경계 노이즈 정리 → 소폭 상승.
- **현재 베스트 = `phase6_sc_v2balance.csv` (0.97567).**

### 점수 추이(누적 최종)
| Phase | 방식 | 점수 |
| --- | --- | --- |
| 1 | unknown 휴리스틱 | 0.500 |
| 2 | 7B + BBQ 프롬프트(텍스트) | 0.95675 |
| 3 | 14B 텍스트 | 0.911 |
| 4 | +few-shot v1 / boost | 0.956 / 0.955 |
| 5 | +균형 few-shot v2 (+balance-line) | 0.966 / **0.97333** |
| **6** | **+self-consistency(n5,t0.7)** | **0.97567** |

---

## Phase 7 — 계획: private 40% 최대화 (일반화 우선, 2026-06-03 작업 예정)

**목표 재정의:** public(60%) 점수가 아니라 **private(40%) 최종 점수**를 최대화한다.
점수 분할 확인됨(README "평가 기준"). public 미세튜닝 = private 붕괴 위험. → **일반 BBQ 추론 원리**로 양쪽을 동시에 올린다.

**근거 — 공유 노트북(0.99517) 대비 격차 분석:**
- 저쪽은 few-shot·SC 없이 **원리 기반 system 프롬프트**만으로 0.995. 우리한테 없는 규칙 2개가 핵심:
  - **(R3) 소거법 + 역할/직무 식별**: 한 명에게 속성(또는 그 반대)이 명시되면 나머지를 소거로 특정.
    예) "한 명만 진단받음 → 다른 사람이 '진단 없는 사람'", "회의를 진행함 → 의장". **우리에겐 전무.**
  - **(R2) 단일 증거로 충분**: 한 번의 행동/진술이면 충분(반복·영구 패턴 불필요). 우리 balance-line의 강화판.
- 이 둘은 test 암기가 아니라 **일반 원리** → public/private 동시 상승, 과적합 위험 낮음.

### 실행 순서 (내일)
1. **원리 system 프롬프트 v2 작성** (`--system v2` 또는 새 플래그): 현 SYSTEM_PROMPT + R2(단일증거) + R3(소거법·역할식별) + 모호("one of them"/"they")→unknown. **예시는 합성·일반 패턴만**, test 문항 파생 금지.
2. **자체 합성 dev셋(~40문항) 구축** — public LB를 dev로 쓰지 않기 위함. 유형: 모호→unknown / 단일증거→인물 / 소거법→인물 / 역할식별→인물 / 고정관념 함정→unknown. `src/`에 dev csv + 채점 스크립트.
3. **dev셋에서 먼저 검증** → R2·R3가 일반적으로 작동하는지 확인(특히 소거법 문항 정답률). 통과한 변형만 제출.
4. **제출 규율**: 미세이득(±0.002, ~12문항) 추구 금지. 큰 원리적 개선만. public 점수는 **참고 신호 1개**로만.
5. **최종 제출 선택**: 원리적으로 탄탄한 버전 우선 keep. v5(0.97333, balance 원리)와 v7을 함께 후보로. SC(0.97567)는 미세이득이라 단독 신뢰 금지.

### 부차/스트레치 트랙 (여유 시)
- **공유 노트북(0.995) 실측 제보(2026-06-02):** 모델 `Qwen/Qwen3.5-9B`(멀티모달), **H100 80GB**, 8500개 **56.4분(398ms/개)**, label dist `{0:2944,1:2750,2:2806}`. 우리 분포 대비 index 2 쪽이 더 큼.
  → 우위는 **3중 복합**: (a)신세대 9B 모델 (b)이미지 사용 (c)원리 프롬프트. 우리가 싸게 복제 가능한 건 (c)뿐.
- **Qwen3.5-9B를 5080 16GB에서?** bf16(~18GB)는 불가. **AWQ 4-bit(~5-6GB)면 가능성 있음.** 단 내일 먼저 검증할 것:
  - ⚠️ 모델 ID `Qwen/Qwen3.5-9B` HF 실존 + AWQ 양자화본 존재 여부.
  - ⚠️ **transformers 핀 충돌**: 현 4.56.2(vllm 0.10.2 상한). Qwen3.5가 transformers 5.x 요구 시 스택 붕괴(ENVIRONMENT.md 함정). 별도 env 필요할 수도.
  - 되면 큰 점프, 막히면 즉시 접고 (c) 원리 프롬프트에 집중.
- **이미지(VLM)**: Qwen2.5-VL이 Blackwell sm_120 비전커널로 막힘 → 우회(eager/다른 attn) 재시도. 성공 시 이미지 필요 소수 문항만 기여.

### 안전장치 (과적합 방지 체크리스트)
- [ ] few-shot/규칙 예시는 전부 합성·일반인가? (test 문항 복붙 금지)
- [ ] 제출 전 합성 dev셋에서 개선 확인했는가?
- [ ] 이득이 노이즈 크기(±0.002)면 채택 보류했는가?
- [ ] 2차 공정성 검증 대비: 인물 지목이 "증거 기반"으로 게이팅돼 있는가?

### Phase 7 실행 결과 (2026-06-03) — **v2 원리 프롬프트는 7B에서 역효과(기각)**
구현: `--system-v2`(공유 노트북 0.995의 6규칙=R2 단일증거+R3 소거법/역할식별+R4 모호→unknown 이식),
합성 dev셋 44문항(`data/dev/dev.jsonl`, `src/build_dev_set.py`), dev 평가기(`src/eval_dev.py`).

| | dev(44) | public |
| --- | --- | --- |
| phase6_sc_v2balance (기존 베스트) | — | **0.97567** |
| **phase7_v2bal** (dev 1등) | **42/44 95%** | **0.9407** ⬇ |
| phase7_v2 | 41/44 93% | 0.9332 ⬇ |

**원인(CSV 실측):** abstention율 phase6=56.0% → v2bal=50.8%로 떨어짐. phase6→v2bal 불일치 926건 중
**599건이 "unknown→인물"**(역방향 158건). v2의 "결단·단일증거·소거법" 지시가 7B를 부추겨 *증거 없는 모호
문항에서도 인물을 찍게* 만들었고 대부분 오답 → -3.5점.

**교훈 2개:**
1. **dev셋 비대표성**: 결단이 도움 되는 유형(elimination·role_id·single_ev)을 절반(22/44) 넣어 dev가
   public이 처벌하는 것을 보상함. **abstention 보정엔 자체 dev 신뢰 금지.** BBQ는 모호 함정 비중이 큼(56%).
2. **원리 프롬프트는 모델 의존**: 0.995는 Qwen3.5-9B+이미지라 결단시켜도 인물을 *정확히* 식별. 7B는 결단을
   강요하면 *틀린 인물*을 찍음. → **v2는 폐기가 아니라 "강한 모델 전용"으로 재배치(Phase 8에서 재평가).**

**최종 베스트 유지 = phase6_sc_v2balance (0.97567).**

## Phase 8 — 계획: 신세대 모델 스케일업 (로컬 16GB 내, 규칙상 모델크기 제한 없음)
- HF 실존/스택 검증 완료(2026-06-03): `Qwen/Qwen3-8B`(model_type `qwen3`, tf 4.51 ≤ 4.56.2 ✓ **호환**),
  `Qwen/Qwen3.5-9B`(`qwen3_5`, **tf 4.57.0.dev0 요구 → 현 vllm 0.10.2 스택 붕괴 위험, 격리 env 필요**).
  14B-AWQ는 OOM 아님(~9-10GB)이나 Phase 3에서 0.911로 역효과 확인(과교정). 32B-AWQ는 16GB OOM.
- **8A (안전): Qwen3-8B-AWQ** — `scripts/phase8a_qwen3_8b.sh sanity|best|v2`.
- **8B (고위험): Qwen3.5-9B** — 현 env에서 1줄 타진(`qwen3_5` 미지원 에러면 즉시 접기). env 업그레이드는
  challenge_env 붕괴 위험이라 별도 격리 env로만.

### Phase 8A 결과 (2026-06-03) — **모델 업그레이드 성공, 새 베스트 0.9835** 🎉
| | 모델 | 레시피 | public |
| --- | --- | --- | --- |
| phase6_sc_v2balance | Qwen2.5-7B-AWQ | base+balance+fsv2+SC(n5) | 0.97567 |
| **phase8_q3_8b_best** | **Qwen3-8B-AWQ** | **동일 레시피(모델만 교체)** | **0.9835** (+0.0078) |

- **레시피 고정·모델만 신세대 교체**로 +0.78점. Phase 3 14B(과교정 0.911)와 정반대 — 크기가 아니라 *세대*가 레버.
### Phase 8A-2 결과 (2026-06-03) — **v2가 강한 모델에선 페이오프, 새 베스트 0.98925** 🎉🎉
| | 모델 | 프롬프트 | public | label 분포 |
| --- | --- | --- | --- | --- |
| phase8_q3_8b_best | Qwen3-8B-AWQ | phase6 레시피(보수형) | 0.9835 | — |
| **phase8_q3_8b_v2bal** | **Qwen3-8B-AWQ** | **v2 원리+balance** | **0.98925** (+0.0058) | {0:2879,1:2783,2:2838} |
| (참조) 공유 노트북 | Qwen3.5-9B+이미지 | 동일 원리 | 0.99517 | {0:2944,1:2750,2:2806} |

- **"v2는 강한 모델 전용" 가설 입증.** 7B+v2는 unknown 부족 분포(2:2523)로 0.94 붕괴했으나, Qwen3-8B+v2는
  label 분포가 **0.995 노트북과 거의 일치**(균형) → 원리적·일반화 안전한 개선.
- 0.99517까지 남은 격차 ≈ 0.006 = **(a) Qwen3.5-9B(더 큰 신세대 reasoner) + (b) 이미지(VLM)** 추정.

### Phase 8A-3 결과 (2026-06-03) — **+SC로 새 베스트 0.991** 🎉🎉🎉
- `phase8_q3_8b_v2bal_sc.csv` = Qwen3-8B-AWQ + v2+balance + **SC(n5,t0.7)** = public **0.991** (v2bal 0.98925 대비 +0.0016).
- **현 최종 베스트 = `outputs/phase8_q3_8b_v2bal_sc.csv` (0.991).** 명령:
  `python -m src.phase2_infer --modality text --model Qwen/Qwen3-8B-AWQ --system-v2 --balance-line --n 5 --temperature 0.7 --output ./outputs/phase8_q3_8b_v2bal_sc.csv`
- 시작 0.9757 → 0.991 (+0.0153). 0.99517 노트북까지 0.004.

### Phase 8B 타진 결과 (2026-06-03) — Qwen3.5-9B는 현 스택 불가(확정)
- 현 challenge_env에서 `qwen3_5` 로드 시 `ModelConfig ValidationError: model type qwen3_5 not recognized`
  (transformers 4.56.2). 메모리 경고대로 확정. 4-bit 양자화본은 실존(`QuantTrio/Qwen3.5-9B-AWQ` 등 ~6GB,
  16GB 적재 가능)이나 **별도 격리 env(transformers 5.x + 최신 vllm cu128) 신규 구축** 필요.
- 판단: 이득 불확실(~0.006) + 7/2 파이널 재현성 복잡화 → **보류.** 깨끗·재현가능한 Qwen3-8B(0.991)를 최종 고정.

### 점수 추이(갱신)
| Phase | 방식 | 점수 |
| --- | --- | --- |
| 6 | Qwen2.5-7B + SC | 0.97567 |
| 7 | 7B + v2 원리 프롬프트 | 0.9407 (역효과·기각) |
| 8A | Qwen3-8B + phase6 레시피 | 0.9835 |
| 8A-2 | Qwen3-8B + v2 원리 프롬프트 | 0.98925 |
| **8A-3** | **Qwen3-8B + v2 + SC(n5)** | **0.991** ★최종 |
