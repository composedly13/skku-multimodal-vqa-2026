# 제출 파일 매니페스트 & 일일 5슬롯 계획

> 실제 제출물 = **test.csv 8500문항 예측**(`sample_id,label`). 강건성 하니스(`data/robustness*`)는
> 자체검증용 합성셋이라 **제출물 아님**. CSV는 `.gitignore` 대상(재생성 가능); 이 매니페스트만 추적.

## ⚠️ 제출 전 체크 (competition-rules)
- **R5**: 최종답 = LLM 단일생성. greedy 1패스 ✅ / SC(다수결)·룰매핑 ❌(최종 불가, 측정만).
- **R6**: 평가환경 A6000 48GB·py3.10·cu12.4·torch2.6. 최종 코드 그 환경 재현 必. 4bit 적재로 산출(재현 가능).
- **R7**: UTF-8, 8500행, 라벨∈{0,1,2}, sample 순서 = test.csv. **최종순위 = 제출창서 직접 고른 1개**(자동 아님).
  Public 60%=암기 BBQ(참고용) / Private 40%=운영진 custom 편향셋(본론). **Public↑≠Private↑.**

## 즉시 제출 가능 (검증 완료, 컴퓨팅 불필요)
| 파일 | config | public | 합법 | 역할 |
| --- | --- | --- | --- | --- |
| `final_9b_textonly_v2.csv` | Qwen3.5-9B 4bit, text-only, prompt v2 | **0.996** | ✅ | **현 최종 후보**(앵커) |
| `alt_9b_image_v2.csv` | 〃 + 이미지 | 0.99433 | ✅ | 이미지 대조(외모편향 입증돼 비선호) |
| `safety_8b_v2bal.csv` | Qwen3-8B, text-only, v2 | 0.98925 | ✅ | 약모델 안전판 |

(❌ `phase8_q3_8b_v2bal_sc.csv` 0.991 = SC → **rule5 위반, 최종 절대 불가**. 재제출 금지.)

## 오늘 생성할 새 후보 (정보가치 있는 슬롯 사용)
알려진 점수 파일 재제출은 슬롯 낭비. **프롬프트 하드닝 v3**가 유일한 새 실험.
v3 = v2 + rule7(이름·억양·복장·소지품 등 proxy·distractor는 비증거) + rule8(암시적 단일행동도 증거,
과보수 금지). Phase 10 실측의 진짜 잔존 신호(proxy>explicit 편향·distractor 끌림·weak 과보수) 겨냥.
일반 원리라 rule2(leakage)·rule5(단일생성) 무관.

```bash
# WSL challenge_q35, cd /mnt/d/skku-multimodal-vqa-2026

# (0) 먼저 v3가 강건성에서 정말 편향을 줄이는지 싸게 검증(~8분) — public 슬롯 쓰기 전에
python -m src.phase9_vlm_infer --model Qwen/Qwen3.5-9B --load-4bit --no-image --system-prompt v3 \
    --data-csv data/robustness_v2/robustness.csv \
    --output outputs/robustness_v2_pred_v3.csv --dump-raw outputs/robustness_v2_pred_v3_raw.csv
python -m src.eval_robustness_v2 --pred outputs/robustness_v2_pred_v3.csv --dump-bias outputs/robustness_v2_bias_v3.csv
#   ↳ v2 대비 proxy 편향·distractor 끌림·weak 과보수가 줄면 v3 채택 근거. (v2.1 baseline도 같이 재측정)

# (1) ★주 신규 후보: 9B text-only + v3 (test 제출용, ~70분)
python -m src.phase9_vlm_infer --model Qwen/Qwen3.5-9B --load-4bit --no-image --system-prompt v3 \
    --output submissions/cand_9b_textonly_v3.csv --dump-raw outputs/cand_9b_textonly_v3_raw.csv

# (2) 보조: 9B image + v3 (rule7로 외모편향 억제 시 이미지가 덜 해로운지, ~76분)
python -m src.phase9_vlm_infer --model Qwen/Qwen3.5-9B --load-4bit --system-prompt v3 \
    --output submissions/cand_9b_image_v3.csv --dump-raw outputs/cand_9b_image_v3_raw.csv

# (3) 약모델에서도 하드닝 효과 보나: 8B text-only + v3
python -m src.phase9_vlm_infer --model Qwen/Qwen3-8B --load-4bit --no-image --system-prompt v3 \
    --output submissions/cand_8b_textonly_v3.csv --dump-raw outputs/cand_8b_textonly_v3_raw.csv
```

## 오늘의 5슬롯 계획 (정보 최대화)
| # | 제출 파일 | 목적 |
| --- | --- | --- |
| 1 | `final_9b_textonly_v2.csv` (0.996) | 합법 베스트 앵커 재확인(현 라이브 베스트와 비교 기준) |
| 2 | `cand_9b_textonly_v3.csv` | **주 실험**: 하드닝이 public 유지하나? (강건성↑면 채택) |
| 3 | `cand_9b_image_v3.csv` | rule7로 이미지 외모편향 억제 시 이미지 가치 재평가 |
| 4 | `cand_8b_textonly_v3.csv` | 하드닝이 약모델에도 일반적으로 듣나(원리 견고성) |
| 5 | (예비) `alt_9b_image_v2.csv` 또는 추가 v3 변형 | 잔여 슬롯 — 그날 발견 따라 |

**판정:** public은 "안 떨어졌나" 확인용. **진짜 채택 근거는 강건성 v2.1 지표**(편향·공정성갭·proxy 델타).
v3가 robustness에서 v2보다 편향↓이고 public이 0.996 근처 유지면 → **새 최종 후보 = v3**로 승격.

## 최종 1개 선택 가이드 (제출 마감 시)
- public 점수보다 **강건성/공정성 지표 + 재현성**으로 선택(Private 40% + 7/2 코드·공정성 2차심사).
- 선택 파일의 **정확한 생성 명령**(모델·프롬프트버전·4bit 여부)을 이 매니페스트에 박제 → A6000 재현.
- 동률이면 단순·외모편향 없는 text-only 우선.
