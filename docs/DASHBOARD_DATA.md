# 대시보드용 데이터 스키마

> 5조 "감시자들" Sentinel 프로젝트 — 시연용 대시보드 저장소(`posco-knda/5_Sentinel_dashboard`)와의 연동 형식입니다.
>
> 대시보드가 실제로 읽는 파일은 `scripts/08_export_dashboard_ts.py`가 만드는
> `outputs/<데이터셋>/dashboard_data/mock.ts`이고, 이 파일을 대시보드 레포의 `src/data/mock.ts`로 복사합니다
> (기본 데이터셋 FD004 → `outputs/FD004/dashboard_data/mock.ts`).
> 아래 JSON들은 07번이 만드는 중간 산출물로, mock.ts의 재료이자 나중에 백엔드 API로 바꿀 때 응답 형식의 기준입니다.

## 1. `model_comparison.json`
모델 성능 비교 탭에 사용. 레코드 배열, 필드:
| 필드 | 설명 |
|---|---|
| `model` | 모델 이름 (`naive_baseline` / `random_forest` / `gru_seq2seq`) |
| `val_MAE`, `val_RMSE`, `val_NASA` | 내부 검증셋(학습에 안 쓴 엔진 — FD004 49대, FD001 20대)의 모든 사이클 기준 성능 |
| `test_MAE`, `test_RMSE`, `test_NASA` | 공식 test 엔진(FD004 248대, FD001 100대)의 마지막 관측 시점 기준 최종 성능 |

MAE/RMSE는 낮을수록 좋음 (단위: cycle). NASA_score도 낮을수록 좋음(비대칭 벌점 — 늦은
예측에 더 큰 벌점을 주는 NASA PHM08 공식 채점식).

## 2. `engines_summary.json`
엔진 목록/카드 뷰에 사용. 공식 test 엔진 각각 1개 레코드:
| 필드 | 설명 |
|---|---|
| `unit` | 엔진 번호 |
| `last_observed_cycle` | 지금까지 관측된 마지막 사이클 번호 |
| `predicted_RUL` | GRU 모델(04번)이 예측한 잔존수명 (cycle) |
| `true_RUL_for_validation_only` | **실제 정답값** — 검증용으로만 포함. **실무 배포 시에는 이 필드가 없다고 가정하고 UI를 설계해야 합니다** (실제 설비는 정답을 모름). |
| `risk_level` | `RED`(≤40 cycle, 정비 경보) / `YELLOW`(40 초과~80 미만, 정비 준비) / `GREEN`(≥80 cycle, 정상). 근거는 `docs/DESIGN_DECISIONS.md` 임의 설정값 #7·#11 |

## 3. `engine_timeseries/{unit}.json`
엔진 하나를 클릭했을 때 상세 그래프를 그리기 위한 데이터. 파일당 엔진 1개:
```json
{
  "unit": 1,
  "cycles": [1, 2, 3, ...],
  "predicted_RUL_curve": [124.8, 123.1, ...],   // cycles와 같은 길이, 각 시점의 예측 RUL
  "sensors": {
    "s2": [641.82, 642.15, ...],                 // cycles와 같은 길이
    "s3": [...], "...": [...]
  }
}
```
`sensors`에는 모델이 쓰는 센서(FD004 15개, FD001 14개)가 들어 있습니다. FD004처럼 비행 조건이 바뀌는
데이터는 **운전조건 보정값**(원래값 − 그 조건의 train 평균 + train 전체 평균, 단위는 원래와 같음)이고,
FD001은 원본값입니다. 센서 이름·단위는 `scripts/common.py`의 `SENSOR_MEANING`·`SENSOR_UNIT`.

## 4. `error_cases.json`
"모델의 한계" 설명 섹션에 사용. `danger_case`(위험한 오류: RUL 과대예측),
`conservative_case`(보수적 오류: RUL 과소예측) 각각 `unit`, `RUL_true`, `RUL_pred`,
`residual` 포함. 대응하는 그래프 이미지는 `outputs/<데이터셋>/figures/error_case_danger.png`,
`error_case_conservative.png`.

## 5. `mock.ts` (08번 — 대시보드가 실제로 읽는 파일)
주요 export: `dataMeta`(데이터셋·모델 이름·엔진 수), `riskThresholds`(위험 40 / 주의 80 — 화면 문구도 이 값을 읽음),
`equipmentRanking`(규칙으로 고른 엔진 9대), `sensorSeries`·`alertLog`(대표 엔진의 마지막 25사이클),
`engineDetails`(엔진별 90% 구간·생존곡선·센서 추이), `trendMeta`(센서 이름·단위·축 범위),
`classifierMetrics`, `featureImportance`, `scenarioCompare`·`costModel`(비용 시뮬레이션).
엔진 선정 규칙과 계산 방식은 08번 스크립트 상단 주석과 `docs/DESIGN_DECISIONS.md` #8~#10 참고.

## 프론트엔드 구현 시 참고
- 이 파일들은 **정적 스냅샷**입니다 (한 번의 파이프라인 실행 결과). 실시간 대시보드를
  만들려면 이 구조를 유지한 채로 백엔드 API가 매번 새로 계산해서 같은 JSON 스키마로
  응답하도록 구현하면 프론트엔드 코드를 거의 바꾸지 않고 연결할 수 있습니다.
- `risk_level`의 3단계 색상 구분은 그대로 신호등 UI(빨강/노랑/초록)에 매핑하면 자연스럽습니다.
