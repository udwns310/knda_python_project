# 대시보드용 데이터 스키마

> 5조 "감시자들" Sentinel 프로젝트 — 대시보드 저장소(`posco-knda/5_Sentinel_dashboard`)에
> 이 폴더의 파일들을 그대로 연결하거나, 백엔드 API 응답을 아래 구조에 맞춰 구현하면 됩니다.
>
> **참고**: 현재 대시보드는 아래 JSON을 직접 읽지 않고, 이 JSON들로부터
> `scripts/08_export_dashboard_ts.py`가 만든 `mock.ts`를 `src/data/mock.ts`로 복사해서 씁니다.
> 아래 스키마는 나중에 백엔드 API로 바꿀 때의 참고용입니다.

## 1. `model_comparison.json`
모델 성능 비교 탭에 사용. 레코드 배열, 필드:
| 필드 | 설명 |
|---|---|
| `model` | 모델 이름 (`naive_baseline` / `random_forest` / `lstm_seq2seq`) |
| `val_MAE`, `val_RMSE`, `val_NASA` | 내부 검증셋(학습에 안 쓴 20개 엔진) 성능 |
| `test_MAE`, `test_RMSE`, `test_NASA` | 공식 test 100개 엔진 기준 최종 성능 (실전과 가장 가까운 수치) |

MAE/RMSE는 낮을수록 좋음 (단위: cycle). NASA_score도 낮을수록 좋음(비대칭 벌점 — 늦은
예측에 더 큰 벌점을 주는 NASA PHM08 공식 채점식).

## 2. `engines_summary.json`
엔진 목록/카드 뷰에 사용. 테스트 세트 100개 엔진 각각 1개 레코드:
| 필드 | 설명 |
|---|---|
| `unit` | 엔진 번호 (1~100) |
| `last_observed_cycle` | 지금까지 관측된 마지막 사이클 번호 |
| `predicted_RUL` | LSTM 모델이 예측한 잔존수명 (cycle) |
| `true_RUL_for_validation_only` | **실제 정답값** — 이 프로젝트가 검증용으로 갖고 있는 C-MAPSS 데이터의 정답. **실무 배포 시에는 이 필드가 없다고 가정하고 UI를 설계해야 합니다** (실제 설비는 정답을 모름). 지금은 "모델이 얼마나 잘 맞았는지" 보여주는 데모/검증 용도로만 포함했습니다. |
| `risk_level` | `RED`(<20 cycle, 즉시 정비 필요) / `YELLOW`(20~60 cycle, 정비 계획 필요) / `GREEN`(>60 cycle, 정상). 임계값 근거는 `docs/DESIGN_DECISIONS.md` 참고 |

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
`sensors`에 포함된 센서는 14개(상수 센서 10개 제외 후 남은 것)이며, 목록과 물리적 의미는
`docs/DESIGN_DECISIONS.md`의 "사용 센서" 표를 참고하세요.

## 4. `error_cases.json`
"모델의 한계" 설명 섹션에 사용. `danger_case`(위험한 오류: RUL 과대예측),
`conservative_case`(보수적 오류: RUL 과소예측) 각각 `unit`, `RUL_true`, `RUL_pred`,
`residual` 포함. 대응하는 그래프 이미지는 `outputs/figures/error_case_danger.png`,
`outputs/figures/error_case_conservative.png`.

## 프론트엔드 구현 시 참고
- 이 파일들은 **정적 스냅샷**입니다 (한 번의 파이프라인 실행 결과). 실시간 대시보드를
  만들려면 이 구조를 유지한 채로 백엔드 API가 매번 새로 계산해서 같은 JSON 스키마로
  응답하도록 구현하면 프론트엔드 코드를 거의 바꾸지 않고 연결할 수 있습니다.
- `risk_level`의 3단계 색상 구분은 그대로 신호등 UI(빨강/노랑/초록)에 매핑하면 자연스럽습니다.
