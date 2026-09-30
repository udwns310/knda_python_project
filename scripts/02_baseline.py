"""
02_baseline.py
===============
선택 과제(RUL 회귀)의 기준 모델입니다. 과정 진행 가이드의 모델링 규칙 "단순 기준 모델을
반드시 먼저 만들고 성능을 기록한다 — 이것이 비교의 출발점"을 회귀 문제에 적용한 것입니다.
(필수 과제인 이진 분류의 기준 모델은 가이드가 예시로 든 "Z-score 기준 모델"로 09번에 있습니다.)

이 스크립트는 센서를 하나도 보지 않는, 제일 단순한 규칙만으로 RUL을 예측합니다.
"이 정도는 데이터 분석 없이 상식으로도 맞출 수 있는 수준"을 기준선으로 잡아두면,
나중에 Random Forest나 GRU가 그보다 얼마나 더 잘하는지(혹은 더 잘하지 못하는지)를
객관적으로 보여줄 수 있습니다.

베이스라인 규칙 (아주 단순합니다):
  "학습 데이터에 있는 엔진들의 평균 수명이 M 사이클이라면,
   지금 t번째 사이클을 돌고 있는 엔진의 남은 수명은 대략 (M - t)일 것이다."
  → RUL_baseline(t) = clip(M - t, 0, RUL_CLIP_VALUE)

이건 그 엔진의 센서값을 전혀 보지 않고 "지금까지 몇 번 돌았는지"만 보고 판단하는
겁니다. 즉, "다른 엔진들이 평균 M 사이클 만에 고장났으니, 이 엔진도 대충
그 정도에 고장나겠지"라는 아주 단순한 가정입니다 (제조사 권장 수명으로 교체 시점을
정하는 방식과 같은 발상).
"""

import json
import numpy as np
import pandas as pd

from common import (
    load_raw, add_rul_labels, split_engines, RUL_CLIP_VALUE,
    mae, rmse, nasa_score,
)

# 폴더 경로는 common.py에서 PC에 상관없이 자동으로 계산됩니다 (0번 섹션 참고).
from common import OUT_METRICS, TRAIN_FILE, TEST_FILE, RUL_FILE

# ---------------------------------------------------------------------------
# 1. 데이터 로드 + train/validation 엔진 분할 (common.py의 split_engines 사용 —
#    모든 모델이 "같은 엔진 분할"을 써야 서로 공정하게 비교할 수 있으므로 반드시
#    common.py의 함수를 그대로 재사용합니다. 절대 여기서 새로 랜덤 분할하지 않습니다.)
# ---------------------------------------------------------------------------
train_raw = load_raw(TRAIN_FILE)
train_raw = add_rul_labels(train_raw)
train_units, val_units = split_engines(train_raw)

train_df = train_raw[train_raw["unit"].isin(train_units)].copy()
val_df = train_raw[train_raw["unit"].isin(val_units)].copy()

print(f"train 엔진 수: {len(train_units)}, validation 엔진 수: {len(val_units)}")

# ---------------------------------------------------------------------------
# 2. 베이스라인 "학습" — train 엔진들의 평균 수명(M) 계산
# ---------------------------------------------------------------------------
mean_life = train_df.groupby("unit")["cycle"].max().mean()
print(f"\ntrain 엔진들의 평균 수명(M) = {mean_life:.1f} 사이클")
print("→ 이 값 하나가 베이스라인의 유일한 '학습 파라미터'입니다.")


def baseline_predict(cycles: np.ndarray) -> np.ndarray:
    raw = mean_life - cycles
    return np.clip(raw, 0, RUL_CLIP_VALUE)


# ---------------------------------------------------------------------------
# 3. validation 엔진들에 대해 평가
# ---------------------------------------------------------------------------
val_pred = baseline_predict(val_df["cycle"].values)
val_true = val_df["RUL"].values

val_metrics = {
    "MAE": mae(val_true, val_pred),
    "RMSE": rmse(val_true, val_pred),
    "NASA_score": nasa_score(val_true, val_pred),
}
print("\n[validation 성능]")
for k, v in val_metrics.items():
    print(f"  {k}: {v:.2f}")

val_out = val_df[["unit", "cycle", "RUL"]].copy()
val_out = val_out.rename(columns={"RUL": "RUL_true"})
val_out["RUL_pred"] = val_pred
val_out.to_csv(f"{OUT_METRICS}/baseline_val_predictions.csv", index=False)

# ---------------------------------------------------------------------------
# 4. 공식 test 세트 평가 (test_FD00X.txt + RUL_FD00X.txt = 정답)
#    이 test 세트는 "마지막에 딱 한 번만" 쓰는 최종 블라인드 테스트입니다.
#    (하이퍼파라미터를 이걸로 조정하면 안 됩니다 — 조정은 항상 validation으로만.)
# ---------------------------------------------------------------------------
test_raw = load_raw(TEST_FILE)
rul_true_file = pd.read_csv(RUL_FILE, header=None, names=["RUL"])
rul_true_file["unit"] = np.arange(1, len(rul_true_file) + 1)

# test 파일은 각 엔진이 "고장 전 어느 시점까지"만 기록되어 있고, RUL 정답 파일은
# 그 마지막으로 기록된 시점에서의 '진짜 남은 수명'을 알려줍니다. 그래서 평가는
# "각 엔진의 마지막 관측 사이클"에서만 이루어집니다 (실전에서도 "지금까지 관측한
# 데이터로 지금 시점의 RUL을 추정"하는 것과 같은 상황입니다).
last_cycle_per_unit = test_raw.groupby("unit")["cycle"].max().reset_index()
test_eval = last_cycle_per_unit.merge(rul_true_file, on="unit")
test_pred = baseline_predict(test_eval["cycle"].values)

test_metrics = {
    "MAE": mae(test_eval["RUL"].values, test_pred),
    "RMSE": rmse(test_eval["RUL"].values, test_pred),
    "NASA_score": nasa_score(test_eval["RUL"].values, test_pred),
}
print("\n[공식 test 세트 성능 (최종 블라인드 평가)]")
for k, v in test_metrics.items():
    print(f"  {k}: {v:.2f}")

test_out = test_eval.copy()
test_out = test_out.rename(columns={"RUL": "RUL_true"})
test_out["RUL_pred"] = test_pred
test_out.to_csv(f"{OUT_METRICS}/baseline_test_predictions.csv", index=False)

with open(f"{OUT_METRICS}/baseline_metrics.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "model": "naive_baseline",
            "description": "학습 엔진 평균수명(M) - 현재사이클, 0~125로 클리핑. 센서 미사용.",
            "mean_life_train": mean_life,
            "validation": val_metrics,
            "official_test": test_metrics,
        },
        f, ensure_ascii=False, indent=2,
    )

print(f"\n저장 완료: {OUT_METRICS}/baseline_metrics.json")
