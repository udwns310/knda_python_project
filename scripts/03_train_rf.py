"""
03_train_rf.py
===============
모델 1: Random Forest 회귀 (표 형태 데이터 기반 모델)

왜 Random Forest를 첫 ML 모델로 골랐는지:
  - 해석이 쉽습니다 (어떤 피처가 중요한지 바로 뽑을 수 있어서, 도메인 해석/발표 단계에서
    "왜 이 센서가 중요한지"를 설명하기 좋습니다 — 01번 EDA에서 뽑은 추세성 상위 센서와
    실제로 일치하는지 06번에서 확인합니다).
  - 과정 진행 가이드의 권장 모델(RandomForest, LogisticRegression, IsolationForest,
    One-Class SVM) 중 회귀에도 그대로 쓸 수 있는 모델입니다.
  - 센서 값 스케일이나 분포 형태에 크게 민감하지 않아서 안정적으로 잘 작동합니다.
  - 학습이 빨라서(수 분 내) 하이퍼파라미터를 실제로 몇 가지 비교해보고 고를 여유가
    있습니다 (아래 2번 참고).

입력 피처: common.py의 build_rf_features()로 만든
  - 각 센서의 현재값
  - 최근 15사이클 이동평균 / 이동표준편차 / 추세기울기
"""

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

from common import (
    load_raw, find_constant_columns, add_rul_labels, split_engines,
    Normalizer, build_rf_features, SENSOR_COLS_RAW,
    mae, rmse, nasa_score,
)

# 폴더 경로는 common.py에서 PC에 상관없이 자동으로 계산됩니다 (0번 섹션 참고).
from common import OUT_METRICS, OUT_MODELS, TRAIN_FILE, TEST_FILE, RUL_FILE

# ---------------------------------------------------------------------------
# 1. 데이터 준비 (baseline과 완전히 동일한 분할을 재사용 — common.py의 seed=42
#    덕분에 항상 같은 엔진들이 validation으로 뽑힙니다. 모델마다 다른 분할을
#    쓰면 비교가 불공정해지므로 이 점이 중요합니다.)
# ---------------------------------------------------------------------------
train_raw = load_raw(TRAIN_FILE)
train_raw = add_rul_labels(train_raw)
train_units, val_units = split_engines(train_raw)

const_cols = find_constant_columns(train_raw[train_raw["unit"].isin(train_units)])
active_sensors = [c for c in SENSOR_COLS_RAW if c not in const_cols]
print(f"사용 센서 {len(active_sensors)}개: {active_sensors}")

norm = Normalizer().fit(train_raw[train_raw["unit"].isin(train_units)], active_sensors)

train_df = norm.transform(train_raw[train_raw["unit"].isin(train_units)])
val_df = norm.transform(train_raw[train_raw["unit"].isin(val_units)])

train_feat = build_rf_features(train_df, active_sensors)
val_feat = build_rf_features(val_df, active_sensors)

feature_cols = [c for c in train_feat.columns if c not in ("unit", "cycle", "RUL")]
print(f"피처 개수: {len(feature_cols)}")

X_train, y_train = train_feat[feature_cols], train_feat["RUL"]
X_val, y_val = val_feat[feature_cols], val_feat["RUL"]

# ---------------------------------------------------------------------------
# 2. [임의 설정값 #5] Random Forest 하이퍼파라미터 — 후보 몇 개를 실제로 학습해서
#    validation MAE가 가장 낮은 조합을 고릅니다 ("감으로 정하지 않고 검증 성능으로
#    검증했다"는 근거를 남기기 위함입니다). 후보 범위는 데이터 규모(학습 샘플
#    FD004 약 4.9만 행·피처 60개, FD001 약 1.7만 행·피처 56개)를 고려해 "너무 얕지도, 지나치게
#    깊어서 과적합 나지도 않을 법한" 상식적인 범위로 잡았습니다. 두 데이터셋 모두 같은 조합이 선택됨.
# ---------------------------------------------------------------------------
candidates = [
    {"n_estimators": 100, "max_depth": 8},
    {"n_estimators": 200, "max_depth": 10},
    {"n_estimators": 300, "max_depth": 12},
    {"n_estimators": 300, "max_depth": None},
]

print("\n[하이퍼파라미터 후보 비교 — validation MAE 기준]")
results = []
trained_models = {}
for cand in candidates:
    t0 = time.time()
    m = RandomForestRegressor(
        n_estimators=cand["n_estimators"],
        max_depth=cand["max_depth"],
        random_state=42,   # 재현성을 위한 고정 시드 (다른 의미 없음, 관례적 값)
        n_jobs=-1,
        min_samples_leaf=5,  # 잎 노드 최소 샘플 5개: 너무 잘게 쪼개져서 과적합하는 것 방지
    )
    m.fit(X_train, y_train)
    pred = m.predict(X_val)
    val_mae = mae(y_val, pred)
    elapsed = time.time() - t0
    key = f"{cand['n_estimators']}_{cand['max_depth']}"
    trained_models[key] = m
    results.append({**cand, "val_MAE": val_mae, "seconds": round(elapsed, 1), "_key": key})
    print(f"  {cand} -> val MAE={val_mae:.2f} ({elapsed:.1f}초)")

best = min(results, key=lambda r: r["val_MAE"])
print(f"\n선택된 하이퍼파라미터: {best}")

# ---------------------------------------------------------------------------
# 3. 최종 모델 = 위 비교 루프에서 이미 학습해둔 "제일 좋은" 모델을 그대로 재사용
#    (똑같은 설정으로 또 학습하면 시간만 두 배로 드므로, 이미 학습된 객체를 사용)
# ---------------------------------------------------------------------------
final_model = trained_models[best["_key"]]
best = {k: v for k, v in best.items() if k != "_key"}
for r in results:
    r.pop("_key", None)

val_pred = final_model.predict(X_val)
val_metrics = {
    "MAE": mae(y_val, val_pred),
    "RMSE": rmse(y_val, val_pred),
    "NASA_score": nasa_score(y_val, val_pred),
}
print("\n[validation 최종 성능]")
for k, v in val_metrics.items():
    print(f"  {k}: {v:.2f}")

val_out = val_feat[["unit", "cycle", "RUL"]].copy().rename(columns={"RUL": "RUL_true"})
val_out["RUL_pred"] = val_pred
val_out.to_csv(f"{OUT_METRICS}/rf_val_predictions.csv", index=False)

# ---------------------------------------------------------------------------
# 4. 공식 test 세트 평가
# ---------------------------------------------------------------------------
test_raw = load_raw(TEST_FILE)
rul_true_file = pd.read_csv(RUL_FILE, header=None, names=["RUL"])
rul_true_file["unit"] = np.arange(1, len(rul_true_file) + 1)

test_norm = norm.transform(test_raw)
test_feat = build_rf_features(test_norm, active_sensors)

# test는 각 엔진의 "마지막으로 관측된 사이클" 시점에서만 평가합니다 (baseline과 동일 규칙).
last_idx = test_feat.groupby("unit")["cycle"].idxmax()
test_last = test_feat.loc[last_idx].merge(rul_true_file, on="unit", suffixes=("_dummy", ""))
test_pred = final_model.predict(test_last[feature_cols])

test_metrics = {
    "MAE": mae(test_last["RUL"].values, test_pred),
    "RMSE": rmse(test_last["RUL"].values, test_pred),
    "NASA_score": nasa_score(test_last["RUL"].values, test_pred),
}
print("\n[공식 test 세트 성능 (최종 블라인드 평가)]")
for k, v in test_metrics.items():
    print(f"  {k}: {v:.2f}")

test_out = test_last[["unit", "cycle", "RUL"]].copy().rename(columns={"RUL": "RUL_true"})
test_out["RUL_pred"] = test_pred
test_out.to_csv(f"{OUT_METRICS}/rf_test_predictions.csv", index=False)

# ---------------------------------------------------------------------------
# 5. 피처 중요도 (도메인 해석용으로 저장)
# ---------------------------------------------------------------------------
importance = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
importance.head(20).to_csv(f"{OUT_METRICS}/rf_feature_importance_top20.csv", header=["importance"])
print("\n[피처 중요도 상위 10개]")
print(importance.head(10))

# 모델과 정규화기, 사용 피처 목록을 저장 (06번 센서 중요도, 10번 교차 검증에서 재사용)
joblib.dump(
    {"model": final_model, "normalizer": norm, "active_sensors": active_sensors,
     "feature_cols": feature_cols, "hyperparams": best},
    f"{OUT_MODELS}/rf_model.joblib",
)

with open(f"{OUT_METRICS}/rf_metrics.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "model": "random_forest",
            "description": "롤링윈도우(15) 평균/표준편차/기울기 + 현재값 피처 기반 RF 회귀",
            "hyperparam_search": results,
            "chosen_hyperparams": best,
            "validation": val_metrics,
            "official_test": test_metrics,
        },
        f, ensure_ascii=False, indent=2,
    )
print(f"\n저장 완료: {OUT_METRICS}/rf_metrics.json, {OUT_MODELS}/rf_model.joblib")
