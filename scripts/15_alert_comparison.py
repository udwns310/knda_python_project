"""
15_alert_comparison.py
======================
같은 조건에서 "RF 분류"와 "GRU(남은 수명 예측 → 40 이하면 경보)"를 비교하는 실험.

배경: 지금까지는 RF 분류가 확률 기준을 0.5 → 0.36으로 낮춰(놓침을 줄이도록) 평가됐고,
GRU는 기준 40을 그대로 썼습니다. 그래서 "RF가 더 잘 잡는다"가 모델 차이인지, 경보 기준을
다르게 잡은 차이인지 알 수 없었습니다. 이 스크립트는 두 가지를 확인합니다.

  실험 1 (validation 49대): 두 방식의 경보를 같은 정비 시뮬레이션(13번과 같은 정책·비용)에 넣어
          "정비 전에 고장 나는 엔진 비율"과 비용을 비교합니다.
  실험 2 (공식 test 248대, 엔진별 마지막 관측 시점): 두 방식의 경보 기준을 쭉 바꿔 가며
          같은 재현율에서 헛경보가 몇 대인지(놓침 ↔ 헛경보 맞교환)를 비교합니다.

실행: python scripts/15_alert_comparison.py   (약 1분)
결과: outputs/FD004/metrics/alert_comparison_cost.csv, alert_comparison_test_sweep.csv, alert_comparison_summary.json
"""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from common import (
    load_raw, find_constant_columns, add_rul_labels, split_engines, Normalizer, build_rf_features,
    SENSOR_COLS_RAW, DANGER_RUL, OUT_METRICS, TRAIN_FILE, TEST_FILE, RUL_FILE,
)

RF_PARAMS = dict(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)  # 09번과 동일
RF_PROB_TH = 0.36                                                                    # 09번에서 고른 값
COST_SETTINGS = {"1:5": 5.0, "1:2": 2.0}
PREP_DAYS, SLOT_INTERVAL, REVIEW_EVERY, N_SIM = 7, (10, 20), 7, 200                   # 13번과 동일

# ---------------------------------------------------------------------------
# 1. 데이터·RF 분류 (09번과 같은 분할·정규화·특성·설정)
# ---------------------------------------------------------------------------
train_raw = add_rul_labels(load_raw(TRAIN_FILE))
train_units, val_units = split_engines(train_raw)
const_cols = find_constant_columns(train_raw[train_raw["unit"].isin(train_units)])
sensors = [c for c in SENSOR_COLS_RAW if c not in const_cols]
norm = Normalizer().fit(train_raw[train_raw["unit"].isin(train_units)], sensors)
train_feat = build_rf_features(norm.transform(train_raw[train_raw["unit"].isin(train_units)]), sensors)
val_feat = build_rf_features(norm.transform(train_raw[train_raw["unit"].isin(val_units)]), sensors)
cols = [c for c in train_feat.columns if c not in ("unit", "cycle", "RUL")]
clf = RandomForestClassifier(**RF_PARAMS).fit(train_feat[cols], (train_feat["RUL"] <= DANGER_RUL).astype(int))

val_feat = val_feat.assign(prob=clf.predict_proba(val_feat[cols])[:, 1])
test_feat = build_rf_features(norm.transform(load_raw(TEST_FILE)), sensors)
rul_true = pd.read_csv(RUL_FILE, header=None, names=["RUL"])
rul_true["unit"] = np.arange(1, len(rul_true) + 1)
test_last = test_feat.loc[test_feat.groupby("unit")["cycle"].idxmax()].merge(rul_true, on="unit")
test_last["prob"] = clf.predict_proba(test_last[cols])[:, 1]
test_last["danger"] = (test_last["RUL"] <= DANGER_RUL).astype(int)

gru_val = pd.read_csv(f"{OUT_METRICS}/gru_val_predictions.csv")
gru_test = pd.read_csv(f"{OUT_METRICS}/gru_test_predictions.csv")
val_feat = val_feat.merge(gru_val[["unit", "cycle", "RUL_pred"]], on=["unit", "cycle"]).sort_values(["unit", "cycle"])
test_last = test_last.merge(gru_test[["unit", "RUL_pred"]], on="unit")
life = load_raw(TRAIN_FILE).groupby("unit")["cycle"].max()

# ---------------------------------------------------------------------------
# 2. 실험 1: 정비 시뮬레이션 (13번과 같은 정책) — 경보 방식만 바꿔서 비교
# ---------------------------------------------------------------------------


def lead_time(rng) -> int:
    wait_review = rng.integers(0, REVIEW_EVERY)
    interval = rng.integers(SLOT_INTERVAL[0], SLOT_INTERVAL[1] + 1)
    return int(wait_review + PREP_DAYS + rng.integers(0, interval))


def simulate(alarm_cycle: dict, cost_fail: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    total_cost = total_ops = failures = wasted = 0.0
    for _ in range(N_SIM):
        for u, a in alarm_cycle.items():
            L = int(life[u])
            m = None if a is None else a + lead_time(rng)
            if m is None or m >= L:
                total_cost += cost_fail
                total_ops += L
                failures += 1
            else:
                total_cost += 1.0
                total_ops += m
                wasted += L - m
    n = N_SIM * len(alarm_cycle)
    return {"cost": 1000 * total_cost / total_ops, "failure_rate": failures / n,
            "avg_wasted_cycles": wasted / max(1, n - failures)}


def first_alarm(signal: pd.Series, cond) -> dict:
    out = {}
    for u, g in val_feat.groupby("unit"):
        hit = g.loc[cond(g), "cycle"]
        out[u] = int(hit.iloc[0]) if len(hit) else None
    return out


rows = []
for t in [0.1, 0.2, 0.3, RF_PROB_TH, 0.4, 0.5, 0.6, 0.7, 0.8]:
    alarm = first_alarm(None, lambda g, t=t: g["prob"] >= t)
    for cname, cf in COST_SETTINGS.items():
        rows.append({"method": "RF 분류", "setting": f"확률 ≥ {t:.2f}", "cost_ratio": cname, **simulate(alarm, cf)})
for n in range(20, 81, 5):
    alarm = first_alarm(None, lambda g, n=n: g["RUL_pred"] <= n)
    for cname, cf in COST_SETTINGS.items():
        rows.append({"method": "GRU", "setting": f"예측 ≤ {n}", "cost_ratio": cname, **simulate(alarm, cf)})
cost = pd.DataFrame(rows)
cost.to_csv(f"{OUT_METRICS}/alert_comparison_cost.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# 3. 실험 2: 공식 test — 경보 기준을 바꿔 가며 놓침 ↔ 헛경보 비교
# ---------------------------------------------------------------------------
y = test_last["danger"].to_numpy()
n_pos, n_neg = int(y.sum()), int((1 - y).sum())


def counts(pred) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return {"TP": tp, "FN": n_pos - tp, "FP": fp, "recall": tp / n_pos}


sweep = []
for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
    sweep.append({"method": "RF 분류", "setting": float(t), **counts((test_last["prob"].to_numpy() >= t).astype(int))})
for n in range(20, 81):
    sweep.append({"method": "GRU", "setting": float(n), **counts((test_last["RUL_pred"].to_numpy() <= n).astype(int))})
sweep = pd.DataFrame(sweep)
sweep.to_csv(f"{OUT_METRICS}/alert_comparison_test_sweep.csv", index=False, encoding="utf-8-sig")

rf_ref = sweep[(sweep.method == "RF 분류") & (sweep.setting == RF_PROB_TH)].iloc[0]
gru_ref = sweep[(sweep.method == "GRU") & (sweep.setting == DANGER_RUL)].iloc[0]
gru = sweep[sweep.method == "GRU"]
rf = sweep[sweep.method == "RF 분류"]
# 같은 재현율(놓침 수)에서 비교
gru_at_rf = gru[gru.TP >= rf_ref.TP].sort_values("setting").iloc[0]      # RF와 같거나 더 높은 재현율이 되는 가장 작은 기준
rf_at_gru = rf[rf.TP >= gru_ref.TP].sort_values("setting", ascending=False).iloc[0]  # GRU와 같은 재현율에서 가장 높은 확률 기준

summary = {
    "test_n": {"danger": n_pos, "safe": n_neg},
    "RF_0.36": {k: float(rf_ref[k]) for k in ("TP", "FN", "FP", "recall")},
    "GRU_40": {k: float(gru_ref[k]) for k in ("TP", "FN", "FP", "recall")},
    "GRU_matched_to_RF_recall": {"setting": float(gru_at_rf.setting), **{k: float(gru_at_rf[k]) for k in ("TP", "FN", "FP", "recall")}},
    "RF_matched_to_GRU_recall": {"setting": float(rf_at_gru.setting), **{k: float(rf_at_gru[k]) for k in ("TP", "FN", "FP", "recall")}},
}
with open(f"{OUT_METRICS}/alert_comparison_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n[실험 1] validation 49대 정비 시뮬레이션 — 정비 전에 고장 나는 엔진 비율 / 비용 (1:5)")
show = cost[cost.cost_ratio == "1:5"][["method", "setting", "failure_rate", "cost", "avg_wasted_cycles"]]
print(show.round(3).to_string(index=False))
print("\n[실험 2] 공식 test 위험 {}대·안전 {}대".format(n_pos, n_neg))
print(json.dumps(summary, ensure_ascii=False, indent=2))
