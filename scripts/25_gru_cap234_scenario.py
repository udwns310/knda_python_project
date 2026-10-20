"""
25_gru_cap234_scenario.py
==========================
하루 비행 횟수가 많은 시나리오(24번)에서 최적 위험 기준 N이 100을 넘는데, 기본 GRU는 남은 수명을 125에서 잘라 학습해
예측값이 125 근처에서 막힙니다(N이 110~120이면 거의 항상 경보). 그래서 **이 시나리오에서만** 남은 수명 라벨 상한을
엔진 수명 중앙값(234)으로 올려 GRU를 다시 학습하고, 24번과 같은 정비 시뮬레이션·공식 test 평가를 다시 합니다.

기본 설정(04번)과 달라지는 것은 라벨 상한(125 → 234) 하나입니다. 구조·시드(42)·엔진 분할·정규화·학습 설정은 같습니다.
기본 모델·본 분석 결과는 건드리지 않고, 결과는 outputs/kr_public/ 에만 저장합니다.

실행: python scripts/25_gru_cap234_scenario.py        (CPU로 십수 분)
결과: outputs/kr_public/gru_cap234_{val,test}_predictions.csv, gru_cap234_summary.json, gru_cap234_threshold.csv
"""

import json
import time

import numpy as np
import pandas as pd
import torch

from common import (
    load_raw, find_constant_columns, split_engines, Normalizer, SENSOR_COLS_RAW, mae, TRAIN_FILE, TEST_FILE, RUL_FILE,
    PROJECT_ROOT, OUT_METRICS, DANGER_RUL,
)
from seq_models import RULGRU, train_recurrent

CAP = 234
SEED = 42
OUT = PROJECT_ROOT / "outputs" / "kr_public"
OUT.mkdir(parents=True, exist_ok=True)

# --- 데이터 (04번과 같은 분할·정규화, 라벨 상한만 CAP) -----------------------------------------
train_raw = load_raw(TRAIN_FILE)
life = train_raw.groupby("unit")["cycle"].max()
train_raw["RUL"] = np.minimum(train_raw["unit"].map(life) - train_raw["cycle"], CAP)
train_units, val_units = split_engines(train_raw)
const_cols = find_constant_columns(train_raw[train_raw["unit"].isin(train_units)])
sensors = [c for c in SENSOR_COLS_RAW if c not in const_cols]
norm = Normalizer().fit(train_raw[train_raw["unit"].isin(train_units)], sensors)
train_df = norm.transform(train_raw[train_raw["unit"].isin(train_units)])
val_df = norm.transform(train_raw[train_raw["unit"].isin(val_units)])


def sequences(df, has_label=True):
    seqs, labels, ids = [], [], []
    for uid, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        seqs.append(torch.tensor(g[sensors].values, dtype=torch.float32))
        if has_label:
            labels.append(torch.tensor(g["RUL"].values / CAP, dtype=torch.float32))
        ids.append(uid)
    return seqs, labels, ids


@torch.no_grad()
def predict(model, seq):
    out = model(seq.unsqueeze(0), torch.tensor([len(seq)])).squeeze(0).numpy()
    return np.clip(out * CAP, 0, CAP)


train_seqs, train_labels, _ = sequences(train_df)
val_seqs, val_labels, val_ids = sequences(val_df)
t0 = time.time()
model = train_recurrent(RULGRU, train_seqs, train_labels, len(sensors), seed=SEED)
print(f"학습 완료 {time.time() - t0:.0f}초")

# --- validation 예측(전체 궤적) ---------------------------------------------------------------
rows = []
for uid, seq, lab in zip(val_ids, val_seqs, val_labels):
    pred = predict(model, seq)
    for c, t_, p_ in zip(np.arange(1, len(seq) + 1), lab.numpy() * CAP, pred):
        rows.append({"unit": uid, "cycle": c, "RUL_true": t_, "RUL_pred": p_})
val = pd.DataFrame(rows)
val.to_csv(OUT / "gru_cap234_val_predictions.csv", index=False)

# --- 공식 test 예측(마지막 시점) --------------------------------------------------------------
test_raw = load_raw(TEST_FILE)
rul_file = pd.read_csv(RUL_FILE, header=None, names=["RUL"])
rul_file["unit"] = np.arange(1, len(rul_file) + 1)
tseqs, _, tids = sequences(norm.transform(test_raw), has_label=False)
test = pd.DataFrame([{"unit": u, "cycle": len(s), "RUL_pred": predict(model, s)[-1]} for u, s in zip(tids, tseqs)])
test = test.merge(rul_file, on="unit").rename(columns={"RUL": "RUL_true"})
test.to_csv(OUT / "gru_cap234_test_predictions.csv", index=False)

# --- 기존 GRU(상한 125)와 정확도 비교 --------------------------------------------------------
old_val = pd.read_csv(f"{OUT_METRICS}/gru_val_predictions.csv")
old_test = pd.read_csv(f"{OUT_METRICS}/gru_test_predictions.csv")
low_v = val["RUL_true"] <= 125
low_t = test["RUL_true"] <= 125
summary = {
    "cap": CAP,
    "val_mae_all": mae(val["RUL_true"], val["RUL_pred"]),
    "val_mae_true_le_125": mae(val.loc[low_v, "RUL_true"], val.loc[low_v, "RUL_pred"]),
    "val_mae_true_le_40": mae(val.loc[val["RUL_true"] <= 40, "RUL_true"], val.loc[val["RUL_true"] <= 40, "RUL_pred"]),
    "test_mae_all": mae(test["RUL_true"], test["RUL_pred"]),
    "test_mae_true_le_125": mae(test.loc[low_t, "RUL_true"], test.loc[low_t, "RUL_pred"]),
    "old125_val_mae_all": mae(old_val["RUL_true"], old_val["RUL_pred"]),
    "old125_test_mae_all": mae(old_test["RUL_true"], old_test["RUL_pred"]),
}
# 기존 모델은 라벨이 125로 잘려 있으므로, 같은 기준(정답 125 이하 시점)으로만 비교
ov = old_val["RUL_true"] <= 125
oldv_pred = old_val.merge(val[["unit", "cycle", "RUL_true"]].rename(columns={"RUL_true": "RUL_raw_cap"}), on=["unit", "cycle"])
summary["old125_val_mae_true_le_40"] = mae(old_val.loc[old_val["RUL_true"] <= 40, "RUL_true"], old_val.loc[old_val["RUL_true"] <= 40, "RUL_pred"])
summary["old125_test_mae_true_le_125"] = mae(old_test.loc[old_test["RUL_true"] <= 125, "RUL_true"], old_test.loc[old_test["RUL_true"] <= 125, "RUL_pred"])
with open(OUT / "gru_cap234_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))

# --- 24번과 같은 정비 시뮬레이션 (시간 단위 = 일 수 × 하루 비행 횟수 k) -------------------------
COST = {"1:5": 5.0, "1:2": 2.0}
PREP, SLOT, REVIEW, N_SIM = 7, (10, 20), 7, 200
vp = val.sort_values(["unit", "cycle"])
N_GRID = list(range(10, CAP + 1, 5))
alarm_at = {n: {u: (int(g.loc[g["RUL_pred"] <= n, "cycle"].iloc[0]) if (g["RUL_pred"] <= n).any() else None)
                for u, g in vp.groupby("unit")} for n in N_GRID}


def lead(rng, k):
    return int((rng.integers(0, REVIEW) + PREP + rng.integers(0, rng.integers(SLOT[0], SLOT[1] + 1))) * k)


def simulate(alarm, cf, k):
    rng = np.random.default_rng(SEED)
    cost = ops = fails = waste = 0.0
    for _ in range(N_SIM):
        for u, a in alarm.items():
            L = int(life[u])
            m = None if a is None else a + lead(rng, k)
            if m is None or m >= L:
                cost += cf; ops += L; fails += 1
            else:
                cost += 1.0; ops += m; waste += L - m
    n = N_SIM * len(alarm)
    return {"cost_per_1000_cycles": 1000 * cost / ops, "failure_rate": fails / n, "avg_wasted_cycles": waste / max(1, n - fails)}


grid = []
for k in (1, 5):
    for cname, cf in COST.items():
        for n in N_GRID:
            grid.append({"cycles_per_day": k, "cost_ratio": cname, "N": n, **simulate(alarm_at[n], cf, k)})
grid = pd.DataFrame(grid)
# 공식 test (고장 임박 = 남은 수명 40 이하; 헛경보 = 경보했지만 실제 남은 수명이 N 초과)
tt = []
for n in N_GRID:
    a = test["RUL_pred"] <= n
    tt.append({"N": n, "test_alarm": int(a.sum()), "test_missed_imminent": int(((test["RUL_true"] <= DANGER_RUL) & ~a).sum()),
               "test_false_alarm": int((a & (test["RUL_true"] > n)).sum())})
grid = grid.merge(pd.DataFrame(tt), on="N")
grid.to_csv(OUT / "gru_cap234_threshold.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200)
for k in (1, 5):
    for c in COST:
        g = grid[(grid.cycles_per_day == k) & (grid.cost_ratio == c)]
        b = g.loc[g["cost_per_1000_cycles"].idxmin()]
        print(f"k={k} 비용{c}: 최적 N={int(b.N)} 비용={b.cost_per_1000_cycles:.2f} 정비 전 고장={b.failure_rate:.3f}")
print(grid[(grid.cycles_per_day == 5) & (grid.cost_ratio == "1:5") & (grid.N.isin([40, 95, 110, 120, 150, 180, 200]))].round(3).to_string(index=False))
