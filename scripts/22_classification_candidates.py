"""
22_classification_candidates.py
===============================
분류 후보(LogisticRegression, RandomForest, HistGradientBoosting, IsolationForest)를 "같은 경보 기준 0.36"으로 다시 비교합니다.

배경: 11번(model screening)은 공정 비교를 위해 모든 후보를 기본 임곗값 0.5로 돌렸습니다. 우리가 최종으로 쓰는 RandomForest는
0.36(09번에서 train 교차검증으로 고름)이라, 발표에서 "0.36으로 돌린 결과" 표가 따로 필요해서 같은 분할·특성으로 다시 계산합니다.

  - 임곗값 0.5와 0.36 두 가지를 모두 계산 (확률을 내는 모델만. IsolationForest는 이상 점수 기준이라 11번과 같은 규칙 사용)
  - 임곗값과 무관한 지표도 함께: ROC-AUC, PR-AUC(Average Precision)
  - 보정 오차(ECE): 예측 확률이 실제 위험 비율과 맞는 정도 (확률을 내는 모델만, 낮을수록 좋음).
    "확률 ≥ 0.36이면 경보"라는 기준이 의미 있으려면 확률이 믿을 만해야 해서 함께 계산
  - 정밀도·재현율·F1 외에 미탐율(FN/(TP+FN)), 오탐율(FP/(FP+TN)), 헛경보비율(FP/(TP+FP)), 혼동행렬 숫자
  - validation(49대, 모든 시점)과 공식 test(248대, 마지막 시점) 둘 다

주의: 0.36은 RandomForest를 기준으로 고른 값이므로 다른 모델에 같은 숫자를 쓰는 것은 "같은 경보 기준"으로 맞춘 비교일 뿐,
각 모델의 최적 임곗값이 아닙니다. 임곗값과 무관한 AUC를 같이 보는 이유입니다.

실행: python scripts/22_classification_candidates.py
결과: outputs/FD004/metrics/classification_candidates_036.csv, classification_candidates_036.json
"""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from common import (
    load_raw, find_constant_columns, add_rul_labels, split_engines, Normalizer, build_rf_features,
    SENSOR_COLS_RAW, DANGER_RUL, TRAIN_FILE, TEST_FILE, RUL_FILE, OUT_METRICS,
)

SEED = 42
TH_CHOSEN, TH_DEFAULT = 0.36, 0.5

train_raw = add_rul_labels(load_raw(TRAIN_FILE))
train_units, val_units = split_engines(train_raw)
tr_raw = train_raw[train_raw["unit"].isin(train_units)]
const_cols = find_constant_columns(tr_raw)
sensors = [c for c in SENSOR_COLS_RAW if c not in const_cols]
norm = Normalizer().fit(tr_raw, sensors)
train_df, val_df = norm.transform(tr_raw), norm.transform(train_raw[train_raw["unit"].isin(val_units)])
test_df = norm.transform(load_raw(TEST_FILE))
rul = pd.read_csv(RUL_FILE, header=None, names=["RUL"])
rul["unit"] = np.arange(1, len(rul) + 1)
y_test = rul.set_index("unit")["RUL"]

train_feat, val_feat, test_feat = (build_rf_features(d, sensors) for d in (train_df, val_df, test_df))
cols = [c for c in train_feat.columns if c not in ("unit", "cycle", "RUL")]
test_last = test_feat.loc[test_feat.groupby("unit")["cycle"].idxmax()].set_index("unit").loc[y_test.index]
Xtr, ytr = train_feat[cols], (train_feat["RUL"] <= DANGER_RUL).astype(int)
Xva, yva = val_feat[cols], (val_feat["RUL"] <= DANGER_RUL).astype(int)
Xte, yte = test_last[cols], (y_test <= DANGER_RUL).astype(int)


def ece(y, p, bins=10):
    """보정 오차(Expected Calibration Error): 확률 구간별 (실제 위험 비율 − 평균 예측 확률)의 가중 평균."""
    y, p = np.asarray(y), np.asarray(p)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return round(float(total), 4)


def metrics(prefix, y, pred, score):
    y, pred = np.asarray(y), np.asarray(pred).astype(int)
    tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
    fp, tn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 0)).sum())
    P, R = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {f"{prefix}_TP": tp, f"{prefix}_FN": fn, f"{prefix}_FP": fp, f"{prefix}_TN": tn,
            f"{prefix}_precision": round(P, 3), f"{prefix}_recall": round(R, 3), f"{prefix}_F1": round(2 * P * R / max(1e-9, P + R), 3),
            f"{prefix}_miss_rate": round(fn / max(1, tp + fn), 3), f"{prefix}_false_alarm_rate": round(fp / max(1, fp + tn), 3),
            f"{prefix}_false_alarm_share": round(fp / max(1, tp + fp), 3),
            f"{prefix}_ROC_AUC": round(float(roc_auc_score(y, score)), 3), f"{prefix}_PR_AUC": round(float(average_precision_score(y, score)), 3)}


rows = []
clfs = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED),
}
for name, clf in clfs.items():
    clf.fit(Xtr, ytr)
    pv, pt = clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]
    for th in (TH_DEFAULT, TH_CHOSEN):
        row = {"model": name, "threshold": th}
        row.update(metrics("val", yva, pv >= th, pv))
        row.update(metrics("test", yte, pt >= th, pt))
        row["val_ECE"], row["test_ECE"] = ece(yva, pv), ece(yte, pt)
        rows.append(row)
        print(name, th, "val F1", row["val_F1"], "P", row["val_precision"], "R", row["val_recall"], "ECE", row["val_ECE"], "| test F1", row["test_F1"])

# IsolationForest: 확률이 없어서 0.36을 쓸 수 없음 → 11번과 같은 규칙(train F1 최대 이상 점수 기준)
iso = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr[train_feat["RUL"] >= 125])
s_tr, s_va, s_te = -iso.score_samples(Xtr), -iso.score_samples(Xva), -iso.score_samples(Xte)
cands = np.quantile(s_tr, np.linspace(0.5, 0.995, 200))
th_iso = cands[int(np.argmax([f1_score(ytr, (s_tr >= c).astype(int)) for c in cands]))]
row = {"model": "IsolationForest (비지도)", "threshold": "이상 점수 기준(train F1 최대)"}
row.update(metrics("val", yva, s_va >= th_iso, s_va))
row.update(metrics("test", yte, s_te >= th_iso, s_te))
rows.append(row)
print("IsolationForest", "val F1", row["val_F1"], "AUC", row["val_ROC_AUC"], "| test F1", row["test_F1"])

df = pd.DataFrame(rows)
df.to_csv(f"{OUT_METRICS}/classification_candidates_036.csv", index=False, encoding="utf-8-sig")
with open(f"{OUT_METRICS}/classification_candidates_036.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print("저장:", f"{OUT_METRICS}/classification_candidates_036.csv")
