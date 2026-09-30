"""
11_model_screening.py
=====================
탐색 실험: "어떤 모델이 최선인가?" — 후보 모델을 같은 조건에서 한 번씩 돌려 비교하고, 최종 모델 선정 근거로 씁니다.
(처음 최종 모델은 참고자료 원형인 LSTM이었으나, 이 비교에서 GRU가 미탐·헛경보·MAE 모두 앞서 04번 최종 모델을
 GRU로 바꿨습니다 — 선정 기준은 docs/DESIGN_DECISIONS.md 임의 설정값 #14)

과제 규칙("머신러닝 모델은 1~2종(최대 3종)만 사용")은 **최종 보고서에 쓰는 모델** 기준입니다. 이 스크립트는
최종 모델을 고르기 위한 근거를 남기는 탐색 단계이고, 보고서 본문에는 여기서 고른 최대 3종만 씁니다.
(과제 문서의 "모델 5~6종 나열" 경고처럼 전부를 깊게 해석하려는 것이 아닙니다.)

공정한 비교를 위해 모든 후보가 같은 것을 씁니다:
  - 같은 엔진 분할(seed=42, 80:20), 같은 정규화(운전조건별 Z-score), 같은 입력
    · 표 형태 모델: 03번 RandomForest와 같은 롤링 특성 (센서 현재값 + 최근 15사이클 평균·표준편차·기울기)
    · 시계열 모델: 04번 GRU와 같은 센서 시퀀스
  - 같은 평가: validation = 검증 엔진의 모든 사이클, test = 공식 test 엔진의 마지막 관측 시점
  - 하이퍼파라미터 탐색 없이 기본값 수준 (RandomForest·GRU는 03·04번 결과를 그대로 가져옴)

후보
  회귀(RUL 숫자 예측): Ridge(선형), RandomForest, ExtraTrees, HistGradientBoosting(부스팅), MLP(얕은 신경망),
                       GRU(04번 최종), LSTM(MathWorks 원형), 1D-CNN
  분류(위험 = 잔여 ≤ 위험 기준, common.py DANGER_RUL): LogisticRegression, RandomForest, HistGradientBoosting, IsolationForest(비지도)
                  — 과제 문서 권장 목록(RandomForest, LogisticRegression, IsolationForest, One-Class SVM) 중심

실행: python scripts/11_model_screening.py  (CMAPSS_DATASET으로 FD001/FD004 선택, FD004는 약 20분)
결과: outputs/<데이터셋>/metrics/model_screening_*.csv, outputs/<데이터셋>/figures/model_screening.png
"""

import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesRegressor, HistGradientBoostingRegressor,
                              HistGradientBoostingClassifier, IsolationForest)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import f1_score, precision_score, recall_score

from common import (
    load_raw, find_constant_columns, add_rul_labels, split_engines, Normalizer, build_rf_features,
    SENSOR_COLS_RAW, DANGER_RUL, mae, rmse, nasa_score,
    DATASET, TRAIN_FILE, TEST_FILE, RUL_FILE, OUT_METRICS, OUT_FIG,
)
from seq_models import engine_sequences, train_recurrent, predict_recurrent, train_cnn, predict_cnn, RULLSTM

SEED = 42

# ---------------------------------------------------------------------------
# 1. 데이터 (03·04·09번과 같은 분할·정규화·특성)
# ---------------------------------------------------------------------------
train_raw = add_rul_labels(load_raw(TRAIN_FILE))
train_units, val_units = split_engines(train_raw)
tr_raw = train_raw[train_raw["unit"].isin(train_units)]
const_cols = find_constant_columns(tr_raw)
sensors = [c for c in SENSOR_COLS_RAW if c not in const_cols]
norm = Normalizer().fit(tr_raw, sensors)
train_df = norm.transform(tr_raw)
val_df = norm.transform(train_raw[train_raw["unit"].isin(val_units)])

test_raw = load_raw(TEST_FILE)
test_df = norm.transform(test_raw)
rul = pd.read_csv(RUL_FILE, header=None, names=["RUL"])
rul["unit"] = np.arange(1, len(rul) + 1)
y_test = rul.set_index("unit")["RUL"]

train_feat = build_rf_features(train_df, sensors)
val_feat = build_rf_features(val_df, sensors)
test_feat = build_rf_features(test_df, sensors)
feature_cols = [c for c in train_feat.columns if c not in ("unit", "cycle", "RUL")]
test_last = test_feat.loc[test_feat.groupby("unit")["cycle"].idxmax()].set_index("unit").loc[y_test.index]

Xtr, ytr = train_feat[feature_cols], train_feat["RUL"]
Xva, yva = val_feat[feature_cols], val_feat["RUL"]
Xte = test_last[feature_cols]
print(f"[{DATASET}] 센서 {len(sensors)}개, 특성 {len(feature_cols)}개, train 행 {len(Xtr):,}, "
      f"validation 행 {len(Xva):,}, test 엔진 {len(Xte)}")


def reg_row(name, kind, p_val, p_test, seconds):
    return {"model": name, "type": kind,
            "val_MAE": round(mae(yva, p_val), 2), "val_RMSE": round(rmse(yva, p_val), 2),
            "test_MAE": round(mae(y_test, p_test), 2), "test_RMSE": round(rmse(y_test, p_test), 2),
            "test_NASA": round(nasa_score(y_test, p_test), 1), "seconds": round(seconds)}


# ---------------------------------------------------------------------------
# 2. 회귀 후보
# ---------------------------------------------------------------------------
reg_rows = []
reg_preds = {}  # 모델 이름 → (validation 예측, test 예측) — 아래 4번 오탐·미탐율 계산에 사용
tabular = {
    "Ridge (선형회귀)": Ridge(alpha=1.0),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=SEED),
    "MLP (신경망 64-32)": MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=300, random_state=SEED),
}
for name, model in tabular.items():
    t0 = time.time()
    model.fit(Xtr, ytr)
    p_val, p_test = np.clip(model.predict(Xva), 0, 125), np.clip(model.predict(Xte), 0, 125)
    reg_preds[name] = (p_val, p_test)
    reg_rows.append(reg_row(name, "표 형태", p_val, p_test, time.time() - t0))
    print(f"  {name}: val MAE {reg_rows[-1]['val_MAE']}, test MAE {reg_rows[-1]['test_MAE']} ({reg_rows[-1]['seconds']}초)")

# 02·03·04번에서 이미 학습·평가한 베이스라인, RandomForest, GRU는 그 결과를 그대로 씀 (같은 분할·정규화)
for name, kind, fname in (("베이스라인 (02번, 센서 미사용)", "기준", "baseline"),
                          ("RandomForest (03번)", "표 형태", "rf"), ("GRU (04번)", "시계열", "gru")):
    with open(f"{OUT_METRICS}/{fname}_metrics.json", encoding="utf-8") as f:
        m = json.load(f)
    v, t = m["validation"], m["official_test"]
    reg_rows.append({"model": name, "type": kind, "val_MAE": round(v["MAE"], 2), "val_RMSE": round(v["RMSE"], 2),
                     "test_MAE": round(t["MAE"], 2), "test_RMSE": round(t["RMSE"], 2),
                     "test_NASA": round(t["NASA_score"], 1), "seconds": None})
    pv = val_feat[["unit", "cycle"]].merge(pd.read_csv(f"{OUT_METRICS}/{fname}_val_predictions.csv"),
                                           on=["unit", "cycle"], how="left")["RUL_pred"].to_numpy()
    pt = pd.read_csv(f"{OUT_METRICS}/{fname}_test_predictions.csv").set_index("unit").loc[y_test.index, "RUL_pred"].to_numpy()
    reg_preds[name] = (pv, pt)

train_seqs, train_labels, _ = engine_sequences(train_df, sensors)
val_seqs, val_labels, _ = engine_sequences(val_df, sensors)
test_seqs, _, test_ids = engine_sequences(test_df, sensors, has_label=False)
order = pd.Series(range(len(test_ids)), index=test_ids).loc[y_test.index]

for name, fit, predict in (
    ("LSTM (MathWorks 원형)", lambda: train_recurrent(RULLSTM, train_seqs, train_labels, len(sensors), SEED),
     predict_recurrent),
    ("1D-CNN (30사이클 창)", lambda: train_cnn(train_seqs, train_labels, len(sensors), SEED), predict_cnn),
):
    t0 = time.time()
    print(f"  {name} 학습 중...")
    model = fit()
    p_val = np.concatenate([predict(model, s) for s in val_seqs])
    p_test = np.array([predict(model, test_seqs[i])[-1] for i in order])
    # 엔진 번호·사이클 순서로 이어 붙였으므로 validation 특성 표(val_feat)와 행 순서가 같음
    reg_preds[name] = (p_val, p_test)
    row = reg_row(name, "시계열", p_val, p_test, time.time() - t0)
    reg_rows.append(row)
    print(f"  {name}: val MAE {row['val_MAE']}, test MAE {row['test_MAE']} ({row['seconds']}초)")

reg_res = pd.DataFrame(reg_rows).sort_values("test_RMSE")
reg_res.to_csv(f"{OUT_METRICS}/model_screening_regression.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# 3. 분류 후보 (위험 = 잔여 ≤ DANGER_RUL, 임곗값은 모두 기본값 0.5 — 공정 비교를 위해 조정하지 않음)
# ---------------------------------------------------------------------------
ytr_c, yva_c = (ytr <= DANGER_RUL).astype(int), (yva <= DANGER_RUL).astype(int)
yte_c = (y_test <= DANGER_RUL).astype(int)


def cls_row(name, pv, pt, seconds):
    return {"model": name,
            "val_F1": round(f1_score(yva_c, pv), 3), "val_Precision": round(precision_score(yva_c, pv, zero_division=0), 3),
            "val_Recall": round(recall_score(yva_c, pv), 3), "test_F1": round(f1_score(yte_c, pt), 3),
            "seconds": round(seconds)}


cls_rows = []
classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED),
}
for name, clf in classifiers.items():
    t0 = time.time()
    clf.fit(Xtr, ytr_c)
    cls_rows.append(cls_row(name, clf.predict(Xva), clf.predict(Xte), time.time() - t0))
    print(f"  {name}: val F1 {cls_rows[-1]['val_F1']}, test F1 {cls_rows[-1]['test_F1']}")

# IsolationForest: 정답 없이 "정상(RUL이 125로 잘린 초반 구간)"만 학습 → 이상 점수가 높을수록 위험.
# 이상 점수 기준값은 train에서 F1이 최대인 값 (09번 Z-score 기준 모델과 같은 방식).
t0 = time.time()
iso = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr[ytr >= 125])
s_tr, s_va, s_te = -iso.score_samples(Xtr), -iso.score_samples(Xva), -iso.score_samples(Xte)
cands = np.quantile(s_tr, np.linspace(0.5, 0.995, 200))
th = cands[int(np.argmax([f1_score(ytr_c, (s_tr >= c).astype(int)) for c in cands]))]
cls_rows.append(cls_row("IsolationForest (비지도)", (s_va >= th).astype(int), (s_te >= th).astype(int), time.time() - t0))
print(f"  IsolationForest: val F1 {cls_rows[-1]['val_F1']}, test F1 {cls_rows[-1]['test_F1']}")

# 참고: 회귀 모델의 예측 RUL을 위험 기준(DANGER_RUL)으로 잘라 분류로 쓴 결과 (09번 "회귀→분류"와 같은 방식)
for name in ("GRU (04번)",):
    pv = (pd.read_csv(f"{OUT_METRICS}/gru_val_predictions.csv")["RUL_pred"] <= DANGER_RUL).astype(int)
    lt = pd.read_csv(f"{OUT_METRICS}/gru_test_predictions.csv").set_index("unit").loc[y_test.index]
    vt = pd.read_csv(f"{OUT_METRICS}/gru_val_predictions.csv")
    cls_rows.append({"model": f"GRU 회귀 → RUL≤{DANGER_RUL} (참고)",
                     "val_F1": round(f1_score((vt["RUL_true"] <= DANGER_RUL).astype(int), pv), 3),
                     "val_Precision": round(precision_score((vt["RUL_true"] <= DANGER_RUL).astype(int), pv), 3),
                     "val_Recall": round(recall_score((vt["RUL_true"] <= DANGER_RUL).astype(int), pv), 3),
                     "test_F1": round(f1_score(yte_c, (lt["RUL_pred"] <= DANGER_RUL).astype(int)), 3), "seconds": None})

cls_res = pd.DataFrame(cls_rows).sort_values("val_F1", ascending=False)
cls_res.to_csv(f"{OUT_METRICS}/model_screening_classification.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# 4. 회귀 모델을 "경보"로 썼을 때의 오탐율·미탐율 (예측 RUL ≤ DANGER_RUL 이면 위험 경보)
# ---------------------------------------------------------------------------
# 같은 MAE라도 "위험을 놓치는 쪽으로 틀리는지, 괜히 경보하는 쪽으로 틀리는지"는 다를 수 있어서,
# 회귀 예측을 대시보드 신호등(빨간불 = 예측 RUL ≤ DANGER_RUL)처럼 썼을 때 얼마나 틀리는지 계산합니다.
#   미탐율 = 실제 위험인데 경보를 못 한 비율      = FN / (TP + FN)   (= 1 − Recall, 낮을수록 좋음)
#   오탐율 = 실제 정상인데 경보를 울린 비율       = FP / (FP + TN)   (낮을수록 좋음)
#   헛경보 비율 = 울린 경보 중 틀린 경보의 비율   = FP / (TP + FP)   (= 1 − Precision)
def alarm_rates(y_true_rul, p_rul):
    t, p = np.asarray(y_true_rul) <= DANGER_RUL, np.asarray(p_rul) <= DANGER_RUL
    tp, fn, fp, tn = (t & p).sum(), (t & ~p).sum(), (~t & p).sum(), (~t & ~p).sum()
    return {"TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
            "미탐율": round(fn / (tp + fn), 3), "오탐율": round(fp / (fp + tn), 3),
            "헛경보비율": round(fp / (tp + fp), 3) if tp + fp else None}


rate_rows = []
for name, (pv, pt) in reg_preds.items():
    v, t = alarm_rates(yva, pv), alarm_rates(y_test, pt)
    mae_row = next(r for r in reg_rows if r["model"] == name)
    rate_rows.append({"model": name, "val_MAE": mae_row["val_MAE"], "test_MAE": mae_row["test_MAE"],
                      **{f"val_{k}": x for k, x in v.items()}, **{f"test_{k}": x for k, x in t.items()}})
rate_res = pd.DataFrame(rate_rows).sort_values("val_미탐율")
rate_res.to_csv(f"{OUT_METRICS}/model_screening_alarm_rates.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# 5. 그림 + 출력
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
r = reg_res.sort_values("test_RMSE", ascending=False)
axes[0].barh(r["model"], r["test_RMSE"],
             color=[{"시계열": "tab:red", "기준": "tab:gray"}.get(t, "tab:blue") for t in r["type"]])
for i, v in enumerate(r["test_RMSE"]):
    axes[0].text(v, i, f" {v:.1f}", va="center", fontsize=8)
axes[0].set_xlabel("공식 test RMSE (cycle, 낮을수록 좋음)")
axes[0].set_title("회귀(RUL) — 빨강: 시계열 / 파랑: 표 형태 / 회색: 기준 모델")
c = cls_res.sort_values("val_F1")
axes[1].barh(c["model"], c["val_F1"], color="tab:green")
for i, v in enumerate(c["val_F1"]):
    axes[1].text(v, i, f" {v:.3f}", va="center", fontsize=8)
axes[1].set_xlim(0, 1.05)
axes[1].set_xlabel("validation F1 (모든 사이클, 높을수록 좋음)")
axes[1].set_title(f"분류(위험 = 잔여 ≤ {DANGER_RUL}) — 임곗값 0.5 고정")
fig.suptitle(f"{DATASET} 모델 비교 (같은 분할·특성·평가, 튜닝 없음)", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT_FIG}/model_screening.png", dpi=120)
plt.close(fig)

print("\n[회귀 — test RMSE 순]")
print(reg_res.to_string(index=False))
print("\n[분류 — validation F1 순]")
print(cls_res.to_string(index=False))
print(f"\n[회귀 예측을 경보로 쓸 때 (예측 RUL ≤ {DANGER_RUL}) — validation 미탐율 순]")
print(rate_res[["model", "val_MAE", "val_미탐율", "val_오탐율", "val_헛경보비율",
                "test_미탐율", "test_오탐율", "test_헛경보비율"]].to_string(index=False))
