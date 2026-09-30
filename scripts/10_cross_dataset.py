"""
10_cross_dataset.py
===================
데이터셋 교차 검증: "한 데이터셋에서 학습한 모델이 다른 데이터셋에서도 통하는가?"

  - FD001로 학습한 모델 → FD004 test에 적용
  - FD004로 학습한 모델 → FD001 test에 적용
  - 비교 기준: 각 데이터셋에서 직접 학습한 모델(같은 데이터 안에서의 성능)

평가하는 모델: GRU 회귀(04번)·RandomForest 회귀(03번) — 저장한 모델 그대로,
RandomForest 분류(09번과 같은 설정으로 학습 데이터에서 다시 학습 — 09번은 모델 파일을 저장하지 않음).

정규화 방식 두 가지를 비교합니다 (모델이 받는 입력이 "학습 때와 같은 기준"이어야 하므로 핵심 변수):
  (A) 원래 기준 그대로 : 학습한 데이터셋의 train 평균·표준편차로 새 데이터를 정규화
      - FD004 모델 → FD001: FD001의 운전조건(고도 0·마하 0·스로틀 100)이 FD004 6개 조건 중 하나라
        FD004 기준값을 그대로 적용할 수 있음
      - FD001 모델 → FD004: FD001에는 운전조건이 하나뿐이라 FD004의 나머지 5개 조건에 대한 기준이
        없음 → 모든 행에 FD001 기준을 그대로 씌움 (현장에서 "다른 조건 데이터를 그냥 넣는" 경우)
  (B) 운전조건 보정    : 새 데이터셋 train의 센서값(정답 RUL은 사용하지 않음)으로 운전조건별
      평균·표준편차를 다시 잡아 정규화 — "새 설비에 센서 데이터만 먼저 모아 기준을 맞춘 뒤 적용"하는 상황

필요: FD004·FD001 둘 다 03·04·09번을 먼저 실행해 outputs/FD004/, outputs/FD001/에 모델과 결과가 있어야 합니다.
실행: python scripts/10_cross_dataset.py   (CMAPSS_DATASET 설정과 무관하게 두 데이터셋을 모두 사용)
결과: outputs/cross_dataset/ (표 csv/json + 그림)
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from common import (
    PROJECT_ROOT, DATA, load_raw, add_rul_labels, split_engines,
    Normalizer, build_rf_features, DANGER_RUL, mae, rmse, nasa_score, out_root,
)
from seq_models import engine_sequences, predict_recurrent, load_seq_model

DATASETS = ["FD004", "FD001"]
OUT_DIR = PROJECT_ROOT / "outputs" / "cross_dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RF_CLS_PARAMS = dict(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)  # 09번과 동일
# 분류기 경보 확률 임곗값: 학습한 데이터셋에서 09번이 고른 값을 그대로 씀
PROB_THRESHOLD = {}
for _ds in DATASETS:
    with open(out_root(_ds) / "metrics" / "classification_metrics.json", encoding="utf-8") as _f:
        PROB_THRESHOLD[_ds] = json.load(_f)["threshold_adjustment"]["chosen"]


def load_dataset(ds):
    train = add_rul_labels(load_raw(f"{DATA}/train_{ds}.txt"))
    test = load_raw(f"{DATA}/test_{ds}.txt")
    rul = pd.read_csv(f"{DATA}/RUL_{ds}.txt", header=None, names=["RUL"])
    rul["unit"] = np.arange(1, len(rul) + 1)
    train_units, _ = split_engines(train)
    return {"train": train, "train_units": train_units, "test": test, "rul": rul}


data = {ds: load_dataset(ds) for ds in DATASETS}


def normalizer_for(src, tgt, sensors, mode):
    """mode A: 학습 데이터셋 기준 / mode B: 적용할 데이터셋 train 센서값으로 운전조건별 기준 재설정."""
    if mode == "B" or src == tgt:
        base = data[tgt] if mode == "B" else data[src]
        return Normalizer().fit(base["train"][base["train"]["unit"].isin(base["train_units"])], sensors)
    s = data[src]
    norm = Normalizer().fit(s["train"][s["train"]["unit"].isin(s["train_units"])], sensors)
    tgt_regimes = set(data[tgt]["test"]["regime"])
    missing = tgt_regimes - set(norm.mean_.index)
    if missing:  # 원래 기준에 없는 운전조건은 학습 데이터셋의 (유일한/첫) 기준을 그대로 씌움
        fallback = norm.mean_.index[0]
        norm.mean_ = pd.concat([norm.mean_] + [norm.mean_.loc[[fallback]].rename(index={fallback: r}) for r in missing])
        norm.std_ = pd.concat([norm.std_] + [norm.std_.loc[[fallback]].rename(index={fallback: r}) for r in missing])
    return norm


def reg_scores(y, p):
    return {"MAE": round(mae(y, p), 2), "RMSE": round(rmse(y, p), 2), "NASA": round(nasa_score(y, p), 1)}


def cls_scores(y, yhat):
    return {"F1": round(f1_score(y, yhat, zero_division=0), 3),
            "Precision": round(precision_score(y, yhat, zero_division=0), 3),
            "Recall": round(recall_score(y, yhat, zero_division=0), 3)}


# 분류기는 학습 데이터셋별로 한 번만 다시 학습 (09번과 같은 특성·설정)
classifiers = {}
for src in DATASETS:
    t0 = time.time()
    rf_pack = joblib.load(out_root(src) / "models" / "rf_model.joblib")
    s = data[src]
    tr = s["train"][s["train"]["unit"].isin(s["train_units"])]
    feat = build_rf_features(rf_pack["normalizer"].transform(tr), rf_pack["active_sensors"])
    clf = RandomForestClassifier(**RF_CLS_PARAMS).fit(feat[rf_pack["feature_cols"]], (feat["RUL"] <= DANGER_RUL).astype(int))
    classifiers[src] = clf
    print(f"[{src}] RandomForest 분류기 재학습 ({time.time() - t0:.0f}초)")

rows = []
for src in DATASETS:
    gru, gru_sensors = load_seq_model(str(out_root(src) / "models" / "gru_model.pt"))
    rf_pack = joblib.load(out_root(src) / "models" / "rf_model.joblib")
    rf_sensors, feature_cols = rf_pack["active_sensors"], rf_pack["feature_cols"]
    for tgt in DATASETS:
        for mode in (["-"] if src == tgt else ["A", "B"]):
            test, rul = data[tgt]["test"], data[tgt]["rul"]
            y_true = rul.set_index("unit")["RUL"]
            danger_true = (y_true <= DANGER_RUL).astype(int)

            # GRU
            norm = normalizer_for(src, tgt, gru_sensors, mode)
            seqs, _, ids = engine_sequences(norm.transform(test), gru_sensors, has_label=False)
            p_gru = pd.Series({u: predict_recurrent(gru, s)[-1] for u, s in zip(ids, seqs)})

            # RandomForest 회귀 · 분류 (같은 롤링 특성)
            norm = normalizer_for(src, tgt, rf_sensors, mode)
            feat = build_rf_features(norm.transform(test), rf_sensors)
            last = feat.loc[feat.groupby("unit")["cycle"].idxmax()].set_index("unit")
            p_rf = pd.Series(rf_pack["model"].predict(last[feature_cols]), index=last.index)
            prob = pd.Series(classifiers[src].predict_proba(last[feature_cols])[:, 1], index=last.index)

            u = y_true.index
            label = "같은 데이터(기준)" if src == tgt else ("A: 원래 기준 그대로" if mode == "A" else "B: 운전조건 보정")
            for name, reg in (("GRU 회귀", p_gru), ("RandomForest 회귀", p_rf)):
                rows.append({"학습": src, "평가": tgt, "정규화": label, "모델": name,
                             **reg_scores(y_true[u], reg[u]),
                             **cls_scores(danger_true[u], (reg[u] <= DANGER_RUL).astype(int))})
            rows.append({"학습": src, "평가": tgt, "정규화": label, "모델": "RandomForest 분류",
                         "MAE": None, "RMSE": None, "NASA": None,
                         **cls_scores(danger_true[u], (prob[u] >= PROB_THRESHOLD[src]).astype(int))})
            print(f"  {src} → {tgt} [{label}] GRU MAE {rows[-3]['MAE']}, RF MAE {rows[-2]['MAE']}, "
                  f"분류 F1 {rows[-1]['F1']}")

res = pd.DataFrame(rows)
res.to_csv(OUT_DIR / "cross_dataset_results.csv", index=False, encoding="utf-8-sig")
with open(OUT_DIR / "cross_dataset_results.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

# 그림: 평가 데이터셋별로 "같은 데이터 학습" vs "다른 데이터 학습(A/B)" 비교
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, tgt in zip(axes, DATASETS):
    sub = res[(res["평가"] == tgt) & (res["모델"] == "GRU 회귀")]
    names = [f"{r['학습']} 학습\n{r['정규화']}" for _, r in sub.iterrows()]
    bars = ax.bar(names, sub["MAE"], color=["tab:blue" if r["학습"] == tgt else "tab:orange" for _, r in sub.iterrows()])
    ax.bar_label(bars, fmt="%.1f")
    ax.set_title(f"{tgt} test에 적용한 GRU의 MAE (낮을수록 좋음)")
    ax.set_ylabel("MAE (cycle)")
    ax.tick_params(axis="x", labelsize=8)
fig.suptitle("데이터셋 교차 검증 — 파랑: 같은 데이터로 학습 / 주황: 다른 데이터로 학습", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "cross_dataset_gru_mae.png", dpi=120)
print(f"\n저장: {OUT_DIR}")
print(res.to_string(index=False))
