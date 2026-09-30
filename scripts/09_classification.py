"""
09_classification.py
====================
주제 F 필수 과제: "고장 임박(위험) 이진 분류"

과제 문서(주제 후보 F)의 필수 과제는
  "잔여 사이클이 임계값(예: 30) 이하인 구간을 '위험'으로 정의하고 이진 분류 모델을 만든다"
이고, RUL 회귀(02~05번에서 한 것)는 "선택 과제"입니다. 이 스크립트가 필수 과제를 담당합니다.
임계값은 예시값 30 대신, 정비 준비 기간·비용 근거로 정한 40을 씁니다 (common.py DANGER_RUL, DESIGN_DECISIONS #11).

과제 공통 요건과 이 스크립트의 대응:
  - Z-score 기준 모델을 먼저 만든다      → 2번 (이동 Z-score 건강지표)
  - scikit-learn 모델 1~2종               → 3번 (RandomForestClassifier)
  - 같은 분할·특성·지표로 비교            → 03번과 같은 엔진 분할(seed=42), 같은 롤링 특성
  - 혼동행렬 + Precision·Recall·F1         → 5번 (정확도 단독 보고 안 함)
  - 임곗값 조정 시 보전 관점 근거          → 4번 [임의 설정값 #12]
  - 오탐 1건·미탐 1건 이상을 시계열 위에 → 7번 (그림 2장)
  - 선택 과제: 회귀 vs 분류 비교           → 5·6번 (GRU 회귀 예측을 임곗값으로 잘라 같은 조건에서 비교)

실행 순서: 01 ~ 06 다음 (03·04번의 예측 결과 CSV를 비교용으로, 06번의 센서 해석 결과를 그래프
          센서 선택용으로 읽음, torch 불필요)
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GroupKFold

from common import (
    load_raw, find_constant_columns, add_rul_labels, split_engines,
    Normalizer, build_rf_features, SENSOR_COLS_RAW, SENSOR_UNIT, DANGER_RUL,
    OUT_FIG, OUT_METRICS, TRAIN_FILE, TEST_FILE, RUL_FILE, RegimeCorrector, MULTI_REGIME, DATASET,
)

# 위험 기준(잔여 ≤ 40사이클, [임의 설정값 #11])은 대시보드 빨간불과 같은 값을 쓰도록
# common.py 7번 섹션의 DANGER_RUL에 있습니다.

# 이동 Z-score 기준 모델 설정
BASELINE_NORMAL_CYCLES = 30  # 각 train 엔진의 처음 30사이클 = "확실히 정상"인 기준 구간
                             # (최소 수명 128이라 이 구간은 모두 잔여 98사이클 이상)
BASELINE_SMOOTH = 15         # 건강지표를 최근 15사이클 평균으로 부드럽게 (RF 롤링 창과 동일)

RF_PARAMS = dict(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)
N_FOLDS = 5          # 확률 임곗값을 고를 때 쓰는 train 엔진 교차검증 폴드 수
ALARM_CONSECUTIVE = 3  # 경보 선행시간 계산 시 "3사이클 연속 경보"부터 경보로 인정 (한 번 튀는 오경보 제외)

plt.rcParams["figure.dpi"] = 110


def scores(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 3),
    }


# ---------------------------------------------------------------------------
# 1. 데이터 준비 (03번 RF 회귀와 완전히 같은 분할·정규화·특성)
# ---------------------------------------------------------------------------
train_raw = add_rul_labels(load_raw(TRAIN_FILE))
train_units, val_units = split_engines(train_raw)
const_cols = find_constant_columns(train_raw[train_raw["unit"].isin(train_units)])
active_sensors = [c for c in SENSOR_COLS_RAW if c not in const_cols]

norm = Normalizer().fit(train_raw[train_raw["unit"].isin(train_units)], active_sensors)
train_df = norm.transform(train_raw[train_raw["unit"].isin(train_units)])
val_df = norm.transform(train_raw[train_raw["unit"].isin(val_units)])

train_feat = build_rf_features(train_df, active_sensors)
val_feat = build_rf_features(val_df, active_sensors)
feature_cols = [c for c in train_feat.columns if c not in ("unit", "cycle", "RUL")]

# RUL은 125에서 잘려 있지만, 위험 기준(40) 이하 구간은 잘리지 않은 값과 같으므로 라벨에 그대로 써도 됩니다.
train_feat["danger"] = (train_feat["RUL"] <= DANGER_RUL).astype(int)
val_feat["danger"] = (val_feat["RUL"] <= DANGER_RUL).astype(int)

test_raw = load_raw(TEST_FILE)
test_feat = build_rf_features(norm.transform(test_raw), active_sensors)
rul_true = pd.read_csv(RUL_FILE, header=None, names=["RUL"])
rul_true["unit"] = np.arange(1, len(rul_true) + 1)
test_last = test_feat.loc[test_feat.groupby("unit")["cycle"].idxmax()].merge(rul_true, on="unit")
test_last["danger"] = (test_last["RUL"] <= DANGER_RUL).astype(int)

print(f"위험 기준: 잔여 ≤ {DANGER_RUL}사이클")
print(f"  train 행 {len(train_feat):,} (위험 {train_feat['danger'].mean():.1%}), "
      f"validation 행 {len(val_feat):,} (위험 {val_feat['danger'].mean():.1%}), "
      f"test 엔진 {len(test_last)}대 (위험 {int(test_last['danger'].sum())}대)")

# ---------------------------------------------------------------------------
# 2. 기준 모델: 이동 Z-score 건강지표
# ---------------------------------------------------------------------------
# 아이디어: "각 센서가 정상일 때보다 몇 표준편차나 벗어났는가(Z-score)"를 센서마다 구해
# 평균 낸 값을 "건강지표"로 씁니다. 값이 클수록 정상에서 멀어진(= 닳은) 상태입니다.
#   - 정상 기준(평균·표준편차)은 train 엔진의 처음 30사이클로만 계산 (validation/test 정보 사용 안 함)
#   - 센서마다 열화될 때 올라가는 것도, 내려가는 것도 있어서, train에서 "사이클이 늘수록
#     올라가는지(+1) 내려가는지(-1)"를 보고 방향을 맞춘 뒤 평균 냅니다.
#   - 한 사이클씩은 값이 흔들리므로 최근 15사이클 평균(이동 평균)으로 부드럽게 만듭니다.
#   - 경보 기준값(τ)은 train 엔진에서 F1이 가장 높은 값으로 정합니다.
#   - 다중 운전조건(FD002·FD004)에서는 비행 조건 차이를 먼저 빼 준 "운전조건 보정값"으로
#     계산합니다 (common.py RegimeCorrector). FD001은 보정값 = 원래값이라 결과가 같습니다.
corrector = RegimeCorrector().fit(train_raw[train_raw["unit"].isin(train_units)], active_sensors)
tr = corrector.transform(train_raw[train_raw["unit"].isin(train_units)])
normal = tr[tr["cycle"] <= BASELINE_NORMAL_CYCLES]
mu, sd = normal[active_sensors].mean(), normal[active_sensors].std().replace(0, 1.0)
direction = np.sign(tr[active_sensors].corrwith(tr["cycle"]))


def health_index(df: pd.DataFrame) -> pd.Series:
    z = ((df[active_sensors] - mu) / sd) * direction
    hi = z.mean(axis=1)
    return hi.groupby(df["unit"]).transform(lambda s: s.rolling(BASELINE_SMOOTH, min_periods=1).mean())


def with_hi(raw: pd.DataFrame) -> pd.DataFrame:
    raw = corrector.transform(raw.sort_values(["unit", "cycle"]))
    raw["HI"] = health_index(raw)
    return raw[["unit", "cycle", "HI"]]


hi_train = train_feat[["unit", "cycle", "danger"]].merge(with_hi(train_raw[train_raw["unit"].isin(train_units)]), on=["unit", "cycle"])
cands = np.quantile(hi_train["HI"], np.linspace(0.5, 0.99, 200))
f1s = [f1_score(hi_train["danger"], (hi_train["HI"] >= c).astype(int)) for c in cands]
tau = float(cands[int(np.argmax(f1s))])
print(f"\n[기준 모델] 이동 Z-score 건강지표 경보 기준 τ = {tau:.2f} (train F1 최대)")

hi_val = val_feat[["unit", "cycle"]].merge(with_hi(train_raw[train_raw["unit"].isin(val_units)]), on=["unit", "cycle"])
hi_test_all = with_hi(test_raw)
hi_test = test_last[["unit", "cycle"]].merge(hi_test_all, on=["unit", "cycle"])

# ---------------------------------------------------------------------------
# 3. 머신러닝 모델: RandomForestClassifier
# ---------------------------------------------------------------------------
# 03번 RF 회귀와 같은 특성(센서 현재값 + 최근 15사이클 평균·표준편차·기울기)을 쓰고,
# 설정도 03번에서 고른 값(나무 300그루, 잎 최소 5개)을 그대로 씁니다 — 회귀와 분류를
# 같은 조건에서 비교하기 위함입니다.
X_train, y_train = train_feat[feature_cols], train_feat["danger"]

# 확률 임곗값을 고르기 위한 "train 안에서의 연습 시험": 엔진 단위 5-겹 교차검증
# (validation 엔진은 최종 평가용으로 남겨 두고, 임곗값 선택에 쓰지 않습니다)
oof = np.zeros(len(train_feat))
for tr_idx, te_idx in GroupKFold(n_splits=N_FOLDS).split(X_train, y_train, groups=train_feat["unit"]):
    m = RandomForestClassifier(**RF_PARAMS).fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
    oof[te_idx] = m.predict_proba(X_train.iloc[te_idx])[:, 1]

clf = RandomForestClassifier(**RF_PARAMS).fit(X_train, y_train)
p_val = clf.predict_proba(val_feat[feature_cols])[:, 1]
p_test = clf.predict_proba(test_last[feature_cols])[:, 1]

# ---------------------------------------------------------------------------
# 4. [임의 설정값 #12] 확률 임곗값 1회 조정 — 보전 관점 근거
# ---------------------------------------------------------------------------
# 기본값 0.5는 "오탐과 미탐을 똑같이 나쁘게" 보는 기준입니다. 그런데 현장에서는
#   - 미탐(위험한데 정상이라고 함) → 고장 날 때까지 운용 → 비계획 정지·응급 수리
#   - 오탐(정상인데 위험이라고 함) → 불필요한 점검 1회
# 로 비용 차이가 큽니다. 대시보드 비용 가정(08번 COST_ASSUMPTIONS) 기준으로
#   고장 1건 놓침 ≈ 24시간 정지 × 1,000만원 + 응급 수리 2,400만원 ≈ 2억 6,400만원
#   불필요 점검 1건 ≈ 0.5시간 × 1,000만원 + 인건비 50만원 ≈ 550만원
# 으로 약 48배입니다. 그래서 "위험 구간을 90% 이상 잡아내는(Recall ≥ 0.9) 임곗값 중에서
# 오탐이 가장 적은(Precision이 가장 높은) 값"을 고릅니다. 사이클 단위 평가에서는 한 번의
# 고장이 여러 사이클에 걸쳐 있어 48배를 그대로 비용 가중치로 쓰면 경보가 과도해지므로,
# 비용 비율은 "재현율 우선"이라는 방향만 정하는 데 쓰고 목표 재현율을 0.9로 두었습니다.
TARGET_RECALL = 0.90
grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
curve = pd.DataFrame([
    {"threshold": t, **{k: v for k, v in scores(y_train, (oof >= t).astype(int)).items()
                        if k in ("precision", "recall", "F1")}}
    for t in grid
])
ok = curve[curve["recall"] >= TARGET_RECALL]
prob_th = float(ok.sort_values(["precision", "threshold"], ascending=[False, False]).iloc[0]["threshold"])
print(f"[임곗값 조정] 0.5 → {prob_th:.2f} (train 교차검증에서 Recall ≥ {TARGET_RECALL} 중 Precision 최대)")

# ---------------------------------------------------------------------------
# 5. 평가: 혼동행렬 + Precision·Recall·F1 (정확도 단독 보고 안 함)
# ---------------------------------------------------------------------------
gru_val = pd.read_csv(f"{OUT_METRICS}/gru_val_predictions.csv")
gru_test = pd.read_csv(f"{OUT_METRICS}/gru_test_predictions.csv")
rf_reg_val = pd.read_csv(f"{OUT_METRICS}/rf_val_predictions.csv")
rf_reg_test = pd.read_csv(f"{OUT_METRICS}/rf_test_predictions.csv")


def align(pred_df, base):
    return base[["unit", "cycle"]].merge(pred_df[["unit", "cycle", "RUL_pred"]], on=["unit", "cycle"])["RUL_pred"].to_numpy()


val_preds = {
    "기준모델(이동 Z-score)": (hi_val["HI"].to_numpy() >= tau).astype(int),
    "RandomForest 분류(임곗값 0.5)": (p_val >= 0.5).astype(int),
    f"RandomForest 분류(임곗값 {prob_th:.2f})": (p_val >= prob_th).astype(int),
    f"RF 회귀→분류(예측 RUL≤{DANGER_RUL})": (align(rf_reg_val, val_feat) <= DANGER_RUL).astype(int),
    f"GRU 회귀→분류(예측 RUL≤{DANGER_RUL})": (align(gru_val, val_feat) <= DANGER_RUL).astype(int),
}
test_preds = {
    "기준모델(이동 Z-score)": (hi_test["HI"].to_numpy() >= tau).astype(int),
    "RandomForest 분류(임곗값 0.5)": (p_test >= 0.5).astype(int),
    f"RandomForest 분류(임곗값 {prob_th:.2f})": (p_test >= prob_th).astype(int),
    f"RF 회귀→분류(예측 RUL≤{DANGER_RUL})": (test_last[["unit"]].merge(rf_reg_test, on="unit")["RUL_pred"].to_numpy() <= DANGER_RUL).astype(int),
    f"GRU 회귀→분류(예측 RUL≤{DANGER_RUL})": (test_last[["unit"]].merge(gru_test, on="unit")["RUL_pred"].to_numpy() <= DANGER_RUL).astype(int),
}

rows = []
for name in val_preds:
    v = scores(val_feat["danger"], val_preds[name])
    t = scores(test_last["danger"], test_preds[name])
    rows.append({"model": name, **{f"val_{k}": x for k, x in v.items()}, **{f"test_{k}": x for k, x in t.items()}})
comparison = pd.DataFrame(rows)
comparison.to_csv(f"{OUT_METRICS}/classification_comparison.csv", index=False)

print(f"\n[validation {len(val_units)}대 · 모든 사이클 기준]")
print(comparison[["model", "val_precision", "val_recall", "val_F1", "val_TP", "val_FN", "val_FP"]].to_string(index=False))
print(f"\n[공식 test {len(test_last)}대 · 마지막 관측 시점 기준]")
print(comparison[["model", "test_precision", "test_recall", "test_F1", "test_TP", "test_FN", "test_FP"]].to_string(index=False))

# 혼동행렬 그림 (기준 모델 vs 최종 RandomForest, validation 기준)
final_name = f"RandomForest 분류(임곗값 {prob_th:.2f})"
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, name in zip(axes, ["기준모델(이동 Z-score)", final_name]):
    cm = confusion_matrix(val_feat["danger"], val_preds[name], labels=[0, 1])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["예측: 정상", "예측: 위험"])
    ax.set_yticks([0, 1], ["실제: 정상", "실제: 위험"])
    ax.set_xlabel("모델 판정 (칸 안의 숫자 = 운행 사이클 수)")
    ax.set_ylabel("실제 상태")
    s = scores(val_feat["danger"], val_preds[name])
    ax.set_title(f"{name}\nPrecision {s['precision']:.2f} · Recall {s['recall']:.2f} · F1 {s['F1']:.2f}", fontsize=10)
fig.suptitle(f"혼동행렬 — {DATASET} validation 엔진 {len(val_units)}대의 모든 사이클 (위험 = 잔여 ≤ {DANGER_RUL}사이클)", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT_FIG}/cls_confusion_matrix.png")
plt.close(fig)

# 임곗값에 따른 Precision/Recall 변화 (임곗값 조정 근거 그림)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(curve["threshold"], curve["precision"], label="Precision (경보 중 진짜 위험 비율)")
ax.plot(curve["threshold"], curve["recall"], label="Recall (진짜 위험 중 잡아낸 비율)")
ax.plot(curve["threshold"], curve["F1"], label="F1", linestyle="--")
ax.axvline(0.5, color="gray", linestyle=":", label="기본 임곗값 0.5")
ax.axvline(prob_th, color="crimson", label=f"선택한 임곗값 {prob_th:.2f}")
ax.axhline(TARGET_RECALL, color="crimson", linestyle=":", linewidth=0.8)
ax.set_xlabel("위험 확률 임곗값 (이 값 이상이면 '위험' 경보)")
ax.set_ylabel("지표 값 (0~1)")
ax.set_title(f"RandomForest 분류 — 임곗값별 성능 ({DATASET} train 엔진 {N_FOLDS}-겹 교차검증)")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(f"{OUT_FIG}/cls_threshold_tradeoff.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 6. 위험 기준(20~60)을 바꿔가며 성능 비교 + 경보 선행시간
# ---------------------------------------------------------------------------
sens = []
for n in (20, 30, 40, 50, 60):
    yt = (train_feat["RUL"] <= n).astype(int)
    yv = (val_feat["RUL"] <= n).astype(int)
    pv = RandomForestClassifier(**RF_PARAMS).fit(X_train, yt).predict_proba(val_feat[feature_cols])[:, 1]
    sens.append({"danger_RUL": n,
                 "RF분류_F1(0.5)": scores(yv, (pv >= 0.5).astype(int))["F1"],
                 "GRU회귀→분류_F1": scores(yv, (align(gru_val, val_feat) <= n).astype(int))["F1"]})
sens = pd.DataFrame(sens)
print("\n[위험 기준별 validation F1]")
print(sens.to_string(index=False))

val_feat["p"] = p_val
val_feat["alarm"] = (p_val >= prob_th).astype(int)
life = train_raw.groupby("unit")["cycle"].max()
lead = []
for u, g in val_feat.groupby("unit"):
    run = g["alarm"].rolling(ALARM_CONSECUTIVE).sum().to_numpy()
    hit = np.where(run >= ALARM_CONSECUTIVE)[0]
    first = int(g["cycle"].iloc[hit[0]]) if len(hit) else None
    lead.append({"unit": int(u), "life": int(life[u]), "first_alarm_cycle": first,
                 "lead_time": (int(life[u]) - first) if first else None})
lead = pd.DataFrame(lead)
print(f"\n[경보 선행시간] 3사이클 연속 경보 기준, 고장 {lead['lead_time'].median():.0f}사이클 전(중앙값)에 첫 경보 "
      f"(범위 {lead['lead_time'].min():.0f}~{lead['lead_time'].max():.0f}, 목표 {DANGER_RUL})")

# ---------------------------------------------------------------------------
# 7. 오탐·미탐 사례를 실제 시계열 위에 표시
# ---------------------------------------------------------------------------
val_feat["FP"] = (val_feat["alarm"] == 1) & (val_feat["danger"] == 0)
val_feat["FN"] = (val_feat["alarm"] == 0) & (val_feat["danger"] == 1)
fp_unit = int(val_feat.groupby("unit")["FP"].sum().idxmax())
fn_unit = int(val_feat.groupby("unit")["FN"].sum().idxmax())
raw_val = corrector.transform(train_raw[train_raw["unit"].isin(val_units)])

# 그래프에 그릴 센서: 06번 도메인 해석에서 RUL 예측에 가장 크게 기여한 센서 1위
# (FD001은 s4 저압터빈 출구 온도, FD004는 s3 고압압축기 출구 온도). 06번 결과가 없으면 s4.
try:
    with open(f"{OUT_METRICS}/domain_interpretation.json", encoding="utf-8") as f:
        _domain = json.load(f)
    PLOT_SENSOR = next(iter(_domain["top_sensors_rf"]))
    _meaning = _domain["sensor_meaning"].get(PLOT_SENSOR, "")
except FileNotFoundError:
    PLOT_SENSOR, _meaning = "s4", ""
PLOT_LABEL = f"{PLOT_SENSOR} {_meaning}".strip()

# 비교용: train 엔진들이 위험 구간에 들어설 때(잔여 = 위험 기준) 그 센서의 평균값
typical_at_danger = tr[tr["RUL"].between(DANGER_RUL - 2, DANGER_RUL + 2)][PLOT_SENSOR].mean()


def plot_case(unit: int, kind: str, filename: str):
    g = val_feat[val_feat["unit"] == unit]
    r = raw_val[raw_val["unit"] == unit].sort_values("cycle")
    danger_start = int(life[unit]) - DANGER_RUL
    mask = g[kind]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(r["cycle"], r[PLOT_SENSOR], color="gray", linewidth=0.8,
             label=f"{PLOT_LABEL} (" + ("운전조건 보정값" if MULTI_REGIME else "원값") + ")")
    ax1.plot(r["cycle"], r[PLOT_SENSOR].rolling(15, min_periods=1).mean(), color="black", label=f"{PLOT_SENSOR} 최근 15사이클 평균")
    ax1.axhline(typical_at_danger, color="purple", linestyle=":", label=f"train 엔진이 위험 진입할 때 평균 ({typical_at_danger:.1f})")
    ax1.axvspan(danger_start, r["cycle"].max(), color="red", alpha=0.08, label=f"실제 위험 구간 (잔여 ≤ {DANGER_RUL})")
    c = g.loc[mask, "cycle"]
    ax1.scatter(c, r.set_index("cycle").loc[c, PLOT_SENSOR], color="orange" if kind == "FP" else "blue", zorder=3, s=18,
                label="오탐 (정상인데 위험 경보)" if kind == "FP" else "미탐 (위험한데 경보 없음)")
    ax1.set_ylabel(f"{PLOT_SENSOR} ({SENSOR_UNIT[PLOT_SENSOR] or '비율'})" + (", 운전조건 보정" if MULTI_REGIME else ""))
    ax1.legend(fontsize=7, loc="upper left")
    ax2.plot(g["cycle"], g["p"], color="teal", label="모델이 본 위험 확률")
    ax2.axhline(prob_th, color="crimson", linestyle="--", label=f"경보 임곗값 {prob_th:.2f}")
    ax2.axvspan(danger_start, r["cycle"].max(), color="red", alpha=0.08)
    ax2.set_ylabel("위험 확률 (0~1)")
    ax2.set_xlabel("운행 사이클 (cycle)")
    ax2.legend(fontsize=7, loc="upper left")
    title = "오탐 사례" if kind == "FP" else "미탐 사례"
    fig.suptitle(f"{title} — {DATASET} validation 엔진 #{unit} (수명 {int(life[unit])}사이클, {title[:2]} {int(mask.sum())}사이클)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/{filename}")
    plt.close(fig)
    return {"unit": unit, "life": int(life[unit]), "count": int(mask.sum()),
            "cycles": [int(x) for x in c], "rul_at_cycles": [int(life[unit]) - int(x) for x in c],
            "sensor": PLOT_SENSOR,
            "sensor_ma15_at_cycles": [round(float(v), 2) for v in r.set_index("cycle")[PLOT_SENSOR].rolling(15, min_periods=1).mean().loc[c]]}


fp_case = plot_case(fp_unit, "FP", "cls_error_case_false_alarm.png")
fn_case = plot_case(fn_unit, "FN", "cls_error_case_missed.png")
print(f"\n[오탐 사례] 엔진 #{fp_case['unit']}: {fp_case['count']}사이클, 그때 실제 잔여 {fp_case['rul_at_cycles']}")
print(f"[미탐 사례] 엔진 #{fn_case['unit']}: {fn_case['count']}사이클, 그때 실제 잔여 {fn_case['rul_at_cycles']}")
print(f"  (비교: train 엔진이 위험 진입할 때 {PLOT_SENSOR} 평균 {typical_at_danger:.2f}, "
      f"오탐 시점 {PLOT_SENSOR} 평균 {fp_case['sensor_ma15_at_cycles']}, 미탐 시점 평균 {fn_case['sensor_ma15_at_cycles']})")

# ---------------------------------------------------------------------------
# 8. 저장
# ---------------------------------------------------------------------------
pd.DataFrame({"unit": val_feat["unit"], "cycle": val_feat["cycle"], "RUL_true": val_feat["RUL"],
              "danger_true": val_feat["danger"], "danger_prob": p_val.round(4), "alarm": val_feat["alarm"],
              "baseline_HI": hi_val["HI"].round(4).to_numpy()}).to_csv(
    f"{OUT_METRICS}/classification_val_predictions.csv", index=False)

with open(f"{OUT_METRICS}/classification_metrics.json", "w", encoding="utf-8") as f:
    json.dump({
        "task": f"고장 임박 이진 분류 (위험 = 잔여 ≤ {DANGER_RUL}사이클)",
        "baseline": {"method": "이동 Z-score 건강지표", "normal_cycles": BASELINE_NORMAL_CYCLES,
                     "smooth_window": BASELINE_SMOOTH, "tau": round(tau, 4)},
        "model": {"method": "RandomForestClassifier", **{k: v for k, v in RF_PARAMS.items() if k != "n_jobs"}},
        "threshold_adjustment": {"default": 0.5, "chosen": prob_th, "rule": f"train 엔진 {N_FOLDS}-겹 교차검증에서 Recall ≥ {TARGET_RECALL} 중 Precision 최대",
                                 "reason": "미탐 1건 비용이 오탐 1건의 약 48배(08번 비용 가정)라 재현율 우선"},
        "comparison": rows,
        "danger_threshold_sensitivity": sens.to_dict(orient="records"),
        "lead_time": {"rule": f"{ALARM_CONSECUTIVE}사이클 연속 경보", "median": float(lead["lead_time"].median()),
                      "per_engine": lead.to_dict(orient="records")},
        "error_cases": {"false_alarm": fp_case, "missed": fn_case,
                        "typical_at_danger_entry": round(float(typical_at_danger), 2)},
    }, f, ensure_ascii=False, indent=2)
print(f"\n저장: {OUT_METRICS}/classification_metrics.json, classification_comparison.csv, "
      f"figures/cls_*.png")
