"""
05_evaluate.py
================
선택 과제(RUL 회귀)의 평가 · 오류 분석 단계입니다.

과정 진행 가이드의 평가 규칙은 "오탐 1건 이상, 미탐 1건 이상을 실제 시계열 그래프 위에 표시하고
왜 틀렸는지 서술"입니다. 이 규칙은 필수 과제인 이진 분류에 대해 09번에서 그대로 수행하고,
여기서는 같은 생각을 숫자를 맞히는 회귀 문제에 맞게 옮겨 적용합니다:
  - "위험한 오류" (분류의 미탐에 대응): 모델이 실제보다 RUL을 더 크게 예측
    → "아직 여유 있다"고 착각해서 정비 시점을 놓칠 위험
  - "보수적 오류" (분류의 오탐에 대응): 모델이 실제보다 RUL을 더 작게 예측
    → 아직 괜찮은데 "곧 고장"이라고 너무 일찍 경고

이 스크립트는:
  1. 베이스라인/RF/GRU 세 모델의 성능을 한 표로 정리
  2. 성능 비교 막대그래프 저장
  3. GRU(최종 회귀 모델) 기준으로 공식 test에서 가장 크게 틀린 "위험한 오류" 1건, "보수적 오류" 1건을
     찾아서 시계열 그래프로 그림. 그래프 안에는 확인된 사실(관측 길이, 주요 센서가 위험 수준에 닿았는지)만
     적고, 원인 해석은 그래프를 보고 docs/FD004_RESULTS.md에 따로 씀
실행 순서: 02·03·04번 다음
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from common import (
    load_raw, add_rul_labels, split_engines, Normalizer, RUL_CLIP_VALUE, RegimeCorrector,
    MULTI_REGIME, DATASET, SENSOR_UNIT, DANGER_RUL, sensor_label,
)
from seq_models import load_seq_model, predict_recurrent

# 폴더 경로는 common.py에서 PC에 상관없이 자동으로 계산됩니다 (0번 섹션 참고).
from common import OUT_METRICS, OUT_FIG, OUT_MODELS, TRAIN_FILE, TEST_FILE, RUL_FILE

# ---------------------------------------------------------------------------
# 1. 세 모델 성능 비교표
# ---------------------------------------------------------------------------
rows = []
for name in ["baseline", "rf", "gru"]:
    with open(f"{OUT_METRICS}/{name}_metrics.json", encoding="utf-8") as f:
        m = json.load(f)
    rows.append({
        "model": m["model"],
        "val_MAE": m["validation"]["MAE"], "val_RMSE": m["validation"]["RMSE"], "val_NASA": m["validation"]["NASA_score"],
        "test_MAE": m["official_test"]["MAE"], "test_RMSE": m["official_test"]["RMSE"], "test_NASA": m["official_test"]["NASA_score"],
    })
comparison = pd.DataFrame(rows)
comparison.to_csv(f"{OUT_METRICS}/model_comparison.csv", index=False)
print("=" * 70)
print("모델 성능 비교 (validation / 공식 test)")
print("=" * 70)
print(comparison.to_string(index=False))

improve_vs_baseline = (1 - comparison.loc[comparison.model == "gru_seq2seq", "test_MAE"].values[0] /
                        comparison.loc[comparison.model == "naive_baseline", "test_MAE"].values[0]) * 100
print(f"\nGRU는 나이브 베이스라인 대비 공식 test MAE를 {improve_vs_baseline:.1f}% 개선했습니다.")

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(comparison))
width = 0.35
ax.bar(x - width / 2, comparison["val_MAE"], width, label="validation MAE")
ax.bar(x + width / 2, comparison["test_MAE"], width, label="공식 test MAE")
ax.set_xticks(x)
ax.set_xticklabels(["나이브 베이스라인", "Random Forest", "GRU"])
ax.set_ylabel("MAE (사이클)")
ax.set_title(f"모델별 RUL 예측 오차(MAE) 비교 — 낮을수록 좋음 ({DATASET})")
ax.legend()
for i, v in enumerate(comparison["test_MAE"]):
    ax.text(i + width / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT_FIG}/model_comparison_mae.png", dpi=130)
plt.close(fig)
print(f"저장: {OUT_FIG}/model_comparison_mae.png")

# ---------------------------------------------------------------------------
# 2. GRU 모델 다시 불러오기 (재학습 없이 04번에서 저장한 가중치만 로드)
# ---------------------------------------------------------------------------
model, active_sensors = load_seq_model(f"{OUT_MODELS}/gru_model.pt")

# 정규화기는 랜덤성이 없는(같은 seed=42로 항상 같은 train/val 분할) 결정론적 계산이라
# 동일하게 재계산해도 04번 스크립트와 완전히 같은 결과가 나옵니다.
train_raw = load_raw(TRAIN_FILE)
train_raw = add_rul_labels(train_raw)
train_units, val_units = split_engines(train_raw)
norm = Normalizer().fit(train_raw[train_raw["unit"].isin(train_units)], active_sensors)
# 그래프용: 운전조건 차이를 뺀 센서값 (FD001은 원래값 그대로)
corrector = RegimeCorrector().fit(train_raw[train_raw["unit"].isin(train_units)], active_sensors)

# 그래프에 그릴 센서 = 03번 RF가 가장 중요하게 본 센서 2개 (특성 중요도를 센서별로 합산).
# 데이터셋마다 다름 — FD004: s3·s17, FD001: s4·s9
_imp = pd.read_csv(f"{OUT_METRICS}/rf_feature_importance_top20.csv", index_col=0)["importance"]
PLOT_SENSORS = list(_imp.groupby(_imp.index.str.split("_").str[0]).sum().sort_values(ascending=False).index[:2])

# 비교 기준: train 엔진들이 위험 구간(잔여 = DANGER_RUL)에 들어설 때의 센서 평균 (운전조건 보정값)
_tr_view = corrector.transform(train_raw[train_raw["unit"].isin(train_units)])
DANGER_LEVEL = {s: float(_tr_view.loc[_tr_view["RUL"] == DANGER_RUL, s].mean()) for s in PLOT_SENSORS}


# ---------------------------------------------------------------------------
# 3. test 세트에서 "위험한 오류" / "보수적 오류" 사례 탐색
# ---------------------------------------------------------------------------
test_pred_df = pd.read_csv(f"{OUT_METRICS}/gru_test_predictions.csv")
test_pred_df["residual"] = test_pred_df["RUL_pred"] - test_pred_df["RUL_true"]

danger_case = test_pred_df.loc[test_pred_df["residual"].idxmax()]  # 예측이 실제보다 훨씬 큼
conservative_case = test_pred_df.loc[test_pred_df["residual"].idxmin()]  # 예측이 실제보다 훨씬 작음

print("\n" + "=" * 70)
print("[오류 사례 분석 — GRU 모델 기준]")
print("=" * 70)
print(f"위험한 오류 사례: 엔진 #{int(danger_case.unit)} "
      f"(실제 RUL={danger_case.RUL_true:.0f}, 예측={danger_case.RUL_pred:.1f}, "
      f"오차=+{danger_case.residual:.1f} → 실제보다 여유있다고 과대평가)")
print(f"보수적 오류 사례: 엔진 #{int(conservative_case.unit)} "
      f"(실제 RUL={conservative_case.RUL_true:.0f}, 예측={conservative_case.RUL_pred:.1f}, "
      f"오차={conservative_case.residual:.1f} → 실제보다 위험하다고 과소평가)")

test_raw = load_raw(TEST_FILE)
test_norm = norm.transform(test_raw)
rul_true_file = pd.read_csv(RUL_FILE, header=None, names=["RUL"])
rul_true_file["unit"] = np.arange(1, len(rul_true_file) + 1)
TEST_LEN_MEDIAN = test_raw.groupby("unit")["cycle"].max().median()


def plot_case(uid, title, filename, note):
    g = test_norm[test_norm["unit"] == uid].sort_values("cycle")
    seq = torch.tensor(g[active_sensors].values, dtype=torch.float32)
    pred_curve = predict_recurrent(model, seq)
    rul_final_true = rul_true_file.loc[rul_true_file.unit == uid, "RUL"].values[0]
    cycles = g["cycle"].values
    # 각 관측 시점에서의 '진짜' RUL: 마지막 시점의 정답(RUL 정답 파일)에, 마지막
    # 시점까지 남은 사이클 수를 거슬러 더해서 역산 (piecewise 클리핑 동일 적용)
    true_curve = np.minimum(rul_final_true + (cycles.max() - cycles), RUL_CLIP_VALUE)

    # 위 두 칸: RF 중요도 상위 센서 2개 (단위가 달라 한 축에 겹치지 않고 칸을 나눔),
    #          점선 = train 엔진들이 위험 구간에 들어설 때의 평균, 선 = 최근 15사이클 평균
    # 아래 칸: 실제 RUL vs GRU 예측 RUL
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1.4]})
    raw_g = corrector.transform(test_raw[test_raw["unit"] == uid].sort_values("cycle"))
    suffix = " (운전조건 보정)" if MULTI_REGIME else ""
    facts = []
    for ax, s in zip(axes[:2], PLOT_SENSORS):
        ma = raw_g[s].rolling(15, min_periods=1).mean()
        ax.plot(raw_g["cycle"], raw_g[s], color="#c9c8c4", linewidth=0.8, label="관측값")
        ax.plot(raw_g["cycle"], ma, color="#2a78d6", linewidth=2, label="최근 15사이클 평균")
        ax.axhline(DANGER_LEVEL[s], color="#eb6834", linestyle="--", linewidth=1.2,
                   label=f"train 엔진이 잔여 {DANGER_RUL}일 때 평균")
        ax.set_ylabel(f"{s} ({SENSOR_UNIT[s]})" if SENSOR_UNIT[s] else s)
        ax.set_title(f"{sensor_label(s)}{suffix}", fontsize=10, loc="left")
        ax.legend(fontsize=8, loc="upper left")
        facts.append(f"{s} 마지막 15사이클 평균 {ma.iloc[-1]:.1f} (위험 진입 평균 {DANGER_LEVEL[s]:.1f})")
    axes[0].set_title(f"엔진 #{uid} ({DATASET} test) — {sensor_label(PLOT_SENSORS[0])}{suffix}", fontsize=10, loc="left")

    axes[2].plot(cycles, true_curve, label=f"실제 RUL (학습 라벨처럼 {RUL_CLIP_VALUE}에서 자름)", color="black", linewidth=2)
    axes[2].plot(cycles, pred_curve, label="GRU 예측 RUL", color="#eb6834", linestyle="--", linewidth=2)
    axes[2].scatter([cycles[-1]], [rul_final_true], color="black", zorder=5,
                    label=f"마지막 시점 실제 RUL (자르지 않은 값) {rul_final_true:.0f}")
    axes[2].scatter([cycles[-1]], [pred_curve[-1]], color="#eb6834", zorder=5)
    axes[2].set_xlabel("운행 사이클 (cycle)")
    axes[2].set_ylabel("RUL (남은 사이클, cycle)")
    axes[2].legend(fontsize=9)
    axes[2].set_title(title, fontsize=10, loc="left")
    n_obs = len(cycles)
    fig.suptitle(f"{note}\n관측 {n_obs}사이클 (test 엔진 중앙값 {TEST_LEN_MEDIAN:.0f}) · " + " · ".join(facts),
                 fontsize=9, wrap=True)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/{filename}", dpi=130)
    plt.close(fig)
    print(f"저장: {OUT_FIG}/{filename}")


plot_case(
    int(danger_case.unit),
    f"위험한 오류 사례 (엔진 #{int(danger_case.unit)}): 실제보다 RUL을 과대예측",
    "error_case_danger.png",
    f"실제 {danger_case.RUL_true:.0f} → 예측 {danger_case.RUL_pred:.1f}: 실제보다 여유 있다고 봄 (정비 시점을 놓칠 위험)",
)
plot_case(
    int(conservative_case.unit),
    f"보수적 오류 사례 (엔진 #{int(conservative_case.unit)}): 실제보다 RUL을 과소예측",
    "error_case_conservative.png",
    f"실제 {conservative_case.RUL_true:.0f} → 예측 {conservative_case.RUL_pred:.1f}: 실제보다 위험하다고 봄 (너무 이른 정비)",
)

# ---------------------------------------------------------------------------
# 4. 오류 사례 요약을 문서화용 JSON으로도 저장 (docs 작성 시 재사용)
# ---------------------------------------------------------------------------
error_summary = {
    "danger_case": {
        "unit": int(danger_case.unit), "RUL_true": float(danger_case.RUL_true),
        "RUL_pred": float(danger_case.RUL_pred), "residual": float(danger_case.residual),
    },
    "conservative_case": {
        "unit": int(conservative_case.unit), "RUL_true": float(conservative_case.RUL_true),
        "RUL_pred": float(conservative_case.RUL_pred), "residual": float(conservative_case.residual),
    },
}
with open(f"{OUT_METRICS}/error_case_summary.json", "w", encoding="utf-8") as f:
    json.dump(error_summary, f, ensure_ascii=False, indent=2)

print("\n평가 및 오류 분석 완료.")
