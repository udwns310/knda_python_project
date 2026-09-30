"""
14_plot_alarm_rates.py
======================
11번 모델 비교 결과(model_screening_alarm_rates.csv)를 그림 한 장으로 보여줍니다.
"회귀 모델의 예측 RUL이 30 이하이면 위험 경보"로 썼을 때, 모델마다
  ① 평균 오차(MAE) — 남은 수명을 평균 몇 사이클 틀리는지
  ② 미탐율 — 실제 위험 구간인데 경보를 못 한 비율 (놓치면 고장 → 가장 중요)
  ③ 오탐율 — 실제 정상 구간인데 경보를 울린 비율 (불필요한 점검)
을 나란히 비교합니다. 세 값은 단위가 달라서 한 그래프에 겹치지 않고 옆으로 나란히 그리며,
모델 순서는 세 칸 모두 같습니다 (미탐율이 낮은 순).

실행: python scripts/14_plot_alarm_rates.py   (11번 실행 후, CMAPSS_DATASET으로 FD004/FD001 선택)
결과: outputs/figures/model_alarm_rates.png
"""

import pandas as pd
import matplotlib.pyplot as plt

from common import DATASET, DANGER_RUL, OUT_METRICS, OUT_FIG

# 색: 모델 종류(시계열 딥러닝 / 표 형태 머신러닝)만 구분, 기준 모델은 회색으로 한 발 물러나게.
# 두 색은 dataviz 팔레트 1·2번 (색각이상 구분 검사 통과: ΔE 24.7)
COLOR = {"시계열": "#2a78d6", "표 형태": "#eb6834", "기준": "#8a8985"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

rates = pd.read_csv(f"{OUT_METRICS}/model_screening_alarm_rates.csv")
kinds = pd.read_csv(f"{OUT_METRICS}/model_screening_regression.csv").set_index("model")["type"]
rates["type"] = rates["model"].map(kinds)
rates = rates.sort_values("val_미탐율", ascending=False)  # 그림 위쪽이 미탐율 낮은(좋은) 모델

panels = [
    ("val_MAE", "평균 오차 MAE (cycle)", "{:.1f}", 1),
    ("val_미탐율", "미탐율 (%) — 위험을 놓친 비율", "{:.1f}%", 100),
    ("val_오탐율", "오탐율 (%) — 괜한 경보 비율", "{:.1f}%", 100),
]
fig, axes = plt.subplots(1, 3, figsize=(14, 0.55 * len(rates) + 1.8), sharey=True)
for ax, (col, title, fmt, scale) in zip(axes, panels):
    vals = rates[col] * scale
    ax.barh(rates["model"], vals, color=[COLOR[t] for t in rates["type"]], height=0.62,
            edgecolor="white", linewidth=2)
    for y, v in enumerate(vals):
        ax.text(v, y, " " + fmt.format(v), va="center", fontsize=9, color=TEXT)
    ax.set_title(title, fontsize=11, color=TEXT, loc="left")
    ax.set_xlim(0, vals.max() * 1.22)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("낮을수록 좋음", fontsize=9, color=MUTED)
axes[0].tick_params(axis="y", labelsize=10, labelcolor=TEXT)

handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[k]) for k in COLOR]
fig.legend(handles, ["시계열 딥러닝", "표 형태 머신러닝", "기준 모델 (센서 미사용)"], loc="lower center",
           ncol=3, frameon=False, fontsize=9)
n_val = rates["val_TP"].iloc[0] + rates["val_FN"].iloc[0]
fig.suptitle(f"{DATASET} — 회귀 모델을 '예측 RUL ≤ {DANGER_RUL}이면 위험 경보'로 썼을 때 "
             f"(validation 엔진의 모든 사이클, 실제 위험 {n_val:,}사이클)",
             fontsize=12, color=TEXT, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.06, 1, 0.95))
fig.savefig(f"{OUT_FIG}/model_alarm_rates.png", dpi=140, facecolor="#fcfcfb")
plt.close(fig)
print(f"저장: {OUT_FIG}/model_alarm_rates.png")
