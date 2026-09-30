"""
14_plot_alarm_rates.py
======================
11번 모델 비교 결과(model_screening_alarm_rates.csv)를 그림 한 장으로 보여줍니다.
"회귀 모델의 예측 RUL이 위험 기준(40) 이하이면 위험 경보"로 썼을 때, 모델마다
  ① 평균 오차(MAE) — 남은 수명을 평균 몇 사이클 틀리는지
  ② 미탐율 — 실제 위험 구간인데 경보를 못 한 비율 (놓치면 고장 → 가장 중요)
  ③ 오탐율 — 실제 정상 구간인데 경보를 울린 비율 (불필요한 점검)
을 나란히 비교합니다. 세 값은 단위가 달라서 한 그래프에 겹치지 않고 옆으로 나란히 그리며,
모델 순서는 세 칸 모두 같습니다 (미탐율이 낮은 순).

두 번째 그림은 같은 값을 산점도로 다시 그립니다: 가로축 = MAE, 세로축 = 미탐율 / 오탐율 / 헛경보비율,
점 하나가 모델 하나이고 직선은 최소제곱 회귀선입니다. "평균 오차가 작은 모델일수록 경보도 정확한가"를 봅니다.
  · 헛경보비율 = 울린 경보 중 틀린 비율 FP/(TP+FP) — 경보를 받은 정비팀 입장의 오탐
  · 기준 모델(센서 미사용)은 MAE가 다른 모델의 2~4배라 한 축에 넣으면 나머지가 한쪽에 몰려, 산점도에서는 빼고 제목 아래에 값만 적음

실행: python scripts/14_plot_alarm_rates.py   (11번 실행 후, CMAPSS_DATASET으로 FD004/FD001 선택)
결과: outputs/<데이터셋>/figures/model_alarm_rates.png, model_alarm_vs_mae.png
"""

import numpy as np
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


def place_labels(ax, xs, ys, labels, fontsize=9):
    """점 옆에 모델 이름을 붙이되, 다른 이름·점과 겹치지 않는 자리를 후보 위치 중에서 차례로 고른다."""
    fig = ax.figure
    fig.canvas.draw()
    to_px = ax.transData.transform
    k = fig.dpi / 72                      # 포인트 → 픽셀
    pts_px = [to_px((a, b)) for a, b in zip(xs, ys)]
    boxes = [(px - 7 * k, py - 7 * k, px + 7 * k, py + 7 * k) for px, py in pts_px]  # 점 자리
    cands = [(8, 0, "left"), (-8, 0, "right"), (0, 11, "center"), (0, -11, "center"),
             (8, 11, "left"), (8, -11, "left"), (-8, 11, "right"), (-8, -11, "right"),
             (8, 22, "left"), (8, -22, "left"), (-8, 22, "right"), (-8, -22, "right"),
             (8, 33, "left"), (8, -33, "left")]
    for (px, py), label, xv, yv in sorted(zip(pts_px, labels, xs, ys), key=lambda t: t[0][1]):
        w, h = len(label) * fontsize * 0.62 * k, fontsize * 1.3 * k
        for dx, dy, ha in cands:
            cx, cy = px + dx * k, py + dy * k
            x0 = cx if ha == "left" else cx - w if ha == "right" else cx - w / 2
            box = (x0, cy - h / 2, x0 + w, cy + h / 2)
            if not any(box[0] < o[2] and o[0] < box[2] and box[1] < o[3] and o[1] < box[3] for o in boxes):
                break
        boxes.append(box)
        far = abs(dy) > 11
        ax.annotate(label, (xv, yv), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=fontsize, color=TEXT, zorder=4,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=5) if far else None)


# ── 두 번째 그림: MAE vs 경보 오류율 산점도 + 회귀선 ─────────────────────────
SHORT = {"GRU (04번)": "GRU", "LSTM (MathWorks 원형)": "LSTM", "MLP (신경망 64-32)": "MLP", "1D-CNN (30사이클 창)": "1D-CNN",
         "HistGradientBoosting": "HistGB", "ExtraTrees": "ExtraTrees", "RandomForest (03번)": "RF",
         "Ridge (선형회귀)": "Ridge"}
base = rates[rates["type"] == "기준"].iloc[0]
pts = rates[rates["type"] != "기준"].copy()
pts["label"] = pts["model"].map(SHORT).fillna(pts["model"])
x = pts["val_MAE"].to_numpy()

scatter_panels = [
    ("val_미탐율", "미탐율 — 실제 위험인데 경보 없음", "FN / (TP+FN)"),
    ("val_오탐율", "오탐율 — 실제 정상인데 경보", "FP / (FP+TN)"),
    ("val_헛경보비율", "헛경보비율 — 울린 경보 중 틀린 비율", "FP / (TP+FP)"),
]
fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
for ax, (col, title, formula) in zip(axes, scatter_panels):
    y = pts[col].to_numpy() * 100
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    xs = np.linspace(x.min() - 0.4, x.max() + 0.4, 50)
    ax.plot(xs, slope * xs + intercept, color=TEXT, linewidth=2, zorder=1)
    ax.scatter(x, y, s=90, c=[COLOR[t] for t in pts["type"]], edgecolors="white", linewidths=2, zorder=3)
    ax.set_title(f"{title}\n{formula}   |   회귀선 y = {slope:.2f}·MAE {intercept:+.1f},  상관계수 r = {r:.2f}",
                 fontsize=10.5, color=TEXT, loc="left")
    ax.set_xlabel("평균 오차 MAE (cycle, validation) — 낮을수록 좋음", fontsize=9, color=MUTED)
    ax.set_ylabel("%  (낮을수록 좋음)", fontsize=9, color=MUTED)
    pad = (y.max() - y.min()) * 0.12 or 1
    ax.set_xlim(x.min() - 0.8, x.max() + 1.6)
    ax.set_ylim(y.min() - pad, y.max() + pad)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    place_labels(ax, x, y, pts["label"].tolist())

handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor=COLOR[k],
                      markeredgecolor="white") for k in ("시계열", "표 형태")]
fig.legend(handles, ["시계열 딥러닝", "표 형태 머신러닝"], loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.suptitle(f"{DATASET} — 회귀 모델의 평균 오차(MAE)와 경보 오류율 (예측 RUL ≤ {DANGER_RUL}이면 경보, "
             f"validation 엔진의 모든 사이클)\n"
             f"점 하나 = 모델 하나, 검은 선 = 최소제곱 회귀선.  참고: 기준 모델(센서 미사용)은 MAE {base['val_MAE']:.1f}, 미탐율 {base['val_미탐율']*100:.1f}%, "
             f"오탐율 {base['val_오탐율']*100:.1f}%, 헛경보비율 {base['val_헛경보비율']*100:.1f}% — 축 범위 밖이라 생략",
             fontsize=11, color=TEXT, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.05, 1, 0.93))
fig.savefig(f"{OUT_FIG}/model_alarm_vs_mae.png", dpi=140, facecolor="#fcfcfb")
plt.close(fig)
print(f"저장: {OUT_FIG}/model_alarm_vs_mae.png")
