"""
13_threshold_cost.py
====================
[임의 설정값 #11] 위험 기준(잔여 ≤ N사이클)의 보전 관점 근거: "정비 리드타임을 지키면서
운행 1사이클당 정비 비용이 가장 싼 N은 얼마인가?"

과정 진행 가이드의 평가 규칙 "임곗값을 조정한 경우, 왜 그 값을 선택했는지 보전 관점(점검 비용 vs
고장 누락 위험)에서 설명"에 대한 정량 근거를 만드는 스크립트입니다.

정비 정책 (모델이 매 사이클 예측하는 RUL을 그대로 사용):
  1) 예측 RUL이 처음으로 N 이하가 되는 사이클에 경보를 울린다.
  2) 경보 후 부품 주문·인력 배치에 L사이클(리드타임)이 걸려, 경보 L사이클 뒤에 정비한다.
  3) 정비 전에 엔진이 고장 나면 → 비계획 정비 (긴 정지 + 응급 수리)
     정비가 고장보다 먼저면     → 계획 정비 (짧은 정지 + 일반 수리), 대신 남은 수명만큼 운행을 덜 함
  경보가 끝까지 안 울리면 고장까지 운행 → 비계획 정비.

비교 지표 = 운행 1사이클당 정비 비용 = (모든 엔진의 정비 비용 합) ÷ (모든 엔진이 실제로 운행한 사이클 합)
  - N이 너무 크면: 고장은 막지만 멀쩡한 엔진을 일찍 정비해 운행 사이클(분모)이 줄어 비싸짐
  - N이 너무 작으면: 경보 뒤 리드타임 안에 고장 나는 엔진이 늘어 비계획 비용(분자)이 커짐
  → 그 사이의 가장 싼 N이 "리드타임을 고려한 최적 위험 기준"입니다.

가정값 (C-MAPSS에는 비용·리드타임 정보가 없어 08번 대시보드 비용 시뮬레이션과 같은 가정을 씀):
  계획 정비 1건   = 4시간 정지 × 1,000만원/시간 + 수리 800만원            = 4,800만원
  비계획 정비 1건 = 24시간 정지 × 1,000만원/시간 + 응급 수리(800만원 × 3) = 26,400만원
  리드타임 L      = 실제 값을 몰라 0·5·10·15·20·30사이클로 바꿔 가며 확인

데이터: validation 엔진(학습에 안 쓴 엔진)의 모든 사이클 예측값 (02·03·04번이 저장한 csv).
공식 test 엔진은 고장 전에 기록이 끊겨 있어 "언제 고장 났는지"를 알 수 없으므로 이 시뮬레이션에 쓸 수 없습니다.

실행: python scripts/13_threshold_cost.py   (몇 초, CMAPSS_DATASET으로 FD004도 가능)
결과: outputs/metrics/threshold_cost.csv, outputs/figures/threshold_cost.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import load_raw, DANGER_RUL, DATASET, TRAIN_FILE, OUT_METRICS, OUT_FIG

COST_PLANNED = 4 * 1000 + 800            # 만원
COST_UNPLANNED = 24 * 1000 + 800 * 3     # 만원
LEAD_TIMES = [0, 5, 10, 15, 20, 30]      # 사이클
# 위험 기준 후보 (예측 RUL ≤ N 이면 경보). 80을 넘으면 엔진 가동 초반부터 경보가 울려 의미가 없어
# 5~80만 봄 (RUL 라벨 상한이 125라 모델 예측도 초반에는 100~125 근처에 머묾).
N_GRID = list(range(5, 81, 5))
MODELS = {"LSTM (04번)": "lstm", "RandomForest (03번)": "rf", "베이스라인 (02번)": "baseline"}

life = load_raw(TRAIN_FILE).groupby("unit")["cycle"].max()


def simulate(pred: pd.DataFrame, n: int, lead: int) -> dict:
    """pred: unit, cycle, RUL_pred (validation 엔진의 모든 사이클)."""
    cost = operated = failures = wasted = 0
    for u, g in pred.groupby("unit"):
        g = g.sort_values("cycle")
        L = int(life[u])
        alarm = g.loc[g["RUL_pred"] <= n, "cycle"]
        maint = int(alarm.iloc[0]) + lead if len(alarm) else None
        if maint is None or maint >= L:            # 정비 전에 고장
            cost += COST_UNPLANNED
            operated += L
            failures += 1
        else:                                      # 계획 정비 (남은 수명 L - maint 만큼 버림)
            cost += COST_PLANNED
            operated += maint
            wasted += L - maint
    n_eng = pred["unit"].nunique()
    return {"cost_per_cycle": cost / operated, "failures": failures, "failure_rate": failures / n_eng,
            "avg_wasted_cycles": wasted / max(1, n_eng - failures)}


rows = []
for name, key in MODELS.items():
    pred = pd.read_csv(f"{OUT_METRICS}/{key}_val_predictions.csv")
    for lead in LEAD_TIMES:
        for n in N_GRID:
            rows.append({"model": name, "lead_time": lead, "N": n, **simulate(pred, n, lead)})
res = pd.DataFrame(rows)

# 참고선: ① 정비 없이 고장까지 운행 ② 고장 시점을 미리 안다고 가정한 이상적인 계획 정비(버리는 수명 0)
val_life = life.loc[pd.read_csv(f"{OUT_METRICS}/lstm_val_predictions.csv")["unit"].unique()]
run_to_failure = COST_UNPLANNED * len(val_life) / val_life.sum()
ideal = COST_PLANNED * len(val_life) / val_life.sum()
res["cost_per_cycle"] = res["cost_per_cycle"].round(2)
res.to_csv(f"{OUT_METRICS}/threshold_cost.csv", index=False, encoding="utf-8-sig")

best = res.loc[res.groupby(["model", "lead_time"])["cost_per_cycle"].idxmin()]
at30 = res[res["N"] == DANGER_RUL].set_index(["model", "lead_time"])["cost_per_cycle"]
best = best.assign(cost_at_N30=[at30[(m, l)] for m, l in zip(best["model"], best["lead_time"])])
best["N30_extra_cost_pct"] = ((best["cost_at_N30"] / best["cost_per_cycle"] - 1) * 100).round(1)

print(f"[{DATASET}] validation 엔진 {len(val_life)}대, 운행 1사이클당 비용 (만원/사이클)")
print(f"  참고: 고장까지 운행 {run_to_failure:.1f} / 고장 시점을 미리 아는 이상적 정비 {ideal:.1f}\n")
print(best[["model", "lead_time", "N", "cost_per_cycle", "failures", "avg_wasted_cycles",
            "cost_at_N30", "N30_extra_cost_pct"]].to_string(index=False))

# 그림: LSTM 기준, 리드타임별 비용 곡선과 최솟값, 그리고 현재 기준(30)
fig, ax = plt.subplots(figsize=(9, 5))
lstm = res[res["model"] == "LSTM (04번)"]
for lead in LEAD_TIMES:
    c = lstm[lstm["lead_time"] == lead]
    line, = ax.plot(c["N"], c["cost_per_cycle"], marker="o", markersize=3, label=f"리드타임 {lead}사이클")
    b = c.loc[c["cost_per_cycle"].idxmin()]
    ax.scatter([b["N"]], [b["cost_per_cycle"]], color=line.get_color(), s=70, zorder=5, edgecolor="black")
ax.axvline(DANGER_RUL, color="crimson", linestyle="--", label=f"현재 위험 기준 N = {DANGER_RUL}")
ax.axhline(run_to_failure, color="gray", linestyle=":", label=f"고장까지 운행 ({run_to_failure:.0f})")
ax.axhline(ideal, color="green", linestyle=":", label=f"이상적 정비 ({ideal:.0f})")
ax.set_ylim(0, run_to_failure * 1.15)
ax.set_xlabel("위험 기준 N (예측 RUL ≤ N 이면 경보, cycle)")
ax.set_ylabel("운행 1사이클당 정비 비용 (만원/cycle)")
ax.set_title(f"리드타임별 위험 기준에 따른 정비 비용 — LSTM, {DATASET} validation {len(val_life)}대 (큰 점 = 최솟값)")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(f"{OUT_FIG}/threshold_cost.png", dpi=130)
plt.close(fig)
print(f"\n저장: {OUT_METRICS}/threshold_cost.csv, {OUT_FIG}/threshold_cost.png")
