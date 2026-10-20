"""
24_threshold_by_cycles_per_day.py
=================================
"하루 비행 횟수가 다르면 비용 기준 최적 위험 기준 N은 얼마인가?" (13번 시뮬레이션을 하루 비행 횟수 k로 확장)

13번(위험 기준 40의 근거)은 논문 가정 "항공기는 하루 1회 비행(1일 = 1사이클)"을 그대로 썼습니다.
같은 운영 조건(주간 회의 0~6일, 정비 준비 7일, 정비 슬롯 10~20일 간격)이라도 하루에 k번 비행하면
경보부터 정비까지 걸리는 시간이 k배 사이클로 늘어납니다. 그 외(비용 비율 1:5·1:2, 무작위 200회 반복, 시드 42,
GRU validation 예측)는 13번과 같고, 시간 단위만 사이클 = (일 수 × k)로 바꿉니다.

k = 1 → 13번 결과와 같아야 합니다(검증용: 최적 N=40).
국내 LCC 실제 값은 23번이 추정한 하루 약 4.6~6.1회입니다 (outputs/kr_public/kr_cycles_per_day.csv).

한계
  - validation 49대 엔진의 시뮬레이션이며 비용은 가정입니다. 실제 항공사 값이 아닙니다.
  - 예측 남은 수명은 125에서 잘려 있어(라벨 상한) N이 100을 넘으면 모델이 구분하기 어려운 구간입니다.
    그래서 N이 크게 나오는 시나리오(k ≥ 5)는 "사이클이 아니라 일(day) 단위로 말해야 한다"는 근거로 쓰는 편이 안전합니다.
  - 하루 비행 횟수는 시나리오 가정이며, 23번의 추정(보유 대수 기준 평균)과 같은 한계가 있습니다.

실행: python scripts/24_threshold_by_cycles_per_day.py   (몇 분 걸릴 수 있음)
결과: outputs/kr_public/threshold_by_cycles_per_day.csv (전체 격자), threshold_by_cycles_per_day_summary.csv, threshold_by_cycles_per_day.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import load_raw, PROJECT_ROOT, DANGER_RUL, TRAIN_FILE, OUT_METRICS

OUT = PROJECT_ROOT / "outputs" / "kr_public"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["figure.dpi"] = 110

COST_SETTINGS = {"1:5": 5.0, "1:2": 2.0}   # 13번과 동일 (de Pater 2022, Lee & Mitici 2023)
PREP_DAYS, SLOT_INTERVAL, REVIEW_EVERY = 7, (10, 20), 7
N_SIM, SEED = 200, 42
K_LIST = [1, 2, 3, 4, 5, 6]
N_GRID = list(range(10, 126, 5))           # 라벨 상한 125까지

life = load_raw(TRAIN_FILE).groupby("unit")["cycle"].max()
pred = pd.read_csv(f"{OUT_METRICS}/gru_val_predictions.csv").sort_values(["unit", "cycle"])
alarms = {n: {u: (int(g.loc[g["RUL_pred"] <= n, "cycle"].iloc[0]) if (g["RUL_pred"] <= n).any() else None)
              for u, g in pred.groupby("unit")} for n in N_GRID}


def lead_time(rng, k: int) -> int:
    """경보 → 정비까지 사이클: (주간 회의 대기 + 준비 7일 + 슬롯 대기) 일 수 × 하루 비행 횟수 k."""
    wait_review = rng.integers(0, REVIEW_EVERY)
    interval = rng.integers(SLOT_INTERVAL[0], SLOT_INTERVAL[1] + 1)
    wait_slot = rng.integers(0, interval)
    return int((wait_review + PREP_DAYS + wait_slot) * k)


def simulate(alarm: dict, cost_fail: float, k: int) -> dict:
    rng = np.random.default_rng(SEED)
    total_cost = total_ops = failures = wasted = 0.0
    for _ in range(N_SIM):
        for u, a in alarm.items():
            L = int(life[u])
            m = None if a is None else a + lead_time(rng, k)
            if m is None or m >= L:
                total_cost += cost_fail
                total_ops += L
                failures += 1
            else:
                total_cost += 1.0
                total_ops += m
                wasted += L - m
    n = N_SIM * len(alarm)
    return {"cost_per_1000_cycles": 1000 * total_cost / total_ops, "failure_rate": failures / n,
            "avg_wasted_cycles": wasted / max(1, n - failures)}


rows = []
for k in K_LIST:
    for cname, cf in COST_SETTINGS.items():
        for n in N_GRID:
            rows.append({"cycles_per_day": k, "cost_ratio": cname, "N": n, **simulate(alarms[n], cf, k)})
grid = pd.DataFrame(rows)
grid.to_csv(OUT / "threshold_by_cycles_per_day.csv", index=False, encoding="utf-8-sig")

summary = []
for (k, c), g in grid.groupby(["cycles_per_day", "cost_ratio"]):
    best = g.loc[g["cost_per_1000_cycles"].idxmin()]
    at40 = g[g["N"] == DANGER_RUL].iloc[0]
    summary.append({"cycles_per_day": k, "cost_ratio": c, "best_N": int(best["N"]), "best_cost": round(best["cost_per_1000_cycles"], 2),
                    "best_failure_rate": round(best["failure_rate"], 3), "best_avg_wasted_cycles": round(best["avg_wasted_cycles"], 1),
                    f"N{DANGER_RUL}_cost": round(at40["cost_per_1000_cycles"], 2), f"N{DANGER_RUL}_failure_rate": round(at40["failure_rate"], 3),
                    f"N{DANGER_RUL}_days_at_this_k": round(DANGER_RUL / k, 1)})
sm = pd.DataFrame(summary)
sm.to_csv(OUT / "threshold_by_cycles_per_day_summary.csv", index=False, encoding="utf-8-sig")
print(sm.to_string(index=False))
chk = sm[(sm["cycles_per_day"] == 1)]["best_N"].tolist()
print("\n검증: 하루 1회에서 최적 N =", chk, "(13번 결과 40과 같아야 함)")

# ---------------------------------------------------------------------------
# 그림: (왼쪽) 비용 곡선 (1:5), (오른쪽) 하루 비행 횟수별 최적 N
# ---------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
cm = plt.get_cmap("Blues")
for i, k in enumerate([1, 3, 5]):
    g = grid[(grid["cycles_per_day"] == k) & (grid["cost_ratio"] == "1:5")]
    a1.plot(g["N"], g["cost_per_1000_cycles"].clip(upper=25), color=cm(0.45 + 0.25 * i), marker="o", markersize=3, label=f"하루 {k}회")
a1.axvline(DANGER_RUL, color="crimson", linestyle="--", linewidth=1, label=f"현재 기준 {DANGER_RUL}")
a1.set_xlabel("위험 기준 N (예측 남은 수명 ≤ N이면 경보)")
a1.set_ylabel("운행 1,000사이클당 정비 비용 (계획 정비 1회 = 1, 비용 1:5)")
a1.set_ylim(0, 25)
a1.set_title("하루 비행 횟수별 비용 곡선", fontsize=10)
a1.legend(fontsize=8)
for c, col, mk in (("1:5", "#05507d", "o"), ("1:2", "#00a5e5", "s")):
    g = sm[sm["cost_ratio"] == c]
    a2.plot(g["cycles_per_day"], g["best_N"], color=col, marker=mk, label=f"비용 {c} 최적 N")
a2.axhline(DANGER_RUL, color="crimson", linestyle="--", linewidth=1, label=f"현재 기준 {DANGER_RUL}")
a2.axhline(125, color="gray", linestyle=":", linewidth=1)
a2.text(1.05, 127, "남은 수명 라벨 상한 125", fontsize=8, color="gray")
try:
    kr = pd.read_csv(PROJECT_ROOT / "outputs" / "kr_public" / "kr_cycles_per_day.csv")
    lc = kr[(kr["구분"] == "LCC") & (kr["항공사"] != "에어프레미아")]["하루_비행수_대당"]
    a2.axvspan(lc.min(), lc.max(), color="#05507d", alpha=0.12, label=f"국내 LCC 추정 {lc.min():.1f}~{lc.max():.1f}회")
except FileNotFoundError:
    pass
a2.set_xlabel("항공기 1대당 하루 비행 횟수")
a2.set_ylabel("비용 최저 위험 기준 N (사이클)")
a2.set_ylim(0, 140)
a2.set_title("하루 비행 횟수별 최적 위험 기준", fontsize=10)
a2.legend(fontsize=8, loc="lower right")
fig.suptitle("같은 정비 준비 기간도 하루에 여러 번 비행하면 사이클로는 몇 배 — 위험 기준은 운항 패턴에 따라 달라진다", fontsize=10.5)
fig.text(0.01, 0.01, "GRU validation 49대 시뮬레이션, 비용은 논문 가정값. 하루 비행 횟수는 시나리오(국내 LCC 구간은 23번의 공공데이터 추정).", fontsize=7, color="gray")
fig.tight_layout(rect=(0, 0.03, 1, 0.94))
fig.savefig(OUT / "threshold_by_cycles_per_day.png")
plt.close(fig)
print("저장:", OUT / "threshold_by_cycles_per_day.png")
