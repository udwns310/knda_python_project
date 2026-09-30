"""
13_threshold_cost.py
====================
[임의 설정값 #11] 위험 기준(예측 RUL ≤ N사이클이면 정비 경보)의 보전 관점 근거:
"정비 준비 기간을 지키면서, 운행 1사이클당 정비 비용이 가장 싼 N은 얼마인가?"

과정 진행 가이드의 평가 규칙 "임곗값을 조정한 경우, 왜 그 값을 선택했는지 보전 관점(점검 비용 vs
고장 누락 위험)에서 설명"에 대한 정량 근거입니다. 정비 준비 기간과 비용 비율은 C-MAPSS 터보팬 엔진으로
정비 일정을 연구한 공개 논문의 가정을 그대로 가져왔습니다.

[문헌 근거]
  (1) de Pater, Reijns, Mitici (2022), "Alarm-based predictive maintenance scheduling for aircraft
      engines with imperfect Remaining Useful Life prognostics", Reliability Engineering & System
      Safety 221, 108341 — C-MAPSS 전 서브셋 사용
      · 추가 정비 준비에 최소 7일 필요, 항공기별 정비 슬롯은 10~20일마다, 정비 계획은 매주 갱신
      · 항공기는 하루 1회 비행 (→ 1일 = 1사이클)
      · 비용: 계획 정비 10,000 / 엔진 고장 50,000 (1 : 5)
      · 유전 알고리즘으로 최적화한 경보 기준: 예측 RUL 49사이클 (안전계수 0.44 함께 사용)
  (2) Lee & Mitici (2023), "Deep reinforcement learning for predictive aircraft maintenance using
      probabilistic Remaining-Useful-Life prognostics", RESS 230, 108908 — FD002 사용
      · 엔진 교체 준비에 며칠이 걸려 30사이클 단위로 정비를 계획
      · 비용: 계획 교체 1 / 비계획 교체 2 (1 : 2)

[정비 정책 — (1)의 운영 방식을 그대로 흉내 냄]
  1) 예측 RUL이 처음으로 N 이하가 되면 경보.
  2) 다음 주간 계획 회의(0~6일 뒤)에서 정비를 잡고, 준비에 7일이 걸린 뒤,
     그 이후 처음 돌아오는 정비 슬롯(10~20일 간격)에 정비.
  3) 정비 전에 고장 나면 비계획 정비, 아니면 계획 정비(남은 수명만큼 운행을 덜 함).
  주간 회의 시점·슬롯 간격·슬롯 위치는 매번 달라서 무작위로 200번 반복한 평균을 씁니다.
  → 경보부터 정비까지 걸리는 시간 = 7~32사이클 (평균 약 17사이클)

[비교 지표] 운행 1사이클당 정비 비용 = 정비 비용 합 ÷ 실제로 운행한 사이클 합
  - N이 너무 크면: 멀쩡한 엔진을 일찍 정비 → 운행 사이클(분모)이 줄어 비쌈
  - N이 너무 작으면: 정비 전에 고장 → 비계획 비용(분자)이 커짐

데이터: validation 엔진(학습에 안 쓴 엔진)의 모든 사이클 예측값 (03번 RF·04번 GRU가 저장한 csv).
공식 test 엔진은 고장 전에 기록이 끊겨 있어 "언제 고장 났는지"를 알 수 없으므로 쓸 수 없습니다.

실행: python scripts/13_threshold_cost.py   (기본 FD004, FD001은 CMAPSS_DATASET=FD001 — 몇 초)
결과: outputs/<데이터셋>/metrics/threshold_cost.csv, threshold_cost_summary.csv, outputs/<데이터셋>/figures/threshold_cost.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import load_raw, DANGER_RUL, DATASET, TRAIN_FILE, OUT_METRICS, OUT_FIG

# 비용 비율 (계획 정비 = 1 기준). 1:5가 문헌 (1)의 주 설정, 1:2는 문헌 (2)로 보수적으로 확인.
COST_SETTINGS = {"1:5 (de Pater 2022)": 5.0, "1:2 (Lee & Mitici 2023)": 2.0}
PREP_DAYS = 7              # 문헌 (1): 추가 정비 준비 최소 7일
SLOT_INTERVAL = (10, 20)   # 문헌 (1): 정비 슬롯 10~20일 간격
REVIEW_EVERY = 7           # 문헌 (1): 정비 계획 매주 갱신
N_SIM = 200                # 무작위 반복 횟수
N_GRID = list(range(10, 81, 5))
MODELS = {"GRU (04번)": "gru", "RandomForest (03번)": "rf"}
OLD_DANGER_RUL = 30        # 이 분석 전에 쓰던 기준 (과제 문서 예시값) — 비교용으로 함께 기록

life = load_raw(TRAIN_FILE).groupby("unit")["cycle"].max()


def lead_time(rng) -> int:
    """경보 → 정비까지 걸리는 사이클 (주간 회의 대기 + 준비 7일 + 다음 슬롯까지 대기)."""
    wait_review = rng.integers(0, REVIEW_EVERY)
    interval = rng.integers(SLOT_INTERVAL[0], SLOT_INTERVAL[1] + 1)
    wait_slot = rng.integers(0, interval)          # 준비가 끝난 뒤 다음 슬롯까지
    return int(wait_review + PREP_DAYS + wait_slot)


def simulate(alarm_cycle: dict, n_sim: int, cost_fail: float, seed: int = 42) -> dict:
    """alarm_cycle: 엔진 → 경보 사이클(None이면 경보 없음)."""
    rng = np.random.default_rng(seed)
    total_cost = total_ops = failures = wasted = 0.0
    for _ in range(n_sim):
        for u, a in alarm_cycle.items():
            L = int(life[u])
            m = None if a is None else a + lead_time(rng)
            if m is None or m >= L:
                total_cost += cost_fail
                total_ops += L
                failures += 1
            else:
                total_cost += 1.0
                total_ops += m
                wasted += L - m
    n = n_sim * len(alarm_cycle)
    return {"cost_per_1000_cycles": 1000 * total_cost / total_ops, "failure_rate": failures / n,
            "avg_wasted_cycles": wasted / max(1, n - failures)}


rows = []
for name, key in MODELS.items():
    pred = pd.read_csv(f"{OUT_METRICS}/{key}_val_predictions.csv").sort_values(["unit", "cycle"])
    for n in N_GRID:
        alarm = {u: (int(g.loc[g["RUL_pred"] <= n, "cycle"].iloc[0]) if (g["RUL_pred"] <= n).any() else None)
                 for u, g in pred.groupby("unit")}
        for cname, cf in COST_SETTINGS.items():
            rows.append({"model": name, "cost_ratio": cname, "N": n, **simulate(alarm, N_SIM, cf)})
res = pd.DataFrame(rows)
res.to_csv(f"{OUT_METRICS}/threshold_cost.csv", index=False, encoding="utf-8-sig")

# 기준값 요약: 최적 N, 그 비용, 현재 기준(DANGER_RUL)과 예전 기준(30)의 결과, 최적의 5% 이내 범위
summary = []
for (m, c), g in res.groupby(["model", "cost_ratio"]):
    best = g.loc[g["cost_per_1000_cycles"].idxmin()]
    near = g[g["cost_per_1000_cycles"] <= best["cost_per_1000_cycles"] * 1.05]["N"]
    row = {"model": m, "cost_ratio": c, "best_N": int(best["N"]),
           "best_cost": round(best["cost_per_1000_cycles"], 2),
           "best_failure_rate": round(best["failure_rate"], 3),
           "best_wasted_cycles": round(best["avg_wasted_cycles"], 1),
           "within_5pct_N": f"{near.min()}~{near.max()}"}
    for n in sorted({DANGER_RUL, OLD_DANGER_RUL}):
        at = g.loc[g["N"] == n].iloc[0]
        row[f"N{n}_failure_rate"] = round(at["failure_rate"], 3)
        row[f"N{n}_extra_cost_pct"] = round((at["cost_per_1000_cycles"] / best["cost_per_1000_cycles"] - 1) * 100, 1)
    summary.append(row)
summary = pd.DataFrame(summary)
summary.to_csv(f"{OUT_METRICS}/threshold_cost_summary.csv", index=False, encoding="utf-8-sig")

rng = np.random.default_rng(0)
leads = np.array([lead_time(rng) for _ in range(10000)])
print(f"[{DATASET}] validation 엔진 {len(pred['unit'].unique())}대, 경보→정비 {leads.min()}~{leads.max()}사이클 "
      f"(평균 {leads.mean():.1f}, 90% 이내 {np.percentile(leads, 90):.0f})")
print(summary.to_string(index=False))

# 비용(비율 두 가지)과 정비 전 고장 비율은 단위가 달라 한 그래프에 겹치지 않고 나란히 그림
COLORS = {"GRU (04번)": "#2a78d6", "RandomForest (03번)": "#eb6834"}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
for ax, (cname, _) in zip(axes[:2], COST_SETTINGS.items()):
    for name in MODELS:
        g = res[(res["model"] == name) & (res["cost_ratio"] == cname)]
        ax.plot(g["N"], g["cost_per_1000_cycles"], marker="o", markersize=4, linewidth=2,
                color=COLORS[name], label=name)
        b = g.loc[g["cost_per_1000_cycles"].idxmin()]
        ax.scatter([b["N"]], [b["cost_per_1000_cycles"]], s=90, color=COLORS[name], edgecolor="white",
                   linewidth=2, zorder=5)
        ax.annotate(f"최저 N={int(b['N'])}", (b["N"], b["cost_per_1000_cycles"]), textcoords="offset points",
                    xytext=(0, -16) if name.startswith("GRU") else (0, 10), ha="center", fontsize=8)
    ax.set_title(f"정비 비용 — 계획:비계획 = {cname}", loc="left", fontsize=10)
    ax.set_ylabel("운행 1,000사이클당 정비 비용\n(계획 정비 1회 = 1)")
for name in MODELS:
    g = res[(res["model"] == name) & (res["cost_ratio"] == list(COST_SETTINGS)[0])]
    axes[2].plot(g["N"], g["failure_rate"] * 100, marker="o", markersize=4, linewidth=2,
                 color=COLORS[name], label=name)
axes[2].set_title("정비 전에 고장 나는 엔진 비율 (비용 비율과 무관)", loc="left", fontsize=10)
axes[2].set_ylabel("정비 전 고장 비율 (%)")
for ax in axes:
    ax.axvline(DANGER_RUL, color="#8a8985", linestyle="--", linewidth=1, label=f"현재 기준 N = {DANGER_RUL}")
    ax.set_xlabel("위험 기준 N (예측 RUL ≤ N 이면 정비 경보, cycle)")
    ax.grid(color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
fig.suptitle(f"{DATASET} — 경보 후 주간 계획·준비 7일·정비 슬롯(10~20일) 대기를 거쳐 정비할 때 "
             f"(de Pater et al. 2022 운영 조건, validation 엔진 {pred['unit'].nunique()}대)",
             fontsize=11, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(f"{OUT_FIG}/threshold_cost.png", dpi=130)
plt.close(fig)
print(f"\n저장: {OUT_METRICS}/threshold_cost*.csv, {OUT_FIG}/threshold_cost.png")
