"""
16_gru_error_cases.py
=====================
GRU(예측 남은 수명 ≤ 40이면 경보)의 오탐·미탐 사례를 09번(RF 분류)과 같은 방식으로 그립니다.
  - 오탐 사례: validation 엔진 중 헛경보 사이클이 가장 많은 엔진
  - 미탐 사례: validation 엔진 중 놓친 사이클이 가장 많은 엔진
09번 그림과 같은 센서(s3, 운전조건 보정값)·같은 위험 구간 표시를 쓰고, 아래 칸만
"위험 확률" 대신 "GRU 예측 남은 수명 vs 실제 남은 수명"을 그립니다.

실행: python scripts/16_gru_error_cases.py
결과: outputs/FD004/figures/gru_error_case_false_alarm.png, gru_error_case_missed.png,
      outputs/FD004/metrics/gru_error_cases.json
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import (
    load_raw, find_constant_columns, add_rul_labels, split_engines, SENSOR_COLS_RAW, display_unit, display_values,
    DANGER_RUL, OUT_FIG, OUT_METRICS, TRAIN_FILE, RegimeCorrector, MULTI_REGIME, DATASET,
)

plt.rcParams["figure.dpi"] = 110
SENSOR = "s3"

train_raw = add_rul_labels(load_raw(TRAIN_FILE))
train_units, val_units = split_engines(train_raw)
const_cols = find_constant_columns(train_raw[train_raw["unit"].isin(train_units)])
sensors = [c for c in SENSOR_COLS_RAW if c not in const_cols]
corrector = RegimeCorrector().fit(train_raw[train_raw["unit"].isin(train_units)], sensors)
tr = corrector.transform(train_raw[train_raw["unit"].isin(train_units)])
raw_val = corrector.transform(train_raw[train_raw["unit"].isin(val_units)])
life = train_raw.groupby("unit")["cycle"].max()
typical_at_danger = tr[tr["RUL"].between(DANGER_RUL - 2, DANGER_RUL + 2)][SENSOR].mean()

gru = pd.read_csv(f"{OUT_METRICS}/gru_val_predictions.csv").sort_values(["unit", "cycle"])
gru["danger"] = gru["RUL_true"] <= DANGER_RUL
gru["alarm"] = gru["RUL_pred"] <= DANGER_RUL
gru["FP"] = gru["alarm"] & ~gru["danger"]
gru["FN"] = ~gru["alarm"] & gru["danger"]
fp_unit = int(gru.groupby("unit")["FP"].sum().idxmax())
fn_unit = int(gru.groupby("unit")["FN"].sum().idxmax())


def plot_case(unit: int, kind: str, filename: str) -> dict:
    g = gru[gru["unit"] == unit]
    r = raw_val[raw_val["unit"] == unit].sort_values("cycle")
    danger_start = int(life[unit]) - DANGER_RUL
    mask = g[kind]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(r["cycle"], display_values(SENSOR, r[SENSOR]), color="gray", linewidth=0.8, label="s3 T30 · HPC(고압압축기) outlet 온도 (운전조건 보정값)")
    ax1.plot(r["cycle"], display_values(SENSOR, r[SENSOR].rolling(15, min_periods=1).mean()), color="black", label="s3 최근 15사이클 평균")
    ax1.axhline(display_values(SENSOR, typical_at_danger), color="purple", linestyle=":", label=f"train 엔진이 위험 진입할 때 평균 ({display_values(SENSOR, typical_at_danger):.1f})")
    ax1.axvspan(danger_start, r["cycle"].max(), color="red", alpha=0.08, label=f"실제 위험 구간 (잔여 ≤ {DANGER_RUL})")
    c = g.loc[mask, "cycle"]
    ax1.scatter(c, display_values(SENSOR, r.set_index("cycle").loc[c, SENSOR]), color="orange" if kind == "FP" else "blue", zorder=3, s=18,
                label="오탐 (정상인데 위험 경보)" if kind == "FP" else "미탐 (위험한데 경보 없음)")
    ax1.set_ylabel(f"{SENSOR} ({display_unit(SENSOR) or '비율'}), 운전조건 보정")
    ax1.legend(fontsize=7, loc="upper left")
    ax2.plot(g["cycle"], g["RUL_pred"], color="teal", label="GRU 예측 남은 수명")
    ax2.plot(g["cycle"], g["RUL_true"], color="gray", linestyle="--", linewidth=1, label="실제 남은 수명 (125에서 자름)")
    ax2.axhline(DANGER_RUL, color="crimson", linestyle="--", label=f"경보 기준 예측 ≤ {DANGER_RUL}")
    ax2.axvspan(danger_start, r["cycle"].max(), color="red", alpha=0.08)
    ax2.set_ylabel("남은 수명 (사이클)")
    ax2.set_xlabel("운행 사이클 (cycle)")
    ax2.legend(fontsize=7, loc="upper right")
    title = "오탐 사례" if kind == "FP" else "미탐 사례"
    fig.suptitle(f"GRU {title} — {DATASET} validation 엔진 #{unit} (수명 {int(life[unit])}사이클, {title[:2]} {int(mask.sum())}사이클)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/{filename}")
    plt.close(fig)
    ma = r.set_index("cycle")[SENSOR].rolling(15, min_periods=1).mean()
    return {"unit": unit, "life": int(life[unit]), "count": int(mask.sum()),
            "cycles": [int(x) for x in c], "rul_at_cycles": [int(life[unit]) - int(x) for x in c],
            "sensor_ma15_at_cycles": [round(float(v), 2) for v in ma.loc[c]],
            "sensor_ma15_at_danger_start": round(float(ma.loc[danger_start]), 2)}


fp_case = plot_case(fp_unit, "FP", "gru_error_case_false_alarm.png")
fn_case = plot_case(fn_unit, "FN", "gru_error_case_missed.png")
with open(f"{OUT_METRICS}/gru_error_cases.json", "w", encoding="utf-8") as f:
    json.dump({"false_alarm": fp_case, "missed": fn_case, "typical_s3_at_danger": round(float(typical_at_danger), 2)},
              f, ensure_ascii=False, indent=2)
print(f"[GRU 오탐 사례] 엔진 #{fp_case['unit']}: {fp_case['count']}사이클, 실제 잔여 {min(fp_case['rul_at_cycles'])}~{max(fp_case['rul_at_cycles'])}")
print(f"[GRU 미탐 사례] 엔진 #{fn_case['unit']}: {fn_case['count']}사이클, 실제 잔여 {min(fn_case['rul_at_cycles'])}~{max(fn_case['rul_at_cycles'])}, "
      f"위험 진입 때 s3 평균 {fn_case['sensor_ma15_at_danger_start']} (train 평균 {typical_at_danger:.1f})")
