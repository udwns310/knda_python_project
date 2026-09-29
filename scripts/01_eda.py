"""
01_eda.py
=========
목적: 본격적인 모델링 들어가기 전에 "데이터가 우리가 생각한 대로 생겼는지" 눈으로
확인하는 단계입니다. 부트캠프 8단계 흐름의 "데이터 이해 / 전처리·기초 분석 / 시계열 분석"
에 해당합니다.

이 스크립트가 하는 일 (실행하면 outputs/figures/ 에 그림 3장이 저장됩니다):
  1. 상수 센서 제거가 제대로 됐는지 확인
  2. 센서별 "추세성(trendability)" 순위 — 어떤 센서가 고장에 가까워질수록 뚜렷하게
     변하는지 확인 (참고자료 1: MathWorks similarity-based RUL 문서의 핵심 아이디어를
     그대로 가져온 부분입니다 — 자세한 설명은 아래 함수 주석 참고)
  3. 대표 엔진 몇 개를 골라서 "센서값이 시간에 따라 어떻게 변하는지" + "RUL 라벨이
     어떻게 생겼는지(평평하다가 뚝 떨어지는 모양)" 를 그림으로 확인
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    load_raw, find_constant_columns, add_rul_labels, SENSOR_COLS_RAW,
)

# 폴더 경로는 common.py에서 PC에 상관없이 자동으로 계산됩니다 (0번 섹션 참고).
from common import DATA, OUT_FIG

train = load_raw(f"{DATA}/train_FD001.txt")
train = add_rul_labels(train)

const_cols = find_constant_columns(train)
active_sensors = [c for c in SENSOR_COLS_RAW if c not in const_cols]

print("=" * 70)
print("[1] 상수 센서 제거 결과")
print("=" * 70)
print(f"제거된 컬럼 ({len(const_cols)}개):", const_cols)
print(f"남은 센서 ({len(active_sensors)}개):", active_sensors)

# ---------------------------------------------------------------------------
# 2. 추세성(Trendability) 랭킹
# ---------------------------------------------------------------------------
# 참고자료 활용: MathWorks "Similarity-Based Remaining Useful Life Estimation" 문서
# (https://www.mathworks.com/help/predmaint/ug/similarity-based-remaining-useful-life-estimation.html)
# 에서는 21개 센서 중 "고장에 가까워질수록 값이 뚜렷하게, 일관되게 변하는" 센서만
# 골라서(trendability 분석) 8개를 추려 health indicator를 만듭니다.
#
# 우리 프로젝트에서는 이 아이디어를 다음과 같이 활용했습니다:
#   - 원문처럼 K-means로 운전조건(regime)을 먼저 나누는 단계는 생략했습니다.
#     이유: 그 문서가 다루는 데이터는 운전조건이 6가지로 섞여 있는 케이스이고,
#     우리가 최종적으로 쓰기로 한 FD001은 운전조건이 1가지뿐이라(문제정의서에도
#     명시) 그 단계 자체가 필요 없습니다 — 이미 "규칙적인 조건" 위에서 측정된
#     데이터라 그대로 추세 분석을 해도 됩니다.
#   - "센서별로 사이클과의 상관관계(|corr|)를 구해서 절대값이 큰 순으로 정렬"하는
#     방식으로 우리 데이터에 맞게 단순화한 trendability 분석을 직접 구현했습니다.
#     상관관계가 높다 = 사이클이 흐를수록(=고장이 가까워질수록) 그 센서 값이
#     일관된 방향으로 움직인다는 뜻이라, 원문의 "선형 추세 기울기로 랭킹" 아이디어와
#     본질적으로 같은 목적을 갖습니다.
#   - 이 랭킹 결과는 04_domain_interpretation 단계에서 "왜 이 센서가 중요한가"를
#     설명할 때 다시 사용합니다.
def trendability_ranking(df: pd.DataFrame, sensor_cols: list) -> pd.Series:
    corrs = {}
    for col in sensor_cols:
        corrs[col] = df[[col, "cycle"]].corr().iloc[0, 1]
    s = pd.Series(corrs).abs().sort_values(ascending=False)
    return s


rank = trendability_ranking(train, active_sensors)
print("\n" + "=" * 70)
print("[2] 센서별 추세성(사이클과의 상관계수 절대값) 랭킹 (상위 8개)")
print("=" * 70)
print(rank.head(8))

# ---------------------------------------------------------------------------
# 3. 대표 엔진 시각화
# ---------------------------------------------------------------------------
top_sensors = rank.head(4).index.tolist()
sample_units = [1, 25, 60]  # 수명이 짧은/중간/긴 엔진 섞어서 하나씩

fig, axes = plt.subplots(len(top_sensors), 1, figsize=(9, 11), sharex=False)
for ax, col in zip(axes, top_sensors):
    for uid in sample_units:
        g = train[train["unit"] == uid]
        ax.plot(g["cycle"], g[col], label=f"engine #{uid}", alpha=0.8)
    ax.set_title(f"sensor {col} (trendability rank corr={rank[col]:.2f})")
    ax.set_xlabel("cycle")
    ax.legend(fontsize=8)
fig.suptitle("추세성 상위 센서들의 실제 시계열 (엔진별)")
fig.tight_layout()
fig.savefig(f"{OUT_FIG}/sensor_trends.png", dpi=130)
print(f"\n저장: {OUT_FIG}/sensor_trends.png")

# RUL 라벨 모양 확인 (평평하다가 뚝 떨어지는지)
fig2, ax2 = plt.subplots(figsize=(8, 5))
for uid in sample_units:
    g = train[train["unit"] == uid]
    ax2.plot(g["cycle"], g["RUL"], label=f"engine #{uid} (수명={g['cycle'].max()})")
ax2.set_title("Piecewise-linear RUL 라벨 모양 (클리핑값=125)")
ax2.set_xlabel("cycle")
ax2.set_ylabel("RUL label")
ax2.legend()
fig2.tight_layout()
fig2.savefig(f"{OUT_FIG}/rul_label_shape.png", dpi=130)
print(f"저장: {OUT_FIG}/rul_label_shape.png")

# 엔진 수명 분포
fig3, ax3 = plt.subplots(figsize=(7, 4))
life = train.groupby("unit")["cycle"].max()
ax3.hist(life, bins=20, edgecolor="white")
ax3.axvline(125, color="red", linestyle="--", label="RUL clip=125")
ax3.set_title(f"train 엔진 100대의 수명 분포 (최소={life.min()}, 최대={life.max()})")
ax3.set_xlabel("수명 (cycle)")
ax3.legend()
fig3.tight_layout()
fig3.savefig(f"{OUT_FIG}/engine_life_distribution.png", dpi=130)
print(f"저장: {OUT_FIG}/engine_life_distribution.png")

print("\nEDA 완료.")
