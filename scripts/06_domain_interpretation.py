"""
06_domain_interpretation.py
=============================
목적: "숫자만 잘 맞추는 모델"에서 끝나지 않고, "왜 그렇게 예측하는지"를 설비 도메인
언어로 설명하는 단계입니다 (부트캠프 8단계 중 "도메인 해석·개선").

1. RF 피처 중요도 상위 센서들이, 참고자료(MathWorks similarity-based RUL)의
   trendability 개념으로 뽑은 "추세성 상위 센서"와 실제로 일치하는지 대조합니다.
   → 두 가지 서로 다른 방법(지도학습 기반 중요도 vs 비지도 상관관계 기반 추세성)이
     같은 센서를 가리킨다면, "모델이 우연이 아니라 진짜 물리적으로 의미있는 신호를
     학습했다"는 근거가 됩니다.
2. C-MAPSS 공식 변수 설명(NASA 문서 기준)을 이용해 상위 센서가 실제로 어떤 물리량인지
   사람이 이해할 수 있는 말로 번역합니다.
"""

import sys, json
import pandas as pd
import matplotlib.pyplot as plt

# 폴더 경로는 common.py에서 PC에 상관없이 자동으로 계산됩니다 (0번 섹션 참고).
from common import DATA, OUT_METRICS, OUT_FIG

# NASA C-MAPSS 데이터 설명서 기준 센서 이름 (물리적 의미) — 공개된 데이터 설명 문서에
# 나온 표준 명칭입니다. 발표 때 "s4가 뭔데요?"라는 질문에 바로 답할 수 있게 정리.
SENSOR_MEANING = {
    "s1": "Fan inlet 온도", "s2": "LPC(저압압축기) outlet 온도", "s3": "HPC(고압압축기) outlet 온도",
    "s4": "LPT(저압터빈) outlet 온도", "s5": "Fan inlet 압력", "s6": "bypass-duct 압력",
    "s7": "HPC outlet 압력", "s8": "물리적 팬 속도(N1)", "s9": "물리적 코어 속도(N2)",
    "s10": "engine pressure ratio", "s11": "HPC outlet 정압", "s12": "연료-압력 비율",
    "s13": "보정된 팬 속도", "s14": "보정된 코어 속도", "s15": "연료 유량비",
    "s16": "bleed enthalpy", "s17": "demanded 팬 속도", "s18": "demanded 보정 팬 속도",
    "s19": "HPT(고압터빈) coolant bleed", "s20": "LPT coolant bleed", "s21": "LPT coolant bleed(2)",
}

# ---------------------------------------------------------------------------
# 1. RF 피처 중요도 vs EDA 추세성 랭킹 비교
# ---------------------------------------------------------------------------
importance = pd.read_csv(f"{OUT_METRICS}/rf_feature_importance_top20.csv", index_col=0)
importance.columns = ["importance"]

# "{센서명}_mean15" 같은 피처명에서 원래 센서 이름만 뽑아내기
importance["sensor"] = importance.index.str.extract(r"(s\d+)")[0].values
sensor_importance = importance.groupby("sensor")["importance"].sum().sort_values(ascending=False)

print("=" * 70)
print("[RF 모델이 실제로 중요하게 쓴 센서 Top 8 (피처 종류 합산)]")
print("=" * 70)
for sensor, imp in sensor_importance.head(8).items():
    print(f"  {sensor} ({SENSOR_MEANING.get(sensor,'?')}) : 중요도 {imp:.3f}")

# EDA에서 만든 trendability 랭킹을 다시 계산 (01_eda.py와 동일 로직, 여기서 재사용)
from common import load_raw, add_rul_labels, find_constant_columns, SENSOR_COLS_RAW
train = add_rul_labels(load_raw(f"{DATA}/train_FD001.txt"))
const_cols = find_constant_columns(train)
active_sensors = [c for c in SENSOR_COLS_RAW if c not in const_cols]
trend_corr = {c: abs(train[[c, "cycle"]].corr().iloc[0, 1]) for c in active_sensors}
trend_rank = pd.Series(trend_corr).sort_values(ascending=False)

top_rf = set(sensor_importance.head(8).index)
top_trend = set(trend_rank.head(8).index)
overlap = top_rf & top_trend
print(f"\nRF 중요도 Top8 ∩ 추세성(trendability) Top8 겹치는 센서: {sorted(overlap)}")
print(f"→ {len(overlap)}/8개가 겹칩니다. 두 개의 독립적인 방법(지도학습 vs 상관관계)이")
print("  같은 센서를 '중요하다'고 가리킨다는 것은, 모델이 노이즈가 아니라 실제")
print("  열화(degradation)와 관련된 물리적 신호를 학습했다는 근거가 됩니다.")

fig, ax = plt.subplots(figsize=(8, 5))
top8 = sensor_importance.head(8)
labels = [f"{s}\n({SENSOR_MEANING.get(s,'')})" for s in top8.index]
ax.barh(labels[::-1], top8.values[::-1], color="tab:blue")
ax.set_xlabel("RF 피처 중요도 (합산)")
ax.set_title("RUL 예측에 가장 크게 기여한 센서 Top 8")
fig.tight_layout()
fig.savefig(f"{OUT_FIG}/feature_importance_top8.png", dpi=130)
print(f"\n저장: {OUT_FIG}/feature_importance_top8.png")

# ---------------------------------------------------------------------------
# 2. 해석 요약을 json/md로 저장 (문서화 단계에서 재사용)
# ---------------------------------------------------------------------------
summary = {
    "top_sensors_rf": {s: float(v) for s, v in sensor_importance.head(8).items()},
    "top_sensors_trendability": {s: float(v) for s, v in trend_rank.head(8).items()},
    "overlap": sorted(overlap),
    "sensor_meaning": SENSOR_MEANING,
}
with open(f"{OUT_METRICS}/domain_interpretation.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"저장: {OUT_METRICS}/domain_interpretation.json")
