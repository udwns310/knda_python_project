"""
06_domain_interpretation.py
=============================
목적: "숫자만 잘 맞추는 모델"에서 끝나지 않고, "왜 그렇게 예측하는지"를 설비 도메인
언어로 설명하는 단계입니다 (과정 진행 가이드 8단계 중 "도메인 해석·개선").

1. RF 피처 중요도 상위 센서들이, 참고자료(MathWorks similarity-based RUL)의
   trendability 개념으로 뽑은 "추세성 상위 센서"와 실제로 일치하는지 대조합니다.
   → 두 가지 서로 다른 방법(지도학습 기반 중요도 vs 비지도 상관관계 기반 추세성)이
     같은 센서를 가리킨다면, "모델이 우연이 아니라 진짜 물리적으로 의미있는 신호를
     학습했다"는 근거가 됩니다.
2. C-MAPSS 변수 설명(Saxena et al. 2008, Table 2 — common.py의 SENSOR_MEANING)을 이용해
   상위 센서가 실제로 어떤 물리량인지 사람이 이해할 수 있는 말로 번역합니다.

실행 순서: 03번(RandomForest 모델 저장) 다음
"""

import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from common import (
    load_raw, add_rul_labels, find_constant_columns, SENSOR_COLS_RAW, SENSOR_MEANING, RegimeCorrector,
)

# 폴더 경로는 common.py에서 PC에 상관없이 자동으로 계산됩니다 (0번 섹션 참고).
from common import OUT_METRICS, OUT_FIG, OUT_MODELS, TRAIN_FILE, DATASET

# ---------------------------------------------------------------------------
# 1. RF 피처 중요도 vs EDA 추세성 랭킹 비교
# ---------------------------------------------------------------------------
# 03번에서 저장한 RandomForest의 특성 중요도(전체 특성)를 센서별로 합산합니다.
# 한 센서마다 특성이 4개(현재값·15사이클 평균·표준편차·기울기)라, 네 개를 더한 값이
# "그 센서가 예측에 기여한 정도"입니다. 전체 특성 중요도의 합은 1입니다.
rf_pack = joblib.load(f"{OUT_MODELS}/rf_model.joblib")
importance = pd.Series(rf_pack["model"].feature_importances_, index=rf_pack["feature_cols"])
# "{센서명}_mean15" 같은 피처명에서 원래 센서 이름만 뽑아내기
sensor_importance = (importance.groupby(importance.index.str.extract(r"^(s\d+)")[0].values).sum()
                     .sort_values(ascending=False))

print("=" * 70)
print(f"[RF 모델이 실제로 중요하게 쓴 센서 Top 8 (센서별 특성 4개 합산, {DATASET})]")
print("=" * 70)
for sensor, imp in sensor_importance.head(8).items():
    print(f"  {sensor} ({SENSOR_MEANING.get(sensor,'?')}) : 중요도 {imp:.3f}")

# EDA에서 만든 trendability 랭킹을 다시 계산 (01_eda.py와 동일 로직, 여기서 재사용)
train = add_rul_labels(load_raw(TRAIN_FILE))
const_cols = find_constant_columns(train)
active_sensors = [c for c in SENSOR_COLS_RAW if c not in const_cols]
# 다중 운전조건이면 조건 차이를 뺀 값으로 추세를 봅니다 (FD001은 원래값 그대로)
train = RegimeCorrector().fit(train, active_sensors).transform(train)
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
ax.set_xlabel("RandomForest 특성 중요도 (센서별 특성 4개 합산, 전체 합 = 1)")
ax.set_title(f"RUL 예측에 가장 크게 기여한 센서 Top 8 ({DATASET})")
fig.tight_layout()
fig.savefig(f"{OUT_FIG}/feature_importance_top8.png", dpi=130)
plt.close(fig)
print(f"\n저장: {OUT_FIG}/feature_importance_top8.png")

# ---------------------------------------------------------------------------
# 2. 해석 요약을 json으로 저장 (05·08·09번 그래프 센서 선택과 문서화에서 재사용)
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
