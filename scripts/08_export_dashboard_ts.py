"""
08_export_dashboard_ts.py
=========================
목적: 5조 웹 대시보드(https://github.com/posco-knda/5_Sentinel_dashboard)가 실제로 읽는
데이터 파일 `src/data/mock.ts`를 파이썬 파이프라인 결과로부터 자동 생성합니다.

07번이 만든 JSON(engines_summary.json, engine_timeseries/*.json)과 04·05·06번의 결과를
읽어서, 대시보드 화면이 기대하는 모양(TypeScript 상수/함수)으로 바꿔 씁니다.
출력: outputs/<데이터셋>/dashboard_data/mock.ts  →  대시보드 레포의 src/data/mock.ts 로 복사

실행 순서: 01 → ... → 07 → 08 (torch 없이도 실행됩니다. 06·07번 결과 파일만 있으면 됨)
대시보드는 과제 제출물이 아닌 발표·시연용이며, 설계 근거는 대시보드 레포의 docs/DATA_MAPPING.md에 있습니다.

RUL 90% 구간 · 생존곡선 계산 방식 (2026-09-29 수정)
---------------------------------------------------
처음 연동 때 쓴 값은 두 가지 문제가 있어 아래 [임의 설정값 #9] 방식으로 다시 계산합니다.
1) 신뢰구간의 위/아래가 뒤집혀 있었음 — 예전 [예측 − 18.6, 예측 + 25.6]. 검증셋에서 실제 RUL은
   예측보다 "최대 25.6 작고, 최대 18.6 큰" 분포라 방향이 반대였고, 고장이 더 빨리 올 수 있는
   위험 쪽 폭을 작게 보여주고 있었음.
2) 생존곡선이 모델 예측을 반영하지 않았음 — 엔진 나이만 보고 train 엔진 수명 분포로 그려서,
   예측 RUL 15.6인 엔진에 "앞으로 40사이클 더 버틸 확률 84%"가 같이 표시되는 모순이 있었음.
"""

import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

from common import (
    load_raw, RUL_CLIP_VALUE, RISK_THRESHOLDS, SENSOR_UNIT, RegimeCorrector, MULTI_REGIME,
    OUT_METRICS, OUT_DASH, TRAIN_FILE, DATASET,
)

# ---------------------------------------------------------------------------
# [임의 설정값 #8] 대시보드에 보여줄 엔진 9대 · 대표 엔진 (규칙으로 자동 선정)
# ---------------------------------------------------------------------------
# 대시보드 화면(카드 9개)은 원래 TCM 주제의 "스탠드 여러 개"를 보여주던 자리라서,
# test 엔진을 다 넣지 않고 발표 데모용으로 9대를 고릅니다. 등급별로 골고루 보이도록:
#   위험(RED) 5대   : 대표 엔진 + 예측 RUL이 가장 낮은 4대
#   주의(YELLOW) 2대: 주의 등급 중 위험 경계에 가장 가까운(예측 RUL이 낮은) 2대
#   정상(GREEN) 2대 : 정상 중 가장 낮은 1대(곧 주의로 넘어갈 엔진) + 가장 여유 있는 1대
# 대표 엔진(모니터링 화면의 메인 차트): 위험 엔진 중 마지막 25사이클 창 "안에서" 정상→주의→위험
# 두 번의 경계 교차가 모두 보이는 엔진 가운데 예측 RUL이 가장 크게 떨어지는 엔진 (열화 과정을 한
# 화면에 보여주기 위함). 조건을 만족하는 엔진이 없으면 창 안에서 가장 크게 떨어지는 위험 엔진.
# 창 길이 25는 기존 대시보드의 시간 슬라이더(0~24, 25칸) 구조에 맞춘 값입니다.
# (처음에는 FD001 결과를 보고 사람이 고른 고정 목록이었으나, FD004 중심으로 바꾸고 위험 기준이
#  바뀔 때마다 다시 골라야 해서 2026-09-30 규칙 기반으로 바꿈)
N_RED, N_YELLOW = 4, 2
FEATURED_WINDOW = 25
MODEL_NAME = "GRU"  # 04번 최종 회귀 모델 (대시보드 문구가 dataMeta.model로 읽음)
DATASET_DESC = {"FD001": "운전조건 1종 · 고장모드 1종", "FD002": "운전조건 6종 · 고장모드 1종",
                "FD003": "운전조건 1종 · 고장모드 2종", "FD004": "운전조건 6종 · 고장모드 2종"}

# ---------------------------------------------------------------------------
# [임의 설정값 #9] RUL 신뢰구간 · 생존곡선 계산 방식
# ---------------------------------------------------------------------------
# GRU는 "RUL = 15.6" 같은 숫자 하나만 내놓습니다. 그 숫자가 얼마나 믿을 만한지를
# 보여주기 위해, 검증셋(학습에 안 쓴 엔진 — FD004 49대, FD001 20대)에서 모델이 "비슷한 값을 예측했던"
# 순간들을 모아 그때 실제 RUL이 어땠는지를 봅니다.
#   예) 예측이 15.6이면, 검증셋에서 예측이 5.6~25.6 사이였던 순간들을 모아서
#       "그때 실제로는 몇 사이클 남아 있었나"의 분포를 구함
#   - 신뢰구간 = 그 분포의 5% ~ 95% 구간 (= 90% 구간)
#   - 생존곡선 S(x) = 그 순간들 중 "실제 RUL이 x보다 컸던" 비율
#                   = "앞으로 x 사이클 더 가동해도 아직 고장나지 않을 확률"
# 이렇게 하면 신뢰구간·생존곡선·예측값이 모두 같은 근거에서 나오므로 서로 모순되지 않습니다.
#
# NEIGHBOR_RADIUS = 10: 너무 좁으면(예: 2) 모이는 사례가 적어 분포가 들쭉날쭉하고,
#   너무 넓으면(예: 30) RUL 10과 40처럼 전혀 다른 상황이 섞입니다. 10이면 위험·주의
#   등급 구간 폭(각 40 cycle)의 4분의 1이라 등급이 크게 섞이지 않으면서 사례가 수백 개 모입니다.
# MIN_NEIGHBORS = 30: 반경 안 사례가 30개보다 적으면 가장 가까운 30개를 씁니다
#   (통계에서 분포 모양을 볼 때 흔히 쓰는 최소 표본 수 관례).
# 실제 RUL은 125로 자르지 않은 원래 값을 씁니다 (공식 test 정답도 자르지 않은 값이라서).
NEIGHBOR_RADIUS = 10
MIN_NEIGHBORS = 30
INTERVAL_QUANTILES = (0.05, 0.95)

# 생존곡선의 음영(불확실성 밴드): 검증 엔진(FD004 49대)을 "엔진 단위로" 다시 뽑는 부트스트랩을
# 1000번 반복해서 5%~95% 범위를 구합니다. 같은 엔진의 연속된 사이클은 서로 거의 같은
# 정보라서, 사이클 단위로 뽑으면 밴드가 실제보다 좁게(과신하게) 나오기 때문입니다.
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
SURVIVAL_GRID = [0, 10, 20, 30, 40, 60, 80, 100, 120, 140]  # 기존 화면의 x축 눈금 그대로

# 센서 추이 차트에 보여줄 점 개수 (엔진 전체 관측 구간에서 고르게 9점 추출)
# 보여줄 센서는 06번 RF 중요도 상위 3개 (FD004: s3·s17·s8, FD001: s4·s9·s3) — 아래 1번에서 결정.
# 값은 07번이 저장한 운전조건 보정값 (FD004는 비행 조건마다 원본값이 크게 튀어서 추세가 안 보이므로).
SENSOR_TREND_POINTS = 9
N_TREND_SENSORS = 3
TREND_DOMAIN_PAD = 0.10  # y축 범위 = train 전체 최소~최대에 범위의 10%씩 여유

# ---------------------------------------------------------------------------
# [임의 설정값 #10] 비용 시뮬레이션 가정값 (단위: 만원, 시간)
# ---------------------------------------------------------------------------
# C-MAPSS에는 비용·정비 시간 정보가 전혀 없어서, 항공기 엔진 정비 맥락에서 상식적으로
# 가정한 "설명용(illustrative)" 값입니다. 실제 현장 값을 구하면 여기만 바꾸면 됩니다.
# (탐지/누락/오탐 건수는 가정이 아니라 모델의 실제 test 결과에서 계산합니다.)
COST_ASSUMPTIONS = {
    "general": {
        "downtimeCostPerHour": 1000.0,  # 엔진 1대 가동중단 1시간당 손실 (=0.1억원)
        "repairUnplannedRatio": 3.0,    # 고장 후 응급 수리비 = 계획 수리비 × 3
        "dangerCaseLoss": 0.0,          # 위험 상태 방치 1건당 추가 손실 (현재 미반영)
        "falseAlarmHours": 0.5,         # 오탐 1건당 불필요한 점검 시간
        "falseAlarmLabor": 50.0,        # 오탐 1건당 점검 인건비
    },
    "EngineRemoval": {
        "hoursPlanned": 4.0,     # RUL 기반으로 미리 계획한 정비의 가동중단 시간
        "hoursUnplanned": 24.0,  # 고장 후 비계획 정비의 가동중단 시간
        "repairPlanned": 800.0,  # 계획 정비 1건 수리비
    },
}

STATUS_OF = {"RED": "critical", "YELLOW": "warning", "GREEN": "good"}


# ---------------------------------------------------------------------------
# 1. 입력 읽기
# ---------------------------------------------------------------------------
with open(f"{OUT_DASH}/engines_summary.json", encoding="utf-8") as f:
    summary = pd.DataFrame(json.load(f)).set_index("unit")


def load_timeseries(unit: int) -> dict:
    with open(f"{OUT_DASH}/engine_timeseries/{unit}.json", encoding="utf-8") as f:
        return json.load(f)


train_raw = load_raw(TRAIN_FILE)

# 검증셋 예측 + 자르지 않은 실제 RUL (= 그 엔진의 마지막 사이클 - 현재 사이클)
val = pd.read_csv(f"{OUT_METRICS}/gru_val_predictions.csv")
life = train_raw.groupby("unit")["cycle"].max()
val["RUL_raw"] = val["unit"].map(life) - val["cycle"]
val_units = np.sort(val["unit"].unique())

with open(f"{OUT_METRICS}/domain_interpretation.json", encoding="utf-8") as f:
    domain = json.load(f)
TREND_SENSORS = list(domain["top_sensors_rf"])[:N_TREND_SENSORS]

# ---------------------------------------------------------------------------
# 2. 신뢰구간 · 생존곡선 ([임의 설정값 #9])
# ---------------------------------------------------------------------------
def neighbors(frame: pd.DataFrame, p: float) -> pd.DataFrame:
    """예측값이 p와 비슷했던(±NEIGHBOR_RADIUS) 검증셋 순간들."""
    dist = (frame["RUL_pred"] - p).abs()
    near = frame[dist <= NEIGHBOR_RADIUS]
    if len(near) < MIN_NEIGHBORS:
        near = frame.loc[dist.nsmallest(MIN_NEIGHBORS).index]
    return near


def survival(rul: np.ndarray) -> np.ndarray:
    return np.array([(rul > x).mean() for x in SURVIVAL_GRID])


rng = np.random.default_rng(BOOTSTRAP_SEED)
by_unit = {u: g for u, g in val.groupby("unit")}
boot_frames = [
    pd.concat([by_unit[u] for u in rng.choice(val_units, size=len(val_units), replace=True)])
    for _ in range(BOOTSTRAP_N)
]


def rul_uncertainty(p: float):
    near = neighbors(val, p)["RUL_raw"].to_numpy()
    lo, hi = np.quantile(near, INTERVAL_QUANTILES)
    interval = {"median": round(p, 1), "lower": round(max(0.0, float(lo)), 1), "upper": round(float(hi), 1)}

    s_mid = survival(near)
    s_boot = np.array([survival(neighbors(bf, p)["RUL_raw"].to_numpy()) for bf in boot_frames])
    s_lo, s_hi = np.quantile(s_boot, INTERVAL_QUANTILES, axis=0)
    curve = [
        {"cycle": x, "median": round(float(m), 4), "lower": round(float(min(l, m)), 4), "upper": round(float(max(h, m)), 4)}
        for x, m, l, h in zip(SURVIVAL_GRID, s_mid, s_lo, s_hi)
    ]
    return interval, curve


# ---------------------------------------------------------------------------
# 3. 엔진 9대 요약 + 상세
# ---------------------------------------------------------------------------
def health_of(p: float) -> int:
    """헬스 점수(0~100) = 예측 RUL을 RUL 상한(125) 대비 비율로 환산."""
    return int(round(min(100.0, 100.0 * p / RUL_CLIP_VALUE)))


red, yellow = RISK_THRESHOLDS["red_at_or_below"], RISK_THRESHOLDS["yellow_below"]


def window_curve(u: int) -> list:
    return load_timeseries(u)["predicted_RUL_curve"][-FEATURED_WINDOW:]


# 대표 엔진 ([임의 설정값 #8])
red_units = summary.index[summary["risk_level"] == "RED"]
drops = {int(u): window_curve(u)[0] - min(window_curve(u)) for u in red_units if len(window_curve(u)) == FEATURED_WINDOW}
candidates = [u for u in drops if window_curve(u)[0] >= yellow and min(window_curve(u)) <= red]
featured = max(candidates or drops, key=lambda u: drops[u])

# 엔진 9대 ([임의 설정값 #8])
by_level = {lv: summary[summary["risk_level"] == lv].sort_values("predicted_RUL") for lv in ("RED", "YELLOW", "GREEN")}
if len(by_level["RED"]) < N_RED + 1 or len(by_level["YELLOW"]) < N_YELLOW or len(by_level["GREEN"]) < 2:
    raise SystemExit(f"등급별 엔진 수가 부족합니다: { {k: len(v) for k, v in by_level.items()} }")
SELECTED_ENGINES = ([featured] + [int(u) for u in by_level["RED"].index if u != featured][:N_RED]
                    + [int(u) for u in by_level["YELLOW"].index[:N_YELLOW]]
                    + [int(by_level["GREEN"].index[0]), int(by_level["GREEN"].index[-1])])

ranking, details = [], {}
for u in SELECTED_ENGINES:
    p = float(summary.loc[u, "predicted_RUL"])
    level = summary.loc[u, "risk_level"]
    ts = load_timeseries(u)
    interval, curve = rul_uncertainty(p)

    idx = np.unique(np.linspace(0, len(ts["cycles"]) - 1, SENSOR_TREND_POINTS).round().astype(int))
    sensor_trend = {
        s: [{"h": ts["cycles"][i], "v": round(float(ts["sensors"][s][i]), 2)} for i in idx]
        for s in TREND_SENSORS
    }
    last_cycle = int(summary.loc[u, "last_observed_cycle"])
    details[f"engine-{u}"] = {
        "equipmentInfo": {
            "line": f"터보팬 엔진 · C-MAPSS {DATASET} ({DATASET_DESC[DATASET]})",
            "installedAt": "—(데이터에 없음)",
            "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
            "team": "5조 감시자들",
            "operatingCycles": last_cycle,
            "modelNo": f"Turbofan ({DATASET})",
        },
        "predictedRulCycle": interval,
        "survivalCurve": curve,
        "sensorTrend": sensor_trend,
        "maintenanceHistory": [{
            "date": f"cycle #{last_cycle} 시점 예측",
            "type": "예정",
            "description": f"RUL 예측 기반 — 잔존 약 {interval['median']} 사이클 소진 시 정비 권고 "
                           f"(90% 구간 {interval['lower']}~{interval['upper']})",
            "status": "scheduled",
        }],
        "status": STATUS_OF[level],
        "health": health_of(p),
    }
    ranking.append({"id": f"engine-{u}", "name": f"Engine #{u}", "health": health_of(p), "status": STATUS_OF[level]})

ranking.sort(key=lambda r: r["health"])

# ---------------------------------------------------------------------------
# 4. 대표 엔진 (모니터링 화면 메인 차트 + 경보 로그) — 선정 규칙은 위 3번
# ---------------------------------------------------------------------------
ts = load_timeseries(featured)
cyc = ts["cycles"][-FEATURED_WINDOW:]
rul = ts["predicted_RUL_curve"][-FEATURED_WINDOW:]
sensor_series = [{"t": f"#{c}", "h": h, "v": round(float(v), 2)} for h, (c, v) in enumerate(zip(cyc, rul))]

first_warn = next((h for h, v in enumerate(rul) if v < yellow), 0)
first_crit = next(h for h, v in enumerate(rul) if v <= red)
alert_log = [
    {"h": first_crit, "time": f"cycle #{cyc[first_crit]}", "equipment": f"Engine #{featured}", "sensor": "예측 RUL",
     "severity": "critical", "action": "정비 일정 즉시 수립", "type": "EngineRemoval"},
    {"h": first_warn, "time": f"cycle #{cyc[first_warn]}", "equipment": f"Engine #{featured}", "sensor": "예측 RUL",
     "severity": "warning", "action": "정비 계획 준비", "type": "EngineRemoval"},
]
anomaly_window = {"start": first_crit, "end": FEATURED_WINDOW - 1}

# ---------------------------------------------------------------------------
# 5. 조기경보 성능 (예측 RUL ≤ 위험 기준을 "위험"으로 보는 신호등 규칙, 공식 test 엔진의 마지막 시점)
# ---------------------------------------------------------------------------
pred_danger = summary["predicted_RUL"] <= red
true_danger = summary["true_RUL_for_validation_only"] <= red
tp = int((pred_danger & true_danger).sum())
fn = int((~pred_danger & true_danger).sum())
fp = int((pred_danger & ~true_danger).sum())
precision, recall = tp / (tp + fp), tp / (tp + fn)
classifier_metrics = {
    "precision": round(precision, 3), "recall": round(recall, 3),
    "f1": round(2 * precision * recall / (precision + recall), 3), "tp": tp, "fn": fn, "fp": fp,
}

# ---------------------------------------------------------------------------
# 6. 센서 중요도 상위 5개 (06번 결과: RF 피처 중요도를 센서별로 합산)
# ---------------------------------------------------------------------------
def short_label(sensor: str) -> str:
    # "LPT(저압터빈) outlet 온도" → "LPT outlet 온도" (카드 폭이 좁아 한글 풀이 괄호만 뺌)
    return re.sub(r"\([가-힣]+\)", "", domain["sensor_meaning"][sensor]).strip()


feature_importance = [
    {"label": f"{short_label(s)} ({s})", "value": round(v, 3)}
    for s, v in list(domain["top_sensors_rf"].items())[:5]
]

# 센서 추이 차트의 y축 범위·이름표 (값이 운전조건 보정값이므로 범위도 보정값 기준)
train_view = RegimeCorrector().fit(train_raw, TREND_SENSORS).transform(train_raw)
trend_meta = {}
for s in TREND_SENSORS:
    lo, hi = train_view[s].min(), train_view[s].max()
    pad = (hi - lo) * TREND_DOMAIN_PAD
    trend_meta[s] = {"unit": SENSOR_UNIT[s], "label": f"{short_label(s)} ({s})",
                     "domain": [round(lo - pad, 1), round(hi + pad, 1)]}

# ---------------------------------------------------------------------------
# 7. 비용 시뮬레이션 (Simulation.tsx의 computeScenario()와 똑같은 식)
# ---------------------------------------------------------------------------
n_engines = len(summary)
cost_model = {
    "nEngines": n_engines,
    "general": COST_ASSUMPTIONS["general"],
    "types": {
        "EngineRemoval": {
            "episodes": tp + fn,          # 실제로 위험 상태였던 엔진 수
            "detected": tp,
            "missed": fn,
            "falseAlarms": fp,
            "dangerCases": tp + fn,       # AS-IS: 모두 위험 상태로 방치됨
            "dangerCasesDetected": 0,     # TO-BE: 탐지된 엔진은 미리 정비해서 방치 0
            "dangerCasesMissed": fn,      # TO-BE: 놓친 엔진만 위험 상태로 남음
            **COST_ASSUMPTIONS["EngineRemoval"],
        }
    },
}


def compute_scenario(cm: dict) -> dict:
    g = cm["general"]
    asis = {"hours": 0.0, "repair": 0.0, "cases": 0}
    tobe = {"hours": 0.0, "repair": 0.0, "cases": 0}
    for t in cm["types"].values():
        rep_u = t["repairPlanned"] * g["repairUnplannedRatio"]
        asis["hours"] += t["episodes"] * t["hoursUnplanned"]
        asis["repair"] += t["episodes"] * rep_u
        asis["cases"] += t["dangerCases"]
        tobe["hours"] += t["detected"] * t["hoursPlanned"] + t["missed"] * t["hoursUnplanned"] + t["falseAlarms"] * g["falseAlarmHours"]
        tobe["repair"] += t["detected"] * t["repairPlanned"] + t["missed"] * rep_u + t["falseAlarms"] * g["falseAlarmLabor"]
        tobe["cases"] += t["dangerCasesDetected"] + t["dangerCasesMissed"]
    loss = lambda x: x["hours"] * g["downtimeCostPerHour"] + x["cases"] * g["dangerCaseLoss"]
    per1000 = 1000 / cm["nEngines"]
    r1 = lambda v: round(v * 10) / 10
    asis_total, tobe_total = asis["repair"] + loss(asis), tobe["repair"] + loss(tobe)
    return {
        "metrics": [
            {"metric": "평균 가동중단 시간 (시간/1,000대 환산)", "asIs": r1(asis["hours"] * per1000), "toBe": r1(tobe["hours"] * per1000)},
            {"metric": "정비 비용 (억원/1,000대 환산)", "asIs": r1(asis["repair"] * per1000 / 10000), "toBe": r1(tobe["repair"] * per1000 / 10000)},
            {"metric": "가동중단 손실 (억원/1,000대 환산)", "asIs": r1(loss(asis) * per1000 / 10000), "toBe": r1(loss(tobe) * per1000 / 10000)},
        ],
        "savingPer1000": r1((asis_total - tobe_total) * per1000 / 10000),
        "savingRate": round((asis_total - tobe_total) / asis_total * 1000) / 10,
        "dangerCaseReduction": round((1 - tobe["cases"] / asis["cases"]) * 100),
    }


scenario = compute_scenario(cost_model)
savings_summary = {
    "perThousandEnginesEok": scenario["savingPer1000"],
    "savingRatePct": scenario["savingRate"],
    "dangerCaseReductionPct": scenario["dangerCaseReduction"],
    "note": f"테스트 엔진 {n_engines}대 결과를 1,000대 규모로 환산한 값입니다. 정비 비용·가동중단 단가는 가정값이고, "
            "탐지/누락/오탐 건수는 실제 모델 성능입니다.",
}

data_meta = {
    "generatedAt": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    "dataset": f"C-MAPSS {DATASET}",
    "datasetDescription": DATASET_DESC[DATASET],
    "sensorValues": "운전조건 보정값" if MULTI_REGIME else "원본값",
    "model": MODEL_NAME,
    "testEngines": n_engines,
    "selectedEngines": [int(r["id"].split("-")[1]) for r in ranking],
    "featuredEngine": featured,
    "note": f"시간축은 엔진 운행 사이클(cycle). 위험 등급(RED/YELLOW/GREEN)은 예측 RUL 기준(≤{red}/<{yellow}/그 외). "
            f"비용·정비시간은 가정값(08_export_dashboard_ts.py COST_ASSUMPTIONS), 탐지 성능(TP/FN/FP)은 {MODEL_NAME} 모델의 실제 test set 결과. "
            "RUL 신뢰구간·생존곡선은 검증셋에서 예측이 비슷했던 사례들의 실제 RUL 분포.",
}

# ---------------------------------------------------------------------------
# 8. mock.ts 쓰기
# ---------------------------------------------------------------------------
J = lambda obj: json.dumps(obj, ensure_ascii=False, indent=2)

ts_code = f"""// ⚠️ 자동 생성 파일 — 분석 레포의 scripts/08_export_dashboard_ts.py 가 만든다.
// 직접 고치지 말고 파이썬 파이프라인을 다시 돌려서 새로 받아올 것.
// 원본 데이터: C-MAPSS {DATASET} ({DATASET_DESC[DATASET]}, NASA 터보팬 엔진 열화 시뮬레이션), 모델: {MODEL_NAME} Seq2Seq 회귀
// (분석 레포 docs/DESIGN_DECISIONS.md, 이 레포 docs/DATA_MAPPING.md 참고)
// 시간축은 엔진 운행 '사이클(cycle)' 입니다.

export type Status = 'good' | 'warning' | 'serious' | 'critical'

export const statusLabel: Record<Status, string> = {{
  good: '정상',
  warning: '주의',
  serious: '경고',
  critical: '위험',
}}

export const dataMeta = {J(data_meta)}

/** 신호등 기준 (분석 레포 common.py의 DANGER_RUL · RISK_THRESHOLDS) — 화면 문구도 이 값을 씀
 *  위험: 예측 RUL ≤ dangerRul, 주의: dangerRul 초과 ~ warningRul 미만, 정상: warningRul 이상 */
export const riskThresholds = {{ dangerRul: {red}, warningRul: {yellow} }}

/** 메인 차트: 대표 엔진(#{featured})의 예측 RUL 추이 — h = 창 안의 사이클 순서(0~{FEATURED_WINDOW - 1}) */
export const sensorSeries: {{ t: string; h: number; v: number }}[] = {J(sensor_series)}

export const anomalyWindow = {J(anomaly_window)}

const START_CYCLE: number = {cyc[0]}

export function formatHour(h: number): string {{
  return `cycle #${{START_CYCLE + Math.round(h)}}`
}}

/** 스크러버 위치(h)에서 대표 엔진의 예측 RUL (사이클 사이는 선형 보간) */
export function getRulAtHour(hour: number): number {{
  const clamped = Math.max(0, Math.min({FEATURED_WINDOW - 1}, hour))
  for (let i = 0; i < sensorSeries.length - 1; i++) {{
    const a = sensorSeries[i]
    const b = sensorSeries[i + 1]
    if (clamped >= a.h && clamped <= b.h) {{
      const frac = (clamped - a.h) / (b.h - a.h)
      return a.v + (b.v - a.v) * frac
    }}
  }}
  return sensorSeries[sensorSeries.length - 1].v
}}

export interface EquipmentRow {{
  id: string
  name: string
  health: number
  status: Status
}}

/** 선택된 {len(ranking)}개 엔진의 헬스 점수(0~100) = 100 × (예측 RUL / {RUL_CLIP_VALUE}).
 *  참고: 각 엔진은 서로 독립된 운행 이력이라 '공유된 시간축'이 없어서,
 *  스크러버(hour)에 따라 순위가 실시간으로 바뀌지는 않고 마지막 관측 시점 기준 스냅샷입니다. */
const equipmentRankingStatic: EquipmentRow[] = {J(ranking)}

export function getEquipmentRankingAtHour(_hour: number): EquipmentRow[] {{
  return equipmentRankingStatic
}}

export const equipmentRanking = equipmentRankingStatic

export interface AlertRow {{
  time: string
  h: number
  equipment: string
  sensor: string
  severity: Status
  action: string
  type?: string
}}

/** 대표 엔진(#{featured})의 예측 RUL이 주의({yellow} 미만)/위험({red} 이하) 임계값을 넘은 시점 — 실제 계산값 */
export const alertLog: AlertRow[] = {J(alert_log)}

export function getAlertsUpToHour(hour: number): AlertRow[] {{
  return alertLog.filter((a) => a.h <= hour)
}}

export const kpis = {{
  totalEquipment: {len(ranking)},
  riskEquipment: equipmentRanking.filter((e) => e.status === 'critical').length,
  avgHealth: equipmentRanking.reduce((s, e) => s + e.health, 0) / equipmentRanking.length,
  todayAlerts: alertLog.length,
  alertsDelta: 0,
}}

export function getKpisAtHour(hour: number) {{
  const ranking = getEquipmentRankingAtHour(hour)
  const riskEquipment = ranking.filter((e) => e.status === 'critical').length
  const avgHealth = ranking.reduce((s, e) => s + e.health, 0) / ranking.length
  const todayAlerts = getAlertsUpToHour(hour).length
  return {{
    totalEquipment: kpis.totalEquipment,
    riskEquipment,
    avgHealth,
    todayAlerts,
    alertsDelta: kpis.alertsDelta,
  }}
}}

// ---------------------------------------------------------------------------
// 엔진별 상세 데이터 (설비 상세 페이지 — /equipment/:id 로 조회)
// predictedRulCycle(90% 구간)·survivalCurve는 검증셋에서 예측이 비슷했던(±{NEIGHBOR_RADIUS} cycle)
// 사례들의 실제 RUL 분포로 계산 (생존곡선 음영 = 검증 엔진 단위 부트스트랩 90% 구간)
// ---------------------------------------------------------------------------
export interface EngineDetail {{
  equipmentInfo: {{ line: string; installedAt: string; lastMaintenance: string; team: string; operatingCycles: number; modelNo: string }}
  predictedRulCycle: {{ median: number; lower: number; upper: number }}
  survivalCurve: {{ cycle: number; median: number; lower: number; upper: number }}[]
  sensorTrend: Record<string, {{ h: number; v: number }}[]>
  maintenanceHistory: {{ date: string; type: string; description: string; status: 'done' | 'scheduled' }}[]
  status: Status
  health: number
}}

export const engineDetails: Record<string, EngineDetail> = {J(details)}

export const DEFAULT_ENGINE_ID = 'engine-{featured}'

/** 센서 추이 3종 (06번 RF 중요도 상위 센서) — 값은 {"운전조건 보정값" if MULTI_REGIME else "원본값"} */
export const trendMeta: Record<string, {{ unit: string; label: string; domain: [number, number] }}> = {J(trend_meta)}

/** 위험 등급(RUL≤{red}) 조기경보 성능 — {MODEL_NAME} 모델, 공식 test {n_engines}개 엔진 기준 실제 계산값
 *  (TP={tp}, FN={fn}, FP={fp}) */
export const classifierMetrics = {J(classifier_metrics)}

/** RF 피처 중요도 상위 5개 센서(그룹 합산) */
export const featureImportance = {J(feature_importance)}

/** AS-IS(사후 정비: 고장까지 운용) vs TO-BE(예지보전: RUL 기반 사전 정비) 비용 시뮬레이션.
 *  탐지/누락/오탐 건수는 {MODEL_NAME} 모델의 실제 test set 결과, 금액·시간 가정은 항공 엔진 정비
 *  맥락의 illustrative 값입니다 (분석 레포 scripts/08_export_dashboard_ts.py 참고). */
export const scenarioCompare = {J(scenario["metrics"])}

export const savingsSummary = {J(savings_summary)}

/** 시나리오 화면 슬라이더가 다시 계산할 때 쓰는 원본 가정치 (계산식은 분석 레포 compute_scenario()와 동일) */
export const costModel = {J(cost_model)}
"""

out_path = f"{OUT_DASH}/mock.ts"
with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(ts_code)

# ---------------------------------------------------------------------------
# 9. 확인용 출력
# ---------------------------------------------------------------------------
for r in ranking:
    u = int(r["id"].split("-")[1])
    d = details[r["id"]]["predictedRulCycle"]
    print(f"  Engine #{u:>3}  {summary.loc[u, 'risk_level']:<6} 예측 RUL {d['median']:>5}  90% 구간 {d['lower']:>5} ~ {d['upper']:>5}")
print(f"대표 엔진: #{featured} (후보 {candidates}), 조기경보 TP/FN/FP = {tp}/{fn}/{fp}")
print(f"저장: {out_path}  →  대시보드 레포 src/data/mock.ts 로 복사하세요.")
