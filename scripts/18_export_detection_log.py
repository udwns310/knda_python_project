"""
18. 탐지 이력(경보 로그) 내보내기 — 대시보드 '탐지 이력' 화면용

공식 test 엔진 248대 각각에 대해, 마지막 관측까지 GRU가 예측한 남은 수명(RUL) 곡선에서
"지금 켜져 있는 경보가 언제부터 켜졌는가"를 찾고, 공식 정답(RUL_FD004)과 맞춰 판정합니다.

규칙 (경계값은 모두 common.py의 RISK_THRESHOLDS를 읽음)
  - 주의 구간: 예측 RUL < 80, 위험 구간: 예측 RUL ≤ 40  (common.risk_level 과 같은 경계)
  - 마지막 관측 시점의 등급은 engines_summary.json 의 risk_level(공식 등급)을 그대로 씁니다.
    (곡선 파일은 소수 둘째 자리로 반올림돼 있어 80.0 같은 경계값에서 등급이 어긋날 수 있음 — 엔진 117)
  - 위험 시작 / 주의 시작 = 마지막 관측 시점까지 구간 안에 '계속' 머무르기 시작한 사이클.
    예측이 중간에 기준을 잠깐 넘었다가 돌아온 경우는 '일시 경보'(구간에 들어갔다 나온 횟수)로 따로 셉니다.
    (처음 교차한 시점을 쓰면 수명 중반의 일시적 흔들림이 '241사이클 앞서 탐지'처럼 보입니다 — 엔진 180)
  - 판정(마지막 관측 시점, 08·09번의 TP/FN/FP와 같은 정의)
      탐지   = 예측 위험(RED) 이고 실제 남은 수명 ≤ 40
      놓침   = 예측 위험이 아닌데 실제 남은 수명 ≤ 40
      헛경보 = 예측 위험(RED) 인데 실제 남은 수명 > 40
      관찰   = 마지막 예측이 주의(YELLOW)이고 실제는 위험이 아님
      해제   = 경보 구간에 들어간 적이 있지만 마지막에는 정상(GREEN)
  - 선행 시간 = 위험 시작 사이클에서 실제 고장까지 남은 사이클
      = (마지막 관측 사이클 - 위험 시작 사이클) + 마지막 관측 시점의 실제 남은 수명
  - 시간축은 엔진마다 자기만의 사이클입니다. 엔진 사이를 줄 세우는 공통 시계는 없습니다.
  - '실제 남은 수명'은 평가용 정답(RUL_FD004)이며 현장에서는 알 수 없는 값입니다.

출력: outputs/<데이터셋>/dashboard_data/detection_log.ts, detection_curves.ts, detection_log.json
      → 대시보드 레포의 src/data/ 로 복사 (직접 고치지 말고 이 스크립트를 다시 돌릴 것)
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from common import RISK_THRESHOLDS, OUT_DASH

RED = RISK_THRESHOLDS["red_at_or_below"]
YELLOW = RISK_THRESHOLDS["yellow_below"]

summary = pd.DataFrame(json.load(open(f"{OUT_DASH}/engines_summary.json", encoding="utf-8"))).set_index("unit")


def runs(mask: np.ndarray):
    """True가 이어지는 구간들을 (시작 인덱스, 끝 인덱스) 목록으로"""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


rows, curves = [], {}
for u in summary.index:
    ts = json.load(open(f"{OUT_DASH}/engine_timeseries/{u}.json", encoding="utf-8"))
    cyc = np.asarray(ts["cycles"], dtype=int)
    pred = np.asarray(ts["predicted_RUL_curve"], dtype=float)
    last_i = len(cyc) - 1
    last = int(cyc[last_i])
    true_end = float(summary.loc[u, "true_RUL_for_validation_only"])
    level = summary.loc[u, "risk_level"]  # 마지막 시점의 공식 등급

    warn_mask, danger_mask = pred < YELLOW, pred <= RED
    warn_mask[last_i] = level in ("YELLOW", "RED")
    danger_mask[last_i] = level == "RED"
    warn_runs, danger_runs = runs(warn_mask), runs(danger_mask)

    true_danger = true_end <= RED
    if not warn_runs and not true_danger:
        continue  # 경보가 한 번도 없었고 실제로도 위험하지 않은 엔진은 로그에 없음

    def lead(i):
        return None if i is None else int(round(true_end + (last - int(cyc[i]))))

    warn_since = next((s for s, e in warn_runs if e == last_i), None)
    danger_since = next((s for s, e in danger_runs if e == last_i), None)
    lead_danger = lead(danger_since)

    pred_danger = level == "RED"
    if pred_danger and true_danger:
        verdict = "tp"
    elif not pred_danger and true_danger:
        verdict = "fn"
    elif pred_danger:
        verdict = "fp"
    elif level == "YELLOW":
        verdict = "watch"
    else:
        verdict = "cleared"

    rows.append({
        "unit": int(u),
        "verdict": verdict,
        "lastCycle": last,
        "warnSince": None if warn_since is None else int(cyc[warn_since]),
        "dangerSince": None if danger_since is None else int(cyc[danger_since]),
        "leadDanger": lead_danger,
        # 위험 경보가 기준(40)보다 일찍, 실제 남은 수명이 40을 넘을 때 켜진 경우
        "earlyDanger": bool(lead_danger is not None and lead_danger > RED),
        # 구간에 들어갔다가 마지막 관측 전에 나온 횟수 (일시 경보)
        "transientWarn": sum(1 for s, e in warn_runs if e != last_i),
        "transientDanger": sum(1 for s, e in danger_runs if e != last_i),
        "predEnd": round(float(pred[last_i]), 1),
        "trueEnd": int(round(true_end)),
    })
    curves[int(u)] = {"c": [int(x) for x in cyc], "p": [round(float(x), 1) for x in pred]}

counts = {v: sum(1 for r in rows if r["verdict"] == v) for v in ("tp", "fn", "fp", "watch", "cleared")}
leads = [r["leadDanger"] for r in rows if r["verdict"] == "tp" and r["leadDanger"] is not None]
summary_out = {
    **counts,
    "engines": int(len(summary)),
    "logged": len(rows),
    "earlyDangerTp": sum(1 for r in rows if r["verdict"] == "tp" and r["earlyDanger"]),
    "withTransientDanger": sum(1 for r in rows if r["transientDanger"] > 0),
    "leadMedianTp": float(np.median(leads)) if leads else None,
    "leadMinTp": int(min(leads)) if leads else None,
    "leadMaxTp": int(max(leads)) if leads else None,
    "red": RED,
    "yellow": YELLOW,
}

# 정렬: 놓침 → 헛경보 → 탐지 → 관찰 → 해제 (같은 판정 안에서는 엔진 번호)
ORDER = {"fn": 0, "fp": 1, "tp": 2, "watch": 3, "cleared": 4}
rows.sort(key=lambda r: (ORDER[r["verdict"]], r["unit"]))

with open(f"{OUT_DASH}/detection_log.json", "w", encoding="utf-8") as f:
    json.dump({"summary": summary_out, "rows": rows}, f, ensure_ascii=False)

J = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"))
stamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
ts_code = f"""// ⚠️ 자동 생성 파일 — 분석 레포의 scripts/18_export_detection_log.py 가 만든다.
// 직접 고치지 말고 파이썬을 다시 돌려서 새로 받아올 것.
// 공식 test {len(summary)}대 엔진의 마지막 관측까지 GRU 예측 RUL이 주의(<{YELLOW})/위험(≤{RED}) 구간에 머무르기 시작한 시점과 판정.
// 시간축은 엔진마다 자기만의 사이클이고, trueEnd(실제 남은 수명)는 평가용 정답이다 (생성: {stamp}).

export type Verdict = 'tp' | 'fn' | 'fp' | 'watch' | 'cleared'

export interface LogRow {{
  unit: number
  verdict: Verdict
  lastCycle: number
  warnSince: number | null
  dangerSince: number | null
  leadDanger: number | null
  earlyDanger: boolean
  transientWarn: number
  transientDanger: number
  predEnd: number
  trueEnd: number
}}

export interface LogSummary {{
  tp: number
  fn: number
  fp: number
  watch: number
  cleared: number
  engines: number
  logged: number
  earlyDangerTp: number
  withTransientDanger: number
  leadMedianTp: number | null
  leadMinTp: number | null
  leadMaxTp: number | null
  red: number
  yellow: number
}}

export const logSummary: LogSummary = {J(summary_out)}

export const logRows: LogRow[] = {J(rows)}
"""
with open(f"{OUT_DASH}/detection_log.ts", "w", encoding="utf-8") as f:
    f.write(ts_code)

# 엔진별 예측 곡선은 크기 때문에 따로 둠 — 화면에서 엔진을 펼칠 때만 불러옴
curves_code = f"""// ⚠️ 자동 생성 파일 — 분석 레포의 scripts/18_export_detection_log.py 가 만든다. 직접 고치지 말 것.
// 엔진별 GRU 예측 RUL 곡선 (c = 사이클, p = 예측 RUL). 탐지 이력 화면에서 엔진을 펼칠 때만 불러온다.

export const detectionCurves: Record<number, {{ c: number[]; p: number[] }}> = {J(curves)}
"""
with open(f"{OUT_DASH}/detection_curves.ts", "w", encoding="utf-8") as f:
    f.write(curves_code)

print(f"로그 {len(rows)}대 / 전체 {len(summary)}대 — " + ", ".join(f"{k}={v}" for k, v in counts.items()))
print(f"탐지 선행 시간(위험 시작 → 실제 고장) 중앙값 {summary_out['leadMedianTp']}, 범위 {summary_out['leadMinTp']}~{summary_out['leadMaxTp']}, "
      f"기준보다 일찍 켜진 탐지 {summary_out['earlyDangerTp']}대, 일시 위험 경보를 겪은 엔진 {summary_out['withTransientDanger']}대")
