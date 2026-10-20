"""
26_env_score_gru.py
====================
"운전 환경을 점수로 만들어 GRU 입력에 넣으면 좋아지는가?" 시험 (강사님 피드백: 온도·습도 같은 환경을 점수화해 넣어 보기).

C-MAPSS에는 날짜·공항 정보가 없어 실제 날씨(기온·습도)를 붙일 수 없습니다. 대신 데이터에 있는 운전 조건 6가지
(고도·마하수·스로틀 조합)를 환경으로 보고, 조건마다 "열 부하 점수"를 만들어 입력에 추가합니다.
습도에 해당하는 값은 데이터에 없어 이 시험에는 넣지 못했습니다.

점수 규칙 (train 엔진 200대에서만 정함 — validation·test는 보지 않음)
  조건별 열 부하 = 그 운전 조건에서 측정된 s3(고압 압축기 출구 온도, 원본 단위)의 평균 → 최소 0, 최대 1로 맞춤
  추가 입력 3개: env_now(지금 조건의 점수) · env_mean(지금까지 평균 점수) · env_cum(지금까지 점수 합 ÷ 200)
  → "가혹한 조건에서 얼마나 오래 운전했는가"를 모델이 알 수 있게 됨. 현재 사이클까지의 값만 쓰므로 미래 정보는 없음.

비교: 17번과 같은 분할(seed 42)·정규화·GRU 구조·학습 설정, 센서 15개만 입력(기준, 17번 base) vs 센서 15개 + 환경 점수 3개.
경보 기준은 예측 남은 수명 ≤ 40으로 고정.

실행: python scripts/26_env_score_gru.py [seed]      결과: outputs/FD004/metrics/env_score_gru_<seed>.json
"""

import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from common import (
    load_raw, find_constant_columns, add_rul_labels, split_engines, Normalizer, SENSOR_COLS_RAW,
    DANGER_RUL, RUL_CLIP_VALUE, OUT_METRICS, TRAIN_FILE, TEST_FILE, RUL_FILE,
)
from seq_models import RULGRU, engine_sequences, train_recurrent, predict_recurrent

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42

train_raw = add_rul_labels(load_raw(TRAIN_FILE))
train_units, val_units = split_engines(train_raw)
const_cols = find_constant_columns(train_raw[train_raw["unit"].isin(train_units)])
sensors = [c for c in SENSOR_COLS_RAW if c not in const_cols]
test_raw = load_raw(TEST_FILE)
rul_true = pd.read_csv(RUL_FILE, header=None, names=["RUL"])
rul_true["unit"] = np.arange(1, len(rul_true) + 1)

# --- 환경(열 부하) 점수: train 엔진에서만 ---------------------------------------------------------
fit_rows = train_raw[train_raw["unit"].isin(train_units)]
heat = fit_rows.groupby("regime")["s3"].mean()
score = (heat - heat.min()) / (heat.max() - heat.min())
print("운전 조건별 열 부하 점수 (0~1):")
print(pd.concat([heat.rename("s3 평균"), score.rename("점수")], axis=1).round(3).to_string())

norm = Normalizer().fit(fit_rows, sensors)


def prep(df_raw):
    df = norm.transform(df_raw).sort_values(["unit", "cycle"]).copy()
    df["env_now"] = df["regime"].map(score).astype(float)
    g = df.groupby("unit")["env_now"]
    df["env_mean"] = g.transform(lambda s: s.expanding().mean())
    df["env_cum"] = g.transform(lambda s: s.cumsum()) / 200.0
    return df


train_df = prep(train_raw[train_raw["unit"].isin(train_units)])
val_df = prep(train_raw[train_raw["unit"].isin(val_units)])
test_df = prep(test_raw)
env_cols = ["env_now", "env_mean", "env_cum"]
cols = sensors + env_cols


def alert_stats(unit, cycle, rul_true_, alarm) -> dict:
    d = pd.DataFrame({"unit": unit, "cycle": cycle, "danger": np.asarray(rul_true_) <= DANGER_RUL, "alarm": np.asarray(alarm)})
    fp = d.alarm & ~d.danger
    fn = ~d.alarm & d.danger
    by_fp = fp.groupby(d.unit).sum()
    by_fn = fn.groupby(d.unit).sum()
    return {
        "val_FP_cycles": int(fp.sum()), "val_FN_cycles": int(fn.sum()),
        "val_engines_with_false_alarm": int((d[~d.danger].groupby("unit")["alarm"].sum() > 0).sum()),
        "FP_179": int(by_fp.get(179, 0)), "FN_245": int(by_fn.get(245, 0)), "FN_235": int(by_fn.get(235, 0)),
    }


tr_s, tr_l, _ = engine_sequences(train_df, cols)
va_s, va_l, va_ids = engine_sequences(val_df, cols)
te_s, _, te_ids = engine_sequences(test_df, cols, has_label=False)
t0 = time.time()
model = train_recurrent(RULGRU, tr_s, tr_l, len(cols), seed=SEED)
rows = []
for uid, seq, lab in zip(va_ids, va_s, va_l):
    pred = predict_recurrent(model, seq)
    for c, t_, p_ in zip(range(1, len(seq) + 1), lab.numpy() * RUL_CLIP_VALUE, pred):
        rows.append((uid, c, t_, p_))
v = pd.DataFrame(rows, columns=["unit", "cycle", "RUL_true", "RUL_pred"])
res = alert_stats(v["unit"], v["cycle"], v["RUL_true"], v["RUL_pred"] <= DANGER_RUL)
res["val_MAE"] = round(float(np.mean(np.abs(v["RUL_true"] - v["RUL_pred"]))), 2)
t = pd.DataFrame([{"unit": u, "pred": predict_recurrent(model, s)[-1]} for u, s in zip(te_ids, te_s)]).merge(rul_true, on="unit")
for th in (40, 42, 46):
    a = (t["pred"] <= th).values
    d = (t["RUL"] <= DANGER_RUL).values
    res[f"test_FN_at_{th}"] = int((~a & d).sum())
    res[f"test_FP_at_{th}"] = int((a & ~d).sum())
res["test_MAE"] = round(float(np.mean(np.abs(t["RUL"] - t["pred"]))), 2)
res["seed"], res["seconds"] = SEED, round(time.time() - t0)
res["regime_score"] = {k: round(float(x), 3) for k, x in score.items()}
v.to_csv(f"{OUT_METRICS}/env_score_gru_{SEED}_val.csv", index=False)
with open(f"{OUT_METRICS}/env_score_gru_{SEED}.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(json.dumps(res, ensure_ascii=False, indent=2))
