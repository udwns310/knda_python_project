"""
29_weather_score_gru.py
========================
실제 공항 기온·습도(기상청 AMOS 2025, 28번이 받은 파일)를 점수로 만들어 GRU 입력에 넣었을 때 성능이 달라지는지 시험합니다
(강사님 피드백: 온도·습도를 점수화해 GRU에 넣어 보기).

⚠ C-MAPSS에는 비행 날짜·공항 기록이 없어 엔진과 날씨를 잇는 열쇠가 없습니다. 그래서 아래 '가정'으로 엔진마다 날씨를 붙입니다.
   - 엔진마다 2025년 중 시작 날짜를 무작위로 정함 (엔진 번호·분할·날씨 시드로 고정)
   - 엔진마다 기준 공항 1곳을 무작위로 정함 (7곳 중, 지점 이름은 확인하지 못해 익명)
   - 하루 비행 5회(25번 시나리오와 같음): 사이클 c의 날짜 = 시작일 + (c-1)//5 일, 시각 = 02·08·14·20시 중 (c-1)%4 번째
   - 그 날짜·시각·공항의 실제 기온(°C)·습도(%)를 가져옴 (결측은 앞 시점 값, 그래도 없으면 전체 평균)
날씨는 엔진 열화 시뮬레이션과 무관하게 붙인 값이라, 개선이 나오더라도 근거로 볼 수 없습니다. 배정을 바꿔 3번(시드 1~3) 반복해
'날씨를 무작위로 붙였을 때의 변동 폭'을 같이 기록합니다. 실제 비행 일시·공항 기록이 생기면 같은 코드로 진짜 시험을 할 수 있습니다.

입력 점수 4개 (현재 사이클까지의 값만 사용, 미래 정보 없음; 점수 규칙은 데이터와 무관한 고정 공식)
   temp_score = clip(기온/35, 0, 1)            더울수록 큼
   hum_score  = 습도/100                          습할수록 큼
   stress_mean = 지금까지 (temp_score+hum_score)/2 의 평균
   stress_cum  = 지금까지 (temp_score+hum_score)/2 의 합 ÷ 200

비교: 26번과 같은 분할(시드 42)·정규화·GRU·학습 설정. 센서 15개만 (기준, 17번 base) vs 센서 15개 + 날씨 점수 4개. 경보 기준은 예측 ≤ 40.

실행: python scripts/29_weather_score_gru.py            (3개 배정 × 약 11분)
결과: outputs/FD004/metrics/weather_gru_<배정시드>.json, weather_gru_summary.json
"""

import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from common import (
    load_raw, find_constant_columns, add_rul_labels, split_engines, Normalizer, SENSOR_COLS_RAW,
    DANGER_RUL, RUL_CLIP_VALUE, OUT_METRICS, TRAIN_FILE, TEST_FILE, RUL_FILE, PROJECT_ROOT,
)
from seq_models import RULGRU, engine_sequences, train_recurrent, predict_recurrent

MODEL_SEED = 42
WX_SEEDS = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
CYCLES_PER_DAY = 5
SLOTS = [2, 8, 14, 20]

# --- 날씨 표: [날짜 365][시각 4][공항 n] ---------------------------------------------------------
wx = pd.read_csv(PROJECT_ROOT / "outputs" / "kr_public" / "airport_weather_2025_slots.csv")
# 분 자료 중 결측 코드가 일부만 섞인 시간 평균(2건: 기온 -40°C 미만, 습도 0 미만)은 결측으로 처리
wx.loc[(wx["temp_c"] < -40) | (wx["hum"] < 0) | (wx["hum"] > 100), ["temp_c", "hum"]] = np.nan
stations = sorted(wx["stn"].unique())
dates = pd.date_range("2025-01-01", periods=365).strftime("%Y-%m-%d")
T = np.full((365, 4, len(stations)), np.nan)
H = np.full((365, 4, len(stations)), np.nan)
d_idx = {d: i for i, d in enumerate(dates)}
s_idx = {s: i for i, s in enumerate(SLOTS)}
st_idx = {s: i for i, s in enumerate(stations)}
for r in wx.itertuples(index=False):
    if r.date in d_idx and r.slot in s_idx:
        T[d_idx[r.date], s_idx[r.slot], st_idx[r.stn]] = r.temp_c
        H[d_idx[r.date], s_idx[r.slot], st_idx[r.stn]] = r.hum
for arr in (T, H):  # 결측: 같은 공항·시각의 앞 날짜 값 → 그래도 없으면 전체 평균
    for s in range(4):
        for k in range(len(stations)):
            col = pd.Series(arr[:, s, k]).ffill().bfill()
            arr[:, s, k] = col.to_numpy()
    arr[np.isnan(arr)] = np.nanmean(arr)
print(f"날씨 표: 공항 {len(stations)}곳, 기온 평균 {T.mean():.1f}°C, 습도 평균 {H.mean():.1f}%", flush=True)

# --- 데이터 (26번과 같은 분할·정규화) ---------------------------------------------------------------
train_raw = add_rul_labels(load_raw(TRAIN_FILE))
train_units, val_units = split_engines(train_raw)
const_cols = find_constant_columns(train_raw[train_raw["unit"].isin(train_units)])
sensors = [c for c in SENSOR_COLS_RAW if c not in const_cols]
test_raw = load_raw(TEST_FILE)
rul_true = pd.read_csv(RUL_FILE, header=None, names=["RUL"])
rul_true["unit"] = np.arange(1, len(rul_true) + 1)
fit_rows = train_raw[train_raw["unit"].isin(train_units)]
norm = Normalizer().fit(fit_rows, sensors)
WX_COLS = ["temp_score", "hum_score", "stress_mean", "stress_cum"]


def add_weather(df_norm, split_tag, wx_seed):
    """엔진마다 시작 날짜·기준 공항을 (분할 이름, 엔진 번호, 배정 시드)로 고정해 정하고 사이클마다 날씨 점수를 붙임."""
    df = df_norm.sort_values(["unit", "cycle"]).copy()
    out = []
    for u, g in df.groupby("unit", sort=False):
        rng = np.random.default_rng([wx_seed, {"train": 1, "val": 2, "test": 3}[split_tag], int(u)])
        start = int(rng.integers(0, 365))
        stn = int(rng.integers(0, len(stations)))
        c = g["cycle"].to_numpy() - 1
        di = (start + c // CYCLES_PER_DAY) % 365
        si = c % 4
        temp, hum = T[di, si, stn], H[di, si, stn]
        ts, hs = np.clip(temp / 35.0, 0, 1), np.clip(hum / 100.0, 0, 1)
        stress = (ts + hs) / 2.0
        n = np.arange(1, len(g) + 1)
        out.append(pd.DataFrame({"temp_score": ts, "hum_score": hs, "stress_mean": np.cumsum(stress) / n,
                                 "stress_cum": np.cumsum(stress) / 200.0}, index=g.index))
    return pd.concat([df, pd.concat(out)[WX_COLS]], axis=1)


def alert_stats(unit, cycle, rul_true_, alarm) -> dict:
    d = pd.DataFrame({"unit": unit, "cycle": cycle, "danger": np.asarray(rul_true_) <= DANGER_RUL, "alarm": np.asarray(alarm)})
    fp, fn = d.alarm & ~d.danger, ~d.alarm & d.danger
    return {"val_FP_cycles": int(fp.sum()), "val_FN_cycles": int(fn.sum()),
            "val_engines_with_false_alarm": int((d[~d.danger].groupby("unit")["alarm"].sum() > 0).sum())}


summary = []
for wx_seed in WX_SEEDS:
    t0 = time.time()
    cols = sensors + WX_COLS
    tr = add_weather(norm.transform(fit_rows), "train", wx_seed)
    va = add_weather(norm.transform(train_raw[train_raw["unit"].isin(val_units)]), "val", wx_seed)
    te = add_weather(norm.transform(test_raw), "test", wx_seed)
    tr_s, tr_l, _ = engine_sequences(tr, cols)
    va_s, va_l, va_ids = engine_sequences(va, cols)
    te_s, _, te_ids = engine_sequences(te, cols, has_label=False)
    model = train_recurrent(RULGRU, tr_s, tr_l, len(cols), seed=MODEL_SEED)
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
        a, d = (t["pred"] <= th).values, (t["RUL"] <= DANGER_RUL).values
        res[f"test_FN_at_{th}"], res[f"test_FP_at_{th}"] = int((~a & d).sum()), int((a & ~d).sum())
    res["test_MAE"] = round(float(np.mean(np.abs(t["RUL"] - t["pred"]))), 2)
    res["weather_seed"], res["seconds"] = wx_seed, round(time.time() - t0)
    with open(f"{OUT_METRICS}/weather_gru_{wx_seed}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    summary.append(res)
    print(json.dumps(res, ensure_ascii=False), flush=True)

with open(f"{OUT_METRICS}/weather_gru_summary.json", "w", encoding="utf-8") as f:
    json.dump({"assumption": "엔진마다 시작 날짜·기준 공항을 무작위 배정, 하루 5회, 실제 AMOS 기온·습도 부착", "runs": summary}, f, ensure_ascii=False, indent=2)
