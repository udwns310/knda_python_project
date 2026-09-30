"""
12_seed_check.py
================
11번 모델 비교에서 GRU가 LSTM보다 좋게 나왔는데, 딥러닝은 시작 가중치(시드)에 따라 결과가
흔들리므로 "한 번의 결과"만으로 결론 내지 않기 위한 확인 실험입니다.
LSTM과 GRU를 시드 3개(42, 1, 2)로 각각 학습해 공식 test 성능의 평균과 흔들림을 비교합니다.
(구조·학습 설정·데이터 분할은 04·11번과 동일, 바뀌는 것은 시드뿐)

실행: python scripts/12_seed_check.py   (FD001 기준 약 20분, CMAPSS_DATASET으로 변경 가능)
결과: outputs/metrics/seed_check.csv
"""

import time

import numpy as np
import pandas as pd

from common import (
    load_raw, find_constant_columns, add_rul_labels, split_engines, Normalizer, SENSOR_COLS_RAW,
    mae, rmse, nasa_score, DATASET, TRAIN_FILE, TEST_FILE, RUL_FILE, OUT_METRICS,
)
from seq_models import engine_sequences, train_recurrent, predict_recurrent, RULLSTM, RULGRU

SEEDS = [42, 1, 2]

train_raw = add_rul_labels(load_raw(TRAIN_FILE))
train_units, _ = split_engines(train_raw)
tr = train_raw[train_raw["unit"].isin(train_units)]
sensors = [c for c in SENSOR_COLS_RAW if c not in find_constant_columns(tr)]
norm = Normalizer().fit(tr, sensors)
seqs, labels, _ = engine_sequences(norm.transform(tr), sensors)
test_seqs, _, test_ids = engine_sequences(norm.transform(load_raw(TEST_FILE)), sensors, has_label=False)
rul = pd.read_csv(RUL_FILE, header=None, names=["RUL"])
y = rul["RUL"].to_numpy()  # test 엔진 1..N 순서 = test_ids 순서(정렬됨)

rows = []
for name, cls in (("LSTM", RULLSTM), ("GRU", RULGRU)):
    for seed in SEEDS:
        t0 = time.time()
        model = train_recurrent(cls, seqs, labels, len(sensors), seed=seed, log=lambda *_: None)
        p = np.array([predict_recurrent(model, s)[-1] for s in test_seqs])
        rows.append({"model": name, "seed": seed, "test_MAE": round(mae(y, p), 2),
                     "test_RMSE": round(rmse(y, p), 2), "test_NASA": round(nasa_score(y, p), 1),
                     "seconds": round(time.time() - t0)})
        print(rows[-1])

res = pd.DataFrame(rows)
res.to_csv(f"{OUT_METRICS}/seed_check.csv", index=False, encoding="utf-8-sig")
summary = res.groupby("model")[["test_MAE", "test_RMSE", "test_NASA"]].agg(["mean", "std"]).round(2)
print(f"\n[{DATASET}] 시드 {SEEDS} 평균 ± 표준편차")
print(summary.to_string())
