"""
04_train_gru.py
================
모델 2: GRU Sequence-to-Sequence 회귀 (딥러닝 모델, 최종 회귀 모델)

참고자료 활용:
  MathWorks "Sequence-to-Sequence Regression Using Deep Learning"
  (https://www.mathworks.com/help/deeplearning/ug/sequence-to-sequence-regression-using-deep-learning.html)
  이 문서는 우리와 완전히 동일한 문제(C-MAPSS 터보팬 엔진, 센서 시계열 → RUL)를
  다루는 공식 튜토리얼이라, 구조를 최대한 그대로 따르되 MATLAB Deep Learning
  Toolbox 전용 기능은 PyTorch 방식으로 바꿔서 구현했습니다. 아래에 "원문 방식"과
  "우리가 실제로 구현한 방식", 그리고 "왜 다르게 했는지"를 항목별로 남깁니다.

  | 항목 | 원문(MathWorks, MATLAB) | 우리 구현(PyTorch) | 이유 |
  |---|---|---|---|
  | 기억 장치(순환층) | LSTM | **GRU** | 모델 선정 기준(DESIGN_DECISIONS #14)으로 원문 LSTM과 비교해 교체 — 아래 참고 |
  | RUL 클리핑 | 150 | 125 | 최소 엔진수명(128)에 맞춤 (common.py 참고) |
  | 정규화 | Z-score (train 통계) | 동일 (FD004는 운전조건별, common.py Normalizer) | 원문 그대로 |
  | 가변 길이 시퀀스 처리 | left-padding | right-padding + pack_padded_sequence | PyTorch의 표준적인 방식이며 효과(패딩이 학습에 영향 안 줌)는 동일 |
  | hidden units | 200 | 200 | 원문 그대로 |
  | FC 레이어 | 50 units | 50 units | 원문 그대로 |
  | Dropout | 0.5 | 0.5 | 원문 그대로 |
  | Optimizer | Adam | Adam | 원문 그대로 |
  | Gradient threshold | 1 | 1 (clip_grad_norm_) | 원문 그대로 |
  | Epoch | 80 | 80 | 원문 그대로 (CPU 학습 시간: FD004 약 13분, FD001 약 4분) |
  | Mini-batch size | 20 | 20 | 원문 그대로 |
  | 타깃 정규화 | symmetric rescaling(자동) | RUL을 0~1로 스케일(÷125) 후 예측 시 다시 ×125 | 개념은 동일(정답 스케일을 작게 만들어 학습 안정화), 구현만 단순화 |
  | 평가 | 부분 시퀀스 마지막 시점 RMSE | 동일 + validation은 전체 궤적 매 시점 평가 | RF/베이스라인과 같은 평가 규약 유지(공정 비교) |

  LSTM → GRU로 바꾼 이유:
    GRU는 LSTM의 기억 장치를 더 단순하게(게이트 3개 → 2개) 만든 구조로, 나머지(층 크기·학습 설정)는 원문 그대로입니다.
    11번에서 원문 구조(LSTM)와 같은 조건으로 비교했을 때 GRU가 위험을 덜 놓치고(미탐율 FD004 11.5% vs 13.0%)
    괜한 경보도 적고(헛경보비율 5.3% vs 8.4%) 평균 오차도 작았습니다(validation MAE 7.82 vs 8.06).
    12번에서 시드를 바꿔 3번씩 학습해도 GRU가 매번 앞섰습니다(FD001 test MAE 평균 9.94 vs 11.63).
    선정 기준(미탐 → 헛경보 → MAE 순서)은 docs/DESIGN_DECISIONS.md 임의 설정값 #14.
"""

import json
import time

import numpy as np
import pandas as pd
import torch

from common import (
    load_raw, find_constant_columns, add_rul_labels, split_engines,
    Normalizer, SENSOR_COLS_RAW, RUL_CLIP_VALUE, mae, rmse, nasa_score,
)
# 모델 구조와 학습 루프는 05·07·10·11·12번에서도 똑같이 필요해서 한 곳(seq_models.py)에 정의해 두고
# 가져다 씁니다 (구조가 파일마다 따로 복사돼 있으면 한쪽만 고쳐지는 실수가 생길 수 있음).
# 11번 모델 비교의 LSTM도 같은 함수로 학습하므로, 두 모델은 순환층 종류만 다르고 나머지는 완전히 같습니다.
from seq_models import RULGRU, engine_sequences, train_recurrent, predict_recurrent

# 폴더 경로는 common.py에서 PC에 상관없이 자동으로 계산됩니다 (0번 섹션 참고).
from common import OUT_METRICS, OUT_MODELS, TRAIN_FILE, TEST_FILE, RUL_FILE

SEED = 42  # 재현성을 위한 시드 고정 (RF와 마찬가지로 42 사용 — 팀 전체 관례)

# ---------------------------------------------------------------------------
# 1. 데이터 준비 (RF와 동일한 엔진 분할/정규화 재사용)
# ---------------------------------------------------------------------------
train_raw = load_raw(TRAIN_FILE)
train_raw = add_rul_labels(train_raw)
train_units, val_units = split_engines(train_raw)

const_cols = find_constant_columns(train_raw[train_raw["unit"].isin(train_units)])
active_sensors = [c for c in SENSOR_COLS_RAW if c not in const_cols]

norm = Normalizer().fit(train_raw[train_raw["unit"].isin(train_units)], active_sensors)
train_df = norm.transform(train_raw[train_raw["unit"].isin(train_units)])
val_df = norm.transform(train_raw[train_raw["unit"].isin(val_units)])

# 엔진별로 (센서 시퀀스, RUL/125 시퀀스, 엔진 번호) — 엔진 하나가 시퀀스 하나
train_seqs, train_labels, train_ids = engine_sequences(train_df, active_sensors)
val_seqs, val_labels, val_ids = engine_sequences(val_df, active_sensors)

print(f"train 엔진(시퀀스) 수: {len(train_seqs)}, validation 엔진 수: {len(val_seqs)}")
print(f"입력 피처 차원(센서 개수): {len(active_sensors)}")

# ---------------------------------------------------------------------------
# 2. 모델 정의 + 학습 — seq_models.py의 RULGRU / train_recurrent
#    GRU(hidden 200) → FC 50 → ReLU → Dropout 0.5 → FC 1, 매 사이클마다 RUL을 하나씩 출력
#    학습: 에폭마다 엔진 순서를 섞어 20대씩 미니배치, 패딩 위치는 마스크로 loss에서 제외,
#          Adam(lr 1e-3), gradient clip 1, 80 epoch
# ---------------------------------------------------------------------------
print("\n학습 시작 (batch=20, epochs=80)")
t_start = time.time()
model = train_recurrent(RULGRU, train_seqs, train_labels, len(active_sensors), seed=SEED)
print(f"학습 완료. 소요 시간: {time.time()-t_start:.1f}초")

# ---------------------------------------------------------------------------
# 3. validation 평가 (전체 궤적 매 시점 — baseline/RF와 동일 규약)
# ---------------------------------------------------------------------------
val_rows = []
for uid, seq, lab in zip(val_ids, val_seqs, val_labels):
    pred = predict_recurrent(model, seq)
    true = lab.numpy() * RUL_CLIP_VALUE
    for c, t_, p_ in zip(np.arange(1, len(seq) + 1), true, pred):
        val_rows.append({"unit": uid, "cycle": c, "RUL_true": t_, "RUL_pred": p_})

val_out = pd.DataFrame(val_rows)
val_metrics = {
    "MAE": mae(val_out["RUL_true"], val_out["RUL_pred"]),
    "RMSE": rmse(val_out["RUL_true"], val_out["RUL_pred"]),
    "NASA_score": nasa_score(val_out["RUL_true"], val_out["RUL_pred"]),
}
print("\n[validation 성능]")
for k, v in val_metrics.items():
    print(f"  {k}: {v:.2f}")
val_out.to_csv(f"{OUT_METRICS}/gru_val_predictions.csv", index=False)

# ---------------------------------------------------------------------------
# 4. 공식 test 세트 평가 (부분 시퀀스 → 마지막 시점 예측만 사용)
# ---------------------------------------------------------------------------
test_raw = load_raw(TEST_FILE)
rul_true_file = pd.read_csv(RUL_FILE, header=None, names=["RUL"])
rul_true_file["unit"] = np.arange(1, len(rul_true_file) + 1)
test_seqs, _, test_ids = engine_sequences(norm.transform(test_raw), active_sensors, has_label=False)

test_rows = [{"unit": uid, "cycle": len(seq), "RUL_pred": predict_recurrent(model, seq)[-1]}
             for uid, seq in zip(test_ids, test_seqs)]
test_out = pd.DataFrame(test_rows).merge(rul_true_file, on="unit").rename(columns={"RUL": "RUL_true"})
test_metrics = {
    "MAE": mae(test_out["RUL_true"], test_out["RUL_pred"]),
    "RMSE": rmse(test_out["RUL_true"], test_out["RUL_pred"]),
    "NASA_score": nasa_score(test_out["RUL_true"], test_out["RUL_pred"]),
}
print("\n[공식 test 세트 성능 (최종 블라인드 평가)]")
for k, v in test_metrics.items():
    print(f"  {k}: {v:.2f}")
test_out.to_csv(f"{OUT_METRICS}/gru_test_predictions.csv", index=False)

torch.save(
    {"arch": "GRU", "model_state": model.state_dict(), "active_sensors": active_sensors,
     "hidden_size": 200, "fc_size": 50, "dropout": 0.5},
    f"{OUT_MODELS}/gru_model.pt",
)

with open(f"{OUT_METRICS}/gru_metrics.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "model": "gru_seq2seq",
            "description": "MathWorks 예제 구조에서 순환층만 GRU로 교체(hidden=200,fc=50,dropout=0.5), "
                           "epochs=80,batch=20,Adam,gradclip=1",
            "epoch_losses_normalized_mse": model.epoch_losses,
            "validation": val_metrics,
            "official_test": test_metrics,
        },
        f, ensure_ascii=False, indent=2,
    )
print(f"\n저장 완료: {OUT_METRICS}/gru_metrics.json, {OUT_MODELS}/gru_model.pt")
