"""
04_train_lstm.py
==================
모델 2: LSTM Sequence-to-Sequence 회귀 (딥러닝 모델)

참고자료 활용:
  MathWorks "Sequence-to-Sequence Regression Using Deep Learning"
  (https://www.mathworks.com/help/deeplearning/ug/sequence-to-sequence-regression-using-deep-learning.html)
  이 문서는 우리와 완전히 동일한 문제(C-MAPSS 터보팬 엔진, 센서 시계열 → RUL)를
  다루는 공식 튜토리얼이라, 구조를 최대한 그대로 따르되 MATLAB Deep Learning
  Toolbox 전용 기능은 PyTorch 방식으로 바꿔서 구현했습니다. 아래에 "원문 방식"과
  "우리가 실제로 구현한 방식", 그리고 "왜 다르게 했는지"를 항목별로 남깁니다.

  | 항목 | 원문(MathWorks, MATLAB) | 우리 구현(PyTorch) | 이유 |
  |---|---|---|---|
  | RUL 클리핑 | 150 | 125 | FD001 최소 엔진수명(128)에 맞춤 (common.py 참고) |
  | 정규화 | Z-score (train 통계) | 동일 (common.py Normalizer) | 원문 그대로 |
  | 가변 길이 시퀀스 처리 | left-padding | right-padding + pack_padded_sequence | PyTorch의 표준적인 방식이며 효과(패딩이 학습에 영향 안 줌)는 동일 |
  | LSTM hidden units | 200 | 200 | 원문 그대로 |
  | FC 레이어 | 50 units | 50 units | 원문 그대로 |
  | Dropout | 0.5 | 0.5 | 원문 그대로 |
  | Optimizer | Adam | Adam | 원문 그대로 |
  | Gradient threshold | 1 | 1 (clip_grad_norm_) | 원문 그대로 |
  | Epoch | 80 | 80 | 원문 그대로 (아래 실제 학습 속도를 보고 조정될 수 있음) |
  | Mini-batch size | 20 | 20 | 원문 그대로 |
  | 타깃 정규화 | symmetric rescaling(자동) | RUL을 0~1로 스케일(÷125) 후 예측 시 다시 ×125 | 개념은 동일(정답 스케일을 작게 만들어 학습 안정화), 구현만 단순화 |
  | 평가 | 부분 시퀀스 마지막 시점 RMSE | 동일 + validation은 전체 궤적 매 시점 평가 | RF/베이스라인과 같은 평가 규약 유지(공정 비교) |
"""

import sys, json, time
sys.path.insert(0, "/home/claude/sentinel_project/scripts")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from common import (
    load_raw, find_constant_columns, add_rul_labels, split_engines,
    Normalizer, SENSOR_COLS_RAW, RUL_CLIP_VALUE, mae, rmse, nasa_score,
)

DATA = "/home/claude/sentinel_project/data"
OUT_METRICS = "/home/claude/sentinel_project/outputs/metrics"
OUT_MODELS = "/home/claude/sentinel_project/outputs/models"

torch.manual_seed(42)  # 재현성을 위한 시드 고정 (RF와 마찬가지로 42 사용 — 팀 전체 관례)

# ---------------------------------------------------------------------------
# 1. 데이터 준비 (RF와 동일한 엔진 분할/정규화 재사용)
# ---------------------------------------------------------------------------
train_raw = load_raw(f"{DATA}/train_FD001.txt")
train_raw = add_rul_labels(train_raw)
train_units, val_units = split_engines(train_raw)

const_cols = find_constant_columns(train_raw[train_raw["unit"].isin(train_units)])
active_sensors = [c for c in SENSOR_COLS_RAW if c not in const_cols]

norm = Normalizer().fit(train_raw[train_raw["unit"].isin(train_units)], active_sensors)
train_df = norm.transform(train_raw[train_raw["unit"].isin(train_units)])
val_df = norm.transform(train_raw[train_raw["unit"].isin(val_units)])


def engine_sequences(df: pd.DataFrame, sensor_cols: list, has_label: bool = True):
    """엔진별로 (센서 시퀀스, RUL 시퀀스, 길이)를 뽑아 리스트로 반환."""
    seqs, labels, units = [], [], []
    for uid, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        seqs.append(torch.tensor(g[sensor_cols].values, dtype=torch.float32))
        if has_label:
            labels.append(torch.tensor(g["RUL"].values / RUL_CLIP_VALUE, dtype=torch.float32))
        units.append(uid)
    return seqs, labels, units


train_seqs, train_labels, train_ids = engine_sequences(train_df, active_sensors)
val_seqs, val_labels, val_ids = engine_sequences(val_df, active_sensors)

print(f"train 엔진(시퀀스) 수: {len(train_seqs)}, validation 엔진 수: {len(val_seqs)}")
print(f"입력 피처 차원(센서 개수): {len(active_sensors)}")


# ---------------------------------------------------------------------------
# 2. 모델 정의
# ---------------------------------------------------------------------------
class RULLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=200, fc_size=50, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_size, 1)

    def forward(self, x_padded, lengths):
        packed = pack_padded_sequence(x_padded, lengths, batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.fc2(out)  # (batch, seq_len, 1)
        return out.squeeze(-1)


model = RULLSTM(n_features=len(active_sensors))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss(reduction="none")

# ---------------------------------------------------------------------------
# 3. 학습 루프
# ---------------------------------------------------------------------------
BATCH_SIZE = 20
EPOCHS = 80
GRAD_CLIP = 1.0

n_train = len(train_seqs)
print(f"\n학습 시작 (batch={BATCH_SIZE}, epochs={EPOCHS})")
t_start = time.time()
epoch_losses = []
rng = np.random.default_rng(42)

for epoch in range(1, EPOCHS + 1):
    order = rng.permutation(n_train)  # "every-epoch" 셔플: 에폭마다 엔진 순서를 섞음
    total_loss, total_count = 0.0, 0
    model.train()
    for start in range(0, n_train, BATCH_SIZE):
        idx = order[start : start + BATCH_SIZE]
        batch_seqs = [train_seqs[i] for i in idx]
        batch_labels = [train_labels[i] for i in idx]
        lengths = torch.tensor([len(s) for s in batch_seqs])

        x_padded = pad_sequence(batch_seqs, batch_first=True)  # (batch, max_len, n_feat)
        y_padded = pad_sequence(batch_labels, batch_first=True)  # (batch, max_len)

        # 패딩된 위치는 길이(lengths) 밖이므로 마스크로 제외하고 loss 계산
        max_len = x_padded.shape[1]
        mask = (torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)).float()

        optimizer.zero_grad()
        pred = model(x_padded, lengths)
        loss_elem = loss_fn(pred, y_padded) * mask
        loss = loss_elem.sum() / mask.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item() * mask.sum().item()
        total_count += mask.sum().item()

    epoch_mse = total_loss / total_count
    epoch_losses.append(epoch_mse)
    if epoch % 10 == 0 or epoch == 1:
        print(f"  epoch {epoch:3d}/{EPOCHS}  train MSE(정규화 스케일)={epoch_mse:.5f}")

print(f"학습 완료. 소요 시간: {time.time()-t_start:.1f}초")

# ---------------------------------------------------------------------------
# 4. 평가용 예측 함수 (엔진 1개씩, 패딩 없이 그대로 넣어서 매 시점 예측 받기)
# ---------------------------------------------------------------------------
model.eval()


@torch.no_grad()
def predict_sequence(seq: torch.Tensor) -> np.ndarray:
    x = seq.unsqueeze(0)  # (1, seq_len, n_feat)
    length = torch.tensor([len(seq)])
    pred_norm = model(x, length).squeeze(0).numpy()
    pred = np.clip(pred_norm * RUL_CLIP_VALUE, 0, RUL_CLIP_VALUE)
    return pred


# ---------------------------------------------------------------------------
# 5. validation 평가 (전체 궤적 매 시점 — baseline/RF와 동일 규약)
# ---------------------------------------------------------------------------
val_rows = []
for uid, seq, lab in zip(val_ids, val_seqs, val_labels):
    pred = predict_sequence(seq)
    true = (lab.numpy() * RUL_CLIP_VALUE)
    cycles = np.arange(1, len(seq) + 1)
    for c, t_, p_ in zip(cycles, true, pred):
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
val_out.to_csv(f"{OUT_METRICS}/lstm_val_predictions.csv", index=False)

# ---------------------------------------------------------------------------
# 6. 공식 test 세트 평가 (부분 시퀀스 → 마지막 시점 예측만 사용)
# ---------------------------------------------------------------------------
test_raw = load_raw(f"{DATA}/test_FD001.txt")
rul_true_file = pd.read_csv(f"{DATA}/RUL_FD001.txt", header=None, names=["RUL"])
rul_true_file["unit"] = np.arange(1, len(rul_true_file) + 1)
test_norm = norm.transform(test_raw)

test_seqs, _, test_ids = engine_sequences(test_norm, active_sensors, has_label=False)

test_rows = []
for uid, seq in zip(test_ids, test_seqs):
    pred = predict_sequence(seq)
    test_rows.append({"unit": uid, "cycle": len(seq), "RUL_pred": pred[-1]})

test_out = pd.DataFrame(test_rows).merge(rul_true_file, on="unit").rename(columns={"RUL": "RUL_true"})
test_metrics = {
    "MAE": mae(test_out["RUL_true"], test_out["RUL_pred"]),
    "RMSE": rmse(test_out["RUL_true"], test_out["RUL_pred"]),
    "NASA_score": nasa_score(test_out["RUL_true"], test_out["RUL_pred"]),
}
print("\n[공식 test 세트 성능 (최종 블라인드 평가)]")
for k, v in test_metrics.items():
    print(f"  {k}: {v:.2f}")
test_out.to_csv(f"{OUT_METRICS}/lstm_test_predictions.csv", index=False)

torch.save(
    {"model_state": model.state_dict(), "active_sensors": active_sensors,
     "hidden_size": 200, "fc_size": 50, "dropout": 0.5},
    f"{OUT_MODELS}/lstm_model.pt",
)

with open(f"{OUT_METRICS}/lstm_metrics.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "model": "lstm_seq2seq",
            "description": "MathWorks 예제 구조 기반 LSTM(hidden=200,fc=50,dropout=0.5), epochs=80,batch=20,Adam,gradclip=1",
            "epoch_losses_normalized_mse": epoch_losses,
            "validation": val_metrics,
            "official_test": test_metrics,
        },
        f, ensure_ascii=False, indent=2,
    )
print(f"\n저장 완료: {OUT_METRICS}/lstm_metrics.json, {OUT_MODELS}/lstm_model.pt")
