"""
seq_models.py
=============
04번(최종 회귀 모델 학습)과 05·07·10·11·12번이 같이 쓰는 시계열 딥러닝 모델 모음입니다.

  - RULLSTM : MathWorks 참고자료 원형 (hidden 200 → FC 50 → 1, dropout 0.5) — 매 사이클마다 RUL 예측.
              11·12번에서 비교 대상으로 사용
  - RULGRU  : LSTM의 기억 장치를 더 단순한 GRU로 바꾼 것 (나머지 동일) — 04번 최종 모델
  - RULCNN  : 최근 30사이클 창(window)을 1차원 합성곱으로 보고 "지금 시점의 RUL" 하나를 예측
              (C-MAPSS 연구에서 많이 쓰는 구조: Babu et al. 2016, Li et al. 2018의 창 방식)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from common import RUL_CLIP_VALUE

BATCH_SIZE, EPOCHS, GRAD_CLIP = 20, 80, 1.0   # MathWorks 예제의 학습 설정 그대로 (04번 참고)
CNN_WINDOW = 30                                # 1D-CNN 입력 창 길이 (test 엔진 최소 관측 31사이클보다 짧게)
CNN_EPOCHS, CNN_BATCH = 30, 256


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
        return self.fc2(out).squeeze(-1)


class RULGRU(RULLSTM):
    def __init__(self, n_features, hidden_size=200, fc_size=50, dropout=0.5):
        super().__init__(n_features, hidden_size, fc_size, dropout)
        self.lstm = nn.GRU(n_features, hidden_size, batch_first=True)  # 같은 자리에 GRU를 끼움


class RULCNN(nn.Module):
    def __init__(self, n_features, window=CNN_WINDOW):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(16 * window, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, x):  # x: (batch, window, n_feat)
        return self.net(x.transpose(1, 2)).squeeze(-1)


# ---------------------------------------------------------------------------
# 데이터 모양 바꾸기
# ---------------------------------------------------------------------------
def engine_sequences(df: pd.DataFrame, sensor_cols: list, has_label: bool = True):
    """엔진별 (센서 시퀀스, RUL/125 시퀀스, 엔진 번호). 엔진 하나가 시퀀스 하나."""
    seqs, labels, units = [], [], []
    for uid, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        seqs.append(torch.tensor(g[sensor_cols].values, dtype=torch.float32))
        if has_label:
            labels.append(torch.tensor(g["RUL"].values / RUL_CLIP_VALUE, dtype=torch.float32))
        units.append(uid)
    return seqs, labels, units


def windows_of(seq: torch.Tensor, window: int = CNN_WINDOW) -> torch.Tensor:
    """시퀀스의 매 시점마다 '최근 window 사이클' 창을 만듭니다. 앞부분이 모자라면 첫 사이클 값으로 채움."""
    pad = seq[:1].repeat(window - 1, 1)
    full = torch.cat([pad, seq], dim=0)
    return full.unfold(0, window, 1).transpose(1, 2)  # (len, window, n_feat)


# ---------------------------------------------------------------------------
# 학습 · 예측
# ---------------------------------------------------------------------------
def train_recurrent(model_cls, seqs, labels, n_features, seed=42, log=print):
    """엔진 단위 미니배치, 패딩 마스크, Adam, gradient clip으로 학습 (04번 설명 참고).
    에폭별 train MSE(정규화 스케일)는 model.epoch_losses에 남겨 둡니다."""
    torch.manual_seed(seed)
    model = model_cls(n_features)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss(reduction="none")
    rng = np.random.default_rng(seed)
    losses = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(seqs))
        tot, cnt = 0.0, 0.0
        for s in range(0, len(seqs), BATCH_SIZE):
            idx = order[s:s + BATCH_SIZE]
            bx = [seqs[i] for i in idx]
            by = [labels[i] for i in idx]
            lengths = torch.tensor([len(x) for x in bx])
            xp, yp = pad_sequence(bx, batch_first=True), pad_sequence(by, batch_first=True)
            mask = (torch.arange(xp.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)).float()
            opt.zero_grad()
            loss = (loss_fn(model(xp, lengths), yp) * mask).sum() / mask.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tot += loss.item() * mask.sum().item()
            cnt += mask.sum().item()
        losses.append(tot / cnt)
        if epoch % 20 == 0:
            log(f"    epoch {epoch}/{EPOCHS} train MSE={tot / cnt:.5f}")
    model.eval()
    model.epoch_losses = losses
    return model


@torch.no_grad()
def predict_recurrent(model, seq: torch.Tensor) -> np.ndarray:
    out = model(seq.unsqueeze(0), torch.tensor([len(seq)])).squeeze(0).numpy()
    return np.clip(out * RUL_CLIP_VALUE, 0, RUL_CLIP_VALUE)


def train_cnn(seqs, labels, n_features, seed=42, log=print):
    torch.manual_seed(seed)
    X = torch.cat([windows_of(s) for s in seqs])
    y = torch.cat(labels)
    model = RULCNN(n_features)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    for epoch in range(1, CNN_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(X), generator=g)
        tot = 0.0
        for s in range(0, len(X), CNN_BATCH):
            idx = perm[s:s + CNN_BATCH]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if epoch % 10 == 0:
            log(f"    epoch {epoch}/{CNN_EPOCHS} train MSE={tot / len(X):.5f}")
    model.eval()
    return model


@torch.no_grad()
def predict_cnn(model, seq: torch.Tensor) -> np.ndarray:
    return np.clip(model(windows_of(seq)).numpy() * RUL_CLIP_VALUE, 0, RUL_CLIP_VALUE)


ARCHS = {"LSTM": RULLSTM, "GRU": RULGRU}


def load_seq_model(path: str):
    """04번에서 저장한 가중치를 불러옵니다 (저장 파일의 arch로 LSTM/GRU 구조를 고름)."""
    ckpt = torch.load(path, weights_only=False)
    model_cls = ARCHS[ckpt.get("arch", "LSTM")]
    model = model_cls(len(ckpt["active_sensors"]), ckpt["hidden_size"], ckpt["fc_size"], ckpt["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["active_sensors"]
