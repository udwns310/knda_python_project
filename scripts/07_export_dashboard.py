"""
07_export_dashboard.py
========================
목적: 5조 웹 대시보드(https://github.com/posco-knda/5_Sentinel_dashboard)에서 바로
불러다 쓸 수 있는 형태로 예측 결과를 정리해서 내보냅니다.

※ 안내: 대시보드가 실제로 읽는 파일(mock.ts)은 이 스크립트가 만든 JSON을 바탕으로
08_export_dashboard_ts.py가 생성합니다. 여기서 만드는 JSON과 스키마 문서(README.md)는
나중에 백엔드 API로 바꿀 때 응답 형태의 기준으로도 쓸 수 있습니다.

내보내는 파일 (outputs/dashboard_data/):
  1. model_comparison.json     — 모델별 성능 비교 (대시보드의 "모델 성능" 탭용)
  2. engines_summary.json      — test 100개 엔진 각각의 현재 상태/위험도 요약 (엔진 목록/카드 뷰용)
  3. engine_timeseries/{unit}.json — 엔진 하나를 클릭했을 때 보여줄 상세 시계열 (센서 + RUL 추이)
  4. error_cases.json          — 오류 사례 2건 (대시보드의 "모델 한계" 설명용)
  5. README.md                 — 각 파일의 필드 설명 (스키마 문서)
"""

import sys, json, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from common import (
    load_raw, add_rul_labels, split_engines, Normalizer, RUL_CLIP_VALUE,
    risk_level,  # 위험도 등급 구간 [임의 설정값 #7]은 common.py 7번 섹션에 있습니다.
)

# 폴더 경로는 common.py에서 PC에 상관없이 자동으로 계산됩니다 (0번 섹션 참고).
from common import DATA, OUT_METRICS, OUT_DASH, OUT_MODELS
os.makedirs(f"{OUT_DASH}/engine_timeseries", exist_ok=True)


# ---------------------------------------------------------------------------
# 1. model_comparison.json
# ---------------------------------------------------------------------------
comparison = pd.read_csv(f"{OUT_METRICS}/model_comparison.csv")
comparison.to_json(f"{OUT_DASH}/model_comparison.json", orient="records", force_ascii=False, indent=2)
print(f"저장: {OUT_DASH}/model_comparison.json")

# ---------------------------------------------------------------------------
# 2. LSTM 모델 재로드 (엔진별 상세 시계열 생성용)
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
        out = self.fc2(out)
        return out.squeeze(-1)


ckpt = torch.load(f"{OUT_MODELS}/lstm_model.pt", weights_only=False)
active_sensors = ckpt["active_sensors"]
model = RULLSTM(len(active_sensors), ckpt["hidden_size"], ckpt["fc_size"], ckpt["dropout"])
model.load_state_dict(ckpt["model_state"])
model.eval()

train_raw = load_raw(f"{DATA}/train_FD001.txt")
train_raw = add_rul_labels(train_raw)
train_units, val_units = split_engines(train_raw)
norm = Normalizer().fit(train_raw[train_raw["unit"].isin(train_units)], active_sensors)


@torch.no_grad()
def predict_sequence(seq_tensor):
    x = seq_tensor.unsqueeze(0)
    length = torch.tensor([len(seq_tensor)])
    pred_norm = model(x, length).squeeze(0).numpy()
    return np.clip(pred_norm * RUL_CLIP_VALUE, 0, RUL_CLIP_VALUE)


test_raw = load_raw(f"{DATA}/test_FD001.txt")
test_norm = norm.transform(test_raw)
rul_true_file = pd.read_csv(f"{DATA}/RUL_FD001.txt", header=None, names=["RUL"])
rul_true_file["unit"] = np.arange(1, len(rul_true_file) + 1)

# ---------------------------------------------------------------------------
# 3. engines_summary.json + engine_timeseries/{unit}.json
# ---------------------------------------------------------------------------
summary_rows = []
for uid in sorted(test_raw["unit"].unique()):
    g_norm = test_norm[test_norm["unit"] == uid].sort_values("cycle")
    g_raw = test_raw[test_raw["unit"] == uid].sort_values("cycle")
    seq = torch.tensor(g_norm[active_sensors].values, dtype=torch.float32)
    pred_curve = predict_sequence(seq)
    rul_true_final = float(rul_true_file.loc[rul_true_file.unit == uid, "RUL"].values[0])
    last_cycle = int(g_raw["cycle"].max())
    rul_pred_final = float(pred_curve[-1])

    summary_rows.append({
        "unit": int(uid),
        "last_observed_cycle": last_cycle,
        "predicted_RUL": round(rul_pred_final, 1),
        "true_RUL_for_validation_only": rul_true_final,  # 실무에서는 모르는 값(정답), 데모/검증용으로만 포함
        "risk_level": risk_level(rul_pred_final),
    })

    # 엔진 상세 시계열 (대시보드에서 엔진 하나 클릭했을 때 그래프 그릴 데이터)
    ts = {
        "unit": int(uid),
        "cycles": g_raw["cycle"].tolist(),
        "predicted_RUL_curve": [round(float(v), 2) for v in pred_curve],
        "sensors": {col: g_raw[col].tolist() for col in active_sensors},
    }
    with open(f"{OUT_DASH}/engine_timeseries/{uid}.json", "w", encoding="utf-8") as f:
        json.dump(ts, f, ensure_ascii=False)

summary_df = pd.DataFrame(summary_rows)
with open(f"{OUT_DASH}/engines_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_rows, f, ensure_ascii=False, indent=2)

print(f"저장: {OUT_DASH}/engines_summary.json ({len(summary_rows)}개 엔진)")
print(f"저장: {OUT_DASH}/engine_timeseries/*.json ({len(summary_rows)}개 파일)")
print("\n위험도 등급 분포:")
print(summary_df["risk_level"].value_counts())

# ---------------------------------------------------------------------------
# 4. error_cases.json
# ---------------------------------------------------------------------------
with open(f"{OUT_METRICS}/error_case_summary.json", encoding="utf-8") as f:
    error_cases = json.load(f)
with open(f"{OUT_DASH}/error_cases.json", "w", encoding="utf-8") as f:
    json.dump(error_cases, f, ensure_ascii=False, indent=2)
print(f"저장: {OUT_DASH}/error_cases.json")

print("\n대시보드 데이터 내보내기 완료.")
