"""
05_evaluate.py
================
선택 과제(RUL 회귀)의 평가 · 오류 분석 단계입니다.

과정 진행 가이드의 평가 규칙은 "오탐 1건 이상, 미탐 1건 이상을 실제 시계열 그래프 위에 표시하고
왜 틀렸는지 서술"입니다. 이 규칙은 필수 과제인 이진 분류에 대해 09번에서 그대로 수행하고,
여기서는 같은 생각을 숫자를 맞히는 회귀 문제에 맞게 옮겨 적용합니다:
  - "위험한 오류" (분류의 미탐에 대응): 모델이 실제보다 RUL을 더 크게 예측
    → "아직 여유 있다"고 착각해서 정비 시점을 놓칠 위험
  - "보수적 오류" (분류의 오탐에 대응): 모델이 실제보다 RUL을 더 작게 예측
    → 아직 괜찮은데 "곧 고장"이라고 너무 일찍 경고

이 스크립트는:
  1. 베이스라인/RF/LSTM 세 모델의 성능을 한 표로 정리
  2. 성능 비교 막대그래프 저장
  3. LSTM 기준으로 공식 test에서 가장 크게 틀린 "위험한 오류" 1건, "보수적 오류" 1건을
     찾아서 시계열 그래프로 그리고, 원인을 그래프 안의 설명 + 출력으로 남김
실행 순서: 02·03·04번 다음
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from common import (
    load_raw, add_rul_labels, split_engines, Normalizer, RUL_CLIP_VALUE, RegimeCorrector,
    MULTI_REGIME, DATASET, SENSOR_UNIT,
)
from seq_models import load_lstm, predict_recurrent

# 폴더 경로는 common.py에서 PC에 상관없이 자동으로 계산됩니다 (0번 섹션 참고).
from common import OUT_METRICS, OUT_FIG, OUT_MODELS, TRAIN_FILE, TEST_FILE, RUL_FILE

# ---------------------------------------------------------------------------
# 1. 세 모델 성능 비교표
# ---------------------------------------------------------------------------
rows = []
for name in ["baseline", "rf", "lstm"]:
    with open(f"{OUT_METRICS}/{name}_metrics.json", encoding="utf-8") as f:
        m = json.load(f)
    rows.append({
        "model": m["model"],
        "val_MAE": m["validation"]["MAE"], "val_RMSE": m["validation"]["RMSE"], "val_NASA": m["validation"]["NASA_score"],
        "test_MAE": m["official_test"]["MAE"], "test_RMSE": m["official_test"]["RMSE"], "test_NASA": m["official_test"]["NASA_score"],
    })
comparison = pd.DataFrame(rows)
comparison.to_csv(f"{OUT_METRICS}/model_comparison.csv", index=False)
print("=" * 70)
print("모델 성능 비교 (validation / 공식 test)")
print("=" * 70)
print(comparison.to_string(index=False))

improve_vs_baseline = (1 - comparison.loc[comparison.model == "lstm_seq2seq", "test_MAE"].values[0] /
                        comparison.loc[comparison.model == "naive_baseline", "test_MAE"].values[0]) * 100
print(f"\nLSTM은 나이브 베이스라인 대비 공식 test MAE를 {improve_vs_baseline:.1f}% 개선했습니다.")

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(comparison))
width = 0.35
ax.bar(x - width / 2, comparison["val_MAE"], width, label="validation MAE")
ax.bar(x + width / 2, comparison["test_MAE"], width, label="공식 test MAE")
ax.set_xticks(x)
ax.set_xticklabels(["나이브 베이스라인", "Random Forest", "LSTM"])
ax.set_ylabel("MAE (사이클)")
ax.set_title(f"모델별 RUL 예측 오차(MAE) 비교 — 낮을수록 좋음 ({DATASET})")
ax.legend()
for i, v in enumerate(comparison["test_MAE"]):
    ax.text(i + width / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT_FIG}/model_comparison_mae.png", dpi=130)
plt.close(fig)
print(f"저장: {OUT_FIG}/model_comparison_mae.png")

# ---------------------------------------------------------------------------
# 2. LSTM 모델 다시 불러오기 (재학습 없이 04번에서 저장한 가중치만 로드)
# ---------------------------------------------------------------------------
model, active_sensors = load_lstm(f"{OUT_MODELS}/lstm_model.pt")

# 정규화기는 랜덤성이 없는(같은 seed=42로 항상 같은 train/val 분할) 결정론적 계산이라
# 동일하게 재계산해도 04번 스크립트와 완전히 같은 결과가 나옵니다.
train_raw = load_raw(TRAIN_FILE)
train_raw = add_rul_labels(train_raw)
train_units, val_units = split_engines(train_raw)
norm = Normalizer().fit(train_raw[train_raw["unit"].isin(train_units)], active_sensors)
# 그래프용: 운전조건 차이를 뺀 센서값 (FD001은 원래값 그대로)
corrector = RegimeCorrector().fit(train_raw[train_raw["unit"].isin(train_units)], active_sensors)


# ---------------------------------------------------------------------------
# 3. test 세트에서 "위험한 오류" / "보수적 오류" 사례 탐색
# ---------------------------------------------------------------------------
test_pred_df = pd.read_csv(f"{OUT_METRICS}/lstm_test_predictions.csv")
test_pred_df["residual"] = test_pred_df["RUL_pred"] - test_pred_df["RUL_true"]

danger_case = test_pred_df.loc[test_pred_df["residual"].idxmax()]  # 예측이 실제보다 훨씬 큼
conservative_case = test_pred_df.loc[test_pred_df["residual"].idxmin()]  # 예측이 실제보다 훨씬 작음

print("\n" + "=" * 70)
print("[오류 사례 분석 — LSTM 모델 기준]")
print("=" * 70)
print(f"위험한 오류 사례: 엔진 #{int(danger_case.unit)} "
      f"(실제 RUL={danger_case.RUL_true:.0f}, 예측={danger_case.RUL_pred:.1f}, "
      f"오차=+{danger_case.residual:.1f} → 실제보다 여유있다고 과대평가)")
print(f"보수적 오류 사례: 엔진 #{int(conservative_case.unit)} "
      f"(실제 RUL={conservative_case.RUL_true:.0f}, 예측={conservative_case.RUL_pred:.1f}, "
      f"오차={conservative_case.residual:.1f} → 실제보다 위험하다고 과소평가)")

test_raw = load_raw(TEST_FILE)
test_norm = norm.transform(test_raw)
rul_true_file = pd.read_csv(RUL_FILE, header=None, names=["RUL"])
rul_true_file["unit"] = np.arange(1, len(rul_true_file) + 1)


def plot_case(uid, title, filename, note):
    g = test_norm[test_norm["unit"] == uid].sort_values("cycle")
    seq = torch.tensor(g[active_sensors].values, dtype=torch.float32)
    pred_curve = predict_recurrent(model, seq)
    rul_final_true = rul_true_file.loc[rul_true_file.unit == uid, "RUL"].values[0]
    cycles = g["cycle"].values
    # 각 관측 시점에서의 '진짜' RUL: 마지막 시점의 정답(RUL 정답 파일)에, 마지막
    # 시점까지 남은 사이클 수를 거슬러 더해서 역산 (piecewise 클리핑 동일 적용)
    true_curve = np.minimum(rul_final_true + (cycles.max() - cycles), RUL_CLIP_VALUE)

    # 위: 추세성이 높은 온도(s4)·압력(s11) 센서 — 단위가 달라 왼쪽/오른쪽 축을 따로 씀
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    raw_g = corrector.transform(test_raw[test_raw["unit"] == uid].sort_values("cycle"))
    suffix = " (운전조건 보정)" if MULTI_REGIME else ""
    axes[0].plot(raw_g["cycle"], raw_g["s4"], color="tab:blue", label="s4 T50 저압터빈 출구 온도")
    axes[0].set_ylabel(f"s4 온도 ({SENSOR_UNIT['s4']}){suffix}", color="tab:blue")
    ax_r = axes[0].twinx()
    ax_r.plot(raw_g["cycle"], raw_g["s11"], color="tab:orange", label="s11 Ps30 고압압축기 출구 정압")
    ax_r.set_ylabel(f"s11 압력 ({SENSOR_UNIT['s11']}){suffix}", color="tab:orange")
    axes[0].legend(handles=axes[0].get_lines() + ax_r.get_lines(), fontsize=8, loc="upper left")
    axes[0].set_title(f"엔진 #{uid} 센서 추이 ({DATASET} test)")

    axes[1].plot(cycles, true_curve, label="실제 RUL", color="black", linewidth=2)
    axes[1].plot(cycles, pred_curve, label="LSTM 예측 RUL", color="tab:red", linestyle="--")
    axes[1].scatter([cycles[-1]], [rul_final_true], color="black", zorder=5)
    axes[1].scatter([cycles[-1]], [pred_curve[-1]], color="tab:red", zorder=5)
    axes[1].set_xlabel("운행 사이클 (cycle)")
    axes[1].set_ylabel("RUL (남은 사이클, cycle)")
    axes[1].legend(fontsize=9)
    axes[1].set_title(title)
    fig.suptitle(note, fontsize=9, wrap=True)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/{filename}", dpi=130)
    plt.close(fig)
    print(f"저장: {OUT_FIG}/{filename}")


plot_case(
    int(danger_case.unit),
    f"위험한 오류 사례 (엔진 #{int(danger_case.unit)}): 실제보다 RUL을 과대예측",
    "error_case_danger.png",
    ("" if DATASET == "FD001" else "[FD001 기준으로 쓴 해석 — 이 데이터셋에서는 그래프를 보고 다시 확인할 것]\n") +
    "원인 추정: 이 엔진은 관측이 끊기는 마지막 시점까지도 s4/s11 등 주요 센서의\n"
    "열화 신호가 아직 뚜렷하게 나타나지 않아, 모델이 '초기 정상 구간'으로 오판했을 가능성이 큽니다.",
)
plot_case(
    int(conservative_case.unit),
    f"보수적 오류 사례 (엔진 #{int(conservative_case.unit)}): 실제보다 RUL을 과소예측",
    "error_case_conservative.png",
    ("" if DATASET == "FD001" else "[FD001 기준으로 쓴 해석 — 이 데이터셋에서는 그래프를 보고 다시 확인할 것]\n") +
    "원인 추정: 이 엔진은 실제 열화 속도가 학습 데이터의 '평균적인 열화 속도'보다\n"
    "느린 개체인데, 모델이 관측된 센서 패턴을 다른(더 빨리 고장난) 엔진들과 비슷하다고\n"
    "판단해 RUL을 실제보다 짧게 예측했을 가능성이 큽니다.",
)

# ---------------------------------------------------------------------------
# 4. 오류 사례 요약을 문서화용 JSON으로도 저장 (docs 작성 시 재사용)
# ---------------------------------------------------------------------------
error_summary = {
    "danger_case": {
        "unit": int(danger_case.unit), "RUL_true": float(danger_case.RUL_true),
        "RUL_pred": float(danger_case.RUL_pred), "residual": float(danger_case.residual),
    },
    "conservative_case": {
        "unit": int(conservative_case.unit), "RUL_true": float(conservative_case.RUL_true),
        "RUL_pred": float(conservative_case.RUL_pred), "residual": float(conservative_case.residual),
    },
}
with open(f"{OUT_METRICS}/error_case_summary.json", "w", encoding="utf-8") as f:
    json.dump(error_summary, f, ensure_ascii=False, indent=2)

print("\n평가 및 오류 분석 완료.")
