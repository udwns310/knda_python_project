"""
27_kr_airport_weather.py
=========================
한국 민간공항 7곳의 기온·습도를 기상청 API허브 공항기상관측(AMOS, 분 자료)에서 받아 월별로 요약합니다.
"우리나라 환경(고온·다습)은 어떤가"를 보여 주는 설명 자료용이며, C-MAPSS 엔진 데이터에는 날짜·공항 정보가 없어
엔진 예측에 직접 연결하지는 못합니다 (연결하려면 항공기별 비행 일시·공항 기록이 필요).

호출 한계: 분 자료는 1회 최대 60분 → 매월 15일의 02·08·14·20시 한 시간씩(분 단위 60개, 7개 지점)을 표본으로 사용합니다.
연간 전체가 아니라 표본이며, 하루 중 시간대가 고르게 들어가도록 4개 시각을 골랐습니다.
지점코드(92, 110, 113, 151, 163, 167, 182)는 API가 돌려주는 값 그대로이며, 공항 이름과의 대응은 확인하지 않아 공항별로
나누지 않고 7곳을 합쳐서 봅니다.

인증키는 환경변수 KMA_AUTHKEY로만 넣습니다 (파일·코드·레포에 넣지 마세요):
    KMA_AUTHKEY='발급받은_키' python scripts/27_kr_airport_weather.py
결과: outputs/kr_public/airport_weather_monthly.csv, airport_weather_summary.json
"""

import io
import json
import os
import urllib.request

import numpy as np
import pandas as pd

from common import PROJECT_ROOT

KEY = os.environ.get("KMA_AUTHKEY")
if not KEY:
    raise SystemExit("환경변수 KMA_AUTHKEY 가 필요합니다.")
OUT = PROJECT_ROOT / "outputs" / "kr_public"
OUT.mkdir(parents=True, exist_ok=True)
YEAR = 2025
COLS = ["stn", "tm", "L_VIS", "R_VIS", "L_RVR", "R_RVR", "CH_MIN", "TA", "TD", "HM", "PS", "PA", "RN", "r1", "r2",
        "WD02", "WD02x", "WD02n", "WS02", "WS02x", "WS02n", "WD10", "WD10x", "WD10n", "WS10", "WS10x", "WS10n"]


def fetch_hour(month: int, hour: int) -> pd.DataFrame:
    # 분 자료: tm = 조회 끝 시각, dtm = 거슬러 올라갈 분 수 (최대 60)
    url = (f"https://apihub.kma.go.kr/api/typ01/url/amos.php?tm={YEAR}{month:02d}15{hour:02d}59&dtm=60&stn=&help=0&authKey={KEY}")
    with urllib.request.urlopen(url, timeout=120) as r:
        text = r.read().decode("cp949", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep=r"\s+", header=None, names=COLS, usecols=range(len(COLS)))
    df["month"], df["hour"] = month, hour
    return df


frames = []
for m in range(1, 13):
    for h in (2, 8, 14, 20):
        try:
            frames.append(fetch_hour(m, h))
        except Exception as e:  # noqa: BLE001
            print(m, h, "실패:", type(e).__name__)
    print(m, "월 완료")
raw = pd.concat(frames, ignore_index=True)
for c in ("TA", "TD", "HM"):
    raw[c] = pd.to_numeric(raw[c], errors="coerce").replace(-99999, np.nan)
raw["temp_c"] = raw["TA"] / 10.0
raw["hum"] = raw["HM"]

monthly = raw.groupby("month").agg(
    temp_mean=("temp_c", "mean"), temp_max=("temp_c", "max"), temp_min=("temp_c", "min"),
    hum_mean=("hum", "mean"), n=("temp_c", "count")).round(1)
monthly.to_csv(OUT / "airport_weather_monthly.csv", encoding="utf-8-sig")
print(monthly.to_string())

stn_mean = raw.groupby("stn").agg(temp_mean=("temp_c", "mean"), hum_mean=("hum", "mean")).round(1)
summary = {
    "year": YEAR, "sample": "매월 15일 02·08·14·20시 각 1시간(분 자료), 공항 7곳 합산", "stations": sorted(raw["stn"].unique().tolist()),
    "temp_max_c": float(raw["temp_c"].max()), "temp_min_c": float(raw["temp_c"].min()),
    "temp_mean_c": round(float(raw["temp_c"].mean()), 1), "hum_mean_pct": round(float(raw["hum"].mean()), 1),
    "share_hum_ge80": round(float((raw["hum"] >= 80).mean()), 3),
    "share_temp_ge30": round(float((raw["temp_c"] >= 30).mean()), 3),
    "summer_hum_mean_pct(6~8월)": round(float(raw[raw["month"].isin([6, 7, 8])]["hum"].mean()), 1),
    "summer_temp_mean_c(6~8월)": round(float(raw[raw["month"].isin([6, 7, 8])]["temp_c"].mean()), 1),
    "winter_temp_mean_c(12~2월)": round(float(raw[raw["month"].isin([12, 1, 2])]["temp_c"].mean()), 1),
}
with open(OUT / "airport_weather_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
