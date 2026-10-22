"""
28_fetch_weather_year.py
=========================
기상청 API허브 공항기상관측(AMOS, 분 자료)에서 2025년 하루 4개 시각(02·08·14·20시, 각 1시간 평균)의 공항 7곳 기온·습도를 받아
outputs/kr_public/airport_weather_2025_slots.csv 로 저장합니다 (29번 실험의 날씨 입력).

분 자료는 1회 최대 60분이라 (365일 × 4시각) = 1,460번 호출합니다. 이미 받은 (날짜, 시각)은 건너뛰어 중간에 끊겨도 이어서 받습니다.
지점코드(92, 110, 113, 151, 163, 167, 182)와 공항 이름의 대응은 확인하지 않아 '지점 A~G'처럼 익명으로 다룹니다.

인증키는 환경변수 KMA_AUTHKEY 로만 넣습니다 (파일·코드·레포에 넣지 마세요):
    KMA_AUTHKEY='발급받은_키' python scripts/28_fetch_weather_year.py
"""

import csv
import datetime as dt
import io
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from common import PROJECT_ROOT

KEY = os.environ.get("KMA_AUTHKEY")
if not KEY:
    raise SystemExit("환경변수 KMA_AUTHKEY 가 필요합니다.")
OUT = PROJECT_ROOT / "outputs" / "kr_public"
OUT.mkdir(parents=True, exist_ok=True)
CSV = OUT / "airport_weather_2025_slots.csv"
SLOTS = (2, 8, 14, 20)
COLS = ["stn", "tm", "L_VIS", "R_VIS", "L_RVR", "R_RVR", "CH_MIN", "TA", "TD", "HM"]


def fetch(day: dt.date, hour: int):
    url = (f"https://apihub.kma.go.kr/api/typ01/url/amos.php?tm={day:%Y%m%d}{hour:02d}59&dtm=60&stn=&help=0&authKey={KEY}")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                text = r.read().decode("cp949", errors="replace")
            lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
            df = pd.read_csv(io.StringIO("\n".join(lines)), sep=r"\s+", header=None, usecols=range(len(COLS)), names=COLS)
            for c in ("TA", "HM"):
                df[c] = pd.to_numeric(df[c], errors="coerce").replace(-99999, np.nan)
            g = df.groupby("stn").agg(temp_c=("TA", lambda s: s.mean() / 10.0), hum=("HM", "mean")).reset_index()
            g["date"], g["slot"] = day.isoformat(), hour
            return g
        except Exception:  # noqa: BLE001
            time.sleep(1.5 * (attempt + 1))
    return None


done = set()
if CSV.exists():
    old = pd.read_csv(CSV)
    done = set(zip(old["date"], old["slot"]))
days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(365)]
todo = [(d, h) for d in days for h in SLOTS if (d.isoformat(), h) not in done]
print("받을 호출:", len(todo), "(이미 받음:", len(done), ")", flush=True)

rows, failed = [], 0
t0 = time.time()
with ThreadPoolExecutor(max_workers=5) as ex:
    for i, res in enumerate(ex.map(lambda a: fetch(*a), todo), 1):
        if res is None:
            failed += 1
        else:
            rows.append(res)
        if i % 100 == 0:
            print(f"  {i}/{len(todo)} 실패 {failed} ({time.time() - t0:.0f}초)", flush=True)
            if rows:
                new = pd.concat(rows, ignore_index=True)
                pd.concat([pd.read_csv(CSV), new]).drop_duplicates(["date", "slot", "stn"]).to_csv(CSV, index=False) if CSV.exists() \
                    else new.to_csv(CSV, index=False)
                rows = []
if rows:
    new = pd.concat(rows, ignore_index=True)
    pd.concat([pd.read_csv(CSV), new]).drop_duplicates(["date", "slot", "stn"]).to_csv(CSV, index=False) if CSV.exists() \
        else new.to_csv(CSV, index=False)
final = pd.read_csv(CSV)
print("완료:", final[["date", "slot"]].drop_duplicates().shape[0], "개 (날짜,시각), 행", len(final), "실패", failed)
