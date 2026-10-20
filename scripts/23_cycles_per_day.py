"""
23_cycles_per_day.py
====================
"우리나라 항공사는 항공기 1대가 하루에 몇 번 비행하는가?"를 공공데이터로 추정합니다.

왜 필요한가: 위험 기준 40(13번)은 논문의 "항공기는 하루 1회 비행(1일 = 1사이클)" 가정으로 정했습니다.
국내 LCC처럼 하루에 여러 번 뜨면 같은 정비 준비 기간이 사이클로는 몇 배라서 기준이 달라집니다 (24번).

계산 (2025년 한 해, 여객기만)
  비행 1회(사이클) 수 = 국내선 출발 편수 + 국제선 출발 편수 + 국제선 도착 편수
      (국내선은 한 번의 비행이 출발·도착 한 쌍이라 출발만 세고, 국제선은 한국 쪽 출발과 도착이 각각 한 번의 비행)
      = 한국공항공사(인천 제외 14개 공항) + 인천공항공사(API)
  하루 비행 횟수(1대당) = 연간 비행 횟수 ÷ (보유 여객기 대수 × 365)

한계 (숫자를 말할 때 함께 말할 것)
  - 보유 대수는 예비 기재·정비 중인 기체를 포함 → 실제 운항 기체 1대당 횟수는 이보다 높을 수 있음
  - 보유 대수는 연초 기준 하나의 값 (연중 기재 변동 미반영)
  - 한국을 거치지 않는 제3국 간 운항은 빠짐 → 약간 낮게 잡힘
  - 티웨이항공은 2025년 자료에서 "트리니티항공(TW)"으로도 표기돼 같은 항공사로 합침
  - "사이클 = 비행 1회"는 논문의 가정이며 C-MAPSS의 사이클 길이가 실제 비행 1회와 같다고 확인된 것은 아님

입력: data/kr_public/ (README에 출처와 내려받은 날짜)
실행: python scripts/23_cycles_per_day.py            (저장된 파일로 계산)
      DATA_GO_KR_KEY='인증키' python scripts/23_cycles_per_day.py --fetch-icn   (인천 API를 다시 받아서 계산)
결과: outputs/kr_public/kr_cycles_per_day.csv, kr_cycles_per_day.png
"""

import json
import os
import sys
import urllib.request

import pandas as pd
import matplotlib.pyplot as plt

from common import PROJECT_ROOT

IN = PROJECT_ROOT / "data" / "kr_public"
OUT = PROJECT_ROOT / "outputs" / "kr_public"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["figure.dpi"] = 110

AIRLINES = ["제주항공", "진에어", "티웨이항공", "에어부산", "이스타항공", "에어서울", "에어로케이", "에어프레미아", "대한항공", "아시아나항공"]
LCC = {"제주항공", "진에어", "티웨이항공", "에어부산", "이스타항공", "에어서울", "에어로케이", "에어프레미아"}
# 한국공항공사 표기(코드 포함) → 표준 이름
KAC_NAME = {"대한항공(KAL)": "대한항공", "아시아나항공(AAR)": "아시아나항공", "에어부산(ABL)": "에어부산", "에어서울(ASV)": "에어서울",
            "이스타항공(ESR)": "이스타항공", "제주항공(JJA)": "제주항공", "진에어(JNA)": "진에어", "트리니티항공(TWB)": "티웨이항공",
            "에어로케이(EOK)": "에어로케이"}
# 인천공항공사 표기가 다른 항공사
ICN_ALIAS = {"티웨이항공": ["티웨이항공", "트리니티항공"], "에어로케이": ["에어로케이", "에어로케이항공"]}


def fetch_icn() -> None:
    """인천공항공사 API를 다시 받아 JSON으로 저장 (인증키는 환경변수로만 받음)."""
    key = os.environ.get("DATA_GO_KR_KEY")
    if not key:
        raise SystemExit("환경변수 DATA_GO_KR_KEY(공공데이터포털 인증키)가 필요합니다.")
    for flag, name in (("I", "intl"), ("D", "dom")):
        q = f"serviceKey={key}&from_month=202501&to_month=202512&pax_cargo=Y&domestic_foreign={flag}&type=json"
        url = "https://apis.data.go.kr/B551177/AviationStatsByAirline/getTotalNumberOfFlight?" + q
        raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read().decode("utf-8")
        j = json.loads(raw)
        if j["response"]["header"]["resultCode"] != "00":
            raise SystemExit(f"API 오류: {j['response']['header']}")
        (IN / f"icn_airline_flights_2025_{name}.json").write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding="utf-8")
        print("저장:", f"icn_airline_flights_2025_{name}.json")


def load_icn(name: str) -> pd.DataFrame:
    j = json.loads((IN / f"icn_airline_flights_2025_{name}.json").read_text(encoding="utf-8"))
    items = j["response"]["body"]["items"]
    df = pd.DataFrame(items["item"] if isinstance(items, dict) else items)
    for c in ("arrFlights", "depFlights", "flights"):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""))  # "39,343" 형태
    return df


if "--fetch-icn" in sys.argv:
    fetch_icn()

kac = pd.read_csv(IN / "kac_aircraft_type_traffic_2025.csv", encoding="utf-8-sig")
kac.columns = ["year", "month", "airport", "dom_intl", "airline", "sched", "usage", "type", "dir", "flights", "pax", "cargo"]
kac["al"] = kac["airline"].map(KAC_NAME)
kac = kac[kac["al"].notna()]
kac_sum = kac.groupby(["al", "dom_intl", "dir"])["flights"].sum()

fleet = pd.read_csv(IN / "kac_fleet_by_type.csv", encoding="utf-8-sig")
fleet_pax = fleet[fleet["구분"] == "여객기"].drop(columns=["구분", "기종별"]).sum()     # 여객기만
pilots = pd.read_csv(IN / "molit_airline_pilots_fleet.csv", encoding="utf-8-sig").set_index("구분")["항공기대수"]

icn_i, icn_d = load_icn("intl"), load_icn("dom")

rows = []
for al in AIRLINES:
    g = lambda dom, d: float(kac_sum.get((al, dom, d), 0.0))
    names = ICN_ALIAS.get(al, [al])
    ri, rd_ = icn_i[icn_i["airline"].isin(names)], icn_d[icn_d["airline"].isin(names)]
    dom = g("국내선", "출발") + float(rd_["depFlights"].sum())
    intl = g("국제선", "출발") + g("국제선", "도착") + float(ri["depFlights"].sum()) + float(ri["arrFlights"].sum())
    total = dom + intl
    n_fleet, n_pilot = float(fleet_pax.get(al, float("nan"))), float(pilots.get(al, float("nan")))
    rows.append({"항공사": al, "구분": "LCC" if al in LCC else "대형항공사", "연간_국내선_비행수": dom, "연간_국제선_비행수": intl, "연간_비행수": total,
                 "국내선_비중": round(dom / total, 3), "보유_여객기": n_fleet, "대수_조종사현황": n_pilot,
                 "하루_비행수_대당": round(total / (n_fleet * 365), 2), "하루_비행수_대당_조종사현황대수": round(total / (n_pilot * 365), 2)})
res = pd.DataFrame(rows)
res.to_csv(OUT / "kr_cycles_per_day.csv", index=False, encoding="utf-8-sig")
print(res[["항공사", "구분", "연간_비행수", "보유_여객기", "하루_비행수_대당", "하루_비행수_대당_조종사현황대수"]].to_string(index=False))

lcc = res[(res["구분"] == "LCC") & (res["항공사"] != "에어프레미아")]["하루_비행수_대당"]
print(f"\n국내 LCC(에어프레미아 제외) 하루 비행 수: 최소 {lcc.min()}, 최대 {lcc.max()}, 평균 {lcc.mean():.1f}")

# ---------------------------------------------------------------------------
# 그림
# ---------------------------------------------------------------------------
order = res.sort_values("하루_비행수_대당")
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#05507d" if c == "LCC" else "#8b9190" for c in order["구분"]]
ax.barh(order["항공사"], order["하루_비행수_대당"], color=colors)
for y, v in enumerate(order["하루_비행수_대당"]):
    ax.text(v + 0.05, y, f"{v:.1f}", va="center", fontsize=9)
ax.axvline(1, color="crimson", linestyle="--", linewidth=1.2, label="논문 가정: 하루 1회")
ax.legend(loc="lower right")
ax.set_xlabel("항공기 1대당 하루 비행 횟수 (2025년, 보유 여객기 기준 평균)")
ax.set_title("국내 항공사의 하루 비행 횟수 추정\n(파랑 = LCC, 회색 = 대형항공사)", fontsize=11)
fig.text(0.01, 0.01, "출처: 한국공항공사·인천공항공사·국토교통부(공공데이터포털). 예비 기재 포함 대수 기준이라 실제 운항 기체는 더 높을 수 있음", fontsize=7, color="gray")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / "kr_cycles_per_day.png")
plt.close(fig)
print("저장:", OUT / "kr_cycles_per_day.csv", OUT / "kr_cycles_per_day.png")
