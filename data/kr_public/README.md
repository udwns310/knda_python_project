# 국내 항공 공공데이터 (하루 비행 횟수 추정용)

C-MAPSS는 실제 항공사 데이터가 아니라 시뮬레이션이라, "우리나라 항공사는 항공기 1대가 하루에 몇 번 비행하는가"를
공공데이터로 추정해 위험 기준(사이클)을 한국 운항 패턴에 맞춰 보는 데 씁니다 (`scripts/23_cycles_per_day.py`).
모두 이용허락범위 제한 없음, 무료 자료이며, 이 폴더의 파일은 **2026-10-19에 내려받은 그대로**입니다
(CSV는 원래 CP949 인코딩이라 UTF-8로만 바꿈).

| 파일 | 출처 (공공데이터포털) | 내용 |
|---|---|---|
| `kac_aircraft_type_traffic_2025.csv` | [한국공항공사_국내공항 기종별 항공운송실적](https://www.data.go.kr/data/15132851/fileData.do) (원본 파일명 2025_기종별_운송현황) | 2025년 1~12월, 공항·항공사·기종별 도착/출발 운항(편), 여객, 화물. **인천공항 제외** 14개 공항 |
| `kac_fleet_by_type.csv` | [한국공항공사_항공기 보유현황](https://www.data.go.kr/data/15002708/fileData.do) | 항공사 10곳의 기종별 보유 대수(여객기·화물기) |
| `molit_airline_pilots_fleet.csv` | [국토교통부_국적항공사 조종사 현황](https://www.data.go.kr/data/15092527/fileData.do) (2026-03-31 기준) | 항공사별 항공기 대수와 조종사 수 (보유 대수 비교용) |
| `icn_airline_flights_2025_intl.json`, `icn_airline_flights_2025_dom.json` | [인천국제공항공사_항공사별 항공 통계 서비스(API)](https://www.data.go.kr/data/15095072/openapi.do) | 2025년 1~12월 여객기 항공사별 도착·출발·합계 편수 (국제선, 국내선) |

## 인천공항 API를 다시 받으려면
공공데이터포털에서 위 API를 활용신청하고 받은 인증키를 **환경변수로만** 넣어 실행합니다 (파일·코드·레포에 넣지 마세요).

```bash
DATA_GO_KR_KEY='발급받은_인증키' python scripts/23_cycles_per_day.py --fetch-icn
```

요청 파라미터: `from_month=202501`, `to_month=202512`, `pax_cargo=Y`(여객기), `domestic_foreign=I`(국제선) 또는 `D`(국내선), `type=json`.

## 공항 날씨 (기온·습도)

`scripts/27_kr_airport_weather.py`가 기상청 API허브의 공항기상관측(AMOS, 분 자료)을 호출해 `outputs/kr_public/airport_weather_*.csv/json`으로 요약합니다.
(원본 응답은 저장하지 않음, 매월 15일 02·08·14·20시 표본, 공항 7곳 합산). 인증키는 환경변수로만 넣습니다.

```bash
KMA_AUTHKEY='발급받은_키' python scripts/27_kr_airport_weather.py
```

### 하루 4개 시각 전체 수집과 날씨 점수 시험

`scripts/28_fetch_weather_year.py`는 2025년 매일 02·08·14·20시(각 1시간 평균)의 공항 7곳 기온·습도를 받아 `outputs/kr_public/airport_weather_2025_slots.csv`로 저장합니다(1,460번 호출, 끊겨도 이어서 받음, 지점 이름 대응은 확인하지 않아 익명).
`scripts/29_weather_score_gru.py`는 이 날씨를 점수 4개로 만들어 GRU 입력에 추가한 시험입니다. C-MAPSS에는 비행 날짜·공항 기록이 없어 엔진마다 시작 날짜와 공항을 무작위로 배정한 가정 시험이므로 개선이 나와도 근거로 볼 수 없습니다.

```bash
KMA_AUTHKEY='발급받은_키' python scripts/28_fetch_weather_year.py
python scripts/29_weather_score_gru.py
```
