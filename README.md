# Sentinel — 터보팬 엔진 고장 임박 예측 (5조 "감시자들")

포스코 K-뉴딜 아카데미 「설비 고장을 예측하는 AI 데이터 분석」 통합 프로젝트 — **주제 F: 터보팬 엔진 고장 임박 예측**

엔진 센서 기록만 보고 **"지금 고장이 임박했는가(남은 운행 30사이클 이하)"** 를 판별하고(필수 과제),
나아가 **"고장까지 몇 사이클 남았는가(잔존수명, RUL)"** 까지 예측합니다(선택 과제).

> 처음 보는 분은 [`docs/EASY_GUIDE.md`](docs/EASY_GUIDE.md)부터 읽어 주세요 — 코드 없이 용어와 개념을 쉽게 설명한 문서입니다.

---

## 결과 요약 (NASA C-MAPSS FD001)

**필수 과제 — 고장 임박 이진 분류** (위험 = 남은 수명 ≤ 30사이클, validation 엔진 20대의 모든 사이클 기준)

| 방법 | Precision | Recall | F1 |
|---|---|---|---|
| 기준 모델: 이동 Z-score 건강지표 | 0.86 | 0.83 | 0.85 |
| **RandomForest 분류 (경보 임곗값 0.40)** | 0.89 | **0.90** | **0.90** |

**선택 과제 — 잔존수명(RUL) 회귀** (공식 test 엔진 100대, 단위: 사이클)

| 모델 | MAE | RMSE | NASA score |
|---|---|---|---|
| 베이스라인 (평균 수명 − 현재 사이클) | 28.08 | 36.62 | 22,964 |
| Random Forest | 14.19 | 19.87 | 1,276 |
| **LSTM** | **10.85** | **14.85** | **288** |

**확장 실험 — 더 어려운 FD004** (비행 조건 6가지 · 고장 원인 2가지): 분류 F1 RandomForest 0.86 (기준 모델 0.55),
RUL LSTM MAE 18.70 (베이스라인 47.42). → [`docs/FD004_RESULTS.md`](docs/FD004_RESULTS.md)

모든 임의 설정값(위험 기준 30, RUL 상한 125, 롤링 창 15, 임곗값 0.40 등)의 근거는 [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md)에 번호별로 정리했습니다.

---

## 실행 방법

### 1. 환경 준비
Python 3.10 이상에서 확인했습니다 (개발: Python 3.14, Windows 11, CPU).
```bash
pip install -r requirements.txt
```

### 2. 데이터 준비
아래 "데이터 출처와 내려받는 법"을 보고 `data/` 폴더에 `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`를 둡니다.

### 3. 순서대로 실행
뒤 단계가 앞 단계 결과를 쓰므로 **번호 순서대로** 실행합니다. 어느 폴더에서 실행해도 됩니다 (Mac/Linux는 `python3`).
```bash
python scripts/01_eda.py                   # 데이터 탐색 (그림 3장)
python scripts/02_baseline.py              # 베이스라인
python scripts/03_train_rf.py              # RandomForest 회귀 (약 4분)
python scripts/04_train_lstm.py            # LSTM 회귀 (약 4분, torch 필요)
python scripts/05_evaluate.py              # 모델 비교 + 오류 사례
python scripts/06_domain_interpretation.py # 센서 중요도 해석
python scripts/07_export_dashboard.py      # 대시보드용 JSON
python scripts/08_export_dashboard_ts.py   # 대시보드 레포가 읽는 mock.ts
python scripts/09_classification.py        # [필수 과제] 고장 임박 이진 분류 (약 2분)
```
결과는 `outputs/` 아래에 저장됩니다 (그림: `outputs/figures/`, 수치: `outputs/metrics/`).
모든 무작위 요소는 `seed=42`로 고정되어 있어 같은 순서로 실행하면 같은 결과가 나옵니다
(LSTM은 PyTorch 버전에 따라 소수점 수준 차이가 있을 수 있음).

### 다른 서브셋(FD002~FD004)으로 실행
환경변수 하나만 바꾸면 같은 스크립트로 실행되고, 결과는 `outputs/FD004/`처럼 따로 저장됩니다.
```powershell
$env:CMAPSS_DATASET="FD004"    # Git Bash: export CMAPSS_DATASET=FD004
python scripts/01_eda.py       # 이후 02~07, 09 동일 (08번은 FD001 전용)
```
자세한 내용: [`docs/PIPELINE_README.md`](docs/PIPELINE_README.md)

---

## 데이터 출처와 내려받는 법

- **데이터**: NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set (Saxena & Goebel, 2008)
- **내려받기**: [NASA PCoE 데이터 저장소](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)에서
  "Turbofan Engine Degradation Simulation" 항목의 `CMAPSSData.zip`을 받아 압축을 풀고,
  `train_FD00X.txt`, `test_FD00X.txt`, `RUL_FD00X.txt`를 `data/` 폴더에 넣습니다.
- **구성 (FD001)**: 엔진 train 100대(20,631행) · test 100대(13,096행), 열 26개(엔진 번호, 사이클, 운전 설정 3개, 센서 21개), 헤더 없는 공백 구분 텍스트
- **주의**: 실제 항공기 데이터가 아니라 NASA가 물리 모델로 만든 **시뮬레이션 데이터**입니다.
- 센서 이름 대응표(s1 = T2 팬 입구 온도 … s21 = W32 저압터빈 냉각 블리드 유량)는 Saxena et al. (2008) Table 2를 따릅니다 (`scripts/06_domain_interpretation.py`).
- 현재 레포의 `data/`에는 팀 내 편의를 위해 FD001 원본이 들어 있습니다. 과제 제출 규격("원본 데이터는 넣지 않고 내려받는 방법만 기재")에 맞추려면 제출용 폴더에서는 빼 주세요.

---

## 폴더 구조

```
knda_python_project/
├── README.md                ← 이 파일 (실행 방법 · 데이터 출처 · 팀원과 역할)
├── requirements.txt         ← 필요 패키지
├── data/                    ← 원본 데이터 (내려받는 법은 위 참고)
├── scripts/
│   ├── common.py            ← 모든 단계가 공유하는 전처리·평가 함수 + 임의 설정값
│   ├── 01_eda.py            ← 데이터 이해·탐색
│   ├── 02_baseline.py       ← 베이스라인 (센서 미사용)
│   ├── 03_train_rf.py       ← RandomForest 회귀
│   ├── 04_train_lstm.py     ← LSTM 회귀
│   ├── 05_evaluate.py       ← 모델 비교 + 회귀 오류 사례
│   ├── 06_domain_interpretation.py ← 센서 중요도·도메인 해석
│   ├── 07_export_dashboard.py      ← 대시보드용 JSON
│   ├── 08_export_dashboard_ts.py   ← 대시보드 레포용 mock.ts
│   ├── 09_classification.py        ← [필수 과제] 고장 임박 이진 분류
│   ├── 10_cross_dataset.py         ← [추가 실험] FD001 ↔ FD004 교차 검증
│   ├── 11_model_screening.py       ← [추가 실험] 후보 모델 비교
│   ├── 12_seed_check.py            ← [추가 실험] 딥러닝 시드별 흔들림 확인
│   ├── 13_threshold_cost.py        ← 위험 기준(30)의 근거: 리드타임별 정비 비용 시뮬레이션
│   └── seq_models.py               ← 시계열 딥러닝 모델(LSTM·GRU·1D-CNN) 정의
├── outputs/
│   ├── figures/             ← 그림 (EDA, 모델 비교, 혼동행렬, 오탐·미탐 사례, 센서 중요도)
│   ├── metrics/             ← 모델별 성능 수치(json/csv), 예측값(csv)
│   ├── dashboard_data/      ← 대시보드 연동 데이터 (스키마: README.md)
│   ├── models/              ← 학습된 모델 (실행 시 생성, 레포에는 없음)
│   ├── FD004/               ← FD004 확장 실험 결과 (같은 구조)
│   └── cross_dataset/       ← 교차 검증 결과
└── docs/
    ├── EASY_GUIDE.md        ← 처음 보는 사람을 위한 쉬운 설명서
    ├── DESIGN_DECISIONS.md  ← 임의 설정값 근거, 오탐·미탐 분석, 분류 vs 회귀 비교
    ├── PIPELINE_README.md   ← 단계별 실행 가이드 (팀원용)
    └── FD004_RESULTS.md     ← FD004 확장 실험 결과
```

과제 권장 구조(`notebooks/`, `src/`)와 대응: 탐색·시계열 분석 = `01`, 모델링 = `02~04`, `09`, 평가 = `05`, `09`, 전처리·특성 = `common.py`

---

## 분석 흐름

| 단계 | 내용 | 스크립트 |
|---|---|---|
| 문제 정의 | 위험 = 남은 수명 ≤ 30사이클, 성공 기준 = 혼동행렬·Precision·Recall·F1 | 문제정의서 |
| 데이터 이해 | 값이 변하지 않는 센서 7개 + 운전 설정 3개 제거 → 센서 14개 사용 | `01`, `common.py` |
| 전처리 | RUL 라벨(125에서 자름), train 기준 Z-score 정규화, 엔진 단위 80:20 분할 | `common.py` |
| 시계열 분석 | 최근 15사이클 이동평균·이동표준편차·기울기 특성, 추세성 랭킹 | `01`, `common.py` |
| 기준 모델 · ML | 이동 Z-score 기준 모델 → RandomForest 분류 / 베이스라인 → RF·LSTM 회귀 | `02~04`, `09` |
| 평가 · 오류 분석 | 혼동행렬, 임곗값 조정 근거(보전 비용), 오탐(#22)·미탐(#93) 사례 | `05`, `09` |
| 도메인 해석 | 핵심 센서 = 저압터빈 출구 온도(s4), 경보 선행시간 약 30사이클 | `06`, `09` |

---

## 팀원과 역할

| 이름 | 역할 | 담당 단계 |
|---|---|---|
| 최영준 | 팀장 (PM) | 문제 정의, 도메인 해석·대시보드 연동, 문서화·발표 총괄 |
| 한형희 | 데이터 담당 | 데이터 이해, 전처리·기초 분석 |
| 최규인 | 시계열·특성 담당 | 전처리·기초 분석, 시계열 분석·특성 생성 |
| 조민우 | 모델링 담당 | 모델링, 평가·오류 분석 |
| 이문용 | 해석·문서 담당 | 평가·오류 분석, 도메인 해석(비용 시뮬레이션) |

(프로젝트 계획서 기준. 코드 작성은 전원이 함께합니다.)

---

## 관련 저장소

- 시연용 웹 대시보드: [`posco-knda/5_Sentinel_dashboard`](https://github.com/posco-knda/5_Sentinel_dashboard) — `08_export_dashboard_ts.py`가 만든 `mock.ts`를 읽습니다.

## 참고자료

- Saxena, A., Goebel, K., Simon, D., Eklund, N. "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM 2008 — 데이터, 센서 대응표, NASA score
- MathWorks, [Sequence-to-Sequence Regression Using Deep Learning](https://www.mathworks.com/help/deeplearning/ug/sequence-to-sequence-regression-using-deep-learning.html) — RUL 클리핑, LSTM 구조
- MathWorks, [Similarity-Based Remaining Useful Life Estimation](https://www.mathworks.com/help/predmaint/ug/similarity-based-remaining-useful-life-estimation.html) — 추세성 분석, 운전조건별 정규화

참고자료를 어떻게 활용했고 어디를 바꿨는지는 `docs/DESIGN_DECISIONS.md` "참고자료 활용 내역"에 정리했습니다.
