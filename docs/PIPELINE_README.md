# Sentinel 파이프라인 실행 가이드 (팀원용)

> 처음 보는 분은 [`EASY_GUIDE.md`](EASY_GUIDE.md)부터 읽어 주세요 — 용어와 개념을 코드 없이 쉽게 설명한 문서입니다.

## 폴더 구조
```
knda_python_project/
  data/                  원본 데이터 (FD004 주, FD001 비교 — train/test/RUL 각 3개 파일)
  scripts/
    common.py                    모든 스크립트가 공유하는 전처리/평가 함수 + 임의 설정값 근거 주석
    seq_models.py                시계열 딥러닝 모델 정의(GRU·LSTM·1D-CNN)와 학습/예측 함수
    01_eda.py                    데이터 이해/탐색 (그림 3장 생성)
    02_baseline.py               [선택 과제] RUL 회귀 기준 모델 (평균 수명 − 현재 사이클)
    03_train_rf.py               [선택 과제] RandomForest 회귀
    04_train_gru.py              [선택 과제] GRU 회귀 (최종 회귀 모델)
    05_evaluate.py               회귀 모델 비교 + 오류 사례 분석
    06_domain_interpretation.py  도메인 해석 (센서 중요도 vs 추세성)
    07_export_dashboard.py       대시보드용 데이터 export (JSON)
    08_export_dashboard_ts.py    대시보드 레포가 실제로 읽는 mock.ts 생성
    09_classification.py         [필수 과제] 고장 임박(잔여 ≤ 40) 이진 분류: Z-score 기준 모델 vs RandomForest
    10_cross_dataset.py          [추가 실험] FD001 ↔ FD004 교차 검증 (한 데이터로 학습 → 다른 데이터에 적용)
    11_model_screening.py        [추가 실험] 후보 모델 비교 (최종 모델 선택 근거)
    12_seed_check.py             [추가 실험] 딥러닝 결과가 시드에 따라 얼마나 흔들리는지 확인
    13_threshold_cost.py         위험 기준 40의 근거: 문헌 운영 조건으로 운행 1사이클당 정비 비용 시뮬레이션
    14_plot_alarm_rates.py       11번 결과를 그림으로: 모델별 MAE·미탐율·오탐율 막대 + MAE 대비 경보 오류율 산점도
  outputs/
    FD004/               주 결과 (아래 4개 폴더)
      figures/             그래프 이미지 (EDA, 모델비교, 오류사례, 혼동행렬, 피처중요도, 정비 비용)
      metrics/             모델별 성능 수치(json/csv), 예측값(csv)
      models/              학습된 모델 파일 (rf_model.joblib, gru_model.pt — 실행 시 생성, 레포에는 없음)
      dashboard_data/      대시보드 연동용 산출물 (mock.ts 등)
    FD001/               비교 결과 (같은 구조)
    cross_dataset/       10번 교차 검증 결과
  docs/
    EASY_GUIDE.md          처음 보는 사람을 위한 쉬운 설명서 (용어·개념·결과 읽는 법)
    FD004_RESULTS.md       주 결과 정리 (분류·회귀·모델 비교·교차 검증·위험 기준)
    DESIGN_DECISIONS.md    임의 설정값 근거 + 참고자료 활용 내역 (필독)
    DASHBOARD_DATA.md      대시보드 데이터(mock.ts, JSON) 형식 설명
    PIPELINE_README.md     이 파일
```

## 실행 방법
반드시 번호 순서대로 실행하세요 (뒤 스크립트가 앞 단계 결과를 재사용합니다).
기본 데이터셋은 **FD004**이고, 결과는 `outputs/FD004/`에 저장됩니다.
폴더 경로는 `scripts/common.py`가 자동으로 계산하므로, 레포를 어디에 clone했든
어느 폴더에서 실행하든 상관없습니다. (Mac/Linux에서는 `python` 대신 `python3`)
```bash
python scripts/01_eda.py
python scripts/02_baseline.py
python scripts/03_train_rf.py              # 약 8분 (하이퍼파라미터 후보 4개 비교 포함)
python scripts/04_train_gru.py             # 약 13분 (80 epoch 학습, CPU 기준)
python scripts/05_evaluate.py
python scripts/06_domain_interpretation.py
python scripts/07_export_dashboard.py
python scripts/08_export_dashboard_ts.py   # 대시보드용 mock.ts 생성 (torch 불필요)
python scripts/09_classification.py        # 필수 과제: 고장 임박 이진 분류 (FD004 약 5분, 위험 기준 민감도 분석 포함)
```
08번이 만든 `outputs/FD004/dashboard_data/mock.ts`를 대시보드 레포의 `src/data/mock.ts`로 복사하면
대시보드에 반영됩니다.
필요 패키지는 루트의 `requirements.txt` 참고 (`pip install -r requirements.txt`).

`outputs/<데이터셋>/models/`(학습된 모델 파일)는 용량 때문에 레포에 올리지 않습니다. 03·04번을
실행하면 자동으로 생성되고, 05·06·07·10번은 이 파일이 있어야 돌아갑니다.

### 추가 실험 (10~14번)
```bash
python scripts/10_cross_dataset.py     # FD001과 FD004 모두 03·04·09번을 먼저 실행해 둬야 함 (약 1분)
python scripts/11_model_screening.py   # 후보 모델 비교 (FD004 약 20분, FD001 약 5분)
python scripts/12_seed_check.py        # GRU·LSTM을 시드 3개로 반복 학습 (GRU 선정 근거) (FD001 약 25분)
python scripts/13_threshold_cost.py    # 위험 기준 40의 근거 — 정비 비용 시뮬레이션 (03·04번 결과 필요, 몇 초)
python scripts/14_plot_alarm_rates.py  # 11번 결과 그림 (몇 초)
```
추가 실험은 최종 모델을 고르고 결과를 검증하기 위한 탐색 단계입니다. 과정 진행 가이드의
"머신러닝 모델은 최대 3종" 규칙은 보고서에 쓰는 최종 모델 기준으로 지킵니다.

## FD001(비교)이나 다른 데이터셋으로 실행하기
환경변수 `CMAPSS_DATASET` 하나만 바꾸면 같은 스크립트로 다른 서브셋을 돌립니다 (기본 FD004).
```powershell
$env:CMAPSS_DATASET="FD001"
python scripts/01_eda.py
# ... 02~09, 11~14 동일
```
Git Bash에서는 `CMAPSS_DATASET=FD001 python scripts/01_eda.py` 처럼 앞에 붙이면 됩니다.
결과는 `outputs/FD001/` 아래에 따로 저장되어 FD004 결과를 덮어쓰지 않습니다. FD001은 데이터가 FD004의 약 1/3이라
03번 약 2분, 04번 약 4분이면 끝납니다. 10번 교차 검증은 두 데이터셋 모두 03·04·09번을 실행해 둬야 합니다.

- FD002·FD003을 쓰려면 [NASA PCoE 데이터 저장소](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)의
  "Turbofan Engine Degradation Simulation" (CMAPSSData.zip)에서 해당 파일 3개를 `data/`에 넣으면 됩니다.
- FD002·FD004는 비행 조건이 6가지로 바뀌는 데이터라, 센서 정규화를 **운전조건별로** 합니다
  (`common.py`의 `assign_regime`, `Normalizer`). 자세한 내용은 `docs/FD004_RESULTS.md`.

## 각 사람이 자기 파트를 이어받으려면
- **데이터/센서 관련 질문**: `scripts/common.py` 상단 주석(센서 이름·단위표 포함) + `docs/DESIGN_DECISIONS.md`
  2장의 임의 설정값 #1~#4
- **모델 성능/튜닝**: `outputs/FD004/metrics/*_metrics.json`에 모델별 상세 수치와
  (RF의 경우) 하이퍼파라미터 탐색 결과가 다 들어있습니다. 값을 바꿔서 재실험하고
  싶으면 해당 스크립트의 "임의 설정값" 표시 부분만 수정하면 됩니다.
- **필수 과제(분류) 결과**: `outputs/FD004/metrics/classification_*.csv/json`,
  `outputs/FD004/figures/cls_*.png` (혼동행렬, 임곗값 근거, 오탐·미탐 사례)
- **위험 기준 40의 근거**: `outputs/FD004/figures/threshold_cost.png`, `outputs/FD004/metrics/threshold_cost_summary.csv`
- **발표/도메인 해석**: `outputs/FD004/figures/feature_importance_top8.png`,
  `outputs/FD004/metrics/domain_interpretation.json` 참고
- **회귀 오류 사례(한계 설명)**: `outputs/FD004/figures/error_case_danger.png`,
  `error_case_conservative.png` + `outputs/FD004/metrics/error_case_summary.json`
- **대시보드 연동**: `docs/DASHBOARD_DATA.md`의 스키마 문서 확인 후 그대로
  가져다 쓰거나 백엔드 API 응답 형식을 맞추면 됩니다.

## 재현성
- 모든 랜덤 요소(train/val 분할, RF, GRU 초기화·배치 순서)에 `seed=42`를 고정해뒀습니다.
  같은 환경에서 위 순서대로 실행하면 같은 숫자가 나옵니다.
- 딥러닝(GRU)은 PyTorch 버전·CPU 종류에 따라 소수점 수준의 차이가 날 수 있습니다.
  시드를 바꿨을 때의 흔들림은 12번 실험 참고.
