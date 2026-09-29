# Sentinel 파이프라인 실행 가이드 (팀원용)

> 처음 보는 분은 [`EASY_GUIDE.md`](EASY_GUIDE.md)부터 읽어 주세요 — 용어와 개념을 코드 없이 쉽게 설명한 문서입니다.

## 폴더 구조
```
sentinel_project/
  data/                  원본 데이터 (train_FD001.txt, test_FD001.txt, RUL_FD001.txt)
  scripts/
    common.py            모든 스크립트가 공유하는 전처리/평가 함수 + 임의 설정값 근거 주석
    01_eda.py             데이터 이해/탐색 (그림 3장 생성)
    02_baseline.py         나이브 베이스라인
    03_train_rf.py          모델 1: Random Forest
    04_train_lstm.py         모델 2: LSTM
    05_evaluate.py             모델 비교 + 오류 사례 분석
    06_domain_interpretation.py 도메인 해석 (센서 중요도)
    07_export_dashboard.py       대시보드용 데이터 export (JSON)
    08_export_dashboard_ts.py     대시보드 레포가 실제로 읽는 mock.ts 생성
    09_classification.py          [필수 과제] 고장 임박(잔여 ≤ 30) 이진 분류: Z-score 기준 모델 vs RandomForest
  outputs/
    figures/             그래프 이미지 (EDA, 모델비교, 오류사례, 피처중요도)
    metrics/              모델별 성능 수치(json/csv), 예측값(csv)
    models/                학습된 모델 파일 (rf_model.joblib, lstm_model.pt)
    dashboard_data/          대시보드 연동용 최종 산출물 (README.md에 스키마 설명)
  docs/
    EASY_GUIDE.md          처음 보는 사람을 위한 쉬운 설명서 (용어·개념·결과 읽는 법)
    DESIGN_DECISIONS.md    임의 설정값 근거 + 참고자료 활용 내역 (필독)
    PIPELINE_README.md      이 파일
```

## 실행 방법
반드시 번호 순서대로 실행하세요 (뒤 스크립트가 앞 단계 결과를 재사용합니다).
폴더 경로는 `scripts/common.py`가 자동으로 계산하므로, 레포를 어디에 clone했든
어느 폴더에서 실행하든 상관없습니다. (Mac/Linux에서는 `python` 대신 `python3`)
```bash
python scripts/01_eda.py
python scripts/02_baseline.py
python scripts/03_train_rf.py          # 약 4분 소요 (하이퍼파라미터 후보 4개 비교 포함)
python scripts/04_train_lstm.py        # 약 2~3분 소요 (80 epoch 학습)
python scripts/05_evaluate.py
python scripts/06_domain_interpretation.py
python scripts/07_export_dashboard.py
python scripts/08_export_dashboard_ts.py   # 대시보드용 mock.ts 생성 (torch 불필요)
python scripts/09_classification.py        # 필수 과제: 고장 임박 이진 분류 (약 2분, torch 불필요)
```
08번이 만든 `outputs/dashboard_data/mock.ts`를 대시보드 레포의 `src/data/mock.ts`로 복사하면
대시보드에 반영됩니다.
필요 패키지: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `torch`, `joblib`
(전부 `pip install` 로 설치 가능)

`outputs/models/`(학습된 모델 파일)는 용량 때문에 레포에 올리지 않습니다. 03·04번을
실행하면 자동으로 생성되고, 05·07번은 이 파일이 있어야 돌아갑니다.

## 각 사람이 자기 파트를 이어받으려면
- **데이터/센서 관련 질문**: `scripts/common.py` 상단 주석 + `docs/DESIGN_DECISIONS.md`
  1~3번 항목 확인
- **모델 성능/튜닝**: `outputs/metrics/*_metrics.json`에 모델별 상세 수치와
  (RF의 경우) 하이퍼파라미터 탐색 결과가 다 들어있습니다. 값을 바꿔서 재실험하고
  싶으면 해당 스크립트의 "임의 설정값" 표시 부분만 수정하면 됩니다.
- **발표/도메인 해석**: `outputs/figures/feature_importance_top8.png`,
  `outputs/metrics/domain_interpretation.json` 참고
- **오류 사례(한계 설명)**: `outputs/figures/error_case_danger.png`,
  `error_case_conservative.png` + `outputs/metrics/error_case_summary.json`
- **대시보드 연동**: `outputs/dashboard_data/README.md`의 스키마 문서 확인 후 그대로
  가져다 쓰거나 백엔드 API 응답 형식을 맞추면 됩니다.

## 재현성
- 모든 랜덤 요소(train/val 분할, RF, LSTM 초기화)에 `seed=42`를 고정해뒀습니다.
  즉, 위 순서대로 그대로 실행하면 이 문서와 같은 숫자가 나와야 합니다.
