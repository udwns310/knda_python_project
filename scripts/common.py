"""
common.py
=========
Sentinel 프로젝트(터보팬 엔진 RUL 예측) 전 과정에서 공통으로 쓰는 함수 모음입니다.

이 파일 하나만 이해하면 나머지 스크립트(02_baseline.py, 03_train_rf.py, ...)를
읽을 때 "이 함수가 뭐하는 거지?"라는 질문 없이 바로 로직을 따라갈 수 있도록
각 함수마다 "무엇을/왜"를 함께 설명해두었습니다.

팀원 안내
--------
- 데이터를 불러오고, RUL(잔존수명) 정답 라벨을 만들고, 모델 입력용 피처를 만드는
  코드는 전부 여기에 있습니다. 모델별 스크립트에서는 이 함수들을 import해서
  씁니다. 같은 전처리를 여러 번 다르게 베껴 쓰다가 실수로 달라지는 것을 막기 위함입니다.
- 아래에서 "임의로 정한 값"이라고 적힌 것들은 전부 docs/DESIGN_DECISIONS.md에
  같은 항목명으로 다시 정리되어 있고, 근거도 그쪽에 자세히 적었습니다.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

# 그래프는 화면에 띄우지 않고 파일로만 저장합니다. 이 설정이 있어야 모니터가 없는 서버나
# 원격 환경에서도 그래프 저장이 오류 없이 동작합니다 (모든 스크립트가 common을 먼저 import).
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402  (backend 설정 뒤에 불러와야 함)

# ---------------------------------------------------------------------------
# 0-1. 어떤 데이터셋을 쓸지 (FD001 / FD002 / FD003 / FD004)
# ---------------------------------------------------------------------------
# 프로젝트의 중심 데이터셋은 FD004(비행 조건 6가지 · 고장 원인 2가지 — 가장 현실에 가까운 서브셋)이고,
# FD001(조건 1가지 · 원인 1가지)은 비교용입니다. 다른 서브셋을 돌리려면 환경변수로 이름만 바꿉니다.
#   PowerShell:  $env:CMAPSS_DATASET="FD001"; python scripts/03_train_rf.py
#   Git Bash:    CMAPSS_DATASET=FD001 python scripts/03_train_rf.py
# 코드 안의 파일 이름(train_FD001.txt 등)과 결과 폴더가 이 값에 맞춰 자동으로 바뀝니다.
DATASET = os.environ.get("CMAPSS_DATASET", "FD004").upper()
if DATASET not in ("FD001", "FD002", "FD003", "FD004"):
    raise SystemExit(f"CMAPSS_DATASET은 FD001~FD004 중 하나여야 합니다 (지금: {DATASET})")
# FD002·FD004는 비행 조건(고도·속도·스로틀)이 6가지로 바뀌며 섞여 있는 "다중 운전조건" 데이터
MULTI_REGIME = DATASET in ("FD002", "FD004")

# ---------------------------------------------------------------------------
# 0-2. 폴더 경로 (모든 스크립트가 여기서 가져다 씀)
# ---------------------------------------------------------------------------
# 경로를 "C:/Users/홍길동/..." 같은 특정 컴퓨터의 절대경로로 적어두면 다른 팀원 PC
# (Windows/Mac/Linux)에서는 실행이 안 됩니다. 그래서 "이 파일(common.py)이 있는 위치"를
# 기준으로 프로젝트 폴더를 계산합니다:
#   common.py 위치 = <프로젝트>/scripts/common.py  →  한 단계 위 = <프로젝트>
# 이렇게 하면 레포를 어디에 clone하든, 어느 폴더에서 실행하든 똑같이 동작합니다.
# 결과 폴더: outputs/FD004/, outputs/FD001/ 처럼 데이터셋별로 따로 저장해서 서로 덮어쓰지 않고
# 나란히 비교할 수 있게 했습니다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = str(PROJECT_ROOT / "data")
TRAIN_FILE = f"{DATA}/train_{DATASET}.txt"
TEST_FILE = f"{DATA}/test_{DATASET}.txt"
RUL_FILE = f"{DATA}/RUL_{DATASET}.txt"


def out_root(dataset: str) -> Path:
    """데이터셋별 결과 폴더 (10번 교차 검증처럼 두 데이터셋을 함께 쓰는 스크립트도 사용)."""
    return PROJECT_ROOT / "outputs" / dataset


_OUT_ROOT = out_root(DATASET)
OUT_FIG = str(_OUT_ROOT / "figures")
OUT_METRICS = str(_OUT_ROOT / "metrics")
OUT_MODELS = str(_OUT_ROOT / "models")
OUT_DASH = str(_OUT_ROOT / "dashboard_data")

if not Path(TRAIN_FILE).exists():
    raise SystemExit(f"{TRAIN_FILE} 파일이 없습니다. NASA PCoE 데이터 저장소에서 CMAPSSData.zip을 받아 "
                     f"train/test/RUL_{DATASET}.txt 세 파일을 data/ 폴더에 넣어 주세요 (README 참고).")

# 결과를 저장할 폴더가 없으면 미리 만들어 둡니다. 특히 models 폴더는 학습된 모델
# 파일(용량이 큼)이라 레포에 올라가 있지 않아서, 처음 clone한 사람은 이 폴더가 없습니다.
for _d in (OUT_FIG, OUT_METRICS, OUT_MODELS, OUT_DASH):
    Path(_d).mkdir(parents=True, exist_ok=True)

# 그래프에 한글(엔진/센서 이름 설명, 제목 등)이 깨지지 않고 나오도록 한글 폰트를
# 지정합니다. 이 설정을 common.py에 한 번만 넣어두면, common.py를 import하는
# 모든 스크립트(01_eda.py, 02_baseline.py, ...)에 자동으로 적용됩니다.
# 팀원마다 운영체제가 달라서, 각 OS에 기본으로 깔려 있는 한글 폰트를 순서대로
# 찾아보고 처음 발견되는 것을 씁니다:
#   Windows → 맑은 고딕(Malgun Gothic), Mac → Apple SD Gothic Neo / AppleGothic,
#   Linux → Noto Sans CJK (한글/일본어/중국어 통합 폰트라 "JP"라는 이름이 붙어
#   있어도 한글이 정상적으로 표시됨), 나눔고딕(NanumGothic)
_KOREAN_FONTS = [
    "Malgun Gothic", "Apple SD Gothic Neo", "AppleGothic",
    "Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic",
]
# Linux에서는 Noto Sans CJK가 설치돼 있어도 matplotlib이 자동으로 못 찾는 경우가
# 있어서, 파일이 있으면 경로를 직접 등록해둡니다 (없으면 그냥 건너뜀).
_CJK_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if _CJK_FONT_PATH.exists():
    fm.fontManager.addfont(str(_CJK_FONT_PATH))
_installed = {f.name for f in fm.fontManager.ttflist}
for _name in _KOREAN_FONTS:
    if _name in _installed:
        matplotlib.rcParams["font.family"] = _name
        break
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# 1. 컬럼 이름 정의
# ---------------------------------------------------------------------------
# C-MAPSS 원본 파일(train/test_FD00X.txt)은 헤더가 없는 공백구분 텍스트 파일입니다.
# 컬럼 순서는 데이터와 함께 배포된 readme.txt 기준 고정 순서입니다.
#   1) unit         : 엔진 번호 (몇 번째 엔진인지)
#   2) cycle        : 그 엔진이 몇 번째 운행 사이클인지 (시간 개념, 1부터 증가)
#   3~5) os1,os2,os3: operational setting 1~3 (그 순간의 운전 조건: 고도/마하수/스로틀 등)
#   6~26) s1~s21    : sensor 1~21 (온도, 압력, 회전수 등 21개 센서 측정값)
ALL_COLS = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
SENSOR_COLS_RAW = [f"s{i}" for i in range(1, 22)]
SETTING_COLS = ["os1", "os2", "os3"]

# 센서 21개의 물리적 의미와 단위 (원자료 기호 · 우리말). 그래프 축 이름과 해석 문서에 씁니다.
# 출처: Saxena, A. et al., "Damage Propagation Modeling for Aircraft Engine Run-to-Failure
#       Simulation", PHM 2008, Table 2 (C-MAPSS 출력 변수 목록 — 데이터 파일의 센서 열 순서와 같음).
# 단위: °R = 랭킨 온도(°F + 459.67), psia = 절대압력(psi), rpm = 분당 회전수, pps = 파운드/초,
#       lbm/s = 질량유량. 빈 문자열은 단위 없는 비율 값입니다.
SENSOR_MEANING = {
    "s1": "T2 · Fan inlet 온도", "s2": "T24 · LPC(저압압축기) outlet 온도",
    "s3": "T30 · HPC(고압압축기) outlet 온도", "s4": "T50 · LPT(저압터빈) outlet 온도",
    "s5": "P2 · Fan inlet 압력", "s6": "P15 · bypass-duct 전압력", "s7": "P30 · HPC outlet 전압력",
    "s8": "Nf · 물리적 팬 속도", "s9": "Nc · 물리적 코어 속도", "s10": "epr · 엔진 압력비(P50/P2)",
    "s11": "Ps30 · HPC outlet 정압", "s12": "phi · 연료 유량/Ps30 비",
    "s13": "NRf · 보정 팬 속도", "s14": "NRc · 보정 코어 속도", "s15": "BPR · 바이패스비",
    "s16": "farB · 연소기 연료-공기비", "s17": "htBleed · 블리드 엔탈피",
    "s18": "Nf_dmd · 요구 팬 속도", "s19": "PCNfR_dmd · 요구 보정 팬 속도",
    "s20": "W31 · HPT(고압터빈) 냉각 블리드 유량", "s21": "W32 · LPT(저압터빈) 냉각 블리드 유량",
}
SENSOR_UNIT = {
    "s1": "°R", "s2": "°R", "s3": "°R", "s4": "°R", "s5": "psia", "s6": "psia", "s7": "psia",
    "s8": "rpm", "s9": "rpm", "s10": "", "s11": "psia", "s12": "pps/psi", "s13": "rpm", "s14": "rpm",
    "s15": "", "s16": "", "s17": "", "s18": "rpm", "s19": "rpm", "s20": "lbm/s", "s21": "lbm/s",
}


def sensor_label(col: str) -> str:
    """그래프 축·범례용 이름. 예) 's4 T50 · LPT(저압터빈) outlet 온도 (°R)'"""
    unit = SENSOR_UNIT.get(col, "")
    return f"{col} {SENSOR_MEANING.get(col, '')}" + (f" ({unit})" if unit else "")

# ---------------------------------------------------------------------------
# 2. [임의 설정값 #1] 상수(=변화 없는) 센서 제거 기준
# ---------------------------------------------------------------------------
# 어떤 센서는 같은 비행 조건에서라면 엔진이 고장에 가까워져도 값이 거의 안 움직입니다.
# 안 움직이는 컬럼은 모델 입력에 넣어봤자 학습에 아무 도움이 안 되고(분산이 0에 가까움 = 정보량 0),
# 오히려 정규화(z-score) 과정에서 "표준편차로 나누기"를 할 때 0에 가까운 수로 나눠서 계산이
# 불안정해질 수 있습니다.
#
# 판단 기준: train 데이터에서 "운전조건 하나 안에서의 표준편차" 중 가장 큰 값이 0.01 미만이면
# "사실상 상수"로 보고 제거합니다. 전체 표준편차를 보면 안 되는 이유: FD004는 비행 조건이 바뀔
# 때마다 센서값이 크게 뛰어서, 사실상 고정된 센서도 "많이 움직이는" 것처럼 보이기 때문입니다.
# (FD001은 운전조건이 하나뿐이라 이 계산이 전체 표준편차와 같습니다.)
#
# 왜 0.01이냐면, 표준편차를 정렬했을 때 "거의 안 움직이는 그룹"과 "의미있게 움직이는 그룹"
# 사이에 뚜렷한 공백이 있고 0.01이 그 사이에 있기 때문입니다.
#   FD004: 제거 대상 최대 0.007  ↔  남는 센서 최소 0.023
#   FD001: os2(0.0003), s6(0.0014), os1(0.0022)  ↔  약 15배 간격  ↔  s15(0.0375), s8(0.071), ...
# → FD004에서 제거: s1, s5, s10, s16, s18, s19, os1, os2, os3 → 센서 15개 사용
#   FD001은 여기에 s6(P15 바이패스 덕트 압력)까지 제거 → 센서 14개 사용
#   원자료 기호로 보면 T2·P2(팬 입구 온도·압력), epr(엔진 압력비), farB(연소기 연료-공기비),
#   Nf_dmd·PCNfR_dmd(요구 팬 속도)로, 비행 조건이 정해지면 고정되는 입구 조건·설정값들이라
#   물리적으로도 변하지 않는 것이 맞습니다. (C-MAPSS 선행 연구들도 이 센서들을 흔히 입력에서 제외합니다.)
CONST_STD_THRESHOLD = 0.01


# ---------------------------------------------------------------------------
# [임의 설정값 #13] 운전조건(regime) 구분 방법 — FD002·FD004용
# ---------------------------------------------------------------------------
# 운전 설정 3개(os1 고도, os2 마하수, os3 스로틀)는 6개 조합 근처에만 찍혀 있고 그 사이
# 값이 없습니다 (FD004 train: 0/0/100, 10/0.25/100, 20/0.7/100, 25/0.62/60, 35/0.84/100,
# 42/0.84/100 — 각 9천~1만5천 행). 그래서 참고자료(MathWorks similarity-based RUL)처럼
# K-means를 돌리지 않고, os1은 정수, os2는 소수 둘째 자리, os3는 정수로 반올림한 조합을
# 그대로 운전조건 이름으로 씁니다 — 결과가 같으면서 무작위성이 없고 설명하기 쉽습니다.
# FD001·FD003은 모든 행이 한 조합으로 묶여 운전조건이 1개가 됩니다.
def assign_regime(df: pd.DataFrame) -> pd.Series:
    return (df["os1"].round(0).abs().astype(str) + "/" + df["os2"].round(2).abs().astype(str)
            + "/" + df["os3"].round(0).astype(str))


def load_raw(path: str) -> pd.DataFrame:
    """train_FD004.txt / test_FD004.txt 같은 원본 텍스트 파일을 DataFrame으로 읽고,
    운전조건 이름(regime) 열을 붙입니다."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLS)
    df["regime"] = assign_regime(df)
    return df


def find_constant_columns(train_df: pd.DataFrame, threshold: float = CONST_STD_THRESHOLD) -> list:
    """train 데이터 기준으로 '사실상 상수'인 센서/운전조건 컬럼 이름 리스트를 반환합니다.

    중요: 반드시 train 데이터만 보고 판단합니다. test 데이터를 같이 보고 판단하면
    "미래 정보(test)를 미리 훔쳐본 것"이 되어 데이터 누수(data leakage)가 됩니다.
    """
    candidate_cols = SENSOR_COLS_RAW + SETTING_COLS
    stds = train_df.groupby("regime")[candidate_cols].std().max()
    return list(stds[stds < threshold].index)


# ---------------------------------------------------------------------------
# 3. [임의 설정값 #2] RUL 라벨 만들기 — Piecewise Linear RUL (구간별 선형 RUL)
# ---------------------------------------------------------------------------
# RUL의 정답값은 "그 사이클 이후 몇 번을 더 운행하고 고장나는가" 입니다.
# 엔진이 마지막으로 기록된 사이클(=고장 시점)을 알고 있으므로
#   RUL(사이클 t) = (엔진의 마지막 사이클) - t
# 로 그냥 계산할 수도 있습니다. 하지만 이렇게 하면 문제가 생깁니다:
#
#   엔진이 갓 가동을 시작한 시점(사이클 1, 2, 3...)의 센서 값은 사실상 "정상" 그 자체라
#   서로 구분이 안 되는데, 정답 RUL은 300, 299, 298...처럼 계속 다른 값을 요구합니다.
#   즉 "센서로는 구분이 안 되는데 정답은 계속 달라야 한다"는 모순이 생겨서 모델이
#   초반 구간에서 억지로 그럴듯한 숫자를 맞추려다 오히려 후반(진짜 중요한, 고장 임박
#   구간)의 예측 품질이 나빠집니다.
#
# 그래서 실제로 많이 쓰는 방법이 "일정 값 이상이면 그냥 그 값으로 잘라버리는" 방식입니다:
#   RUL_label(t) = min(진짜 RUL(t), CLIP_VALUE)
# 이렇게 하면 "아직 한참 남았다"는 구간은 전부 같은 값(CLIP_VALUE)으로 취급되고,
# 모델은 "진짜로 값이 변하기 시작하는" 열화(degradation) 구간 학습에 집중하게 됩니다.
#
# CLIP_VALUE를 얼마로 할지가 임의 설정값입니다. 우리는 125로 정했습니다. 근거:
#   1) 참고자료로 준 MathWorks의 LSTM 튜토리얼
#      (https://www.mathworks.com/help/deeplearning/ug/sequence-to-sequence-regression-using-deep-learning.html)
#      도 정확히 이 방식(피스와이즈 선형 RUL)을 쓰고, 거기서는 150을 컷오프로 사용합니다.
#   2) 그런데 우리 train 데이터의 "가장 일찍 고장난 엔진"의 총 수명은 128 사이클입니다
#      (01_eda.py의 수명 분포 그림으로 확인: FD001·FD004 모두 최소 128).
#      만약 MathWorks 예제처럼 150을 그대로 쓰면, 이 128사이클짜리 엔진은 "평평한 구간"이
#      단 한 번도 나타나지 않고 시작부터 끝까지 계속 RUL이 줄어드는 것으로 학습되어,
#      다른 엔진들과 학습 신호가 일관되지 않습니다.
#   3) 그래서 "가장 짧은 엔진 수명(128)보다 확실히 작은 값"으로 125를 선택해, 모든
#      엔진이 최소 몇 사이클이라도 "평평한 구간"을 갖도록 했습니다. C-MAPSS를 다룬 선행
#      연구들도 대체로 120~130 범위의 상한을 쓰고 있어 결과를 비교하기에도 무리가 없습니다.
RUL_CLIP_VALUE = 125


def add_rul_labels(df: pd.DataFrame) -> pd.DataFrame:
    """train 스타일 데이터(각 엔진이 고장까지 끝까지 기록됨)에 RUL 정답 컬럼을 추가합니다.

    test 데이터처럼 '고장 전에 관측이 끊긴' 데이터에는 이 함수를 쓰지 않고,
    별도로 RUL_FD00X.txt(정답 파일)의 값을 이용합니다 (02~05번 test 평가 참고).
    """
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    raw_rul = max_cycle - df["cycle"]
    df["RUL"] = np.minimum(raw_rul, RUL_CLIP_VALUE)
    return df


# ---------------------------------------------------------------------------
# 4. 센서 값 정규화 (Z-score)
# ---------------------------------------------------------------------------
# 센서마다 단위와 값 범위가 완전히 다릅니다 (어떤 센서는 8000~9000대 값, 어떤 센서는
# 0~1 사이 값). 이 상태로 모델에 넣으면 "숫자가 큰 센서"가 실제 중요도와 상관없이
# 더 크게 취급될 수 있습니다. 그래서 Z-score 정규화를 합니다:
#   normalized = (원래값 - 평균) / 표준편차
# → 정규화 후에는 모든 센서가 평균 0, 표준편차 1인 비슷한 스케일이 됩니다.
#
# 중요한 원칙: 평균/표준편차는 반드시 "train 데이터에서만" 계산합니다.
# (MathWorks LSTM 참고자료에도 명시된 원칙이고, 과정 진행 가이드의 "스케일러는 학습 구간에서만
# fit" 규칙과도 같습니다.) test 데이터는 이 train 기준 값을 그대로 "적용"만 받습니다.
# test 데이터로 새로 평균/표준편차를 계산해버리면 "아직 보면 안 되는 미래 데이터의
# 통계 정보"가 모델에 흘러들어가는 데이터 누수가 됩니다.
#
# 다중 운전조건(FD002·FD004)에서는 평균/표준편차를 "운전조건별로 따로" 계산합니다.
# 같은 엔진 상태라도 고도 0에서와 고도 42에서의 센서값은 전혀 다르기 때문에, 전체 평균으로
# 빼면 "비행 조건이 바뀐 것"이 "엔진이 닳은 것"처럼 보입니다. 운전조건마다 그 조건의
# 평균을 빼 주면 조건 차이가 사라지고 열화 신호만 남습니다. (MathWorks similarity-based RUL
# 참고자료의 "운전조건별 정규화" 단계와 같은 아이디어.) FD001은 운전조건이 하나라 예전과 같습니다.
class Normalizer:
    """train으로 학습(fit)하고 train/test 양쪽에 동일하게 적용(transform)하는 정규화기.
    운전조건(regime)별로 평균·표준편차를 따로 계산합니다."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.cols_ = None

    def fit(self, df: pd.DataFrame, cols: list):
        self.cols_ = cols
        g = df.groupby("regime")[cols]
        self.mean_ = g.mean()
        self.std_ = g.std().replace(0, 1.0).fillna(1.0)  # 혹시 모를 0 나눗셈 방지
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        unknown = set(df["regime"]) - set(self.mean_.index)
        if unknown:
            raise ValueError(f"train에 없던 운전조건이 있습니다: {unknown}")
        df = df.copy()
        mean = self.mean_.loc[df["regime"]].to_numpy()
        std = self.std_.loc[df["regime"]].to_numpy()
        df[self.cols_] = (df[self.cols_].to_numpy() - mean) / std
        return df


class RegimeCorrector:
    """그래프·추세 분석용: 센서값에서 "운전조건 때문에 생긴 차이"만 빼고 원래 단위는 유지합니다.
        보정값 = 원래값 − (그 운전조건의 train 평균) + (train 전체 평균)
    예) s4 온도가 고도에 따라 1,100~1,420°R로 뛰어도, 보정 후에는 한 줄의 열화 추세로 보입니다.
    FD001은 운전조건이 하나라 보정값 = 원래값입니다 (그래프와 수치가 예전과 동일)."""

    def fit(self, df: pd.DataFrame, cols: list):
        self.cols_ = cols
        self.regime_mean_ = df.groupby("regime")[cols].mean()
        self.global_mean_ = df[cols].mean()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if not MULTI_REGIME:
            return df
        df[self.cols_] = (df[self.cols_].to_numpy() - self.regime_mean_.loc[df["regime"]].to_numpy()
                          + self.global_mean_.to_numpy())
        return df


# ---------------------------------------------------------------------------
# 5. [임의 설정값 #3] 롤링 윈도우(이동 구간) 피처 — Random Forest용
# ---------------------------------------------------------------------------
# Random Forest 같은 '표(table) 기반' 모델은 한 시점의 센서값 하나만 보면 그 시점이
# "막 정상 상태에서 흔들리는 것"인지 "진짜로 계속 나빠지는 추세"인지 구분을 못합니다.
# 그래서 "최근 WINDOW 사이클 동안의 평균/표준편차/기울기(추세)"를 새로운 피처로
# 만들어서 넣어줍니다. 이렇게 하면 모델이 "지금 값 자체"뿐 아니라 "최근 변화 추이"까지
# 참고할 수 있습니다.
#
# WINDOW = 15로 정했습니다. 근거:
#   - train에서 가장 짧은 엔진 수명이 128 사이클입니다. 윈도우가 너무 크면(예: 50)
#     이 엔진은 학습 데이터의 상당 부분이 "직전 50개가 다 안 채워진" 초반 구간이 됩니다.
#     15면 128사이클짜리 엔진도 창이 꽉 찬 학습 샘플이 100개 넘게 남습니다.
#   - 너무 작으면(예: 3~5) 노이즈에 민감해서 "추세"보다 "순간 흔들림"을 더 많이 반영합니다.
#   - 15는 이 둘의 절충점으로 잡은 값이며, 팀에서 시간이 되면 5/10/15/20으로 바꿔가며
#     검증 성능(MAE)을 비교해보는 것을 추천합니다 (아래 build_rf_features 함수의
#     window 인자를 바꾸기만 하면 됩니다).
ROLLING_WINDOW = 15


def build_rf_features(df: pd.DataFrame, sensor_cols: list, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Random Forest 입력용 피처를 만듭니다: 원본 센서값 + 최근 window 사이클의
    이동평균(mean) / 이동표준편차(std) / 선형추세 기울기(slope)를 엔진(unit)별로 계산합니다.

    반드시 unit별로 그룹을 나눠서 계산합니다 (다른 엔진의 데이터가 섞이면 안 되므로),
    그리고 각 그룹 안에서는 cycle 순서를 유지한 채로만 계산합니다 (미래 정보 누수 방지:
    시점 t의 피처를 만들 때 t보다 미래인 t+1, t+2...의 값을 절대 사용하지 않습니다).
    """
    df = df.sort_values(["unit", "cycle"]).copy()
    feature_frames = []

    for uid, g in df.groupby("unit"):
        g = g.sort_values("cycle").reset_index(drop=True)
        feats = {"unit": g["unit"], "cycle": g["cycle"]}
        t = g["cycle"].astype(float)
        for col in sensor_cols:
            roll = g[col].rolling(window=window, min_periods=1)
            feats[f"{col}_mean{window}"] = roll.mean()
            feats[f"{col}_std{window}"] = roll.std().fillna(0)
            # 기울기(slope): 최근 window개 지점에 직선을 하나 그었을 때의 기울기.
            # 양수면 값이 오르는 추세, 음수면 내려가는 추세라는 뜻입니다.
            feats[f"{col}_slope{window}"] = _rolling_slope_vectorized(g[col], t, window)
        feat_df = pd.DataFrame(feats)
        # 원본 센서값(현재 시점 값)도 그대로 같이 둡니다 — "현재 상태"와 "최근 추세"를
        # 모델이 둘 다 볼 수 있게 하기 위함입니다.
        for col in sensor_cols:
            feat_df[col] = g[col].values
        if "RUL" in g.columns:
            feat_df["RUL"] = g["RUL"].values
        feature_frames.append(feat_df)

    return pd.concat(feature_frames, ignore_index=True)


def _rolling_slope_vectorized(y: pd.Series, t: pd.Series, window: int) -> np.ndarray:
    """롤링(이동) 구간 최소제곱 직선의 기울기를, 반복문 없이 pandas rolling().sum()
    만으로 계산하는 벡터화(vectorized) 버전입니다.

    수학적으로 최소제곱 직선의 기울기는
        slope = (n*Σ(t*y) - Σt*Σy) / (n*Σ(t^2) - (Σt)^2)
    로 구할 수 있고, 여기서 t는 "몇 번째 사이클인가"(실제 cycle 번호)를 그대로
    사용해도 됩니다 — x축을 통째로 평행이동해도(t=0,1,2... 로 다시 매기지 않아도)
    직선의 '기울기'는 변하지 않기 때문입니다(절편만 달라짐). 덕분에 매 시점마다
    파이썬 반복문+np.polyfit을 부르는 대신, pandas의 rolling(window).sum()
    (C로 구현되어 매우 빠름) 4번만 호출해서 한 번에 계산할 수 있습니다
    (시점마다 np.polyfit을 부르는 방식보다 수백 배 빠름).
    """
    n_roll = t.rolling(window=window, min_periods=1).count()
    s_t = t.rolling(window=window, min_periods=1).sum()
    s_y = y.rolling(window=window, min_periods=1).sum()
    s_tt = (t * t).rolling(window=window, min_periods=1).sum()
    s_ty = (t * y).rolling(window=window, min_periods=1).sum()

    denom = n_roll * s_tt - s_t * s_t
    numer = n_roll * s_ty - s_t * s_y
    slope = np.where(denom.abs() > 1e-9, numer / denom.replace(0, np.nan), 0.0)
    slope = np.nan_to_num(slope, nan=0.0)
    # 표본이 2개 미만인 맨 첫 지점은 기울기를 정의할 수 없으므로 0으로 둡니다.
    slope = np.where(n_roll.values < 2, 0.0, slope)
    return slope


# ---------------------------------------------------------------------------
# 6. [임의 설정값 #4] train/validation 엔진 분할
# ---------------------------------------------------------------------------
# 과정 진행 가이드의 "시간 순서를 유지한 분할, 무작위 셔플 금지" 원칙을 C-MAPSS 구조에
# 맞게 해석한 방식입니다 (주제 F 권장 흐름: "개체 단위로 학습/검증을 나눈다").
# C-MAPSS는 "서로 다른 엔진 각각이 독립적으로 가동 시작~고장까지 기록된" 데이터라서,
# 하나의 연속된 시간축이 아니라 "엔진 수만큼의 독립적인 시계열 묶음"입니다.
#
# 만약 한 엔진의 사이클들을 앞부분은 train, 뒷부분은 validation으로 나누면
# (=흔히 하는 "시간순 분할"의 문자 그대로 해석) 같은 엔진의 정보가 train과
# validation에 둘 다 섞여 들어가서 "이 엔진을 이미 어느정도 알고 있는 상태에서
# 미래를 맞추는" 상황이 되어 실제 신규 엔진에 대한 일반화 성능을 과대평가하게 됩니다.
#
# 그래서 우리는 "엔진 단위로" 나눕니다: 전체 엔진 중 일부는 통째로 train,
# 나머지 엔진은 통째로 validation. 이러면 validation에 들어간 엔진은 모델이
# 학습 중에 단 한 사이클도 보지 못한, 완전히 새로운 엔진이 되어 진짜 실전과
# 비슷한 평가가 됩니다. (각 엔진 '내부'에서는 여전히 cycle 순서를 그대로 유지하고
# 있으므로 시간 역행은 없습니다.)
#
# 분할 비율 80:20, seed=42로 고정했습니다.
#   - 80:20은 FD001의 엔진이 100대뿐인 점을 고려한 선택입니다 (validation 20대면 통계가
#     너무 불안정하지 않으면서 train 80대로도 학습에 충분). FD004는 249대 → 200:49.
#   - seed=42는 "재현 가능성"을 위한 고정값입니다 (팀원이 같은 코드를 실행해도
#     항상 같은 엔진이 validation으로 뽑히도록). 42라는 숫자 자체에 특별한
#     의미는 없고, 코드/데이터과학 커뮤니티에서 관례적으로 많이 쓰는 시드값입니다.
VAL_SPLIT_RATIO = 0.2
SPLIT_SEED = 42


def split_engines(df: pd.DataFrame, val_ratio: float = VAL_SPLIT_RATIO, seed: int = SPLIT_SEED):
    """엔진(unit) 단위로 train/validation을 나눠서, 각각의 unit 리스트를 반환합니다."""
    rng = np.random.default_rng(seed)
    units = df["unit"].unique()
    units_sorted = np.sort(units)
    shuffled = rng.permutation(units_sorted)
    n_val = max(1, int(len(shuffled) * val_ratio))
    val_units = set(shuffled[:n_val].tolist())
    train_units = set(shuffled[n_val:].tolist())
    return train_units, val_units


# ---------------------------------------------------------------------------
# 7. [임의 설정값 #11] 위험(고장 임박) 기준: 잔여 ≤ 40사이클
# ---------------------------------------------------------------------------
# 과제 필수 과제(09_classification.py)의 "위험" 라벨과 대시보드의 빨간불(RED) · 정비 경보가 같은
# 기준을 쓰도록 여기 한 곳에 둡니다.
# 근거 (13_threshold_cost.py, docs/DESIGN_DECISIONS.md #11):
#   C-MAPSS 터보팬 엔진 정비 일정 연구의 운영 조건 — 추가 정비 준비 최소 7일, 정비 슬롯 10~20일
#   간격, 주간 계획, 하루 1비행, 계획:비계획 정비 비용 1:5 (de Pater, Reijns & Mitici 2022,
#   Reliability Engineering & System Safety 221, 108341) — 으로 "예측 RUL ≤ N이면 경보 → 정비"를
#   시뮬레이션하면, FD004 GRU(04번 최종 모델)에서 운행 1사이클당 정비 비용이 가장 싼 N은 40이었습니다
#   (비용 비율을 1:2로 바꿔도 40 — Lee & Mitici 2023, RESS 230, 108908).
#   N = 30이면 경보 뒤 정비까지 걸리는 시간(평균 17사이클) 안에 엔진 14.3%가 고장 나고,
#   N을 40보다 크게 잡으면 비용이 완만하게만 늘어 (45: +2%, 50: +5%) 불확실할 땐 크게 잡는 편이 안전합니다.
#   과제 문서 주제 F의 예시값(30)과 문제정의서의 "예: 30"에서 출발했지만, 위 근거로 40으로 확정했습니다.
DANGER_RUL = 40

# ---------------------------------------------------------------------------
# [임의 설정값 #7] 대시보드 위험도 등급 구간 (07·08 대시보드 export가 같이 씀)
# ---------------------------------------------------------------------------
# 예측된 RUL(잔존수명, 단위: cycle)을 대시보드에서 신호등처럼 바로 보여주기 위해
# 3단계 등급으로 나눕니다.
#   위험(RED): 예측 RUL ≤ 40 사이클   → 정비 경보, 정비 일정을 지금 잡아야 하는 수준 (= 위 DANGER_RUL)
#   주의(YELLOW): 40 초과 ~ 80 미만   → 정비 계획을 준비해야 하는 수준
#   정상(GREEN): 80 사이클 이상       → 당장은 여유 있는 수준
# 주의 구간 상한(80)은 "위험 기준의 두 배 = 정비 준비를 시작할 여유"로 둔 값입니다
# (위험 기준을 20 → 30 → 40으로 바꿀 때마다 같은 규칙을 적용). 현업 정보를 얻으면 교체를 추천합니다.
RISK_THRESHOLDS = {"red_at_or_below": DANGER_RUL, "yellow_below": 2 * DANGER_RUL}


def risk_level(rul_pred: float) -> str:
    if rul_pred <= RISK_THRESHOLDS["red_at_or_below"]:
        return "RED"
    elif rul_pred < RISK_THRESHOLDS["yellow_below"]:
        return "YELLOW"
    return "GREEN"


# ---------------------------------------------------------------------------
# 8. 평가지표 함수들
# ---------------------------------------------------------------------------
def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def nasa_score(y_true, y_pred):
    """C-MAPSS 데이터를 만든 NASA PHM08 챌린지의 공식 평가 스코어입니다.
    (출처: Saxena, A. et al., "Damage Propagation Modeling for Aircraft Engine
    Run-to-Failure Simulation", PHM 2008.)

    MAE/RMSE와 다른 점: 이 점수는 "늦게 예측하는 것(실제보다 RUL을 크게 예측해서
    정비 시점을 놓치는 것)"에 "일찍 예측하는 것(RUL을 작게 예측해서 너무 일찍
    정비하는 것)"보다 훨씬 큰 벌점을 줍니다. 실제 설비 운영에서는 "고장을 못 맞춰서
    사고가 나는 것"이 "너무 일찍 정비해서 비용이 조금 더 드는 것"보다 훨씬 위험하기
    때문에, 이 비대칭적 벌점 구조가 실제 의사결정 상황과 더 잘 맞습니다.

    d = 예측 - 실제
      d < 0 (예측이 실제보다 작음 = 일찍 경고, 보수적) → exp(-d/13) - 1
      d >= 0 (예측이 실제보다 큼 = 늦게 경고, 위험)   → exp(d/10) - 1
    """
    d = np.asarray(y_pred) - np.asarray(y_true)
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(score))
