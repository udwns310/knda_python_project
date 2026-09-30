// ⚠️ 자동 생성 파일 — 분석 레포의 scripts/08_export_dashboard_ts.py 가 만든다.
// 직접 고치지 말고 파이썬 파이프라인을 다시 돌려서 새로 받아올 것.
// 원본 데이터: C-MAPSS FD001 (운전조건 1종 · 고장모드 1종, NASA 터보팬 엔진 열화 시뮬레이션), 모델: LSTM Seq2Seq
// (분석 레포 docs/DESIGN_DECISIONS.md, 이 레포 docs/DATA_MAPPING.md 참고)
// 시간축은 엔진 운행 '사이클(cycle)' 입니다.

export type Status = 'good' | 'warning' | 'serious' | 'critical'

export const statusLabel: Record<Status, string> = {
  good: '정상',
  warning: '주의',
  serious: '경고',
  critical: '위험',
}

export const dataMeta = {
  "generatedAt": "2026-09-30T12:38:27",
  "dataset": "C-MAPSS FD001",
  "datasetDescription": "운전조건 1종 · 고장모드 1종",
  "sensorValues": "원본값",
  "testEngines": 100,
  "selectedEngines": [
    34,
    42,
    81,
    76,
    41,
    32,
    46,
    3,
    47
  ],
  "featuredEngine": 41,
  "note": "시간축은 엔진 운행 사이클(cycle). 위험 등급(RED/YELLOW/GREEN)은 예측 RUL 기준(≤40/<80/그 외). 비용·정비시간은 가정값(08_export_dashboard_ts.py COST_ASSUMPTIONS), 탐지 성능(TP/FN/FP)은 LSTM 모델의 실제 test set 결과. RUL 신뢰구간·생존곡선은 검증셋에서 예측이 비슷했던 사례들의 실제 RUL 분포."
}

/** 신호등 기준 (분석 레포 common.py의 DANGER_RUL · RISK_THRESHOLDS) — 화면 문구도 이 값을 씀
 *  위험: 예측 RUL ≤ dangerRul, 주의: dangerRul 초과 ~ warningRul 미만, 정상: warningRul 이상 */
export const riskThresholds = { dangerRul: 40, warningRul: 80 }

/** 메인 차트: 대표 엔진(#41)의 예측 RUL 추이 — h = 창 안의 사이클 순서(0~24) */
export const sensorSeries: { t: string; h: number; v: number }[] = [
  {
    "t": "#99",
    "h": 0,
    "v": 85.06
  },
  {
    "t": "#100",
    "h": 1,
    "v": 79.45
  },
  {
    "t": "#101",
    "h": 2,
    "v": 76.31
  },
  {
    "t": "#102",
    "h": 3,
    "v": 68.06
  },
  {
    "t": "#103",
    "h": 4,
    "v": 66.69
  },
  {
    "t": "#104",
    "h": 5,
    "v": 60.29
  },
  {
    "t": "#105",
    "h": 6,
    "v": 57.12
  },
  {
    "t": "#106",
    "h": 7,
    "v": 45.0
  },
  {
    "t": "#107",
    "h": 8,
    "v": 42.68
  },
  {
    "t": "#108",
    "h": 9,
    "v": 31.08
  },
  {
    "t": "#109",
    "h": 10,
    "v": 36.78
  },
  {
    "t": "#110",
    "h": 11,
    "v": 29.07
  },
  {
    "t": "#111",
    "h": 12,
    "v": 27.96
  },
  {
    "t": "#112",
    "h": 13,
    "v": 16.09
  },
  {
    "t": "#113",
    "h": 14,
    "v": 9.55
  },
  {
    "t": "#114",
    "h": 15,
    "v": 9.52
  },
  {
    "t": "#115",
    "h": 16,
    "v": 7.61
  },
  {
    "t": "#116",
    "h": 17,
    "v": 8.87
  },
  {
    "t": "#117",
    "h": 18,
    "v": 9.06
  },
  {
    "t": "#118",
    "h": 19,
    "v": 10.97
  },
  {
    "t": "#119",
    "h": 20,
    "v": 10.35
  },
  {
    "t": "#120",
    "h": 21,
    "v": 10.29
  },
  {
    "t": "#121",
    "h": 22,
    "v": 10.5
  },
  {
    "t": "#122",
    "h": 23,
    "v": 10.76
  },
  {
    "t": "#123",
    "h": 24,
    "v": 12.27
  }
]

export const anomalyWindow = {
  "start": 9,
  "end": 24
}

const START_CYCLE: number = 99

export function formatHour(h: number): string {
  return `cycle #${START_CYCLE + Math.round(h)}`
}

/** 스크러버 위치(h)에서 대표 엔진의 예측 RUL (사이클 사이는 선형 보간) */
export function getRulAtHour(hour: number): number {
  const clamped = Math.max(0, Math.min(24, hour))
  for (let i = 0; i < sensorSeries.length - 1; i++) {
    const a = sensorSeries[i]
    const b = sensorSeries[i + 1]
    if (clamped >= a.h && clamped <= b.h) {
      const frac = (clamped - a.h) / (b.h - a.h)
      return a.v + (b.v - a.v) * frac
    }
  }
  return sensorSeries[sensorSeries.length - 1].v
}

export interface EquipmentRow {
  id: string
  name: string
  health: number
  status: Status
}

/** 선택된 9개 엔진의 헬스 점수(0~100) = 100 × (예측 RUL / 125).
 *  참고: 각 엔진은 서로 독립된 운행 이력이라 '공유된 시간축'이 없어서,
 *  스크러버(hour)에 따라 순위가 실시간으로 바뀌지는 않고 마지막 관측 시점 기준 스냅샷입니다. */
const equipmentRankingStatic: EquipmentRow[] = [
  {
    "id": "engine-34",
    "name": "Engine #34",
    "health": 8,
    "status": "critical"
  },
  {
    "id": "engine-42",
    "name": "Engine #42",
    "health": 8,
    "status": "critical"
  },
  {
    "id": "engine-81",
    "name": "Engine #81",
    "health": 9,
    "status": "critical"
  },
  {
    "id": "engine-76",
    "name": "Engine #76",
    "health": 9,
    "status": "critical"
  },
  {
    "id": "engine-41",
    "name": "Engine #41",
    "health": 10,
    "status": "critical"
  },
  {
    "id": "engine-32",
    "name": "Engine #32",
    "health": 33,
    "status": "warning"
  },
  {
    "id": "engine-46",
    "name": "Engine #46",
    "health": 36,
    "status": "warning"
  },
  {
    "id": "engine-3",
    "name": "Engine #3",
    "health": 64,
    "status": "good"
  },
  {
    "id": "engine-47",
    "name": "Engine #47",
    "health": 100,
    "status": "good"
  }
]

export function getEquipmentRankingAtHour(_hour: number): EquipmentRow[] {
  return equipmentRankingStatic
}

export const equipmentRanking = equipmentRankingStatic

export interface AlertRow {
  time: string
  h: number
  equipment: string
  sensor: string
  severity: Status
  action: string
  type?: string
}

/** 대표 엔진(#41)의 예측 RUL이 주의(80 미만)/위험(40 이하) 임계값을 넘은 시점 — 실제 계산값 */
export const alertLog: AlertRow[] = [
  {
    "h": 9,
    "time": "cycle #108",
    "equipment": "Engine #41",
    "sensor": "예측 RUL",
    "severity": "critical",
    "action": "정비 일정 즉시 수립",
    "type": "EngineRemoval"
  },
  {
    "h": 1,
    "time": "cycle #100",
    "equipment": "Engine #41",
    "sensor": "예측 RUL",
    "severity": "warning",
    "action": "정비 계획 준비",
    "type": "EngineRemoval"
  }
]

export function getAlertsUpToHour(hour: number): AlertRow[] {
  return alertLog.filter((a) => a.h <= hour)
}

export const kpis = {
  totalEquipment: 9,
  riskEquipment: equipmentRanking.filter((e) => e.status === 'critical').length,
  avgHealth: equipmentRanking.reduce((s, e) => s + e.health, 0) / equipmentRanking.length,
  todayAlerts: alertLog.length,
  alertsDelta: 0,
}

export function getKpisAtHour(hour: number) {
  const ranking = getEquipmentRankingAtHour(hour)
  const riskEquipment = ranking.filter((e) => e.status === 'critical').length
  const avgHealth = ranking.reduce((s, e) => s + e.health, 0) / ranking.length
  const todayAlerts = getAlertsUpToHour(hour).length
  return {
    totalEquipment: kpis.totalEquipment,
    riskEquipment,
    avgHealth,
    todayAlerts,
    alertsDelta: kpis.alertsDelta,
  }
}

// ---------------------------------------------------------------------------
// 엔진별 상세 데이터 (설비 상세 페이지 — /equipment/:id 로 조회)
// predictedRulCycle(90% 구간)·survivalCurve는 검증셋에서 예측이 비슷했던(±10 cycle)
// 사례들의 실제 RUL 분포로 계산 (생존곡선 음영 = 검증 엔진 단위 부트스트랩 90% 구간)
// ---------------------------------------------------------------------------
export interface EngineDetail {
  equipmentInfo: { line: string; installedAt: string; lastMaintenance: string; team: string; operatingCycles: number; modelNo: string }
  predictedRulCycle: { median: number; lower: number; upper: number }
  survivalCurve: { cycle: number; median: number; lower: number; upper: number }[]
  sensorTrend: Record<string, { h: number; v: number }[]>
  maintenanceHistory: { date: string; type: string; description: string; status: 'done' | 'scheduled' }[]
  status: Status
  health: number
}

export const engineDetails: Record<string, EngineDetail> = {
  "engine-41": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종 · 고장모드 1종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 123,
      "modelNo": "Turbofan (FD001)"
    },
    "predictedRulCycle": {
      "median": 12.3,
      "lower": 1.0,
      "upper": 27.0
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 0.9727,
        "lower": 0.9642,
        "upper": 0.9808
      },
      {
        "cycle": 10,
        "median": 0.5597,
        "lower": 0.5149,
        "upper": 0.5996
      },
      {
        "cycle": 20,
        "median": 0.1824,
        "lower": 0.1218,
        "upper": 0.2411
      },
      {
        "cycle": 30,
        "median": 0.0189,
        "lower": 0.0,
        "upper": 0.0444
      },
      {
        "cycle": 40,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 60,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 80,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 100,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 120,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 140,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      }
    ],
    "sensorTrend": {
      "s4": [
        {
          "h": 1,
          "v": 1396.42
        },
        {
          "h": 16,
          "v": 1393.64
        },
        {
          "h": 31,
          "v": 1397.03
        },
        {
          "h": 47,
          "v": 1402.65
        },
        {
          "h": 62,
          "v": 1398.73
        },
        {
          "h": 77,
          "v": 1396.42
        },
        {
          "h": 93,
          "v": 1404.04
        },
        {
          "h": 108,
          "v": 1410.03
        },
        {
          "h": 123,
          "v": 1416.76
        }
      ],
      "s9": [
        {
          "h": 1,
          "v": 9053.25
        },
        {
          "h": 16,
          "v": 9041.94
        },
        {
          "h": 31,
          "v": 9051.23
        },
        {
          "h": 47,
          "v": 9048.2
        },
        {
          "h": 62,
          "v": 9042.02
        },
        {
          "h": 77,
          "v": 9050.08
        },
        {
          "h": 93,
          "v": 9047.96
        },
        {
          "h": 108,
          "v": 9039.95
        },
        {
          "h": 123,
          "v": 9036.88
        }
      ],
      "s3": [
        {
          "h": 1,
          "v": 1586.07
        },
        {
          "h": 16,
          "v": 1584.22
        },
        {
          "h": 31,
          "v": 1576.68
        },
        {
          "h": 47,
          "v": 1589.93
        },
        {
          "h": 62,
          "v": 1589.09
        },
        {
          "h": 77,
          "v": 1580.19
        },
        {
          "h": 93,
          "v": 1586.0
        },
        {
          "h": 108,
          "v": 1585.56
        },
        {
          "h": 123,
          "v": 1594.24
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #123 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 12.3 사이클 소진 시 정비 권고 (90% 구간 1.0~27.0)",
        "status": "scheduled"
      }
    ],
    "status": "critical",
    "health": 10
  },
  "engine-34": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종 · 고장모드 1종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 203,
      "modelNo": "Turbofan (FD001)"
    },
    "predictedRulCycle": {
      "median": 9.6,
      "lower": 1.0,
      "upper": 25.0
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 0.9525,
        "lower": 0.9475,
        "upper": 0.9569
      },
      {
        "cycle": 10,
        "median": 0.4774,
        "lower": 0.4226,
        "upper": 0.5259
      },
      {
        "cycle": 20,
        "median": 0.1235,
        "lower": 0.0699,
        "upper": 0.1777
      },
      {
        "cycle": 30,
        "median": 0.019,
        "lower": 0.0,
        "upper": 0.0488
      },
      {
        "cycle": 40,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 60,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 80,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 100,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 120,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 140,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      }
    ],
    "sensorTrend": {
      "s4": [
        {
          "h": 1,
          "v": 1402.56
        },
        {
          "h": 26,
          "v": 1387.99
        },
        {
          "h": 51,
          "v": 1395.88
        },
        {
          "h": 77,
          "v": 1405.09
        },
        {
          "h": 102,
          "v": 1393.27
        },
        {
          "h": 127,
          "v": 1402.14
        },
        {
          "h": 153,
          "v": 1406.22
        },
        {
          "h": 178,
          "v": 1416.91
        },
        {
          "h": 203,
          "v": 1427.49
        }
      ],
      "s9": [
        {
          "h": 1,
          "v": 9071.43
        },
        {
          "h": 26,
          "v": 9059.68
        },
        {
          "h": 51,
          "v": 9068.14
        },
        {
          "h": 77,
          "v": 9070.73
        },
        {
          "h": 102,
          "v": 9061.87
        },
        {
          "h": 127,
          "v": 9067.89
        },
        {
          "h": 153,
          "v": 9051.18
        },
        {
          "h": 178,
          "v": 9056.19
        },
        {
          "h": 203,
          "v": 9036.27
        }
      ],
      "s3": [
        {
          "h": 1,
          "v": 1574.94
        },
        {
          "h": 26,
          "v": 1586.31
        },
        {
          "h": 51,
          "v": 1578.49
        },
        {
          "h": 77,
          "v": 1583.53
        },
        {
          "h": 102,
          "v": 1591.2
        },
        {
          "h": 127,
          "v": 1586.2
        },
        {
          "h": 153,
          "v": 1588.56
        },
        {
          "h": 178,
          "v": 1594.82
        },
        {
          "h": 203,
          "v": 1600.38
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #203 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 9.6 사이클 소진 시 정비 권고 (90% 구간 1.0~25.0)",
        "status": "scheduled"
      }
    ],
    "status": "critical",
    "health": 8
  },
  "engine-42": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종 · 고장모드 1종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 156,
      "modelNo": "Turbofan (FD001)"
    },
    "predictedRulCycle": {
      "median": 10.0,
      "lower": 1.0,
      "upper": 25.0
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 0.9535,
        "lower": 0.9486,
        "upper": 0.9576
      },
      {
        "cycle": 10,
        "median": 0.4884,
        "lower": 0.4344,
        "upper": 0.5339
      },
      {
        "cycle": 20,
        "median": 0.1302,
        "lower": 0.0761,
        "upper": 0.1807
      },
      {
        "cycle": 30,
        "median": 0.0186,
        "lower": 0.0,
        "upper": 0.048
      },
      {
        "cycle": 40,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 60,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 80,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 100,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 120,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 140,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      }
    ],
    "sensorTrend": {
      "s4": [
        {
          "h": 1,
          "v": 1401.03
        },
        {
          "h": 20,
          "v": 1399.05
        },
        {
          "h": 40,
          "v": 1394.26
        },
        {
          "h": 59,
          "v": 1395.43
        },
        {
          "h": 79,
          "v": 1406.66
        },
        {
          "h": 98,
          "v": 1397.51
        },
        {
          "h": 117,
          "v": 1405.5
        },
        {
          "h": 137,
          "v": 1412.28
        },
        {
          "h": 156,
          "v": 1425.12
        }
      ],
      "s9": [
        {
          "h": 1,
          "v": 9055.08
        },
        {
          "h": 20,
          "v": 9048.56
        },
        {
          "h": 40,
          "v": 9048.32
        },
        {
          "h": 59,
          "v": 9046.22
        },
        {
          "h": 79,
          "v": 9047.07
        },
        {
          "h": 98,
          "v": 9041.46
        },
        {
          "h": 117,
          "v": 9046.54
        },
        {
          "h": 137,
          "v": 9039.98
        },
        {
          "h": 156,
          "v": 9026.89
        }
      ],
      "s3": [
        {
          "h": 1,
          "v": 1586.33
        },
        {
          "h": 20,
          "v": 1589.14
        },
        {
          "h": 40,
          "v": 1590.99
        },
        {
          "h": 59,
          "v": 1590.66
        },
        {
          "h": 79,
          "v": 1589.98
        },
        {
          "h": 98,
          "v": 1583.73
        },
        {
          "h": 117,
          "v": 1594.99
        },
        {
          "h": 137,
          "v": 1591.07
        },
        {
          "h": 156,
          "v": 1599.55
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #156 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 10.0 사이클 소진 시 정비 권고 (90% 구간 1.0~25.0)",
        "status": "scheduled"
      }
    ],
    "status": "critical",
    "health": 8
  },
  "engine-81": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종 · 고장모드 1종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 213,
      "modelNo": "Turbofan (FD001)"
    },
    "predictedRulCycle": {
      "median": 10.7,
      "lower": 1.0,
      "upper": 26.0
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 0.9619,
        "lower": 0.9544,
        "upper": 0.9699
      },
      {
        "cycle": 10,
        "median": 0.5135,
        "lower": 0.4604,
        "upper": 0.5576
      },
      {
        "cycle": 20,
        "median": 0.1502,
        "lower": 0.0918,
        "upper": 0.2051
      },
      {
        "cycle": 30,
        "median": 0.0179,
        "lower": 0.0,
        "upper": 0.0461
      },
      {
        "cycle": 40,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 60,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 80,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 100,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 120,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 140,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      }
    ],
    "sensorTrend": {
      "s4": [
        {
          "h": 1,
          "v": 1402.35
        },
        {
          "h": 27,
          "v": 1401.75
        },
        {
          "h": 54,
          "v": 1401.8
        },
        {
          "h": 81,
          "v": 1402.88
        },
        {
          "h": 107,
          "v": 1392.44
        },
        {
          "h": 133,
          "v": 1409.0
        },
        {
          "h": 160,
          "v": 1416.5
        },
        {
          "h": 187,
          "v": 1419.91
        },
        {
          "h": 213,
          "v": 1427.83
        }
      ],
      "s9": [
        {
          "h": 1,
          "v": 9065.91
        },
        {
          "h": 27,
          "v": 9063.92
        },
        {
          "h": 54,
          "v": 9065.22
        },
        {
          "h": 81,
          "v": 9065.01
        },
        {
          "h": 107,
          "v": 9071.65
        },
        {
          "h": 133,
          "v": 9063.52
        },
        {
          "h": 160,
          "v": 9077.21
        },
        {
          "h": 187,
          "v": 9066.9
        },
        {
          "h": 213,
          "v": 9075.26
        }
      ],
      "s3": [
        {
          "h": 1,
          "v": 1593.57
        },
        {
          "h": 27,
          "v": 1589.42
        },
        {
          "h": 54,
          "v": 1581.42
        },
        {
          "h": 81,
          "v": 1588.7
        },
        {
          "h": 107,
          "v": 1582.05
        },
        {
          "h": 133,
          "v": 1583.42
        },
        {
          "h": 160,
          "v": 1584.31
        },
        {
          "h": 187,
          "v": 1590.9
        },
        {
          "h": 213,
          "v": 1598.35
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #213 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 10.7 사이클 소진 시 정비 권고 (90% 구간 1.0~26.0)",
        "status": "scheduled"
      }
    ],
    "status": "critical",
    "health": 9
  },
  "engine-76": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종 · 고장모드 1종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 205,
      "modelNo": "Turbofan (FD001)"
    },
    "predictedRulCycle": {
      "median": 11.3,
      "lower": 1.0,
      "upper": 26.0
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 0.9626,
        "lower": 0.9554,
        "upper": 0.9704
      },
      {
        "cycle": 10,
        "median": 0.5231,
        "lower": 0.4722,
        "upper": 0.5663
      },
      {
        "cycle": 20,
        "median": 0.156,
        "lower": 0.0968,
        "upper": 0.2133
      },
      {
        "cycle": 30,
        "median": 0.0176,
        "lower": 0.0,
        "upper": 0.0451
      },
      {
        "cycle": 40,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 60,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 80,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 100,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 120,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 140,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      }
    ],
    "sensorTrend": {
      "s4": [
        {
          "h": 1,
          "v": 1395.95
        },
        {
          "h": 27,
          "v": 1398.03
        },
        {
          "h": 52,
          "v": 1398.51
        },
        {
          "h": 77,
          "v": 1403.38
        },
        {
          "h": 103,
          "v": 1401.18
        },
        {
          "h": 129,
          "v": 1404.09
        },
        {
          "h": 154,
          "v": 1408.39
        },
        {
          "h": 179,
          "v": 1418.3
        },
        {
          "h": 205,
          "v": 1420.07
        }
      ],
      "s9": [
        {
          "h": 1,
          "v": 9057.45
        },
        {
          "h": 27,
          "v": 9060.68
        },
        {
          "h": 52,
          "v": 9061.38
        },
        {
          "h": 77,
          "v": 9065.54
        },
        {
          "h": 103,
          "v": 9063.56
        },
        {
          "h": 129,
          "v": 9068.19
        },
        {
          "h": 154,
          "v": 9074.13
        },
        {
          "h": 179,
          "v": 9096.17
        },
        {
          "h": 205,
          "v": 9114.99
        }
      ],
      "s3": [
        {
          "h": 1,
          "v": 1583.52
        },
        {
          "h": 27,
          "v": 1586.12
        },
        {
          "h": 52,
          "v": 1576.04
        },
        {
          "h": 77,
          "v": 1581.95
        },
        {
          "h": 103,
          "v": 1589.78
        },
        {
          "h": 129,
          "v": 1582.11
        },
        {
          "h": 154,
          "v": 1590.15
        },
        {
          "h": 179,
          "v": 1595.51
        },
        {
          "h": 205,
          "v": 1603.48
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #205 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 11.3 사이클 소진 시 정비 권고 (90% 구간 1.0~26.0)",
        "status": "scheduled"
      }
    ],
    "status": "critical",
    "health": 9
  },
  "engine-32": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종 · 고장모드 1종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 145,
      "modelNo": "Turbofan (FD001)"
    },
    "predictedRulCycle": {
      "median": 41.6,
      "lower": 27.0,
      "upper": 55.3
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 1.0,
        "lower": 1.0,
        "upper": 1.0
      },
      {
        "cycle": 10,
        "median": 1.0,
        "lower": 1.0,
        "upper": 1.0
      },
      {
        "cycle": 20,
        "median": 0.9966,
        "lower": 0.9899,
        "upper": 1.0
      },
      {
        "cycle": 30,
        "median": 0.9116,
        "lower": 0.852,
        "upper": 0.9614
      },
      {
        "cycle": 40,
        "median": 0.483,
        "lower": 0.3858,
        "upper": 0.5667
      },
      {
        "cycle": 60,
        "median": 0.017,
        "lower": 0.0,
        "upper": 0.0452
      },
      {
        "cycle": 80,
        "median": 0.0034,
        "lower": 0.0,
        "upper": 0.0092
      },
      {
        "cycle": 100,
        "median": 0.0034,
        "lower": 0.0,
        "upper": 0.0092
      },
      {
        "cycle": 120,
        "median": 0.0034,
        "lower": 0.0,
        "upper": 0.0092
      },
      {
        "cycle": 140,
        "median": 0.0034,
        "lower": 0.0,
        "upper": 0.0092
      }
    ],
    "sensorTrend": {
      "s4": [
        {
          "h": 1,
          "v": 1400.1
        },
        {
          "h": 19,
          "v": 1393.45
        },
        {
          "h": 37,
          "v": 1397.16
        },
        {
          "h": 55,
          "v": 1396.31
        },
        {
          "h": 73,
          "v": 1399.36
        },
        {
          "h": 91,
          "v": 1395.72
        },
        {
          "h": 109,
          "v": 1395.8
        },
        {
          "h": 127,
          "v": 1400.5
        },
        {
          "h": 145,
          "v": 1406.22
        }
      ],
      "s9": [
        {
          "h": 1,
          "v": 9073.46
        },
        {
          "h": 19,
          "v": 9070.44
        },
        {
          "h": 37,
          "v": 9073.0
        },
        {
          "h": 55,
          "v": 9071.5
        },
        {
          "h": 73,
          "v": 9074.13
        },
        {
          "h": 91,
          "v": 9081.1
        },
        {
          "h": 109,
          "v": 9085.01
        },
        {
          "h": 127,
          "v": 9085.62
        },
        {
          "h": 145,
          "v": 9105.13
        }
      ],
      "s3": [
        {
          "h": 1,
          "v": 1580.94
        },
        {
          "h": 19,
          "v": 1583.3
        },
        {
          "h": 37,
          "v": 1581.43
        },
        {
          "h": 55,
          "v": 1579.59
        },
        {
          "h": 73,
          "v": 1578.16
        },
        {
          "h": 91,
          "v": 1590.79
        },
        {
          "h": 109,
          "v": 1589.32
        },
        {
          "h": 127,
          "v": 1589.43
        },
        {
          "h": 145,
          "v": 1587.37
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #145 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 41.6 사이클 소진 시 정비 권고 (90% 구간 27.0~55.3)",
        "status": "scheduled"
      }
    ],
    "status": "warning",
    "health": 33
  },
  "engine-46": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종 · 고장모드 1종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 146,
      "modelNo": "Turbofan (FD001)"
    },
    "predictedRulCycle": {
      "median": 45.1,
      "lower": 32.0,
      "upper": 65.4
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 1.0,
        "lower": 1.0,
        "upper": 1.0
      },
      {
        "cycle": 10,
        "median": 1.0,
        "lower": 1.0,
        "upper": 1.0
      },
      {
        "cycle": 20,
        "median": 1.0,
        "lower": 1.0,
        "upper": 1.0
      },
      {
        "cycle": 30,
        "median": 0.9707,
        "lower": 0.9412,
        "upper": 0.9929
      },
      {
        "cycle": 40,
        "median": 0.652,
        "lower": 0.5482,
        "upper": 0.7346
      },
      {
        "cycle": 60,
        "median": 0.0769,
        "lower": 0.0067,
        "upper": 0.1686
      },
      {
        "cycle": 80,
        "median": 0.011,
        "lower": 0.0035,
        "upper": 0.0203
      },
      {
        "cycle": 100,
        "median": 0.011,
        "lower": 0.0035,
        "upper": 0.0203
      },
      {
        "cycle": 120,
        "median": 0.011,
        "lower": 0.0035,
        "upper": 0.0203
      },
      {
        "cycle": 140,
        "median": 0.011,
        "lower": 0.0035,
        "upper": 0.0203
      }
    ],
    "sensorTrend": {
      "s4": [
        {
          "h": 1,
          "v": 1397.15
        },
        {
          "h": 19,
          "v": 1406.43
        },
        {
          "h": 37,
          "v": 1401.64
        },
        {
          "h": 55,
          "v": 1407.95
        },
        {
          "h": 73,
          "v": 1413.27
        },
        {
          "h": 92,
          "v": 1408.75
        },
        {
          "h": 110,
          "v": 1405.57
        },
        {
          "h": 128,
          "v": 1404.69
        },
        {
          "h": 146,
          "v": 1413.17
        }
      ],
      "s9": [
        {
          "h": 1,
          "v": 9065.72
        },
        {
          "h": 19,
          "v": 9061.02
        },
        {
          "h": 37,
          "v": 9069.18
        },
        {
          "h": 55,
          "v": 9073.34
        },
        {
          "h": 73,
          "v": 9060.11
        },
        {
          "h": 92,
          "v": 9069.98
        },
        {
          "h": 110,
          "v": 9080.79
        },
        {
          "h": 128,
          "v": 9082.89
        },
        {
          "h": 146,
          "v": 9080.61
        }
      ],
      "s3": [
        {
          "h": 1,
          "v": 1582.81
        },
        {
          "h": 19,
          "v": 1592.47
        },
        {
          "h": 37,
          "v": 1594.53
        },
        {
          "h": 55,
          "v": 1588.92
        },
        {
          "h": 73,
          "v": 1583.99
        },
        {
          "h": 92,
          "v": 1594.74
        },
        {
          "h": 110,
          "v": 1585.42
        },
        {
          "h": 128,
          "v": 1593.39
        },
        {
          "h": 146,
          "v": 1594.28
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #146 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 45.1 사이클 소진 시 정비 권고 (90% 구간 32.0~65.4)",
        "status": "scheduled"
      }
    ],
    "status": "warning",
    "health": 36
  },
  "engine-3": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종 · 고장모드 1종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 126,
      "modelNo": "Turbofan (FD001)"
    },
    "predictedRulCycle": {
      "median": 80.0,
      "lower": 53.0,
      "upper": 118.0
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 1.0,
        "lower": 1.0,
        "upper": 1.0
      },
      {
        "cycle": 10,
        "median": 1.0,
        "lower": 1.0,
        "upper": 1.0
      },
      {
        "cycle": 20,
        "median": 1.0,
        "lower": 1.0,
        "upper": 1.0
      },
      {
        "cycle": 30,
        "median": 1.0,
        "lower": 1.0,
        "upper": 1.0
      },
      {
        "cycle": 40,
        "median": 0.9974,
        "lower": 0.9924,
        "upper": 1.0
      },
      {
        "cycle": 60,
        "median": 0.75,
        "lower": 0.6569,
        "upper": 0.8251
      },
      {
        "cycle": 80,
        "median": 0.2908,
        "lower": 0.1908,
        "upper": 0.3902
      },
      {
        "cycle": 100,
        "median": 0.1199,
        "lower": 0.071,
        "upper": 0.1776
      },
      {
        "cycle": 120,
        "median": 0.0459,
        "lower": 0.0346,
        "upper": 0.0587
      },
      {
        "cycle": 140,
        "median": 0.0408,
        "lower": 0.0306,
        "upper": 0.0514
      }
    ],
    "sensorTrend": {
      "s4": [
        {
          "h": 1,
          "v": 1408.39
        },
        {
          "h": 17,
          "v": 1405.4
        },
        {
          "h": 32,
          "v": 1405.97
        },
        {
          "h": 48,
          "v": 1411.59
        },
        {
          "h": 63,
          "v": 1402.3
        },
        {
          "h": 79,
          "v": 1412.77
        },
        {
          "h": 95,
          "v": 1408.57
        },
        {
          "h": 110,
          "v": 1409.54
        },
        {
          "h": 126,
          "v": 1418.89
        }
      ],
      "s9": [
        {
          "h": 1,
          "v": 9053.65
        },
        {
          "h": 17,
          "v": 9051.43
        },
        {
          "h": 32,
          "v": 9054.94
        },
        {
          "h": 48,
          "v": 9056.76
        },
        {
          "h": 63,
          "v": 9050.53
        },
        {
          "h": 79,
          "v": 9052.76
        },
        {
          "h": 95,
          "v": 9049.88
        },
        {
          "h": 110,
          "v": 9046.74
        },
        {
          "h": 126,
          "v": 9049.26
        }
      ],
      "s3": [
        {
          "h": 1,
          "v": 1589.92
        },
        {
          "h": 17,
          "v": 1588.45
        },
        {
          "h": 32,
          "v": 1586.34
        },
        {
          "h": 48,
          "v": 1586.63
        },
        {
          "h": 63,
          "v": 1588.79
        },
        {
          "h": 79,
          "v": 1586.45
        },
        {
          "h": 95,
          "v": 1595.37
        },
        {
          "h": 110,
          "v": 1582.46
        },
        {
          "h": 126,
          "v": 1589.75
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #126 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 80.0 사이클 소진 시 정비 권고 (90% 구간 53.0~118.0)",
        "status": "scheduled"
      }
    ],
    "status": "good",
    "health": 64
  },
  "engine-47": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종 · 고장모드 1종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 73,
      "modelNo": "Turbofan (FD001)"
    },
    "predictedRulCycle": {
      "median": 125.0,
      "lower": 97.0,
      "upper": 222.0
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 1.0,
        "lower": 1.0,
        "upper": 1.0
      },
      {
        "cycle": 10,
        "median": 1.0,
        "lower": 1.0,
        "upper": 1.0
      },
      {
        "cycle": 20,
        "median": 1.0,
        "lower": 1.0,
        "upper": 1.0
      },
      {
        "cycle": 30,
        "median": 1.0,
        "lower": 1.0,
        "upper": 1.0
      },
      {
        "cycle": 40,
        "median": 1.0,
        "lower": 1.0,
        "upper": 1.0
      },
      {
        "cycle": 60,
        "median": 1.0,
        "lower": 1.0,
        "upper": 1.0
      },
      {
        "cycle": 80,
        "median": 0.9949,
        "lower": 0.9888,
        "upper": 1.0
      },
      {
        "cycle": 100,
        "median": 0.9395,
        "lower": 0.8931,
        "upper": 0.9812
      },
      {
        "cycle": 120,
        "median": 0.8451,
        "lower": 0.7539,
        "upper": 0.9286
      },
      {
        "cycle": 140,
        "median": 0.6779,
        "lower": 0.5524,
        "upper": 0.7949
      }
    ],
    "sensorTrend": {
      "s4": [
        {
          "h": 1,
          "v": 1415.87
        },
        {
          "h": 10,
          "v": 1418.18
        },
        {
          "h": 19,
          "v": 1413.49
        },
        {
          "h": 28,
          "v": 1411.35
        },
        {
          "h": 37,
          "v": 1410.57
        },
        {
          "h": 46,
          "v": 1415.89
        },
        {
          "h": 55,
          "v": 1408.82
        },
        {
          "h": 64,
          "v": 1402.61
        },
        {
          "h": 73,
          "v": 1402.79
        }
      ],
      "s9": [
        {
          "h": 1,
          "v": 9045.79
        },
        {
          "h": 10,
          "v": 9048.75
        },
        {
          "h": 19,
          "v": 9046.35
        },
        {
          "h": 28,
          "v": 9051.07
        },
        {
          "h": 37,
          "v": 9041.39
        },
        {
          "h": 46,
          "v": 9040.82
        },
        {
          "h": 55,
          "v": 9043.84
        },
        {
          "h": 64,
          "v": 9044.41
        },
        {
          "h": 73,
          "v": 9038.32
        }
      ],
      "s3": [
        {
          "h": 1,
          "v": 1586.87
        },
        {
          "h": 10,
          "v": 1592.77
        },
        {
          "h": 19,
          "v": 1593.85
        },
        {
          "h": 28,
          "v": 1593.51
        },
        {
          "h": 37,
          "v": 1595.63
        },
        {
          "h": 46,
          "v": 1591.5
        },
        {
          "h": 55,
          "v": 1584.69
        },
        {
          "h": 64,
          "v": 1584.39
        },
        {
          "h": 73,
          "v": 1587.09
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #73 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 125.0 사이클 소진 시 정비 권고 (90% 구간 97.0~222.0)",
        "status": "scheduled"
      }
    ],
    "status": "good",
    "health": 100
  }
}

export const DEFAULT_ENGINE_ID = 'engine-41'

/** 센서 추이 3종 (06번 RF 중요도 상위 센서) — 값은 원본값 */
export const trendMeta: Record<string, { unit: string; label: string; domain: [number, number] }> = {
  "s4": {
    "unit": "°R",
    "label": "T50 · LPT outlet 온도 (s4)",
    "domain": [
      1376.3,
      1447.4
    ]
  },
  "s9": {
    "unit": "rpm",
    "label": "Nc · 물리적 코어 속도 (s9)",
    "domain": [
      8999.4,
      9266.9
    ]
  },
  "s3": {
    "unit": "°R",
    "label": "T30 · HPC outlet 온도 (s3)",
    "domain": [
      1566.5,
      1621.5
    ]
  }
}

/** 위험 등급(RUL≤40) 조기경보 성능 — LSTM 모델, 공식 test 100개 엔진 기준 실제 계산값
 *  (TP=28, FN=0, FP=1) */
export const classifierMetrics = {
  "precision": 0.966,
  "recall": 1.0,
  "f1": 0.982,
  "tp": 28,
  "fn": 0,
  "fp": 1
}

/** RF 피처 중요도 상위 5개 센서(그룹 합산) */
export const featureImportance = [
  {
    "label": "T50 · LPT outlet 온도 (s4)",
    "value": 0.589
  },
  {
    "label": "Nc · 물리적 코어 속도 (s9)",
    "value": 0.094
  },
  {
    "label": "T30 · HPC outlet 온도 (s3)",
    "value": 0.067
  },
  {
    "label": "Ps30 · HPC outlet 정압 (s11)",
    "value": 0.053
  },
  {
    "label": "W32 · LPT 냉각 블리드 유량 (s21)",
    "value": 0.037
  }
]

/** AS-IS(사후 정비: 고장까지 운용) vs TO-BE(예지보전: RUL 기반 사전 정비) 비용 시뮬레이션.
 *  탐지/누락/오탐 건수는 LSTM 모델의 실제 test set 결과, 금액·시간 가정은 항공 엔진 정비
 *  맥락의 illustrative 값입니다 (분석 레포 scripts/08_export_dashboard_ts.py 참고). */
export const scenarioCompare = [
  {
    "metric": "평균 가동중단 시간 (시간/1,000대 환산)",
    "asIs": 6720.0,
    "toBe": 1125.0
  },
  {
    "metric": "정비 비용 (억원/1,000대 환산)",
    "asIs": 67.2,
    "toBe": 22.4
  },
  {
    "metric": "가동중단 손실 (억원/1,000대 환산)",
    "asIs": 672.0,
    "toBe": 112.5
  }
]

export const savingsSummary = {
  "perThousandEnginesEok": 604.2,
  "savingRatePct": 81.7,
  "dangerCaseReductionPct": 100,
  "note": "테스트 엔진 100대 결과를 1,000대 규모로 환산한 값입니다. 정비 비용·가동중단 단가는 가정값이고, 탐지/누락/오탐 건수는 실제 모델 성능입니다."
}

/** 시나리오 화면 슬라이더가 다시 계산할 때 쓰는 원본 가정치 (계산식은 분석 레포 compute_scenario()와 동일) */
export const costModel = {
  "nEngines": 100,
  "general": {
    "downtimeCostPerHour": 1000.0,
    "repairUnplannedRatio": 3.0,
    "dangerCaseLoss": 0.0,
    "falseAlarmHours": 0.5,
    "falseAlarmLabor": 50.0
  },
  "types": {
    "EngineRemoval": {
      "episodes": 28,
      "detected": 28,
      "missed": 0,
      "falseAlarms": 1,
      "dangerCases": 28,
      "dangerCasesDetected": 0,
      "dangerCasesMissed": 0,
      "hoursPlanned": 4.0,
      "hoursUnplanned": 24.0,
      "repairPlanned": 800.0
    }
  }
}
