// ⚠️ 자동 생성 파일 — 분석 레포의 scripts/08_export_dashboard_ts.py 가 만든다.
// 직접 고치지 말고 파이썬 파이프라인을 다시 돌려서 새로 받아올 것.
// 원본 데이터: C-MAPSS FD001 (NASA 터보팬 엔진 열화 시뮬레이션), 모델: LSTM Seq2Seq
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
  "generatedAt": "2026-09-30T10:59:04",
  "dataset": "C-MAPSS FD001",
  "testEngines": 100,
  "selectedEngines": [
    34,
    42,
    81,
    76,
    37,
    91,
    62,
    43,
    47
  ],
  "featuredEngine": 37,
  "note": "시간축은 엔진 운행 사이클(cycle). 위험 등급(RED/YELLOW/GREEN)은 예측 RUL 기준(≤30/<60/그 외). 비용·정비시간은 가정값(08_export_dashboard_ts.py COST_ASSUMPTIONS), 탐지 성능(TP/FN/FP)은 LSTM 모델의 실제 test set 결과. RUL 신뢰구간·생존곡선은 검증셋에서 예측이 비슷했던 사례들의 실제 RUL 분포."
}

/** 신호등 기준 (분석 레포 common.py의 DANGER_RUL · RISK_THRESHOLDS) — 화면 문구도 이 값을 씀
 *  위험: 예측 RUL ≤ dangerRul, 주의: dangerRul 초과 ~ warningRul 미만, 정상: warningRul 이상 */
export const riskThresholds = { dangerRul: 30, warningRul: 60 }

/** 메인 차트: 대표 엔진(#37)의 예측 RUL 추이 — h = 창 안의 사이클 순서(0~24) */
export const sensorSeries: { t: string; h: number; v: number }[] = [
  {
    "t": "#97",
    "h": 0,
    "v": 77.2
  },
  {
    "t": "#98",
    "h": 1,
    "v": 72.93
  },
  {
    "t": "#99",
    "h": 2,
    "v": 69.78
  },
  {
    "t": "#100",
    "h": 3,
    "v": 67.03
  },
  {
    "t": "#101",
    "h": 4,
    "v": 62.7
  },
  {
    "t": "#102",
    "h": 5,
    "v": 62.58
  },
  {
    "t": "#103",
    "h": 6,
    "v": 67.92
  },
  {
    "t": "#104",
    "h": 7,
    "v": 62.42
  },
  {
    "t": "#105",
    "h": 8,
    "v": 58.88
  },
  {
    "t": "#106",
    "h": 9,
    "v": 56.23
  },
  {
    "t": "#107",
    "h": 10,
    "v": 58.44
  },
  {
    "t": "#108",
    "h": 11,
    "v": 42.65
  },
  {
    "t": "#109",
    "h": 12,
    "v": 34.39
  },
  {
    "t": "#110",
    "h": 13,
    "v": 29.99
  },
  {
    "t": "#111",
    "h": 14,
    "v": 22.41
  },
  {
    "t": "#112",
    "h": 15,
    "v": 18.63
  },
  {
    "t": "#113",
    "h": 16,
    "v": 14.03
  },
  {
    "t": "#114",
    "h": 17,
    "v": 12.68
  },
  {
    "t": "#115",
    "h": 18,
    "v": 11.67
  },
  {
    "t": "#116",
    "h": 19,
    "v": 13.16
  },
  {
    "t": "#117",
    "h": 20,
    "v": 12.52
  },
  {
    "t": "#118",
    "h": 21,
    "v": 12.07
  },
  {
    "t": "#119",
    "h": 22,
    "v": 12.42
  },
  {
    "t": "#120",
    "h": 23,
    "v": 14.27
  },
  {
    "t": "#121",
    "h": 24,
    "v": 15.17
  }
]

export const anomalyWindow = {
  "start": 13,
  "end": 24
}

const START_CYCLE: number = 97

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
    "id": "engine-37",
    "name": "Engine #37",
    "health": 12,
    "status": "critical"
  },
  {
    "id": "engine-91",
    "name": "Engine #91",
    "health": 26,
    "status": "warning"
  },
  {
    "id": "engine-62",
    "name": "Engine #62",
    "health": 26,
    "status": "warning"
  },
  {
    "id": "engine-43",
    "name": "Engine #43",
    "health": 48,
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

/** 대표 엔진(#37)의 예측 RUL이 주의(60 미만)/위험(30 이하) 임계값을 넘은 시점 — 실제 계산값 */
export const alertLog: AlertRow[] = [
  {
    "h": 13,
    "time": "cycle #110",
    "equipment": "Engine #37",
    "sensor": "예측 RUL",
    "severity": "critical",
    "action": "정비 일정 즉시 수립",
    "type": "EngineRemoval"
  },
  {
    "h": 8,
    "time": "cycle #105",
    "equipment": "Engine #37",
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
  "engine-34": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종)",
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
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종)",
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
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종)",
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
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종)",
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
  "engine-37": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 121,
      "modelNo": "Turbofan (FD001)"
    },
    "predictedRulCycle": {
      "median": 15.2,
      "lower": 2.0,
      "upper": 29.0
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 0.9838,
        "lower": 0.9754,
        "upper": 0.991
      },
      {
        "cycle": 10,
        "median": 0.6349,
        "lower": 0.5897,
        "upper": 0.6758
      },
      {
        "cycle": 20,
        "median": 0.2475,
        "lower": 0.1866,
        "upper": 0.3046
      },
      {
        "cycle": 30,
        "median": 0.0365,
        "lower": 0.0104,
        "upper": 0.0669
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
          "v": 1396.14
        },
        {
          "h": 16,
          "v": 1393.71
        },
        {
          "h": 31,
          "v": 1398.58
        },
        {
          "h": 46,
          "v": 1404.38
        },
        {
          "h": 61,
          "v": 1397.56
        },
        {
          "h": 76,
          "v": 1405.52
        },
        {
          "h": 91,
          "v": 1409.21
        },
        {
          "h": 106,
          "v": 1413.67
        },
        {
          "h": 121,
          "v": 1413.38
        }
      ],
      "s9": [
        {
          "h": 1,
          "v": 9051.31
        },
        {
          "h": 16,
          "v": 9052.93
        },
        {
          "h": 31,
          "v": 9059.87
        },
        {
          "h": 46,
          "v": 9059.41
        },
        {
          "h": 61,
          "v": 9049.06
        },
        {
          "h": 76,
          "v": 9052.06
        },
        {
          "h": 91,
          "v": 9053.14
        },
        {
          "h": 106,
          "v": 9049.81
        },
        {
          "h": 121,
          "v": 9041.71
        }
      ],
      "s3": [
        {
          "h": 1,
          "v": 1591.3
        },
        {
          "h": 16,
          "v": 1583.34
        },
        {
          "h": 31,
          "v": 1586.18
        },
        {
          "h": 46,
          "v": 1591.41
        },
        {
          "h": 61,
          "v": 1585.43
        },
        {
          "h": 76,
          "v": 1583.69
        },
        {
          "h": 91,
          "v": 1587.53
        },
        {
          "h": 106,
          "v": 1585.11
        },
        {
          "h": 121,
          "v": 1593.15
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #121 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 15.2 사이클 소진 시 정비 권고 (90% 구간 2.0~29.0)",
        "status": "scheduled"
      }
    ],
    "status": "critical",
    "health": 12
  },
  "engine-91": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 234,
      "modelNo": "Turbofan (FD001)"
    },
    "predictedRulCycle": {
      "median": 31.9,
      "lower": 19.4,
      "upper": 47.0
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
        "median": 0.9348,
        "lower": 0.8918,
        "upper": 0.9734
      },
      {
        "cycle": 30,
        "median": 0.5788,
        "lower": 0.4914,
        "upper": 0.6642
      },
      {
        "cycle": 40,
        "median": 0.1603,
        "lower": 0.0816,
        "upper": 0.2391
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
          "v": 1403.1
        },
        {
          "h": 30,
          "v": 1405.01
        },
        {
          "h": 59,
          "v": 1407.19
        },
        {
          "h": 88,
          "v": 1411.19
        },
        {
          "h": 117,
          "v": 1407.59
        },
        {
          "h": 147,
          "v": 1411.37
        },
        {
          "h": 176,
          "v": 1425.03
        },
        {
          "h": 205,
          "v": 1417.68
        },
        {
          "h": 234,
          "v": 1419.74
        }
      ],
      "s9": [
        {
          "h": 1,
          "v": 9053.39
        },
        {
          "h": 30,
          "v": 9053.59
        },
        {
          "h": 59,
          "v": 9048.82
        },
        {
          "h": 88,
          "v": 9042.95
        },
        {
          "h": 117,
          "v": 9046.86
        },
        {
          "h": 147,
          "v": 9045.86
        },
        {
          "h": 176,
          "v": 9048.67
        },
        {
          "h": 205,
          "v": 9049.44
        },
        {
          "h": 234,
          "v": 9052.13
        }
      ],
      "s3": [
        {
          "h": 1,
          "v": 1591.39
        },
        {
          "h": 30,
          "v": 1591.83
        },
        {
          "h": 59,
          "v": 1598.0
        },
        {
          "h": 88,
          "v": 1592.07
        },
        {
          "h": 117,
          "v": 1591.37
        },
        {
          "h": 147,
          "v": 1594.28
        },
        {
          "h": 176,
          "v": 1588.15
        },
        {
          "h": 205,
          "v": 1597.16
        },
        {
          "h": 234,
          "v": 1605.05
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #234 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 31.9 사이클 소진 시 정비 권고 (90% 구간 19.4~47.0)",
        "status": "scheduled"
      }
    ],
    "status": "warning",
    "health": 26
  },
  "engine-62": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 232,
      "modelNo": "Turbofan (FD001)"
    },
    "predictedRulCycle": {
      "median": 32.8,
      "lower": 21.0,
      "upper": 47.3
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
        "median": 0.9548,
        "lower": 0.9226,
        "upper": 0.9862
      },
      {
        "cycle": 30,
        "median": 0.6158,
        "lower": 0.5344,
        "upper": 0.698
      },
      {
        "cycle": 40,
        "median": 0.1751,
        "lower": 0.0965,
        "upper": 0.2566
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
          "v": 1393.21
        },
        {
          "h": 30,
          "v": 1393.17
        },
        {
          "h": 59,
          "v": 1398.03
        },
        {
          "h": 88,
          "v": 1398.21
        },
        {
          "h": 117,
          "v": 1399.34
        },
        {
          "h": 145,
          "v": 1399.97
        },
        {
          "h": 174,
          "v": 1410.83
        },
        {
          "h": 203,
          "v": 1406.82
        },
        {
          "h": 232,
          "v": 1407.86
        }
      ],
      "s9": [
        {
          "h": 1,
          "v": 9057.4
        },
        {
          "h": 30,
          "v": 9052.16
        },
        {
          "h": 59,
          "v": 9066.55
        },
        {
          "h": 88,
          "v": 9054.86
        },
        {
          "h": 117,
          "v": 9055.98
        },
        {
          "h": 145,
          "v": 9064.18
        },
        {
          "h": 174,
          "v": 9064.88
        },
        {
          "h": 203,
          "v": 9074.77
        },
        {
          "h": 232,
          "v": 9096.2
        }
      ],
      "s3": [
        {
          "h": 1,
          "v": 1584.09
        },
        {
          "h": 30,
          "v": 1583.95
        },
        {
          "h": 59,
          "v": 1586.31
        },
        {
          "h": 88,
          "v": 1577.81
        },
        {
          "h": 117,
          "v": 1583.13
        },
        {
          "h": 145,
          "v": 1586.28
        },
        {
          "h": 174,
          "v": 1585.44
        },
        {
          "h": 203,
          "v": 1584.76
        },
        {
          "h": 232,
          "v": 1594.78
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #232 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 32.8 사이클 소진 시 정비 권고 (90% 구간 21.0~47.3)",
        "status": "scheduled"
      }
    ],
    "status": "warning",
    "health": 26
  },
  "engine-43": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 172,
      "modelNo": "Turbofan (FD001)"
    },
    "predictedRulCycle": {
      "median": 60.1,
      "lower": 40.0,
      "upper": 81.0
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
        "median": 0.9419,
        "lower": 0.8986,
        "upper": 0.9791
      },
      {
        "cycle": 60,
        "median": 0.2946,
        "lower": 0.1806,
        "upper": 0.4097
      },
      {
        "cycle": 80,
        "median": 0.0539,
        "lower": 0.0376,
        "upper": 0.0717
      },
      {
        "cycle": 100,
        "median": 0.0456,
        "lower": 0.0292,
        "upper": 0.0636
      },
      {
        "cycle": 120,
        "median": 0.0456,
        "lower": 0.0292,
        "upper": 0.0636
      },
      {
        "cycle": 140,
        "median": 0.0456,
        "lower": 0.0292,
        "upper": 0.0636
      }
    ],
    "sensorTrend": {
      "s4": [
        {
          "h": 1,
          "v": 1402.16
        },
        {
          "h": 22,
          "v": 1402.72
        },
        {
          "h": 44,
          "v": 1401.53
        },
        {
          "h": 65,
          "v": 1404.99
        },
        {
          "h": 87,
          "v": 1402.32
        },
        {
          "h": 108,
          "v": 1403.49
        },
        {
          "h": 129,
          "v": 1408.38
        },
        {
          "h": 151,
          "v": 1406.72
        },
        {
          "h": 172,
          "v": 1418.4
        }
      ],
      "s9": [
        {
          "h": 1,
          "v": 9047.72
        },
        {
          "h": 22,
          "v": 9047.82
        },
        {
          "h": 44,
          "v": 9052.74
        },
        {
          "h": 65,
          "v": 9050.76
        },
        {
          "h": 87,
          "v": 9053.0
        },
        {
          "h": 108,
          "v": 9049.12
        },
        {
          "h": 129,
          "v": 9052.88
        },
        {
          "h": 151,
          "v": 9059.19
        },
        {
          "h": 172,
          "v": 9052.93
        }
      ],
      "s3": [
        {
          "h": 1,
          "v": 1588.76
        },
        {
          "h": 22,
          "v": 1588.7
        },
        {
          "h": 44,
          "v": 1588.3
        },
        {
          "h": 65,
          "v": 1582.45
        },
        {
          "h": 87,
          "v": 1591.45
        },
        {
          "h": 108,
          "v": 1583.13
        },
        {
          "h": 129,
          "v": 1589.55
        },
        {
          "h": 151,
          "v": 1591.29
        },
        {
          "h": 172,
          "v": 1594.19
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #172 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 60.1 사이클 소진 시 정비 권고 (90% 구간 40.0~81.0)",
        "status": "scheduled"
      }
    ],
    "status": "good",
    "health": 48
  },
  "engine-47": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종)",
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

export const DEFAULT_ENGINE_ID = 'engine-37'

export const trendMeta: Record<string, { unit: string; domain: [number, number] }> = {
  "s4": {
    "unit": "°R",
    "domain": [
      1376.3,
      1447.4
    ]
  },
  "s9": {
    "unit": "rpm",
    "domain": [
      8999.4,
      9266.9
    ]
  },
  "s3": {
    "unit": "°R",
    "domain": [
      1566.5,
      1621.5
    ]
  }
}

/** 위험 등급(RUL≤30) 조기경보 성능 — LSTM 모델, 공식 test 100개 엔진 기준 실제 계산값
 *  (TP=24, FN=1, FP=1) */
export const classifierMetrics = {
  "precision": 0.96,
  "recall": 0.96,
  "f1": 0.96,
  "tp": 24,
  "fn": 1,
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
    "asIs": 6000.0,
    "toBe": 1205.0
  },
  {
    "metric": "정비 비용 (억원/1,000대 환산)",
    "asIs": 60.0,
    "toBe": 21.6
  },
  {
    "metric": "가동중단 손실 (억원/1,000대 환산)",
    "asIs": 600.0,
    "toBe": 120.5
  }
]

export const savingsSummary = {
  "perThousandEnginesEok": 517.8,
  "savingRatePct": 78.5,
  "dangerCaseReductionPct": 96,
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
      "episodes": 25,
      "detected": 24,
      "missed": 1,
      "falseAlarms": 1,
      "dangerCases": 25,
      "dangerCasesDetected": 0,
      "dangerCasesMissed": 1,
      "hoursPlanned": 4.0,
      "hoursUnplanned": 24.0,
      "repairPlanned": 800.0
    }
  }
}
