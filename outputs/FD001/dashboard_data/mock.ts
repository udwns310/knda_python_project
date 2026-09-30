// ⚠️ 자동 생성 파일 — 분석 레포의 scripts/08_export_dashboard_ts.py 가 만든다.
// 직접 고치지 말고 파이썬 파이프라인을 다시 돌려서 새로 받아올 것.
// 원본 데이터: C-MAPSS FD001 (운전조건 1종 · 고장모드 1종, NASA 터보팬 엔진 열화 시뮬레이션), 모델: GRU Seq2Seq 회귀
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
  "generatedAt": "2026-09-30T14:28:08",
  "dataset": "C-MAPSS FD001",
  "datasetDescription": "운전조건 1종 · 고장모드 1종",
  "sensorValues": "원본값",
  "model": "GRU",
  "testEngines": 100,
  "selectedEngines": [
    34,
    81,
    76,
    42,
    18,
    62,
    58,
    27,
    78
  ],
  "featuredEngine": 18,
  "note": "시간축은 엔진 운행 사이클(cycle). 위험 등급(RED/YELLOW/GREEN)은 예측 RUL 기준(≤40/<80/그 외). 비용·정비시간은 가정값(08_export_dashboard_ts.py COST_ASSUMPTIONS), 탐지 성능(TP/FN/FP)은 GRU 모델의 실제 test set 결과. RUL 신뢰구간·생존곡선은 검증셋에서 예측이 비슷했던 사례들의 실제 RUL 분포."
}

/** 신호등 기준 (분석 레포 common.py의 DANGER_RUL · RISK_THRESHOLDS) — 화면 문구도 이 값을 씀
 *  위험: 예측 RUL ≤ dangerRul, 주의: dangerRul 초과 ~ warningRul 미만, 정상: warningRul 이상 */
export const riskThresholds = { dangerRul: 40, warningRul: 80 }

/** 메인 차트: 대표 엔진(#18)의 예측 RUL 추이 — h = 창 안의 사이클 순서(0~24) */
export const sensorSeries: { t: string; h: number; v: number }[] = [
  {
    "t": "#109",
    "h": 0,
    "v": 82.63
  },
  {
    "t": "#110",
    "h": 1,
    "v": 81.86
  },
  {
    "t": "#111",
    "h": 2,
    "v": 82.22
  },
  {
    "t": "#112",
    "h": 3,
    "v": 79.05
  },
  {
    "t": "#113",
    "h": 4,
    "v": 72.72
  },
  {
    "t": "#114",
    "h": 5,
    "v": 74.03
  },
  {
    "t": "#115",
    "h": 6,
    "v": 72.83
  },
  {
    "t": "#116",
    "h": 7,
    "v": 72.94
  },
  {
    "t": "#117",
    "h": 8,
    "v": 69.59
  },
  {
    "t": "#118",
    "h": 9,
    "v": 63.77
  },
  {
    "t": "#119",
    "h": 10,
    "v": 57.43
  },
  {
    "t": "#120",
    "h": 11,
    "v": 55.72
  },
  {
    "t": "#121",
    "h": 12,
    "v": 48.09
  },
  {
    "t": "#122",
    "h": 13,
    "v": 43.8
  },
  {
    "t": "#123",
    "h": 14,
    "v": 47.09
  },
  {
    "t": "#124",
    "h": 15,
    "v": 43.07
  },
  {
    "t": "#125",
    "h": 16,
    "v": 41.33
  },
  {
    "t": "#126",
    "h": 17,
    "v": 34.33
  },
  {
    "t": "#127",
    "h": 18,
    "v": 31.54
  },
  {
    "t": "#128",
    "h": 19,
    "v": 28.78
  },
  {
    "t": "#129",
    "h": 20,
    "v": 25.19
  },
  {
    "t": "#130",
    "h": 21,
    "v": 24.62
  },
  {
    "t": "#131",
    "h": 22,
    "v": 25.65
  },
  {
    "t": "#132",
    "h": 23,
    "v": 25.66
  },
  {
    "t": "#133",
    "h": 24,
    "v": 23.01
  }
]

export const anomalyWindow = {
  "start": 17,
  "end": 24
}

const START_CYCLE: number = 109

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
    "health": 6,
    "status": "critical"
  },
  {
    "id": "engine-81",
    "name": "Engine #81",
    "health": 8,
    "status": "critical"
  },
  {
    "id": "engine-76",
    "name": "Engine #76",
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
    "id": "engine-18",
    "name": "Engine #18",
    "health": 18,
    "status": "critical"
  },
  {
    "id": "engine-62",
    "name": "Engine #62",
    "health": 33,
    "status": "warning"
  },
  {
    "id": "engine-58",
    "name": "Engine #58",
    "health": 36,
    "status": "warning"
  },
  {
    "id": "engine-27",
    "name": "Engine #27",
    "health": 65,
    "status": "good"
  },
  {
    "id": "engine-78",
    "name": "Engine #78",
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

/** 대표 엔진(#18)의 예측 RUL이 주의(80 미만)/위험(40 이하) 임계값을 넘은 시점 — 실제 계산값 */
export const alertLog: AlertRow[] = [
  {
    "h": 17,
    "time": "cycle #126",
    "equipment": "Engine #18",
    "sensor": "예측 RUL",
    "severity": "critical",
    "action": "정비 일정 즉시 수립",
    "type": "EngineRemoval"
  },
  {
    "h": 3,
    "time": "cycle #112",
    "equipment": "Engine #18",
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
  "engine-18": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종 · 고장모드 1종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 133,
      "modelNo": "Turbofan (FD001)"
    },
    "predictedRulCycle": {
      "median": 23.0,
      "lower": 12.3,
      "upper": 36.0
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
        "median": 0.9829,
        "lower": 0.9675,
        "upper": 0.9956
      },
      {
        "cycle": 20,
        "median": 0.636,
        "lower": 0.6013,
        "upper": 0.6703
      },
      {
        "cycle": 30,
        "median": 0.2184,
        "lower": 0.1754,
        "upper": 0.2579
      },
      {
        "cycle": 40,
        "median": 0.0107,
        "lower": 0.0,
        "upper": 0.0259
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
          "v": 1395.19
        },
        {
          "h": 17,
          "v": 1398.77
        },
        {
          "h": 34,
          "v": 1395.26
        },
        {
          "h": 51,
          "v": 1408.2
        },
        {
          "h": 67,
          "v": 1392.12
        },
        {
          "h": 83,
          "v": 1402.25
        },
        {
          "h": 100,
          "v": 1399.01
        },
        {
          "h": 117,
          "v": 1410.29
        },
        {
          "h": 133,
          "v": 1419.18
        }
      ],
      "s9": [
        {
          "h": 1,
          "v": 9074.23
        },
        {
          "h": 17,
          "v": 9061.88
        },
        {
          "h": 34,
          "v": 9060.28
        },
        {
          "h": 51,
          "v": 9061.96
        },
        {
          "h": 67,
          "v": 9063.69
        },
        {
          "h": 83,
          "v": 9059.28
        },
        {
          "h": 100,
          "v": 9057.78
        },
        {
          "h": 117,
          "v": 9062.13
        },
        {
          "h": 133,
          "v": 9056.09
        }
      ],
      "s3": [
        {
          "h": 1,
          "v": 1586.9
        },
        {
          "h": 17,
          "v": 1590.23
        },
        {
          "h": 34,
          "v": 1591.52
        },
        {
          "h": 51,
          "v": 1583.99
        },
        {
          "h": 67,
          "v": 1586.89
        },
        {
          "h": 83,
          "v": 1593.37
        },
        {
          "h": 100,
          "v": 1593.49
        },
        {
          "h": 117,
          "v": 1591.96
        },
        {
          "h": 133,
          "v": 1600.45
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #133 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 23.0 사이클 소진 시 정비 권고 (90% 구간 12.3~36.0)",
        "status": "scheduled"
      }
    ],
    "status": "critical",
    "health": 18
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
      "median": 6.9,
      "lower": 0.0,
      "upper": 17.0
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 0.9403,
        "lower": 0.9369,
        "upper": 0.9435
      },
      {
        "cycle": 10,
        "median": 0.3433,
        "lower": 0.306,
        "upper": 0.3785
      },
      {
        "cycle": 20,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
      },
      {
        "cycle": 30,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
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
        "description": "RUL 예측 기반 — 잔존 약 6.9 사이클 소진 시 정비 권고 (90% 구간 0.0~17.0)",
        "status": "scheduled"
      }
    ],
    "status": "critical",
    "health": 6
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
      "median": 9.4,
      "lower": 1.0,
      "upper": 20.0
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 0.9504,
        "lower": 0.9475,
        "upper": 0.9529
      },
      {
        "cycle": 10,
        "median": 0.4541,
        "lower": 0.4226,
        "upper": 0.4824
      },
      {
        "cycle": 20,
        "median": 0.0471,
        "lower": 0.0229,
        "upper": 0.0753
      },
      {
        "cycle": 30,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
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
        "description": "RUL 예측 기반 — 잔존 약 9.4 사이클 소진 시 정비 권고 (90% 구간 1.0~20.0)",
        "status": "scheduled"
      }
    ],
    "status": "critical",
    "health": 8
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
      "median": 10.3,
      "lower": 1.0,
      "upper": 22.0
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 0.9549,
        "lower": 0.95,
        "upper": 0.9604
      },
      {
        "cycle": 10,
        "median": 0.4822,
        "lower": 0.4458,
        "upper": 0.5147
      },
      {
        "cycle": 20,
        "median": 0.0784,
        "lower": 0.0443,
        "upper": 0.1166
      },
      {
        "cycle": 30,
        "median": 0.0,
        "lower": 0.0,
        "upper": 0.0
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
        "description": "RUL 예측 기반 — 잔존 약 10.3 사이클 소진 시 정비 권고 (90% 구간 1.0~22.0)",
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
      "median": 10.5,
      "lower": 1.0,
      "upper": 22.0
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 0.9552,
        "lower": 0.9504,
        "upper": 0.9608
      },
      {
        "cycle": 10,
        "median": 0.4858,
        "lower": 0.4508,
        "upper": 0.5198
      },
      {
        "cycle": 20,
        "median": 0.0825,
        "lower": 0.0467,
        "upper": 0.1234
      },
      {
        "cycle": 30,
        "median": 0.0024,
        "lower": 0.0,
        "upper": 0.0068
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
        "description": "RUL 예측 기반 — 잔존 약 10.5 사이클 소진 시 정비 권고 (90% 구간 1.0~22.0)",
        "status": "scheduled"
      }
    ],
    "status": "critical",
    "health": 8
  },
  "engine-62": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종 · 고장모드 1종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 232,
      "modelNo": "Turbofan (FD001)"
    },
    "predictedRulCycle": {
      "median": 41.0,
      "lower": 30.0,
      "upper": 52.0
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
        "median": 0.9429,
        "lower": 0.9007,
        "upper": 0.9777
      },
      {
        "cycle": 40,
        "median": 0.5111,
        "lower": 0.4312,
        "upper": 0.5911
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
        "description": "RUL 예측 기반 — 잔존 약 41.0 사이클 소진 시 정비 권고 (90% 구간 30.0~52.0)",
        "status": "scheduled"
      }
    ],
    "status": "warning",
    "health": 33
  },
  "engine-58": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종 · 고장모드 1종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 176,
      "modelNo": "Turbofan (FD001)"
    },
    "predictedRulCycle": {
      "median": 44.7,
      "lower": 35.0,
      "upper": 55.1
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
        "median": 0.7066,
        "lower": 0.6326,
        "upper": 0.7766
      },
      {
        "cycle": 60,
        "median": 0.0077,
        "lower": 0.0,
        "upper": 0.0163
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
          "v": 1397.36
        },
        {
          "h": 23,
          "v": 1396.43
        },
        {
          "h": 45,
          "v": 1394.24
        },
        {
          "h": 67,
          "v": 1405.36
        },
        {
          "h": 89,
          "v": 1406.31
        },
        {
          "h": 110,
          "v": 1407.87
        },
        {
          "h": 132,
          "v": 1407.41
        },
        {
          "h": 154,
          "v": 1416.26
        },
        {
          "h": 176,
          "v": 1418.42
        }
      ],
      "s9": [
        {
          "h": 1,
          "v": 9052.53
        },
        {
          "h": 23,
          "v": 9051.56
        },
        {
          "h": 45,
          "v": 9047.34
        },
        {
          "h": 67,
          "v": 9055.74
        },
        {
          "h": 89,
          "v": 9045.95
        },
        {
          "h": 110,
          "v": 9046.02
        },
        {
          "h": 132,
          "v": 9042.46
        },
        {
          "h": 154,
          "v": 9055.08
        },
        {
          "h": 176,
          "v": 9058.12
        }
      ],
      "s3": [
        {
          "h": 1,
          "v": 1589.01
        },
        {
          "h": 23,
          "v": 1580.41
        },
        {
          "h": 45,
          "v": 1584.25
        },
        {
          "h": 67,
          "v": 1584.57
        },
        {
          "h": 89,
          "v": 1588.17
        },
        {
          "h": 110,
          "v": 1581.24
        },
        {
          "h": 132,
          "v": 1590.32
        },
        {
          "h": 154,
          "v": 1582.29
        },
        {
          "h": 176,
          "v": 1596.72
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #176 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 44.7 사이클 소진 시 정비 권고 (90% 구간 35.0~55.1)",
        "status": "scheduled"
      }
    ],
    "status": "warning",
    "health": 36
  },
  "engine-27": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종 · 고장모드 1종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 140,
      "modelNo": "Turbofan (FD001)"
    },
    "predictedRulCycle": {
      "median": 80.8,
      "lower": 56.0,
      "upper": 96.6
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
        "median": 0.8114,
        "lower": 0.727,
        "upper": 0.8843
      },
      {
        "cycle": 80,
        "median": 0.1686,
        "lower": 0.0868,
        "upper": 0.2616
      },
      {
        "cycle": 100,
        "median": 0.0343,
        "lower": 0.0135,
        "upper": 0.0598
      },
      {
        "cycle": 120,
        "median": 0.02,
        "lower": 0.0112,
        "upper": 0.0292
      },
      {
        "cycle": 140,
        "median": 0.02,
        "lower": 0.0112,
        "upper": 0.0292
      }
    ],
    "sensorTrend": {
      "s4": [
        {
          "h": 1,
          "v": 1394.82
        },
        {
          "h": 18,
          "v": 1398.13
        },
        {
          "h": 36,
          "v": 1395.57
        },
        {
          "h": 53,
          "v": 1394.86
        },
        {
          "h": 71,
          "v": 1400.76
        },
        {
          "h": 88,
          "v": 1399.33
        },
        {
          "h": 105,
          "v": 1396.29
        },
        {
          "h": 123,
          "v": 1401.69
        },
        {
          "h": 140,
          "v": 1407.16
        }
      ],
      "s9": [
        {
          "h": 1,
          "v": 9048.04
        },
        {
          "h": 18,
          "v": 9054.52
        },
        {
          "h": 36,
          "v": 9051.27
        },
        {
          "h": 53,
          "v": 9055.51
        },
        {
          "h": 71,
          "v": 9046.51
        },
        {
          "h": 88,
          "v": 9047.1
        },
        {
          "h": 105,
          "v": 9050.77
        },
        {
          "h": 123,
          "v": 9049.78
        },
        {
          "h": 140,
          "v": 9049.45
        }
      ],
      "s3": [
        {
          "h": 1,
          "v": 1585.88
        },
        {
          "h": 18,
          "v": 1582.67
        },
        {
          "h": 36,
          "v": 1580.93
        },
        {
          "h": 53,
          "v": 1581.38
        },
        {
          "h": 71,
          "v": 1580.97
        },
        {
          "h": 88,
          "v": 1585.53
        },
        {
          "h": 105,
          "v": 1583.27
        },
        {
          "h": 123,
          "v": 1588.1
        },
        {
          "h": 140,
          "v": 1586.56
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #140 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 80.8 사이클 소진 시 정비 권고 (90% 구간 56.0~96.6)",
        "status": "scheduled"
      }
    ],
    "status": "good",
    "health": 65
  },
  "engine-78": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD001 (운전조건 1종 · 고장모드 1종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 72,
      "modelNo": "Turbofan (FD001)"
    },
    "predictedRulCycle": {
      "median": 125.0,
      "lower": 106.0,
      "upper": 211.0
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
        "median": 0.9979,
        "lower": 0.9937,
        "upper": 1.0
      },
      {
        "cycle": 100,
        "median": 0.9672,
        "lower": 0.933,
        "upper": 0.9932
      },
      {
        "cycle": 120,
        "median": 0.8661,
        "lower": 0.7992,
        "upper": 0.9267
      },
      {
        "cycle": 140,
        "median": 0.6743,
        "lower": 0.5746,
        "upper": 0.7638
      }
    ],
    "sensorTrend": {
      "s4": [
        {
          "h": 1,
          "v": 1401.11
        },
        {
          "h": 10,
          "v": 1400.78
        },
        {
          "h": 19,
          "v": 1392.97
        },
        {
          "h": 28,
          "v": 1399.02
        },
        {
          "h": 37,
          "v": 1402.25
        },
        {
          "h": 45,
          "v": 1400.87
        },
        {
          "h": 54,
          "v": 1396.4
        },
        {
          "h": 63,
          "v": 1394.0
        },
        {
          "h": 72,
          "v": 1389.33
        }
      ],
      "s9": [
        {
          "h": 1,
          "v": 9060.41
        },
        {
          "h": 10,
          "v": 9056.18
        },
        {
          "h": 19,
          "v": 9064.88
        },
        {
          "h": 28,
          "v": 9059.29
        },
        {
          "h": 37,
          "v": 9061.15
        },
        {
          "h": 45,
          "v": 9060.58
        },
        {
          "h": 54,
          "v": 9062.78
        },
        {
          "h": 63,
          "v": 9063.28
        },
        {
          "h": 72,
          "v": 9062.94
        }
      ],
      "s3": [
        {
          "h": 1,
          "v": 1574.83
        },
        {
          "h": 10,
          "v": 1590.83
        },
        {
          "h": 19,
          "v": 1583.45
        },
        {
          "h": 28,
          "v": 1585.38
        },
        {
          "h": 37,
          "v": 1587.37
        },
        {
          "h": 45,
          "v": 1578.84
        },
        {
          "h": 54,
          "v": 1589.54
        },
        {
          "h": 63,
          "v": 1584.53
        },
        {
          "h": 72,
          "v": 1580.66
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #72 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 125.0 사이클 소진 시 정비 권고 (90% 구간 106.0~211.0)",
        "status": "scheduled"
      }
    ],
    "status": "good",
    "health": 100
  }
}

export const DEFAULT_ENGINE_ID = 'engine-18'

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

/** 위험 등급(RUL≤40) 조기경보 성능 — GRU 모델, 공식 test 100개 엔진 기준 실제 계산값
 *  (TP=27, FN=1, FP=1) */
export const classifierMetrics = {
  "precision": 0.964,
  "recall": 0.964,
  "f1": 0.964,
  "tp": 27,
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
 *  탐지/누락/오탐 건수는 GRU 모델의 실제 test set 결과, 금액·시간 가정은 항공 엔진 정비
 *  맥락의 illustrative 값입니다 (분석 레포 scripts/08_export_dashboard_ts.py 참고). */
export const scenarioCompare = [
  {
    "metric": "평균 가동중단 시간 (시간/1,000대 환산)",
    "asIs": 6720.0,
    "toBe": 1325.0
  },
  {
    "metric": "정비 비용 (억원/1,000대 환산)",
    "asIs": 67.2,
    "toBe": 24.0
  },
  {
    "metric": "가동중단 손실 (억원/1,000대 환산)",
    "asIs": 672.0,
    "toBe": 132.5
  }
]

export const savingsSummary = {
  "perThousandEnginesEok": 582.6,
  "savingRatePct": 78.8,
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
      "episodes": 28,
      "detected": 27,
      "missed": 1,
      "falseAlarms": 1,
      "dangerCases": 28,
      "dangerCasesDetected": 0,
      "dangerCasesMissed": 1,
      "hoursPlanned": 4.0,
      "hoursUnplanned": 24.0,
      "repairPlanned": 800.0
    }
  }
}
