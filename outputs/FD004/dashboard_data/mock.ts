// ⚠️ 자동 생성 파일 — 분석 레포의 scripts/08_export_dashboard_ts.py 가 만든다.
// 직접 고치지 말고 파이썬 파이프라인을 다시 돌려서 새로 받아올 것.
// 원본 데이터: C-MAPSS FD004 (운전조건 6종 · 고장모드 2종, NASA 터보팬 엔진 열화 시뮬레이션), 모델: LSTM Seq2Seq
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
  "generatedAt": "2026-09-30T12:33:35",
  "dataset": "C-MAPSS FD004",
  "datasetDescription": "운전조건 6종 · 고장모드 2종",
  "sensorValues": "운전조건 보정값",
  "testEngines": 248,
  "selectedEngines": [
    58,
    158,
    31,
    155,
    187,
    197,
    77,
    199,
    246
  ],
  "featuredEngine": 187,
  "note": "시간축은 엔진 운행 사이클(cycle). 위험 등급(RED/YELLOW/GREEN)은 예측 RUL 기준(≤40/<80/그 외). 비용·정비시간은 가정값(08_export_dashboard_ts.py COST_ASSUMPTIONS), 탐지 성능(TP/FN/FP)은 LSTM 모델의 실제 test set 결과. RUL 신뢰구간·생존곡선은 검증셋에서 예측이 비슷했던 사례들의 실제 RUL 분포."
}

/** 신호등 기준 (분석 레포 common.py의 DANGER_RUL · RISK_THRESHOLDS) — 화면 문구도 이 값을 씀
 *  위험: 예측 RUL ≤ dangerRul, 주의: dangerRul 초과 ~ warningRul 미만, 정상: warningRul 이상 */
export const riskThresholds = { dangerRul: 40, warningRul: 80 }

/** 메인 차트: 대표 엔진(#187)의 예측 RUL 추이 — h = 창 안의 사이클 순서(0~24) */
export const sensorSeries: { t: string; h: number; v: number }[] = [
  {
    "t": "#97",
    "h": 0,
    "v": 91.29
  },
  {
    "t": "#98",
    "h": 1,
    "v": 89.62
  },
  {
    "t": "#99",
    "h": 2,
    "v": 92.37
  },
  {
    "t": "#100",
    "h": 3,
    "v": 93.01
  },
  {
    "t": "#101",
    "h": 4,
    "v": 83.44
  },
  {
    "t": "#102",
    "h": 5,
    "v": 83.31
  },
  {
    "t": "#103",
    "h": 6,
    "v": 81.34
  },
  {
    "t": "#104",
    "h": 7,
    "v": 79.51
  },
  {
    "t": "#105",
    "h": 8,
    "v": 77.83
  },
  {
    "t": "#106",
    "h": 9,
    "v": 76.18
  },
  {
    "t": "#107",
    "h": 10,
    "v": 75.2
  },
  {
    "t": "#108",
    "h": 11,
    "v": 65.67
  },
  {
    "t": "#109",
    "h": 12,
    "v": 57.22
  },
  {
    "t": "#110",
    "h": 13,
    "v": 58.66
  },
  {
    "t": "#111",
    "h": 14,
    "v": 57.64
  },
  {
    "t": "#112",
    "h": 15,
    "v": 54.89
  },
  {
    "t": "#113",
    "h": 16,
    "v": 53.55
  },
  {
    "t": "#114",
    "h": 17,
    "v": 45.69
  },
  {
    "t": "#115",
    "h": 18,
    "v": 46.46
  },
  {
    "t": "#116",
    "h": 19,
    "v": 42.28
  },
  {
    "t": "#117",
    "h": 20,
    "v": 42.34
  },
  {
    "t": "#118",
    "h": 21,
    "v": 41.15
  },
  {
    "t": "#119",
    "h": 22,
    "v": 39.11
  },
  {
    "t": "#120",
    "h": 23,
    "v": 37.73
  },
  {
    "t": "#121",
    "h": 24,
    "v": 35.24
  }
]

export const anomalyWindow = {
  "start": 22,
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
    "id": "engine-58",
    "name": "Engine #58",
    "health": 1,
    "status": "critical"
  },
  {
    "id": "engine-158",
    "name": "Engine #158",
    "health": 2,
    "status": "critical"
  },
  {
    "id": "engine-31",
    "name": "Engine #31",
    "health": 5,
    "status": "critical"
  },
  {
    "id": "engine-155",
    "name": "Engine #155",
    "health": 7,
    "status": "critical"
  },
  {
    "id": "engine-187",
    "name": "Engine #187",
    "health": 28,
    "status": "critical"
  },
  {
    "id": "engine-197",
    "name": "Engine #197",
    "health": 32,
    "status": "warning"
  },
  {
    "id": "engine-77",
    "name": "Engine #77",
    "health": 32,
    "status": "warning"
  },
  {
    "id": "engine-199",
    "name": "Engine #199",
    "health": 64,
    "status": "good"
  },
  {
    "id": "engine-246",
    "name": "Engine #246",
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

/** 대표 엔진(#187)의 예측 RUL이 주의(80 미만)/위험(40 이하) 임계값을 넘은 시점 — 실제 계산값 */
export const alertLog: AlertRow[] = [
  {
    "h": 22,
    "time": "cycle #119",
    "equipment": "Engine #187",
    "sensor": "예측 RUL",
    "severity": "critical",
    "action": "정비 일정 즉시 수립",
    "type": "EngineRemoval"
  },
  {
    "h": 7,
    "time": "cycle #104",
    "equipment": "Engine #187",
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
  "engine-187": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD004 (운전조건 6종 · 고장모드 2종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 121,
      "modelNo": "Turbofan (FD004)"
    },
    "predictedRulCycle": {
      "median": 35.2,
      "lower": 19.0,
      "upper": 62.0
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
        "median": 0.9302,
        "lower": 0.905,
        "upper": 0.9538
      },
      {
        "cycle": 30,
        "median": 0.6304,
        "lower": 0.5634,
        "upper": 0.6983
      },
      {
        "cycle": 40,
        "median": 0.2971,
        "lower": 0.2267,
        "upper": 0.3705
      },
      {
        "cycle": 60,
        "median": 0.0539,
        "lower": 0.0136,
        "upper": 0.0941
      },
      {
        "cycle": 80,
        "median": 0.0389,
        "lower": 0.0,
        "upper": 0.0771
      },
      {
        "cycle": 100,
        "median": 0.0389,
        "lower": 0.0,
        "upper": 0.0771
      },
      {
        "cycle": 120,
        "median": 0.0389,
        "lower": 0.0,
        "upper": 0.0771
      },
      {
        "cycle": 140,
        "median": 0.0389,
        "lower": 0.0,
        "upper": 0.0771
      }
    ],
    "sensorTrend": {
      "s3": [
        {
          "h": 1,
          "v": 1417.88
        },
        {
          "h": 16,
          "v": 1412.76
        },
        {
          "h": 31,
          "v": 1415.68
        },
        {
          "h": 46,
          "v": 1413.34
        },
        {
          "h": 61,
          "v": 1416.41
        },
        {
          "h": 76,
          "v": 1412.83
        },
        {
          "h": 91,
          "v": 1421.33
        },
        {
          "h": 106,
          "v": 1414.98
        },
        {
          "h": 121,
          "v": 1423.62
        }
      ],
      "s17": [
        {
          "h": 1,
          "v": 346.13
        },
        {
          "h": 16,
          "v": 349.23
        },
        {
          "h": 31,
          "v": 346.6
        },
        {
          "h": 46,
          "v": 348.6
        },
        {
          "h": 61,
          "v": 347.23
        },
        {
          "h": 76,
          "v": 348.13
        },
        {
          "h": 91,
          "v": 347.98
        },
        {
          "h": 106,
          "v": 348.23
        },
        {
          "h": 121,
          "v": 348.23
        }
      ],
      "s8": [
        {
          "h": 1,
          "v": 2228.71
        },
        {
          "h": 16,
          "v": 2228.87
        },
        {
          "h": 31,
          "v": 2228.73
        },
        {
          "h": 46,
          "v": 2228.71
        },
        {
          "h": 61,
          "v": 2228.85
        },
        {
          "h": 76,
          "v": 2228.7
        },
        {
          "h": 91,
          "v": 2228.79
        },
        {
          "h": 106,
          "v": 2228.6
        },
        {
          "h": 121,
          "v": 2228.5
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #121 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 35.2 사이클 소진 시 정비 권고 (90% 구간 19.0~62.0)",
        "status": "scheduled"
      }
    ],
    "status": "critical",
    "health": 28
  },
  "engine-58": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD004 (운전조건 6종 · 고장모드 2종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 185,
      "modelNo": "Turbofan (FD004)"
    },
    "predictedRulCycle": {
      "median": 0.9,
      "lower": 0.0,
      "upper": 13.0
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 0.8899,
        "lower": 0.877,
        "upper": 0.9002
      },
      {
        "cycle": 10,
        "median": 0.1124,
        "lower": 0.0729,
        "upper": 0.1545
      },
      {
        "cycle": 20,
        "median": 0.0164,
        "lower": 0.0,
        "upper": 0.0359
      },
      {
        "cycle": 30,
        "median": 0.0164,
        "lower": 0.0,
        "upper": 0.0359
      },
      {
        "cycle": 40,
        "median": 0.0164,
        "lower": 0.0,
        "upper": 0.0359
      },
      {
        "cycle": 60,
        "median": 0.0164,
        "lower": 0.0,
        "upper": 0.0359
      },
      {
        "cycle": 80,
        "median": 0.0164,
        "lower": 0.0,
        "upper": 0.0359
      },
      {
        "cycle": 100,
        "median": 0.0164,
        "lower": 0.0,
        "upper": 0.0359
      },
      {
        "cycle": 120,
        "median": 0.0164,
        "lower": 0.0,
        "upper": 0.0359
      },
      {
        "cycle": 140,
        "median": 0.0164,
        "lower": 0.0,
        "upper": 0.0359
      }
    ],
    "sensorTrend": {
      "s3": [
        {
          "h": 1,
          "v": 1410.61
        },
        {
          "h": 24,
          "v": 1414.85
        },
        {
          "h": 47,
          "v": 1414.77
        },
        {
          "h": 70,
          "v": 1420.43
        },
        {
          "h": 93,
          "v": 1410.78
        },
        {
          "h": 116,
          "v": 1413.34
        },
        {
          "h": 139,
          "v": 1424.09
        },
        {
          "h": 162,
          "v": 1423.49
        },
        {
          "h": 185,
          "v": 1429.0
        }
      ],
      "s17": [
        {
          "h": 1,
          "v": 346.97
        },
        {
          "h": 24,
          "v": 346.07
        },
        {
          "h": 47,
          "v": 346.6
        },
        {
          "h": 70,
          "v": 348.23
        },
        {
          "h": 93,
          "v": 347.6
        },
        {
          "h": 116,
          "v": 349.23
        },
        {
          "h": 139,
          "v": 348.23
        },
        {
          "h": 162,
          "v": 347.97
        },
        {
          "h": 185,
          "v": 352.13
        }
      ],
      "s8": [
        {
          "h": 1,
          "v": 2228.68
        },
        {
          "h": 24,
          "v": 2228.68
        },
        {
          "h": 47,
          "v": 2228.78
        },
        {
          "h": 70,
          "v": 2228.63
        },
        {
          "h": 93,
          "v": 2228.8
        },
        {
          "h": 116,
          "v": 2228.64
        },
        {
          "h": 139,
          "v": 2228.53
        },
        {
          "h": 162,
          "v": 2228.79
        },
        {
          "h": 185,
          "v": 2229.0
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #185 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 0.9 사이클 소진 시 정비 권고 (90% 구간 0.0~13.0)",
        "status": "scheduled"
      }
    ],
    "status": "critical",
    "health": 1
  },
  "engine-158": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD004 (운전조건 6종 · 고장모드 2종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 171,
      "modelNo": "Turbofan (FD004)"
    },
    "predictedRulCycle": {
      "median": 1.9,
      "lower": 0.0,
      "upper": 15.0
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 0.9002,
        "lower": 0.8891,
        "upper": 0.9098
      },
      {
        "cycle": 10,
        "median": 0.1527,
        "lower": 0.1096,
        "upper": 0.1961
      },
      {
        "cycle": 20,
        "median": 0.0163,
        "lower": 0.0,
        "upper": 0.0364
      },
      {
        "cycle": 30,
        "median": 0.0163,
        "lower": 0.0,
        "upper": 0.0364
      },
      {
        "cycle": 40,
        "median": 0.0163,
        "lower": 0.0,
        "upper": 0.0364
      },
      {
        "cycle": 60,
        "median": 0.0163,
        "lower": 0.0,
        "upper": 0.0364
      },
      {
        "cycle": 80,
        "median": 0.0163,
        "lower": 0.0,
        "upper": 0.0364
      },
      {
        "cycle": 100,
        "median": 0.0163,
        "lower": 0.0,
        "upper": 0.0364
      },
      {
        "cycle": 120,
        "median": 0.0163,
        "lower": 0.0,
        "upper": 0.0364
      },
      {
        "cycle": 140,
        "median": 0.0163,
        "lower": 0.0,
        "upper": 0.0364
      }
    ],
    "sensorTrend": {
      "s3": [
        {
          "h": 1,
          "v": 1417.37
        },
        {
          "h": 22,
          "v": 1411.98
        },
        {
          "h": 43,
          "v": 1416.45
        },
        {
          "h": 65,
          "v": 1411.23
        },
        {
          "h": 86,
          "v": 1418.56
        },
        {
          "h": 107,
          "v": 1416.21
        },
        {
          "h": 129,
          "v": 1418.91
        },
        {
          "h": 150,
          "v": 1422.36
        },
        {
          "h": 171,
          "v": 1431.0
        }
      ],
      "s17": [
        {
          "h": 1,
          "v": 348.98
        },
        {
          "h": 22,
          "v": 348.23
        },
        {
          "h": 43,
          "v": 347.6
        },
        {
          "h": 65,
          "v": 347.98
        },
        {
          "h": 86,
          "v": 348.97
        },
        {
          "h": 107,
          "v": 349.6
        },
        {
          "h": 129,
          "v": 348.97
        },
        {
          "h": 150,
          "v": 347.98
        },
        {
          "h": 171,
          "v": 351.07
        }
      ],
      "s8": [
        {
          "h": 1,
          "v": 2228.61
        },
        {
          "h": 22,
          "v": 2228.5
        },
        {
          "h": 43,
          "v": 2228.72
        },
        {
          "h": 65,
          "v": 2228.62
        },
        {
          "h": 86,
          "v": 2228.73
        },
        {
          "h": 107,
          "v": 2228.77
        },
        {
          "h": 129,
          "v": 2228.74
        },
        {
          "h": 150,
          "v": 2228.57
        },
        {
          "h": 171,
          "v": 2228.62
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #171 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 1.9 사이클 소진 시 정비 권고 (90% 구간 0.0~15.0)",
        "status": "scheduled"
      }
    ],
    "status": "critical",
    "health": 2
  },
  "engine-31": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD004 (운전조건 6종 · 고장모드 2종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 134,
      "modelNo": "Turbofan (FD004)"
    },
    "predictedRulCycle": {
      "median": 6.1,
      "lower": 0.0,
      "upper": 19.0
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 0.9311,
        "lower": 0.9258,
        "upper": 0.9362
      },
      {
        "cycle": 10,
        "median": 0.2757,
        "lower": 0.2292,
        "upper": 0.3234
      },
      {
        "cycle": 20,
        "median": 0.0408,
        "lower": 0.0123,
        "upper": 0.0735
      },
      {
        "cycle": 30,
        "median": 0.0225,
        "lower": 0.0,
        "upper": 0.0499
      },
      {
        "cycle": 40,
        "median": 0.0225,
        "lower": 0.0,
        "upper": 0.0499
      },
      {
        "cycle": 60,
        "median": 0.0225,
        "lower": 0.0,
        "upper": 0.0499
      },
      {
        "cycle": 80,
        "median": 0.0225,
        "lower": 0.0,
        "upper": 0.0499
      },
      {
        "cycle": 100,
        "median": 0.0225,
        "lower": 0.0,
        "upper": 0.0499
      },
      {
        "cycle": 120,
        "median": 0.0225,
        "lower": 0.0,
        "upper": 0.0499
      },
      {
        "cycle": 140,
        "median": 0.0225,
        "lower": 0.0,
        "upper": 0.0499
      }
    ],
    "sensorTrend": {
      "s3": [
        {
          "h": 1,
          "v": 1414.22
        },
        {
          "h": 18,
          "v": 1410.47
        },
        {
          "h": 34,
          "v": 1415.3
        },
        {
          "h": 51,
          "v": 1415.51
        },
        {
          "h": 67,
          "v": 1415.35
        },
        {
          "h": 84,
          "v": 1416.0
        },
        {
          "h": 101,
          "v": 1419.79
        },
        {
          "h": 117,
          "v": 1424.17
        },
        {
          "h": 134,
          "v": 1432.84
        }
      ],
      "s17": [
        {
          "h": 1,
          "v": 347.98
        },
        {
          "h": 18,
          "v": 345.97
        },
        {
          "h": 34,
          "v": 348.13
        },
        {
          "h": 51,
          "v": 349.07
        },
        {
          "h": 67,
          "v": 346.98
        },
        {
          "h": 84,
          "v": 348.23
        },
        {
          "h": 101,
          "v": 348.98
        },
        {
          "h": 117,
          "v": 350.23
        },
        {
          "h": 134,
          "v": 350.6
        }
      ],
      "s8": [
        {
          "h": 1,
          "v": 2228.77
        },
        {
          "h": 18,
          "v": 2228.72
        },
        {
          "h": 34,
          "v": 2228.73
        },
        {
          "h": 51,
          "v": 2228.76
        },
        {
          "h": 67,
          "v": 2228.74
        },
        {
          "h": 84,
          "v": 2228.75
        },
        {
          "h": 101,
          "v": 2228.8
        },
        {
          "h": 117,
          "v": 2228.67
        },
        {
          "h": 134,
          "v": 2228.98
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #134 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 6.1 사이클 소진 시 정비 권고 (90% 구간 0.0~19.0)",
        "status": "scheduled"
      }
    ],
    "status": "critical",
    "health": 5
  },
  "engine-155": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD004 (운전조건 6종 · 고장모드 2종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 161,
      "modelNo": "Turbofan (FD004)"
    },
    "predictedRulCycle": {
      "median": 8.3,
      "lower": 0.0,
      "upper": 21.0
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 0.9395,
        "lower": 0.9351,
        "upper": 0.9437
      },
      {
        "cycle": 10,
        "median": 0.3444,
        "lower": 0.2989,
        "upper": 0.3888
      },
      {
        "cycle": 20,
        "median": 0.0568,
        "lower": 0.027,
        "upper": 0.089
      },
      {
        "cycle": 30,
        "median": 0.0235,
        "lower": 0.0,
        "upper": 0.0524
      },
      {
        "cycle": 40,
        "median": 0.0235,
        "lower": 0.0,
        "upper": 0.0524
      },
      {
        "cycle": 60,
        "median": 0.0235,
        "lower": 0.0,
        "upper": 0.0524
      },
      {
        "cycle": 80,
        "median": 0.0235,
        "lower": 0.0,
        "upper": 0.0524
      },
      {
        "cycle": 100,
        "median": 0.0235,
        "lower": 0.0,
        "upper": 0.0524
      },
      {
        "cycle": 120,
        "median": 0.0235,
        "lower": 0.0,
        "upper": 0.0524
      },
      {
        "cycle": 140,
        "median": 0.0235,
        "lower": 0.0,
        "upper": 0.0524
      }
    ],
    "sensorTrend": {
      "s3": [
        {
          "h": 1,
          "v": 1420.43
        },
        {
          "h": 21,
          "v": 1415.94
        },
        {
          "h": 41,
          "v": 1419.04
        },
        {
          "h": 61,
          "v": 1418.77
        },
        {
          "h": 81,
          "v": 1425.12
        },
        {
          "h": 101,
          "v": 1415.54
        },
        {
          "h": 121,
          "v": 1415.8
        },
        {
          "h": 141,
          "v": 1424.11
        },
        {
          "h": 161,
          "v": 1427.05
        }
      ],
      "s17": [
        {
          "h": 1,
          "v": 347.07
        },
        {
          "h": 21,
          "v": 346.23
        },
        {
          "h": 41,
          "v": 347.13
        },
        {
          "h": 61,
          "v": 347.6
        },
        {
          "h": 81,
          "v": 347.6
        },
        {
          "h": 101,
          "v": 346.97
        },
        {
          "h": 121,
          "v": 347.98
        },
        {
          "h": 141,
          "v": 350.6
        },
        {
          "h": 161,
          "v": 348.6
        }
      ],
      "s8": [
        {
          "h": 1,
          "v": 2228.6
        },
        {
          "h": 21,
          "v": 2228.67
        },
        {
          "h": 41,
          "v": 2228.71
        },
        {
          "h": 61,
          "v": 2228.69
        },
        {
          "h": 81,
          "v": 2228.72
        },
        {
          "h": 101,
          "v": 2228.76
        },
        {
          "h": 121,
          "v": 2228.78
        },
        {
          "h": 141,
          "v": 2228.85
        },
        {
          "h": 161,
          "v": 2228.93
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #161 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 8.3 사이클 소진 시 정비 권고 (90% 구간 0.0~21.0)",
        "status": "scheduled"
      }
    ],
    "status": "critical",
    "health": 7
  },
  "engine-197": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD004 (운전조건 6종 · 고장모드 2종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 196,
      "modelNo": "Turbofan (FD004)"
    },
    "predictedRulCycle": {
      "median": 40.4,
      "lower": 24.0,
      "upper": 72.5
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
        "median": 0.9852,
        "lower": 0.9746,
        "upper": 0.9937
      },
      {
        "cycle": 30,
        "median": 0.7969,
        "lower": 0.7453,
        "upper": 0.8434
      },
      {
        "cycle": 40,
        "median": 0.4688,
        "lower": 0.3926,
        "upper": 0.5451
      },
      {
        "cycle": 60,
        "median": 0.099,
        "lower": 0.0501,
        "upper": 0.1496
      },
      {
        "cycle": 80,
        "median": 0.0434,
        "lower": 0.0035,
        "upper": 0.0829
      },
      {
        "cycle": 100,
        "median": 0.0399,
        "lower": 0.0,
        "upper": 0.0794
      },
      {
        "cycle": 120,
        "median": 0.0399,
        "lower": 0.0,
        "upper": 0.0794
      },
      {
        "cycle": 140,
        "median": 0.0399,
        "lower": 0.0,
        "upper": 0.0794
      }
    ],
    "sensorTrend": {
      "s3": [
        {
          "h": 1,
          "v": 1413.11
        },
        {
          "h": 25,
          "v": 1410.79
        },
        {
          "h": 50,
          "v": 1416.83
        },
        {
          "h": 74,
          "v": 1413.62
        },
        {
          "h": 99,
          "v": 1414.05
        },
        {
          "h": 123,
          "v": 1421.49
        },
        {
          "h": 147,
          "v": 1425.78
        },
        {
          "h": 172,
          "v": 1425.58
        },
        {
          "h": 196,
          "v": 1419.84
        }
      ],
      "s17": [
        {
          "h": 1,
          "v": 347.98
        },
        {
          "h": 25,
          "v": 346.97
        },
        {
          "h": 50,
          "v": 346.98
        },
        {
          "h": 74,
          "v": 346.6
        },
        {
          "h": 99,
          "v": 348.23
        },
        {
          "h": 123,
          "v": 347.07
        },
        {
          "h": 147,
          "v": 348.13
        },
        {
          "h": 172,
          "v": 347.98
        },
        {
          "h": 196,
          "v": 349.23
        }
      ],
      "s8": [
        {
          "h": 1,
          "v": 2228.85
        },
        {
          "h": 25,
          "v": 2228.65
        },
        {
          "h": 50,
          "v": 2228.84
        },
        {
          "h": 74,
          "v": 2228.69
        },
        {
          "h": 99,
          "v": 2228.87
        },
        {
          "h": 123,
          "v": 2228.94
        },
        {
          "h": 147,
          "v": 2228.73
        },
        {
          "h": 172,
          "v": 2229.14
        },
        {
          "h": 196,
          "v": 2229.42
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #196 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 40.4 사이클 소진 시 정비 권고 (90% 구간 24.0~72.5)",
        "status": "scheduled"
      }
    ],
    "status": "warning",
    "health": 32
  },
  "engine-77": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD004 (운전조건 6종 · 고장모드 2종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 108,
      "modelNo": "Turbofan (FD004)"
    },
    "predictedRulCycle": {
      "median": 40.5,
      "lower": 24.0,
      "upper": 72.6
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
        "median": 0.9869,
        "lower": 0.9775,
        "upper": 0.9947
      },
      {
        "cycle": 30,
        "median": 0.7998,
        "lower": 0.7487,
        "upper": 0.847
      },
      {
        "cycle": 40,
        "median": 0.4717,
        "lower": 0.3957,
        "upper": 0.5481
      },
      {
        "cycle": 60,
        "median": 0.0992,
        "lower": 0.05,
        "upper": 0.1501
      },
      {
        "cycle": 80,
        "median": 0.0435,
        "lower": 0.0035,
        "upper": 0.0831
      },
      {
        "cycle": 100,
        "median": 0.04,
        "lower": 0.0,
        "upper": 0.0795
      },
      {
        "cycle": 120,
        "median": 0.04,
        "lower": 0.0,
        "upper": 0.0795
      },
      {
        "cycle": 140,
        "median": 0.04,
        "lower": 0.0,
        "upper": 0.0795
      }
    ],
    "sensorTrend": {
      "s3": [
        {
          "h": 1,
          "v": 1410.41
        },
        {
          "h": 14,
          "v": 1414.56
        },
        {
          "h": 28,
          "v": 1409.21
        },
        {
          "h": 41,
          "v": 1415.39
        },
        {
          "h": 55,
          "v": 1423.72
        },
        {
          "h": 68,
          "v": 1415.29
        },
        {
          "h": 81,
          "v": 1413.77
        },
        {
          "h": 95,
          "v": 1423.0
        },
        {
          "h": 108,
          "v": 1420.25
        }
      ],
      "s17": [
        {
          "h": 1,
          "v": 347.13
        },
        {
          "h": 14,
          "v": 347.13
        },
        {
          "h": 28,
          "v": 347.23
        },
        {
          "h": 41,
          "v": 347.23
        },
        {
          "h": 55,
          "v": 346.98
        },
        {
          "h": 68,
          "v": 346.23
        },
        {
          "h": 81,
          "v": 347.98
        },
        {
          "h": 95,
          "v": 347.23
        },
        {
          "h": 108,
          "v": 349.23
        }
      ],
      "s8": [
        {
          "h": 1,
          "v": 2228.74
        },
        {
          "h": 14,
          "v": 2228.74
        },
        {
          "h": 28,
          "v": 2228.64
        },
        {
          "h": 41,
          "v": 2228.7
        },
        {
          "h": 55,
          "v": 2228.73
        },
        {
          "h": 68,
          "v": 2228.61
        },
        {
          "h": 81,
          "v": 2228.69
        },
        {
          "h": 95,
          "v": 2228.48
        },
        {
          "h": 108,
          "v": 2228.38
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #108 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 40.5 사이클 소진 시 정비 권고 (90% 구간 24.0~72.6)",
        "status": "scheduled"
      }
    ],
    "status": "warning",
    "health": 32
  },
  "engine-199": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD004 (운전조건 6종 · 고장모드 2종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 177,
      "modelNo": "Turbofan (FD004)"
    },
    "predictedRulCycle": {
      "median": 80.0,
      "lower": 52.7,
      "upper": 132.3
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
        "median": 0.9965,
        "lower": 0.9892,
        "upper": 1.0
      },
      {
        "cycle": 60,
        "median": 0.8244,
        "lower": 0.7559,
        "upper": 0.8869
      },
      {
        "cycle": 80,
        "median": 0.4461,
        "lower": 0.3455,
        "upper": 0.5505
      },
      {
        "cycle": 100,
        "median": 0.1674,
        "lower": 0.0918,
        "upper": 0.2461
      },
      {
        "cycle": 120,
        "median": 0.0656,
        "lower": 0.019,
        "upper": 0.1196
      },
      {
        "cycle": 140,
        "median": 0.0398,
        "lower": 0.0044,
        "upper": 0.0766
      }
    ],
    "sensorTrend": {
      "s3": [
        {
          "h": 1,
          "v": 1414.51
        },
        {
          "h": 23,
          "v": 1421.88
        },
        {
          "h": 45,
          "v": 1416.22
        },
        {
          "h": 67,
          "v": 1428.96
        },
        {
          "h": 89,
          "v": 1415.34
        },
        {
          "h": 111,
          "v": 1429.95
        },
        {
          "h": 133,
          "v": 1419.64
        },
        {
          "h": 155,
          "v": 1418.54
        },
        {
          "h": 177,
          "v": 1420.7
        }
      ],
      "s17": [
        {
          "h": 1,
          "v": 348.98
        },
        {
          "h": 23,
          "v": 347.23
        },
        {
          "h": 45,
          "v": 346.98
        },
        {
          "h": 67,
          "v": 347.23
        },
        {
          "h": 89,
          "v": 347.98
        },
        {
          "h": 111,
          "v": 346.6
        },
        {
          "h": 133,
          "v": 350.23
        },
        {
          "h": 155,
          "v": 349.23
        },
        {
          "h": 177,
          "v": 350.13
        }
      ],
      "s8": [
        {
          "h": 1,
          "v": 2228.53
        },
        {
          "h": 23,
          "v": 2228.58
        },
        {
          "h": 45,
          "v": 2228.62
        },
        {
          "h": 67,
          "v": 2228.5
        },
        {
          "h": 89,
          "v": 2228.59
        },
        {
          "h": 111,
          "v": 2228.82
        },
        {
          "h": 133,
          "v": 2228.5
        },
        {
          "h": 155,
          "v": 2228.51
        },
        {
          "h": 177,
          "v": 2228.88
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #177 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 80.0 사이클 소진 시 정비 권고 (90% 구간 52.7~132.3)",
        "status": "scheduled"
      }
    ],
    "status": "good",
    "health": 64
  },
  "engine-246": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD004 (운전조건 6종 · 고장모드 2종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 29,
      "modelNo": "Turbofan (FD004)"
    },
    "predictedRulCycle": {
      "median": 125.0,
      "lower": 106.0,
      "upper": 361.0
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
        "median": 0.9967,
        "lower": 0.9934,
        "upper": 0.9993
      },
      {
        "cycle": 100,
        "median": 0.9655,
        "lower": 0.9486,
        "upper": 0.9803
      },
      {
        "cycle": 120,
        "median": 0.8982,
        "lower": 0.8624,
        "upper": 0.9297
      },
      {
        "cycle": 140,
        "median": 0.794,
        "lower": 0.7366,
        "upper": 0.844
      }
    ],
    "sensorTrend": {
      "s3": [
        {
          "h": 1,
          "v": 1415.86
        },
        {
          "h": 5,
          "v": 1409.03
        },
        {
          "h": 8,
          "v": 1413.52
        },
        {
          "h": 11,
          "v": 1413.67
        },
        {
          "h": 15,
          "v": 1416.96
        },
        {
          "h": 19,
          "v": 1407.96
        },
        {
          "h": 22,
          "v": 1417.21
        },
        {
          "h": 25,
          "v": 1416.97
        },
        {
          "h": 29,
          "v": 1417.51
        }
      ],
      "s17": [
        {
          "h": 1,
          "v": 347.23
        },
        {
          "h": 5,
          "v": 347.23
        },
        {
          "h": 8,
          "v": 347.07
        },
        {
          "h": 11,
          "v": 346.23
        },
        {
          "h": 15,
          "v": 346.98
        },
        {
          "h": 19,
          "v": 347.23
        },
        {
          "h": 22,
          "v": 346.07
        },
        {
          "h": 25,
          "v": 344.13
        },
        {
          "h": 29,
          "v": 346.97
        }
      ],
      "s8": [
        {
          "h": 1,
          "v": 2228.68
        },
        {
          "h": 5,
          "v": 2228.8
        },
        {
          "h": 8,
          "v": 2228.62
        },
        {
          "h": 11,
          "v": 2228.69
        },
        {
          "h": 15,
          "v": 2228.71
        },
        {
          "h": 19,
          "v": 2228.74
        },
        {
          "h": 22,
          "v": 2228.7
        },
        {
          "h": 25,
          "v": 2228.71
        },
        {
          "h": 29,
          "v": 2228.68
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #29 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 125.0 사이클 소진 시 정비 권고 (90% 구간 106.0~361.0)",
        "status": "scheduled"
      }
    ],
    "status": "good",
    "health": 100
  }
}

export const DEFAULT_ENGINE_ID = 'engine-187'

/** 센서 추이 3종 (06번 RF 중요도 상위 센서) — 값은 운전조건 보정값 */
export const trendMeta: Record<string, { unit: string; label: string; domain: [number, number] }> = {
  "s3": {
    "unit": "°R",
    "label": "T30 · HPC outlet 온도 (s3)",
    "domain": [
      1391.1,
      1450.7
    ]
  },
  "s17": {
    "unit": "",
    "label": "htBleed · 블리드 엔탈피 (s17)",
    "domain": [
      340.9,
      355.4
    ]
  },
  "s8": {
    "unit": "rpm",
    "label": "Nf · 물리적 팬 속도 (s8)",
    "domain": [
      2227.2,
      2231.2
    ]
  }
}

/** 위험 등급(RUL≤40) 조기경보 성능 — LSTM 모델, 공식 test 248개 엔진 기준 실제 계산값
 *  (TP=60, FN=9, FP=6) */
export const classifierMetrics = {
  "precision": 0.909,
  "recall": 0.87,
  "f1": 0.889,
  "tp": 60,
  "fn": 9,
  "fp": 6
}

/** RF 피처 중요도 상위 5개 센서(그룹 합산) */
export const featureImportance = [
  {
    "label": "T30 · HPC outlet 온도 (s3)",
    "value": 0.504
  },
  {
    "label": "htBleed · 블리드 엔탈피 (s17)",
    "value": 0.142
  },
  {
    "label": "Nf · 물리적 팬 속도 (s8)",
    "value": 0.112
  },
  {
    "label": "Ps30 · HPC outlet 정압 (s11)",
    "value": 0.039
  },
  {
    "label": "T50 · LPT outlet 온도 (s4)",
    "value": 0.03
  }
]

/** AS-IS(사후 정비: 고장까지 운용) vs TO-BE(예지보전: RUL 기반 사전 정비) 비용 시뮬레이션.
 *  탐지/누락/오탐 건수는 LSTM 모델의 실제 test set 결과, 금액·시간 가정은 항공 엔진 정비
 *  맥락의 illustrative 값입니다 (분석 레포 scripts/08_export_dashboard_ts.py 참고). */
export const scenarioCompare = [
  {
    "metric": "평균 가동중단 시간 (시간/1,000대 환산)",
    "asIs": 6677.4,
    "toBe": 1850.8
  },
  {
    "metric": "정비 비용 (억원/1,000대 환산)",
    "asIs": 66.8,
    "toBe": 28.2
  },
  {
    "metric": "가동중단 손실 (억원/1,000대 환산)",
    "asIs": 667.7,
    "toBe": 185.1
  }
]

export const savingsSummary = {
  "perThousandEnginesEok": 521.2,
  "savingRatePct": 71.0,
  "dangerCaseReductionPct": 87,
  "note": "테스트 엔진 248대 결과를 1,000대 규모로 환산한 값입니다. 정비 비용·가동중단 단가는 가정값이고, 탐지/누락/오탐 건수는 실제 모델 성능입니다."
}

/** 시나리오 화면 슬라이더가 다시 계산할 때 쓰는 원본 가정치 (계산식은 분석 레포 compute_scenario()와 동일) */
export const costModel = {
  "nEngines": 248,
  "general": {
    "downtimeCostPerHour": 1000.0,
    "repairUnplannedRatio": 3.0,
    "dangerCaseLoss": 0.0,
    "falseAlarmHours": 0.5,
    "falseAlarmLabor": 50.0
  },
  "types": {
    "EngineRemoval": {
      "episodes": 69,
      "detected": 60,
      "missed": 9,
      "falseAlarms": 6,
      "dangerCases": 69,
      "dangerCasesDetected": 0,
      "dangerCasesMissed": 9,
      "hoursPlanned": 4.0,
      "hoursUnplanned": 24.0,
      "repairPlanned": 800.0
    }
  }
}
