// ⚠️ 자동 생성 파일 — 분석 레포의 scripts/08_export_dashboard_ts.py 가 만든다.
// 직접 고치지 말고 파이썬 파이프라인을 다시 돌려서 새로 받아올 것.
// 원본 데이터: C-MAPSS FD004 (운전조건 6종 · 고장모드 2종, NASA 터보팬 엔진 열화 시뮬레이션), 모델: GRU Seq2Seq 회귀
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
  "generatedAt": "2026-09-30T14:23:27",
  "dataset": "C-MAPSS FD004",
  "datasetDescription": "운전조건 6종 · 고장모드 2종",
  "sensorValues": "운전조건 보정값",
  "model": "GRU",
  "testEngines": 248,
  "selectedEngines": [
    31,
    103,
    158,
    58,
    198,
    70,
    185,
    221,
    10
  ],
  "featuredEngine": 198,
  "note": "시간축은 엔진 운행 사이클(cycle). 위험 등급(RED/YELLOW/GREEN)은 예측 RUL 기준(≤40/<80/그 외). 비용·정비시간은 가정값(08_export_dashboard_ts.py COST_ASSUMPTIONS), 탐지 성능(TP/FN/FP)은 GRU 모델의 실제 test set 결과. RUL 신뢰구간·생존곡선은 검증셋에서 예측이 비슷했던 사례들의 실제 RUL 분포."
}

/** 신호등 기준 (분석 레포 common.py의 DANGER_RUL · RISK_THRESHOLDS) — 화면 문구도 이 값을 씀
 *  위험: 예측 RUL ≤ dangerRul, 주의: dangerRul 초과 ~ warningRul 미만, 정상: warningRul 이상 */
export const riskThresholds = { dangerRul: 40, warningRul: 80 }

/** 메인 차트: 대표 엔진(#198)의 예측 RUL 추이 — h = 창 안의 사이클 순서(0~24) */
export const sensorSeries: { t: string; h: number; v: number }[] = [
  {
    "t": "#91",
    "h": 0,
    "v": 57.97
  },
  {
    "t": "#92",
    "h": 1,
    "v": 58.55
  },
  {
    "t": "#93",
    "h": 2,
    "v": 53.5
  },
  {
    "t": "#94",
    "h": 3,
    "v": 50.31
  },
  {
    "t": "#95",
    "h": 4,
    "v": 46.92
  },
  {
    "t": "#96",
    "h": 5,
    "v": 45.75
  },
  {
    "t": "#97",
    "h": 6,
    "v": 43.18
  },
  {
    "t": "#98",
    "h": 7,
    "v": 40.62
  },
  {
    "t": "#99",
    "h": 8,
    "v": 38.95
  },
  {
    "t": "#100",
    "h": 9,
    "v": 37.27
  },
  {
    "t": "#101",
    "h": 10,
    "v": 36.65
  },
  {
    "t": "#102",
    "h": 11,
    "v": 35.62
  },
  {
    "t": "#103",
    "h": 12,
    "v": 35.2
  },
  {
    "t": "#104",
    "h": 13,
    "v": 32.89
  },
  {
    "t": "#105",
    "h": 14,
    "v": 34.58
  },
  {
    "t": "#106",
    "h": 15,
    "v": 35.31
  },
  {
    "t": "#107",
    "h": 16,
    "v": 32.54
  },
  {
    "t": "#108",
    "h": 17,
    "v": 31.92
  },
  {
    "t": "#109",
    "h": 18,
    "v": 28.51
  },
  {
    "t": "#110",
    "h": 19,
    "v": 27.11
  },
  {
    "t": "#111",
    "h": 20,
    "v": 25.2
  },
  {
    "t": "#112",
    "h": 21,
    "v": 22.84
  },
  {
    "t": "#113",
    "h": 22,
    "v": 20.58
  },
  {
    "t": "#114",
    "h": 23,
    "v": 19.67
  },
  {
    "t": "#115",
    "h": 24,
    "v": 17.41
  }
]

export const anomalyWindow = {
  "start": 8,
  "end": 24
}

const START_CYCLE: number = 91

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
    "id": "engine-31",
    "name": "Engine #31",
    "health": 7,
    "status": "critical"
  },
  {
    "id": "engine-103",
    "name": "Engine #103",
    "health": 7,
    "status": "critical"
  },
  {
    "id": "engine-158",
    "name": "Engine #158",
    "health": 8,
    "status": "critical"
  },
  {
    "id": "engine-58",
    "name": "Engine #58",
    "health": 9,
    "status": "critical"
  },
  {
    "id": "engine-198",
    "name": "Engine #198",
    "health": 14,
    "status": "critical"
  },
  {
    "id": "engine-70",
    "name": "Engine #70",
    "health": 32,
    "status": "warning"
  },
  {
    "id": "engine-185",
    "name": "Engine #185",
    "health": 33,
    "status": "warning"
  },
  {
    "id": "engine-221",
    "name": "Engine #221",
    "health": 64,
    "status": "good"
  },
  {
    "id": "engine-10",
    "name": "Engine #10",
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

/** 대표 엔진(#198)의 예측 RUL이 주의(80 미만)/위험(40 이하) 임계값을 넘은 시점 — 실제 계산값 */
export const alertLog: AlertRow[] = [
  {
    "h": 8,
    "time": "cycle #99",
    "equipment": "Engine #198",
    "sensor": "예측 RUL",
    "severity": "critical",
    "action": "정비 일정 즉시 수립",
    "type": "EngineRemoval"
  },
  {
    "h": 0,
    "time": "cycle #91",
    "equipment": "Engine #198",
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
  "engine-198": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD004 (운전조건 6종 · 고장모드 2종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 115,
      "modelNo": "Turbofan (FD004)"
    },
    "predictedRulCycle": {
      "median": 17.4,
      "lower": 2.0,
      "upper": 26.0
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 0.9861,
        "lower": 0.9804,
        "upper": 0.9912
      },
      {
        "cycle": 10,
        "median": 0.5869,
        "lower": 0.5533,
        "upper": 0.6183
      },
      {
        "cycle": 20,
        "median": 0.147,
        "lower": 0.1019,
        "upper": 0.1929
      },
      {
        "cycle": 30,
        "median": 0.0348,
        "lower": 0.002,
        "upper": 0.0678
      },
      {
        "cycle": 40,
        "median": 0.0328,
        "lower": 0.0,
        "upper": 0.0655
      },
      {
        "cycle": 60,
        "median": 0.0328,
        "lower": 0.0,
        "upper": 0.0655
      },
      {
        "cycle": 80,
        "median": 0.0328,
        "lower": 0.0,
        "upper": 0.0655
      },
      {
        "cycle": 100,
        "median": 0.0328,
        "lower": 0.0,
        "upper": 0.0655
      },
      {
        "cycle": 120,
        "median": 0.0328,
        "lower": 0.0,
        "upper": 0.0655
      },
      {
        "cycle": 140,
        "median": 0.0328,
        "lower": 0.0,
        "upper": 0.0655
      }
    ],
    "sensorTrend": {
      "s3": [
        {
          "h": 1,
          "v": 1420.31
        },
        {
          "h": 15,
          "v": 1418.15
        },
        {
          "h": 29,
          "v": 1421.03
        },
        {
          "h": 44,
          "v": 1425.57
        },
        {
          "h": 58,
          "v": 1422.62
        },
        {
          "h": 72,
          "v": 1422.97
        },
        {
          "h": 87,
          "v": 1420.87
        },
        {
          "h": 101,
          "v": 1419.15
        },
        {
          "h": 115,
          "v": 1424.73
        }
      ],
      "s17": [
        {
          "h": 1,
          "v": 349.13
        },
        {
          "h": 15,
          "v": 346.97
        },
        {
          "h": 29,
          "v": 349.07
        },
        {
          "h": 44,
          "v": 347.97
        },
        {
          "h": 58,
          "v": 347.6
        },
        {
          "h": 72,
          "v": 349.07
        },
        {
          "h": 87,
          "v": 348.23
        },
        {
          "h": 101,
          "v": 350.13
        },
        {
          "h": 115,
          "v": 349.97
        }
      ],
      "s8": [
        {
          "h": 1,
          "v": 2228.76
        },
        {
          "h": 15,
          "v": 2228.81
        },
        {
          "h": 29,
          "v": 2228.69
        },
        {
          "h": 44,
          "v": 2228.75
        },
        {
          "h": 58,
          "v": 2228.74
        },
        {
          "h": 72,
          "v": 2228.67
        },
        {
          "h": 87,
          "v": 2228.58
        },
        {
          "h": 101,
          "v": 2228.85
        },
        {
          "h": 115,
          "v": 2228.83
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #115 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 17.4 사이클 소진 시 정비 권고 (90% 구간 2.0~26.0)",
        "status": "scheduled"
      }
    ],
    "status": "critical",
    "health": 14
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
      "median": 8.9,
      "lower": 0.0,
      "upper": 16.0
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 0.9293,
        "lower": 0.9247,
        "upper": 0.9337
      },
      {
        "cycle": 10,
        "median": 0.241,
        "lower": 0.1975,
        "upper": 0.2852
      },
      {
        "cycle": 20,
        "median": 0.0303,
        "lower": 0.0,
        "upper": 0.0662
      },
      {
        "cycle": 30,
        "median": 0.0303,
        "lower": 0.0,
        "upper": 0.0662
      },
      {
        "cycle": 40,
        "median": 0.0303,
        "lower": 0.0,
        "upper": 0.0662
      },
      {
        "cycle": 60,
        "median": 0.0303,
        "lower": 0.0,
        "upper": 0.0662
      },
      {
        "cycle": 80,
        "median": 0.0303,
        "lower": 0.0,
        "upper": 0.0662
      },
      {
        "cycle": 100,
        "median": 0.0303,
        "lower": 0.0,
        "upper": 0.0662
      },
      {
        "cycle": 120,
        "median": 0.0303,
        "lower": 0.0,
        "upper": 0.0662
      },
      {
        "cycle": 140,
        "median": 0.0303,
        "lower": 0.0,
        "upper": 0.0662
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
        "description": "RUL 예측 기반 — 잔존 약 8.9 사이클 소진 시 정비 권고 (90% 구간 0.0~16.0)",
        "status": "scheduled"
      }
    ],
    "status": "critical",
    "health": 7
  },
  "engine-103": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD004 (운전조건 6종 · 고장모드 2종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 145,
      "modelNo": "Turbofan (FD004)"
    },
    "predictedRulCycle": {
      "median": 9.0,
      "lower": 0.0,
      "upper": 16.1
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 0.9299,
        "lower": 0.9252,
        "upper": 0.9342
      },
      {
        "cycle": 10,
        "median": 0.2475,
        "lower": 0.2018,
        "upper": 0.2925
      },
      {
        "cycle": 20,
        "median": 0.0315,
        "lower": 0.0,
        "upper": 0.0673
      },
      {
        "cycle": 30,
        "median": 0.0315,
        "lower": 0.0,
        "upper": 0.0673
      },
      {
        "cycle": 40,
        "median": 0.0315,
        "lower": 0.0,
        "upper": 0.0673
      },
      {
        "cycle": 60,
        "median": 0.0315,
        "lower": 0.0,
        "upper": 0.0673
      },
      {
        "cycle": 80,
        "median": 0.0315,
        "lower": 0.0,
        "upper": 0.0673
      },
      {
        "cycle": 100,
        "median": 0.0315,
        "lower": 0.0,
        "upper": 0.0673
      },
      {
        "cycle": 120,
        "median": 0.0315,
        "lower": 0.0,
        "upper": 0.0673
      },
      {
        "cycle": 140,
        "median": 0.0315,
        "lower": 0.0,
        "upper": 0.0673
      }
    ],
    "sensorTrend": {
      "s3": [
        {
          "h": 1,
          "v": 1414.31
        },
        {
          "h": 19,
          "v": 1414.5
        },
        {
          "h": 37,
          "v": 1414.38
        },
        {
          "h": 55,
          "v": 1413.19
        },
        {
          "h": 73,
          "v": 1412.99
        },
        {
          "h": 91,
          "v": 1415.44
        },
        {
          "h": 109,
          "v": 1412.62
        },
        {
          "h": 127,
          "v": 1425.07
        },
        {
          "h": 145,
          "v": 1426.17
        }
      ],
      "s17": [
        {
          "h": 1,
          "v": 347.98
        },
        {
          "h": 19,
          "v": 347.23
        },
        {
          "h": 37,
          "v": 346.23
        },
        {
          "h": 55,
          "v": 346.13
        },
        {
          "h": 73,
          "v": 345.97
        },
        {
          "h": 91,
          "v": 347.97
        },
        {
          "h": 109,
          "v": 348.23
        },
        {
          "h": 127,
          "v": 349.97
        },
        {
          "h": 145,
          "v": 350.23
        }
      ],
      "s8": [
        {
          "h": 1,
          "v": 2228.85
        },
        {
          "h": 19,
          "v": 2228.94
        },
        {
          "h": 37,
          "v": 2228.87
        },
        {
          "h": 55,
          "v": 2228.73
        },
        {
          "h": 73,
          "v": 2228.76
        },
        {
          "h": 91,
          "v": 2228.75
        },
        {
          "h": 109,
          "v": 2228.73
        },
        {
          "h": 127,
          "v": 2228.8
        },
        {
          "h": 145,
          "v": 2228.36
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #145 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 9.0 사이클 소진 시 정비 권고 (90% 구간 0.0~16.1)",
        "status": "scheduled"
      }
    ],
    "status": "critical",
    "health": 7
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
      "median": 10.1,
      "lower": 0.0,
      "upper": 17.0
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 0.9367,
        "lower": 0.9318,
        "upper": 0.9414
      },
      {
        "cycle": 10,
        "median": 0.2911,
        "lower": 0.25,
        "upper": 0.3321
      },
      {
        "cycle": 20,
        "median": 0.031,
        "lower": 0.0,
        "upper": 0.0672
      },
      {
        "cycle": 30,
        "median": 0.031,
        "lower": 0.0,
        "upper": 0.0672
      },
      {
        "cycle": 40,
        "median": 0.031,
        "lower": 0.0,
        "upper": 0.0672
      },
      {
        "cycle": 60,
        "median": 0.031,
        "lower": 0.0,
        "upper": 0.0672
      },
      {
        "cycle": 80,
        "median": 0.031,
        "lower": 0.0,
        "upper": 0.0672
      },
      {
        "cycle": 100,
        "median": 0.031,
        "lower": 0.0,
        "upper": 0.0672
      },
      {
        "cycle": 120,
        "median": 0.031,
        "lower": 0.0,
        "upper": 0.0672
      },
      {
        "cycle": 140,
        "median": 0.031,
        "lower": 0.0,
        "upper": 0.0672
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
        "description": "RUL 예측 기반 — 잔존 약 10.1 사이클 소진 시 정비 권고 (90% 구간 0.0~17.0)",
        "status": "scheduled"
      }
    ],
    "status": "critical",
    "health": 8
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
      "median": 11.2,
      "lower": 0.0,
      "upper": 18.0
    },
    "survivalCurve": [
      {
        "cycle": 0,
        "median": 0.941,
        "lower": 0.9367,
        "upper": 0.9451
      },
      {
        "cycle": 10,
        "median": 0.3379,
        "lower": 0.3009,
        "upper": 0.3768
      },
      {
        "cycle": 20,
        "median": 0.0352,
        "lower": 0.0038,
        "upper": 0.0711
      },
      {
        "cycle": 30,
        "median": 0.0314,
        "lower": 0.0,
        "upper": 0.0674
      },
      {
        "cycle": 40,
        "median": 0.0314,
        "lower": 0.0,
        "upper": 0.0674
      },
      {
        "cycle": 60,
        "median": 0.0314,
        "lower": 0.0,
        "upper": 0.0674
      },
      {
        "cycle": 80,
        "median": 0.0314,
        "lower": 0.0,
        "upper": 0.0674
      },
      {
        "cycle": 100,
        "median": 0.0314,
        "lower": 0.0,
        "upper": 0.0674
      },
      {
        "cycle": 120,
        "median": 0.0314,
        "lower": 0.0,
        "upper": 0.0674
      },
      {
        "cycle": 140,
        "median": 0.0314,
        "lower": 0.0,
        "upper": 0.0674
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
        "description": "RUL 예측 기반 — 잔존 약 11.2 사이클 소진 시 정비 권고 (90% 구간 0.0~18.0)",
        "status": "scheduled"
      }
    ],
    "status": "critical",
    "health": 9
  },
  "engine-70": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD004 (운전조건 6종 · 고장모드 2종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 190,
      "modelNo": "Turbofan (FD004)"
    },
    "predictedRulCycle": {
      "median": 40.5,
      "lower": 25.0,
      "upper": 63.8
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
        "median": 0.9906,
        "lower": 0.9812,
        "upper": 0.9976
      },
      {
        "cycle": 30,
        "median": 0.7762,
        "lower": 0.7344,
        "upper": 0.8166
      },
      {
        "cycle": 40,
        "median": 0.3842,
        "lower": 0.3204,
        "upper": 0.446
      },
      {
        "cycle": 60,
        "median": 0.0557,
        "lower": 0.0038,
        "upper": 0.1029
      },
      {
        "cycle": 80,
        "median": 0.0463,
        "lower": 0.0,
        "upper": 0.0897
      },
      {
        "cycle": 100,
        "median": 0.0429,
        "lower": 0.0,
        "upper": 0.0839
      },
      {
        "cycle": 120,
        "median": 0.0429,
        "lower": 0.0,
        "upper": 0.0839
      },
      {
        "cycle": 140,
        "median": 0.0429,
        "lower": 0.0,
        "upper": 0.0839
      }
    ],
    "sensorTrend": {
      "s3": [
        {
          "h": 1,
          "v": 1418.6
        },
        {
          "h": 25,
          "v": 1416.93
        },
        {
          "h": 48,
          "v": 1409.86
        },
        {
          "h": 72,
          "v": 1414.27
        },
        {
          "h": 95,
          "v": 1414.79
        },
        {
          "h": 119,
          "v": 1414.01
        },
        {
          "h": 143,
          "v": 1416.73
        },
        {
          "h": 166,
          "v": 1418.71
        },
        {
          "h": 190,
          "v": 1420.04
        }
      ],
      "s17": [
        {
          "h": 1,
          "v": 346.97
        },
        {
          "h": 25,
          "v": 347.97
        },
        {
          "h": 48,
          "v": 345.97
        },
        {
          "h": 72,
          "v": 349.97
        },
        {
          "h": 95,
          "v": 347.23
        },
        {
          "h": 119,
          "v": 346.23
        },
        {
          "h": 143,
          "v": 348.23
        },
        {
          "h": 166,
          "v": 347.97
        },
        {
          "h": 190,
          "v": 349.07
        }
      ],
      "s8": [
        {
          "h": 1,
          "v": 2228.71
        },
        {
          "h": 25,
          "v": 2228.75
        },
        {
          "h": 48,
          "v": 2228.7
        },
        {
          "h": 72,
          "v": 2228.71
        },
        {
          "h": 95,
          "v": 2228.76
        },
        {
          "h": 119,
          "v": 2228.78
        },
        {
          "h": 143,
          "v": 2228.88
        },
        {
          "h": 166,
          "v": 2228.9
        },
        {
          "h": 190,
          "v": 2229.04
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #190 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 40.5 사이클 소진 시 정비 권고 (90% 구간 25.0~63.8)",
        "status": "scheduled"
      }
    ],
    "status": "warning",
    "health": 32
  },
  "engine-185": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD004 (운전조건 6종 · 고장모드 2종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 203,
      "modelNo": "Turbofan (FD004)"
    },
    "predictedRulCycle": {
      "median": 40.7,
      "lower": 25.0,
      "upper": 63.0
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
        "median": 0.9914,
        "lower": 0.9821,
        "upper": 0.9982
      },
      {
        "cycle": 30,
        "median": 0.7819,
        "lower": 0.7401,
        "upper": 0.8208
      },
      {
        "cycle": 40,
        "median": 0.3901,
        "lower": 0.325,
        "upper": 0.451
      },
      {
        "cycle": 60,
        "median": 0.0547,
        "lower": 0.0047,
        "upper": 0.1001
      },
      {
        "cycle": 80,
        "median": 0.0445,
        "lower": 0.0,
        "upper": 0.0875
      },
      {
        "cycle": 100,
        "median": 0.0411,
        "lower": 0.0,
        "upper": 0.0793
      },
      {
        "cycle": 120,
        "median": 0.0411,
        "lower": 0.0,
        "upper": 0.0793
      },
      {
        "cycle": 140,
        "median": 0.0411,
        "lower": 0.0,
        "upper": 0.0793
      }
    ],
    "sensorTrend": {
      "s3": [
        {
          "h": 1,
          "v": 1412.53
        },
        {
          "h": 26,
          "v": 1413.29
        },
        {
          "h": 51,
          "v": 1415.99
        },
        {
          "h": 77,
          "v": 1424.32
        },
        {
          "h": 102,
          "v": 1414.39
        },
        {
          "h": 127,
          "v": 1420.89
        },
        {
          "h": 153,
          "v": 1417.92
        },
        {
          "h": 178,
          "v": 1428.94
        },
        {
          "h": 203,
          "v": 1419.05
        }
      ],
      "s17": [
        {
          "h": 1,
          "v": 348.23
        },
        {
          "h": 26,
          "v": 347.13
        },
        {
          "h": 51,
          "v": 346.97
        },
        {
          "h": 77,
          "v": 348.23
        },
        {
          "h": 102,
          "v": 347.13
        },
        {
          "h": 127,
          "v": 346.23
        },
        {
          "h": 153,
          "v": 347.07
        },
        {
          "h": 178,
          "v": 349.13
        },
        {
          "h": 203,
          "v": 348.13
        }
      ],
      "s8": [
        {
          "h": 1,
          "v": 2228.71
        },
        {
          "h": 26,
          "v": 2228.72
        },
        {
          "h": 51,
          "v": 2228.72
        },
        {
          "h": 77,
          "v": 2228.71
        },
        {
          "h": 102,
          "v": 2228.81
        },
        {
          "h": 127,
          "v": 2228.7
        },
        {
          "h": 153,
          "v": 2228.72
        },
        {
          "h": 178,
          "v": 2228.82
        },
        {
          "h": 203,
          "v": 2228.89
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #203 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 40.7 사이클 소진 시 정비 권고 (90% 구간 25.0~63.0)",
        "status": "scheduled"
      }
    ],
    "status": "warning",
    "health": 33
  },
  "engine-221": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD004 (운전조건 6종 · 고장모드 2종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 176,
      "modelNo": "Turbofan (FD004)"
    },
    "predictedRulCycle": {
      "median": 80.5,
      "lower": 61.0,
      "upper": 168.0
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
        "median": 0.9612,
        "lower": 0.94,
        "upper": 0.9782
      },
      {
        "cycle": 80,
        "median": 0.4877,
        "lower": 0.4012,
        "upper": 0.5741
      },
      {
        "cycle": 100,
        "median": 0.1373,
        "lower": 0.0572,
        "upper": 0.221
      },
      {
        "cycle": 120,
        "median": 0.0777,
        "lower": 0.0027,
        "upper": 0.1525
      },
      {
        "cycle": 140,
        "median": 0.0644,
        "lower": 0.0027,
        "upper": 0.1324
      }
    ],
    "sensorTrend": {
      "s3": [
        {
          "h": 1,
          "v": 1411.98
        },
        {
          "h": 23,
          "v": 1416.96
        },
        {
          "h": 45,
          "v": 1424.25
        },
        {
          "h": 67,
          "v": 1423.46
        },
        {
          "h": 89,
          "v": 1413.41
        },
        {
          "h": 110,
          "v": 1405.89
        },
        {
          "h": 132,
          "v": 1413.17
        },
        {
          "h": 154,
          "v": 1421.67
        },
        {
          "h": 176,
          "v": 1412.08
        }
      ],
      "s17": [
        {
          "h": 1,
          "v": 346.07
        },
        {
          "h": 23,
          "v": 346.98
        },
        {
          "h": 45,
          "v": 345.98
        },
        {
          "h": 67,
          "v": 347.98
        },
        {
          "h": 89,
          "v": 347.07
        },
        {
          "h": 110,
          "v": 346.07
        },
        {
          "h": 132,
          "v": 348.13
        },
        {
          "h": 154,
          "v": 347.6
        },
        {
          "h": 176,
          "v": 345.98
        }
      ],
      "s8": [
        {
          "h": 1,
          "v": 2228.62
        },
        {
          "h": 23,
          "v": 2228.76
        },
        {
          "h": 45,
          "v": 2228.7
        },
        {
          "h": 67,
          "v": 2228.71
        },
        {
          "h": 89,
          "v": 2228.75
        },
        {
          "h": 110,
          "v": 2228.69
        },
        {
          "h": 132,
          "v": 2228.69
        },
        {
          "h": 154,
          "v": 2228.73
        },
        {
          "h": 176,
          "v": 2228.84
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #176 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 80.5 사이클 소진 시 정비 권고 (90% 구간 61.0~168.0)",
        "status": "scheduled"
      }
    ],
    "status": "good",
    "health": 64
  },
  "engine-10": {
    "equipmentInfo": {
      "line": "터보팬 엔진 · C-MAPSS FD004 (운전조건 6종 · 고장모드 2종)",
      "installedAt": "—(데이터에 없음)",
      "lastMaintenance": "—(단일 run-to-failure 데이터, 실제 교체 이력 없음)",
      "team": "5조 감시자들",
      "operatingCycles": 23,
      "modelNo": "Turbofan (FD004)"
    },
    "predictedRulCycle": {
      "median": 125.0,
      "lower": 125.0,
      "upper": 368.0
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
        "median": 1.0,
        "lower": 1.0,
        "upper": 1.0
      },
      {
        "cycle": 100,
        "median": 0.993,
        "lower": 0.9851,
        "upper": 0.9988
      },
      {
        "cycle": 120,
        "median": 0.9639,
        "lower": 0.9406,
        "upper": 0.9812
      },
      {
        "cycle": 140,
        "median": 0.8944,
        "lower": 0.8505,
        "upper": 0.9311
      }
    ],
    "sensorTrend": {
      "s3": [
        {
          "h": 1,
          "v": 1408.34
        },
        {
          "h": 4,
          "v": 1417.98
        },
        {
          "h": 7,
          "v": 1419.55
        },
        {
          "h": 9,
          "v": 1414.21
        },
        {
          "h": 12,
          "v": 1417.04
        },
        {
          "h": 15,
          "v": 1412.1
        },
        {
          "h": 17,
          "v": 1411.2
        },
        {
          "h": 20,
          "v": 1412.74
        },
        {
          "h": 23,
          "v": 1423.67
        }
      ],
      "s17": [
        {
          "h": 1,
          "v": 346.98
        },
        {
          "h": 4,
          "v": 346.07
        },
        {
          "h": 7,
          "v": 347.07
        },
        {
          "h": 9,
          "v": 347.6
        },
        {
          "h": 12,
          "v": 345.6
        },
        {
          "h": 15,
          "v": 347.07
        },
        {
          "h": 17,
          "v": 347.07
        },
        {
          "h": 20,
          "v": 347.6
        },
        {
          "h": 23,
          "v": 348.13
        }
      ],
      "s8": [
        {
          "h": 1,
          "v": 2228.83
        },
        {
          "h": 4,
          "v": 2228.74
        },
        {
          "h": 7,
          "v": 2228.73
        },
        {
          "h": 9,
          "v": 2228.76
        },
        {
          "h": 12,
          "v": 2228.77
        },
        {
          "h": 15,
          "v": 2228.7
        },
        {
          "h": 17,
          "v": 2228.8
        },
        {
          "h": 20,
          "v": 2228.71
        },
        {
          "h": 23,
          "v": 2228.78
        }
      ]
    },
    "maintenanceHistory": [
      {
        "date": "cycle #23 시점 예측",
        "type": "예정",
        "description": "RUL 예측 기반 — 잔존 약 125.0 사이클 소진 시 정비 권고 (90% 구간 125.0~368.0)",
        "status": "scheduled"
      }
    ],
    "status": "good",
    "health": 100
  }
}

export const DEFAULT_ENGINE_ID = 'engine-198'

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

/** 위험 등급(RUL≤40) 조기경보 성능 — GRU 모델, 공식 test 248개 엔진 기준 실제 계산값
 *  (TP=61, FN=8, FP=1) */
export const classifierMetrics = {
  "precision": 0.984,
  "recall": 0.884,
  "f1": 0.931,
  "tp": 61,
  "fn": 8,
  "fp": 1
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
 *  탐지/누락/오탐 건수는 GRU 모델의 실제 test set 결과, 금액·시간 가정은 항공 엔진 정비
 *  맥락의 illustrative 값입니다 (분석 레포 scripts/08_export_dashboard_ts.py 참고). */
export const scenarioCompare = [
  {
    "metric": "평균 가동중단 시간 (시간/1,000대 환산)",
    "asIs": 6677.4,
    "toBe": 1760.1
  },
  {
    "metric": "정비 비용 (억원/1,000대 환산)",
    "asIs": 66.8,
    "toBe": 27.4
  },
  {
    "metric": "가동중단 손실 (억원/1,000대 환산)",
    "asIs": 667.7,
    "toBe": 176.0
  }
]

export const savingsSummary = {
  "perThousandEnginesEok": 531.1,
  "savingRatePct": 72.3,
  "dangerCaseReductionPct": 88,
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
      "detected": 61,
      "missed": 8,
      "falseAlarms": 1,
      "dangerCases": 69,
      "dangerCasesDetected": 0,
      "dangerCasesMissed": 8,
      "hoursPlanned": 4.0,
      "hoursUnplanned": 24.0,
      "repairPlanned": 800.0
    }
  }
}
