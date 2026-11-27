// ─── Deterministic ADAS Simulator ───────────────────────────────────────────
// Generates realistic FrameInferencePayload objects at ~30 Hz
// so the UI can be fully demonstrated without a GPU backend.

import {
  FrameInferencePayload,
  TrackedObject,
  LaneLine,
  WeatherCondition,
  ObjectClass,
  RiskLevel,
} from '../types/adas';

// ─── Scenario presets ───────────────────────────────────────────────────────
export type ScenarioName =
  | 'highway'
  | 'pedestrian_crossing'
  | 'lead_car_brake'
  | 'school_zone'
  | 'heavy_rain_night';

interface SimObject {
  id: number;
  cls: ObjectClass;
  x: number; y: number;       // normalised centre
  w: number; h: number;       // normalised size
  vx: number; vy: number;     // velocity (norm/s)
  distance: number;
  speed: number;
  signLabel?: string;
  lightState?: 'red' | 'yellow' | 'green';
}

interface SimState {
  frameId: number;
  t: number;
  weather: WeatherCondition;
  egoSpeed: number;
  objects: SimObject[];
  scenario: ScenarioName;
}

const CLASS_HEIGHTS: Record<string, number> = {
  car: 1.5, truck: 3.2, bus: 3.4, pedestrian: 1.7,
  rider: 1.7, bicycle: 1.2, motorcycle: 1.3,
  traffic_light: 0.6, traffic_sign: 0.7,
};

function clamp(v: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, v)); }
function rand(a: number, b: number) { return a + Math.random() * (b - a); }

// ─── Default lane geometry ──────────────────────────────────────────────────
function makeLanes(): LaneLine[] {
  return [
    { side: 'left_ego',  marking_type: 'white_solid',  points_norm: [[0.30, 1],[0.42, 0.55],[0.47, 0.35]] },
    { side: 'right_ego', marking_type: 'white_dashed', points_norm: [[0.70, 1],[0.58, 0.55],[0.53, 0.35]] },
    { side: 'left_adj',  marking_type: 'yellow_solid', points_norm: [[0.12, 1],[0.36, 0.55],[0.44, 0.35]] },
    { side: 'right_adj', marking_type: 'white_solid',  points_norm: [[0.88, 1],[0.64, 0.55],[0.56, 0.35]] },
  ];
}

// ─── Drivable area ──────────────────────────────────────────────────────────
function makeDrivable(): [number, number][] {
  return [[0.12,1],[0.36,0.55],[0.44,0.35],[0.56,0.35],[0.64,0.55],[0.88,1]];
}

// ─── Build scenario objects ─────────────────────────────────────────────────
function scenarioObjects(name: ScenarioName, t: number): SimObject[] {
  const objs: SimObject[] = [];
  const id = (i: number) => i + 1;

  switch (name) {
    case 'highway': {
      // Lead car cruising
      objs.push({ id: id(0), cls:'car', x:0.50, y:0.52 + Math.sin(t*0.3)*0.01, w:0.08, h:0.06,
        vx:0, vy:-0.005, distance: 28 + Math.sin(t)*3, speed: 80, });
      // Truck right lane
      objs.push({ id: id(1), cls:'truck', x:0.65, y:0.48 + Math.sin(t*0.2)*0.008, w:0.10, h:0.09,
        vx:0, vy:-0.003, distance: 42, speed: 70, });
      // Motorcycle left
      objs.push({ id: id(2), cls:'motorcycle', x:0.35, y:0.56, w:0.04, h:0.05,
        vx:0.002, vy:-0.008, distance: 18, speed: 95, });
      break;
    }
    case 'pedestrian_crossing': {
      // Pedestrian crossing from left
      const px = 0.25 + (t % 8) * 0.06;
      objs.push({ id: id(0), cls:'pedestrian', x: clamp(px, 0.1, 0.9), y:0.62, w:0.03, h:0.08,
        vx:0.06, vy:0, distance: clamp(12 - (t % 8)*1.2, 3, 15), speed: 5, });
      // Stopped car
      objs.push({ id: id(1), cls:'car', x:0.50, y:0.50, w:0.09, h:0.06,
        vx:0, vy:0, distance: 22, speed: 0, });
      // Speed limit sign
      objs.push({ id: id(2), cls:'traffic_sign', x:0.15, y:0.38, w:0.04, h:0.05,
        vx:0, vy:0, distance: 30, speed: 0, signLabel:'SPEED LIMIT 30', });
      break;
    }
    case 'lead_car_brake': {
      // Lead car braking hard
      const dist = clamp(35 - (t % 6) * 6, 4, 40);
      const yPos = 0.45 + (1 - dist / 40) * 0.20;
      const sz = 0.06 + (1 - dist / 40) * 0.06;
      objs.push({ id: id(0), cls:'car', x:0.50, y: yPos, w: sz, h: sz * 0.7,
        vx:0, vy:0.01, distance: dist, speed: clamp(90 - (t % 6)*15, 0, 90), });
      objs.push({ id: id(1), cls:'car', x:0.35, y:0.44, w:0.07, h:0.05,
        vx:-0.001, vy:-0.002, distance: 38, speed: 85, });
      break;
    }
    case 'school_zone': {
      // School zone sign
      objs.push({ id: id(0), cls:'traffic_sign', x:0.82, y:0.36, w:0.045, h:0.055,
        vx:0, vy:0, distance: 25, speed: 0, signLabel:'SCHOOL ZONE', });
      // Red traffic light
      objs.push({ id: id(1), cls:'traffic_light', x:0.50, y:0.28, w:0.025, h:0.06,
        vx:0, vy:0, distance: 32, speed: 0, lightState: (t % 12 < 7) ? 'red' : 'green', });
      // Child pedestrian
      objs.push({ id: id(2), cls:'pedestrian', x: 0.55 + Math.sin(t * 0.5) * 0.08, y:0.58, w:0.025, h:0.065,
        vx:0.02, vy:0, distance: 14, speed: 4, });
      // Parked bus
      objs.push({ id: id(3), cls:'bus', x:0.78, y:0.52, w:0.11, h:0.09,
        vx:0, vy:0, distance: 20, speed: 0, });
      break;
    }
    case 'heavy_rain_night': {
      objs.push({ id: id(0), cls:'car', x:0.48, y:0.54, w:0.09, h:0.065,
        vx:0.001, vy:-0.004, distance: 20, speed: 60, });
      objs.push({ id: id(1), cls:'truck', x:0.68, y:0.46, w:0.11, h:0.10,
        vx:0, vy:-0.002, distance: 45, speed: 55, });
      objs.push({ id: id(2), cls:'bicycle', x:0.30, y:0.60, w:0.035, h:0.05,
        vx:0.003, vy:-0.001, distance: 12, speed: 15, });
      break;
    }
  }
  return objs;
}

function weatherForScenario(s: ScenarioName): WeatherCondition {
  if (s === 'heavy_rain_night') return Math.random() > 0.5 ? 'rain' : 'night';
  return 'clear_day';
}

function riskFromTTC(ttc: number | null, dist: number, cls: ObjectClass): RiskLevel {
  if (ttc !== null && ttc < 2.5) return 'critical';
  if (ttc !== null && ttc < 4.0) return 'caution';
  if ((cls === 'pedestrian' || cls === 'bicycle' || cls === 'rider') && dist < 8) return 'critical';
  if (dist < 12) return 'caution';
  return 'safe';
}

// ─── Main simulator class ───────────────────────────────────────────────────
export class AdasSimulator {
  private state: SimState;
  private modelVariant = 'Hybrid-S (CNN + 2-Layer Transformer)';
  private precision: 'FP32' | 'FP16' | 'INT8' = 'FP32';
  private weatherOverride: WeatherCondition | null = null;

  constructor() {
    this.state = {
      frameId: 0,
      t: 0,
      weather: 'clear_day',
      egoSpeed: 72,
      objects: [],
      scenario: 'highway',
    };
  }

  setScenario(name: ScenarioName) { this.state.scenario = name; this.state.t = 0; this.state.frameId = 0; }
  setModel(v: string) { this.modelVariant = v; }
  setPrecision(p: 'FP32' | 'FP16' | 'INT8') { this.precision = p; }
  setWeatherOverride(w: WeatherCondition | null) { this.weatherOverride = w; }

  step(dt: number): FrameInferencePayload {
    this.state.t += dt;
    this.state.frameId++;

    const scenario = this.state.scenario;
    const weather = this.weatherOverride ?? weatherForScenario(scenario);
    const simObjs = scenarioObjects(scenario, this.state.t);

    // ego speed varies by scenario
    let egoSpeed = 72;
    if (scenario === 'school_zone') egoSpeed = 28;
    if (scenario === 'heavy_rain_night') egoSpeed = 50;
    if (scenario === 'lead_car_brake') egoSpeed = 85;
    if (scenario === 'pedestrian_crossing') egoSpeed = 35;
    this.state.egoSpeed = egoSpeed + Math.sin(this.state.t) * 2;

    // convert SimObjects → TrackedObjects
    const objects: TrackedObject[] = simObjs.map(o => {
      const relSpeed = this.state.egoSpeed - o.speed;
      const closing = relSpeed > 0;
      const ttc = closing && o.distance > 0 ? (o.distance / (relSpeed / 3.6)) : null;
      const risk = riskFromTTC(ttc, o.distance, o.cls);

      return {
        track_id: o.id,
        cls: o.cls,
        confidence: clamp(0.78 + Math.random() * 0.20, 0, 1),
        bbox: [
          clamp(o.x - o.w / 2, 0, 1),
          clamp(o.y - o.h / 2, 0, 1),
          clamp(o.x + o.w / 2, 0, 1),
          clamp(o.y + o.h / 2, 0, 1),
        ] as [number, number, number, number],
        distance_m: +o.distance.toFixed(1),
        abs_speed_kmh: +o.speed.toFixed(1),
        rel_speed_kmh: +relSpeed.toFixed(1),
        ttc_s: ttc !== null ? +ttc.toFixed(2) : null,
        risk_level: risk,
        trajectory_norm: [
          [o.x, o.y],
          [o.x + o.vx * 30, o.y + o.vy * 30],
          [o.x + o.vx * 60, o.y + o.vy * 60],
        ],
        history_norm: [
          [o.x - o.vx * 20, o.y - o.vy * 20],
          [o.x - o.vx * 10, o.y - o.vy * 10],
          [o.x, o.y],
        ],
        sign_label: o.signLabel,
        sign_confidence: o.signLabel ? 0.92 + Math.random() * 0.07 : undefined,
        light_state: o.lightState,
      };
    });

    const hasBrake = objects.some(o =>
      (o.ttc_s !== null && o.ttc_s < 2.5) ||
      ((o.cls === 'pedestrian' || o.cls === 'bicycle') && o.distance_m < 8)
    );

    const alerts: string[] = [];
    objects.forEach(o => {
      if (o.risk_level === 'critical') {
        alerts.push(`[ALERT] CRITICAL: ${o.cls.toUpperCase()} ID:${String(o.track_id).padStart(2,'0')} | Dist: ${o.distance_m}m | TTC: ${o.ttc_s?.toFixed(1) ?? 'N/A'}s`);
      }
      if (o.sign_label) {
        alerts.push(`[SIGN] GTSRB: ${o.sign_label} (${((o.sign_confidence ?? 0) * 100).toFixed(1)}%)`);
      }
    });

    // simulated latency by model
    const latBase = this.modelVariant.includes('YOLOv8') ? 8 : this.modelVariant.includes('RT-DETR') ? 22 : 12;
    const precMul = this.precision === 'INT8' ? 0.5 : this.precision === 'FP16' ? 0.7 : 1.0;
    const jitter = () => rand(0.8, 1.2);

    const latency = {
      preprocess: +(1.2 * jitter()).toFixed(1),
      backbone: +(latBase * 0.45 * precMul * jitter()).toFixed(1),
      transformer_neck: +(latBase * 0.30 * precMul * jitter()).toFixed(1),
      tracker: +(1.8 * jitter()).toFixed(1),
      telemetry: +(0.4 * jitter()).toFixed(1),
      total: 0,
    };
    latency.total = +(latency.preprocess + latency.backbone + latency.transformer_neck + latency.tracker + latency.telemetry).toFixed(1);

    return {
      timestamp_ms: Date.now(),
      frame_id: this.state.frameId,
      model_variant: this.modelVariant,
      precision: this.precision,
      fps: +(1000 / Math.max(latency.total, 1)).toFixed(1),
      latency_ms: latency,
      ego_speed_kmh: +this.state.egoSpeed.toFixed(1),
      weather: { condition: weather, confidence: 0.90 + Math.random() * 0.09 },
      drivable_area_polygon_norm: makeDrivable(),
      lanes: makeLanes(),
      objects,
      emergency_brake_warning: hasBrake,
      alerts,
    };
  }
}
