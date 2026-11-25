// ─── ADAS Inference Payload Types ───────────────────────────────────────────

export type ObjectClass =
  | 'car' | 'truck' | 'bus' | 'pedestrian' | 'rider'
  | 'bicycle' | 'motorcycle' | 'traffic_light' | 'traffic_sign';

export type LaneMarkingType =
  | 'white_dashed' | 'white_solid' | 'yellow_solid' | 'double_yellow';

export type WeatherCondition = 'clear_day' | 'night' | 'rain' | 'fog';
export type RiskLevel = 'safe' | 'caution' | 'critical';
export type InputMode = 'simulator' | 'camera' | 'video';

export interface TrackedObject {
  track_id: number;
  cls: ObjectClass;
  confidence: number;
  bbox: [number, number, number, number];   // normalised [x1,y1,x2,y2]
  distance_m: number;
  abs_speed_kmh: number;
  rel_speed_kmh: number;
  ttc_s: number | null;
  risk_level: RiskLevel;
  trajectory_norm: [number, number][];
  history_norm: [number, number][];
  sign_label?: string;
  sign_confidence?: number;
  light_state?: 'red' | 'yellow' | 'green' | 'unknown';
}

export interface LaneLine {
  side: 'left_ego' | 'right_ego' | 'left_adj' | 'right_adj';
  marking_type: LaneMarkingType;
  points_norm: [number, number][];
}

export interface FrameInferencePayload {
  timestamp_ms: number;
  frame_id: number;
  model_variant: string;
  precision: 'FP32' | 'FP16' | 'INT8';
  fps: number;
  latency_ms: {
    preprocess: number;
    backbone: number;
    transformer_neck: number;
    tracker: number;
    telemetry: number;
    total: number;
  };
  ego_speed_kmh: number;
  weather: { condition: WeatherCondition; confidence: number };
  drivable_area_polygon_norm: [number, number][];
  lanes: LaneLine[];
  objects: TrackedObject[];
  emergency_brake_warning: boolean;
  alerts: string[];
}

export interface LogEntry {
  ts: number;
  level: 'INFO' | 'ALERT' | 'SIGN' | 'SYS';
  msg: string;
}
