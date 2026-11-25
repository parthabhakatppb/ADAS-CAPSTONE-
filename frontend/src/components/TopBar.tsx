import React from 'react';
import { InputMode, WeatherCondition } from '../types/adas';
import { ScenarioName } from '../engine/simulator';
import {
  Activity, Camera, Film, Volume2, VolumeX,
  CloudRain, Sun, Moon, CloudFog,
} from 'lucide-react';

interface TopBarProps {
  mode: InputMode;
  onModeChange: (m: InputMode) => void;
  connected: boolean;
  modelVariant: string;
  onModelChange: (v: string) => void;
  precision: 'FP32' | 'FP16' | 'INT8';
  onPrecisionChange: (p: 'FP32' | 'FP16' | 'INT8') => void;
  weatherOverride: string | null;
  onWeatherChange: (w: string | null) => void;
  scenario: ScenarioName;
  onScenarioChange: (s: ScenarioName) => void;
  muted: boolean;
  onMuteToggle: () => void;
}

const MODELS = [
  'Hybrid-S (CNN + 1-Layer ViT)',
  'Hybrid-S (CNN + 2-Layer ViT)',
  'YOLOv8-S (Pure CNN Baseline)',
  'RT-DETR (Pure Transformer)',
];

const SCENARIOS: { key: ScenarioName; label: string }[] = [
  { key: 'highway', label: 'Normal Highway' },
  { key: 'pedestrian_crossing', label: 'Pedestrian Crossing' },
  { key: 'lead_car_brake', label: 'Lead-Car Brake' },
  { key: 'school_zone', label: 'School Zone + Red Light' },
  { key: 'heavy_rain_night', label: 'Rain / Night Stress' },
];

const modeBtnClass = (active: boolean) =>
  `px-3 py-1.5 text-xs font-semibold rounded-md transition-all ${
    active
      ? 'bg-drivable/20 text-drivable border border-drivable/40'
      : 'bg-white/5 text-gray-400 border border-transparent hover:bg-white/10 hover:text-white'
  }`;

const TopBar: React.FC<TopBarProps> = ({
  mode, onModeChange, connected,
  modelVariant, onModelChange,
  precision, onPrecisionChange,
  weatherOverride, onWeatherChange,
  scenario, onScenarioChange,
  muted, onMuteToggle,
}) => {
  const statusPill = mode === 'simulator'
    ? { label: 'SIMULATION', color: 'bg-drivable' }
    : connected
      ? { label: 'BACKEND LIVE', color: 'bg-safe' }
      : { label: 'DISCONNECTED', color: 'bg-critical' };

  return (
    <div className="flex items-center gap-3 h-14 px-4 bg-card border-b border-border flex-shrink-0 select-none">
      {/* Title */}
      <div className="flex items-center gap-2 mr-2">
        <Activity className="w-5 h-5 text-drivable" />
        <span className="text-sm font-bold tracking-wide whitespace-nowrap">HYBRID CNN–ViT ADAS</span>
        <span className={`${statusPill.color} text-[10px] font-bold px-2 py-0.5 rounded-full text-black`}>
          {statusPill.label}
        </span>
      </div>

      {/* Mode switcher */}
      <div className="flex gap-1">
        <button className={modeBtnClass(mode === 'simulator')} onClick={() => onModeChange('simulator')}>
          <Activity className="w-3 h-3 inline mr-1" />Simulator
        </button>
        <button className={modeBtnClass(mode === 'camera')} onClick={() => onModeChange('camera')}>
          <Camera className="w-3 h-3 inline mr-1" />Camera
        </button>
        <button className={modeBtnClass(mode === 'video')} onClick={() => onModeChange('video')}>
          <Film className="w-3 h-3 inline mr-1" />Video
        </button>
      </div>

      {/* Scenario selector (simulator only) */}
      {mode === 'simulator' && (
        <select
          value={scenario}
          onChange={e => onScenarioChange(e.target.value as ScenarioName)}
          className="bg-white/5 border border-border text-xs text-white rounded px-2 py-1.5"
        >
          {SCENARIOS.map(s => <option key={s.key} value={s.key}>{s.label}</option>)}
        </select>
      )}

      {/* Model selector */}
      <select
        value={modelVariant}
        onChange={e => onModelChange(e.target.value)}
        className="bg-white/5 border border-border text-xs text-white rounded px-2 py-1.5 max-w-[200px]"
      >
        {MODELS.map(m => <option key={m} value={m}>{m}</option>)}
      </select>

      {/* Precision */}
      <div className="flex gap-0.5 bg-white/5 rounded p-0.5">
        {(['FP32', 'FP16', 'INT8'] as const).map(p => (
          <button
            key={p}
            onClick={() => onPrecisionChange(p)}
            className={`px-2 py-1 text-[10px] font-bold rounded ${
              precision === p ? 'bg-drivable/20 text-drivable' : 'text-gray-400 hover:text-white'
            }`}
          >{p}</button>
        ))}
      </div>

      {/* Weather override */}
      <select
        value={weatherOverride ?? 'auto'}
        onChange={e => onWeatherChange(e.target.value === 'auto' ? null : e.target.value)}
        className="bg-white/5 border border-border text-xs text-white rounded px-2 py-1.5"
      >
        <option value="auto">Weather: Auto</option>
        <option value="clear_day">Clear Day</option>
        <option value="night">Night</option>
        <option value="rain">Rain</option>
        <option value="fog">Fog</option>
      </select>

      <div className="flex-1" />

      {/* Mute */}
      <button onClick={onMuteToggle} className="p-1.5 rounded hover:bg-white/10 text-gray-400 hover:text-white">
        {muted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
      </button>
    </div>
  );
};

export default TopBar;
