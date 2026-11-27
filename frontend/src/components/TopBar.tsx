import React from 'react';
import { InputMode } from '../types/adas';
import { ScenarioName } from '../engine/simulator';
import HudSelect, { HudSelectOption } from './HudSelect';
import {
  Activity, Camera, Film, Volume2, VolumeX,
  CloudRain, Sun, Moon, CloudFog, Cpu, Route, Sparkles,
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

const MODELS: HudSelectOption[] = [
  { value: 'Hybrid-S (CNN + 1-Layer Transformer)', label: 'Hybrid-S (CNN + 1-Layer Transformer)' },
  { value: 'Hybrid-S (CNN + 2-Layer Transformer)', label: 'Hybrid-S (CNN + 2-Layer Transformer)' },
  { value: 'YOLOv8-S (Pure CNN Baseline)', label: 'YOLOv8-S (Pure CNN Baseline)' },
  { value: 'RT-DETR (Pure Transformer)', label: 'RT-DETR (Pure Transformer)' },
];

const SCENARIOS: HudSelectOption[] = [
  { value: 'highway', label: 'Normal Highway' },
  { value: 'pedestrian_crossing', label: 'Pedestrian Crossing' },
  { value: 'lead_car_brake', label: 'Lead-Car Brake' },
  { value: 'school_zone', label: 'School Zone + Red Light' },
  { value: 'heavy_rain_night', label: 'Rain / Night Stress' },
];

const WEATHERS: HudSelectOption[] = [
  { value: 'auto', label: 'Weather: Auto', icon: <Sparkles className="w-3.5 h-3.5" /> },
  { value: 'clear_day', label: 'Clear Day', icon: <Sun className="w-3.5 h-3.5" /> },
  { value: 'night', label: 'Night', icon: <Moon className="w-3.5 h-3.5" /> },
  { value: 'rain', label: 'Rain', icon: <CloudRain className="w-3.5 h-3.5" /> },
  { value: 'fog', label: 'Fog', icon: <CloudFog className="w-3.5 h-3.5" /> },
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
        <span className="text-sm font-bold tracking-wide whitespace-nowrap">HYBRID CNN–TRANSFORMER ADAS</span>
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
        <HudSelect
          ariaLabel="Simulation Scenario"
          value={scenario}
          options={SCENARIOS}
          onChange={v => onScenarioChange(v as ScenarioName)}
          icon={<Route className="w-3.5 h-3.5" />}
          menuClassName="min-w-[195px]"
        />
      )}

      {/* Model selector */}
      <HudSelect
        ariaLabel="Model Variant"
        value={modelVariant}
        options={MODELS}
        onChange={onModelChange}
        icon={<Cpu className="w-3.5 h-3.5" />}
        className="max-w-[220px]"
        menuClassName="min-w-[235px]"
      />

      {/* Precision */}
      <div className="flex gap-0.5 bg-white/5 border border-border rounded-md p-0.5">
        {(['FP32', 'FP16', 'INT8'] as const).map(p => (
          <button
            key={p}
            onClick={() => onPrecisionChange(p)}
            className={`px-2 py-1 text-[10px] font-bold rounded transition-colors ${
              precision === p ? 'bg-drivable/20 text-drivable' : 'text-gray-400 hover:text-white'
            }`}
          >{p}</button>
        ))}
      </div>

      {/* Weather override */}
      <HudSelect
        ariaLabel="Weather Override"
        value={weatherOverride ?? 'auto'}
        options={WEATHERS}
        onChange={v => onWeatherChange(v === 'auto' ? null : v)}
        menuClassName="min-w-[160px]"
      />

      <div className="flex-1" />

      {/* Mute */}
      <button onClick={onMuteToggle} className="p-1.5 rounded hover:bg-white/10 text-gray-400 hover:text-white">
        {muted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
      </button>
    </div>
  );
};

export default TopBar;
