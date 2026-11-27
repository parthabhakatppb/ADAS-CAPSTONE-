import React from 'react';
import { FrameInferencePayload, LogEntry } from '../types/adas';
import BEVRadar from './BEVRadar';
import {
  Gauge, AlertTriangle, Activity, List,
} from 'lucide-react';

interface TelemetrySidebarProps {
  payload: FrameInferencePayload | null;
}

const riskBadge = (r: string) => {
  const c = r === 'critical' ? 'bg-critical text-white' : r === 'caution' ? 'bg-caution text-black' : 'bg-safe/20 text-safe';
  return <span className={`${c} px-1.5 py-0.5 rounded text-[9px] font-bold uppercase`}>{r}</span>;
};

const Card: React.FC<{ title: string; icon: React.ReactNode; children: React.ReactNode }> = ({ title, icon, children }) => (
  <div className="bg-card border border-border rounded-lg p-3">
    <div className="flex items-center gap-1.5 mb-2 text-gray-400 text-[11px] font-semibold uppercase tracking-wider">
      {icon}{title}
    </div>
    {children}
  </div>
);

const Stat: React.FC<{ label: string; value: string; color?: string }> = ({ label, value, color }) => (
  <div className="flex justify-between items-baseline">
    <span className="text-gray-500 text-[10px] uppercase">{label}</span>
    <span className={`font-mono tabular-nums text-sm font-bold ${color ?? 'text-white'}`}>{value}</span>
  </div>
);

const TelemetrySidebar: React.FC<TelemetrySidebarProps> = ({ payload }) => {
  if (!payload) {
    return (
      <div className="w-80 bg-primary border-l border-border p-3 flex items-center justify-center text-gray-500 text-sm">
        Waiting for data…
      </div>
    );
  }

  const nearest = payload.objects.length
    ? payload.objects.reduce((a, b) => a.distance_m < b.distance_m ? a : b)
    : null;
  const lowestTTC = payload.objects.filter(o => o.ttc_s !== null).sort((a, b) => (a.ttc_s ?? 999) - (b.ttc_s ?? 999))[0] ?? null;
  const ttcColor = lowestTTC
    ? (lowestTTC.ttc_s! < 2.5 ? 'text-critical' : lowestTTC.ttc_s! < 4 ? 'text-caution' : 'text-safe')
    : 'text-gray-500';

  // Pipeline latency bar
  const stages = [
    { name: 'Pre', ms: payload.latency_ms.preprocess, color: '#6366f1' },
    { name: 'CNN', ms: payload.latency_ms.backbone, color: '#06B6D4' },
    { name: 'Attn', ms: payload.latency_ms.transformer_neck, color: '#8b5cf6' },
    { name: 'Trk', ms: payload.latency_ms.tracker, color: '#F59E0B' },
    { name: 'Tel', ms: payload.latency_ms.telemetry, color: '#10B981' },
  ];
  const totalMs = payload.latency_ms.total;
  const budget = 33.3; // 30 FPS budget

  return (
    <div className="w-80 bg-primary border-l border-border flex flex-col gap-2 p-2 overflow-y-auto flex-shrink-0">
      {/* BEV Radar */}
      <Card title="Spatial Radar (BEV)" icon={<Activity className="w-3 h-3" />}>
        <div className="h-44">
          <BEVRadar payload={payload} />
        </div>
      </Card>

      {/* Ego & Threat Telemetry */}
      <Card title="Ego & Threat Telemetry" icon={<Gauge className="w-3 h-3" />}>
        <div className="space-y-1">
          <Stat label="Ego Speed" value={`${payload.ego_speed_kmh.toFixed(0)} km/h`} />
          <Stat label="Nearest Object" value={nearest ? `${nearest.distance_m}m (${nearest.cls})` : '—'} />
          <Stat label="Closing Speed" value={nearest ? `${nearest.rel_speed_kmh.toFixed(1)} km/h` : '—'} color={nearest && nearest.rel_speed_kmh > 10 ? 'text-caution' : undefined} />
          <Stat label="Lowest TTC" value={lowestTTC ? `${lowestTTC.ttc_s!.toFixed(1)}s` : '—'} color={ttcColor} />
          <Stat label="Weather" value={`${payload.weather.condition.replace('_', ' ').toUpperCase()} (${(payload.weather.confidence * 100).toFixed(0)}%)`} />
        </div>
      </Card>

      {/* Performance & Latency */}
      <Card title="Pipeline Latency" icon={<Activity className="w-3 h-3" />}>
        <div className="space-y-1.5">
          <Stat label="FPS" value={payload.fps.toFixed(0)} color={payload.fps > 25 ? 'text-safe' : payload.fps > 15 ? 'text-caution' : 'text-critical'} />
          <Stat label="Total" value={`${totalMs.toFixed(1)}ms`} color={totalMs < budget ? 'text-safe' : 'text-critical'} />
          {/* stacked bar */}
          <div className="flex h-3 rounded overflow-hidden bg-white/5">
            {stages.map(s => (
              <div key={s.name} style={{ width: `${(s.ms / budget) * 100}%`, backgroundColor: s.color }} title={`${s.name}: ${s.ms.toFixed(1)}ms`} />
            ))}
          </div>
          <div className="flex justify-between text-[8px] text-gray-500">
            {stages.map(s => <span key={s.name}>{s.name}</span>)}
          </div>
          <Stat label="Model" value={payload.model_variant.split('(')[0].trim()} />
          <Stat label="Precision" value={payload.precision} />
        </div>
      </Card>

      {/* Tracked Objects */}
      <Card title={`Tracked Objects (${payload.objects.length})`} icon={<List className="w-3 h-3" />}>
        <div className="max-h-36 overflow-y-auto space-y-1">
          {payload.objects
            .sort((a, b) => (a.ttc_s ?? 999) - (b.ttc_s ?? 999))
            .map(obj => (
              <div key={obj.track_id} className="flex items-center justify-between text-[10px] px-1 py-0.5 rounded bg-white/[0.03]">
                <span className="text-gray-400">#{String(obj.track_id).padStart(2, '0')}</span>
                <span className="text-white font-semibold w-14 truncate">{obj.cls}</span>
                <span className="font-mono tabular-nums text-gray-300 w-10 text-right">{obj.distance_m}m</span>
                <span className="font-mono tabular-nums text-gray-300 w-14 text-right">{obj.rel_speed_kmh > 0 ? '+' : ''}{obj.rel_speed_kmh.toFixed(0)}km/h</span>
                <span className="font-mono tabular-nums w-10 text-right" style={{ color: obj.ttc_s !== null && obj.ttc_s < 2.5 ? '#EF4444' : obj.ttc_s !== null && obj.ttc_s < 4 ? '#F59E0B' : '#10B981' }}>
                  {obj.ttc_s !== null ? `${obj.ttc_s.toFixed(1)}s` : '—'}
                </span>
                {riskBadge(obj.risk_level)}
              </div>
            ))}
          {payload.objects.length === 0 && <span className="text-gray-600 text-[10px]">No objects tracked</span>}
        </div>
      </Card>
    </div>
  );
};

export default TelemetrySidebar;
