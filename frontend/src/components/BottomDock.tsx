import React, { useState, useRef, useEffect } from 'react';
import { LogEntry, FrameInferencePayload } from '../types/adas';
import HudSelect from './HudSelect';
import {
  Play, Pause, SkipBack, SkipForward, Download,
  Terminal, BarChart3, Film, Trash2, Gauge,
} from 'lucide-react';

interface BottomDockProps {
  logs: LogEntry[];
  onClearLogs: () => void;
  payload: FrameInferencePayload | null;
  mode: string;
  videoFileName: string | null;
  isPlaying: boolean;
  videoProgress: number;
  videoDuration: number;
  playbackRate: number;
  onTogglePlay: () => void;
  onSkipBack: () => void;
  onSkipForward: () => void;
  onSeek: (sec: number) => void;
  onPlaybackRateChange: (rate: number) => void;
}

type DockTab = 'video' | 'terminal' | 'ablation';

const TAB_CLASS = (active: boolean) =>
  `flex items-center gap-1 px-3 py-1.5 text-[11px] font-semibold rounded-t transition-colors ${
    active ? 'bg-card text-white border-t border-x border-border' : 'text-gray-500 hover:text-gray-300'
  }`;

// Mock ablation data
const ABLATION = [
  { model: 'YOLOv8-S', mAP: 42.1, fps: 142 },
  { model: 'Hybrid-S (1-Layer)', mAP: 44.8, fps: 98 },
  { model: 'Hybrid-S (2-Layer)', mAP: 46.3, fps: 78 },
  { model: 'RT-DETR', mAP: 47.9, fps: 32 },
];

const BottomDock: React.FC<BottomDockProps> = ({
  logs,
  onClearLogs,
  payload,
  mode,
  videoFileName,
  isPlaying,
  videoProgress,
  videoDuration,
  playbackRate,
  onTogglePlay,
  onSkipBack,
  onSkipForward,
  onSeek,
  onPlaybackRateChange,
}) => {
  const [tab, setTab] = useState<DockTab>('terminal');
  const [collapsed, setCollapsed] = useState(false);
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  if (collapsed) {
    return (
      <div
        className="h-6 bg-card border-t border-border flex items-center justify-center cursor-pointer text-gray-500 text-xs hover:text-white"
        onClick={() => setCollapsed(false)}
      >
        ▲ Expand Dock
      </div>
    );
  }

  return (
    <div className="h-48 bg-primary border-t border-border flex flex-col flex-shrink-0">
      {/* Tab bar */}
      <div className="flex items-center gap-1 px-2 pt-1 bg-primary">
        <button className={TAB_CLASS(tab === 'video')} onClick={() => setTab('video')}>
          <Film className="w-3 h-3" />Video Controls
        </button>
        <button className={TAB_CLASS(tab === 'terminal')} onClick={() => setTab('terminal')}>
          <Terminal className="w-3 h-3" />Terminal Log
        </button>
        <button className={TAB_CLASS(tab === 'ablation')} onClick={() => setTab('ablation')}>
          <BarChart3 className="w-3 h-3" />Ablation
        </button>
        <div className="flex-1" />
        <button onClick={() => setCollapsed(true)} className="text-gray-500 hover:text-white text-xs px-2">▼</button>
      </div>

      {/* Tab panels */}
      <div className="flex-1 overflow-hidden bg-card rounded-b">
        {/* Video controls */}
        {tab === 'video' && (
          <div className="flex items-center gap-4 p-4 h-full">
            <div className="flex items-center gap-2">
              <button
                onClick={onSkipBack}
                disabled={mode !== 'video' || !videoFileName}
                className="p-2 rounded bg-white/5 hover:bg-white/10 text-white disabled:opacity-40"
              >
                <SkipBack className="w-4 h-4" />
              </button>
              <button
                onClick={onTogglePlay}
                disabled={mode !== 'video' || !videoFileName}
                className="p-2 rounded bg-drivable/20 hover:bg-drivable/30 text-drivable disabled:opacity-40"
              >
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              </button>
              <button
                onClick={onSkipForward}
                disabled={mode !== 'video' || !videoFileName}
                className="p-2 rounded bg-white/5 hover:bg-white/10 text-white disabled:opacity-40"
              >
                <SkipForward className="w-4 h-4" />
              </button>
            </div>
            <div className="flex-1">
              <div className="flex justify-between text-[10px] text-gray-400 mb-1">
                <span>{videoFileName ? `File: ${videoFileName}` : 'Timeline — drag & drop a video file in Video mode'}</span>
                {videoDuration > 0 && (
                  <span className="font-mono tabular-nums">
                    {videoProgress.toFixed(1)}s / {videoDuration.toFixed(1)}s
                  </span>
                )}
              </div>
              <input
                type="range"
                min={0}
                max={videoDuration || 100}
                step={0.1}
                value={videoProgress}
                disabled={mode !== 'video' || !videoFileName}
                onChange={(e) => onSeek(parseFloat(e.target.value))}
                className="w-full accent-drivable cursor-pointer disabled:opacity-40"
              />
            </div>
            <HudSelect
              ariaLabel="Playback Rate"
              value={String(playbackRate)}
              onChange={(v) => onPlaybackRateChange(parseFloat(v))}
              placement="top"
              align="right"
              mono
              icon={<Gauge className="w-3.5 h-3.5" />}
              className="min-w-[78px]"
              menuClassName="min-w-[104px]"
              options={[
                { value: '0.25', label: '0.25x' },
                { value: '0.5', label: '0.5x' },
                { value: '1', label: '1x' },
                { value: '2', label: '2x' },
              ]}
            />
          </div>
        )}

        {/* Terminal log */}
        {tab === 'terminal' && (
          <div className="h-full flex flex-col">
            <div className="flex items-center gap-2 px-3 py-1 border-b border-border text-[10px]">
              <span className="text-gray-500">Live Telemetry Log ({logs.length} entries)</span>
              <div className="flex-1" />
              <button onClick={onClearLogs} className="text-gray-500 hover:text-white flex items-center gap-0.5">
                <Trash2 className="w-3 h-3" />Clear
              </button>
              <button className="text-gray-500 hover:text-white flex items-center gap-0.5">
                <Download className="w-3 h-3" />Export
              </button>
            </div>
            <div className="flex-1 overflow-y-auto px-3 py-1 font-mono text-[11px] leading-relaxed">
              {logs.slice(-100).map((entry, i) => {
                const col = entry.level === 'ALERT' ? 'text-critical' : entry.level === 'SIGN' ? 'text-drivable' : entry.level === 'SYS' ? 'text-caution' : 'text-gray-400';
                return (
                  <div key={i} className={col}>
                    <span className="text-gray-600">[{new Date(entry.ts).toLocaleTimeString()}]</span> {entry.msg}
                  </div>
                );
              })}
              {logs.length === 0 && <span className="text-gray-600">No events yet — start the simulator…</span>}
              <div ref={logEndRef} />
            </div>
          </div>
        )}

        {/* Ablation chart */}
        {tab === 'ablation' && (
          <div className="h-full p-3 overflow-auto">
            <table className="w-full text-[11px]">
              <thead>
                <tr className="text-gray-500 border-b border-border">
                  <th className="text-left py-1">Model</th>
                  <th className="text-right py-1">mAP@0.5</th>
                  <th className="text-right py-1">FPS</th>
                  <th className="text-left py-1 pl-3">mAP vs FPS</th>
                </tr>
              </thead>
              <tbody>
                {ABLATION.map(row => (
                  <tr key={row.model} className="border-b border-white/5">
                    <td className="py-1.5 text-white font-semibold">{row.model}</td>
                    <td className="py-1.5 text-right font-mono tabular-nums text-safe">{row.mAP.toFixed(1)}</td>
                    <td className="py-1.5 text-right font-mono tabular-nums text-drivable">{row.fps}</td>
                    <td className="py-1.5 pl-3">
                      <div className="flex items-center gap-1">
                        <div className="h-2 rounded bg-safe/40" style={{ width: `${row.mAP * 1.5}px` }} />
                        <div className="h-2 rounded bg-drivable/40" style={{ width: `${row.fps * 0.5}px` }} />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default BottomDock;
