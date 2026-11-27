import React, { useRef, useEffect, useState } from 'react';
import { FrameInferencePayload } from '../types/adas';
import { resizeCanvas } from '../engine/canvasRenderer';
import { Upload, CameraOff } from 'lucide-react';

interface ViewportHUDProps {
  canvasRef: React.RefObject<HTMLCanvasElement>;
  videoRef: React.RefObject<HTMLVideoElement>;
  payload: FrameInferencePayload | null;
  mode: string;
  cameraError: string | null;
  videoFileName: string | null;
  onVideoFileSelect: (file: File) => void;
}

const ViewportHUD: React.FC<ViewportHUDProps> = ({
  canvasRef,
  videoRef,
  payload,
  mode,
  cameraError,
  videoFileName,
  onVideoFileSelect,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  // Resize canvas when container resizes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const obs = new ResizeObserver(() => resizeCanvas(canvas));
    obs.observe(canvas);
    return () => obs.disconnect();
  }, [canvasRef]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (mode !== 'video') return;
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('video/')) {
      onVideoFileSelect(file);
    }
  };

  return (
    <div
      ref={containerRef}
      className="relative flex-1 bg-black overflow-hidden"
      onDragOver={(e) => { if (mode === 'video') { e.preventDefault(); setDragging(true); } }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      {/* Offscreen video element (kept 1px visible so browser decodes frames) */}
      <video ref={videoRef} playsInline muted className="fixed w-px h-px opacity-0 pointer-events-none" />

      {/* Hidden file input for Video Upload */}
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onVideoFileSelect(file);
        }}
      />

      {/* Canvas fills the viewport area */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
      />

      {/* Video Upload prompt when in Video mode with no file loaded */}
      {mode === 'video' && (!videoFileName || dragging) && (
        <div
          onClick={() => fileInputRef.current?.click()}
          className={`absolute inset-0 flex flex-col items-center justify-center cursor-pointer transition-colors ${
            dragging ? 'bg-drivable/20 border-2 border-dashed border-drivable' : 'bg-black/60 hover:bg-black/50'
          }`}
        >
          <Upload className="w-10 h-10 text-drivable mb-3" />
          <p className="text-sm font-semibold text-white">Drop a dashcam video here or click to browse</p>
          <p className="text-xs text-gray-400 mt-1">Supports .mp4, .webm, .mov — streamed live to Hybrid CNN-Transformer</p>
        </div>
      )}

      {/* Camera error overlay */}
      {mode === 'camera' && cameraError && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 px-6 text-center">
          <CameraOff className="w-10 h-10 text-critical mb-3" />
          <p className="text-sm font-semibold text-white">Camera Unavailable</p>
          <p className="text-xs text-gray-400 mt-1 max-w-md">{cameraError}</p>
        </div>
      )}

      {/* Top-right: traffic light state widget */}
      {payload && (() => {
        const light = payload.objects.find(o => o.cls === 'traffic_light' && o.light_state);
        if (!light) return null;
        const col = light.light_state === 'red' ? '#EF4444'
          : light.light_state === 'yellow' ? '#F59E0B' : '#10B981';
        return (
          <div className="absolute top-12 right-3 bg-black/60 rounded-lg px-3 py-2 flex items-center gap-2">
            <div className="w-4 h-4 rounded-full" style={{ backgroundColor: col, boxShadow: `0 0 8px ${col}` }} />
            <span className="text-xs font-bold text-white">{light.light_state?.toUpperCase()}</span>
          </div>
        );
      })()}

      {/* Bottom: active traffic signs strip */}
      {payload && (() => {
        const signs = payload.objects.filter(o => o.sign_label);
        if (!signs.length) return null;
        return (
          <div className="absolute bottom-6 left-3 flex gap-2">
            {signs.map(s => (
              <div key={s.track_id} className="bg-drivable/20 border border-drivable/40 rounded px-2 py-1 text-[10px] font-bold text-drivable">
                {s.sign_label} ({((s.sign_confidence ?? 0) * 100).toFixed(0)}%)
              </div>
            ))}
          </div>
        );
      })()}

      {/* FPS / latency overlay */}
      {payload && (
        <div className="absolute top-2 left-3 bg-black/50 rounded px-2 py-1 font-mono text-[11px] tabular-nums text-safe">
          {payload.fps.toFixed(0)} FPS · {payload.latency_ms.total.toFixed(1)}ms
        </div>
      )}
    </div>
  );
};

export default ViewportHUD;
