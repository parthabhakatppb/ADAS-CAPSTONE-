import React, { useRef, useEffect } from 'react';
import { FrameInferencePayload } from '../types/adas';

interface BEVRadarProps {
  payload: FrameInferencePayload | null;
}

const riskColor = (r: string) => r === 'critical' ? '#EF4444' : r === 'caution' ? '#F59E0B' : '#10B981';

const BEVRadar: React.FC<BEVRadarProps> = ({ payload }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !payload) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const w = rect.width;
    const h = rect.height;
    const cx = w / 2;
    const egoY = h - 20;

    // Background
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, w, h);

    // Distance rings
    ctx.strokeStyle = '#1E293B';
    ctx.lineWidth = 1;
    const rings = [10, 25, 50];
    const maxDist = 60;
    rings.forEach(d => {
      const r = (d / maxDist) * (h - 40);
      ctx.beginPath();
      ctx.arc(cx, egoY, r, Math.PI, 2 * Math.PI);
      ctx.stroke();
      ctx.fillStyle = '#4b5563';
      ctx.font = '9px Inter';
      ctx.fillText(`${d}m`, cx + r + 2, egoY - 2);
    });

    // Ego lane corridor
    ctx.fillStyle = 'rgba(6,182,212,0.08)';
    ctx.beginPath();
    ctx.moveTo(cx - 18, egoY);
    ctx.lineTo(cx - 40, 10);
    ctx.lineTo(cx + 40, 10);
    ctx.lineTo(cx + 18, egoY);
    ctx.closePath();
    ctx.fill();

    // Ego vehicle icon
    ctx.fillStyle = '#06B6D4';
    ctx.fillRect(cx - 6, egoY - 10, 12, 18);
    ctx.fillStyle = '#fff';
    ctx.font = '8px Inter';
    ctx.fillText('EGO', cx - 8, egoY + 20);

    // Objects
    payload.objects.forEach(obj => {
      // Map bbox center X → BEV horizontal, distance → BEV vertical
      const bboxCx = (obj.bbox[0] + obj.bbox[2]) / 2;
      const ox = cx + (bboxCx - 0.5) * w * 1.5;
      const oy = egoY - (Math.min(obj.distance_m, maxDist) / maxDist) * (h - 40);
      const col = riskColor(obj.risk_level);

      // Velocity vector
      if (obj.rel_speed_kmh > 0) {
        ctx.strokeStyle = col;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(ox, oy);
        ctx.lineTo(ox, oy + Math.min(obj.rel_speed_kmh * 0.3, 20));
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Dot
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(ox, oy, 5, 0, Math.PI * 2);
      ctx.fill();

      // Label
      ctx.fillStyle = '#d1d5db';
      ctx.font = '8px Inter';
      ctx.fillText(`${obj.cls.slice(0, 3).toUpperCase()} ${obj.distance_m}m`, ox + 7, oy + 3);
    });
  }, [payload]);

  return <canvas ref={canvasRef} className="w-full h-full rounded" />;
};

export default BEVRadar;
