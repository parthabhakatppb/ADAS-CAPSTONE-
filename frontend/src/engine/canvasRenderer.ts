// ─── High-performance HTML5 Canvas renderer ────────────────────────────────
// All drawing happens outside React state via requestAnimationFrame.

import { FrameInferencePayload, TrackedObject } from '../types/adas';

// ─── Colour palette ─────────────────────────────────────────────────────────
const C = {
  safe:     '#10B981',
  caution:  '#F59E0B',
  critical: '#EF4444',
  drivable: 'rgba(6,182,212,0.18)',
  drivableStroke: 'rgba(6,182,212,0.4)',
  laneDash: '#ffffff',
  laneYellow: '#ffd700',
  trail:    'rgba(255,255,255,0.25)',
  bg:       '#080B12',
  hud:      'rgba(16,21,34,0.75)',
};

function riskColor(r: string): string {
  return r === 'critical' ? C.critical : r === 'caution' ? C.caution : C.safe;
}

// ─── Resize helper ──────────────────────────────────────────────────────────
export function resizeCanvas(canvas: HTMLCanvasElement) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width  = rect.width  * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

// ─── Coordinate helpers (normalised → pixel) ────────────────────────────────
function nx(v: number, w: number) { return v * w; }
function ny(v: number, h: number) { return v * h; }

// ─── Draw the full simulator road scene background ──────────────────────────
export function drawSimulatorBackground(ctx: CanvasRenderingContext2D, w: number, h: number, t: number, weather: string) {
  // Sky gradient
  const sky = ctx.createLinearGradient(0, 0, 0, h * 0.4);
  if (weather === 'night') {
    sky.addColorStop(0, '#0a0e1a');
    sky.addColorStop(1, '#141b2d');
  } else if (weather === 'rain' || weather === 'fog') {
    sky.addColorStop(0, '#3a3f4b');
    sky.addColorStop(1, '#6b7280');
  } else {
    sky.addColorStop(0, '#1e3a5f');
    sky.addColorStop(1, '#4a8cc7');
  }
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, w, h * 0.4);

  // Ground
  ctx.fillStyle = '#374151';
  ctx.fillRect(0, h * 0.4, w, h * 0.6);

  // Road surface
  ctx.fillStyle = '#1f2937';
  ctx.beginPath();
  ctx.moveTo(w * 0.12, h);
  ctx.lineTo(w * 0.36, h * 0.4);
  ctx.lineTo(w * 0.64, h * 0.4);
  ctx.lineTo(w * 0.88, h);
  ctx.closePath();
  ctx.fill();

  // Animated lane dashes
  const dashCount = 12;
  ctx.strokeStyle = '#d1d5db';
  ctx.lineWidth = 2;
  for (let i = 0; i < dashCount; i++) {
    const p = ((i / dashCount) + (t * 0.15) % 1) % 1;
    const perspective = 0.4 + p * 0.6;
    const yy = h * perspective;
    const cx = w * 0.5;
    const halfGap = w * (0.02 + p * 0.08);
    if (i % 2 === 0) {
      ctx.beginPath();
      ctx.moveTo(cx, yy);
      ctx.lineTo(cx, yy + h * 0.03);
      ctx.stroke();
    }
  }

  // Weather particles
  if (weather === 'rain') {
    ctx.strokeStyle = 'rgba(150,200,255,0.3)';
    ctx.lineWidth = 1;
    for (let i = 0; i < 80; i++) {
      const rx = (Math.sin(i * 73.1 + t * 400) * 0.5 + 0.5) * w;
      const ry = ((i * 17.3 + t * 600) % h);
      ctx.beginPath();
      ctx.moveTo(rx, ry);
      ctx.lineTo(rx - 2, ry + 12);
      ctx.stroke();
    }
  }
  if (weather === 'fog') {
    ctx.fillStyle = 'rgba(200,210,220,0.15)';
    ctx.fillRect(0, h * 0.3, w, h * 0.4);
  }
}

// ─── Draw drivable area polygon ─────────────────────────────────────────────
function drawDrivable(ctx: CanvasRenderingContext2D, w: number, h: number, pts: [number, number][]) {
  if (!pts || pts.length < 3) return;
  ctx.fillStyle = C.drivable;
  ctx.strokeStyle = C.drivableStroke;
  ctx.lineWidth = 1;
  ctx.beginPath();
  pts.forEach(([x, y], i) => {
    if (i === 0) ctx.moveTo(nx(x, w), ny(y, h));
    else ctx.lineTo(nx(x, w), ny(y, h));
  });
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

// ─── Draw lane lines ────────────────────────────────────────────────────────
function drawLanes(ctx: CanvasRenderingContext2D, w: number, h: number, lanes: FrameInferencePayload['lanes']) {
  lanes.forEach(lane => {
    const isYellow = lane.marking_type === 'yellow_solid' || lane.marking_type === 'double_yellow';
    ctx.strokeStyle = isYellow ? C.laneYellow : '#ffffff';
    ctx.lineWidth = lane.marking_type === 'double_yellow' ? 3 : 2;
    ctx.setLineDash(lane.marking_type === 'white_dashed' ? [8, 8] : []);
    ctx.beginPath();
    lane.points_norm.forEach(([x, y], i) => {
      if (i === 0) ctx.moveTo(nx(x, w), ny(y, h));
      else ctx.lineTo(nx(x, w), ny(y, h));
    });
    ctx.stroke();
  });
  ctx.setLineDash([]);
}

// ─── Draw corner-bracket bounding boxes ─────────────────────────────────────
function drawObjects(ctx: CanvasRenderingContext2D, w: number, h: number, objects: TrackedObject[]) {
  objects.forEach(obj => {
    const [x1n, y1n, x2n, y2n] = obj.bbox;
    const x1 = nx(x1n, w), y1 = ny(y1n, h);
    const x2 = nx(x2n, w), y2 = ny(y2n, h);
    const bw = x2 - x1, bh = y2 - y1;
    const col = riskColor(obj.risk_level);
    const corner = Math.min(bw, bh) * 0.25;

    ctx.strokeStyle = col;
    ctx.lineWidth = 2;

    // Corner brackets
    // top-left
    ctx.beginPath(); ctx.moveTo(x1, y1 + corner); ctx.lineTo(x1, y1); ctx.lineTo(x1 + corner, y1); ctx.stroke();
    // top-right
    ctx.beginPath(); ctx.moveTo(x2 - corner, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + corner); ctx.stroke();
    // bottom-left
    ctx.beginPath(); ctx.moveTo(x1, y2 - corner); ctx.lineTo(x1, y2); ctx.lineTo(x1 + corner, y2); ctx.stroke();
    // bottom-right
    ctx.beginPath(); ctx.moveTo(x2 - corner, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - corner); ctx.stroke();

    // Label above box
    const label = `ID:${String(obj.track_id).padStart(2,'0')} ${obj.cls.toUpperCase()} ${(obj.confidence * 100).toFixed(0)}% | ${obj.distance_m}m`;
    ctx.font = '600 11px Inter, system-ui, sans-serif';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = 'rgba(0,0,0,0.65)';
    ctx.fillRect(x1, y1 - 16, tw + 8, 15);
    ctx.fillStyle = col;
    ctx.fillText(label, x1 + 4, y1 - 4);

    // TTC under box if critical
    if (obj.ttc_s !== null && obj.ttc_s < 4) {
      const ttcLabel = `TTC: ${obj.ttc_s.toFixed(1)}s`;
      ctx.font = 'bold 11px Inter, system-ui, sans-serif';
      ctx.fillStyle = C.critical;
      ctx.fillText(ttcLabel, x1, y2 + 12);
    }

    // Motion trail (fading circles along history)
    if (obj.history_norm.length > 1) {
      obj.history_norm.forEach(([tx, ty], i) => {
        const alpha = (i + 1) / obj.history_norm.length * 0.4;
        ctx.fillStyle = `rgba(255,255,255,${alpha})`;
        ctx.beginPath();
        ctx.arc(nx(tx, w), ny(ty, h), 2, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // Trajectory arrows
    if (obj.trajectory_norm.length > 1) {
      ctx.strokeStyle = riskColor(obj.risk_level);
      ctx.globalAlpha = 0.5;
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      obj.trajectory_norm.forEach(([tx, ty], i) => {
        if (i === 0) ctx.moveTo(nx(tx, w), ny(ty, h));
        else ctx.lineTo(nx(tx, w), ny(ty, h));
      });
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }

    // Traffic light state badge
    if (obj.light_state && obj.light_state !== 'unknown') {
      const lCol = obj.light_state === 'red' ? '#EF4444' : obj.light_state === 'yellow' ? '#F59E0B' : '#10B981';
      ctx.fillStyle = lCol;
      ctx.beginPath();
      ctx.arc(x2 + 10, y1 + bh / 2, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.font = '9px Inter';
      ctx.fillText(obj.light_state.toUpperCase(), x2 + 20, y1 + bh / 2 + 3);
    }

    // Sign badge
    if (obj.sign_label) {
      ctx.fillStyle = 'rgba(6,182,212,0.8)';
      const stw = ctx.measureText(obj.sign_label).width;
      ctx.fillRect(x1, y2 + 2, stw + 10, 14);
      ctx.fillStyle = '#fff';
      ctx.font = '10px Inter';
      ctx.fillText(obj.sign_label, x1 + 5, y2 + 13);
    }
  });
}

// ─── Emergency brake banner ─────────────────────────────────────────────────
function drawBrakeWarning(ctx: CanvasRenderingContext2D, w: number, h: number, payload: FrameInferencePayload, t: number) {
  if (!payload.emergency_brake_warning) return;

  // Pulsing red vignette
  const pulse = 0.5 + Math.sin(t * 8) * 0.3;
  const grad = ctx.createRadialGradient(w / 2, h / 2, h * 0.3, w / 2, h / 2, h * 0.9);
  grad.addColorStop(0, 'rgba(239,68,68,0)');
  grad.addColorStop(1, `rgba(239,68,68,${pulse * 0.15})`);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  // Top banner
  ctx.fillStyle = `rgba(239,68,68,${0.85 + Math.sin(t * 6) * 0.1})`;
  ctx.fillRect(0, 0, w, 36);
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 15px Inter, system-ui, sans-serif';

  const critical = payload.objects.find(o => o.risk_level === 'critical');
  const msg = critical
    ? `⚠ BRAKE WARNING — COLLISION IMMINENT (TTC: ${critical.ttc_s?.toFixed(1) ?? '?'}s | ${critical.cls.toUpperCase()} ID:${String(critical.track_id).padStart(2,'0')})`
    : '⚠ BRAKE WARNING — COLLISION IMMINENT';
  ctx.fillText(msg, 12, 24);
}

// ─── HUD footer disclaimer ──────────────────────────────────────────────────
function drawFooter(ctx: CanvasRenderingContext2D, w: number, h: number) {
  ctx.fillStyle = 'rgba(16,21,34,0.6)';
  ctx.fillRect(0, h - 18, w, 18);
  ctx.fillStyle = '#6b7280';
  ctx.font = '9px Inter';
  ctx.fillText('PERCEPTION & WARNING ONLY — NO PHYSICAL ACTUATION', 8, h - 5);
}

// ─── Main render function ───────────────────────────────────────────────────
export function renderFrame(
  canvas: HTMLCanvasElement,
  payload: FrameInferencePayload | null,
  isSimulator: boolean,
  time: number,
  videoEl?: HTMLVideoElement | null,
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const rect = canvas.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;

  ctx.clearRect(0, 0, w, h);

  if (isSimulator && payload) {
    drawSimulatorBackground(ctx, w, h, time, payload.weather.condition);
  } else if (videoEl && videoEl.readyState >= 2) {
    ctx.drawImage(videoEl, 0, 0, w, h);
  }

  if (payload) {
    drawDrivable(ctx, w, h, payload.drivable_area_polygon_norm);
    drawLanes(ctx, w, h, payload.lanes);
    drawObjects(ctx, w, h, payload.objects);
    drawBrakeWarning(ctx, w, h, payload, time);
  }
  drawFooter(ctx, w, h);
}
