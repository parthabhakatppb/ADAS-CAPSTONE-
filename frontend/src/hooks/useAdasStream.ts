// ─── Unified ADAS stream hook ───────────────────────────────────────────────
// Manages Live Simulator, Webcam, and Video Upload modes with Spring Boot WS.

import { useRef, useState, useEffect, useCallback } from 'react';
import { FrameInferencePayload, InputMode, LogEntry } from '../types/adas';
import { AdasSimulator, ScenarioName } from '../engine/simulator';
import { renderFrame, resizeCanvas } from '../engine/canvasRenderer';

const WS_PROTOCOL = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_URL = typeof window !== 'undefined'
  ? `${WS_PROTOCOL}//${window.location.host}/ws/infer`
  : 'ws://127.0.0.1:8000/ws/infer';

export function useAdasStream() {
  const [mode, setMode] = useState<InputMode>('simulator');
  const [connected, setConnected] = useState(false);
  const [payload, setPayload] = useState<FrameInferencePayload | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [scenario, setScenario] = useState<ScenarioName>('highway');
  const [modelVariant, setModelVariant] = useState('Hybrid-S (CNN + 2-Layer Transformer)');
  const [precision, setPrecision] = useState<'FP32' | 'FP16' | 'INT8'>('FP32');
  const [weatherOverride, setWeatherOverride] = useState<string | null>(null);

  // Camera & Video state
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [videoFileName, setVideoFileName] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [videoProgress, setVideoProgress] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const simRef = useRef(new AdasSimulator());
  const rafRef = useRef(0);
  const wsRef = useRef<WebSocket | null>(null);
  const latestPayloadRef = useRef<FrameInferencePayload | null>(null);
  const timeRef = useRef(0);
  const lastHudUpdate = useRef(0);
  const lastFrameSent = useRef(0);
  const videoUrlRef = useRef<string | null>(null);

  const addSysLog = useCallback((msg: string) => {
    setLogs(prev => [...prev.slice(-200), { ts: Date.now(), level: 'SYS', msg }]);
  }, []);

  // Push alerts to log
  const pushAlerts = useCallback((p: FrameInferencePayload) => {
    if (!p.alerts || !p.alerts.length) return;
    const now = Date.now();
    const newEntries: LogEntry[] = p.alerts.map(msg => ({
      ts: now,
      level: msg.startsWith('[ALERT]') ? 'ALERT' : msg.startsWith('[SIGN]') ? 'SIGN' : 'INFO',
      msg,
    }));
    setLogs(prev => [...prev.slice(-200), ...newEntries]);
  }, []);

  // Helper: capture current video frame as JPEG Blob and send over WS
  const captureAndSend = useCallback((video: HTMLVideoElement) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (video.readyState < 2 || video.videoWidth === 0) return;

    if (!offscreenRef.current) {
      offscreenRef.current = document.createElement('canvas');
    }
    const off = offscreenRef.current;
    const targetW = 640;
    const targetH = Math.max(360, Math.round((video.videoHeight / video.videoWidth) * targetW));
    if (off.width !== targetW || off.height !== targetH) {
      off.width = targetW;
      off.height = targetH;
    }
    const octx = off.getContext('2d');
    if (!octx) return;
    octx.drawImage(video, 0, 0, targetW, targetH);
    off.toBlob(
      (blob) => {
        if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(blob);
        }
      },
      'image/jpeg',
      0.75,
    );
  }, []);

  // ─── Simulator loop ─────────────────────────────────────────────────────
  useEffect(() => {
    if (mode !== 'simulator') return;

    const sim = simRef.current;
    sim.setScenario(scenario);
    sim.setModel(modelVariant);
    sim.setPrecision(precision);
    if (weatherOverride) sim.setWeatherOverride(weatherOverride as any);
    else sim.setWeatherOverride(null);

    let lastTime = performance.now();

    const loop = (now: number) => {
      const dt = (now - lastTime) / 1000;
      lastTime = now;
      timeRef.current += dt;

      const frame = sim.step(dt);
      latestPayloadRef.current = frame;

      if (canvasRef.current) {
        renderFrame(canvasRef.current, frame, true, timeRef.current);
      }

      if (now - lastHudUpdate.current > 80) {
        lastHudUpdate.current = now;
        setPayload(frame);
        pushAlerts(frame);
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [mode, scenario, modelVariant, precision, weatherOverride, pushAlerts]);

  // ─── WebSocket connection (camera / video) with auto-reconnect ──────────
  useEffect(() => {
    if (mode === 'simulator') {
      wsRef.current?.close();
      wsRef.current = null;
      setConnected(false);
      return;
    }

    let cancelled = false;
    let reconnectTimer: number | undefined;
    latestPayloadRef.current = null;
    setPayload(null);

    const connect = () => {
      if (cancelled) return;
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelled) { ws.close(); return; }
        setConnected(true);
        addSysLog('[SYS] Connected to Spring Boot ADAS backend (ws://localhost:8000/ws/infer)');
        ws.send(JSON.stringify({
          type: 'config',
          model_variant: modelVariant,
          precision,
          conf_threshold: 0.25,
          weather_override: weatherOverride,
        }));
      };

      ws.onclose = () => {
        setConnected(false);
        if (!cancelled) {
          addSysLog('[SYS] Backend disconnected — retrying in 2s...');
          reconnectTimer = window.setTimeout(connect, 2000);
        }
      };

      ws.onerror = () => {
        // onclose will handle reconnect
      };

      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data.type === 'status') {
            addSysLog(`[SYS] Inference model ${data.model_ready ? 'READY' : 'warming up...'}`);
            return;
          }
          if (data.type === 'config_ack' || data.error) {
            if (data.error) addSysLog(`[SYS] Backend error: ${data.error}`);
            return;
          }
          const frame = data as FrameInferencePayload;
          latestPayloadRef.current = frame;
          setPayload(frame);
          pushAlerts(frame);
        } catch {
          /* ignore malformed frames */
        }
      };
    };

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [mode, pushAlerts, addSysLog]);

  // Sync config updates to open WS
  useEffect(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'config',
        model_variant: modelVariant,
        precision,
        conf_threshold: 0.25,
        weather_override: weatherOverride,
      }));
    }
  }, [modelVariant, precision, weatherOverride]);

  // ─── Camera mode: getUserMedia + render + stream frames ─────────────────
  useEffect(() => {
    if (mode !== 'camera') {
      setCameraError(null);
      return;
    }

    let active = true;
    let mediaStream: MediaStream | null = null;
    const video = videoRef.current;

    const startCamera = async () => {
      try {
        setCameraError(null);
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        if (!active) {
          mediaStream.getTracks().forEach(t => t.stop());
          return;
        }
        if (video) {
          video.removeAttribute('src');
          video.srcObject = mediaStream;
          await video.play();
          addSysLog('[SYS] Camera stream started — sending frames to Hybrid CNN-Transformer');
        }
      } catch (err: any) {
        const msg = err?.message || 'Camera access denied or unavailable';
        setCameraError(msg);
        addSysLog(`[SYS] Camera error: ${msg}`);
      }
    };

    startCamera();

    const loop = (now: number) => {
      if (!active) return;
      timeRef.current = now / 1000;
      if (canvasRef.current) {
        renderFrame(canvasRef.current, latestPayloadRef.current, false, timeRef.current, video);
      }
      if (video && now - lastFrameSent.current > 120) {
        lastFrameSent.current = now;
        captureAndSend(video);
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);

    return () => {
      active = false;
      cancelAnimationFrame(rafRef.current);
      if (mediaStream) {
        mediaStream.getTracks().forEach(t => t.stop());
      }
      if (video) {
        video.srcObject = null;
      }
    };
  }, [mode, captureAndSend, addSysLog]);

  // ─── Video upload mode: render + stream frames while playing ────────────
  useEffect(() => {
    if (mode !== 'video') {
      setIsPlaying(false);
      return;
    }

    let active = true;
    const video = videoRef.current;

    const loop = (now: number) => {
      if (!active) return;
      timeRef.current = now / 1000;
      if (canvasRef.current) {
        renderFrame(canvasRef.current, latestPayloadRef.current, false, timeRef.current, video);
      }
      if (video && !video.paused && !video.ended) {
        setVideoProgress(video.currentTime);
        if (now - lastFrameSent.current > 120) {
          lastFrameSent.current = now;
          captureAndSend(video);
        }
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);

    return () => {
      active = false;
      cancelAnimationFrame(rafRef.current);
      if (video) {
        video.pause();
      }
    };
  }, [mode, captureAndSend]);

  // Video upload helpers
  const loadVideoFile = useCallback((file: File) => {
    const video = videoRef.current;
    if (!video) return;
    if (videoUrlRef.current) {
      URL.revokeObjectURL(videoUrlRef.current);
    }
    const url = URL.createObjectURL(file);
    videoUrlRef.current = url;
    video.srcObject = null;
    video.src = url;
    video.playbackRate = playbackRate;
    setVideoFileName(file.name);
    video.onloadedmetadata = () => {
      setVideoDuration(video.duration || 0);
      setVideoProgress(0);
      video.play().then(() => setIsPlaying(true)).catch(() => {});
      addSysLog(`[SYS] Loaded video: ${file.name} (${video.videoWidth}x${video.videoHeight})`);
    };
    video.onended = () => setIsPlaying(false);
  }, [playbackRate, addSysLog]);

  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video || !video.src) return;
    if (video.paused) {
      video.play().then(() => setIsPlaying(true)).catch(() => {});
    } else {
      video.pause();
      setIsPlaying(false);
    }
  }, []);

  const seekTo = useCallback((seconds: number) => {
    const video = videoRef.current;
    if (!video || !video.src) return;
    video.currentTime = Math.max(0, Math.min(video.duration || 0, seconds));
    setVideoProgress(video.currentTime);
    setTimeout(() => captureAndSend(video), 60);
  }, [captureAndSend]);

  const skipBy = useCallback((delta: number) => {
    const video = videoRef.current;
    if (!video || !video.src) return;
    seekTo(video.currentTime + delta);
  }, [seekTo]);

  const changePlaybackRate = useCallback((rate: number) => {
    setPlaybackRate(rate);
    if (videoRef.current) {
      videoRef.current.playbackRate = rate;
    }
  }, []);

  // ─── Canvas resize observer ─────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const obs = new ResizeObserver(() => resizeCanvas(canvas));
    obs.observe(canvas);
    resizeCanvas(canvas);
    return () => obs.disconnect();
  }, []);

  return {
    mode, setMode,
    connected,
    payload,
    logs, setLogs,
    scenario, setScenario,
    modelVariant, setModelVariant,
    precision, setPrecision,
    weatherOverride, setWeatherOverride,
    canvasRef,
    videoRef,
    cameraError,
    videoFileName,
    isPlaying,
    videoProgress,
    videoDuration,
    playbackRate,
    loadVideoFile,
    togglePlay,
    seekTo,
    skipBy,
    changePlaybackRate,
  };
}
