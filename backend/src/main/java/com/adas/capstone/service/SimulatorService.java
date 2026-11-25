package com.adas.capstone.service;

import com.adas.capstone.model.*;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Server-side ADAS simulator — generates realistic FrameInferencePayload
 * objects so the WebSocket pipeline can be tested end-to-end without a
 * GPU or the PyTorch model loaded.
 */
@Service
public class SimulatorService {

    private String modelVariant = "Hybrid-S (CNN + 2-Layer ViT)";
    private String precision = "FP32";
    private double confThreshold = 0.35;
    private final AtomicInteger frameCounter = new AtomicInteger(0);
    private final long startTime = System.currentTimeMillis();
    private final Random rng = new Random();

    public void setModelVariant(String v) { this.modelVariant = v; }
    public void setPrecision(String v) { this.precision = v; }
    public void setConfThreshold(double v) { this.confThreshold = v; }

    private double clamp(double v, double lo, double hi) { return Math.max(lo, Math.min(hi, v)); }
    private double jitter() { return 0.85 + rng.nextDouble() * 0.30; }

    public FrameInferencePayload generateFrame() {
        int frame = frameCounter.incrementAndGet();
        double t = (System.currentTimeMillis() - startTime) / 1000.0;

        double egoSpeed = 65 + Math.sin(t * 0.3) * 8;

        // ── Objects ──────────────────────────────────────────────────────
        List<TrackedObject> objects = new ArrayList<>();

        // Lead car
        double dist = clamp(25 + Math.sin(t * 0.5) * 10, 5, 50);
        double relSpeed = 12 + Math.sin(t * 0.4) * 5;
        Double ttc = relSpeed > 0 ? dist / (relSpeed / 3.6) : null;
        String risk = (ttc != null && ttc < 2.5) ? "critical" : (ttc != null && ttc < 4) ? "caution" : "safe";

        TrackedObject car = new TrackedObject();
        car.setTrackId(1);
        car.setCls("car");
        car.setConfidence(Math.round((0.82 + rng.nextDouble() * 0.15) * 100.0) / 100.0);
        car.setBbox(new double[]{0.42, 0.44, 0.58, 0.56});
        car.setDistanceM(Math.round(dist * 10.0) / 10.0);
        car.setAbsSpeedKmh(Math.round((egoSpeed - relSpeed) * 10.0) / 10.0);
        car.setRelSpeedKmh(Math.round(relSpeed * 10.0) / 10.0);
        car.setTtcS(ttc != null ? Math.round(ttc * 100.0) / 100.0 : null);
        car.setRiskLevel(risk);
        car.setTrajectoryNorm(List.of(new double[]{0.50, 0.50}, new double[]{0.50, 0.48}, new double[]{0.50, 0.46}));
        car.setHistoryNorm(List.of(new double[]{0.50, 0.54}, new double[]{0.50, 0.52}, new double[]{0.50, 0.50}));
        objects.add(car);

        // Pedestrian (periodic)
        if (Math.sin(t * 0.2) > 0.3) {
            double px = 0.25 + (t % 6) * 0.07;
            double pDist = clamp(14 - (t % 6) * 1.5, 3, 20);
            String pRisk = pDist < 8 ? "critical" : pDist < 12 ? "caution" : "safe";

            TrackedObject ped = new TrackedObject();
            ped.setTrackId(7);
            ped.setCls("pedestrian");
            ped.setConfidence(Math.round((0.75 + rng.nextDouble() * 0.20) * 100.0) / 100.0);
            ped.setBbox(new double[]{
                    Math.round(clamp(px - 0.015, 0, 1) * 1000.0) / 1000.0,
                    0.55,
                    Math.round(clamp(px + 0.015, 0, 1) * 1000.0) / 1000.0,
                    0.68
            });
            ped.setDistanceM(Math.round(pDist * 10.0) / 10.0);
            ped.setAbsSpeedKmh(5.0);
            ped.setRelSpeedKmh(Math.round((egoSpeed - 5) * 10.0) / 10.0);
            Double pTtc = egoSpeed > 5 ? pDist / ((egoSpeed - 5) / 3.6) : null;
            ped.setTtcS(pTtc != null ? Math.round(pTtc * 100.0) / 100.0 : null);
            ped.setRiskLevel(pRisk);
            ped.setTrajectoryNorm(List.of(new double[]{px, 0.62}, new double[]{px + 0.03, 0.62}));
            ped.setHistoryNorm(List.of(new double[]{px - 0.03, 0.62}, new double[]{px, 0.62}));
            objects.add(ped);
        }

        // Truck in adjacent lane (always visible)
        TrackedObject truck = new TrackedObject();
        truck.setTrackId(3);
        truck.setCls("truck");
        truck.setConfidence(0.91);
        truck.setBbox(new double[]{0.68, 0.42, 0.82, 0.54});
        truck.setDistanceM(38.0);
        truck.setAbsSpeedKmh(72.0);
        truck.setRelSpeedKmh(Math.round((egoSpeed - 72) * 10.0) / 10.0);
        truck.setTtcS(null);
        truck.setRiskLevel("safe");
        truck.setTrajectoryNorm(List.of(new double[]{0.75, 0.48}, new double[]{0.75, 0.46}));
        truck.setHistoryNorm(List.of(new double[]{0.75, 0.52}, new double[]{0.75, 0.50}));
        objects.add(truck);

        // ── Brake warning ────────────────────────────────────────────────
        boolean hasBrake = objects.stream().anyMatch(o ->
                (o.getTtcS() != null && o.getTtcS() < 2.5) ||
                (("pedestrian".equals(o.getCls()) || "bicycle".equals(o.getCls())) && o.getDistanceM() < 8)
        );

        // ── Alerts ───────────────────────────────────────────────────────
        List<String> alerts = new ArrayList<>();
        for (TrackedObject o : objects) {
            if ("critical".equals(o.getRiskLevel())) {
                alerts.add(String.format("[ALERT] CRITICAL: %s ID:%02d | Dist: %.1fm | TTC: %ss",
                        o.getCls().toUpperCase(), o.getTrackId(), o.getDistanceM(),
                        o.getTtcS() != null ? String.format("%.1f", o.getTtcS()) : "N/A"));
            }
        }

        // ── Latency ──────────────────────────────────────────────────────
        double latBase = modelVariant.contains("YOLOv8") ? 8 : modelVariant.contains("RT-DETR") ? 22 : 12;
        double precMul = "INT8".equals(precision) ? 0.5 : "FP16".equals(precision) ? 0.7 : 1.0;

        LatencyBreakdown lat = new LatencyBreakdown();
        lat.setPreprocess(Math.round(1.2 * jitter() * 10.0) / 10.0);
        lat.setBackbone(Math.round(latBase * 0.45 * precMul * jitter() * 10.0) / 10.0);
        lat.setTransformerNeck(Math.round(latBase * 0.30 * precMul * jitter() * 10.0) / 10.0);
        lat.setTracker(Math.round(1.8 * jitter() * 10.0) / 10.0);
        lat.setTelemetry(Math.round(0.4 * jitter() * 10.0) / 10.0);
        lat.computeTotal();

        // ── Lanes ────────────────────────────────────────────────────────
        List<LaneLine> lanes = List.of(
            new LaneLine("left_ego",  "white_solid",  List.of(new double[]{0.30,1}, new double[]{0.42,0.55}, new double[]{0.47,0.35})),
            new LaneLine("right_ego", "white_dashed", List.of(new double[]{0.70,1}, new double[]{0.58,0.55}, new double[]{0.53,0.35})),
            new LaneLine("left_adj",  "yellow_solid", List.of(new double[]{0.12,1}, new double[]{0.36,0.55}, new double[]{0.44,0.35})),
            new LaneLine("right_adj", "white_solid",  List.of(new double[]{0.88,1}, new double[]{0.64,0.55}, new double[]{0.56,0.35}))
        );

        // ── Drivable area ────────────────────────────────────────────────
        List<double[]> drivable = List.of(
            new double[]{0.12,1}, new double[]{0.36,0.55}, new double[]{0.44,0.35},
            new double[]{0.56,0.35}, new double[]{0.64,0.55}, new double[]{0.88,1}
        );

        // ── Assemble payload ─────────────────────────────────────────────
        FrameInferencePayload payload = new FrameInferencePayload();
        payload.setTimestampMs(System.currentTimeMillis());
        payload.setFrameId(frame);
        payload.setModelVariant(modelVariant);
        payload.setPrecision(precision);
        payload.setFps(Math.round(1000.0 / Math.max(lat.getTotal(), 1) * 10.0) / 10.0);
        payload.setLatencyMs(lat);
        payload.setEgoSpeedKmh(Math.round(egoSpeed * 10.0) / 10.0);
        payload.setWeather(new WeatherInfo("clear_day", 0.94));
        payload.setDrivableAreaPolygonNorm(drivable);
        payload.setLanes(lanes);
        payload.setObjects(objects);
        payload.setEmergencyBrakeWarning(hasBrake);
        payload.setAlerts(alerts);

        return payload;
    }
}
