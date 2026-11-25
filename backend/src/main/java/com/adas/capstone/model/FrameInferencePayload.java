package com.adas.capstone.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

/**
 * Full per-frame inference payload — mirrors the TypeScript FrameInferencePayload.
 * When persisted to MongoDB it is embedded inside a TelemetrySession document.
 */
public class FrameInferencePayload {

    @JsonProperty("timestamp_ms")
    private long timestampMs;

    @JsonProperty("frame_id")
    private int frameId;

    @JsonProperty("model_variant")
    private String modelVariant;

    private String precision;       // FP32 | FP16 | INT8
    private double fps;

    @JsonProperty("latency_ms")
    private LatencyBreakdown latencyMs;

    @JsonProperty("ego_speed_kmh")
    private double egoSpeedKmh;

    private WeatherInfo weather;

    @JsonProperty("drivable_area_polygon_norm")
    private List<double[]> drivableAreaPolygonNorm;

    private List<LaneLine> lanes;
    private List<TrackedObject> objects;

    @JsonProperty("emergency_brake_warning")
    private boolean emergencyBrakeWarning;

    private List<String> alerts;

    public FrameInferencePayload() {}

    // ── Getters & Setters ────────────────────────────────────────────────
    public long getTimestampMs() { return timestampMs; }
    public void setTimestampMs(long v) { this.timestampMs = v; }

    public int getFrameId() { return frameId; }
    public void setFrameId(int v) { this.frameId = v; }

    public String getModelVariant() { return modelVariant; }
    public void setModelVariant(String v) { this.modelVariant = v; }

    public String getPrecision() { return precision; }
    public void setPrecision(String v) { this.precision = v; }

    public double getFps() { return fps; }
    public void setFps(double v) { this.fps = v; }

    public LatencyBreakdown getLatencyMs() { return latencyMs; }
    public void setLatencyMs(LatencyBreakdown v) { this.latencyMs = v; }

    public double getEgoSpeedKmh() { return egoSpeedKmh; }
    public void setEgoSpeedKmh(double v) { this.egoSpeedKmh = v; }

    public WeatherInfo getWeather() { return weather; }
    public void setWeather(WeatherInfo v) { this.weather = v; }

    public List<double[]> getDrivableAreaPolygonNorm() { return drivableAreaPolygonNorm; }
    public void setDrivableAreaPolygonNorm(List<double[]> v) { this.drivableAreaPolygonNorm = v; }

    public List<LaneLine> getLanes() { return lanes; }
    public void setLanes(List<LaneLine> v) { this.lanes = v; }

    public List<TrackedObject> getObjects() { return objects; }
    public void setObjects(List<TrackedObject> v) { this.objects = v; }

    public boolean isEmergencyBrakeWarning() { return emergencyBrakeWarning; }
    public void setEmergencyBrakeWarning(boolean v) { this.emergencyBrakeWarning = v; }

    public List<String> getAlerts() { return alerts; }
    public void setAlerts(List<String> v) { this.alerts = v; }
}
