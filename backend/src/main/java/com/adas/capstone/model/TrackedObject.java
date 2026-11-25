package com.adas.capstone.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

/**
 * Mirrors the TypeScript TrackedObject interface.
 * Embedded inside FrameInferencePayload — not a standalone MongoDB document.
 */
public class TrackedObject {
    @JsonProperty("track_id")
    private int trackId;
    private String cls;           // car, truck, bus, pedestrian, rider, bicycle, motorcycle, traffic_light, traffic_sign
    private double confidence;
    private double[] bbox;        // [x1, y1, x2, y2] normalised
    @JsonProperty("distance_m")
    private double distanceM;
    @JsonProperty("abs_speed_kmh")
    private double absSpeedKmh;
    @JsonProperty("rel_speed_kmh")
    private double relSpeedKmh;
    @JsonProperty("ttc_s")
    private Double ttcS;          // null if not closing
    @JsonProperty("risk_level")
    private String riskLevel;     // safe | caution | critical
    @JsonProperty("trajectory_norm")
    private List<double[]> trajectoryNorm;
    @JsonProperty("history_norm")
    private List<double[]> historyNorm;
    @JsonProperty("sign_label")
    private String signLabel;
    @JsonProperty("sign_confidence")
    private Double signConfidence;
    @JsonProperty("light_state")
    private String lightState;    // red | yellow | green | unknown

    public TrackedObject() {}

    // ── Getters & Setters ────────────────────────────────────────────────
    public int getTrackId() { return trackId; }
    public void setTrackId(int trackId) { this.trackId = trackId; }

    public String getCls() { return cls; }
    public void setCls(String cls) { this.cls = cls; }

    public double getConfidence() { return confidence; }
    public void setConfidence(double confidence) { this.confidence = confidence; }

    public double[] getBbox() { return bbox; }
    public void setBbox(double[] bbox) { this.bbox = bbox; }

    public double getDistanceM() { return distanceM; }
    public void setDistanceM(double distanceM) { this.distanceM = distanceM; }

    public double getAbsSpeedKmh() { return absSpeedKmh; }
    public void setAbsSpeedKmh(double absSpeedKmh) { this.absSpeedKmh = absSpeedKmh; }

    public double getRelSpeedKmh() { return relSpeedKmh; }
    public void setRelSpeedKmh(double relSpeedKmh) { this.relSpeedKmh = relSpeedKmh; }

    public Double getTtcS() { return ttcS; }
    public void setTtcS(Double ttcS) { this.ttcS = ttcS; }

    public String getRiskLevel() { return riskLevel; }
    public void setRiskLevel(String riskLevel) { this.riskLevel = riskLevel; }

    public List<double[]> getTrajectoryNorm() { return trajectoryNorm; }
    public void setTrajectoryNorm(List<double[]> trajectoryNorm) { this.trajectoryNorm = trajectoryNorm; }

    public List<double[]> getHistoryNorm() { return historyNorm; }
    public void setHistoryNorm(List<double[]> historyNorm) { this.historyNorm = historyNorm; }

    public String getSignLabel() { return signLabel; }
    public void setSignLabel(String signLabel) { this.signLabel = signLabel; }

    public Double getSignConfidence() { return signConfidence; }
    public void setSignConfidence(Double signConfidence) { this.signConfidence = signConfidence; }

    public String getLightState() { return lightState; }
    public void setLightState(String lightState) { this.lightState = lightState; }
}
