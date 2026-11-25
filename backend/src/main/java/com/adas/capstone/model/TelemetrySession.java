package com.adas.capstone.model;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/**
 * MongoDB document representing one telemetry session (from connect to disconnect).
 * Stores aggregate stats and the last N alerts for review.
 */
@Document(collection = "telemetry_sessions")
public class TelemetrySession {

    @Id
    private String id;
    private Instant startTime;
    private Instant endTime;
    private String modelVariant;
    private String precision;
    private int totalFrames;
    private double avgFps;
    private double avgLatencyMs;
    private int brakeWarningCount;
    private List<String> recentAlerts = new ArrayList<>();

    public TelemetrySession() {
        this.startTime = Instant.now();
    }

    // ── Getters & Setters ────────────────────────────────────────────────
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public Instant getStartTime() { return startTime; }
    public void setStartTime(Instant v) { this.startTime = v; }

    public Instant getEndTime() { return endTime; }
    public void setEndTime(Instant v) { this.endTime = v; }

    public String getModelVariant() { return modelVariant; }
    public void setModelVariant(String v) { this.modelVariant = v; }

    public String getPrecision() { return precision; }
    public void setPrecision(String v) { this.precision = v; }

    public int getTotalFrames() { return totalFrames; }
    public void setTotalFrames(int v) { this.totalFrames = v; }

    public double getAvgFps() { return avgFps; }
    public void setAvgFps(double v) { this.avgFps = v; }

    public double getAvgLatencyMs() { return avgLatencyMs; }
    public void setAvgLatencyMs(double v) { this.avgLatencyMs = v; }

    public int getBrakeWarningCount() { return brakeWarningCount; }
    public void setBrakeWarningCount(int v) { this.brakeWarningCount = v; }

    public List<String> getRecentAlerts() { return recentAlerts; }
    public void setRecentAlerts(List<String> v) { this.recentAlerts = v; }
}
