package com.adas.capstone.model;

import com.fasterxml.jackson.annotation.JsonProperty;

public class LatencyBreakdown {
    private double preprocess;
    private double backbone;
    @JsonProperty("transformer_neck")
    private double transformerNeck;
    private double tracker;
    private double telemetry;
    private double total;

    public LatencyBreakdown() {}

    public double getPreprocess() { return preprocess; }
    public void setPreprocess(double v) { this.preprocess = v; }

    public double getBackbone() { return backbone; }
    public void setBackbone(double v) { this.backbone = v; }

    public double getTransformerNeck() { return transformerNeck; }
    public void setTransformerNeck(double v) { this.transformerNeck = v; }

    public double getTracker() { return tracker; }
    public void setTracker(double v) { this.tracker = v; }

    public double getTelemetry() { return telemetry; }
    public void setTelemetry(double v) { this.telemetry = v; }

    public double getTotal() { return total; }
    public void setTotal(double v) { this.total = v; }

    public void computeTotal() {
        this.total = Math.round((preprocess + backbone + transformerNeck + tracker + telemetry) * 10.0) / 10.0;
    }
}
