package com.adas.capstone.model;

public class WeatherInfo {
    private String condition;   // clear_day | night | rain | fog
    private double confidence;

    public WeatherInfo() {}
    public WeatherInfo(String condition, double confidence) {
        this.condition = condition;
        this.confidence = confidence;
    }

    public String getCondition() { return condition; }
    public void setCondition(String condition) { this.condition = condition; }

    public double getConfidence() { return confidence; }
    public void setConfidence(double confidence) { this.confidence = confidence; }
}
