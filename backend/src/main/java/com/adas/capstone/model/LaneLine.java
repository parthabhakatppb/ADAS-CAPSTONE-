package com.adas.capstone.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

/**
 * Mirrors the TypeScript LaneLine interface.
 */
public class LaneLine {
    private String side;         // left_ego | right_ego | left_adj | right_adj
    @JsonProperty("marking_type")
    private String markingType;  // white_dashed | white_solid | yellow_solid | double_yellow
    @JsonProperty("points_norm")
    private List<double[]> pointsNorm;

    public LaneLine() {}

    public LaneLine(String side, String markingType, List<double[]> pointsNorm) {
        this.side = side;
        this.markingType = markingType;
        this.pointsNorm = pointsNorm;
    }

    public String getSide() { return side; }
    public void setSide(String side) { this.side = side; }

    public String getMarkingType() { return markingType; }
    public void setMarkingType(String markingType) { this.markingType = markingType; }

    public List<double[]> getPointsNorm() { return pointsNorm; }
    public void setPointsNorm(List<double[]> pointsNorm) { this.pointsNorm = pointsNorm; }
}
