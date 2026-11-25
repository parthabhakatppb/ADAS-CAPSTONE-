package com.adas.capstone.controller;

import com.adas.capstone.model.AlertLog;
import com.adas.capstone.model.TelemetrySession;
import com.adas.capstone.service.TelemetryService;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * REST endpoints for querying persisted telemetry data from MongoDB.
 */
@RestController
@RequestMapping("/api/telemetry")
public class TelemetryController {

    private final TelemetryService telemetryService;

    public TelemetryController(TelemetryService telemetryService) {
        this.telemetryService = telemetryService;
    }

    @GetMapping("/sessions")
    public List<TelemetrySession> recentSessions() {
        return telemetryService.recentSessions();
    }

    @GetMapping("/sessions/{sessionId}/alerts")
    public List<AlertLog> sessionAlerts(@PathVariable String sessionId) {
        return telemetryService.alertsForSession(sessionId);
    }

    @GetMapping("/alerts/recent")
    public List<AlertLog> recentAlerts() {
        return telemetryService.recentAlerts();
    }
}
