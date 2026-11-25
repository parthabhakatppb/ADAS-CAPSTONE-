package com.adas.capstone.service;

import com.adas.capstone.model.AlertLog;
import com.adas.capstone.model.FrameInferencePayload;
import com.adas.capstone.model.TelemetrySession;
import com.adas.capstone.repository.AlertLogRepository;
import com.adas.capstone.repository.TelemetrySessionRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.List;

/**
 * Persists telemetry sessions and alert logs to MongoDB.
 */
@Service
public class TelemetryService {

    private static final Logger log = LoggerFactory.getLogger(TelemetryService.class);

    private final TelemetrySessionRepository sessionRepo;
    private final AlertLogRepository alertRepo;

    public TelemetryService(TelemetrySessionRepository sessionRepo, AlertLogRepository alertRepo) {
        this.sessionRepo = sessionRepo;
        this.alertRepo = alertRepo;
    }

    public TelemetrySession createSession(String modelVariant, String precision) {
        TelemetrySession session = new TelemetrySession();
        session.setModelVariant(modelVariant);
        session.setPrecision(precision);
        try {
            session = sessionRepo.save(session);
            log.info("Created telemetry session: {}", session.getId());
        } catch (Exception e) {
            session.setId("local-" + java.util.UUID.randomUUID());
            log.warn("MongoDB unavailable ({}), using in-memory session {}", e.getMessage(), session.getId());
        }
        return session;
    }

    public void endSession(String sessionId, int totalFrames, double avgFps, double avgLatency, int brakeCount) {
        if (sessionId == null || sessionId.startsWith("local-")) return;
        try {
            sessionRepo.findById(sessionId).ifPresent(s -> {
                s.setEndTime(Instant.now());
                s.setTotalFrames(totalFrames);
                s.setAvgFps(avgFps);
                s.setAvgLatencyMs(avgLatency);
                s.setBrakeWarningCount(brakeCount);
                sessionRepo.save(s);
                log.info("Ended telemetry session: {} ({} frames)", sessionId, totalFrames);
            });
        } catch (Exception e) {
            log.warn("Failed to end MongoDB session {}: {}", sessionId, e.getMessage());
        }
    }

    public void logAlerts(String sessionId, FrameInferencePayload payload) {
        if (sessionId == null || sessionId.startsWith("local-")) return;
        if (payload.getAlerts() == null || payload.getAlerts().isEmpty()) return;
        try {
            for (String msg : payload.getAlerts()) {
                String level = msg.startsWith("[ALERT]") ? "ALERT"
                             : msg.startsWith("[SIGN]") ? "SIGN"
                             : msg.startsWith("[SYS]") ? "SYS" : "INFO";
                AlertLog entry = new AlertLog(sessionId, level, msg, payload.getFrameId(), payload.getModelVariant());
                alertRepo.save(entry);
            }
        } catch (Exception e) {
            log.debug("Skipped MongoDB alert log: {}", e.getMessage());
        }
    }

    public List<TelemetrySession> recentSessions() {
        try {
            return sessionRepo.findTop20ByOrderByStartTimeDesc();
        } catch (Exception e) {
            return List.of();
        }
    }

    public List<AlertLog> recentAlerts() {
        try {
            return alertRepo.findTop100ByOrderByTimestampDesc();
        } catch (Exception e) {
            return List.of();
        }
    }

    public List<AlertLog> alertsForSession(String sessionId) {
        try {
            return alertRepo.findBySessionIdOrderByTimestampDesc(sessionId);
        } catch (Exception e) {
            return List.of();
        }
    }
}
