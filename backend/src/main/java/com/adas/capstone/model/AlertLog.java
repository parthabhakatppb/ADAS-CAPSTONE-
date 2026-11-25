package com.adas.capstone.model;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.Instant;

/**
 * MongoDB document for persisted alert/event log entries.
 */
@Document(collection = "alert_logs")
public class AlertLog {

    @Id
    private String id;
    private String sessionId;
    private Instant timestamp;
    private String level;    // ALERT | SIGN | SYS | INFO
    private String message;
    private int frameId;
    private String modelVariant;

    public AlertLog() {}

    public AlertLog(String sessionId, String level, String message, int frameId, String modelVariant) {
        this.sessionId = sessionId;
        this.timestamp = Instant.now();
        this.level = level;
        this.message = message;
        this.frameId = frameId;
        this.modelVariant = modelVariant;
    }

    // ── Getters & Setters ────────────────────────────────────────────────
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getSessionId() { return sessionId; }
    public void setSessionId(String v) { this.sessionId = v; }

    public Instant getTimestamp() { return timestamp; }
    public void setTimestamp(Instant v) { this.timestamp = v; }

    public String getLevel() { return level; }
    public void setLevel(String v) { this.level = v; }

    public String getMessage() { return message; }
    public void setMessage(String v) { this.message = v; }

    public int getFrameId() { return frameId; }
    public void setFrameId(int v) { this.frameId = v; }

    public String getModelVariant() { return modelVariant; }
    public void setModelVariant(String v) { this.modelVariant = v; }
}
