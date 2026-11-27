package com.adas.capstone.websocket;

import com.adas.capstone.model.FrameInferencePayload;
import com.adas.capstone.model.TelemetrySession;
import com.adas.capstone.service.ModelInferenceService;
import com.adas.capstone.service.SimulatorService;
import com.adas.capstone.service.TelemetryService;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.*;
import org.springframework.web.socket.handler.AbstractWebSocketHandler;

import java.util.Map;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * WebSocket handler at /ws/infer.
 *
 * Protocol:
 *   Client -> Server (text):   { "type": "config", "model_variant": "...", "precision": "...", "conf_threshold": 0.25, "weather_override": null }
 *   Client -> Server (binary): JPEG frame bytes from Camera or Video upload
 *   Server -> Client (text):   FrameInferencePayload JSON
 */
@Component
public class InferenceWebSocketHandler extends AbstractWebSocketHandler {

    private static final Logger log = LoggerFactory.getLogger(InferenceWebSocketHandler.class);

    private final SimulatorService simulator;
    private final ModelInferenceService inferenceService;
    private final TelemetryService telemetry;
    private final ObjectMapper mapper;

    private final ExecutorService inferencePool = Executors.newFixedThreadPool(2);

    private final Map<String, String> sessionIds = new ConcurrentHashMap<>();
    private final Map<String, Integer> frameCounts = new ConcurrentHashMap<>();
    private final Map<String, Double> fpsAccum = new ConcurrentHashMap<>();
    private final Map<String, Double> latAccum = new ConcurrentHashMap<>();
    private final Map<String, Integer> brakeCounts = new ConcurrentHashMap<>();
    private final Map<String, AtomicBoolean> inFlight = new ConcurrentHashMap<>();

    private final Map<String, String> modelVariants = new ConcurrentHashMap<>();
    private final Map<String, String> precisions = new ConcurrentHashMap<>();
    private final Map<String, Double> confThresholds = new ConcurrentHashMap<>();
    private final Map<String, String> weatherOverrides = new ConcurrentHashMap<>();

    public InferenceWebSocketHandler(
            SimulatorService simulator,
            ModelInferenceService inferenceService,
            TelemetryService telemetry,
            ObjectMapper mapper) {
        this.simulator = simulator;
        this.inferenceService = inferenceService;
        this.telemetry = telemetry;
        this.mapper = mapper;
    }

    @Override
    public void afterConnectionEstablished(WebSocketSession session) {
        log.info("WebSocket connected: {}", session.getId());

        TelemetrySession ts = telemetry.createSession("Hybrid-S (CNN + 2-Layer Transformer)", "FP32");
        sessionIds.put(session.getId(), ts.getId());
        frameCounts.put(session.getId(), 0);
        fpsAccum.put(session.getId(), 0.0);
        latAccum.put(session.getId(), 0.0);
        brakeCounts.put(session.getId(), 0);
        inFlight.put(session.getId(), new AtomicBoolean(false));
        modelVariants.put(session.getId(), "Hybrid-S (CNN + 2-Layer Transformer)");
        precisions.put(session.getId(), "FP32");
        confThresholds.put(session.getId(), 0.25);

        try {
            String statusMsg = String.format(
                    "{\"type\":\"status\",\"model_ready\":%s}",
                    inferenceService.isReady());
            session.sendMessage(new TextMessage(statusMsg));
        } catch (Exception e) {
            log.warn("Could not send initial status to {}: {}", session.getId(), e.getMessage());
        }
    }

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) {
        try {
            JsonNode node = mapper.readTree(message.getPayload());
            if ("config".equals(node.path("type").asText())) {
                if (node.has("model_variant") && !node.get("model_variant").isNull()) {
                    String mv = node.get("model_variant").asText();
                    modelVariants.put(session.getId(), mv);
                    simulator.setModelVariant(mv);
                }
                if (node.has("precision") && !node.get("precision").isNull()) {
                    String p = node.get("precision").asText();
                    precisions.put(session.getId(), p);
                    simulator.setPrecision(p);
                }
                if (node.has("conf_threshold") && !node.get("conf_threshold").isNull()) {
                    double ct = node.get("conf_threshold").asDouble();
                    confThresholds.put(session.getId(), ct);
                    simulator.setConfThreshold(ct);
                }
                if (node.has("weather_override")) {
                    if (node.get("weather_override").isNull()) {
                        weatherOverrides.remove(session.getId());
                    } else {
                        weatherOverrides.put(session.getId(), node.get("weather_override").asText());
                    }
                }
                session.sendMessage(new TextMessage("{\"type\":\"config_ack\",\"status\":\"ok\"}"));
            }
        } catch (Exception e) {
            log.warn("Failed to parse text message: {}", e.getMessage());
        }
    }

    @Override
    protected void handleBinaryMessage(WebSocketSession session, BinaryMessage message) {
        AtomicBoolean busy = inFlight.get(session.getId());
        if (busy == null || !busy.compareAndSet(false, true)) {
            // Drop frame if previous frame is still being inferred to avoid queue lag
            return;
        }

        byte[] bytes = new byte[message.getPayload().remaining()];
        message.getPayload().get(bytes);

        String sidKey = session.getId();
        String mv = modelVariants.getOrDefault(sidKey, "Hybrid-S (CNN + 2-Layer Transformer)");
        String prec = precisions.getOrDefault(sidKey, "FP32");
        double conf = confThresholds.getOrDefault(sidKey, 0.25);
        String weather = weatherOverrides.get(sidKey);

        inferencePool.submit(() -> {
            try {
                if (!session.isOpen()) return;
                String json = inferenceService.inferFrameJson(bytes, mv, prec, conf, weather);
                synchronized (session) {
                    if (session.isOpen()) {
                        session.sendMessage(new TextMessage(json));
                    }
                }

                FrameInferencePayload payload = mapper.readValue(json, FrameInferencePayload.class);
                if (payload != null && payload.getLatencyMs() != null) {
                    int fc = frameCounts.merge(sidKey, 1, Integer::sum);
                    fpsAccum.merge(sidKey, payload.getFps(), Double::sum);
                    latAccum.merge(sidKey, payload.getLatencyMs().getTotal(), Double::sum);
                    if (payload.isEmergencyBrakeWarning()) {
                        brakeCounts.merge(sidKey, 1, Integer::sum);
                    }
                    if (fc % 10 == 0) {
                        String mongoSid = sessionIds.get(sidKey);
                        if (mongoSid != null) {
                            telemetry.logAlerts(mongoSid, payload);
                        }
                    }
                }
            } catch (Exception e) {
                log.warn("Error inferring/sending frame for {}: {}", sidKey, e.getMessage());
            } finally {
                busy.set(false);
            }
        });
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) {
        log.info("WebSocket disconnected: {} ({})", session.getId(), status);

        String sid = sessionIds.remove(session.getId());
        int fc = frameCounts.getOrDefault(session.getId(), 0);
        double avgFps = fc > 0 ? fpsAccum.getOrDefault(session.getId(), 0.0) / fc : 0;
        double avgLat = fc > 0 ? latAccum.getOrDefault(session.getId(), 0.0) / fc : 0;
        int brakes = brakeCounts.getOrDefault(session.getId(), 0);

        if (sid != null) {
            telemetry.endSession(sid, fc, avgFps, avgLat, brakes);
        }

        frameCounts.remove(session.getId());
        fpsAccum.remove(session.getId());
        latAccum.remove(session.getId());
        brakeCounts.remove(session.getId());
        inFlight.remove(session.getId());
        modelVariants.remove(session.getId());
        precisions.remove(session.getId());
        confThresholds.remove(session.getId());
    }

    @Override
    public void handleTransportError(WebSocketSession session, Throwable exception) {
        log.error("Transport error on {}: {}", session.getId(), exception.getMessage());
    }
}
