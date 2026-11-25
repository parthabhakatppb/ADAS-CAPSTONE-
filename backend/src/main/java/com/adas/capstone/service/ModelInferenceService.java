package com.adas.capstone.service;

import com.adas.capstone.model.FrameInferencePayload;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Manages the persistent Python HybridADASModel worker subprocess and runs
 * real PyTorch inference on incoming JPEG camera/video frames.
 */
@Service
public class ModelInferenceService {

    private static final Logger log = LoggerFactory.getLogger(ModelInferenceService.class);

    @Value("${app.inference.python-cmd:python}")
    private String pythonCmd;

    private final ObjectMapper mapper;
    private final SimulatorService fallbackSimulator;

    private Process workerProcess;
    private BufferedWriter workerIn;
    private BufferedReader workerOut;
    private volatile boolean ready = false;
    private final Object lock = new Object();

    public ModelInferenceService(ObjectMapper mapper, SimulatorService fallbackSimulator) {
        this.mapper = mapper;
        this.fallbackSimulator = fallbackSimulator;
    }

    @PostConstruct
    public void startWorker() {
        Thread starter = new Thread(this::initWorkerProcess, "python-worker-init");
        starter.setDaemon(true);
        starter.start();
    }

    private Path resolveRepoRoot() {
        Path cwd = Paths.get("").toAbsolutePath();
        if (Files.exists(cwd.resolve("scripts").resolve("inference_worker.py"))) {
            return cwd;
        }
        Path parent = cwd.getParent();
        if (parent != null && Files.exists(parent.resolve("scripts").resolve("inference_worker.py"))) {
            return parent;
        }
        return cwd;
    }

    private void initWorkerProcess() {
        try {
            Path root = resolveRepoRoot();
            Path script = root.resolve("scripts").resolve("inference_worker.py");
            if (!Files.exists(script)) {
                log.error("Inference worker script not found at {}", script);
                return;
            }

            log.info("Starting Python HybridADASModel worker: {} {}", pythonCmd, script);
            ProcessBuilder pb = new ProcessBuilder(pythonCmd, "-u", script.toString());
            pb.directory(root.toFile());
            pb.redirectError(ProcessBuilder.Redirect.INHERIT);

            workerProcess = pb.start();
            workerIn = new BufferedWriter(new OutputStreamWriter(workerProcess.getOutputStream(), StandardCharsets.UTF_8));
            workerOut = new BufferedReader(new InputStreamReader(workerProcess.getInputStream(), StandardCharsets.UTF_8));

            String banner;
            while ((banner = workerOut.readLine()) != null) {
                String trimmed = banner.trim();
                if (!trimmed.startsWith("{")) {
                    log.debug("Python worker stdout: {}", trimmed);
                    continue;
                }
                JsonNode node = mapper.readTree(trimmed);
                if ("ready".equals(node.path("status").asText())) {
                    ready = true;
                    log.info("Python HybridADASModel worker READY: {}", trimmed);
                    return;
                }
            }
            log.warn("Python worker exited without sending ready banner");
        } catch (Exception e) {
            log.error("Failed to start Python inference worker: {}", e.getMessage());
        }
    }

    public boolean isReady() {
        return ready && workerProcess != null && workerProcess.isAlive();
    }

    public String inferFrameJson(byte[] jpegBytes, String modelVariant, String precision, double confThreshold, String weatherOverride) {
        if (!isReady()) {
            try {
                FrameInferencePayload sim = fallbackSimulator.generateFrame();
                return mapper.writeValueAsString(sim);
            } catch (Exception e) {
                return "{\"error\":\"Inference worker warming up\"}";
            }
        }

        synchronized (lock) {
            try {
                Map<String, Object> req = new LinkedHashMap<>();
                req.put("image_b64", Base64.getEncoder().encodeToString(jpegBytes));
                req.put("model_variant", modelVariant);
                req.put("precision", precision);
                req.put("conf_threshold", confThreshold);
                if (weatherOverride != null && !weatherOverride.isBlank()) {
                    req.put("weather_override", weatherOverride);
                }

                workerIn.write(mapper.writeValueAsString(req));
                workerIn.newLine();
                workerIn.flush();

                String line = workerOut.readLine();
                if (line == null) {
                    ready = false;
                    log.warn("Python inference worker closed stdout; restarting...");
                    startWorker();
                    return mapper.writeValueAsString(fallbackSimulator.generateFrame());
                }
                return line;
            } catch (Exception e) {
                log.error("Inference error: {}", e.getMessage());
                try {
                    return mapper.writeValueAsString(fallbackSimulator.generateFrame());
                } catch (Exception ignored) {
                    return "{\"error\":\"Inference failed\"}";
                }
            }
        }
    }

    @PreDestroy
    public void stopWorker() {
        ready = false;
        if (workerProcess != null && workerProcess.isAlive()) {
            workerProcess.destroy();
        }
    }
}
