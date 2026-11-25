"""
Hybrid CNN-Vision Transformer ADAS Live Perception & Safety Telemetry Engine
Stage 1: Monocular Input Stream (Dashcam video or live camera)
Stage 2: Hybrid Vision Engine (CSPDarknet + MHSA Transformer Neck + Decoupled Head + NMS)
Stage 3: Multi-Object Tracker (ByteTrack with persistent ID assignment)
Stage 4: ADAS Telemetry Engine (Pinhole Distance, Speed Derivatives, TTC, & FCW/AEB Alerts)
+ Stage-2 Traffic Sign Classifier (MobileNetV3-Small GTSRB)
"""
import argparse
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
import sys

# Supervision for ByteTrack
try:
    import supervision as sv
    HAS_SUPERVISION = True
except ImportError:
    HAS_SUPERVISION = False

from models.hybrid_model import build_model_variant
from models.sign_classifier import GTSRBClassifier, GTSRB_CLASSES
from configs.phase1_classes import PHASE1_CLASSES, NUM_PHASE1_CLASSES
from utils.metrics import postprocess_detections

# Class real-world standard heights for monocular distance estimation (in meters)
REAL_WORLD_HEIGHTS = {
    0: 1.50,  # car
    1: 3.20,  # van_truck
    2: 3.40,  # tram / bus
    3: 1.70,  # pedestrian
    4: 1.20,  # cyclist
    5: 0.80,  # traffic_sign
}

# Alert BGR colors
ALERT_COLORS = {
    "SAFE": (60, 200, 60),           # Green
    "CAUTION_FCW": (0, 165, 255),    # Orange
    "CRITICAL_BRAKE": (0, 0, 255),   # Red
}


class ByteTrackTracker:
    """
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box (Zhang et al. ECCV 2022).
    Two-stage association:
      Stage 1: Match high-confidence detections with active tracks using IoU.
      Stage 2: Match remaining tracks with low-confidence detections to recover occluded objects.
    """
    def __init__(self, high_thresh=0.35, low_thresh=0.10, match_thresh=0.3, max_lost=25):
        self.next_id = 1
        self.tracks = {}  # track_id -> dict(box, score, cls, lost, hits)
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_lost = max_lost

    def _iou(self, boxes1, boxes2):
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = (rb - lt).clip(min=0)
        inter = wh[..., 0] * wh[..., 1]
        union = area1[:, None] + area2[None, :] - inter
        return inter / np.clip(union, a_min=1e-7, a_max=None)

    def update(self, boxes, scores, classes):
        if len(boxes) == 0:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int), np.empty((0,), dtype=int)

        high_mask = scores >= self.high_thresh
        low_mask = (scores >= self.low_thresh) & (~high_mask)

        high_boxes, high_scores, high_classes = boxes[high_mask], scores[high_mask], classes[high_mask]
        low_boxes, low_scores, low_classes = boxes[low_mask], scores[low_mask], classes[low_mask]

        active_track_ids = list(self.tracks.keys())
        matched_tracks = set()
        matched_high_dets = set()
        out_boxes, out_scores, out_classes, out_ids = [], [], [], []

        # --- Stage 1: Match high-confidence detections ---
        if len(active_track_ids) > 0 and len(high_boxes) > 0:
            track_boxes = np.stack([self.tracks[tid]["box"] for tid in active_track_ids])
            ious1 = self._iou(high_boxes, track_boxes)

            flat_order = np.argsort(-ious1, axis=None)
            for idx in flat_order:
                d_idx, t_idx = np.unravel_index(idx, ious1.shape)
                if ious1[d_idx, t_idx] < self.match_thresh:
                    break
                tid = active_track_ids[t_idx]
                if d_idx in matched_high_dets or tid in matched_tracks:
                    continue
                matched_high_dets.add(d_idx)
                matched_tracks.add(tid)
                prev_b = self.tracks[tid]["box"]
                curr_b = high_boxes[d_idx]
                smoothed_b = 0.75 * curr_b + 0.25 * prev_b
                self.tracks[tid].update({
                    "box": smoothed_b,
                    "score": high_scores[d_idx],
                    "cls": high_classes[d_idx],
                    "lost": 0,
                    "hits": self.tracks[tid]["hits"] + 1,
                })
                out_boxes.append(smoothed_b)
                out_scores.append(high_scores[d_idx])
                out_classes.append(high_classes[d_idx])
                out_ids.append(tid)

        # --- Stage 2: Match remaining tracks with low-confidence detections ---
        unmatched_track_ids = [tid for tid in active_track_ids if tid not in matched_tracks]
        matched_low_dets = set()

        if len(unmatched_track_ids) > 0 and len(low_boxes) > 0:
            unmatched_track_boxes = np.stack([self.tracks[tid]["box"] for tid in unmatched_track_ids])
            ious2 = self._iou(low_boxes, unmatched_track_boxes)

            flat_order = np.argsort(-ious2, axis=None)
            for idx in flat_order:
                d_idx, t_idx = np.unravel_index(idx, ious2.shape)
                if ious2[d_idx, t_idx] < 0.20:
                    break
                tid = unmatched_track_ids[t_idx]
                if d_idx in matched_low_dets or tid in matched_tracks:
                    continue
                matched_low_dets.add(d_idx)
                matched_tracks.add(tid)
                prev_b = self.tracks[tid]["box"]
                curr_b = low_boxes[d_idx]
                smoothed_b = 0.60 * curr_b + 0.40 * prev_b
                self.tracks[tid].update({
                    "box": smoothed_b,
                    "score": low_scores[d_idx],
                    "cls": low_classes[d_idx],
                    "lost": 0,
                    "hits": self.tracks[tid]["hits"] + 1,
                })
                out_boxes.append(smoothed_b)
                out_scores.append(low_scores[d_idx])
                out_classes.append(low_classes[d_idx])
                out_ids.append(tid)

        # --- Initialize new tracks from unmatched high-confidence detections ---
        for d_idx in range(len(high_boxes)):
            if d_idx not in matched_high_dets:
                tid = self.next_id
                self.next_id += 1
                b = high_boxes[d_idx]
                s = high_scores[d_idx]
                c = high_classes[d_idx]
                self.tracks[tid] = {
                    "box": b,
                    "score": s,
                    "cls": c,
                    "lost": 0,
                    "hits": 1,
                }
                out_boxes.append(b)
                out_scores.append(s)
                out_classes.append(c)
                out_ids.append(tid)

        # --- Age unmatched tracks ---
        for tid in active_track_ids:
            if tid not in matched_tracks:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]

        if len(out_boxes) == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int), np.empty((0,), dtype=int)

        return np.array(out_boxes), np.array(out_scores), np.array(out_classes), np.array(out_ids, dtype=int)


class ADASEngine:
    def __init__(self, focal_length_y=720.0, min_dist=1.0, max_dist=120.0):
        self.focal_length_y = focal_length_y
        self.min_dist = min_dist
        self.max_dist = max_dist
        # Track history: track_id -> deque of (timestamp, distance_m, centroid_y)
        self.history = defaultdict(lambda: deque(maxlen=15))

    def estimate_distance(self, bbox, cls_id):
        """Pinhole monocular distance proxy: Z = (f * H_real) / H_pixel"""
        x1, y1, x2, y2 = bbox
        h_pixel = max(y2 - y1, 1.0)
        h_real = REAL_WORLD_HEIGHTS.get(int(cls_id), 1.50)
        distance = (self.focal_length_y * h_real) / h_pixel
        return float(np.clip(distance, self.min_dist, self.max_dist))

    def compute_ttc_and_speed(self, track_id, current_dist, current_time, bbox):
        """
        Calculates relative speed and Time-To-Collision (TTC) using derivative over track history.
        """
        x1, y1, x2, y2 = bbox
        cy = (y1 + y2) / 2.0
        hist = self.history[track_id]
        hist.append((current_time, current_dist, cy))

        if len(hist) < 3:
            return 0.0, float("inf"), "SAFE"

        dt = hist[-1][0] - hist[0][0]
        if dt < 0.05:
            return 0.0, float("inf"), "SAFE"

        # Change in distance over dt (negative slope means approaching)
        dd = hist[-1][1] - hist[0][1]
        v_closing = -dd / dt  # m/s (+ means getting closer)
        v_rel_kmh = v_closing * 3.6

        if v_closing > 0.5:
            ttc = current_dist / v_closing
        else:
            ttc = float("inf")

        # Emergency Braking Warning Condition:
        # TTC < 2.5s OR pedestrian/cyclist < 8.0m in vehicle path
        cls_id = getattr(self, "current_cls", None)
        is_vulnerable = cls_id in (3, 4)

        if ttc < 2.5 or (is_vulnerable and current_dist < 8.0):
            status = "CRITICAL_BRAKE"
        elif ttc < 4.0 or current_dist < 15.0:
            status = "CAUTION_FCW"
        else:
            status = "SAFE"

        return v_rel_kmh, ttc, status


class ADASPerceptionPipeline:
    def __init__(
        self,
        weights_path="runs/phase1_hybrid_s/best.pt",
        variant="hybrid_s",
        sign_weights_path=None,
        conf_thresh=0.25,
        iou_thresh=0.45,
        focal_length=720.0,
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # 1. Load Hybrid Detector
        print(f"Loading {variant} Hybrid Vision Engine on {self.device}...")
        self.detector = build_model_variant(
            variant,
            num_classes=NUM_PHASE1_CLASSES,
            use_drivable_head=False,
            use_weather_head=False,
        ).to(self.device)

        if Path(weights_path).exists():
            ckpt = torch.load(weights_path, map_location=self.device)
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            self.detector.load_state_dict(state_dict)
            print(f"Loaded detector weights from {weights_path}")
        else:
            print(f"Warning: Checkpoint {weights_path} not found. Running with initialized weights.")
        self.detector.eval()

        # 2. Stage-2 GTSRB Sign Classifier
        self.sign_classifier = GTSRBClassifier(num_classes=43).to(self.device)
        if sign_weights_path and Path(sign_weights_path).exists():
            s_ckpt = torch.load(sign_weights_path, map_location=self.device)
            self.sign_classifier.load_state_dict(s_ckpt)
            print(f"Loaded GTSRB sign classifier from {sign_weights_path}")
        self.sign_classifier.eval()

        # 3. Tracker: ByteTrack Multi-Object Tracker
        self.tracker = ByteTrackTracker(high_thresh=conf_thresh, low_thresh=0.10)
        print("Initialized ByteTrack multi-object tracker.")

        # 4. Telemetry Engine
        self.adas_engine = ADASEngine(focal_length_y=focal_length)

    @torch.no_grad()
    def process_frame(self, frame_bgr, timestamp_s):
        h, w = frame_bgr.shape[:2]

        # Preprocess for model (640x640)
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0

        # Stage 2: Hybrid Perception Forward Pass
        preds = self.detector(img_tensor)
        cls_logits = preds["cls_logits"][0]
        boxes = preds["boxes"][0]

        # NMS Post-Processing
        kept_boxes, kept_scores, kept_classes = postprocess_detections(
            cls_logits, boxes, conf_thresh=self.conf_thresh, iou_thresh=self.iou_thresh
        )

        # Scale coordinates back to original frame resolution
        scale_x = w / 640.0
        scale_y = h / 640.0
        if len(kept_boxes) > 0:
            kept_boxes[:, [0, 2]] *= scale_x
            kept_boxes[:, [1, 3]] *= scale_y

        # Stage 3: Multi-Object Tracking (ByteTrack)
        boxes_t, scores_t, cls_t, ids_t = self.tracker.update(kept_boxes, kept_scores, kept_classes)
        tracked_objects = []
        for b, s, c, tid in zip(boxes_t, scores_t, cls_t, ids_t):
            tracked_objects.append({
                "box": b,
                "score": float(s),
                "cls": int(c),
                "id": int(tid),
            })

        # Stage 4: ADAS Telemetry (Distance, Speed, TTC, Alerts)
        active_critical_alert = False
        min_ttc = float("inf")
        closest_distance = float("inf")
        annotated_tracks = []

        for obj in tracked_objects:
            bbox = obj["box"]
            cls_id = obj["cls"]
            track_id = obj["id"]

            dist_m = self.adas_engine.estimate_distance(bbox, cls_id)
            self.adas_engine.current_cls = cls_id
            speed_kmh, ttc_s, alert_state = self.adas_engine.compute_ttc_and_speed(
                track_id, dist_m, timestamp_s, bbox
            )

            # Stage-2 GTSRB Sign Classification (if traffic_sign detected)
            sign_name = None
            if cls_id == 5:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    _, s_name, s_conf = self.sign_classifier.predict_crop(crop, device=self.device)
                    sign_name = f"{s_name} ({s_conf*100:.0f}%)"

            if alert_state == "CRITICAL_BRAKE":
                active_critical_alert = True

            min_ttc = min(min_ttc, ttc_s)
            closest_distance = min(closest_distance, dist_m)

            annotated_tracks.append({
                "box": bbox,
                "cls_id": cls_id,
                "track_id": track_id,
                "score": obj["score"],
                "dist_m": dist_m,
                "speed_kmh": speed_kmh,
                "ttc_s": ttc_s,
                "alert": alert_state,
                "sign_name": sign_name,
            })

        system_status = "CRITICAL_BRAKE" if active_critical_alert else ("CAUTION_FCW" if min_ttc < 4.0 else "SAFE")
        return annotated_tracks, system_status, closest_distance, min_ttc


def draw_adas_hud(frame, tracks, system_status, closest_dist, min_ttc, fps):
    h, w = frame.shape[:2]

    # Draw tracks & telemetry
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t["box"]]
        alert = t["alert"]
        color = ALERT_COLORS.get(alert, (200, 200, 200))
        cls_name = PHASE1_CLASSES.get(t["cls_id"], "Obj")
        if t["sign_name"]:
            cls_name = t["sign_name"]

        # Bounding box
        thickness = 3 if alert == "CRITICAL_BRAKE" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label tag
        dist_str = f"{t['dist_m']:.1f}m"
        ttc_str = f"TTC:{t['ttc_s']:.1f}s" if t["ttc_s"] < 20.0 else "TTC:inf"
        spd_str = f"{t['speed_kmh']:+.1f}km/h"
        label = f"ID:{t['track_id']} {cls_name} | {dist_str} | {spd_str} | {ttc_str}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Top HUD Banner
    banner_color = ALERT_COLORS.get(system_status, (60, 200, 60))
    cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.rectangle(frame, (0, 50), (w, 55), banner_color, -1)

    hud_text = f"ADAS PILOT: {system_status} | FPS: {fps:.1f} | Nearest: {closest_dist:.1f}m | Min TTC: {min_ttc:.1f}s"
    cv2.putText(frame, hud_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Emergency Flashing Warning if CRITICAL_BRAKE
    if system_status == "CRITICAL_BRAKE":
        cv2.rectangle(frame, (w // 4, h - 80), (3 * w // 4, h - 20), (0, 0, 255), -1)
        cv2.putText(
            frame,
            "EMERGENCY BRAKE ALERT - COLLISION IMMINENT",
            (w // 4 + 20, h - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


def run_live_demo(
    video_path,
    output_path=None,
    weights_path="runs/phase1_hybrid_s/best.pt",
    variant="hybrid_s",
    sign_weights=None,
    conf_thresh=0.25,
    max_frames=None,
    no_display=False,
):
    pipeline = ADASPerceptionPipeline(
        weights_path=weights_path,
        variant=variant,
        sign_weights_path=sign_weights,
        conf_thresh=conf_thresh,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video source: {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"\n=== Starting Live ADAS Inference ===")
    print(f"Source: {video_path} ({w}x{h} @ {fps:.1f} FPS)")
    if output_path:
        print(f"Output: {output_path}")

    frame_count = 0
    t_start = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and frame_count >= max_frames):
                break

            t_now = frame_count / fps
            frame_count += 1

            t_proc0 = time.time()
            tracks, status, closest_dist, min_ttc = pipeline.process_frame(frame, t_now)
            proc_time = time.time() - t_proc0
            inst_fps = 1.0 / max(proc_time, 1e-4)

            annotated = draw_adas_hud(frame, tracks, status, closest_dist, min_ttc, inst_fps)

            # Terminal log for active alerts
            if status != "SAFE" or frame_count % 30 == 0:
                print(
                    f"Frame {frame_count:04d} | Status: {status:<15s} | "
                    f"Closest: {closest_dist:5.1f}m | Min TTC: {min_ttc:4.1f}s | "
                    f"Tracks: {len(tracks):2d} | FPS: {inst_fps:4.1f}"
                )

            if writer:
                writer.write(annotated)

            if not no_display:
                cv2.imshow("Hybrid CNN-Transformer ADAS Perception", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        if not no_display:
            cv2.destroyAllWindows()

    total_time = time.time() - t_start
    print(f"\nInference completed: {frame_count} frames in {total_time:.2f}s ({frame_count/max(total_time, 1e-4):.1f} FPS)")


def main():
    ap = argparse.ArgumentParser(description="Live Hybrid CNN-ViT ADAS Perception Demo")
    ap.add_argument("--video", default="sample_dashcam.mp4", help="Video path or camera index")
    ap.add_argument("--weights", default="runs/phase1_hybrid_s/best.pt", help="Hybrid model checkpoint")
    ap.add_argument("--variant", default="hybrid_s", choices=["cnn_baseline", "hybrid_n", "hybrid_s", "hybrid_m"])
    ap.add_argument("--sign_classifier", default=None, help="Stage-2 GTSRB sign classifier checkpoint")
    ap.add_argument("--out", default="adas_output.mp4", help="Path to save annotated video")
    ap.add_argument("--conf_thresh", type=float, default=0.25)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--no_display", action="store_true", help="Disable GUI window display")
    args = ap.parse_args()

    # If video does not exist, build a synthetic dashcam test clip
    video_source = args.video
    if not Path(video_source).exists():
        print(f"Video {video_source} not found. Generating synthetic test dashcam clip 'sample_dashcam.mp4'...")
        video_source = "sample_dashcam.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w, h, vid_fps = 640, 480, 20
        writer = cv2.VideoWriter(video_source, fourcc, vid_fps, (w, h))

        for f_idx in range(60):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            # Sky
            frame[:220, :] = [220, 180, 130]
            # Road
            frame[220:, :] = [70, 70, 70]
            # Lane markings
            cv2.line(frame, (320, 220), (80, 480), (255, 255, 255), 3)
            cv2.line(frame, (320, 220), (560, 480), (255, 255, 255), 3)

            # Approaching Car (getting larger and closer)
            scale = 1.0 + f_idx * 0.03
            car_w = int(60 * scale)
            car_h = int(40 * scale)
            car_cx = 320
            car_cy = int(240 + f_idx * 3.0)
            cv2.rectangle(
                frame,
                (car_cx - car_w // 2, car_cy - car_h // 2),
                (car_cx + car_w // 2, car_cy + car_h // 2),
                (50, 50, 220),
                -1,
            )

            # Pedestrian on shoulder
            ped_h_sz = 50
            ped_w_sz = 18
            ped_x = 120
            ped_y = 230
            cv2.rectangle(frame, (ped_x, ped_y), (ped_x + ped_w_sz, ped_y + ped_h_sz), (0, 200, 0), -1)

            writer.write(frame)
        writer.release()
        print("Generated sample_dashcam.mp4 (60 frames).")

    run_live_demo(
        video_source,
        output_path=args.out,
        weights_path=args.weights,
        variant=args.variant,
        sign_weights=args.sign_classifier,
        conf_thresh=args.conf_thresh,
        max_frames=args.max_frames,
        no_display=args.no_display,
    )


if __name__ == "__main__":
    main()
