"""
Comprehensive evaluation metrics for ADAS object detection:
- IoU calculation
- Vectorized Non-Maximum Suppression (NMS)
- Precision, Recall, F1
- Average Precision (AP@0.5, AP@0.5:0.95) following COCO / VOC conventions
"""
import numpy as np
import torch


def box_iou_numpy(box1, box2):
    """
    box1: (N, 4) [x1, y1, x2, y2]
    box2: (M, 4) [x1, y1, x2, y2]
    Returns: (N, M) IoU matrix
    """
    if len(box1) == 0 or len(box2) == 0:
        return np.zeros((len(box1), len(box2)), dtype=np.float32)

    area1 = (box1[:, 2] - box1[:, 0]).clip(min=0) * (box1[:, 3] - box1[:, 1]).clip(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clip(min=0) * (box2[:, 3] - box2[:, 1]).clip(min=0)

    lt = np.maximum(box1[:, None, :2], box2[None, :, :2])
    rb = np.minimum(box1[:, None, 2:], box2[None, :, 2:])
    wh = (rb - lt).clip(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, a_min=1e-7, a_max=None)


def nms_numpy(boxes, scores, iou_thresh=0.45):
    """
    Single-class Non-Maximum Suppression.
    boxes: (N, 4) [x1, y1, x2, y2]
    scores: (N,)
    """
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clip(min=0)
        h = (yy2 - yy1).clip(min=0)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.clip(union, a_min=1e-7, a_max=None)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


def postprocess_detections(cls_logits, boxes, conf_thresh=0.25, iou_thresh=0.45):
    """
    Post-process model outputs for one batch element:
    cls_logits: (N, num_classes) raw logits
    boxes: (N, 4) xyxy pixel coords
    Returns:
        kept_boxes: (K, 4)
        kept_scores: (K,)
        kept_classes: (K,)
    """
    scores = cls_logits.sigmoid()
    max_scores, class_ids = scores.max(dim=-1)

    mask = max_scores > conf_thresh
    if not mask.any():
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    b = boxes[mask].cpu().numpy()
    s = max_scores[mask].cpu().numpy()
    c = class_ids[mask].cpu().numpy()

    # Per-class NMS (or batched offset NMS)
    unique_classes = np.unique(c)
    final_boxes, final_scores, final_classes = [], [], []

    for cls in unique_classes:
        cls_mask = c == cls
        cls_boxes = b[cls_mask]
        cls_scores = s[cls_mask]

        keep = nms_numpy(cls_boxes, cls_scores, iou_thresh=iou_thresh)
        final_boxes.append(cls_boxes[keep])
        final_scores.append(cls_scores[keep])
        final_classes.append(c[cls_mask][keep])

    if len(final_boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.concatenate(final_boxes, axis=0),
        np.concatenate(final_scores, axis=0),
        np.concatenate(final_classes, axis=0),
    )


def compute_ap(recall, precision):
    """
    Compute Average Precision using continuous 101-point COCO integration.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute maximum precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 101-point interpolation
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)


def evaluate_predictions(all_preds, all_gts, num_classes, iou_thresholds=None):
    """
    Calculate per-class and mean Precision, Recall, mAP@0.5, mAP@0.5:0.95.
    all_preds: list of per-image dicts: {"boxes": (K, 4), "scores": (K,), "classes": (K,)}
    all_gts: list of per-image dicts: {"boxes": (M, 4), "classes": (M,)}
    """
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.50, 0.95, 10)

    per_class_metrics = {}

    for cls_id in range(num_classes):
        # Collect all predictions and ground truths for this class
        cls_preds = []
        n_pos = 0

        for img_idx, (p, g) in enumerate(zip(all_preds, all_gts)):
            g_mask = g["classes"] == cls_id
            g_boxes = g["boxes"][g_mask]
            n_pos += len(g_boxes)

            p_mask = p["classes"] == cls_id
            p_boxes = p["boxes"][p_mask]
            p_scores = p["scores"][p_mask]

            for box, score in zip(p_boxes, p_scores):
                cls_preds.append({
                    "img_idx": img_idx,
                    "box": box,
                    "score": score,
                })

        if n_pos == 0:
            per_class_metrics[cls_id] = {
                "targets": 0,
                "precision": 0.0,
                "recall": 0.0,
                "map50": 0.0,
                "map50_95": 0.0,
            }
            continue

        if len(cls_preds) == 0:
            per_class_metrics[cls_id] = {
                "targets": n_pos,
                "precision": 0.0,
                "recall": 0.0,
                "map50": 0.0,
                "map50_95": 0.0,
            }
            continue

        # Sort predictions by descending score
        cls_preds = sorted(cls_preds, key=lambda x: x["score"], reverse=True)
        pred_boxes = np.stack([p["box"] for p in cls_preds])
        pred_imgs = np.array([p["img_idx"] for p in cls_preds])

        aps_at_ious = []
        for iou_th in iou_thresholds:
            tp = np.zeros(len(cls_preds))
            fp = np.zeros(len(cls_preds))

            # Track which GT boxes have been matched per image
            matched = {i: set() for i in range(len(all_gts))}

            for p_idx, (p_box, img_i) in enumerate(zip(pred_boxes, pred_imgs)):
                g_mask = all_gts[img_i]["classes"] == cls_id
                gt_boxes_img = all_gts[img_i]["boxes"][g_mask]

                if len(gt_boxes_img) == 0:
                    fp[p_idx] = 1
                    continue

                ious = box_iou_numpy(p_box[None, :], gt_boxes_img)[0]
                best_gt_idx = np.argmax(ious)
                max_iou = ious[best_gt_idx]

                if max_iou >= iou_th:
                    if best_gt_idx not in matched[img_i]:
                        tp[p_idx] = 1
                        matched[img_i].add(best_gt_idx)
                    else:
                        fp[p_idx] = 1
                else:
                    fp[p_idx] = 1

            cumsum_tp = np.cumsum(tp)
            cumsum_fp = np.cumsum(fp)
            recall = cumsum_tp / n_pos
            precision = cumsum_tp / np.maximum(cumsum_tp + cumsum_fp, 1e-7)

            ap = compute_ap(recall, precision)
            aps_at_ious.append(ap)

            if np.isclose(iou_th, 0.50):
                final_p = float(precision[-1]) if len(precision) > 0 else 0.0
                final_r = float(recall[-1]) if len(recall) > 0 else 0.0
                map50 = ap

        per_class_metrics[cls_id] = {
            "targets": n_pos,
            "precision": final_p,
            "recall": final_r,
            "map50": map50,
            "map50_95": float(np.mean(aps_at_ious)),
        }

    # Compute overall mean metrics (macro average across classes with targets > 0)
    valid_classes = [c for c in per_class_metrics if per_class_metrics[c]["targets"] > 0]
    if valid_classes:
        mean_p = float(np.mean([per_class_metrics[c]["precision"] for c in valid_classes]))
        mean_r = float(np.mean([per_class_metrics[c]["recall"] for c in valid_classes]))
        mean_map50 = float(np.mean([per_class_metrics[c]["map50"] for c in valid_classes]))
        mean_map50_95 = float(np.mean([per_class_metrics[c]["map50_95"] for c in valid_classes]))
    else:
        mean_p, mean_r, mean_map50, mean_map50_95 = 0.0, 0.0, 0.0, 0.0

    return {
        "per_class": per_class_metrics,
        "mean": {
            "precision": mean_p,
            "recall": mean_r,
            "map50": mean_map50,
            "map50_95": mean_map50_95,
        }
    }


@torch.no_grad()
def run_validation_evaluation(model, loader, device, num_classes, conf_thresh=0.25, iou_thresh=0.45):
    """
    Run full metric validation over a DataLoader.
    """
    model.eval()
    all_preds, all_gts = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        boxes_list = batch["boxes"]
        classes_list = batch["classes"]

        preds = model(images)
        cls_logits = preds["cls_logits"]
        pred_boxes = preds["boxes"]

        B = images.shape[0]
        for b in range(B):
            k_boxes, k_scores, k_classes = postprocess_detections(
                cls_logits[b], pred_boxes[b], conf_thresh=conf_thresh, iou_thresh=iou_thresh
            )
            all_preds.append({
                "boxes": k_boxes,
                "scores": k_scores,
                "classes": k_classes,
            })

            gt_b = boxes_list[b].cpu().numpy() if torch.is_tensor(boxes_list[b]) else np.array(boxes_list[b])
            gt_c = classes_list[b].cpu().numpy() if torch.is_tensor(classes_list[b]) else np.array(classes_list[b])
            all_gts.append({
                "boxes": gt_b,
                "classes": gt_c,
            })

    return evaluate_predictions(all_preds, all_gts, num_classes=num_classes)
