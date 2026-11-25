"""
Phase 1 Validation Script: Evaluates a trained hybrid detector checkpoint on
validation data, reporting per-class Precision, Recall, mAP@0.5, and mAP@0.5:0.95.

Usage:
    python scripts/val_phase1.py \
        --data_root ./data/phase1_yolo/merged \
        --weights runs/phase1_hybrid_s/best.pt \
        --variant hybrid_s
"""
import argparse
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.hybrid_model import HybridADASModel
from data.phase1_dataset import Phase1Dataset, collate_fn
from configs.phase1_classes import PHASE1_CLASSES, NUM_PHASE1_CLASSES
from utils.metrics import run_validation_evaluation

VARIANT_PRESETS = {
    "cnn_baseline": dict(width_mult=0.50, depth_mult=0.33, num_transformer_layers=0),
    "hybrid_n":     dict(width_mult=0.25, depth_mult=0.33, num_transformer_layers=1),
    "hybrid_s":     dict(width_mult=0.50, depth_mult=0.33, num_transformer_layers=2),
    "hybrid_m":     dict(width_mult=0.75, depth_mult=0.67, num_transformer_layers=4),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data/phase1_yolo/merged", help="Path to merged dataset")
    ap.add_argument("--weights", required=True, help="Path to checkpoint .pt")
    ap.add_argument("--variant", default="hybrid_s", choices=list(VARIANT_PRESETS.keys()))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--conf_thresh", type=float, default=0.25)
    ap.add_argument("--iou_thresh", type=float, default=0.45)
    ap.add_argument("--num_workers", type=int, default=2)
    return ap.parse_args()


def print_metrics_table(eval_results):
    per_class = eval_results["per_class"]
    mean = eval_results["mean"]

    sep = "+" + "-" * 18 + "+" + "-" * 10 + "+" + "-" * 12 + "+" + "-" * 10 + "+" + "-" * 11 + "+" + "-" * 14 + "+"
    header = f"| {'Class':<16} | {'Targets':<8} | {'Precision':<10} | {'Recall':<8} | {'mAP@0.5':<9} | {'mAP@0.5:0.95':<12} |"
    print("\n" + sep)
    print(header)
    print(sep)

    total_targets = 0
    for cls_id in sorted(per_class.keys()):
        m = per_class[cls_id]
        name = PHASE1_CLASSES.get(cls_id, f"class_{cls_id}")
        total_targets += m["targets"]
        print(f"| {name:<16} | {m['targets']:<8d} | {m['precision']:<10.4f} | {m['recall']:<8.4f} | {m['map50']:<9.4f} | {m['map50_95']:<12.4f} |")

    print(sep)
    print(f"| {'all (mean)':<16} | {total_targets:<8d} | {mean['precision']:<10.4f} | {mean['recall']:<8.4f} | {mean['map50']:<9.4f} | {mean['map50_95']:<12.4f} |")
    print(sep + "\n")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_ds = Phase1Dataset(args.data_root, split="val", img_size=args.img_size, augment=False)
    if len(val_ds) == 0:
        print(f"Warning: No images found in {args.data_root}/images/val!")
        return

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers
    )

    cfg = VARIANT_PRESETS[args.variant]
    model = HybridADASModel(
        num_classes=NUM_PHASE1_CLASSES,
        use_drivable_head=False,
        use_weather_head=False,
        **cfg
    ).to(device)

    print(f"Loading weights from {args.weights}...")
    ckpt = torch.load(args.weights, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    t0 = time.time()
    results = run_validation_evaluation(
        model, val_loader, device,
        num_classes=NUM_PHASE1_CLASSES,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh
    )
    eval_time = time.time() - t0

    print_metrics_table(results)
    print(f"Validation completed in {eval_time:.1f}s ({len(val_ds)} images, {len(val_ds)/eval_time:.1f} FPS)")


if __name__ == "__main__":
    main()
