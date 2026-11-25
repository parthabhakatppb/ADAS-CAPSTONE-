"""
Unified YOLO dataset: loads images and YOLO-format labels from merged IDD+KITTI+DAWN
dataset directory structure:
  <data_root>/images/<split>/*.{jpg,png,jpeg}
  <data_root>/labels/<split>/*.txt
"""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class UnifiedYoloDataset(Dataset):
    def __init__(self, data_root, split="train", img_size=640, augment=None):
        self.root = Path(data_root)
        self.img_dir = self.root / "images" / split
        self.lbl_dir = self.root / "labels" / split
        self.img_size = img_size
        self.augment = augment if augment is not None else (split == "train")

        self.image_files = []
        if self.img_dir.exists():
            for ext in ["*.png", "*.ppm", "*.jpg", "*.jpeg"]:
                self.image_files.extend(list(self.img_dir.glob(ext)))
            self.image_files = sorted(self.image_files)

    def __len__(self):
        return len(self.image_files)

    def _load_boxes(self, stem):
        txt = self.lbl_dir / f"{stem}.txt"
        boxes, classes = [], []
        if txt.exists() and txt.stat().st_size > 0:
            for line in txt.read_text(encoding="utf-8", errors="ignore").strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    c, cx, cy, w, h = map(float, parts[:5])
                    boxes.append([cx, cy, w, h])
                    classes.append(int(c))
        return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int64)

    def _apply_augmentation(self, img, boxes_norm):
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            if len(boxes_norm):
                boxes_norm = boxes_norm.copy()
                boxes_norm[:, 0] = 1.0 - boxes_norm[:, 0]

        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-15, 15)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        return img, boxes_norm

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        stem = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))

        boxes_norm, classes = self._load_boxes(stem)

        if self.augment:
            img, boxes_norm = self._apply_augmentation(img, boxes_norm)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if len(boxes_norm):
            cx, cy, bw, bh = boxes_norm[:, 0], boxes_norm[:, 1], boxes_norm[:, 2], boxes_norm[:, 3]
            x1 = (cx - bw / 2.0) * self.img_size
            y1 = (cy - bh / 2.0) * self.img_size
            x2 = (cx + bw / 2.0) * self.img_size
            y2 = (cy + bh / 2.0) * self.img_size
            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        else:
            boxes_xyxy = np.zeros((0, 4), dtype=np.float32)

        return {
            "image": img_t,
            "boxes": torch.from_numpy(boxes_xyxy),
            "classes": torch.from_numpy(classes),
            "stem": stem,
        }


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    boxes = [b["boxes"] for b in batch]
    classes = [b["classes"] for b in batch]
    stems = [b["stem"] for b in batch]
    return {"image": images, "boxes": boxes, "classes": classes, "stems": stems}
