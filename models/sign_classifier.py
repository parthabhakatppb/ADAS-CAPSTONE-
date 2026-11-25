"""
Stage-2 Traffic Sign Classifier: MobileNetV3-Small on GTSRB (43 classes).
Used in the detect-then-classify pipeline:
Stage 1 (HybridADASModel) detects the traffic sign bounding box ->
Stage 2 (GTSRBClassifier) crops the region and classifies the sign type.
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

GTSRB_CLASSES = {
    0: "Speed limit 20", 1: "Speed limit 30", 2: "Speed limit 50", 3: "Speed limit 60",
    4: "Speed limit 70", 5: "Speed limit 80", 6: "End speed limit 80", 7: "Speed limit 100",
    8: "Speed limit 120", 9: "No overtaking", 10: "No overtaking (trucks)",
    11: "Priority next intersection", 12: "Priority road", 13: "Yield", 14: "Stop",
    15: "No vehicles", 16: "No trucks", 17: "No entry", 18: "General caution",
    19: "Dangerous curve left", 20: "Dangerous curve right", 21: "Double curve",
    22: "Bumpy road", 23: "Slippery road", 24: "Road narrows right",
    25: "Road work", 26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Ice/snow", 31: "Wild animals crossing",
    32: "End of all limits", 33: "Turn right ahead", 34: "Turn left ahead",
    35: "Ahead only", 36: "Go straight or right", 37: "Go straight or left",
    38: "Keep right", 39: "Keep left", 40: "Roundabout mandatory",
    41: "End no overtaking", 42: "End no overtaking (trucks)",
}


class GTSRBClassifier(nn.Module):
    def __init__(self, num_classes=43, pretrained=False, input_size=48):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        # Lightweight MobileNetV3-Small backbone
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)

        # Replace final classifier layer
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        return self.backbone(x)

    @torch.no_grad()
    def predict_crop(self, crop, device="cpu"):
        """
        Takes a cropped image (numpy array BGR or RGB, or PIL Image) and predicts sign class.
        Returns:
            class_id: int
            class_name: str
            confidence: float
        """
        if isinstance(crop, np.ndarray):
            if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                return -1, "Unknown", 0.0
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
        elif isinstance(crop, Image.Image):
            pil_img = crop
        else:
            return -1, "Unknown", 0.0

        tensor = self.transform(pil_img).unsqueeze(0).to(device)
        self.eval()
        logits = self.forward(tensor)
        probs = torch.softmax(logits, dim=-1)
        conf, cls_id = probs.max(dim=-1)

        c_id = cls_id.item()
        c_name = GTSRB_CLASSES.get(c_id, f"Sign_{c_id}")
        return c_id, c_name, conf.item()
