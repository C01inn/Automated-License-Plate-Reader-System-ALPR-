"""
Count the cars in a photo, fully offline.

Uses torchvision's Faster R-CNN (MobileNetV3 backbone) trained on COCO, which is small enough to
run on a Raspberry Pi CPU. Weights (~75 MB) are downloaded once to ~/.cache/torch.

Usage:
    python utils/count_cars.py path/to/image.jpg [--min-score 0.5]
"""

import argparse
import sys
import warnings

import cv2
import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)

warnings.filterwarnings("ignore")

# COCO categories counted as a car. SUVs and pickups are often labelled "truck".
CAR_LABELS = {"car", "truck"}

_model = None


def _load_model():
    global _model
    if _model is None:
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        _model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, box_score_thresh=0.05).eval()
        _model.categories = weights.meta["categories"]
    return _model


def count_cars(image_path: str, min_score: float = 0.5) -> int:
    """Return the number of cars detected in the image at image_path."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    model = _load_model()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    with torch.inference_mode():
        out = model([tensor])[0]

    return sum(
        1
        for label, score in zip(out["labels"].tolist(), out["scores"].tolist())
        if score >= min_score and model.categories[label] in CAR_LABELS
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Count the cars in an image (offline).")
    ap.add_argument("image", help="path to the image")
    ap.add_argument("--min-score", type=float, default=0.5, help="detection confidence threshold (0-1)")
    args = ap.parse_args()

    try:
        print(count_cars(args.image, args.min_score))
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
