import argparse

import torch
from torchvision.ops import box_iou

from src.common.config import MIDV2020_DIR, MODELS_DIR
from src.field_detection.dataset import FieldDetectionDataset
from src.field_detection.model import LABELS, load_model


@torch.no_grad()
def evaluate(model, dataset, device: str, score_threshold: float = 0.5, iou_threshold: float = 0.5) -> dict:
    hits = {label: 0 for label in LABELS[1:]}
    totals = {label: 0 for label in LABELS[1:]}

    for image, target in dataset:
        pred = model([image.to(device)])[0]
        for gt_box, gt_label in zip(target["boxes"], target["labels"]):
            label_name = LABELS[gt_label.item()]
            totals[label_name] += 1
            mask = (pred["labels"] == gt_label) & (pred["scores"] > score_threshold)
            if mask.any():
                candidate_boxes = pred["boxes"][mask]
                ious = box_iou(gt_box.unsqueeze(0).to(candidate_boxes.device), candidate_boxes)
                if ious.max().item() > iou_threshold:
                    hits[label_name] += 1

    return {label: (hits[label] / totals[label] if totals[label] else 0.0) for label in LABELS[1:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(MODELS_DIR / "field_detector.pt"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)

    valid_ds = FieldDetectionDataset(MIDV2020_DIR / "valid")
    results = evaluate(model, valid_ds, device)
    for label, recall in results.items():
        print(f"{label}: recall@IoU0.5 = {recall:.2%}")


if __name__ == "__main__":
    main()
