"""Detect every tamper-prone field region with the trained field detector,
crop each one, classify authentic-vs-tampered per crop, and aggregate into
one verdict (tampered if any region is flagged tampered).

This replaces whole-image classification (src.forgery_detection.inference),
which was tried five ways this session (raw RGB, ELA, at two resolutions
and two data scales) and never beat chance - see project notes. Localizing
before classifying is what fixed OCR (src.field_detection +
DetectorFieldExtractor); this applies the same fix to forgery detection,
and the crop-based classifier it feeds measurably found real signal where
whole-image classification found none (100% precision / 21.5% recall on
the held-out validation crops).
"""
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from src.field_detection.model import LABELS as FIELD_LABELS
from src.forgery_detection.dataset import build_transforms
from src.forgery_detection.model import LABELS as FORGERY_LABELS

# Field-detector boxes fit tightly; padding keeps the boundary/seam visible
# to the crop classifier, not just the interior - the same fix that solved
# document_number OCR misreads (DetectorFieldExtractor.BOX_PADDING) applies
# here too. This value must match scripts/generate_forgery_crops.py's -
# a mismatch would be a train/inference confound, the same class of bug
# already caught twice this session for JPEG recompression quality.
CROP_PADDING = 24
DETECTION_SCORE_THRESHOLD = 0.5


def crop_box(image: Image.Image, box: list[float], padding: int = CROP_PADDING) -> Image.Image:
    x0, y0, x1, y1 = box
    x0 = max(0, int(x0 - padding))
    y0 = max(0, int(y0 - padding))
    x1 = min(image.width, int(x1 + padding))
    y1 = min(image.height, int(y1 + padding))
    return image.crop((x0, y0, x1, y1))


@torch.no_grad()
def predict_regions(
    image: Image.Image, field_model, forgery_model, device: str, preprocessing: str = "raw"
) -> dict:
    input_tensor = TF.to_tensor(image).to(device)
    pred = field_model([input_tensor])[0]

    transform = build_transforms(train=False, preprocessing=preprocessing)
    regions = []
    for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
        if score < DETECTION_SCORE_THRESHOLD:
            continue
        crop = crop_box(image, box.tolist())
        crop_tensor = transform(crop).unsqueeze(0).to(device)
        logits = forgery_model(crop_tensor)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax())
        regions.append(
            {
                "category": FIELD_LABELS[label.item()],
                "label": FORGERY_LABELS[pred_idx],
                "confidence": float(probs[pred_idx]),
            }
        )

    if not regions:
        return {"label": "authentic", "confidence": 0.0, "regions": []}

    tampered_regions = [r for r in regions if r["label"] == "tampered"]
    if tampered_regions:
        worst = max(tampered_regions, key=lambda r: r["confidence"])
        return {"label": "tampered", "confidence": worst["confidence"], "regions": regions}

    # every region says authentic - report the least confident one, so the
    # number reflects genuine uncertainty rather than an arbitrary pick.
    weakest = min(regions, key=lambda r: r["confidence"])
    return {"label": "authentic", "confidence": weakest["confidence"], "regions": regions}
