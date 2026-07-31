import torch
import torch.nn.functional as F
from PIL import Image

from src.forgery_detection.dataset import build_transforms
from src.forgery_detection.model import LABELS

_transform = build_transforms(train=False)


@torch.no_grad()
def predict(image: Image.Image, model, device: str) -> dict:
    input_tensor = _transform(image).unsqueeze(0).to(device)
    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1)[0]
    pred_idx = int(probs.argmax())
    return {
        "label": LABELS[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {label: float(p) for label, p in zip(LABELS, probs.tolist())},
    }
