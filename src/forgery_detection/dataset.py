import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.common.config import ROOT
from src.forgery_detection.forensics import compute_ela

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224


def build_transforms(train: bool, preprocessing: str = "ela") -> transforms.Compose:
    if preprocessing == "ela":
        # ELA runs first, at the image's native resolution, so the JPEG-block
        # artifacts it exposes aren't blurred away by resizing beforehand.
        # Brightness/contrast jitter is skipped for this input - it's a
        # difference map, not a photo, so photometric augmentation has no
        # natural rationale here.
        ops = [transforms.Lambda(compute_ela), transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    elif preprocessing == "raw":
        ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
        if train:
            ops.append(transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1))
    else:
        raise ValueError(f"unknown preprocessing: {preprocessing!r}")
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


class ForgeryDataset(Dataset):
    """Reads a manifest CSV produced by scripts/generate_synthetic_tampering.py:
    filepath,label,technique,source_image (label 0 = authentic, 1 = tampered).
    """

    def __init__(self, manifest_path: Path, train: bool, preprocessing: str = "ela"):
        with open(manifest_path, newline="") as f:
            self.rows = list(csv.DictReader(f))
        self.transform = build_transforms(train, preprocessing)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(ROOT / row["filepath"]).convert("RGB")
        label = int(row["label"])
        return self.transform(image), label
