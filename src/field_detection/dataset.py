from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from src.common.coco_utils import largest_box, load_coco_split
from src.field_detection.model import CATEGORY_TO_LABEL


class FieldDetectionDataset(Dataset):
    """MIDV-2020's own annotated field boxes, adapted to torchvision's
    detection format: one box per category per image (the largest instance
    when a category repeats - the same convention already used by
    `TrOCRFieldExtractor` and `tampering.py`), for the categories in
    `CATEGORY_TO_LABEL`.
    """

    def __init__(self, split_dir: Path):
        self.split_dir = split_dir
        self.records = [
            r for r in load_coco_split(split_dir) if any(c in r.fields for c in CATEGORY_TO_LABEL)
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image = Image.open(self.split_dir / rec.file_name).convert("RGB")

        boxes, labels = [], []
        for category, label in CATEGORY_TO_LABEL.items():
            if category not in rec.fields:
                continue
            x, y, w, h = largest_box(rec.fields[category])
            boxes.append([x, y, x + w, y + h])
            labels.append(label)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return TF.to_tensor(image), target


def collate_fn(batch):
    return tuple(zip(*batch))
