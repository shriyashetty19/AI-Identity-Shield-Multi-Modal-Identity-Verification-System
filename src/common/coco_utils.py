"""Minimal helpers for reading the COCO-format annotations exported by
Roboflow for the MIDV-2020 dataset."""
import json
from pathlib import Path
from typing import NamedTuple


class DocumentRecord(NamedTuple):
    image_id: int
    file_name: str
    width: int
    height: int
    # category name -> list of [x, y, w, h] boxes (a field can repeat, e.g.
    # primary_identifier shows up in both the visual zone and the MRZ).
    fields: dict[str, list[list[float]]]


def load_coco_split(split_dir: Path) -> list[DocumentRecord]:
    """Load `_annotations.coco.json` from a Roboflow export directory."""
    coco = json.loads((split_dir / "_annotations.coco.json").read_text())
    categories = {c["id"]: c["name"] for c in coco["categories"]}

    fields_by_image: dict[int, dict[str, list[list[float]]]] = {}
    for ann in coco["annotations"]:
        name = categories[ann["category_id"]]
        fields_by_image.setdefault(ann["image_id"], {}).setdefault(name, []).append(ann["bbox"])

    records = []
    for img in coco["images"]:
        records.append(
            DocumentRecord(
                image_id=img["id"],
                file_name=img["file_name"],
                width=img["width"],
                height=img["height"],
                fields=fields_by_image.get(img["id"], {}),
            )
        )
    return records


def largest_box(boxes: list[list[float]]) -> list[float]:
    """Pick the largest box when a category has more than one instance."""
    return max(boxes, key=lambda b: b[2] * b[3])
