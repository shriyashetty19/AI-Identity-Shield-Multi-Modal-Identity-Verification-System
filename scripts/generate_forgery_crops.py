"""Build a crop-based forgery-detection training set: instead of
classifying the whole document, crop just the field region a tampering
technique touched (with padding so the boundary/seam is visible, not just
the interior) and classify authentic-vs-tampered on that crop.

Rationale: the whole-image classifier (src/forgery_detection/train.py) was
tried five ways this session (raw RGB, ELA, at 1x and 5x data scale) and
never beat chance. The OCR half of this project had the exact same
"whole image, no idea where to look" problem - fixing it by localizing
first (src/field_detection/) took it from unusable to genuinely accurate.
This applies the same fix here.

Crops both the authentic and tampered image at the SAME field category per
pair (never independently random) - matching the field-category
distribution between classes is required, or the model could shortcut on
"which field type appears more often in each class" instead of learning
real tamper evidence. That's the same shape of leak already found and
fixed twice this session for JPEG recompression quality - one confound
fixed doesn't mean the next one won't show up elsewhere.

Crops are saved as PNG (lossless) specifically to avoid adding a *third*
JPEG re-compression pass on top of the two the whole-image pipeline
already applies (original acquisition + tampering.py's final recompress) -
an extra JPEG save, even at a fixed quality, is one more place a leak could
hide.

Run after scripts/generate_synthetic_tampering.py - reuses its manifest to
know which technique/field applied to each authentic/tampered pair.
Idempotent - safe to re-run.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from src.common.coco_utils import largest_box, load_coco_split
from src.common.config import DATA_DIR, MIDV2020_DIR, MIDV2020_TAMPER_DIR, ROOT
from src.forgery_detection.region_inference import CROP_PADDING as PADDING

CROP_DIR = DATA_DIR / "midv2020_tamper_crops"


def target_category(technique: str) -> str | None:
    if technique == "photo_swap":
        return "face_image"
    if ":" in technique:
        return technique.split(":", 1)[1]
    return None


def crop_box(image: Image.Image, box: list[float], padding: int = PADDING) -> Image.Image:
    x, y, w, h = box
    x0 = max(0, int(x - padding))
    y0 = max(0, int(y - padding))
    x1 = min(image.width, int(x + w + padding))
    y1 = min(image.height, int(y + h + padding))
    return image.crop((x0, y0, x1, y1))


def process_split(split: str) -> None:
    manifest_path = MIDV2020_TAMPER_DIR / f"{split}_manifest.csv"
    with manifest_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    records_by_name = {rec.file_name: rec for rec in load_coco_split(MIDV2020_DIR / split)}

    out_dir = CROP_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    crop_rows = []
    for i in range(0, len(rows), 2):
        authentic_row, tampered_row = rows[i], rows[i + 1]
        if authentic_row["label"] != "0" or tampered_row["label"] != "1":
            raise ValueError(f"expected an (authentic, tampered) pair at manifest rows {i}/{i + 1}")

        category = target_category(tampered_row["technique"])
        rec = records_by_name[authentic_row["source_image"]]
        if category is None or category not in rec.fields:
            continue
        box = largest_box(rec.fields[category])

        authentic_img = Image.open(ROOT / authentic_row["filepath"]).convert("RGB")
        tampered_img = Image.open(ROOT / tampered_row["filepath"]).convert("RGB")

        authentic_crop = crop_box(authentic_img, box)
        tampered_crop = crop_box(tampered_img, box)

        authentic_name = Path(authentic_row["filepath"]).stem + "_crop.png"
        tampered_name = Path(tampered_row["filepath"]).stem + "_crop.png"
        authentic_crop.save(out_dir / authentic_name)
        tampered_crop.save(out_dir / tampered_name)

        crop_rows.append(
            {
                "filepath": str((out_dir / authentic_name).relative_to(ROOT)),
                "label": 0,
                "technique": "none",
                "category": category,
                "source_image": authentic_row["source_image"],
            }
        )
        crop_rows.append(
            {
                "filepath": str((out_dir / tampered_name).relative_to(ROOT)),
                "label": 1,
                "technique": tampered_row["technique"],
                "category": category,
                "source_image": tampered_row["source_image"],
            }
        )

    manifest_out = CROP_DIR / f"{split}_manifest.csv"
    with manifest_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "technique", "category", "source_image"])
        writer.writeheader()
        writer.writerows(crop_rows)

    n = len(crop_rows) // 2
    print(f"{split}: {n} authentic + {n} tampered crops -> {manifest_out}")


def main() -> None:
    for split in ["train", "valid"]:
        if (MIDV2020_TAMPER_DIR / f"{split}_manifest.csv").exists():
            process_split(split)


if __name__ == "__main__":
    main()
