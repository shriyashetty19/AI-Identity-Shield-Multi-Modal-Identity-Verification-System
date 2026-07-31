"""Build the forgery-detection training set: for every authentic MIDV-2020
image, generate several tampered counterparts (splice or photo-swap) and
write a manifest CSV listing every (image, label, technique).

copy_paste is deliberately excluded here - it relocates a patch *within the
same image*, so there's no cross-source signal (different compression
history, different resampling) for a classifier to learn from. A classical
self-similarity detector was tried instead (src/forgery_detection/
copy_move.py) but didn't separate tampered from authentic on this document
type either - copy_paste detection is an open problem, not solved by
excluding it from this CNN's training set, just correctly scoped out of it.

VARIANTS_PER_IMAGE multiplies the ~960 source images into a larger training
set (each variant redraws its own random technique/field/donor/quality), the
best data-scale lever available without new source images.

Run after scripts/download_midv2020.py. Idempotent - safe to re-run.
"""
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.coco_utils import load_coco_split
from src.common.config import MIDV2020_DIR, MIDV2020_TAMPER_DIR, ROOT
from src.forgery_detection.tampering import RECOMPRESS_QUALITY_RANGE, _recompress, apply_random_tampering
from PIL import Image

SEED = 42
CNN_TECHNIQUES = ["splice", "photo_swap"]
VARIANTS_PER_IMAGE = 5


def process_split(split: str) -> None:
    split_dir = MIDV2020_DIR / split
    out_dir = MIDV2020_TAMPER_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_coco_split(split_dir)
    rng = random.Random(SEED)

    manifest_path = MIDV2020_TAMPER_DIR / f"{split}_manifest.csv"
    rows = []
    for i, rec in enumerate(records):
        image = Image.open(split_dir / rec.file_name).convert("RGB")

        for variant in range(VARIANTS_PER_IMAGE):
            # Authentic row: re-encoded through the same JPEG quality range
            # the tampering techniques use below, so the model can't tell
            # the classes apart just by detecting a second JPEG
            # re-compression pass - it has to learn the actual tamper
            # evidence instead.
            authentic_out = _recompress(image, rng.randint(*RECOMPRESS_QUALITY_RANGE))
            authentic_name = f"authentic_v{variant}_{rec.file_name}"
            authentic_out.save(out_dir / authentic_name, quality=90)
            rows.append(
                {
                    "filepath": str((out_dir / authentic_name).relative_to(ROOT)),
                    "label": 0,
                    "technique": "none",
                    "source_image": rec.file_name,
                }
            )

            donor = records[rng.randrange(len(records) - 1)] if len(records) > 1 else rec
            if donor.image_id == rec.image_id and len(records) > 1:
                donor = records[(i + 1) % len(records)]

            donor_image = Image.open(split_dir / donor.file_name).convert("RGB")

            tampered, technique = apply_random_tampering(
                image, rec.fields, donor_image, donor.fields, rng, techniques=CNN_TECHNIQUES
            )
            out_name = f"tampered_v{variant}_{rec.file_name}"
            tampered.save(out_dir / out_name, quality=90)

            rows.append(
                {
                    "filepath": str((out_dir / out_name).relative_to(ROOT)),
                    "label": 1,
                    "technique": technique,
                    "source_image": rec.file_name,
                }
            )

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "technique", "source_image"])
        writer.writeheader()
        writer.writerows(rows)

    n = len(records) * VARIANTS_PER_IMAGE
    print(f"{split}: {n} authentic + {n} tampered -> {manifest_path}")


def main() -> None:
    if not MIDV2020_DIR.exists():
        raise SystemExit("data/midv2020 not found - run scripts/download_midv2020.py first.")
    for split in ["train", "valid"]:
        if (MIDV2020_DIR / split).exists():
            process_split(split)


if __name__ == "__main__":
    main()
