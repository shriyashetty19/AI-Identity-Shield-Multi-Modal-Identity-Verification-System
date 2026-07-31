"""Measure real OCR field-extraction accuracy against a small hand-labeled
ground-truth sample (data/midv2020_ocr_ground_truth.csv), for:

- TrOCRFieldExtractor: reads MIDV-2020's own annotated field boxes (best
  case - the field location is already known). Currently unavailable in
  this environment (transformers/tokenizer incompatibility - see extract.py).
- DonutFieldExtractor: zero-shot DocVQA-style prompting on the whole image.
- DetectorFieldExtractor: detect-then-read using the trained
  src.field_detection model + easyocr - what the FastAPI backend actually
  uses on arbitrary uploads.

MIDV-2020's export ships no transcribed text, so this hand-labeled sample is
the only source of ground truth available - see data/midv2020_ocr_ground_truth.csv
for how it was built.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from src.common.coco_utils import load_coco_split
from src.common.config import DATA_DIR, MIDV2020_DIR, MODELS_DIR
from src.ocr.extract import DetectorFieldExtractor, DonutFieldExtractor, TrOCRFieldExtractor
from src.ocr.validate import validate_date, validate_document_number

GROUND_TRUTH_CSV = DATA_DIR / "midv2020_ocr_ground_truth.csv"


def name_matches(extracted: str | None, expected: str) -> bool:
    if not extracted:
        return False
    tokens = [t.upper() for t in extracted.replace("/", " ").split()]
    return expected.upper() in tokens


def date_matches(extracted: str | None, expected_iso: str) -> bool:
    if not extracted:
        return False
    result = validate_date(extracted)
    return result["valid"] and result["normalized"] == expected_iso


def document_number_matches(extracted: str | None, expected: str) -> bool:
    if not extracted:
        return False
    result = validate_document_number(extracted)
    return result["valid"] and result["normalized"] == expected


MATCHERS = {
    "name": name_matches,
    "date_of_birth": date_matches,
    "document_number": document_number_matches,
}


def main() -> None:
    with GROUND_TRUTH_CSV.open(newline="") as f:
        ground_truth = list(csv.DictReader(f))

    valid_dir = MIDV2020_DIR / "valid"
    records_by_name = {rec.file_name: rec for rec in load_coco_split(valid_dir)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    extractors = {"donut": DonutFieldExtractor(device=device)}

    try:
        extractors["trocr"] = TrOCRFieldExtractor(device=device)
    except Exception as exc:
        print(f"TrOCRFieldExtractor unavailable in this environment, skipping: {exc}")

    field_checkpoint = MODELS_DIR / "field_detector.pt"
    if field_checkpoint.exists():
        extractors["detector"] = DetectorFieldExtractor(field_checkpoint, device=device)
    else:
        print(f"no field-detector checkpoint at {field_checkpoint}, skipping")

    results = {name: {f: 0 for f in MATCHERS} for name in extractors}
    total = len(ground_truth)

    for row in ground_truth:
        record = records_by_name[row["filename"]]
        image = Image.open(valid_dir / row["filename"]).convert("RGB")

        extracted = {"donut": extractors["donut"].extract_fields(image)}
        if "trocr" in extractors:
            extracted["trocr"] = extractors["trocr"].extract_fields_from_boxes(image, record.fields)
        if "detector" in extractors:
            extracted["detector"] = extractors["detector"].extract_fields(image)

        print(f"\n{row['filename']}")
        for field, matcher in MATCHERS.items():
            expected = row[field]
            print(f"  {field}: expected={expected!r}")
            for name in extractors:
                value = extracted[name].get(field)
                ok = matcher(value, expected)
                results[name][field] += ok
                print(f"    {name}={value!r} {'OK' if ok else 'MISS'}")

    print(f"\n=== accuracy over {total} hand-labeled samples ===")
    for extractor, field_counts in results.items():
        print(f"{extractor}:")
        for field, correct in field_counts.items():
            print(f"  {field}: {correct}/{total} ({correct / total:.0%})")


if __name__ == "__main__":
    main()
