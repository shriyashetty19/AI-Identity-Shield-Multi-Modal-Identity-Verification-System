"""Synthetic document tampering generators.

Each function takes PIL images plus the field bounding boxes parsed by
`src.common.coco_utils.load_coco_split` and returns a tampered copy of the
image. These simulate the three most common real-world ID-forgery patterns:

- copy_paste: a region is cloned and pasted elsewhere in the same document
  (classic copy-move forgery, e.g. duplicating a digit to alter a number).
- splice: a field's content is replaced with the equivalent field cropped
  from a different donor document (e.g. swapping in a different birth date).
- photo_swap: the portrait photo is replaced with a donor's portrait
  (identity-photo substitution fraud).
"""
import io
import random

from PIL import Image

from src.common.coco_utils import largest_box

TEXT_FIELDS = [
    "date_of_birth",
    "date_of_expiry",
    "date_of_issue",
    "document_number",
    "personal_number",
    "primary_identifier",
    "secondary_identifier",
    "place_of_birth",
]

# Every technique's final re-encode - and the untampered control image's
# matching re-encode in scripts/generate_synthetic_tampering.py - must draw
# from this exact same quality range. Any difference here becomes a
# class-correlated shortcut the classifier can exploit instead of learning
# real tamper evidence (this happened once already: the ranges used to
# differ per technique).
RECOMPRESS_QUALITY_RANGE = (45, 85)


def _box_to_int(box: list[float]) -> tuple[int, int, int, int]:
    x, y, w, h = box
    return int(x), int(y), int(x + w), int(y + h)


def _recompress(image: Image.Image, quality: int) -> Image.Image:
    """Re-encode at a different JPEG quality to mimic the compression-artifact
    discontinuities real splices leave behind."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def copy_paste(image: Image.Image, fields: dict, rng: random.Random) -> tuple[Image.Image, str] | None:
    candidates = [f for f in TEXT_FIELDS if f in fields]
    if not candidates:
        return None
    field = rng.choice(candidates)
    box = _box_to_int(largest_box(fields[field]))
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0

    out = image.copy()
    patch = image.crop(box)

    dx = rng.choice([-1, 1]) * rng.randint(w // 2, w)
    dy = rng.randint(-h // 2, h // 2)
    nx0 = min(max(x0 + dx, 0), out.width - w)
    ny0 = min(max(y0 + dy, 0), out.height - h)
    out.paste(patch, (nx0, ny0))
    return _recompress(out, rng.randint(*RECOMPRESS_QUALITY_RANGE)), f"copy_paste:{field}"


def splice(
    image: Image.Image,
    fields: dict,
    donor_image: Image.Image,
    donor_fields: dict,
    rng: random.Random,
) -> tuple[Image.Image, str] | None:
    candidates = [f for f in TEXT_FIELDS if f in fields and f in donor_fields]
    if not candidates:
        return None
    field = rng.choice(candidates)
    x0, y0, x1, y1 = _box_to_int(largest_box(fields[field]))
    w, h = x1 - x0, y1 - y0

    donor_box = _box_to_int(largest_box(donor_fields[field]))
    donor_patch = donor_image.crop(donor_box).resize((w, h))

    out = image.copy()
    out.paste(donor_patch, (x0, y0))
    return _recompress(out, rng.randint(*RECOMPRESS_QUALITY_RANGE)), f"splice:{field}"


def photo_swap(
    image: Image.Image,
    fields: dict,
    donor_image: Image.Image,
    donor_fields: dict,
    rng: random.Random,
) -> tuple[Image.Image, str] | None:
    if "face_image" not in fields or "face_image" not in donor_fields:
        return None
    x0, y0, x1, y1 = _box_to_int(largest_box(fields["face_image"]))
    w, h = x1 - x0, y1 - y0

    donor_box = _box_to_int(largest_box(donor_fields["face_image"]))
    donor_patch = donor_image.crop(donor_box).resize((w, h))

    out = image.copy()
    out.paste(donor_patch, (x0, y0))
    return _recompress(out, rng.randint(*RECOMPRESS_QUALITY_RANGE)), "photo_swap"


TECHNIQUES = ["copy_paste", "splice", "photo_swap"]


def apply_random_tampering(
    image: Image.Image,
    fields: dict,
    donor_image: Image.Image,
    donor_fields: dict,
    rng: random.Random,
    techniques: list[str] | None = None,
) -> tuple[Image.Image, str]:
    """Try tampering techniques in a random order and return the first that
    is applicable (some documents lack the fields a technique needs).

    `techniques` restricts which of TECHNIQUES to consider - e.g. the
    forgery-detection CNN's training set excludes copy_paste, since it
    relocates a patch *within the same image* and leaves no cross-source
    signal (different compression history, different resampling) for a
    classifier to find. (A classical self-similarity detector was tried as
    an alternative for this case - see copy_move.py - but didn't separate
    tampered from authentic on this document type either; copy_paste
    detection remains an open problem here, not solved by excluding it.)
    """
    order = (techniques if techniques is not None else TECHNIQUES)[:]
    rng.shuffle(order)
    for technique in order:
        if technique == "copy_paste":
            result = copy_paste(image, fields, rng)
        elif technique == "splice":
            result = splice(image, fields, donor_image, donor_fields, rng)
        else:
            result = photo_swap(image, fields, donor_image, donor_fields, rng)
        if result is not None:
            return result
    # Every document has a face_image box in this dataset, so this is
    # unreachable in practice, but fall back to a no-op crop-recompress
    # rather than raising.
    return _recompress(image, 60), "recompress_only"
