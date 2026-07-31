import random

from PIL import Image

from src.forgery_detection.tampering import apply_random_tampering, copy_paste, photo_swap, splice

FIELDS = {
    "date_of_birth": [[10, 10, 20, 8]],
    "document_number": [[10, 30, 25, 8]],
    "face_image": [[5, 45, 15, 15]],
}


def _blank(size=(64, 64), color=(200, 200, 200)):
    return Image.new("RGB", size, color)


def test_copy_paste_preserves_image_size():
    image = _blank()
    result = copy_paste(image, FIELDS, random.Random(0))
    assert result is not None
    tampered, technique = result
    assert tampered.size == image.size
    assert technique.startswith("copy_paste:")


def test_splice_pastes_donor_field():
    image = _blank(color=(200, 200, 200))
    donor = _blank(color=(10, 10, 10))
    result = splice(image, FIELDS, donor, FIELDS, random.Random(1))
    assert result is not None
    tampered, technique = result
    assert tampered.size == image.size
    assert technique.startswith("splice:")


def test_photo_swap_requires_face_image_field():
    image = _blank()
    donor = _blank(color=(10, 10, 10))
    assert photo_swap(image, {}, donor, FIELDS, random.Random(2)) is None
    result = photo_swap(image, FIELDS, donor, FIELDS, random.Random(2))
    assert result is not None
    assert result[1] == "photo_swap"


def test_apply_random_tampering_always_returns_a_result():
    image = _blank()
    donor = _blank(color=(50, 50, 50))
    for seed in range(10):
        tampered, technique = apply_random_tampering(image, FIELDS, donor, FIELDS, random.Random(seed))
        assert tampered.size == image.size
        assert technique
