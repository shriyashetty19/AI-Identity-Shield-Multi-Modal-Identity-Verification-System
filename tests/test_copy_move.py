import numpy as np
from PIL import Image

from src.forgery_detection.copy_move import detect_copy_move


def _textured_image(seed=0, size=256):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_detects_a_duplicated_patch():
    image = _textured_image()
    patch = image.crop((10, 10, 100, 100))
    tampered = image.copy()
    tampered.paste(patch, (140, 140))

    result = detect_copy_move(tampered)
    assert result["detected"] is True
    assert result["match_count"] >= 6


def test_does_not_flag_an_untampered_image():
    image = _textured_image()
    result = detect_copy_move(image)
    assert result["detected"] is False
