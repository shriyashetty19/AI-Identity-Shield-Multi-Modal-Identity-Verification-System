"""Error Level Analysis (ELA): a classic image-forensics preprocessing step.

Re-compresses the image at a fixed JPEG quality and returns the amplified
pixel-wise difference from the input. Untouched regions settle into a stable
low error level after one more compression pass; content pasted in from a
different source image (splice, photo-swap) carries a different compression
history and re-quantizes differently, showing up as a brighter patch in the
difference. This is the standard first-line technique in the forgery-
detection literature for exposing exactly the kind of compression-history
mismatch `tampering.py`'s splice/photo-swap techniques leave behind - feeding
it to the classifier instead of raw RGB gives the model a signal to find
instead of forcing it to discover (or shortcut around) compression artifacts
on its own.
"""
import io

from PIL import Image, ImageChops

ELA_QUALITY = 90
ELA_SCALE = 15


def compute_ela(image: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=ELA_QUALITY)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    diff = ImageChops.difference(image.convert("RGB"), recompressed)
    return diff.point(lambda p: min(255, p * ELA_SCALE))
