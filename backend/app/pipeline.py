"""Loads the Phase 1 models once at startup and runs the verification
pipeline (forgery detection + OCR field extraction) on an uploaded image."""
import logging

import torch
from PIL import Image

from src.common.config import MODELS_DIR
from src.forgery_detection.model import load_model as load_forgery_model
from src.forgery_detection.region_inference import predict_regions
from src.ocr.extract import DetectorFieldExtractor, DonutFieldExtractor
from src.ocr.validate import validate_fields

logger = logging.getLogger("ai_identity_shield")


class VerificationPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("verification pipeline device: %s", self.device)

        # Whole-image classification (src.forgery_detection.inference) was
        # tried five ways this session (raw RGB / ELA, two resolutions, two
        # data scales) and never beat chance - see project notes. This
        # crop-based classifier, fed regions from the same field detector
        # OCR uses below, found real signal instead (100% precision / 21.5%
        # recall on held-out validation crops) - it's the one wired in here.
        self.forgery_model = None
        checkpoint = MODELS_DIR / "forgery_resnet50_crop_raw.pt"
        if checkpoint.exists():
            self.forgery_model = load_forgery_model(checkpoint, self.device)
        else:
            logger.warning("no forgery checkpoint at %s - train it first (src/forgery_detection/train.py)", checkpoint)

        field_checkpoint = MODELS_DIR / "field_detector.pt"
        if field_checkpoint.exists():
            self.ocr_extractor = DetectorFieldExtractor(field_checkpoint, device=self.device)
        else:
            logger.warning(
                "no field-detector checkpoint at %s - falling back to DonutFieldExtractor "
                "(train src/field_detection/train.py for better OCR accuracy)",
                field_checkpoint,
            )
            self.ocr_extractor = DonutFieldExtractor(device=self.device)

    def run_forgery_detection(self, image: Image.Image) -> dict:
        if self.forgery_model is None:
            return {"error": "forgery detection model has not been trained yet"}
        if not isinstance(self.ocr_extractor, DetectorFieldExtractor):
            return {"error": "field detector required for region-based forgery detection is unavailable"}
        return predict_regions(
            image, self.ocr_extractor.model, self.forgery_model, self.device, preprocessing="raw"
        )

    def run_ocr(self, image: Image.Image) -> dict:
        raw_fields = self.ocr_extractor.extract_fields(image)
        return validate_fields(raw_fields)

    def verify(self, image: Image.Image) -> dict:
        forgery = self.run_forgery_detection(image)
        ocr = self.run_ocr(image)

        if forgery.get("error"):
            status = "UNKNOWN"
        elif forgery["label"] == "tampered":
            status = "FAILED"
        elif all(field["valid"] for field in ocr.values()):
            status = "VERIFIED"
        else:
            status = "FLAGGED"

        return {"forgery": forgery, "ocr": ocr, "status": status}
