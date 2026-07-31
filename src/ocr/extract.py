"""OCR field extraction.

Three extractors:

- `TrOCRFieldExtractor` recognizes text from a *known crop* of a document
  (e.g. the name/DOB/document-number boxes annotated in MIDV-2020). Used in
  the Phase 1 notebook to demonstrate/inspect OCR quality against the
  dataset's own field annotations. Currently broken in this environment -
  `microsoft/trocr-base-printed`'s tokenizer fails to load under
  `transformers==5.14.1` (a real library incompatibility, not a missing
  dependency - see project notes).

- `DonutFieldExtractor` answers free-form questions about a *whole* document
  image with no bounding boxes needed (DocVQA-style prompting). Measured
  against a 15-sample hand-labeled ground truth: document_number 87%, name
  40%, date_of_birth 7% - the model can't reliably tell apart the
  document's three near-identical date fields when shown the whole image
  with no idea where to look.

- `DetectorFieldExtractor` fixes that root cause: it localizes each field
  with the trained `src.field_detection` model first, crops, and reads the
  crop with easyocr - detect-then-read instead of whole-image zero-shot
  prompting, removing the field-disambiguation ambiguity Donut can't
  handle. This is what the FastAPI backend uses at inference time.
"""
import re

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from transformers import DonutProcessor, TrOCRProcessor, VisionEncoderDecoderModel

from src.common.coco_utils import largest_box

TROCR_MODEL = "microsoft/trocr-base-printed"
DONUT_MODEL = "naver-clova-ix/donut-base-finetuned-docvqa"

# category name in the MIDV-2020 annotations -> field name we expose
FIELD_BOXES = {
    "name": "primary_identifier",
    "date_of_birth": "date_of_birth",
    "document_number": "document_number",
}

DONUT_QUESTIONS = {
    "name": "What is the surname of the passport holder?",
    "date_of_birth": "What is the date of birth shown on the passport?",
    "document_number": "What is the document number?",
}


class TrOCRFieldExtractor:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
        self.model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL).to(device)
        self.model.eval()

    def read(self, crop: Image.Image) -> str:
        pixel_values = self.processor(images=crop, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values, max_new_tokens=32)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def extract_fields_from_boxes(self, image: Image.Image, fields: dict) -> dict:
        results = {}
        for field_name, category in FIELD_BOXES.items():
            if category not in fields:
                results[field_name] = None
                continue
            x, y, w, h = largest_box(fields[category])
            crop = image.crop((int(x), int(y), int(x + w), int(y + h)))
            results[field_name] = self.read(crop)
        return results


class DonutFieldExtractor:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.processor = DonutProcessor.from_pretrained(DONUT_MODEL)
        self.model = VisionEncoderDecoderModel.from_pretrained(DONUT_MODEL).to(device)
        self.model.eval()

    def ask(self, image: Image.Image, question: str) -> str:
        task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
        decoder_input_ids = self.processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=self.model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(
            self.processor.tokenizer.pad_token, ""
        )
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
        parsed = self.processor.token2json(sequence)
        return parsed.get("answer", "").strip()

    def extract_fields(self, image: Image.Image) -> dict:
        return {field: self.ask(image, question) for field, question in DONUT_QUESTIONS.items()}


class DetectorFieldExtractor:
    """Detect each field's location with the trained src.field_detection
    model, crop it, and read the crop with easyocr - detect-then-read
    instead of DonutFieldExtractor's whole-image zero-shot prompting."""

    LABEL_TO_FIELD = {
        "primary_identifier": "name",
        "date_of_birth": "date_of_birth",
        "document_number": "document_number",
    }

    def __init__(self, checkpoint_path, device: str = "cpu", score_threshold: float = 0.5):
        from src.field_detection.model import load_model

        self.device = device
        self.model = load_model(checkpoint_path, device)
        self.score_threshold = score_threshold

        import easyocr

        self.reader = easyocr.Reader(["en"], gpu=(device == "cuda"))

    def _read(self, crop: Image.Image) -> str | None:
        result = self.reader.readtext(np.array(crop))
        if not result:
            return None
        return " ".join(r[1] for r in result).strip()

    # Detected boxes fit the annotated field tightly - just tight enough that
    # the crop sometimes clips the first character (measured: document_number
    # readings were misreading a clipped "C" as "6" in most failures). A
    # small margin gives easyocr the full glyph without meaningfully
    # including neighboring content in these single-line/single-field boxes.
    BOX_PADDING = 8

    @torch.no_grad()
    def extract_fields(self, image: Image.Image) -> dict:
        from src.field_detection.model import LABELS

        input_tensor = TF.to_tensor(image).to(self.device)
        pred = self.model([input_tensor])[0]

        results = {field: None for field in self.LABEL_TO_FIELD.values()}
        # torchvision detection models return predictions score-sorted
        # descending, so the first box seen per label is the best one.
        for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            if score < self.score_threshold:
                continue
            field = self.LABEL_TO_FIELD.get(LABELS[label.item()])
            if field is None or results[field] is not None:
                continue
            x0, y0, x1, y1 = (int(v) for v in box.tolist())
            x0 = max(0, x0 - self.BOX_PADDING)
            y0 = max(0, y0 - self.BOX_PADDING)
            x1 = min(image.width, x1 + self.BOX_PADDING)
            y1 = min(image.height, y1 + self.BOX_PADDING)
            results[field] = self._read(image.crop((x0, y0, x1, y1)))
        return results
