"""Basic format validation for OCR-extracted ID fields. This checks *shape*
(does it look like a name/date/document number), not correctness against any
ground truth - there is none available for a freshly uploaded document."""
import re

from dateutil import parser as date_parser

NAME_RE = re.compile(r"^[A-Za-z][A-Za-z .,'\-]{1,60}$")
DOCUMENT_NUMBER_RE = re.compile(r"^[A-Z0-9]{5,12}$")


def validate_name(text: str | None) -> dict:
    if not text:
        return {"raw": text, "valid": False, "reason": "empty"}
    cleaned = text.strip()
    valid = bool(NAME_RE.match(cleaned))
    return {"raw": text, "valid": valid, "reason": None if valid else "does not look like a name"}


def validate_date(text: str | None) -> dict:
    if not text:
        return {"raw": text, "valid": False, "reason": "empty", "normalized": None}
    try:
        parsed = date_parser.parse(text.strip(), dayfirst=True, fuzzy=True)
    except (ValueError, OverflowError):
        return {"raw": text, "valid": False, "reason": "unparseable date", "normalized": None}
    return {"raw": text, "valid": True, "reason": None, "normalized": parsed.date().isoformat()}


def validate_document_number(text: str | None) -> dict:
    if not text:
        return {"raw": text, "valid": False, "reason": "empty"}
    cleaned = re.sub(r"[\s\-]", "", text.strip().upper())
    valid = bool(DOCUMENT_NUMBER_RE.match(cleaned))
    return {
        "raw": text,
        "valid": valid,
        "reason": None if valid else "does not match expected alphanumeric pattern",
        "normalized": cleaned if valid else None,
    }


def validate_fields(fields: dict) -> dict:
    return {
        "name": validate_name(fields.get("name")),
        "date_of_birth": validate_date(fields.get("date_of_birth")),
        "document_number": validate_document_number(fields.get("document_number")),
    }
