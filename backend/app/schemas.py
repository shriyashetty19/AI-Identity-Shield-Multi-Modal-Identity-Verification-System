from pydantic import BaseModel


class RegionResult(BaseModel):
    category: str
    label: str
    confidence: float


class ForgeryResult(BaseModel):
    label: str | None = None
    confidence: float | None = None
    error: str | None = None
    regions: list[RegionResult] | None = None


class FieldResult(BaseModel):
    raw: str | None = None
    valid: bool = False
    reason: str | None = None
    normalized: str | None = None


class OCRResult(BaseModel):
    name: FieldResult
    date_of_birth: FieldResult
    document_number: FieldResult


class VerificationResponse(BaseModel):
    filename: str
    forgery: ForgeryResult
    ocr: OCRResult | None = None
    status: str
