from src.ocr.validate import validate_date, validate_document_number, validate_name


def test_validate_name_accepts_plausible_names():
    assert validate_name("John Smith")["valid"] is True
    assert validate_name("O'Brien-Jones")["valid"] is True


def test_validate_name_rejects_empty_or_numeric():
    assert validate_name(None)["valid"] is False
    assert validate_name("")["valid"] is False
    assert validate_name("12345")["valid"] is False


def test_validate_date_parses_common_formats():
    result = validate_date("12/05/1990")
    assert result["valid"] is True
    assert result["normalized"] == "1990-05-12"


def test_validate_date_rejects_garbage():
    result = validate_date("not a date")
    assert result["valid"] is False
    assert result["normalized"] is None


def test_validate_document_number_accepts_alphanumeric():
    result = validate_document_number("AB123456")
    assert result["valid"] is True
    assert result["normalized"] == "AB123456"


def test_validate_document_number_rejects_too_short_or_symbols():
    assert validate_document_number("AB")["valid"] is False
    assert validate_document_number("!!invalid!!")["valid"] is False
