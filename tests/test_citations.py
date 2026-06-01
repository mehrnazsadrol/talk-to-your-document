from frontend.components.citations import format_locator, format_timestamp


def test_format_timestamp_zero():
    assert format_timestamp(0) == "00:00"


def test_format_timestamp_truncates_fractional_seconds():
    assert format_timestamp(65.4) == "01:05"


def test_format_timestamp_widens_past_one_hour():
    assert format_timestamp(3725) == "1:02:05"


def test_format_locator_audio_uses_en_dash():
    metadata = {"source_type": "audio", "start_seconds": 12, "end_seconds": 34}
    assert format_locator(metadata) == "00:12–00:34"


def test_format_locator_pdf_with_page_number():
    assert format_locator({"source_type": "pdf", "page_number": 7}) == "p. 7"


def test_format_locator_pdf_without_page_number_returns_empty():
    assert format_locator({"source_type": "pdf"}) == ""
