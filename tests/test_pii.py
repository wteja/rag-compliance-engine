from app.pii import redact, redact_with_counts


def test_redact_with_counts_masks_and_counts_email():
    clean, counts = redact_with_counts("Contact me at jane@acme.com please.")
    assert "jane@acme.com" not in clean
    assert counts.get("EMAIL_ADDRESS") == 1


def test_redact_with_counts_clean_text_returns_empty_dict():
    clean, counts = redact_with_counts("This project is progressing well.")
    assert clean == "This project is progressing well."
    assert counts == {}


def test_redact_returns_only_text():
    out = redact("Call 415-555-0199 now.")
    assert isinstance(out, str)
    assert "415-555-0199" not in out
