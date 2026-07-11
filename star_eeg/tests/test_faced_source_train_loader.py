import pytest

from star_eeg.data.faced_source_train_loader import parse_source_key


def test_source_key_parser_accepts_only_subjects_1_through_80():
    assert parse_source_key("sub000.pkl-0-0")["subject"] == 1
    assert parse_source_key("sub079.pkl-27-2")["subject"] == 80
    with pytest.raises(PermissionError):
        parse_source_key("sub080.pkl-0-0")
    with pytest.raises(PermissionError):
        parse_source_key("sub100.pkl-0-0")


def test_source_key_parser_rejects_unexpected_key_shape():
    with pytest.raises(ValueError):
        parse_source_key("test-subject-101")
