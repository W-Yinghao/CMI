import pytest

from star_eeg.data.faced_split_contract import (
    FACED_SPLITS,
    assert_exact_split,
    assert_source_train_subject,
    contract_payload,
    role_for_subject,
)


def test_exact_faced_subject_firewall():
    assert_exact_split(FACED_SPLITS)
    assert role_for_subject(1) == "source_train"
    assert role_for_subject(80) == "source_train"
    assert role_for_subject(81) == "source_val"
    assert role_for_subject(100) == "source_val"
    assert role_for_subject(101) == "target_test"
    assert role_for_subject(123) == "target_test"
    payload = contract_payload()
    assert payload["expected_segments"] == {
        "source_train": 6720,
        "source_val": 1680,
        "target_test": 1932,
    }
    assert len(payload["faced_split_hash"]) == 64


def test_non_source_subject_rejected_from_anchor():
    with pytest.raises(PermissionError):
        assert_source_train_subject(81)
    with pytest.raises(PermissionError):
        assert_source_train_subject(101)
