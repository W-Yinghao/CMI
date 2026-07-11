import pytest

from star_eeg.config import STAR01
from star_eeg.data.anchor_manifest import build_anchor_manifest, synthetic_preview_records
from star_eeg.data.shuffled_label_manifest import (
    build_shuffled_manifest,
    validate_frozen_shuffled_manifest,
)


def _anchor():
    return build_anchor_manifest(
        synthetic_preview_records(),
        dataset_manifest_hash="1" * 64,
        preview_only=True,
    )


def test_fixed_shuffle_is_deterministic_source_only_and_histogram_preserving():
    anchor = _anchor()
    first = build_shuffled_manifest(anchor, STAR01.permutation_seed)
    second = build_shuffled_manifest(anchor, STAR01.permutation_seed)
    assert first == second
    assert first["source_val_participated"] is False
    assert first["target_test_participated"] is False
    assert all(pair["before"] == pair["after"] for pair in first["within_subject_histograms"].values())
    validate_frozen_shuffled_manifest(first)


def test_frozen_hash_detects_mutation():
    manifest = build_shuffled_manifest(_anchor(), STAR01.permutation_seed)
    manifest["records"][0]["shuffled_label"] = (manifest["records"][0]["shuffled_label"] + 1) % 9
    with pytest.raises(ValueError):
        validate_frozen_shuffled_manifest(manifest)
