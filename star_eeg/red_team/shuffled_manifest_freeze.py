"""Red-team checks for the fixed within-subject shuffled control."""

from typing import Dict, Mapping

from star_eeg.data.shuffled_label_manifest import (
    build_shuffled_manifest,
    validate_frozen_shuffled_manifest,
)


def evaluate_shuffled_manifest(
    anchor_manifest: Mapping[str, object],
    shuffled_manifest: Mapping[str, object],
    permutation_seed: int,
) -> Dict[str, object]:
    validate_frozen_shuffled_manifest(shuffled_manifest)
    repeated = build_shuffled_manifest(anchor_manifest, permutation_seed)
    checks = {
        "fixed_manifest_hash": repeated["shuffled_manifest_hash"] == shuffled_manifest.get("shuffled_manifest_hash"),
        "source_train_only": shuffled_manifest.get("split") == "source_train",
        "source_val_excluded": shuffled_manifest.get("source_val_participated") is False,
        "target_test_excluded": shuffled_manifest.get("target_test_participated") is False,
        "within_subject_histograms_preserved": all(
            pair.get("before") == pair.get("after")
            for pair in shuffled_manifest.get("within_subject_histograms", {}).values()
        ),
        "no_epoch_reshuffle": shuffled_manifest.get("reshuffle_each_epoch") is False,
        "permutation_rng_separate": shuffled_manifest.get("permutation_rng_separate_from_training_rng") is True,
        "same_semantic_contract_both_model_seeds": shuffled_manifest.get("same_semantic_manifest_for_model_seeds") == [0, 1],
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "shuffled_manifest_hash": shuffled_manifest.get("shuffled_manifest_hash"),
    }
