from collections import Counter

from star_eeg.data.anchor_batch_stream import (
    build_anchor_batches,
    build_exposure_matched_shuffled_manifest,
)
from star_eeg.data.faced_split_contract import canonical_hash


def _full_anchor_manifest():
    labels = [0] * 9 + [1] * 9 + [2] * 9 + [3] * 9 + [4] * 12 + [5] * 9 + [6] * 9 + [7] * 9 + [8] * 9
    records = [
        {
            "sample_id": f"sub{subject:03d}-{index:03d}",
            "subject": subject,
            "label": label,
            "split": "source_train",
        }
        for subject in range(1, 81)
        for index, label in enumerate(labels)
    ]
    core = {
        "split": "source_train",
        "n_records": len(records),
        "records": records,
    }
    return {**core, "anchor_manifest_hash": canonical_hash(core)}


def test_full_exposure_stream_is_exact_and_true_shuffled_marginals_match():
    anchor = _full_anchor_manifest()
    shuffled = build_exposure_matched_shuffled_manifest(anchor)
    batches, exposure_rows = build_anchor_batches(anchor, shuffled, model_seed=0)
    assert len(batches) == 750
    assert all(len(batch["sample_ids"]) == 64 for batch in batches)
    subject_counts = Counter(
        int(sample_id.split("-")[0][3:])
        for batch in batches
        for sample_id in batch["sample_ids"]
    )
    assert set(subject_counts.values()) == {600}
    assert all(row["marginal_equal"] for row in exposure_rows)
    assert shuffled["n_changed_labels"] > 0


def test_model_seed_changes_order_not_semantic_manifest():
    anchor = _full_anchor_manifest()
    shuffled = build_exposure_matched_shuffled_manifest(anchor)
    s0, _ = build_anchor_batches(anchor, shuffled, model_seed=0)
    s1, _ = build_anchor_batches(anchor, shuffled, model_seed=1)
    assert [row["x_id_hash"] for row in s0] != [row["x_id_hash"] for row in s1]
    assert shuffled["same_semantic_manifest_for_model_seeds"] == [0, 1]
