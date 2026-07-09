from __future__ import annotations

from tta_mech_eeg.normalization.artifact_inventory import build_bn_artifact_inventory


def test_bn_artifact_inventory_marks_current_cedar01f_as_not_feasible():
    payload = build_bn_artifact_inventory()
    assert payload["summary"]["total_records"] == 18
    assert payload["summary"]["feature_artifact_hashes_match_handoff"] is True
    assert payload["summary"]["ready_records"] == 0
    assert payload["summary"]["rejected_records"] == 18
    assert payload["summary"]["has_any_model_checkpoint"] is False
    assert payload["summary"]["has_any_bn_buffers"] is False
    assert payload["summary"]["has_any_raw_or_preprocessed_input"] is False
    assert payload["summary"]["has_any_forward_ready_artifact"] is False
    assert payload["summary"]["feasibility"] == "TTA_MECH_02B_NOT_FEASIBLE_FROM_CURRENT_ARTIFACTS"
    assert all(row["has_source_split"] for row in payload["records"])
    assert all(row["status"] == "REJECT" for row in payload["records"])
