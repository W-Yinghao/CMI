import inspect

import pytest

from star_eeg.data.anchor_manifest import AnchorRecord, build_anchor_manifest
from star_eeg.objectives.task_anchor import source_task_anchor_step
from star_eeg.red_team.target_label_quarantine import inspect_training_signatures


def test_target_y_absent_from_training_function_signatures():
    result = inspect_training_signatures()
    assert result["status"] == "PASS"
    parameters = inspect.signature(source_task_anchor_step).parameters
    assert "source_y" in parameters
    assert "target_y" not in parameters


def test_anchor_manifest_rejects_source_val_and_target_rows():
    for subject, split in ((81, "source_val"), (101, "target_test")):
        with pytest.raises(PermissionError):
            build_anchor_manifest(
                [AnchorRecord(sample_id=f"bad-{subject}", subject=subject, label=0, split=split)],
                dataset_manifest_hash="0" * 64,
                preview_only=True,
            )
