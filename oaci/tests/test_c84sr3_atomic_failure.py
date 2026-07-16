from pathlib import Path
import shutil

import pytest

from oaci.multidataset.c84s_common import (
    C84SContractError, atomic_publish_directory,
)
from oaci.multidataset.c84sr1_common import write_stage_receipt
from oaci.multidataset.c84sr1_synthetic import (
    synthetic_contexts, synthetic_label_rows,
)
from oaci.multidataset.c84sr1_stage_a_labels import run_stage_a_from_rows
from oaci.multidataset.c84sr3_stage_a_replay import run_replay
from oaci.multidataset.c84sr3_stage_b_selection import run_stage_b


def test_cleanup_failure_never_masks_primary_exception(tmp_path, monkeypatch):
    final = tmp_path / "final"
    original_rmtree = shutil.rmtree
    holder = {}

    def writer(staging: Path):
        handle = (staging / "open.csv").open("w", encoding="utf-8")
        holder["handle"] = handle
        handle.write("partial\n")
        raise C84SContractError("PRIMARY_SELECTION_FAILURE")

    writer.cleanup_on_failure = lambda: holder["handle"].close()

    def fail_rmtree(_path):
        raise OSError(39, "Directory not empty")

    monkeypatch.setattr(shutil, "rmtree", fail_rmtree)
    with pytest.raises(C84SContractError, match="PRIMARY_SELECTION_FAILURE") as caught:
        atomic_publish_directory(final, writer)
    assert not final.exists()
    assert holder["handle"].closed
    assert any("staging cleanup failed" in note for note in caught.value.__notes__)
    residuals = list(tmp_path.glob(".final.staging-*"))
    assert len(residuals) == 1
    original_rmtree(residuals[0])


def test_injected_stage_b_failure_publishes_no_final_root(tmp_path):
    registry, labels, _ = synthetic_label_rows()
    stage_a_root = tmp_path / "stage_a"
    run_stage_a_from_rows(
        guard_receipt={"C84S_authorized": True},
        frozen_registry_rows=registry, label_rows=labels,
        output_root=stage_a_root,
    )
    final = tmp_path / "selection"
    with pytest.raises(RuntimeError, match="injected C84SR3 Stage-B context failure"):
        run_stage_b(
            stage_a_handoff_path=stage_a_root / "C84S_STAGE_A_HANDOFF.json",
            final_root=final, contexts=synthetic_contexts(), chains=8,
            synthetic=True, failure_injection_context=0,
        )
    assert not final.exists()
    assert not list(tmp_path.glob(".selection.staging-*"))


def test_v5_stage_a_replay_requires_and_records_v5_receipt(tmp_path, monkeypatch):
    receipt = tmp_path / "authorization_consumed.json"
    write_stage_receipt(receipt, {
        "schema_version": "test", "stage": "C84S_V5_authorization_consumed",
        "C84S_authorized": True, "historical_V4_authorization_reused": False,
    })
    monkeypatch.setattr(
        "oaci.multidataset.c84sr3_stage_a_replay.replay_historical_stage_a",
        lambda: {
            "status": "PASS", "label_loader_calls": 0,
            "target_label_rows_reloaded": 0,
            "evaluation_descriptor_released_to_Stage_B": False,
        },
    )
    output = tmp_path / "replay"
    result = run_replay(receipt_path=receipt, output_root=output)
    assert result["schema_version"] == "c84sr3_immutable_stage_a_replay_v1"
    assert (output / "C84S_STAGE_A_REPLAY.json").is_file()

