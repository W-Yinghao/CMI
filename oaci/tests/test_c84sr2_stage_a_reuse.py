from pathlib import Path

from oaci.multidataset.c84s_common import read_json
from oaci.multidataset.c84sr1_common import write_stage_receipt
from oaci.multidataset.c84sr2_stage_a_replay import (
    replay_historical_stage_a, run_replay,
)


def test_historical_stage_a_replays_without_label_loader_or_evaluation_release():
    replay = replay_historical_stage_a()
    assert replay["status"] == "PASS"
    assert replay["files_replayed"] == 11
    assert replay["label_loader_calls"] == 0
    assert replay["target_label_rows_reloaded"] == 0
    assert replay["evaluation_descriptor_released_to_Stage_B"] is False


def test_stage_a_replay_publishes_only_an_identity_receipt(tmp_path: Path):
    receipt = tmp_path / "authorization_consumed.json"
    write_stage_receipt(receipt, {
        "schema_version": "test", "stage": "C84S_V4_authorization_consumed",
        "C84S_authorized": True,
    })
    root = tmp_path / "stage_a_replay"
    result = run_replay(receipt_path=receipt, output_root=root)
    assert result["status"] == "PASS"
    assert sorted(path.name for path in root.iterdir()) == ["C84S_STAGE_A_REPLAY.json"]
    frozen = read_json(root / "C84S_STAGE_A_REPLAY.json")
    assert frozen["label_loader_calls"] == 0
    assert "construction_handoff_path" in frozen
    assert "evaluation_seal_path" in frozen
