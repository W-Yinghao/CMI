from pathlib import Path

import pytest

from star_eeg.artifacts.close_star01a_finals import _write_completion_csv
from star_eeg.training.approval_lock import APPROVED_CELLS


def test_completion_matrix_uses_lf_and_never_overwrites(tmp_path):
    path = tmp_path / "matrix.csv"
    _write_completion_csv(path, [{"cell": APPROVED_CELLS[0], "status": "PASS"}])
    assert b"\r" not in path.read_bytes()
    assert path.read_text().splitlines() == ["cell,status", f"{APPROVED_CELLS[0]},PASS"]
    with pytest.raises(FileExistsError):
        _write_completion_csv(
            path, [{"cell": APPROVED_CELLS[1], "status": "PASS"}]
        )


def test_afterok_submitter_and_closure_artifact_names_are_frozen():
    repo_root = Path(__file__).resolve().parents[2]
    submitter = (repo_root / "star_eeg/runners/submit_star01a_blind_chain.py").read_text()
    closure = (repo_root / "star_eeg/artifacts/close_star01a_finals.py").read_text()
    assert "--dependency=afterok:" in submitter
    assert "star01_final_checkpoint_manifest.json" in closure
    assert "star01_training_completion_matrix.csv" in closure
    assert "star01_closure_go_nogo.json" in closure
    assert "target_scoring_allowed\": False" in closure


def test_exact_six_cell_closure_universe():
    assert APPROVED_CELLS == (
        "H200_SSL_CONT_s0",
        "H200_SSL_CONT_s1",
        "H200_STAR_TRUE_s0",
        "H200_STAR_TRUE_s1",
        "H200_STAR_SHUFFLED_s0",
        "H200_STAR_SHUFFLED_s1",
    )
