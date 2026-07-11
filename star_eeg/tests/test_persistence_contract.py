import pytest

from star_eeg.training.persistence import (
    append_telemetry_row,
    atomic_write_bytes_no_overwrite,
    atomic_write_json_no_overwrite,
    claim_attempt_directory,
    freeze_completed_attempt,
    no_temporary_files,
    open_telemetry,
    tree_is_read_only,
    validate_telemetry_file,
)


def _telemetry_row(step):
    return {
        "step": step,
        "loss": 1.0,
        "encoder_grad_norm_before_clipping": 1.0,
        "encoder_grad_norm_after_clipping": 0.5,
        "model_grad_norm_before_clipping": 1.0,
        "model_grad_norm_after_clipping": 0.5,
        "temporary_head_grad_norm_before_clipping": 0.0,
        "temporary_head_grad_norm_after_clipping": 0.0,
        "parameter_delta_norm": 0.1,
        "learning_rate": 5e-4,
        "nan_inf_status": "PASS",
    }


def test_attempt_claim_and_atomic_write_never_overwrite(tmp_path):
    attempt = tmp_path / "cell" / "attempt_01"
    claim_attempt_directory(attempt, "attempt_01")
    payload = attempt / "payload.json"
    atomic_write_json_no_overwrite(payload, {"status": "first"})
    with pytest.raises(FileExistsError):
        atomic_write_json_no_overwrite(payload, {"status": "second"})
    with pytest.raises(FileExistsError):
        claim_attempt_directory(attempt, "attempt_01")
    assert no_temporary_files(attempt)


def test_telemetry_is_immediately_valid_jsonl(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    handle, digest = open_telemetry(path)
    with handle:
        first_hash = append_telemetry_row(handle, digest, _telemetry_row(1))
        assert path.read_text().count("\n") == 1
        final_hash = append_telemetry_row(handle, digest, _telemetry_row(2))
    audit = validate_telemetry_file(path, 2)
    assert audit["rows"] == 2
    assert audit["final_step"] == 2
    assert audit["sha256"] == final_hash
    assert first_hash != final_hash


def test_completed_attempt_tree_freezes_read_only(tmp_path):
    attempt = tmp_path / "cell" / "attempt_01"
    claim_attempt_directory(attempt, "attempt_01")
    checkpoint_dir = attempt / "checkpoints"
    checkpoint_dir.mkdir()
    atomic_write_bytes_no_overwrite(checkpoint_dir / "step_0005.pth", b"payload")
    atomic_write_json_no_overwrite(attempt / "completion.json", {"status": "COMPLETE"})
    freeze_completed_attempt(attempt)
    assert tree_is_read_only(attempt)
    for path in [attempt, *attempt.rglob("*")]:
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            path.chmod(0o644)
