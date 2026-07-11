import pytest

from star_eeg.training import approval_lock
from star_eeg.training.persistence import atomic_write_json_no_overwrite


def _fake_repo(tmp_path, monkeypatch):
    commit = "a" * 40
    for label, relative in approval_lock.ARTIFACT_BINDINGS.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if label == "training_tasks_csv":
            rows = ["task_id,variant,seed"]
            for task_id, cell in enumerate(approval_lock.APPROVED_CELLS):
                variant, seed = cell.rsplit("_s", 1)
                rows.append(f"{task_id},{variant},{seed}")
            path.write_text("\n".join(rows) + "\n")
        else:
            path.write_text(f"{label}\n")

    def fake_git(_repo_root, *args):
        if args == ("rev-parse", "HEAD"):
            return commit
        if args == ("branch", "--show-current"):
            return "project/star-task-anchor"
        if args == ("status", "--porcelain", "--untracked-files=no"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(approval_lock, "_git", fake_git)
    return commit


def test_approval_lock_binds_commit_cells_and_artifacts(tmp_path, monkeypatch):
    commit = _fake_repo(tmp_path, monkeypatch)
    payload = approval_lock.build_approval_payload(
        tmp_path,
        approved_execution_commit=commit,
        approved_attempt_id="attempt_01",
        issued_at_utc="2026-07-11T00:00:00+00:00",
    )
    manifest_hash = payload[approval_lock.APPROVAL_HASH_FIELD]
    path = tmp_path / f"star01a_approval.{manifest_hash}.json"
    atomic_write_json_no_overwrite(path, payload)
    path.chmod(0o444)
    audit = approval_lock.validate_approval_lock(
        path,
        tmp_path,
        variant="H200_STAR_TRUE",
        model_seed=0,
    )
    assert audit["status"] == "PASS"
    assert audit["approved_execution_commit"] == commit
    assert tuple(audit["approved_cells"]) == approval_lock.APPROVED_CELLS
    assert audit["target_scoring"] == "BLOCKED"


def test_approval_lock_fails_if_bound_artifact_changes(tmp_path, monkeypatch):
    commit = _fake_repo(tmp_path, monkeypatch)
    payload = approval_lock.build_approval_payload(
        tmp_path,
        approved_execution_commit=commit,
        approved_attempt_id="attempt_01",
        issued_at_utc="2026-07-11T00:00:00+00:00",
    )
    manifest_hash = payload[approval_lock.APPROVAL_HASH_FIELD]
    path = tmp_path / f"star01a_approval.{manifest_hash}.json"
    atomic_write_json_no_overwrite(path, payload)
    path.chmod(0o444)
    changed = tmp_path / approval_lock.ARTIFACT_BINDINGS["runner_source"]
    changed.write_text("changed\n")
    with pytest.raises(PermissionError):
        approval_lock.validate_approval_lock(path, tmp_path)


def test_required_pm_binding_fields_are_present():
    required = {
        "h200_immutable_manifest",
        "anchor_manifest",
        "shuffled_manifest",
        "compute_match_contract",
        "training_tasks_csv",
        "runner_source",
        "slurm_source",
    }
    assert required <= set(approval_lock.ARTIFACT_BINDINGS)
    assert len(approval_lock.APPROVED_CELLS) == 6


def test_formal_array_environment_is_exact_and_cell_bound():
    environment = {
        "SLURM_JOB_ID": "123_2",
        "SLURM_ARRAY_JOB_ID": "123",
        "SLURM_ARRAY_TASK_ID": "2",
        "SLURM_ARRAY_TASK_COUNT": "6",
        "SLURM_ARRAY_TASK_MIN": "0",
        "SLURM_ARRAY_TASK_MAX": "5",
        "STAR_ATTEMPT_ID": "attempt_01",
    }
    audit = approval_lock.validate_six_cell_array_environment(
        "H200_STAR_TRUE", 0, environment
    )
    assert audit["array_task_id"] == 2
    assert audit["expected_cell"] == "H200_STAR_TRUE_s0"
    with pytest.raises(PermissionError):
        approval_lock.validate_six_cell_array_environment(
            "H200_STAR_TRUE", 1, environment
        )
    with pytest.raises(PermissionError):
        approval_lock.validate_six_cell_array_environment(
            "H200_STAR_TRUE", 0, {**environment, "SLURM_ARRAY_TASK_COUNT": "5"}
        )
