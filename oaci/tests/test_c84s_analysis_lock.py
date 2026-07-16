from __future__ import annotations

import ast
from pathlib import Path

import pytest

from oaci.multidataset import c84s_runtime_guard as runtime
from oaci.multidataset.c84s_common import C84SContractError, read_json


def test_frozen_protocol_and_complete_field_replay() -> None:
    replay = runtime.verify_protocol_inputs()
    assert replay["hashes"]["complete_field"] == runtime.EXPECTED["complete_field"]
    assert len(replay["manifest"]["field_descriptors"]) == 1944


def test_no_authorization_record_exists_at_readiness() -> None:
    assert not runtime.AUTHORIZATION_PATH.exists()


def test_static_process_isolation_passes() -> None:
    checks = runtime.static_isolation_audit()
    assert len(checks) == len(runtime.IMPLEMENTATION_PATHS) + 3
    assert all(row["pass"] for row in checks)


def test_authorization_fails_closed_when_absent() -> None:
    with pytest.raises(C84SContractError, match="authorization record is absent"):
        runtime.verify_authorization_record({"status": runtime.LOCK_READY_STATUS}, "a" * 64)


def test_execution_lock_when_present_binds_scope() -> None:
    if not runtime.LOCK_PATH.exists():
        pytest.skip("analysis execution lock is generated after implementation commit")
    digest = runtime.verify_lock_self()
    lock = read_json(runtime.LOCK_PATH)
    assert len(digest) == 64
    assert lock["status"] == runtime.LOCK_READY_STATUS
    assert lock["external_field"]["target_artifacts"] == 1944
    assert lock["external_field"]["context_digest_sidecars"] == 1944
    assert lock["authorization"]["record_present_at_lock"] is False
    assert lock["forbidden"]["training"] is True
    assert lock["forbidden"]["forward"] is True
    assert lock["forbidden"]["GPU"] is True


def test_no_C84S_module_imports_training_or_GPU_packages() -> None:
    forbidden = {"torch", "mne", "moabb", "cupy"}
    for relative in runtime.IMPLEMENTATION_PATHS:
        tree = ast.parse((runtime.REPO_ROOT / relative).read_text(encoding="utf-8"))
        imports = {
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        }
        assert not any(name.split(".")[0] in forbidden for name in imports)

