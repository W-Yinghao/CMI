import hashlib
import json
from pathlib import Path

import pytest

from oaci.multidataset import c84fr2_lock as lock_generator
from oaci.multidataset import c84fr2_runtime_guard as runtime


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "oaci/reports"


def test_lock_generator_protocol_ancestry_helpers():
    assert lock_generator.git_is_ancestor(lock_generator.PROTOCOL_COMMIT, "HEAD")
    assert lock_generator.git_is_ancestor(lock_generator.TARGET_PROTOCOL_COMMIT, "HEAD")
    assert not lock_generator.git_is_ancestor("0" * 40, "HEAD")


def test_runtime_fails_closed_when_v2_lock_is_absent(tmp_path):
    with pytest.raises(runtime.C84FR2RuntimeError, match="execution lock is absent"):
        runtime.require_authorization_and_lock(
            lock_path=tmp_path / "missing.json",
            lock_sha_path=tmp_path / "missing.sha256",
            authorization_path=tmp_path / "authorization.json",
            output_root=tmp_path / "output",
        )


def test_v2_lock_is_self_consistent_and_not_authorized():
    path = REPORTS / "C84F_TARGET_STAGE_EXECUTION_LOCK_V2.json"
    sidecar = REPORTS / "C84F_TARGET_STAGE_EXECUTION_LOCK_V2.sha256"
    expected = sidecar.read_text(encoding="ascii").split()[0]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    lock = json.loads(path.read_text(encoding="utf-8"))
    assert lock["status"] == "LOCKED_READY_FOR_DIRECT_PI_REAUTHORIZATION_NOT_AUTHORIZED"
    assert lock["implementation"]["entrypoint"].endswith(
        "oaci.multidataset.c84fr2_target_stage run-real"
    )
    assert lock["implementation"]["target_stage_training_callable"] is False
    assert lock["numerical_gates"]["same_backend_GPU_PyTorch_float32_max_abs"] == 1e-6
    assert lock["numerical_gates"]["historical_2e5_widened"] is False
    assert lock["numerical_gates"]["cross_backend_diagnostic_only"] is True
    assert lock["schemas"] == {
        "target_artifact": "c84f_target_unlabeled_v2",
        "context_digest_sidecar": "c84f_target_context_and_digest_index_v2",
        "complete_field_manifest": "c84f_complete_field_manifest_v2",
    }
    assert len(lock["frozen_target_input_source"]["partial_target_objects"]) == 11
    assert lock["frozen_target_input_source"]["partial_target_artifacts_reusable"] is False
    assert lock["authorization"]["record_present_at_lock"] is False
    assert all(lock["forbidden"].values())


def test_runtime_bound_registry_covers_target_only_transitive_dependencies():
    required = {
        "oaci/multidataset/c84f_target_instrumentation.py",
        "oaci/multidataset/c84f_field_manifest.py",
        "oaci/multidataset/c84f_runtime_guard.py",
        "oaci/multidataset/c84fr1_runtime_guard.py",
        "oaci/multidataset/c84fr2_target_numerical_replay.py",
        "oaci/multidataset/c84fr2_target_stage.py",
        "oaci/multidataset/c84fr2_runtime_guard.py",
        "oaci/models/factory.py",
        "oaci/models/shallow.py",
    }
    assert required <= set(lock_generator.IMPLEMENTATION_FILES)


def test_runtime_replays_frozen_inputs_before_authorization():
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    function = source[source.index("def require_authorization_and_lock"):]
    assert function.index("verify_frozen_model_and_target_inputs") < function.index(
        "verify_authorization_record"
    )


def test_no_C84S_execution_lock_was_created():
    assert not (REPORTS / "C84S_EXECUTION_LOCK.json").exists()
    assert not (REPORTS / "C84S_EXECUTION_LOCK.sha256").exists()
