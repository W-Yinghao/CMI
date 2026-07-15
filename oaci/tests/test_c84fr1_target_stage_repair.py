import ast
import hashlib
import json
from pathlib import Path

import pytest

from oaci.multidataset import c84f_target_instrumentation as target_stage
from oaci.multidataset import c84fr1_lock as lock_generator
from oaci.multidataset import c84fr1_runtime_guard as runtime
from oaci.multidataset import c84fr1_target_stage_repair as repair


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "oaci/reports"


def _raw(path, byte_count, digest, order=("path", "bytes", "sha256")):
    values = {"path": path, "bytes": byte_count, "sha256": digest}
    return {key: values[key] for key in order}


def test_raw_file_rows_ignore_dictionary_insertion_order():
    digest_a = "a" * 64
    digest_b = "b" * 64
    first = [
        _raw("/z.edf", 20, digest_b, ("sha256", "path", "bytes")),
        _raw("/a.edf", 10, digest_a, ("bytes", "sha256", "path")),
    ]
    second = [
        _raw("/a.edf", 10, digest_a),
        _raw("/z.edf", 20, digest_b),
    ]
    assert target_stage.canonical_raw_file_rows(first) == target_stage.canonical_raw_file_rows(second)
    assert [row["path"] for row in target_stage.canonical_raw_file_rows(first)] == [
        "/a.edf", "/z.edf",
    ]


@pytest.mark.parametrize(
    "row",
    [
        {"path": "/a.edf", "bytes": 10},
        {"path": "/a.edf", "bytes": 10, "sha256": "a" * 64, "extra": 1},
        {"path": "", "bytes": 10, "sha256": "a" * 64},
        {"path": "/a.edf", "bytes": 0, "sha256": "a" * 64},
        {"path": "/a.edf", "bytes": 10, "sha256": "invalid"},
    ],
)
def test_raw_file_rows_fail_closed_on_schema_or_identity_drift(row):
    with pytest.raises(target_stage.C84FTargetInstrumentationError):
        target_stage.canonical_raw_file_rows([row])


def test_raw_manifest_payload_has_canonical_dataset_path_order():
    rows = [
        {"dataset": "PhysionetMI", "path": "/z", "bytes": 2, "sha256": "b" * 64},
        {"dataset": "Cho2017", "path": "/a", "bytes": 1, "sha256": "a" * 64},
    ]
    payload = repair.target_raw_manifest_payload(rows)
    assert [(row["dataset"], row["path"]) for row in payload["files"]] == [
        ("Cho2017", "/a"), ("PhysionetMI", "/z"),
    ]
    assert payload["target_labels"] == 0


def test_historical_raw_manifest_requires_exact_replay(tmp_path):
    payload = repair.target_raw_manifest_payload([
        {"dataset": "Cho2017", "path": "/a", "bytes": 1, "sha256": "a" * 64},
    ])
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    replay = repair.require_exact_historical_raw_manifest(payload, path)
    assert replay["exact_replay"] is True
    assert replay["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    changed = {**payload, "file_count": 2}
    with pytest.raises(repair.C84FR1TargetStageError):
        repair.require_exact_historical_raw_manifest(changed, path)


def test_target_repair_module_has_no_training_import_or_entrypoint():
    source_path = ROOT / "oaci/multidataset/c84fr1_target_stage_repair.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("training" in name or name.startswith("oaci.train") for name in imported)
    function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert not any(name.startswith("train") for name in function_names)


def test_repair_protocol_and_failure_evidence_replay():
    protocol = REPORTS / "C84FR1_TARGET_REGISTRY_CANONICAL_ORDER_REPAIR_PROTOCOL.json"
    sidecar = REPORTS / "C84FR1_TARGET_REGISTRY_CANONICAL_ORDER_REPAIR_PROTOCOL.sha256"
    expected = sidecar.read_text(encoding="ascii").split()[0]
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == expected
    payload = json.loads(protocol.read_text(encoding="utf-8"))
    assert payload["repair"]["retraining_allowed"] is False
    assert payload["repair"]["target_label_access"] is False
    failure = json.loads((REPORTS / "C84F_FAILED_ATTEMPT_896185.json").read_text())
    assert failure["frozen_state"]["candidate_units"] == 1944
    assert failure["frozen_state"]["target_unlabeled_artifacts"] == 0
    assert all(value == 0 for value in failure["isolation"].values())


def test_runtime_fails_when_replacement_lock_is_absent(tmp_path):
    with pytest.raises(runtime.C84FR1RuntimeError, match="execution lock is absent"):
        runtime.require_authorization_and_lock(
            lock_path=tmp_path / "missing.json",
            lock_sha_path=tmp_path / "missing.sha256",
            authorization_path=tmp_path / "authorization.json",
            output_root=tmp_path / "output",
        )


def test_lock_generator_checks_git_ancestry_by_return_code():
    assert lock_generator.git_is_ancestor(lock_generator.PROTOCOL_COMMIT, "HEAD")
    assert not lock_generator.git_is_ancestor("0" * 40, "HEAD")


def test_target_runtime_transitive_repository_dependencies_are_lock_bound():
    required = {
        "oaci/__init__.py",
        "oaci/support_graph.py",
        "oaci/multidataset/c84_dataset_registry.py",
        "oaci/multidataset/c84_dataset_registry_v2.py",
        "oaci/multidataset/c84r_montage_repair.py",
        "oaci/multidataset/c84fl2_protocol.py",
        "oaci/multidataset/c84f_field_manifest.py",
        "oaci/multidataset/c84f_target_instrumentation.py",
        "oaci/multidataset/c84f_runtime_guard.py",
        "oaci/multidataset/c84fr1_runtime_guard.py",
        "oaci/multidataset/c84fr1_target_stage_repair.py",
        "oaci/models/__init__.py",
        "oaci/models/factory.py",
        "oaci/models/shallow.py",
        "oaci/models/output.py",
    }
    assert required <= set(lock_generator.IMPLEMENTATION_FILES)


def test_canary_artifacts_replay_before_fresh_authorization_check():
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    function = source[source.index("def require_authorization_and_lock"):]
    assert function.index("verify_dual_canary_reuse") < function.index("verify_authorization_record")
