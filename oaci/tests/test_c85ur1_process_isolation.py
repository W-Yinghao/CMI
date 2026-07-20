from __future__ import annotations

import ast
import builtins
import hashlib
import json
from pathlib import Path

import pytest

from oaci.multidataset.c84s_common import C84SContractError, sha256_file
from oaci.theory import c85u_stage_u2_v2
from oaci.theory.c85u_runtime_guard_v2 import RuntimeOpenPolicyV2
from oaci.theory.c85u_stage_u2_v2 import _validate_u1_handoff_before_u2
from oaci.theory.c85u_u1_registry_v2 import build_u1_runtime_registry

from .c85ur1_test_support import make_shadow_context


REPO_ROOT = Path(__file__).resolve().parents[2]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.add(node.module or "")
    return result


def test_v1_lock_bound_implementation_is_unchanged() -> None:
    lock = json.loads((REPO_ROOT / "oaci/reports/C85U_EXECUTION_LOCK.json").read_text())
    for row in lock["bound_repository_objects"]:
        path = REPO_ROOT / row["path"]
        assert path.stat().st_size == row["size_bytes"]
        assert sha256_file(path) == row["sha256"]


def test_u1_and_u2_v2_static_registry_isolation() -> None:
    theory = REPO_ROOT / "oaci/theory"
    u1 = theory / "c85u_u1_registry_v2.py"
    u2 = theory / "c85u_u2_registry_v2.py"
    u1_imports = _imports(u1)
    assert not any(
        token in imported.lower()
        for imported in u1_imports
        for token in ("stage_b", "q0", "analysis", "inference", "taxonomy")
    )
    assert "c85u_input_registry" not in (theory / "c85u_stage_u1_v2.py").read_text()
    u2_text = u2.read_text(encoding="utf-8")
    assert "/projects/" not in u2_text
    assert "stage_a_labels" not in u2_text
    assert "target_evaluation_label_view" not in u2_text


def test_u1_registry_dynamic_open_trap_rejects_forbidden_roots(monkeypatch) -> None:
    forbidden_fragments = (
        "stage_b_selection_freeze",
        "stage_c_scientific_result",
        "method_context_decisions",
    )
    original_open = builtins.open
    observed_forbidden: list[str] = []

    def trapped_open(file, *args, **kwargs):
        text = str(file)
        if any(fragment in text for fragment in forbidden_fragments):
            observed_forbidden.append(text)
            raise AssertionError(f"forbidden U1 open: {text}")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", trapped_open)
    registry = build_u1_runtime_registry()
    assert len(registry.contexts) == 944
    assert len(registry.target_artifact_rows) == 1_944
    assert registry.target_artifact_total_bytes == 48_018_748_054
    assert observed_forbidden == []


def test_runtime_file_open_policy_fails_closed(tmp_path: Path) -> None:
    allowed = (tmp_path / "allowed").resolve()
    forbidden = (tmp_path / "forbidden").resolve()
    policy = RuntimeOpenPolicyV2(frozenset({allowed}))
    assert policy.require_allowed(allowed) == allowed
    with pytest.raises(RuntimeError, match="file-open policy rejected"):
        policy.require_allowed(forbidden)
    assert policy.allowed_opens == 1
    assert policy.forbidden_opens == 1


def test_direct_u2_without_context_fails_before_any_open(monkeypatch, tmp_path: Path) -> None:
    calls: list[object] = []
    original_open = builtins.open

    def trapped_open(file, *args, **kwargs):
        calls.append(file)
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", trapped_open)
    with pytest.raises(SystemExit):
        c85u_stage_u2_v2.main([
            "run-real", "--utility-root", str(tmp_path / "utility"),
            "--output-root", str(tmp_path / "result"),
        ])
    assert calls == []


def test_u2_rejects_another_attempt_handoff_before_utility_open(tmp_path: Path) -> None:
    first = make_shadow_context(tmp_path / "first", attempt_id="first")
    second = make_shadow_context(tmp_path / "second", attempt_id="second")
    utility = first.output_root / "stage_u1_candidate_utility_v2"
    utility.mkdir()
    handoff = utility / "C85U_STAGE_U1_HANDOFF.json"
    value = {
        "execution_lock_sha256": first.execution_lock_sha256,
        "execution_lock_commit": first.execution_lock_commit,
        "authorization_file_sha256": first.authorization_file_sha256,
        "authorization_binding_sha256": first.authorization_binding_sha256,
        "authorization_id": first.authorization_id,
        "attempt_id": first.attempt_id,
        "parent_output_root": str(first.output_root),
        "U1_output_root": str(utility),
        "protected_replay_sha256": "1" * 64,
    }
    handoff.write_text(json.dumps(value), encoding="utf-8")
    digest = hashlib.sha256(handoff.read_bytes()).hexdigest()
    handoff.with_suffix(".sha256").write_text(
        f"{digest}  {handoff.name}\n", encoding="ascii",
    )
    with pytest.raises(C84SContractError, match="attempt linkage"):
        _validate_u1_handoff_before_u2(
            context=second, handoff_path=handoff, utility_root=utility,
        )
