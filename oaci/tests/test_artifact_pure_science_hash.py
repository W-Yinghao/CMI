"""Commit-independent pure-science hash vs provenance-bound artifact hash.

The artifact_scientific_hash binds the git commit/tree (so it changes across commits even when only a
test or doc file changed); artifact_pure_science_hash binds ONLY the science (manifest + execution config
+ model spec + fold result) so identical science yields an identical hash across commits. These tests use
the fake two-level closed loop with INJECTED git evidence to vary the "commit" without touching science.

Standalone (`python -m oaci.tests.test_artifact_pure_science_hash`) and pytest-compatible.
"""
from __future__ import annotations

import json
import os
import tempfile

import oaci.protocol
from oaci.artifacts.summary import compare_artifact_summary_to_memory, read_completed_artifact
from oaci.artifacts.verify import verify_artifact_tree
from oaci.artifacts.writer import (GitEvidence, artifact_pure_science_hash, git_evidence_hash,
                                   pure_science_context_hash)
from oaci.runner.demo import build_demo_summary
from oaci.runner.fake_artifact import run_fake_two_level

_MAN = os.path.join(os.path.dirname(oaci.protocol.__file__), "fake_runner_v1.yaml")
_ORDER = ("ERM", "OACI", "global_lpc", "uniform")
_C = {}


def _ge(commit_char):
    c, t = commit_char * 40, commit_char.upper() * 40           # commit AND tree differ
    return GitEvidence(c, t, ("oaci",), (), True, git_evidence_hash(c, t, ("oaci",), (), True))


def _run(commit_char, seed=0):
    return run_fake_two_level(_MAN, tempfile.mkdtemp(), model_seed=seed, method_order=_ORDER,
                              repo_root="/x", git_evidence=_ge(commit_char))


def _two_commits():
    """Same science, two different injected commits."""
    if "p" not in _C:
        _C["p"] = (_run("a"), _run("b"))
    return _C["p"]


# ============ commit-independence vs commit-sensitivity ============
def test_pure_science_hash_does_not_change_when_git_commit_changes_only():
    a, b = _two_commits()
    assert a.write_result.artifact_pure_science_hash == b.write_result.artifact_pure_science_hash
    assert a.write_result.pure_context_hash == b.write_result.pure_context_hash


def test_provenance_bound_hash_changes_when_git_commit_changes():
    a, b = _two_commits()
    assert a.write_result.artifact_scientific_hash != b.write_result.artifact_scientific_hash
    assert a.write_result.context_hash != b.write_result.context_hash
    assert a.write_result.provenance_hash != b.write_result.provenance_hash


# ============ the pure hash still binds the real science ============
def test_pure_science_hash_changes_when_manifest_science_changes():
    base_mpay = {"a": 1, "b": [2, 3]}
    ec = ((0, {"x": 1}), (1, {"x": 1}))
    ms = ((0, {"m": 1}), (1, {"m": 1}))
    h0 = pure_science_context_hash(base_mpay, ec, ms)
    assert pure_science_context_hash({"a": 2, "b": [2, 3]}, ec, ms) != h0     # manifest payload
    assert pure_science_context_hash(base_mpay, ((0, {"x": 9}), (1, {"x": 1})), ms) != h0  # exec config
    assert pure_science_context_hash(base_mpay, ec, ((0, {"m": 9}), (1, {"m": 1}))) != h0  # model spec
    # and the top-level pure hash binds the manifest hash + pure context hash
    assert artifact_pure_science_hash("FOLD", "MAN-A", h0) != artifact_pure_science_hash("FOLD", "MAN-B", h0)


def test_pure_science_hash_changes_when_checkpoint_or_prediction_changes():
    # a different model_seed -> different trained checkpoints/predictions -> different fold_result_hash,
    # so BOTH the pure-science and the provenance-bound hash change (same injected commit).
    s0, s1 = _run("c", seed=0), _run("c", seed=1)
    assert s0.fold_result.fold_result_hash != s1.fold_result.fold_result_hash
    assert s0.write_result.artifact_pure_science_hash != s1.write_result.artifact_pure_science_hash
    assert s0.write_result.artifact_scientific_hash != s1.write_result.artifact_scientific_hash


# ============ verifier + summary + demo ============
def test_verifier_recomputes_both_hashes():
    a, _ = _two_commits()
    rep = verify_artifact_tree(a.write_result.artifact_dir, deep=True)
    assert rep.ok
    assert rep.artifact_scientific_hash == a.write_result.artifact_scientific_hash
    assert rep.artifact_pure_science_hash == a.write_result.artifact_pure_science_hash


def test_verifier_rejects_a_tampered_pure_science_hash():
    a = _run("d")
    marker_p = os.path.join(a.write_result.artifact_dir, "COMMITTED.json")
    with open(marker_p) as f:
        marker = json.load(f)
    marker["artifact_pure_science_hash"] = "0" * 64                # tamper
    with open(marker_p, "w") as f:
        json.dump(marker, f)
    rep = verify_artifact_tree(a.write_result.artifact_dir, deep=True)
    assert not rep.ok
    assert any("pure-science hash does not recompute" in m for _, m in rep.errors)


def test_summary_reports_both_hashes_and_compare_detects_pure_mismatch():
    a, _ = _two_commits()
    summ = read_completed_artifact(a.write_result.artifact_dir, deep_verify=True)
    assert summ.artifact_pure_science_hash == a.write_result.artifact_pure_science_hash
    assert summ.provenance_hash == a.write_result.provenance_hash
    # correct compare passes; a wrong pure hash is reported as a mismatch
    ok = compare_artifact_summary_to_memory(summ, a.fold_result, a.context,
                                            artifact_scientific_hash=a.write_result.artifact_scientific_hash,
                                            artifact_pure_science_hash=a.write_result.artifact_pure_science_hash)
    assert ok.ok
    bad = compare_artifact_summary_to_memory(summ, a.fold_result, a.context,
                                             artifact_scientific_hash=a.write_result.artifact_scientific_hash,
                                             artifact_pure_science_hash="0" * 64)
    assert not bad.ok and "artifact_pure_science_hash" in bad.mismatches


def test_demo_reports_both_hashes():
    a, _ = _two_commits()
    d = build_demo_summary(a)
    for k in ("artifact_scientific_hash", "artifact_pure_science_hash", "pure_context_hash", "provenance_hash"):
        assert k in d and d[k]
    assert d["artifact_pure_science_hash"] == a.write_result.artifact_pure_science_hash


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} artifact-pure-science-hash tests")


if __name__ == "__main__":
    _run_all()
