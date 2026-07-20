"""A2b-2b-ii: fake artifact write -> deep verify -> read -> compare closed loop, CLI demo, corruption.

Standalone (``python -m oaci.tests.test_runner_fake_artifact``) and pytest-compatible.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile

import oaci.protocol
from oaci.artifacts.canonical_json import decode_canonical_json
from oaci.artifacts.summary import read_completed_artifact
from oaci.artifacts.verify import verify_artifact_tree
from oaci.artifacts.writer import (GitEvidence, collect_git_evidence, context_scientific_hash,
                                   git_evidence_hash)
from oaci.runner.fake import DEFAULT_METHOD_ORDER
from oaci.runner.fake_artifact import build_fake_artifact_context, run_fake_two_level

from oaci.tests.test_runner_artifacts import _rewrite_index_and_marker

_MAN = os.path.join(os.path.dirname(oaci.protocol.__file__), "fake_runner_v1.yaml")
_REV = tuple(reversed(DEFAULT_METHOD_ORDER))
_C = {}


def _ge():
    c, t = "c" * 40, "t" * 40
    return GitEvidence(c, t, ("oaci",), (), True, git_evidence_hash(c, t, ("oaci",), (), True))


def _run(model_seed=0, order=DEFAULT_METHOD_ORDER):
    key = (model_seed, tuple(order))
    if key not in _C:
        repo = os.path.join(tempfile.gettempdir(), "oaci-fake-repo-marker")
        _C[key] = run_fake_two_level(_MAN, tempfile.mkdtemp(prefix="oaci-fakeart-"), model_seed=model_seed,
                                     method_order=order, repo_root=repo, git_evidence=_ge())
    return _C[key]


def _fresh_copy():
    src = _run().write_result.artifact_dir
    dst = os.path.join(tempfile.mkdtemp(), os.path.basename(src))
    shutil.copytree(src, dst)
    return dst


def _git_repo(dirty_file=None):
    d = tempfile.mkdtemp()
    for cmd in (["init", "-q"], ["config", "user.email", "a@b.c"], ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", d, *cmd], check=True)
    os.makedirs(os.path.join(d, "oaci"))
    with open(os.path.join(d, "oaci", "keep.py"), "w") as f:
        f.write("x\n")
    subprocess.run(["git", "-C", d, "add", "-A"], check=True)
    subprocess.run(["git", "-C", d, "commit", "-q", "-m", "init"], check=True)
    if dirty_file:
        with open(os.path.join(d, "oaci", dirty_file), "w") as f:
            f.write("y\n")
    return d


# ============================ verifier CLI / counts ============================
def test_verify_cli_uses_current_report_fields():
    from oaci.artifacts import verify as V
    buf = io.StringIO()
    old = sys.stdout; sys.stdout = buf
    try:
        rc = V._main([_run().write_result.artifact_dir])
    finally:
        sys.stdout = old
    assert rc == 0 and "verified_checkpoints" in buf.getvalue() and "total_files" in buf.getvalue()


def test_artifact_write_result_counts_are_unambiguous():
    wr = _run().write_result
    assert wr.n_total_files == wr.n_indexed_files + 2 and wr.n_unique_checkpoints >= 1


# ============================ git evidence ============================
def test_git_evidence_is_collected_not_claimed():
    ge = collect_git_evidence(_git_repo(), ("oaci",))
    assert len(ge.commit) == 40 and len(ge.tree_hash) == 40 and ge.clean and not ge.status_entries


def test_git_evidence_rejects_dirty_tracked_staged_and_untracked_scientific_files():
    ge = collect_git_evidence(_git_repo(dirty_file="dirty.py"), ("oaci",))
    assert not ge.clean and ge.status_entries
    fake = _run().fake_fold
    try:
        build_fake_artifact_context(fake, _run().fold_result, repo_root="/x", git_evidence=ge)
    except ValueError:
        pass
    else:
        raise AssertionError("a dirty scientific tree must be rejected")


def test_context_hash_binds_commit_and_tree_hash():
    g1, g2 = _ge(), GitEvidence("d" * 40, "t" * 40, ("oaci",), (), True,
                                git_evidence_hash("d" * 40, "t" * 40, ("oaci",), (), True))
    ctx = _run().context
    h1 = context_scientific_hash(ctx.manifest_payload, ctx.execution_config_payloads, ctx.model_spec_payloads, g1)
    h2 = context_scientific_hash(ctx.manifest_payload, ctx.execution_config_payloads, ctx.model_spec_payloads, g2)
    assert h1 != h2 and h1 == ctx.context_hash


def test_deep_verifier_recomputes_context_hash():
    tree = _fresh_copy()
    doc = decode_canonical_json(open(os.path.join(tree, "context", "provenance.json"), "rb").read())
    doc["body"]["commit"] = "f" * 40                          # tamper the recorded commit
    from oaci.artifacts.canonical_json import canonical_json_bytes
    with open(os.path.join(tree, "context", "provenance.json"), "wb") as f:
        f.write(canonical_json_bytes(doc))
    _rewrite_index_and_marker(tree)                           # fix file shas so only the logical check can fail
    rep = verify_artifact_tree(tree, deep=True)
    assert not rep.ok


# ============================ payload identity ============================
def test_manifest_payload_roundtrip_matches_manifest_hash():
    from oaci.protocol.manifest_v2 import load_v2, manifest_logical_payload, manifest_payload_hash
    m = load_v2(_MAN)
    assert manifest_payload_hash(manifest_logical_payload(m)) == m.freeze()["sha256"]


def test_context_payloads_match_actual_execution_config_and_model_spec():
    from oaci.runner.config import execution_config_logical_payload, model_spec_logical_payload
    from oaci.runner.keys import canonical_json_hash
    fake = _run().fake_fold
    assert canonical_json_hash(execution_config_logical_payload(fake.execution_config)) \
        == fake.execution_config.execution_config_hash
    assert canonical_json_hash(model_spec_logical_payload(fake.model_spec)) == fake.model_spec.model_spec_hash


# ============================ closed loop ============================
def test_fake_write_verify_read_compare_closed_loop():
    r = _run()
    assert r.verification.ok and r.loaded_summary.fold_result_hash == r.fold_result.fold_result_hash
    assert r.comparison_hash and os.path.exists(os.path.join(r.write_result.artifact_dir, "COMMITTED.json"))


def test_loaded_summary_matches_every_in_memory_logical_hash():
    from oaci.artifacts.summary import compare_artifact_summary_to_memory
    r = _run()
    cmp = compare_artifact_summary_to_memory(r.loaded_summary, r.fold_result, r.context,
                                             artifact_scientific_hash=r.write_result.artifact_scientific_hash)
    assert cmp.ok and not cmp.mismatches


def test_loaded_support_tables_match_exactly():
    r = _run()
    for lvl, lr in r.fold_result.level_items:
        s = r.loaded_summary.levels[int(lvl)]
        assert s.eligibility_counts == _rows_int(lr.support_state.support_graph.counts)
        assert s.reference_prior == tuple(float(x) for x in lr.support_state.support_graph.reference_prior.tolist())
    assert r.loaded_summary.levels[1].cell_mass[0][1] == 0.0   # deleted cell


def test_loaded_checkpoint_refs_and_hashes_match():
    r = _run()
    for lvl, lr in r.fold_result.level_items:
        mem = {lr.erm_stage.checkpoint.model_hash}
        for _, m in lr.method_items:
            mem.add(m.selection.model_hash)
            mem.update(c.model_hash for c in m.train_result.trajectory)
        assert set(r.loaded_summary.levels[int(lvl)].checkpoint_hashes) == mem


def test_loaded_prediction_and_metrics_hashes_match():
    r = _run()
    lr = r.fold_result.levels[0]
    s = r.loaded_summary.levels[0]
    for name, m in lr.method_items:
        ms = dict(s.method_items)[name]
        assert dict(ms.prediction_content_hashes)["target_audit"] == m.target_predictions.prediction_content_hash()
        assert dict(ms.metrics_hashes)["target_audit"] == m.target_metrics.metrics_hash


def test_artifact_index_sha_is_exposed_in_summary():
    r = _run()
    assert len(r.loaded_summary.artifact_index_sha256) == 64


# ============================ reproducibility ============================
def test_same_seed_reproduces_fold_and_artifact_scientific_hashes():
    a = run_fake_two_level(_MAN, tempfile.mkdtemp(), model_seed=0, repo_root="/x", git_evidence=_ge())
    b = run_fake_two_level(_MAN, tempfile.mkdtemp(), model_seed=0, repo_root="/x", git_evidence=_ge())
    assert a.fold_result.fold_result_hash == b.fold_result.fold_result_hash
    assert a.write_result.artifact_scientific_hash == b.write_result.artifact_scientific_hash


def test_permuted_method_order_reproduces_all_scientific_hashes():
    a, b = _run(order=DEFAULT_METHOD_ORDER), _run(order=_REV)
    assert a.fold_result.fold_result_hash == b.fold_result.fold_result_hash
    assert a.write_result.artifact_scientific_hash == b.write_result.artifact_scientific_hash


def test_different_model_seed_changes_fold_and_artifact_hashes():
    a, b = _run(model_seed=0), _run(model_seed=1)
    assert a.fold_result.fold_result_hash != b.fold_result.fold_result_hash
    assert a.write_result.artifact_scientific_hash != b.write_result.artifact_scientific_hash


def test_different_model_seed_keeps_fake_data_and_fold_scope_hashes():
    a, b = _run(model_seed=0), _run(model_seed=1)
    assert a.fake_fold.fake_data_hash == b.fake_fold.fake_data_hash
    assert a.fold_result.fold_scope.fold_scope_hash == b.fold_result.fold_scope.fold_scope_hash


# ============================ corruption / interruption ============================
def test_corrupt_fake_prediction_reports_exact_path():
    tree = _fresh_copy()
    target = os.path.join("levels", "level-000", "methods", "OACI", "target_audit.npz")
    with open(os.path.join(tree, target), "ab") as f:
        f.write(b"\x00")
    rep = verify_artifact_tree(tree, deep=True)
    assert not rep.ok and any(p == target for p, _ in rep.errors)


def test_corrupt_fake_checkpoint_reports_exact_path():
    tree = _fresh_copy()
    ck_dir = os.path.join(tree, "levels", "level-000", "checkpoints")
    stem = next(f[:-3] for f in os.listdir(ck_dir) if f.endswith(".pt"))
    rel = os.path.join("levels", "level-000", "checkpoints", stem + ".pt")
    with open(os.path.join(tree, rel), "ab") as f:
        f.write(b"\x00")
    rep = verify_artifact_tree(tree, deep=True)
    assert not rep.ok and any(p == rel for p, _ in rep.errors)


def test_coordinated_file_sha_rewrite_still_fails_deep_verification():
    import torch
    tree = _fresh_copy()
    ck_dir = os.path.join(tree, "levels", "level-000", "checkpoints")
    stem = next(f[:-3] for f in os.listdir(ck_dir) if f.endswith(".pt"))
    pt = os.path.join(ck_dir, stem + ".pt")
    st = torch.load(pt, map_location="cpu", weights_only=True)
    k0 = sorted(st)[0]; st[k0] = st[k0] + 1.0
    with open(pt, "wb") as f:
        torch.save(st, f)
    _rewrite_index_and_marker(tree)                           # shas now match; only state_hash can catch it
    rep = verify_artifact_tree(tree, deep=True)
    assert not rep.ok and any(stem in p for p, _ in rep.errors)


def test_interrupted_write_leaves_no_committed_tree():
    import oaci.artifacts.writer as W
    out = tempfile.mkdtemp()
    orig = W._Tree.checkpoint
    W._Tree.checkpoint = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom mid-write"))
    try:
        run_fake_two_level(_MAN, out, model_seed=0, repo_root="/x", git_evidence=_ge())
    except RuntimeError:
        pass
    else:
        raise AssertionError("the interrupted write should have raised")
    finally:
        W._Tree.checkpoint = orig
    committed = [d for d in os.listdir(out) if os.path.exists(os.path.join(out, d, "COMMITTED.json"))]
    staging = [d for d in os.listdir(out) if d.startswith(".tmp-oaci-artifact-")]
    assert not committed and not staging


# ============================ CLI ============================
def test_demo_outputs_canonical_json_and_success_exit():
    from oaci.runner.demo import main
    repo = _git_repo()
    buf = io.BytesIO()

    class _W:
        buffer = buf
    old = sys.stdout; sys.stdout = _W()
    try:
        rc = main(["--manifest", _MAN, "--output-root", tempfile.mkdtemp(), "--model-seed", "0",
                   "--method-order", "ERM,OACI,global_lpc,uniform", "--repo-root", repo])
    finally:
        sys.stdout = old
    assert rc == 0
    doc = decode_canonical_json(buf.getvalue())
    assert "efficacy" in doc["notice"] and doc["deep_verification_ok"] and len(doc["levels"]) == 2


def test_verifier_cli_success_and_failure_exit_codes():
    from oaci.artifacts import verify as V
    assert V._main([_run().write_result.artifact_dir]) == 0
    bad = _fresh_copy(); os.remove(os.path.join(bad, "COMMITTED.json"))
    assert V._main([bad]) == 1
    assert V._main([]) == 2


def test_no_oaci_runtime_import_from_cmi_or_h2cmi():
    import oaci.runner.fake_artifact  # noqa: F401
    import oaci.runner.demo  # noqa: F401
    bad = [m for m in sys.modules if m == "cmi" or m.startswith("cmi.") or m == "h2cmi" or m.startswith("h2cmi.")]
    assert not bad, f"oaci must not import cmi/h2cmi at runtime: {bad}"


def _rows_int(a):
    import numpy as np
    return tuple(tuple(float(x) for x in row) for row in np.asarray(a).tolist())


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} runner-fake-artifact tests")


if __name__ == "__main__":
    _run_all()
