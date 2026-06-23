"""A2b-2a: versioned artifact schema, deterministic codecs, checkpoint store, atomic commit, verifier.

Standalone (``python -m oaci.tests.test_runner_artifacts``) and pytest-compatible.
"""
from __future__ import annotations

import dataclasses
import os
import shutil
import tempfile

import numpy as np
import torch

from oaci.artifacts import plan_codec as P, prediction_codec as PR, support_codec as SC
from oaci.artifacts.atomic import COMMIT_MARKER, INDEX_NAME, StagingDir, safe_relpath
from oaci.artifacts.canonical_json import canonical_json_bytes, canonical_json_hash, decode_canonical_json
from oaci.artifacts.checkpoint import read_checkpoint_file, write_checkpoint_file
from oaci.artifacts.deterministic_npz import (deterministic_npz_bytes, read_verified_npz, to_unicode_array,
                                             write_deterministic_npz)
from oaci.artifacts.writer import (ArtifactContext, artifact_scientific_hash, context_scientific_hash,
                                   write_artifact_tree_atomic)
from oaci.artifacts.verify import verify_artifact_tree
from oaci.runner import assemble_fold_run

from oaci.tests.test_runner_finalize import _complete


def _context(lr, manifest_hash="m"):
    mp = {"dataset_id": "FAKE"}
    ecp = ((0, {"execution_config_hash": lr.execution_config_hash}),)
    msp = ((0, {"model_spec_hash": lr.model_spec_hash}),)
    return ArtifactContext(mp, manifest_hash, ecp, msp, "gitabc", True, context_scientific_hash(mp, ecp, msp))


_W = {}


def _written():
    if "w" not in _W:
        lr, ctx = _complete()
        fr = assemble_fold_run(ctx[4], {0: lr})
        root = tempfile.mkdtemp(prefix="oaci-art-")
        res = write_artifact_tree_atomic(fr, _context(lr), root)
        _W["w"] = (res, fr, lr, ctx, root)
    return _W["w"]


def _fresh_tree():
    res, fr, lr, ctx, _ = _written()
    dst = tempfile.mkdtemp(prefix="oaci-artc-")
    out = os.path.join(dst, os.path.basename(res.artifact_dir))
    shutil.copytree(res.artifact_dir, out)
    return out


# ============================ context / scientific hash ============================
def test_artifact_context_hashes_match_runner_result():
    res, fr, lr, _, _ = _written()
    assert res.fold_result_hash == fr.fold_result_hash
    assert res.artifact_scientific_hash == artifact_scientific_hash(fr.fold_result_hash, "m", res.context_hash)


def test_artifact_scientific_hash_is_independent_of_file_bytes():
    # the logical hash is H(schema, fold_result, manifest, context) -- not the .pt/.npz bytes
    a = artifact_scientific_hash("foldH", "manH", "ctxH")
    assert a == artifact_scientific_hash("foldH", "manH", "ctxH")
    assert a != artifact_scientific_hash("foldH2", "manH", "ctxH")


# ============================ canonical JSON ============================
def test_canonical_json_roundtrips_nonfinite_values():
    v = {"a": float("nan"), "b": float("inf"), "c": float("-inf"), "arr": np.array([1.0, np.nan])}
    d = decode_canonical_json(canonical_json_bytes(v))
    assert d["a"] != d["a"] and d["b"] == float("inf") and d["c"] == float("-inf")
    assert np.isnan(d["arr"][1]) and d["arr"][0] == 1.0


def test_canonical_json_rejects_reserved_tag_collision():
    try:
        canonical_json_bytes({"$float": "nan"})
    except ValueError:
        pass
    else:
        raise AssertionError("a reserved $float key in an ordinary mapping must be rejected")


def test_canonical_json_is_key_order_independent():
    assert canonical_json_bytes({"a": 1, "b": 2}) == canonical_json_bytes({"b": 2, "a": 1})
    assert canonical_json_hash({"x": [1, 2], "y": 3}) == canonical_json_hash({"y": 3, "x": [1, 2]})


# ============================ deterministic NPZ ============================
def test_deterministic_npz_is_byte_identical():
    a = {"q": np.arange(6), "g": to_unicode_array(["a", "bb", "ccc"]), "f": np.array([1.5, 2.5])}
    assert deterministic_npz_bytes(a) == deterministic_npz_bytes({k: a[k] for k in reversed(list(a))})


def test_npz_rejects_object_dtype():
    try:
        deterministic_npz_bytes({"o": np.array(["a", "b"], dtype=object)})
    except TypeError:
        pass
    else:
        raise AssertionError("object dtype must be rejected")


def test_npz_preserves_unicode_dtype_shape_and_values():
    with tempfile.TemporaryDirectory() as td:
        arrs = {"ids": to_unicode_array(["alpha", "b"]), "m": np.array([[1, 2], [3, 4]], dtype=np.int64)}
        meta = write_deterministic_npz(os.path.join(td, "a.npz"), arrs)
        back = read_verified_npz(os.path.join(td, "a.npz"), meta)
        assert list(back["ids"]) == ["alpha", "b"] and back["m"].dtype == np.int64 and back["m"].shape == (2, 2)


# ============================ checkpoint store ============================
def test_checkpoint_writer_rejects_non_tensor_payload():
    with tempfile.TemporaryDirectory() as td:
        try:
            write_checkpoint_file(os.path.join(td, "c.pt"), "h", {"w": [1, 2, 3]})
        except TypeError:
            pass
        else:
            raise AssertionError("a non-tensor checkpoint value must be rejected")


def test_checkpoint_roundtrip_preserves_state_hash():
    _, _, lr, _, _ = _written()
    sel = lr.methods["OACI"].selection
    with tempfile.TemporaryDirectory() as td:
        meta = write_checkpoint_file(os.path.join(td, "c.pt"), sel.model_hash, sel.model_state)
        st = read_checkpoint_file(os.path.join(td, "c.pt"), meta)
        from oaci.train.checkpoint import state_hash
        assert state_hash(st) == sel.model_hash


def test_checkpoint_store_deduplicates_shared_hash():
    res, fr, lr, _, _ = _written()
    # global_lpc and uniform share a checkpoint -> fewer physical .pt files than (method,role) pairs
    hashes = {m.selection.model_hash for _, m in lr.method_items}
    ck_dir = os.path.join(res.artifact_dir, "levels", "level-000", "checkpoints")
    pts = [f for f in os.listdir(ck_dir) if f.endswith(".pt")]
    assert len(hashes) < 4 and all(any(p.startswith(h) for p in pts) for h in hashes)


def test_checkpoint_corruption_is_detected():
    _, _, lr, _, _ = _written()
    sel = lr.methods["ERM"].selection
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "c.pt")
        meta = write_checkpoint_file(p, sel.model_hash, sel.model_state)
        with open(p, "ab") as f:
            f.write(b"\x00")
        try:
            read_checkpoint_file(p, meta)
        except ValueError:
            pass
        else:
            raise AssertionError("a corrupt checkpoint file must be detected")


def test_checkpoint_metadata_corruption_is_detected():
    _, _, lr, _, _ = _written()
    sel = lr.methods["ERM"].selection
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "c.pt")
        meta = write_checkpoint_file(p, sel.model_hash, sel.model_state)
        bad = {**meta, "model_hash": "deadbeef"}
        try:
            read_checkpoint_file(p, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("a corrupt checkpoint metadata must be detected")


# ============================ plan / prediction round-trips ============================
def test_support_graph_roundtrip_preserves_hash():
    _, _, _, ctx, _ = _written()
    g = ctx[2].support_graph
    _, b, a = SC.encode_support(ctx[2])
    assert SC.decode_support_graph(b, a).support_hash() == g.support_hash()


def test_task_plan_roundtrip_preserves_epoch_step_structure():
    _, _, _, ctx, _ = _written()
    p = ctx[5].stage2_task
    _, b, a = P.encode_task_plan(p)
    q = P.decode_task_plan(b, a)
    assert q.plan_hash == p.plan_hash and len(q.epochs) == len(p.epochs)
    assert [len(e) for e in q.epochs] == [len(e) for e in p.epochs]


def test_alignment_plan_roundtrip_preserves_all_microbatch_boundaries():
    _, _, _, ctx, _ = _written()
    p = ctx[5].full_domain_alignment
    _, b, a = P.encode_alignment_plan(p)
    q = P.decode_alignment_plan(b, a)
    assert q.plan_hash == p.plan_hash
    assert [len(lb.microbatches) for lb in q.warmup_batches] == [len(lb.microbatches) for lb in p.warmup_batches]
    assert [len(gs.critic_batches) for gs in q.game_steps] == [len(gs.critic_batches) for gs in p.game_steps]


def test_fold_plan_roundtrip_preserves_string_group_mapping():
    _, _, _, ctx, _ = _written()
    p = ctx[5].selection_fold_plan
    _, b, a = P.encode_fold_plan(p)
    q = P.decode_fold_plan(b, a)
    assert q.plan_hash == p.plan_hash and q.fold_of_group == p.fold_of_group and q.domain_of_group == p.domain_of_group


def test_bootstrap_plan_roundtrip_preserves_zero_multiplicities():
    _, _, _, ctx, _ = _written()
    p = ctx[5].selection_bootstrap_plan
    _, b, a = P.encode_bootstrap_plan(p)
    q = P.decode_bootstrap_plan(b, a)
    assert q.plan_hash == p.plan_hash
    assert q.candidate_draws[0].group_multiplicities == p.candidate_draws[0].group_multiplicities


def test_prediction_bundle_roundtrip_preserves_all_hashes():
    _, _, lr, _, _ = _written()
    bnd = lr.methods["OACI"].target_predictions
    _, b, a = PR.encode_prediction(bnd)
    q = PR.decode_prediction(b, a)
    assert q.prediction_content_hash() == bnd.prediction_content_hash()
    assert q.audit_signature_hash == bnd.audit_signature_hash and q.bundle_hash == bnd.bundle_hash


def test_metrics_roundtrip_preserves_nan_and_status():
    from oaci.eval.calibration import fixed_bin_edges
    from oaci.runner import evaluate_prediction_bundle
    from oaci.tests.test_runner_finalize import _missing_class_bundle
    em = evaluate_prediction_bundle(_missing_class_bundle(), bin_edges=fixed_bin_edges(5))
    _, b, _ = PR.encode_metrics(em)
    q = PR.decode_metrics(b)
    assert np.isnan(q.mean_domain_reference_bacc) and q.domain_reference_status == em.domain_reference_status
    assert q.metrics_hash == em.metrics_hash


# ============================ writer gate ============================
def test_writer_recomputes_method_level_and_fold_hashes():
    res, fr, lr, _, root = _written()
    for tamper in (dataclasses.replace(fr, fold_result_hash="bad"),
                   dataclasses.replace(fr, level_items=((0, dataclasses.replace(lr, level_result_hash="bad")),)),):
        try:
            write_artifact_tree_atomic(tamper, _context(lr), tempfile.mkdtemp())
        except ValueError:
            pass
        else:
            raise AssertionError("a tampered logical hash must be rejected by the writer")


def test_writer_rejects_noncomplete_result():
    from oaci.runner import RunnerPhase
    _, fr, lr, _, _ = _written()
    bad = dataclasses.replace(fr, level_items=((0, dataclasses.replace(lr, phase=RunnerPhase.AUDIT)),))
    _expect_reject(bad, lr)


def test_writer_rejects_false_required_invariant():
    _, fr, lr, _, _ = _written()
    inv = tuple((k, (False if k == "phase_complete" else v)) for k, v in lr.invariant_items)
    bad = dataclasses.replace(fr, level_items=((0, dataclasses.replace(lr, invariant_items=inv)),))
    _expect_reject(bad, lr)


def test_writer_accepts_numeric_zero_oaci_invariant():
    res, _, lr, _, _ = _written()
    inv = dict(lr.invariant_items)
    assert inv["oaci_rejected_ineligible_rows"] == 0 and inv["n_unique_checkpoints"] >= 1
    assert verify_artifact_tree(res.artifact_dir, deep=False).ok    # the real tree was accepted


def test_writer_rejects_target_fit_ids():
    _, fr, lr, _, _ = _written()
    prov = dataclasses.replace(lr.provenance, target_fit_ids=frozenset({"t0"}))
    bad = dataclasses.replace(fr, level_items=((0, dataclasses.replace(lr, provenance=prov)),))
    _expect_reject(bad, lr)


def test_writer_rejects_selection_snapshot_mismatch():
    _, fr, lr, _, _ = _written()
    bad = dataclasses.replace(fr, level_items=((0, dataclasses.replace(lr, selection_snapshot_hash="bad")),))
    _expect_reject(bad, lr)


def test_writer_rejects_checkpoint_hash_mismatch():
    _, fr, lr, _, _ = _written()
    es = lr.erm_stage
    bad_state = dict(es.checkpoint.model_state)
    k0 = sorted(bad_state)[0]
    bad_state[k0] = bad_state[k0] + 1.0                            # same model_hash string, different bytes
    es2 = dataclasses.replace(es, checkpoint=dataclasses.replace(es.checkpoint, model_state=bad_state))
    bad = dataclasses.replace(fr, level_items=((0, dataclasses.replace(lr, erm_stage=es2)),))
    _expect_reject(bad, lr)


def _expect_reject(bad_fr, lr):
    try:
        write_artifact_tree_atomic(bad_fr, _context(lr), tempfile.mkdtemp())
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("the writer must reject an inconsistent result")


# ============================ atomic commit ============================
def test_atomic_failure_leaves_no_committed_result():
    parent = tempfile.mkdtemp()
    final = os.path.join(parent, "art")
    try:
        with StagingDir(final) as st:
            st.write_bytes("a.json", b"x")
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert not os.path.exists(final)
    assert not [d for d in os.listdir(parent) if d.startswith(".tmp-oaci-artifact-")]


def test_existing_destination_is_not_replaced():
    lr_ctx = _written()
    res, fr, lr, _, _ = lr_ctx
    root = tempfile.mkdtemp()
    write_artifact_tree_atomic(fr, _context(lr), root)
    try:
        write_artifact_tree_atomic(fr, _context(lr), root)
    except FileExistsError:
        pass
    else:
        raise AssertionError("an existing destination must not be overwritten by default")


def test_commit_marker_is_written_last():
    res, _, _, _, _ = _written()
    index = decode_canonical_json(open(os.path.join(res.artifact_dir, INDEX_NAME), "rb").read())
    rels = {e["relative_path"] for e in index["files"]}
    assert COMMIT_MARKER not in rels and INDEX_NAME not in rels   # the marker is outside the index
    assert os.path.exists(os.path.join(res.artifact_dir, COMMIT_MARKER))


def test_parent_directory_is_fsynced():
    import oaci.artifacts.atomic as A
    seen = []
    orig = A._fsync_dir
    A._fsync_dir = lambda p: seen.append(os.path.abspath(p))
    try:
        _, fr, lr, _, _ = _written()
        root = tempfile.mkdtemp()
        r = write_artifact_tree_atomic(fr, _context(lr), root)
    finally:
        A._fsync_dir = orig
    assert os.path.abspath(root) in seen                          # the final parent was fsynced


def test_path_traversal_is_rejected():
    for rel in ("../x", "a/../b", "/abs", "a//b"):
        try:
            safe_relpath(rel)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe path {rel!r} must be rejected")


def test_symlink_output_is_rejected():
    base = tempfile.mkdtemp()
    link = os.path.join(base, "link")
    os.symlink(base, link)
    _, fr, lr, _, _ = _written()
    try:
        write_artifact_tree_atomic(fr, _context(lr), link)
    except ValueError:
        pass
    else:
        raise AssertionError("writing through a symlink output must be rejected")


# ============================ verifier ============================
def test_verifier_rejects_missing_commit_marker():
    tree = _fresh_tree()
    os.remove(os.path.join(tree, COMMIT_MARKER))
    rep = verify_artifact_tree(tree, deep=False)
    assert not rep.ok and any(p == COMMIT_MARKER for p, _ in rep.errors)


def test_verifier_rejects_unindexed_extra_file():
    tree = _fresh_tree()
    with open(os.path.join(tree, "levels", "extra.txt"), "w") as f:
        f.write("x")
    rep = verify_artifact_tree(tree, deep=False)
    assert not rep.ok and any("extra.txt" in p for p, _ in rep.errors)


def test_verifier_reports_exact_corrupt_path():
    tree = _fresh_tree()
    target = os.path.join("levels", "level-000", "methods", "OACI", "method.json")
    with open(os.path.join(tree, target), "ab") as f:
        f.write(b" ")
    rep = verify_artifact_tree(tree, deep=False)
    assert not rep.ok and any(p == target for p, _ in rep.errors)


def test_deep_verifier_recomputes_all_logical_hashes():
    res, _, _, _, _ = _written()
    rep = verify_artifact_tree(res.artifact_dir, deep=True)
    assert rep.ok and rep.n_checkpoints >= 1 and rep.n_plans >= 7
    assert rep.artifact_scientific_hash == res.artifact_scientific_hash


# ============================ A2b-2b artifact residuals ============================
def test_staging_directories_are_unique_for_same_destination():
    base = tempfile.mkdtemp()
    final = os.path.join(base, "art")
    a = StagingDir(final).__enter__()
    b = StagingDir(final).__enter__()
    assert a.staging != b.staging and os.path.isdir(a.staging) and os.path.isdir(b.staging)


def test_concurrent_staging_writers_do_not_delete_each_other():
    base = tempfile.mkdtemp()
    final = os.path.join(base, "art")
    a = StagingDir(final).__enter__()
    a.write_bytes("x.json", b"a")
    b = StagingDir(final).__enter__()       # a second writer must not wipe the first's staging
    assert os.path.exists(os.path.join(a.staging, "x.json"))
    a.__exit__(None, None, None)
    assert os.path.isdir(b.staging)         # closing A leaves B intact


def test_all_existing_parent_symlink_components_are_rejected():
    base = tempfile.mkdtemp()
    real = os.path.join(base, "real"); os.makedirs(real)
    link = os.path.join(base, "link"); os.symlink(real, link)
    # final = base/link/sub/art -> the 'link' ancestor (not the immediate parent) is a symlink
    final = os.path.join(link, "sub", "art")
    try:
        StagingDir(final).__enter__()
    except ValueError:
        pass
    else:
        raise AssertionError("a symlinked ancestor component must be rejected")


def test_deep_verifier_weights_only_loads_every_checkpoint():
    res, _, lr, _, _ = _written()
    rep = verify_artifact_tree(res.artifact_dir, deep=True)
    n_pt = len({m.selection.model_hash for _, m in lr.method_items} | {lr.erm_stage.checkpoint.model_hash})
    assert rep.ok and rep.n_verified_checkpoints == rep.n_checkpoints >= n_pt


def test_deep_verifier_rejects_state_hash_mismatch_after_index_rewrite():
    import torch
    from oaci.artifacts.canonical_json import canonical_json_bytes
    from oaci.artifacts.atomic import COMMIT_MARKER, INDEX_NAME
    tree = _fresh_tree()
    ck_dir = os.path.join(tree, "levels", "level-000", "checkpoints")
    stem = next(f[:-3] for f in os.listdir(ck_dir) if f.endswith(".pt"))
    pt = os.path.join(ck_dir, stem + ".pt")
    st = torch.load(pt, map_location="cpu", weights_only=True)
    k0 = sorted(st)[0]; st[k0] = st[k0] + 1.0                  # change bytes (state hash now != stem)
    with open(pt, "wb") as f:
        torch.save(st, f)
    # rewrite the index + marker so the file sha matches (only the state-hash check can catch it)
    _rewrite_index_and_marker(tree)
    rep = verify_artifact_tree(tree, deep=True)
    assert not rep.ok and any(stem in p for p, _ in rep.errors)


def test_deep_verifier_rejects_orphan_pt_or_metadata():
    import shutil as _sh
    tree = _fresh_tree()
    ck_dir = os.path.join(tree, "levels", "level-000", "checkpoints")
    stem = next(f[:-3] for f in os.listdir(ck_dir) if f.endswith(".pt"))
    extra = os.path.join(ck_dir, "deadbeef.pt")
    _sh.copy(os.path.join(ck_dir, stem + ".pt"), extra)       # a .pt with no .json (orphan)
    _rewrite_index_and_marker(tree)
    rep = verify_artifact_tree(tree, deep=True)
    assert not rep.ok and any("deadbeef" in p for p, _ in rep.errors)


def test_artifact_summary_counts_indexed_and_total_files_unambiguously():
    res, fr, _, _, _ = _written()
    from oaci.artifacts.summary import read_completed_artifact
    s = read_completed_artifact(res.artifact_dir, deep_verify=True)
    assert s.n_total_files == s.n_indexed_files + 2          # + index + marker
    assert s.artifact_scientific_hash == res.artifact_scientific_hash and s.fold_result_hash == fr.fold_result_hash
    assert s.n_verified_checkpoints >= 1 and s.n_verified_plans >= 7


def _rewrite_index_and_marker(tree):
    """Recompute the index file shas + marker index-sha (so only logical/state checks can fail)."""
    import hashlib
    from oaci.artifacts.atomic import COMMIT_MARKER, INDEX_NAME
    from oaci.artifacts.canonical_json import canonical_json_bytes
    index = decode_canonical_json(open(os.path.join(tree, INDEX_NAME), "rb").read())
    files = {os.path.relpath(os.path.join(r, fn), tree)
             for r, _, fns in os.walk(tree) for fn in fns} - {COMMIT_MARKER, INDEX_NAME}
    entries = []
    for rel in sorted(files):
        ap = os.path.join(tree, rel)
        prev = next((e for e in index["files"] if e["relative_path"] == rel), None)
        kind = prev["artifact_kind"] if prev else "checkpoint_pt"
        logical = prev["logical_hash"] if prev else rel.rsplit("/", 1)[-1][:-3]
        with open(ap, "rb") as f:
            data = f.read()
        entries.append({"relative_path": rel, "artifact_kind": kind, "schema_version": "oaci-artifact-v1",
                        "byte_size": len(data), "file_sha256": hashlib.sha256(data).hexdigest(),
                        "logical_hash": logical})
    idx_bytes = canonical_json_bytes({"files": sorted(entries, key=lambda e: e["relative_path"])})
    with open(os.path.join(tree, INDEX_NAME), "wb") as f:
        f.write(idx_bytes)
    marker = decode_canonical_json(open(os.path.join(tree, COMMIT_MARKER), "rb").read())
    marker["artifact_index_sha256"] = hashlib.sha256(idx_bytes).hexdigest()
    with open(os.path.join(tree, COMMIT_MARKER), "wb") as f:
        f.write(canonical_json_bytes(marker))


def test_fake_mlp_architecture_has_no_python_defaults():
    import os as _os
    from oaci.protocol.freeze import default_confirmatory_path
    from oaci.protocol.manifest_v2 import load_v2
    p = _os.path.join(_os.path.dirname(default_confirmatory_path()), "smoke_v1.yaml")
    m = load_v2(p)
    m.backbone.name = "mlp"; m.backbone.mlp_z_dim = None      # mlp without explicit dims must be rejected
    for fld in ("temporal_filters", "temporal_kernel_samples", "pool_kernel_samples",
                "pool_stride_samples", "dropout", "safe_log_eps"):
        setattr(m.backbone, fld, None)
    try:
        m.validate_ranges()
    except ValueError:
        pass
    else:
        raise AssertionError("an mlp backbone with no frozen dims must be rejected")


def test_no_oaci_runtime_import_from_cmi_or_h2cmi():
    import sys
    import oaci.artifacts.writer  # noqa: F401
    import oaci.artifacts.verify  # noqa: F401
    bad = [m for m in sys.modules if m == "cmi" or m.startswith("cmi.") or m == "h2cmi" or m.startswith("h2cmi.")]
    assert not bad, f"oaci must not import cmi/h2cmi at runtime: {bad}"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} runner-artifact tests")


if __name__ == "__main__":
    _run_all()
