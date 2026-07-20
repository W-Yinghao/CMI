"""A2b-1b-ii-a: selection snapshot + lock + fixed source-audit leakage (phase stops at AUDIT).

Standalone (``python -m oaci.tests.test_runner_audit``) and pytest-compatible. Reuses the in-memory
level fixtures from test_runner_train_select.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch

import oaci.runner.audit as audit_mod
from oaci.runner import (DEFAULT_METHOD_ORDER, RunnerPhase, build_training_data_for_design,
                         make_selection_snapshot, overlap_probe_sample_ids, run_post_selection_audit,
                         scientific_value_hash)
from oaci.runner.scientific_hash import leakage_result_hash
from oaci.train.data import population_signature_hash

from oaci.tests.test_runner_train_select import _exec_cfg, _MSPEC, _factory, _rows, _run

_CPU = torch.device("cpu")


def _audit(res, ctx):
    return run_post_selection_audit(res, ctx[0], ctx[4], _exec_cfg(), _MSPEC, _factory, _CPU)


# ----- one shared estimable audit (read-only tests) -----
_C = {}


def _est():
    if "a" not in _C:
        res, ctx = _run()
        _C["a"] = (_audit(res, ctx), res, ctx)
    return _C["a"]


def _rows_no_audit_overlap():
    """source_train: 2 domains × both classes; source_audit: Ad0 only c0, Ad1 only c1 (no class has
    support in >= 2 audit domains -> audit is non-estimable); target: Td0 both classes."""
    rows = []
    for d in range(2):
        dom = f"Sd{d}"
        for r in range(4):
            for c in (0, 1):
                rows.append(dict(sid=f"source_train_{dom}_r{r}_c{c}", dom=dom, grp=f"{dom}-rec{r}",
                                 unit=f"source_train_{dom}_r{r}_c{c}", y=c, mass=1.0, role="source_train"))
    for d, c in ((0, 0), (1, 1)):
        dom = f"Ad{d}"
        for r in range(3):
            rows.append(dict(sid=f"source_audit_{dom}_r{r}_c{c}", dom=dom, grp=f"{dom}-rec{r}",
                             unit=f"source_audit_{dom}_r{r}_c{c}", y=c, mass=1.0, role="source_audit"))
    for r in range(2):
        for c in (0, 1):
            rows.append(dict(sid=f"target_audit_Td0_r{r}_c{c}", dom="Td0", grp=f"Td0-rec{r}",
                             unit=f"target_audit_Td0_r{r}_c{c}", y=c, mass=1.0, role="target_audit"))
    return rows


class _O:
    def __init__(self, **k):
        self.__dict__.update(k)


# ============================ shared overlap-probe primitive ============================
def test_overlap_probe_ids_are_shared_by_selection_and_audit():
    ai, res, ctx = _est()
    fd, ss, lp, fs, plans = ctx[0], ctx[2], ctx[3], ctx[4], ctx[5]
    sel_ids = set(overlap_probe_sample_ids(plans.selection_design, ss.support_graph))
    assert set(res.provenance.selection_fit_ids) == sel_ids
    aud_ids = set(overlap_probe_sample_ids(fs.source_audit.design, fs.source_audit.support_graph))
    assert set(ai.provenance.audit_estimator_fit_ids) == aud_ids
    assert aud_ids and aud_ids <= set(fd.role_ids("source_audit"))     # only source-audit rows


def test_overlap_probe_ids_are_canonical_and_support_aware():
    des = _O(sample_id=("d", "c", "b", "a"), y=np.array([0, 0, 1, 1]), d=np.array([0, 1, 0, 1]))
    sg = _O(comparable_classes=[0], support_of_class={0: [0]})        # class 0 only in domain 0
    assert overlap_probe_sample_ids(des, sg) == ("d",)               # c excluded (d∉support), b/a not comparable
    des2 = _O(sample_id=("z", "a"), y=np.array([0, 0]), d=np.array([0, 0]))
    assert overlap_probe_sample_ids(des2, sg) == ("a", "z")          # canonical sort
    dup = _O(sample_id=("a", "a"), y=np.array([0, 0]), d=np.array([0, 0]))
    try:
        overlap_probe_sample_ids(dup, sg)
    except ValueError:
        pass
    else:
        raise AssertionError("a duplicate overlap sample id must be rejected")


# ============================ selection snapshot ============================
def test_selection_snapshot_is_method_order_invariant():
    _, res, _ = _est()
    sel = res.selected_methods
    a = make_selection_snapshot(sel).snapshot_hash
    b = make_selection_snapshot({k: sel[k] for k in reversed(list(sel))}).snapshot_hash
    assert a == b


def test_selection_snapshot_binds_complete_leakage_result():
    _, res, _ = _est()
    snap = make_selection_snapshot(res.selected_methods)
    erm = next(m for m in snap.methods if m.method_name == "ERM")
    assert erm.selection_leakage_hash == leakage_result_hash(res.selected_methods["ERM"].selection_leakage)
    bad = dict(res.selected_methods)                                 # estimable selection with leakage stripped
    bad["ERM"] = dataclasses.replace(bad["ERM"], selection_leakage=None)
    try:
        make_selection_snapshot(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("an estimable selection with no leakage must be rejected")


def test_selection_snapshot_recomputes_and_checks_model_state_hash():
    _, res, _ = _est()
    for m in make_selection_snapshot(res.selected_methods).methods:
        assert m.recomputed_state_hash == m.model_hash


def test_selection_snapshot_detects_state_or_selection_mutation():
    _, res, _ = _est()
    sm = res.selected_methods["ERM"]
    tampered = dict(res.selected_methods)
    tampered["ERM"] = dataclasses.replace(sm, selection=dataclasses.replace(sm.selection, model_hash="deadbeef"))
    try:
        make_selection_snapshot(tampered)
    except ValueError:
        pass
    else:
        raise AssertionError("a mutated selected model hash must be detected")


def test_lock_requires_complete_four_method_selection():
    _, res, _ = _est()
    incomplete = {k: v for k, v in res.selected_methods.items() if k != "uniform"}
    try:
        make_selection_snapshot(incomplete)
    except ValueError:
        pass
    else:
        raise AssertionError("a snapshot needs exactly the four methods")


# ============================ lock + ordering ============================
def test_audit_requires_intermediate_selection_phase():
    res, ctx = _run()
    _audit(res, ctx)                                                 # first audit moves provenance to AUDIT
    try:
        _audit(res, ctx)                                            # second must refuse
    except ValueError:
        pass
    else:
        raise AssertionError("audit must require a SELECTION-phase provenance")


def test_selection_is_locked_before_audit_transition():
    ai, _, _ = _est()
    assert ai.invariants["selection_locked_event_index"] is not None
    assert ai.provenance.phase == RunnerPhase.AUDIT


def test_all_selection_events_precede_first_audit_fit_event():
    ai, _, _ = _est()
    assert ai.invariants["all_audit_fits_after_lock"]
    assert ai.invariants["first_audit_fit_event_index"] > ai.invariants["selection_locked_event_index"]


# ============================ audit TrainingData ============================
def test_audit_training_data_follows_design_order():
    _, _, ctx = _est()
    fd, design = ctx[0], ctx[4].source_audit.design
    td = build_training_data_for_design(fd, design)
    assert td.sample_id == tuple(design.sample_id)


def test_audit_training_data_matches_design_population_exactly():
    _, _, ctx = _est()
    fd, design = ctx[0], ctx[4].source_audit.design
    td = build_training_data_for_design(fd, design)
    assert population_signature_hash(td) == design.population_hash
    assert set(td.sample_id) == set(fd.role_ids("source_audit"))


def test_audit_training_data_rejects_metadata_mismatch():
    _, _, ctx = _est()
    fd, design = ctx[0], ctx[4].source_audit.design
    y2 = np.array(design.y); y2[0] = 1 - int(y2[0])
    try:
        build_training_data_for_design(fd, dataclasses.replace(design, y=y2))
    except ValueError:
        pass
    else:
        raise AssertionError("design metadata disagreeing with FoldData must be rejected")


def test_audit_training_data_rejects_any_target_id():
    _, _, ctx = _est()
    fd, design = ctx[0], ctx[4].source_audit.design
    tgt = fd.role_ids("target_audit")[0]
    bad_ids = (tgt,) + tuple(design.sample_id[1:])
    try:
        build_training_data_for_design(fd, dataclasses.replace(design, sample_id=bad_ids))
    except ValueError:
        pass
    else:
        raise AssertionError("a target id in the audit design must be rejected")


def test_audit_uses_exact_scope_fold_and_bootstrap_objects():
    res, ctx = _run()
    fs = ctx[4]
    seen = []
    orig = audit_mod.make_leakage_score_key

    def cap(feat, sg, fold, boot, critic):
        seen.append((sg, fold, boot))
        return orig(feat, sg, fold, boot, critic)

    audit_mod.make_leakage_score_key = cap
    try:
        _audit(res, ctx)
    finally:
        audit_mod.make_leakage_score_key = orig
    sa = fs.source_audit
    assert seen and all(sg is sa.support_graph and fold is sa.fold_plan and boot is sa.bootstrap_plan
                        for sg, fold, boot in seen)


# ============================ audit scoring / cache reuse ============================
def test_audit_fit_ids_are_only_overlap_probe_rows():
    ai, _, ctx = _est()
    expect = set(overlap_probe_sample_ids(ctx[4].source_audit.design, ctx[4].source_audit.support_graph))
    assert set(ai.provenance.audit_estimator_fit_ids) == expect
    assert ai.invariants["audit_fit_id_count"] == len(expect)


def test_audit_feature_cache_computes_once_per_unique_hash():
    ai, _, _ = _est()
    cs = ai.audit_cache_stats
    assert cs.feature_requests == 4 and cs.feature_computes == cs.n_unique_selected_models
    assert cs.feature_hits == 4 - cs.n_unique_selected_models and ai.invariants["feature_cache_reuse_valid"]


def test_audit_score_cache_computes_once_per_unique_hash():
    ai, _, _ = _est()
    cs = ai.audit_cache_stats
    assert cs.score_requests == 4 and cs.score_computes == cs.n_unique_selected_models
    assert cs.score_hits == 4 - cs.n_unique_selected_models and ai.invariants["score_cache_reuse_valid"]


def test_same_selected_hash_is_reused_across_methods():
    ai, _, _ = _est()
    by_hash = {}
    for n, r in ai.audit_method_items:
        by_hash.setdefault(r.model_hash, []).append(r.leakage_hash)
    assert any(len(v) > 1 for v in by_hash.values())                # the smoke shares global_lpc/uniform
    for v in by_hash.values():
        assert len(set(v)) == 1                                     # same checkpoint -> identical audit leakage


def test_audit_factory_seed_is_method_order_independent():
    res1, c1 = _run(order=DEFAULT_METHOD_ORDER)
    res2, c2 = _run(order=tuple(reversed(DEFAULT_METHOD_ORDER)))
    a1 = {n: r.leakage_hash for n, r in _audit(res1, c1).audit_method_items}
    a2 = {n: r.leakage_hash for n, r in _audit(res2, c2).audit_method_items}
    assert a1 == a2


# ============================ non-estimable audit ============================
def test_audit_nonestimable_performs_zero_feature_and_probe_work():
    res, ctx = _run(rows=_rows_no_audit_overlap())
    assert ctx[4].source_audit.status != "estimable"
    ai = _audit(res, ctx)
    cs = ai.audit_cache_stats
    assert (cs.feature_requests, cs.feature_computes, cs.feature_hits) == (0, 0, 0)
    assert (cs.score_requests, cs.score_computes, cs.score_hits) == (0, 0, 0)
    assert not ai.provenance.audit_estimator_fit_ids
    for _, r in ai.audit_method_items:
        assert r.status != "estimable" and r.leakage is None and r.leakage_hash is None


def test_audit_nonestimable_keeps_selection_unchanged():
    res, ctx = _run(rows=_rows_no_audit_overlap())
    ai = _audit(res, ctx)
    assert ai.selection_snapshot_before.snapshot_hash == ai.selection_snapshot_after.snapshot_hash
    assert ai.phase == RunnerPhase.AUDIT


def test_audit_numerical_failure_is_not_hidden():
    res, ctx = _run()
    orig = audit_mod.compute_leakage_score
    audit_mod.compute_leakage_score = lambda *a, **k: (_ for _ in ()).throw(FloatingPointError("nonfinite probe"))
    try:
        _audit(res, ctx)
    except FloatingPointError:
        raised = True
    else:
        raised = False
    finally:
        audit_mod.compute_leakage_score = orig
    assert raised                                                    # not converted to non-estimable


def test_accepted_replicate_failure_propagates_candidate_id():
    res, ctx = _run()
    orig = audit_mod.compute_leakage_score
    audit_mod.compute_leakage_score = lambda *a, **k: (_ for _ in ()).throw(ValueError("accepted replicate cand_42 failed"))
    msg = ""
    try:
        _audit(res, ctx)
    except ValueError as e:
        msg = str(e)
    finally:
        audit_mod.compute_leakage_score = orig
    assert "cand_42" in msg


# ============================ final-shape invariants ============================
def test_selection_snapshot_before_after_audit_is_identical():
    ai, _, _ = _est()
    assert ai.selection_snapshot_before.snapshot_hash == ai.selection_snapshot_after.snapshot_hash
    assert ai.invariants["selection_snapshot_unchanged"]


def test_audit_phase_stops_at_audit_not_complete():
    ai, _, _ = _est()
    assert ai.phase == RunnerPhase.AUDIT and ai.provenance.phase == RunnerPhase.AUDIT


def test_target_fit_ids_remain_empty():
    ai, _, _ = _est()
    assert not ai.provenance.target_fit_ids and ai.invariants["target_fit_ids_empty"]


def test_same_seed_reproduces_audit_leakage_hashes():
    r1, c1 = _run(model_seed=7)
    r2, c2 = _run(model_seed=7)
    a1 = _audit(r1, c1)
    a2 = _audit(r2, c2)
    assert {n: r.leakage_hash for n, r in a1.audit_method_items} == {n: r.leakage_hash for n, r in a2.audit_method_items}
    assert a1.selection_snapshot_before.snapshot_hash == a2.selection_snapshot_before.snapshot_hash


def test_permuted_method_order_reproduces_audit_results():
    r1, c1 = _run(order=DEFAULT_METHOD_ORDER)
    r2, c2 = _run(order=tuple(reversed(DEFAULT_METHOD_ORDER)))
    a1 = _audit(r1, c1)
    a2 = _audit(r2, c2)
    assert a1.selection_snapshot_after.snapshot_hash == a2.selection_snapshot_after.snapshot_hash
    assert a1.audit_cache_stats.stats_hash == a2.audit_cache_stats.stats_hash


def test_no_oaci_runtime_import_from_cmi_or_h2cmi():
    import sys
    import oaci.runner.audit  # noqa: F401
    bad = [m for m in sys.modules if m == "cmi" or m.startswith("cmi.") or m == "h2cmi" or m.startswith("h2cmi.")]
    assert not bad, f"oaci must not import cmi/h2cmi at runtime: {bad}"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} runner-audit tests")


if __name__ == "__main__":
    _run_all()
