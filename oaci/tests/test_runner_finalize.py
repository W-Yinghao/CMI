"""A2b-1b-ii-b: three-role eval-unit predictions + metrics + final in-memory ABI (phase COMPLETE).

Standalone (``python -m oaci.tests.test_runner_finalize``) and pytest-compatible.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch

from oaci.eval.artifacts import PredictionBundle
from oaci.eval.calibration import fixed_bin_edges
from oaci.runner import (DEFAULT_METHOD_ORDER, DeletionCell, RoleView, RowPredictionArtifact, RunnerPhase,
                         aggregate_role_to_bundle, assemble_fold_run, build_fold_scope, build_frozen_maps,
                         build_level_plans, build_level_population, build_level_support, evaluate_prediction_bundle,
                         finalize_level_run, level0_reference_prior, make_deletion_schedule, predict_checkpoint,
                         run_level_complete, run_level_through_audit)
from oaci.runner.config import RunnerExecutionConfig
from oaci.runner.keys import FoldKey, RunKey
from oaci.runner.predict import PredictionCacheKey

from oaci.runner import FoldData
from oaci.tests.test_runner_train_select import _exec_cfg, _MSPEC, _factory, _fold, _rows, _scope_cfg

_CPU = torch.device("cpu")
_REV = tuple(reversed(DEFAULT_METHOD_ORDER))


from oaci.protocol.manifest_v2 import manifest_payload_hash

_MANIFEST_PAYLOAD = {"protocol_id": "oaci-test", "status": "smoke"}
_MANIFEST_HASH = manifest_payload_hash(_MANIFEST_PAYLOAD)


def _complete(rows=None, model_seed=0, order=DEFAULT_METHOD_ORDER, level=0, deletion=None, support_m=2):
    rows = rows if rows is not None else _rows()
    fd = _fold(rows)
    maps = build_frozen_maps(["c0", "c1"], sorted({r["dom"] for r in rows if r["role"] == "source_train"}),
                             ["Ad0", "Ad1", "Td0"])
    sch = make_deletion_schedule(deletion if deletion is not None else [DeletionCell("Sd0", "c1")], fd, maps)
    ref = level0_reference_prior(fd, maps)
    cfg = _scope_cfg(support_m)
    rk = RunKey(FoldKey(_MANIFEST_HASH, "ds", "f0", 1, 2), level, model_seed)
    ss = build_level_support(fd, maps, level, sch, ref, support_m=support_m)
    lp = build_level_population(fd, maps, ss)
    fs = build_fold_scope(rk.fold_key, maps, fd, sch, cfg)
    plans = build_level_plans(fs, level, ss, lp, cfg, model_seed=model_seed)
    lr = run_level_complete(rk, fd, ss, lp, fs, plans, _exec_cfg(), _MSPEC, _factory, _CPU, method_order=order)
    return lr, (fd, maps, ss, lp, fs, plans, rk)


def _audit(rows=None, level=0, deletion=None, support_m=2):
    rows = rows if rows is not None else _rows()
    fd = _fold(rows)
    maps = build_frozen_maps(["c0", "c1"], sorted({r["dom"] for r in rows if r["role"] == "source_train"}),
                             ["Ad0", "Ad1", "Td0"])
    sch = make_deletion_schedule(deletion if deletion is not None else [DeletionCell("Sd0", "c1")], fd, maps)
    ref = level0_reference_prior(fd, maps)
    cfg = _scope_cfg(support_m)
    rk = RunKey(FoldKey("m", "ds", "f0", 1, 2), level, 0)
    ss = build_level_support(fd, maps, level, sch, ref, support_m=support_m)
    lp = build_level_population(fd, maps, ss)
    fs = build_fold_scope(rk.fold_key, maps, fd, sch, cfg)
    plans = build_level_plans(fs, level, ss, lp, cfg, model_seed=0)
    ai = run_level_through_audit(rk, fd, ss, lp, fs, plans, _exec_cfg(), _MSPEC, _factory, _CPU)
    return ai, (fd, maps, ss, lp, fs, plans, rk)


_DC = {}


def _done():
    if "a" not in _DC:
        _DC["a"] = _complete()
    return _DC["a"]


def _shared_pair(lr):
    """The two methods that selected the same checkpoint in the default fixture."""
    by = {}
    for n, m in lr.method_items:
        by.setdefault(m.selection.model_hash, []).append(n)
    return next(v for v in by.values() if len(v) > 1)


# ---------- hand-built role views for unit-aggregation tests ----------
def _rv(sids, eu, y, dom, grp, mass):
    return RoleView(role="source_guard", row_indices=np.arange(len(sids)), X=torch.zeros(len(sids), 1),
                    y=np.array(y, np.int64), sample_id=tuple(sids), domain_id=tuple(dom),
                    group_id=tuple(grp), eval_unit_id=tuple(eu), sample_mass=np.array(mass, float),
                    population_hash="p", tensor_hash="t")


def _art(sids, logits):
    return RowPredictionArtifact(sample_id=tuple(sids), logits=np.asarray(logits, float), model_hash="m",
                                 role="source_guard", population_hash="p", tensor_hash="t", content_hash="c")


def _agg(rv, art, dmap):
    return aggregate_role_to_bundle(art, rv, method_name="ERM", selected_model_hash="m", domain_map=dmap,
                                    class_names=("c0", "c1"), model_seed=0, fold_key_hash="fk", support_hash="sh",
                                    split_manifest_hash="sm", preprocess_hash="pp", risk_metric="ce",
                                    prob_floor=1e-6, deletion_level=0)


def _missing_class_bundle():
    return PredictionBundle(sample_id=["a", "b", "c", "d"], logits=[[2, 0], [0, 2], [2, 0], [2, 0]],
                            y=[0, 1, 0, 0], domain=[0, 0, 1, 1], group=["g0", "g0", "g1", "g1"], method="ERM",
                            seed=0, split_id="s", split_role="target_audit", deletion_level=0, class_names=("c0", "c1"))


# ============================ residual fixes ============================
def test_runner_config_consumes_ece_bins():
    import os
    from oaci.protocol.freeze import default_confirmatory_path
    from oaci.protocol.manifest_v2 import load_v2
    p = os.path.join(os.path.dirname(default_confirmatory_path()), "smoke_v1.yaml")
    m = load_v2(p)
    assert RunnerExecutionConfig.from_manifest(m).ece_bins == m.evaluation.ece_bins
    base = RunnerExecutionConfig.from_manifest(load_v2(p)).execution_config_hash
    m2 = load_v2(p); m2.evaluation.ece_bins = 20
    assert RunnerExecutionConfig.from_manifest(m2).execution_config_hash != base


def test_prediction_bundle_freezes_class_names_and_metadata():
    b = _missing_class_bundle()
    assert isinstance(b.class_names, tuple)
    assert len(b.bundle_hash) == 64 and len(b.audit_signature_hash) == 64
    try:
        b.method = "X"                                          # frozen dataclass
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("PredictionBundle must be frozen")


def test_source_guard_role_identity_is_not_sample_id_only():
    rows = _rows()
    fd = _fold(rows)
    rv = fd.make_role_view("source_guard", fd.source_train_idx)
    assert len(rv.population_hash) == 64 and len(rv.tensor_hash) == 64 and rv.population_hash != rv.tensor_hash
    flipped = [dict(r) for r in rows]
    for r in flipped:                                            # flip a source-train label -> identity changes
        if r["role"] == "source_train":
            r["y"] = 1 - r["y"]; break
    rv2 = _fold(flipped).make_role_view("source_guard", fd.source_train_idx)
    assert rv2.population_hash != rv.population_hash             # binds metadata, not only sample ids


def test_role_view_population_and_tensor_hash_are_row_order_invariant():
    rows = _rows()
    a = _fold(rows).make_role_view("source_audit")
    perm = list(np.random.default_rng(1).permutation(len(rows)))
    b = _fold([rows[i] for i in perm]).make_role_view("source_audit")
    assert a.population_hash == b.population_hash and a.tensor_hash == b.tensor_hash


def test_role_view_tensor_hash_changes_with_X():
    rows = _rows()
    fd_a = _fold(rows)
    Xb = fd_a.X.clone(); Xb[int(fd_a.source_audit_idx[0])] += 7.0          # perturb one audit row's X
    fd_b = FoldData.from_arrays(
        X=Xb, y=np.asarray(fd_a.y), sample_id=list(fd_a.sample_id), domain_id=list(fd_a.domain_id),
        group_id=list(fd_a.group_id), support_unit_id=list(fd_a.support_unit_id),
        mass_unit_id=list(fd_a.mass_unit_id), eval_unit_id=list(fd_a.eval_unit_id),
        sample_mass=np.asarray(fd_a.sample_mass), class_names=list(fd_a.class_names),
        source_train_idx=np.asarray(fd_a.source_train_idx), source_audit_idx=np.asarray(fd_a.source_audit_idx),
        target_audit_idx=np.asarray(fd_a.target_audit_idx), preprocess_hash=fd_a.preprocess_hash,
        split_manifest_hash=fd_a.split_manifest_hash, preprocess_fit_ids=frozenset())
    a = fd_a.make_role_view("source_audit"); b = fd_b.make_role_view("source_audit")
    assert a.population_hash == b.population_hash and a.tensor_hash != b.tensor_hash


# ============================ prediction ============================
def test_prediction_forward_is_rng_mode_and_state_safe():
    lr, ctx = _done()
    fd, rk = ctx[0], ctx[6]
    rv = fd.make_role_view("source_audit")
    sel = lr.methods["ERM"].selection
    before = torch.random.get_rng_state()
    predict_checkpoint(sel.model_state, sel.model_hash, _factory, rv, factory_seed=123, chunk_size=None, device=_CPU)
    assert torch.equal(before, torch.random.get_rng_state())


def test_prediction_cache_key_binds_role_population_tensor_and_chunk():
    base = PredictionCacheKey("m", "source_audit", "p", "t", "spec", 512)
    for k in (PredictionCacheKey("m", "target_audit", "p", "t", "spec", 512),
              PredictionCacheKey("m", "source_audit", "p2", "t", "spec", 512),
              PredictionCacheKey("m", "source_audit", "p", "t2", "spec", 512),
              PredictionCacheKey("m", "source_audit", "p", "t", "spec", 256)):
        assert k != base


def test_prediction_cache_computes_once_per_unique_hash_per_role():
    lr, _ = _done()
    pc = lr.prediction_cache_stats
    K = dict(lr.invariant_items)["n_unique_checkpoints"]
    assert pc.source_guard_computes == K and pc.source_audit_computes == K and pc.target_computes == K


def test_prediction_factory_seed_is_method_order_independent():
    a, _ = _done()
    b, _ = _complete(order=_REV)
    da = {n: m.target_predictions.prediction_content_hash() for n, m in a.method_items}
    db = {n: m.target_predictions.prediction_content_hash() for n, m in b.method_items}
    assert da == db


def test_nonfinite_or_wrong_class_dimension_logits_fail():
    try:
        PredictionBundle(sample_id=["a"], logits=[[np.inf, 0]], y=[0], domain=[0], group=["g"], method="ERM",
                         seed=0, split_id="s", split_role="target_audit", deletion_level=0, class_names=("c0", "c1"))
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite logits must fail")
    try:
        PredictionBundle(sample_id=["a"], logits=[[0, 0, 0]], y=[0], domain=[0], group=["g"], method="ERM",
                         seed=0, split_id="s", split_role="target_audit", deletion_level=0, class_names=("c0", "c1"))
    except ValueError:
        pass
    else:
        raise AssertionError("C != len(class_names) must fail")


# ============================ eval-unit aggregation ============================
def test_eval_unit_requires_constant_label_domain_and_group():
    rv = _rv(("a", "b"), ("u", "u"), [0, 1], ["d0", "d0"], ["g", "g"], [1.0, 1.0])
    try:
        _agg(rv, _art(("a", "b"), [[1, 0], [0, 1]]), {"d0": 0})
    except ValueError:
        pass
    else:
        raise AssertionError("an eval unit spanning labels must fail")


def test_eval_unit_aggregation_uses_sample_mass():
    rv = _rv(("a", "b"), ("u", "u"), [0, 0], ["d0", "d0"], ["g", "g"], [3.0, 1.0])
    b = _agg(rv, _art(("a", "b"), [[10, 0], [0, 10]]), {"d0": 0})
    p = np.exp(b.logits[0])
    assert abs(p[0] / p[1] - 3.0) < 0.05                        # 3:1 mass weighting recovered


def test_eval_unit_aggregation_is_window_duplication_invariant():
    one = _agg(_rv(("a",), ("u",), [0], ["d0"], ["g"], [1.0]), _art(("a",), [[2, 0]]), {"d0": 0})
    dup = _agg(_rv(("a", "b"), ("u", "u"), [0, 0], ["d0", "d0"], ["g", "g"], [0.5, 0.5]),
               _art(("a", "b"), [[2, 0], [2, 0]]), {"d0": 0})
    assert np.allclose(one.logits, dup.logits)


def test_eval_unit_bundle_uses_unit_ids_not_window_ids():
    b = _agg(_rv(("a", "b"), ("u", "u"), [0, 0], ["d0", "d0"], ["g", "g"], [1.0, 1.0]),
             _art(("a", "b"), [[1, 0], [1, 0]]), {"d0": 0})
    assert list(b.sample_id) == ["u"]


def test_probability_floor_is_applied_and_renormalized():
    b = _agg(_rv(("a",), ("u",), [0], ["d0"], ["g"], [1.0]), _art(("a",), [[1000.0, -1000.0]]), {"d0": 0})
    p = np.exp(b.logits[0])
    assert abs(p.sum() - 1.0) < 1e-9 and p.min() >= 1e-6 * 0.99


def test_metrics_use_eval_units_not_windows():
    rv = _rv(("a", "b", "c", "d"), ("u0", "u0", "u1", "u1"), [0, 0, 1, 1], ["d0", "d0", "d1", "d1"],
             ["g0", "g0", "g1", "g1"], [0.5, 0.5, 0.5, 0.5])
    b = _agg(rv, _art(("a", "b", "c", "d"), [[2, 0], [2, 0], [0, 2], [0, 2]]), {"d0": 0, "d1": 1})
    assert b.n == 2                                             # 4 windows -> 2 eval units
    evaluate_prediction_bundle(b, bin_edges=fixed_bin_edges(5))


def test_source_guard_contains_current_level_rows_only():
    l0, _ = _done()
    l1, _ = _complete(level=1)
    deleted = {s for s in l0.methods["ERM"].source_guard_predictions.sample_id.tolist()
               if s.startswith("source_train_Sd0_") and s.endswith("_c1")}
    present1 = set(l1.methods["ERM"].source_guard_predictions.sample_id.tolist())
    assert deleted and not (deleted & present1)                 # the deleted (Sd0,c1) units are gone at level 1


# ============================ frozen populations ============================
def test_source_audit_bundle_uses_frozen_population_and_tensor_hash():
    lr, ctx = _done()
    fd = ctx[0]
    b = lr.methods["OACI"].source_audit_predictions
    assert b.audit_tensor_hash == fd.source_audit_tensor_hash


def test_target_bundle_uses_frozen_population_and_tensor_hash():
    lr, ctx = _done()
    fd = ctx[0]
    assert lr.methods["OACI"].target_predictions.audit_tensor_hash == fd.target_tensor_hash


def test_target_prediction_records_no_fit_ids():
    lr, _ = _done()
    assert not lr.provenance.target_fit_ids


def test_same_checkpoint_has_same_row_prediction_hash_across_methods():
    lr, _ = _done()
    a, b = _shared_pair(lr)[:2]
    ma, mb = lr.methods[a], lr.methods[b]
    for role in ("source_guard_predictions", "source_audit_predictions", "target_predictions"):
        assert np.array_equal(getattr(ma, role).logits, getattr(mb, role).logits)     # same forward reused
    assert ma.source_audit_predictions.audit_signature_hash == mb.source_audit_predictions.audit_signature_hash
    assert ma.source_audit_predictions.prediction_content_hash() != mb.source_audit_predictions.prediction_content_hash()


# ============================ metrics ============================
def test_metrics_use_one_fixed_bin_edge_array():
    lr, _ = _done()
    b = lr.methods["OACI"].target_predictions
    m5 = evaluate_prediction_bundle(b, bin_edges=fixed_bin_edges(5))
    m50 = evaluate_prediction_bundle(b, bin_edges=fixed_bin_edges(50))
    assert m5.metrics_hash != m50.metrics_hash                  # the bin edges are an input, fixed per run


def test_reference_bacc_missing_class_status_is_preserved():
    m = evaluate_prediction_bundle(_missing_class_bundle(), bin_edges=fixed_bin_edges(5))
    assert np.isnan(m.mean_domain_reference_bacc) and m.domain_reference_status == "nonestimable_missing_class_domain"


def test_observed_bacc_and_class_coverage_are_reported():
    m = evaluate_prediction_bundle(_missing_class_bundle(), bin_edges=fixed_bin_edges(5))
    assert not np.isnan(m.mean_domain_observed_bacc) and dict(m.domain_class_coverage_items)[1] == 0.5


def test_domain_ece_mean_and_worst_are_reported():
    m = evaluate_prediction_bundle(_missing_class_bundle(), bin_edges=fixed_bin_edges(5))
    assert np.isfinite(m.mean_domain_ece) and np.isfinite(m.worst_domain_ece)


# ============================ finalize ============================
def test_finalize_requires_audit_phase():
    ai, ctx = _audit()
    bad = dataclasses.replace(ai, phase=RunnerPhase.SELECTION)
    try:
        finalize_level_run(bad, ctx[0], ctx[4], ctx[2], ctx[3], ctx[5], _exec_cfg(), _MSPEC, _factory, _CPU)
    except ValueError:
        pass
    else:
        raise AssertionError("finalize must require an AUDIT-phase intermediate")


def test_finalize_rejects_changed_selection_snapshot():
    ai, ctx = _audit()
    bad = dataclasses.replace(ai, selection_snapshot_after=dataclasses.replace(
        ai.selection_snapshot_after, snapshot_hash="deadbeef"))
    try:
        finalize_level_run(bad, ctx[0], ctx[4], ctx[2], ctx[3], ctx[5], _exec_cfg(), _MSPEC, _factory, _CPU)
    except RuntimeError:
        pass
    else:
        raise AssertionError("finalize must reject a changed selection snapshot")


def test_finalize_reaches_complete_phase():
    lr, _ = _done()
    assert lr.phase == RunnerPhase.COMPLETE and lr.provenance.phase == RunnerPhase.COMPLETE


def test_final_provenance_snapshot_is_immutable():
    lr, _ = _done()
    assert len(lr.provenance.provenance_hash) == 64
    try:
        lr.provenance.phase = RunnerPhase.AUDIT
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("the final provenance snapshot must be immutable")


def test_final_target_fit_ids_are_empty():
    lr, _ = _done()
    assert not lr.provenance.target_fit_ids and dict(lr.invariant_items)["target_fit_ids_empty"]


def test_inactive_method_receives_erm_predictions_and_metrics():
    lr, _ = _complete(rows=_rows(src_dom=1))
    inactive = [n for n, m in lr.method_items if not m.active]
    assert inactive
    for n in inactive:
        m = lr.methods[n]
        assert m.selection.selected_erm and m.selection.model_hash == m.shared_erm_hash
        assert m.target_predictions.n > 0 and np.isfinite(m.target_metrics.pooled_nll)


def test_source_audit_signatures_match_across_methods():
    lr, _ = _done()
    sigs = {m.source_audit_predictions.audit_signature_hash for _, m in lr.method_items}
    assert len(sigs) == 1 and dict(lr.invariant_items)["source_audit_signature_match"]


def test_target_signatures_match_across_methods():
    lr, _ = _done()
    sigs = {m.target_predictions.audit_signature_hash for _, m in lr.method_items}
    assert len(sigs) == 1 and dict(lr.invariant_items)["target_signature_match"]


def test_prediction_cache_stats_match_4_K_4minusK():
    lr, _ = _done()
    pc = lr.prediction_cache_stats
    K = dict(lr.invariant_items)["n_unique_checkpoints"]
    for req, comp, hit in ((pc.source_guard_requests, pc.source_guard_computes, pc.source_guard_hits),
                           (pc.source_audit_requests, pc.source_audit_computes, pc.source_audit_hits),
                           (pc.target_requests, pc.target_computes, pc.target_hits)):
        assert (req, comp, hit) == (4, K, 4 - K)


# ============================ fold assembly ============================
def test_fold_assembly_rejects_mismatched_scope():
    lr, ctx = _done()
    other_fs = dataclasses.replace(ctx[4], fold_key=FoldKey("m", "OTHER", "f0", 1, 2))
    try:
        assemble_fold_run(other_fs, {0: lr})
    except ValueError:
        pass
    else:
        raise AssertionError("a level FoldKey disagreeing with the scope must fail")


def test_fold_assembly_rejects_noncomplete_level():
    lr, ctx = _done()
    try:
        assemble_fold_run(ctx[4], {0: dataclasses.replace(lr, phase=RunnerPhase.AUDIT)})
    except ValueError:
        pass
    else:
        raise AssertionError("a non-COMPLETE level must fail assembly")


def test_audit_and_target_signatures_are_level_invariant():
    l0, c0 = _done()
    l1, _ = _complete(level=1)
    fr = assemble_fold_run(c0[4], {0: l0, 1: l1})
    assert len(fr.levels) == 2
    sa = {m.source_audit_predictions.audit_signature_hash for lr in (l0, l1) for _, m in lr.method_items}
    ta = {m.target_predictions.audit_signature_hash for lr in (l0, l1) for _, m in lr.method_items}
    assert len(sa) == 1 and len(ta) == 1


def test_uniform_prior_is_level_invariant():
    l0, _ = _done()
    l1, _ = _complete(level=1)
    assert (l0.methods["uniform"].training_diagnostics["prior_matrix_hash"]
            == l1.methods["uniform"].training_diagnostics["prior_matrix_hash"])


# ============================ integration ============================
def test_method_result_hash_binds_selection_risk_epoch_score_and_reason():
    from oaci.runner.finalize import method_result_hash
    m = _done()[0].methods["OACI"]
    audit = type("A", (), {"status": m.audit_status, "leakage_hash": None})()
    bundles = {"source_guard": m.source_guard_predictions, "source_audit": m.source_audit_predictions,
               "target_audit": m.target_predictions}
    mets = {"source_guard": m.source_guard_metrics, "source_audit": m.source_audit_metrics,
            "target_audit": m.target_metrics}
    tm = _TM(m)
    base = method_result_hash("OACI", tm, _SM(m.selection, m), audit, bundles, mets)
    for fld, val in (("R_src", m.selection.R_src + 1.0), ("selected_epoch", m.selection.selected_epoch + 5),
                     ("selection_reason", m.selection.selection_reason + "X"), ("n_feasible", m.selection.n_feasible + 1)):
        sel2 = dataclasses.replace(m.selection, **{fld: val})
        assert method_result_hash("OACI", tm, _SM(sel2, m), audit, bundles, mets) != base


def test_level_result_hash_binds_plans_erm_config_and_model_spec():
    a, _ = _done()
    b, _ = _complete(model_seed=0)
    assert a.level_result_hash == b.level_result_hash               # same seed/plans/config -> same hash
    c, _ = _complete(model_seed=99)
    assert a.level_result_hash != c.level_result_hash               # different ERM/plans -> different hash
    assert a.execution_config_hash == b.execution_config_hash and a.model_spec_hash == b.model_spec_hash


class _TM:
    def __init__(self, m):
        self.active = m.active; self.inactive_reason = m.inactive_reason
        self.shared_erm_hash = m.shared_erm_hash; self.shared_tau = m.shared_tau
        self.shared_stage2_task_plan_hash = m.shared_stage2_task_plan_hash
        self.train_result = m.train_result; self.training_diagnostics = dict(m.training_diagnostics_items)


class _SM:
    def __init__(self, selection, m):
        self.selection = selection; self.selection_status = m.selection_status
        self.selection_leakage = m.selection_leakage


def test_runner_single_level_complete_in_memory():
    lr, _ = _done()
    assert set(dict(lr.method_items)) == {"ERM", "OACI", "global_lpc", "uniform"}
    for _, m in lr.method_items:
        for b in (m.source_guard_predictions, m.source_audit_predictions, m.target_predictions):
            assert isinstance(b, PredictionBundle)
        assert np.isfinite(m.target_metrics.pooled_nll)
    assert len(lr.level_result_hash) == 64


def test_runner_complete_nonestimable_selection_and_audit():
    lr, ctx = _complete(rows=_rows_all_nonestimable(), support_m=2, deletion=[])
    assert ctx[5].selection_status != "estimable"               # selection non-estimable
    assert all(m.selection.selected_erm for _, m in lr.method_items)
    assert dict(lr.invariant_items)["n_unique_checkpoints"] == 1
    for _, m in lr.method_items:
        assert m.target_predictions.n > 0


def test_same_seed_reproduces_final_result_hashes():
    a, _ = _complete(model_seed=11)
    b, _ = _complete(model_seed=11)
    assert a.level_result_hash == b.level_result_hash


def test_permuted_method_order_reproduces_final_results():
    a, _ = _done()
    b, _ = _complete(order=_REV)
    assert a.level_result_hash == b.level_result_hash


def test_no_oaci_runtime_import_from_cmi_or_h2cmi():
    import sys
    import oaci.runner.finalize  # noqa: F401
    bad = [m for m in sys.modules if m == "cmi" or m.startswith("cmi.") or m == "h2cmi" or m.startswith("h2cmi.")]
    assert not bad, f"oaci must not import cmi/h2cmi at runtime: {bad}"


def _rows_all_nonestimable():
    """source_train Sd0=c0 / Sd1=c1, source_audit Ad0=c0 / Ad1=c1 (both non-estimable); target normal."""
    rows = []
    for d, c in ((0, 0), (1, 1)):
        dom = f"Sd{d}"
        for r in range(4):
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


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} runner-finalize tests")


if __name__ == "__main__":
    _run_all()
