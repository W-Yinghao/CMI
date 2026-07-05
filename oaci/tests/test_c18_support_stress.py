"""C18 Controlled Support-Mismatch x Identifiability Stress Test. Deterministic masks from source metadata
only; target labels joined post hoc + diagnostic-only; non-estimable cells never smoothed; finite-only
filtering (None/NaN/+-inf) before every fit; LOTO probe + permutation baseline + non-deployability; S6/S7
severity match; boundary from per-class recall; order-invariant severity; report forbids the causal over-claim;
and S0 recompute self-consistency (the mechanism that makes S0 reproduce C17)."""
from __future__ import annotations

import json
import os

import numpy as np

from oaci.support_graph import build_support_graph
from oaci.support_stress import feature_inventory, masks, schema, severity_response, support_metrics
from oaci.support_stress import source_signal_recompute as ssr
from oaci.support_stress import stress_plan as sp
from oaci.support_stress.report import _guard_forbidden
from oaci.support_stress.taxonomy import severity_taxonomy


def _sg(counts=None):
    counts = np.array([[20, 18, 15, 10], [16, 12, 9, 8], [14, 10, 8, 8], [12, 9, 8, 7], [10, 8, 8, 6], [9, 8, 7, 5]],
                      dtype=np.int64) if counts is None else counts
    return build_support_graph(counts, 8, cell_mass=counts.astype(float) / counts.sum(),
                               reference_prior=np.ones(counts.shape[1]) / counts.shape[1],
                               domain_names=[f"s{i}" for i in range(counts.shape[0])],
                               class_names=[str(c) for c in range(counts.shape[1])])


def test_boundary_classes_are_c16_losers():
    bnd = sp.boundary_classes_from_c16({"0": {"mean_recall_delta": 0.026}, "1": {"mean_recall_delta": -0.038},
                                        "2": {"mean_recall_delta": 0.033}, "3": {"mean_recall_delta": -0.032}})
    assert bnd == (1, 3)                                              # the opposite-direction losers


def test_support_masks_are_deterministic():
    sg = _sg()
    a = sp.all_regime_plans(sg.counts, sg.cell_mass, sg.eligible, sg.m, boundary_classes=(1, 3), seed=0, target=1, level=0)
    b = sp.all_regime_plans(sg.counts, sg.cell_mass, sg.eligible, sg.m, boundary_classes=(1, 3), seed=0, target=1, level=0)
    for r in schema.REGIME_ORDER:
        assert a[r].as_row() == b[r].as_row()                        # identical across calls (S7 seeded, no RNG drift)


def test_support_masks_use_source_metadata_only():
    # plans depend only on (counts, cell_mass, eligible, m, boundary_classes, fold key) — no target labels
    sg = _sg()
    p = sp.build_regime_plan("S6_boundary_aligned_mask", sg.counts, sg.cell_mass, sg.eligible, sg.m,
                             boundary_classes=(1, 3), seed=0, target=1, level=0)
    assert all(a.cls in (1, 3) for a in p.actions)                   # boundary-aligned touches only boundary classes


def test_boundary_aligned_and_random_masks_match_severity():
    sg = _sg()
    plans = sp.all_regime_plans(sg.counts, sg.cell_mass, sg.eligible, sg.m, boundary_classes=(1, 3), seed=0, target=1, level=0)
    assert plans["S6_boundary_aligned_mask"].severity_n_cells == plans["S7_random_matched_mask"].severity_n_cells


def test_nonestimable_cells_are_not_smoothed():
    sg = _sg()
    p = sp.build_regime_plan("S4_missing_cells", sg.counts, sg.cell_mass, sg.eligible, sg.m,
                             boundary_classes=(1, 3), seed=0, target=1, level=0)
    na = masks.actions_by_name(p, sg.domain_names)
    m = masks.apply_to_support_graph(na, sg)
    for (dname, c), a in na.items():
        d = list(sg.domain_names).index(dname)
        assert m.counts[d, c] == 0 and m.cell_mass[d, c] == 0.0      # deleted -> zero, never imputed/smoothed


def test_estimability_loss_counts_manual_cells():
    sg = _sg()
    p = sp.build_regime_plan("S4_missing_cells", sg.counts, sg.cell_mass, sg.eligible, sg.m,
                             boundary_classes=(1, 3), seed=0, target=1, level=0, n_perturb=2)
    na = masks.actions_by_name(p, sg.domain_names)
    loss = support_metrics.estimability_loss(sg, masks.apply_to_support_graph(na, sg))
    assert loss["eligible_cells_removed"] == len(p.deleted_cells()) == 2


def test_feature_inventory_complete_and_static_excluded():
    feature_inventory.assert_inventory_complete()
    assert set(feature_inventory.static_features()) == {"R_src", "balanced_err", "train_surrogate", "epoch"}
    try:
        feature_inventory.assert_only_recomputable_used(["src__train_surrogate"])
    except ValueError:
        pass
    else:
        raise AssertionError("static training-log feature must be rejected for mask-stress claims")


def test_finite_filter_drops_none_nan_inf_before_standardization():
    from oaci.identifiability.multivariate_probe import _finite, _matrix
    assert _finite(0.5) and not _finite(None) and not _finite(float("nan"))
    assert not _finite(float("inf")) and not _finite(float("-inf"))
    # a column that is None/NaN/inf across all rows is DROPPED (not every row dropped)
    rows = [{"src__" + s: 0.1 * i for s in schema.RECOMPUTABLE_UNDER_MASK}
            | {"src__source_guard_worst_bacc": (None if i % 3 == 0 else float("nan") if i % 3 == 1 else float("inf")),
               "src__" + "epoch": 1, "src__R_src": 1.0, "src__balanced_err": 0.3, "src__train_surrogate": 1.0,
               "tgt__target_bacc_good": bool(i % 2), "seed": 0, "target": (i % 3) + 1, "level": 0}
            for i in range(12)]
    X, y, gt, gs, fold, cols = _matrix(rows)
    assert "src__source_guard_worst_bacc" not in cols and np.isfinite(X).all()


def test_rank_correlation_handles_constant_and_missing_series():
    from oaci.identifiability.univariate import _spearman
    assert _spearman([1, 1, 1], [1, 2, 3]) is None
    assert _spearman([1, 2, 3], [3, 2, 1]) < -0.99


def _synth_atlas(n_targets=4, n_per=6, seed=0):
    rng = np.random.RandomState(seed); rows = []
    for t in range(1, n_targets + 1):
        for k in range(n_per):
            good = k % 2 == 0
            base = {"seed": 0, "target": t, "level": 0, "model_hash": f"m{t}{k}",
                    "diagnostic_only_non_deployable": True, "tgt__target_bacc_good": good,
                    "tgt__target_bacc_delta": (0.02 if good else -0.02), "tgt__target_nll_delta": -0.01}
            for s in schema.RECOMPUTABLE_UNDER_MASK:
                base["src__" + s] = (0.6 if good else 0.4) + rng.randn() * 0.05
            for s in schema.STATIC_TRAINING_LOG_ONLY:
                base["src__" + s] = float(rng.randn())
            rows.append(base)
    return rows


def test_probe_uses_leave_one_target_out_and_permutation_baseline_and_no_selector():
    from oaci.identifiability.multivariate_probe import multivariate_probe
    m = multivariate_probe(_synth_atlas(), n_perm=15)
    assert set(m["per_target_auc"]) <= {"1", "2", "3", "4"}
    assert m["permutation_p"] is not None and m["loto_auc"] is not None
    assert m["non_deployable"] is True and "selector" not in m and "chosen_model_hash" not in m


def test_target_labels_diagnostic_only_and_joined_post_hoc():
    import inspect
    # recompute_candidate takes NO target argument (targets cannot influence source signals)
    sig = inspect.signature(ssr.recompute_candidate)
    assert not any("target_worst" in p or "tgt" in p for p in sig.parameters)
    rows = _synth_atlas()
    assert all(r["diagnostic_only_non_deployable"] for r in rows)
    from oaci.support_stress.identifiability_stress import _assert_no_target_features
    _assert_no_target_features(rows)                                 # no tgt__ leaks into src__ feature set


def test_severity_response_is_order_invariant():
    probe = {r: {"loto_auc": 0.5 + 0.01 * i, "beats_permutation": True, "permutation_p": 0.01, "n_used": 100}
             for i, r in enumerate(schema.REGIME_ORDER)}
    boundary = {r: {"source_target_recall_delta_corr": 0.3} for r in schema.REGIME_ORDER}
    leak = {r: {"source_estimable_fraction": 1.0} for r in schema.REGIME_ORDER}
    a = severity_response.severity_response(probe, boundary, leak)
    # shuffle dict order -> same result (keyed by regime, not order)
    probe2 = {r: probe[r] for r in reversed(schema.REGIME_ORDER)}
    b = severity_response.severity_response(probe2, boundary, leak)
    assert a["severity_rows"] == b["severity_rows"] and a["order_invariant"]


def test_taxonomy_deterministic_and_diagnostic_only():
    probe = {r: {"loto_auc": 0.60, "beats_permutation": True} for r in schema.REGIME_ORDER}
    t1 = severity_taxonomy(probe_by_regime=probe); t2 = severity_taxonomy(probe_by_regime=probe)
    assert t1["case_label"] == t2["case_label"] and t1["diagnostic_only_non_deployable"]
    assert t1["case_label"] in (schema.CASE_PRESERVED, schema.CASE_COLLAPSED_II, schema.CASE_COLLAPSED_IV,
                                schema.CASE_BOUNDARY_DESTROYED, schema.CASE_ABSTENTION_DOMINANT, schema.CASE_INCONCLUSIVE)


def test_report_forbids_support_mismatch_causal_overclaim():
    for bad in ("support mismatch caused the original OACI failure", "OACI is rescued",
                "target oracle is deployable", "BNCI001 naturally demonstrates support mismatch"):
        try:
            _guard_forbidden("# C18\n\n" + bad + ".\n")
        except ValueError:
            continue
        raise AssertionError(f"forbidden claim not caught: {bad}")
    _guard_forbidden("# C18\n\ncontrolled support perturbations modulate source-side identifiability.\n")   # allowed


def _write_fake_extraction(root, c10root):
    """Minimal valid persisted extraction (1 seed x 2 targets x 1 level) + matching C10, enough for
    build_regime_atlas + self_check_s0 + identity_probe to run and be S0-self-consistent."""
    from oaci.eval.calibration import fixed_bin_edges
    edges = fixed_bin_edges(15); rng = np.random.RandomState(3)
    for t in (1, 2, 3):
        fdir = os.path.join(root, f"seed-0-target-{t:03d}", "level-0"); os.makedirs(fdir, exist_ok=True)
        gdoms = [f"BNCI2014_001|subject-00{d}" for d in range(4, 7)]                       # 3 guard domains
        adoms = [f"BNCI2014_001|subject-00{d}" for d in (2, 3)]                            # 2 audit domains
        n_cand = 5
        for role, doms in (("source_guard", gdoms), ("source_audit", adoms)):
            dom = np.array([d for d in doms for _ in range(4) for _ in range(3)], dtype=object)  # 3 units/cell
            y = np.array([c for _ in doms for c in range(4) for _ in range(3)], dtype=np.int64)
            n = len(y)
            lg = np.stack([rng.randn(n, 4) + (np.eye(4)[y] * (0.5 + 0.3 * ci)) for ci in range(n_cand)])
            np.savez(os.path.join(fdir, f"units-{role}.npz"), domain_raw=dom, y=y,
                     group=np.array([f"g{i}" for i in range(n)], dtype=object),
                     sample_id=np.array([f"{role}-{t}-{i}" for i in range(n)], dtype=object))
            np.save(os.path.join(fdir, f"logits-{role}.npy"), lg)
        for dz, doms in (("selection", gdoms), ("audit", adoms)):
            nrow = len(doms) * 4 * 3
            np.savez(os.path.join(fdir, f"featz-{dz}.npz"), Z=rng.randn(n_cand, nrow, 8).astype(np.float32),
                     y=np.array([c for _ in doms for c in range(4) for _ in range(3)], dtype=np.int64),
                     d=np.array([str(i) for i in range(len(doms)) for _ in range(12)], dtype=object),
                     group=np.array([f"g{i}" for i in range(nrow)], dtype=object),
                     sample_id=np.array([f"{dz}-{i}" for i in range(nrow)], dtype=object))
        for name, doms in (("support-source.npz", gdoms), ("support-audit.npz", adoms)):
            counts = np.full((len(doms), 4), 12, dtype=np.int64)
            np.savez(os.path.join(fdir, name), counts=counts, cell_mass=counts.astype(float) / counts.sum(),
                     m=np.int64(3), reference_prior=np.ones(4) / 4,
                     domain_names=np.array(doms, dtype=object), class_names=np.array([str(c) for c in range(4)], dtype=object))
        json.dump({"critic": {"capacities": [0], "l2_C": 1.0, "max_iter": 50, "prob_floor": 1e-6,
                              "feature_seed_base": 10000}, "selection_n_folds": 2, "audit_n_folds": 2,
                   "sel_ok": True, "sa_ok": True}, open(os.path.join(fdir, "config.json"), "w"))
        cand_meta = [{"index": ci, "origin": ("ERM" if ci == 0 else "OACI"), "model_hash": f"m{t}{ci}",
                      "epoch": 10 * ci, "lambda": 0.1, "R_src": 0.8, "balanced_err": 0.3, "train_surrogate": 1.1,
                      "feasible": True, "is_erm": ci == 0} for ci in range(n_cand)]
        json.dump(cand_meta, open(os.path.join(fdir, "cand_meta.json"), "w"))
        # s0 scalars: worst-domain via the SAME recompute path (guarantees self-consistency); leakage placeholder
        fld = ssr.load_fold_level(root, 0, t, 0)
        s0 = []
        for ci, cm in enumerate(cand_meta):
            rec = ssr.recompute_candidate(fld, ci, {}, {}, edges=edges, classes=list(range(4)), with_leakage=False)
            s0.append({**cm, **{k: rec[k] for k in schema.RECOMPUTABLE_UNDER_MASK if k in rec},
                       "selection_leakage_point": 1.0, "audit_leakage_point": 0.5})
        json.dump(s0, open(os.path.join(fdir, "s0_scalars.json"), "w"))
        # fake C10 fold: candidates with target metrics (ERM ref) + selected
        os.makedirs(c10root, exist_ok=True)
        cs = [{"model_hash": f"m{t}{ci}", "is_erm": ci == 0, "feasible": True,
               "target_worst_bacc": (0.55 if ci % 2 else 0.50), "target_worst_nll": 1.1, "target_worst_ece": 0.1}
              for ci in range(n_cand)]
        json.dump({"seed": 0, "target": t, "levels": {"0": {"candidates": cs, "n_candidates": n_cand,
                   "selected": {"ERM": f"m{t}0", "OACI": f"m{t}1"}}}},
                  open(os.path.join(c10root, f"seed-0-target-{t:03d}.json"), "w"))


def test_s0_recompute_self_consistent_and_pipeline_runs(tmp_path=None):
    import tempfile
    d = tempfile.mkdtemp()
    ex = os.path.join(d, "ex"); c10 = os.path.join(d, "c10")
    _write_fake_extraction(ex, c10)
    # S0 recompute-from-logits == extracted s0_scalars (the mechanism guaranteeing S0 reproduces C17)
    sc = ssr.self_check_s0(ex, 0, 1)
    assert sc["all_ok"] and sc["worst_diff"] == 0.0
    # build a regime atlas (masks applied) and run the mask-stress probe end-to-end
    from oaci.support_stress.identifiability_stress import mask_stress_probe
    m = mask_stress_probe(ex, c10, "S2_rare_cells", boundary_classes=(1, 3), n_perm=10)
    assert m["regime"] == "S2_rare_cells" and m["n_used"] >= 0 and "loto_auc" in m


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} c18-support-stress tests")


if __name__ == "__main__":
    _run_all()
