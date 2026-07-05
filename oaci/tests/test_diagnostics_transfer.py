"""C10 Part-1 artifact-only diagnostics: transfer/optimism computations, order-invariance, canonical-JSON
serializability, case A/B/C scaffold, and the no-cross-package-import guard. Synthetic fold records (the
c8_loader output shape) so the tests need no real LOSO root."""
from __future__ import annotations

from oaci.diagnostics.report import to_json, render_report_md
from oaci.diagnostics.transfer import (audit_to_target_transfer, run_all_transfer, selection_to_audit_optimism)

_ROLE_KEYS = ("worst_bacc", "worst_nll", "worst_ece", "mean_bacc", "mean_nll", "mean_ece",
              "pooled_bacc", "pooled_nll", "pooled_ece", "reference_status")


def _roles(worst_bacc, worst_nll, worst_ece=0.1):
    def one(b, n, e):
        return {"worst_bacc": b, "worst_nll": n, "worst_ece": e, "mean_bacc": b, "mean_nll": n, "mean_ece": e,
                "pooled_bacc": b, "pooled_nll": n, "pooled_ece": e, "reference_status": "estimable"}
    return {"source_audit": one(worst_bacc, worst_nll, worst_ece),
            "source_guard": one(worst_bacc + 0.02, worst_nll - 0.02, worst_ece),
            "target_audit": one(worst_bacc, worst_nll, worst_ece)}


def _method(method, *, sel_ucl, audit_ucl, tb, tn, epoch=100, lam=1.0, rsrc=0.8):
    return {"method": method, "level": 0, "active": True, "selected_erm": method == "ERM",
            "used_erm_fallback": False, "selection_reason": "stage2_best", "selection_status": "estimable",
            "selected_epoch": epoch, "selection_score": sel_ucl, "R_src": rsrc, "n_feasible": 20,
            "selected_model_hash": f"{method}h", "selected_lambda": (None if method == "ERM" else lam),
            "sel_leakage": {"bootstrap_ucl": sel_ucl, "L_abs": sel_ucl, "extractable_LQ_ov": sel_ucl,
                            "percentile_ucl": sel_ucl},
            "audit_leakage": {"bootstrap_ucl": audit_ucl, "L_abs": audit_ucl, "extractable_LQ_ov": audit_ucl},
            "roles": _roles(tb, tn)}


def _fold(seed, target, *, oaci_sel, oaci_audit, oaci_tb, oaci_tn, erm_sel=1.7, erm_audit=0.8,
          erm_tb=0.50, erm_tn=1.20, lam=1.0):
    def lvl():
        md = {"ERM": _method("ERM", sel_ucl=erm_sel, audit_ucl=erm_audit, tb=erm_tb, tn=erm_tn),
              "OACI": _method("OACI", sel_ucl=oaci_sel, audit_ucl=oaci_audit, tb=oaci_tb, tn=oaci_tn, lam=lam),
              "global_lpc": _method("global_lpc", sel_ucl=1.5, audit_ucl=0.82, tb=erm_tb - 0.01, tn=erm_tn),
              "uniform": _method("uniform", sel_ucl=1.5, audit_ucl=0.82, tb=erm_tb - 0.01, tn=erm_tn)}
        return {"k1": {"status": "stop_no_detectable_heldout_leakage_reduction", "observed_delta": 0.01,
                       "p_lower": 0.3, "p_two_sided": 0.6}, "methods": md}
    return {"seed": seed, "target": target, "artifact_dir": f"/x/s{seed}t{target}",
            "levels": {0: lvl(), 1: lvl()}}


def _optimism_folds():
    """OACI always cuts selection leakage (−0.5) but audit is unchanged/noisy → selection-optimism."""
    return [_fold(s, t, oaci_sel=1.2, oaci_audit=(0.80 + (0.02 if (s + t) % 2 else -0.02)),
                  oaci_tb=0.50 - 0.005 * ((s + t) % 3), oaci_tn=1.15) for s in (0, 1, 2) for t in range(1, 10)]


def test_selection_to_audit_optimism_computation():
    o = selection_to_audit_optimism(_optimism_folds())
    assert o["n_fold_levels"] == 54
    assert o["delta_selection_leakage"]["mean"] < -0.4          # sel leakage cut everywhere
    assert o["n_selection_reduced"] == 54
    assert abs(o["delta_audit_leakage"]["mean"]) < 0.05         # audit barely moves
    assert o["n_audit_reduced"] < 54                            # not universal ⇒ optimism


def test_audit_leakage_to_target_metric_correlation():
    # construct a clean POSITIVE mechanism: more audit-leakage reduction ⇒ more target bAcc gain
    folds = []
    for i, (s, t) in enumerate([(s, t) for s in (0, 1, 2) for t in range(1, 10)]):
        red = 0.01 * (i - 27)                                   # ranges negative..positive
        folds.append(_fold(s, t, oaci_sel=1.2, oaci_audit=0.8 + red, oaci_tb=0.50 - red, oaci_tn=1.2))
    a = audit_to_target_transfer(folds)
    r = a["corr_audit_vs_target_worst_bacc"]["pearson"]["r"]
    assert r is not None and r < -0.9                           # Δaudit<0 ↔ Δbacc>0 ⇒ strong negative corr


def test_transfer_metrics_are_order_invariant():
    folds = _optimism_folds()
    a = run_all_transfer(folds)
    b = run_all_transfer(list(reversed(folds)))
    assert a["selection_to_audit_optimism"]["delta_audit_leakage"]["mean"] == \
        b["selection_to_audit_optimism"]["delta_audit_leakage"]["mean"]
    assert a["harm_localization"]["total_bacc_loss"] == b["harm_localization"]["total_bacc_loss"]
    assert a["method_comparison"]["ERM"]["n_bacc_harmed"] == b["method_comparison"]["ERM"]["n_bacc_harmed"]


def test_c10_report_is_canonical_json_serializable():
    from oaci.artifacts.canonical_json import canonical_json_bytes
    folds = _optimism_folds()
    js = to_json(folds, run_all_transfer(folds))
    blob = canonical_json_bytes(js)                             # raises on any int mapping key
    assert blob and b'"part1_transfer"' in blob
    assert isinstance(render_report_md(folds, run_all_transfer(folds)), str)


def test_c10_report_contains_case_ABC_decision_fields():
    folds = _optimism_folds()
    js = to_json(folds, run_all_transfer(folds))
    cd = js["case_determination"]
    assert "final_case" in cd and cd["final_case"] is None      # decided in C10b
    for k in ("A_source_only_selector_works", "B_only_source_audit_oracle_works", "C_oracle_also_fails"):
        assert k in cd["decision_rule"]
    assert js["part1_case_lean"]["part1_lean"] in ("case_C_candidate", "ambiguous_needs_part2")


def test_no_oaci_diagnostics_import_from_cmi_or_h2cmi():
    import sys
    import oaci.diagnostics.c8_loader, oaci.diagnostics.transfer, oaci.diagnostics.report  # noqa: F401
    leaked = [m for m in sys.modules if m == "cmi" or m.startswith("cmi.")
              or m == "h2cmi" or m.startswith("h2cmi.")]
    assert leaked == [], f"diagnostics must not import cmi/h2cmi at runtime: {leaked}"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} c10-diagnostics tests")


if __name__ == "__main__":
    _run_all()
