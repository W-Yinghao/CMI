"""C14 falsification battery: G0 integrity gating, combining K1/K2/oracle/anti-transfer into the verdict,
order-invariance, weak-nominal-K1 not counted as success, oracle-failure closure. Synthetic C8/C10/C12
evidence dicts (only the fields the gates read)."""
from __future__ import annotations

from oaci.falsification.battery import final_verdict, run_battery


def c8(*, k2="stop_no_reproducible_gain", sweep="stop_no_detectable_heldout_leakage_reduction",
       n_nom=11, n_bh=0, n_bonf=0, deep=True, tfit=True, nfolds=27):
    return {"all_deep_verified": deep, "all_target_fit_empty": tfit, "n_folds": nfolds,
            "k1_overall": {"k1_sweep_status": sweep, "n_leakage_reduction_detected": n_nom, "n_tests": 54,
                           "observed_delta_mean": -0.03,
                           "multiplicity": {"n_bh_survive": n_bh, "n_bonferroni_survive": n_bonf}},
            "k2": {"k2_status": k2, "reproduced_endpoints": None}, "k2_agg": {}}


def c10(*, oracle=False, identity_pass=True, flips=0, s5="stop_no_reproducible_gain", src_opt=True):
    return {"part1_transfer": {
        "selection_to_audit_optimism": {"delta_selection_leakage": {"mean": (-0.326 if src_opt else -0.01)},
                                        "delta_audit_leakage": {"mean": (0.008 if src_opt else -0.009)},
                                        "corr_selection_vs_audit_delta": {"pearson": {"r": (0.004 if src_opt else 0.9)}},
                                        "n_selection_reduced": 54, "n_audit_reduced": 25, "n_fold_levels": 54},
        "audit_to_target_transfer": {"corr_audit_vs_target_worst_bacc": {"pearson": {"r": -0.06}},
                                     "corr_audit_vs_target_worst_nll": {"pearson": {"r": -0.13}}}},
        "part2_selector_replay": {
            "identity": {"all_pass": identity_pass, "total_argmax_flips": flips, "max_logit_diff": 1.8e-15,
                         "n_all_match": 216, "n_checks": 216, "n_byte_hash_match": 64, "n_numeric_only": 152},
            "selectors": {"S0_current": {"k2_status": "stop_no_reproducible_gain"},
                          "S1_leakage_worst_source_bacc": {"k2_status": "stop_no_reproducible_gain"},
                          "S5_source_audit_oracle": {"k2_status": s5}},
            "oracle_reproducible": oracle, "source_only_reproducible": ([] if not oracle else []),
            "s0_current_k2": "stop_no_reproducible_gain", "final_case": "C_oracle_also_fails"}}


def _cell(t, te, lv, *, ssg, esg, dnll, dbacc, blow, fb=False):
    return {"target": t, "temp": te, "level": lv, "src_source_guard_nll": ssg, "erm_source_guard_nll": esg,
            "d_nll_vs_erm": dnll, "d_bacc_vs_erm": dbacc, "src_fallback_erm": fb, "target_nll_blowup": blow,
            "erm_target_nll": 1.2, "src_target_nll": 1.2 + (dnll or 0.0)}


def c12(cells, *, verdict="stop_SRC_pivot_measurement_only"):
    active = [c for c in cells if not c["src_fallback_erm"]]
    nblow = sum(c["target_nll_blowup"] for c in cells)
    nnt = sum(1 for c in active if c["src_source_guard_nll"] < c["erm_source_guard_nll"] and c["d_nll_vs_erm"] > 0)
    return {"cells": cells, "targets": sorted({c["target"] for c in cells}), "temperatures": sorted({c["temp"] for c in cells}),
            "verdict": {"verdict": verdict, "n_target_nll_blowup": nblow, "n_source_improved_not_transferred": nnt,
                        "n_fallback": sum(1 for c in cells if c["src_fallback_erm"]), "n_cells": len(cells),
                        "n_active": len(active)}}


_ANTI = [_cell(1, 0.1, 0, ssg=1.10, esg=1.20, dnll=+0.9, dbacc=-0.05, blow=True),
         _cell(3, 0.1, 0, ssg=1.12, esg=1.20, dnll=+0.2, dbacc=-0.01, blow=False),
         _cell(1, 0.1, 1, ssg=1.20, esg=1.20, dnll=0.0, dbacc=0.0, blow=False, fb=True)]


def test_battery_requires_deep_verified_artifacts():
    bat = run_battery(c8(deep=False), c10(), c12(_ANTI))
    assert bat["gates"]["G0_integrity"]["status"] == "invalid_evidence"
    assert bat["verdict"]["control_hypothesis_status"] == "invalid_evidence"


def test_battery_rejects_target_fit_ids():
    bat = run_battery(c8(tfit=False), c10(), c12(_ANTI))
    assert bat["gates"]["G0_integrity"]["status"] == "invalid_evidence"
    # also a replay-identity failure invalidates
    bat2 = run_battery(c8(), c10(identity_pass=False, flips=3), c12(_ANTI))
    assert bat2["gates"]["G0_integrity"]["status"] == "invalid_evidence"


def test_battery_combines_k1_k2_oracle_and_antitransfer():
    bat = run_battery(c8(), c10(), c12(_ANTI))
    r = bat["verdict"]["falsification_reasons"]
    assert set(r) == {"falsified_by_no_endpoint_transfer", "falsified_by_oracle_failure",
                      "falsified_by_source_target_antitransfer"}
    assert bat["verdict"]["control_hypothesis_status"] == "falsified"


def test_battery_verdict_is_order_invariant():
    bat = run_battery(c8(), c10(), c12(_ANTI))
    gm = bat["gates"]
    # verdict depends only on the gate SET, not the presentation order
    import collections
    shuffled = collections.OrderedDict(reversed(list(gm.items())))
    assert final_verdict(dict(shuffled)) == bat["verdict"]


def test_weak_nominal_k1_signal_is_not_reported_as_success():
    bat = run_battery(c8(n_nom=11, n_bh=0), c10(), c12(_ANTI))
    g2 = bat["gates"]["G2_heldout_leakage"]
    assert g2["status"] == "weak_nominal_nonmultiplicity_signal" and g2["weak_nominal_signal"] is True
    assert "heldout_leakage_reduction_detected" not in g2["status"]      # weak != detected/success


def test_oracle_failure_triggers_control_hypothesis_closure():
    bat = run_battery(c8(k2="reproducible_gain"), c10(oracle=False, s5="stop_no_reproducible_gain"), c12(_ANTI))
    # even if K2 nominally 'gain', an oracle failure still contributes a falsification reason
    assert bat["gates"]["G4_oracle_rescue"]["status"] == "oracle_fails_to_rescue"
    assert "falsified_by_oracle_failure" in bat["verdict"]["falsification_reasons"]
    # and a clean supported case: K2 gain + oracle rescue + no anti-transfer
    clean = c12([_cell(1, 0.1, 0, ssg=1.10, esg=1.20, dnll=-0.1, dbacc=+0.05, blow=False)])
    good = run_battery(c8(k2="reproducible_gain", sweep="leakage_reduction_detected", n_nom=40, n_bh=20),
                       c10(oracle=True, s5="reproducible_gain"), clean)
    assert good["verdict"]["control_hypothesis_status"] == "control_hypothesis_supported"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} falsification-battery tests")


if __name__ == "__main__":
    _run_all()
