"""C10b counterfactual selector replay: role-gating (no selector reads target; S1–S4 never read source_audit),
conservative ERM fallback, oracle can pick a non-current checkpoint, K2 aggregation + case A/B/C, order-
invariance, canonical serializability. Synthetic candidate tables (candidate_replay output schema) — no GPU.
The byte-exact GPU replay identity is proven separately by the identity probe and asserted in-module."""
from __future__ import annotations

from oaci.diagnostics.report import aggregate_selector_replay
from oaci.diagnostics.selectors import (Gated, AccessLog, SELECTORS, run_selectors_on_level, s4_conservative_source_only,
                                        s5_source_audit_oracle)


def _cand(h, *, is_erm=False, feasible=True, sel_leak=1.0, sg_bacc=0.5, sg_nll=1.2, sg_ece=0.1,
          sa_bacc=0.5, sa_nll=1.2, sa_leak=0.8, tb=0.5, tn=1.2, epoch=100):
    return {"origin": "ERM" if is_erm else "OACI", "model_hash": h, "epoch": epoch, "lambda": 1.0,
            "R_src": 0.8, "balanced_err": 0.3, "train_surrogate": 1.1, "feasible": feasible, "is_erm": is_erm,
            "source_guard_worst_bacc": sg_bacc, "source_guard_worst_nll": sg_nll, "source_guard_worst_ece": sg_ece,
            "source_guard_pred_hash": f"sg{h}", "source_audit_worst_bacc": sa_bacc, "source_audit_worst_nll": sa_nll,
            "source_audit_worst_ece": 0.1, "source_audit_pred_hash": f"sa{h}",
            "target_worst_bacc": tb, "target_worst_nll": tn, "target_worst_ece": 0.1, "target_pred_hash": f"t{h}",
            "selection_leakage_point": sel_leak, "audit_leakage_point": sa_leak}


# ---- role gating ----
def test_source_only_selectors_cannot_read_source_audit_or_target():
    rows = [_cand("erm", is_erm=True), _cand("o1", sel_leak=0.5)]
    res = run_selectors_on_level(rows, selected_oaci_hash="o1")
    for name in ("S1_leakage_worst_source_bacc", "S2_leakage_worst_source_nll", "S3_leakage_calibration",
                 "S4_conservative_source_only", "S0_current"):
        a = res[name]["access"]
        assert not a["target_read"] and "target" not in a["roles_actually_read"]
        assert "source_audit" not in a["roles_actually_read"], f"{name} read source_audit"
        assert a["forbidden_fields"] == []


def test_source_audit_oracle_cannot_read_target():
    res = run_selectors_on_level([_cand("erm", is_erm=True), _cand("o1")], selected_oaci_hash="o1")
    a = res["S5_source_audit_oracle"]["access"]
    assert not a["target_read"] and "target" not in a["roles_actually_read"] and a["forbidden_fields"] == []
    assert "source_audit" in a["roles_actually_read"]                     # oracle DOES use source_audit


def test_gated_raises_on_forbidden_field():
    log = AccessLog({"source_guard"})
    g = Gated(_cand("x"), log)
    assert g["source_guard_worst_bacc"] == 0.5                            # allowed
    try:
        _ = g["target_worst_bacc"]
    except PermissionError:
        assert log.forbidden == ["target_worst_bacc"]
        return
    raise AssertionError("reading a target field under a source_guard-only gate must raise")


def test_target_metrics_are_evaluation_only():
    # target metrics appear in the CHOICE result (post-selection eval) but never in any access read set
    res = run_selectors_on_level([_cand("erm", is_erm=True, tb=0.4), _cand("o1", tb=0.6)], selected_oaci_hash="o1")
    for name, c in res.items():
        assert "target_worst_bacc" in c and not c["access"]["target_read"]


def test_conservative_selector_falls_back_to_erm():
    # no OACI candidate beats ERM leakage by margin -> S4 returns ERM
    rows = [_cand("erm", is_erm=True, sel_leak=0.9), _cand("o1", sel_leak=0.88)]   # improvement < margin 0.05
    h, log = s4_conservative_source_only(rows, {"margins": {"bacc": .02, "nll": .05, "ece": .02, "leakage": .05}})
    assert h == "erm"
    # a big leakage improvement + guards pass -> picks OACI
    rows2 = [_cand("erm", is_erm=True, sel_leak=0.9, sg_bacc=0.5), _cand("o2", sel_leak=0.5, sg_bacc=0.51)]
    h2, _ = s4_conservative_source_only(rows2, {"margins": {"bacc": .02, "nll": .05, "ece": .02, "leakage": .05}})
    assert h2 == "o2"


def test_oracle_selector_can_choose_non_current_checkpoint():
    # o2 has the best source_audit worst bAcc -> oracle picks it even though current selection is o1
    rows = [_cand("erm", is_erm=True, sa_bacc=0.50), _cand("o1", sa_bacc=0.52), _cand("o2", sa_bacc=0.66)]
    h, _ = s5_source_audit_oracle(rows, {})
    assert h == "o2"


# ---- K2 aggregation + case A/B/C ----
def _replay(make_cands):
    folds = []
    for s in (0, 1, 2):
        for t in range(1, 10):
            cands = make_cands(s, t)
            for L in (0, 1):
                pass
            folds.append({"seed": s, "target": t, "artifact_dir": f"/x/s{s}t{t}", "identity": [],
                          "levels": {str(L): {"n_candidates": len(cands), "candidates": cands,
                                              "selected": {"ERM": "erm", "OACI": "o1"}} for L in (0, 1)}})
    return folds


def test_case_C_when_no_selector_beats_erm():
    # every OACI candidate has worse target than ERM -> no selector reproduces gain -> case C
    def mk(s, t):
        return [_cand("erm", is_erm=True, tb=0.60, tn=1.00, sa_bacc=0.55),
                _cand("o1", sel_leak=0.5, tb=0.50, tn=1.20, sa_bacc=0.50)]
    agg = aggregate_selector_replay(_replay(mk))
    assert agg["final_case"] == "C_oracle_also_fails"
    assert agg["selectors"]["S0_current"]["k2_status"] == "stop_no_reproducible_gain"
    assert agg["access_invariants_ok"]


def test_case_A_when_source_only_selector_reproduces_gain():
    # a source-guard-passing, low-leakage OACI candidate ALSO has better target everywhere -> S1..S4 gain
    def mk(s, t):
        return [_cand("erm", is_erm=True, sel_leak=0.9, sg_bacc=0.50, tb=0.50, tn=1.20),
                _cand("good", sel_leak=0.4, sg_bacc=0.52, tb=0.60, tn=1.00, sa_bacc=0.60)]
    agg = aggregate_selector_replay(_replay(mk))
    assert agg["final_case"] == "A_source_only_selector_works"
    assert "S1_leakage_worst_source_bacc" in agg["source_only_reproducible"]


def test_case_B_when_only_oracle_reproduces_gain():
    # the good-target checkpoint is hidden from source-only signal (worse source_guard, higher leakage)
    # but visible to the source_audit oracle (best source_audit bAcc)
    def mk(s, t):
        return [_cand("erm", is_erm=True, sel_leak=0.5, sg_bacc=0.55, tb=0.50, tn=1.20, sa_bacc=0.50),
                _cand("hidden", sel_leak=0.9, sg_bacc=0.40, tb=0.62, tn=1.00, sa_bacc=0.70)]
    agg = aggregate_selector_replay(_replay(mk))
    assert agg["final_case"] == "B_only_source_audit_oracle_works"
    assert agg["oracle_reproducible"] and not agg["source_only_reproducible"]


def test_selector_replay_is_order_invariant():
    def mk(s, t):
        return [_cand("erm", is_erm=True, tb=0.5), _cand("o1", sel_leak=0.5, tb=0.55), _cand("o2", sel_leak=0.7, tb=0.6)]
    folds = _replay(mk)
    a = aggregate_selector_replay(folds)
    rev = [{**f, "levels": {L: {**lv, "candidates": list(reversed(lv["candidates"]))}
                            for L, lv in f["levels"].items()}} for f in reversed(folds)]
    b = aggregate_selector_replay(rev)
    assert a["final_case"] == b["final_case"]
    assert a["selectors"]["S5_source_audit_oracle"]["k2_status"] == b["selectors"]["S5_source_audit_oracle"]["k2_status"]


def test_selector_replay_summary_is_canonical_serializable():
    from oaci.artifacts.canonical_json import canonical_json_bytes
    def mk(s, t):
        return [_cand("erm", is_erm=True), _cand("o1", sel_leak=0.5)]
    agg = aggregate_selector_replay(_replay(mk))
    blob = canonical_json_bytes(agg)
    assert blob and b'"final_case"' in blob and b'"selectors"' in blob


def test_all_six_selectors_present_and_access_invariants_hold():
    agg = aggregate_selector_replay(_replay(lambda s, t: [_cand("erm", is_erm=True), _cand("o1", sel_leak=0.5)]))
    assert set(agg["selectors"]) == set(SELECTORS) and len(SELECTORS) == 6
    assert agg["access_invariants_ok"]


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} c10-selector-replay tests")


if __name__ == "__main__":
    _run_all()
