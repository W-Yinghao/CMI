"""C8 multi-seed aggregation logic (K1 counts + real multi-seed K2). Synthetic fold records. Standalone."""
from __future__ import annotations

from oaci.confirmatory.c8_aggregate import aggregate_c8, render_c8_report_md

_DET = "leakage_reduction_detected"
_STOP = "stop_no_detectable_heldout_leakage_reduction"


def _fold(seed, target, *, deep=True, tfit=True, fam="FAM", prov="PROV", k1=_STOP,
          eb=0.40, ob=0.42, en=1.20, on=1.10):
    levels = {L: {"k1": {"k1_status": k1, "observed_delta": 0.01, "p_lower": 0.30, "permutation_plan_hash": "ph"},
                  "worst": {"ERM": {"bacc": eb, "nll": en}, "OACI": {"bacc": ob, "nll": on}}} for L in (0, 1)}
    return {"seed": seed, "target": target, "deep_verification_ok": deep, "target_fit_empty": tfit,
            "protocol_family": fam, "provenance_hash": prov, "context_hash": f"c{seed}{target}", "levels": levels}


def _all(**kw):
    return [_fold(s, t, **kw) for s in (0, 1, 2) for t in range(1, 10)]


def test_c8_aggregate_requires_27_folds_one_family_one_provenance():
    r = aggregate_c8(_all(), seeds=[0, 1, 2])
    assert r["n_folds"] == 27 and r["seeds"] == [0, 1, 2] and r["targets"] == list(range(1, 10))
    assert r["protocol_family"] == "FAM" and r["provenance_hash"] == "PROV"
    for bad in (_all()[:-1], [dict(_fold(0, 1), provenance_hash="X")] + _all()[1:],
                [dict(_fold(0, 1), deep_verification_ok=False)] + _all()[1:],
                [dict(_fold(0, 1), target_fit_empty=False)] + _all()[1:]):
        try:
            aggregate_c8(bad, seeds=[0, 1, 2])
        except ValueError:
            continue
        raise AssertionError("bad fold set must be rejected")


def test_c8_k1_counts_per_level():
    folds = _all(k1=_STOP)
    folds[0]["levels"][0]["k1"]["k1_status"] = _DET                # one detected at (seed0,target1,level0)
    r = aggregate_c8(folds, seeds=[0, 1, 2])
    assert r["k1_counts"][0]["leakage_reduction_detected"] == 1 and r["k1_counts"][0]["n"] == 27
    assert r["k1_counts"][1]["stop_no_detectable_heldout_leakage_reduction"] == 27
    assert len(r["k1_per_fold"]) == 54                             # 27 folds × 2 levels


def test_c8_k2_reproducible_gain_across_seeds():
    r = aggregate_c8(_all(eb=0.40, ob=0.42, en=1.20, on=1.10), seeds=[0, 1, 2])   # OACI better everywhere
    assert r["k2"]["k2_status"] == "reproducible_gain" and r["k2"]["n_seeds"] == 3
    assert len(r["k2_units"]) == 6                                 # 3 seeds × 2 levels


def test_c8_k2_stops_when_no_reproducible_gain():
    r = aggregate_c8(_all(eb=0.45, ob=0.40, en=1.10, on=1.25), seeds=[0, 1, 2])   # OACI worse everywhere
    assert r["k2"]["k2_status"] == "stop_no_reproducible_gain"


def test_c8_worst_target_uses_min_bacc_max_nll():
    folds = _all(ob=0.50, on=1.00)
    folds[0]["levels"][0]["worst"]["OACI"] = {"bacc": 0.20, "nll": 2.00}          # one bad target drags worst
    r = aggregate_c8(folds, seeds=[0, 1, 2])
    u = next(u for u in r["k2_units"] if u["seed"] == 0 and u["level"] == 0)
    # worst OACI bAcc = 0.20 (< ERM 0.40) -> Δ negative; worst OACI NLL = 2.00 (> ERM 1.20) -> Δ positive
    assert u["deltas"]["worst_domain_bacc"] < 0 and u["deltas"]["worst_domain_nll"] > 0


def test_c8_report_md_has_k1_and_k2():
    md = render_c8_report_md(aggregate_c8(_all(), seeds=[0, 1, 2]))
    assert "## K1" in md and "## K2" in md and "minimum-seed" in md
    assert "detected" in md and "required_min_seeds" in md


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} c8-aggregate tests")


if __name__ == "__main__":
    _run_all()
