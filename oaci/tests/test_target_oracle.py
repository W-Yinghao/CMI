"""C16-A target-oracle ceiling: the C1/C2/C3 taxonomy, non-deployable labeling, canonical serialization.
Synthetic 27-fold replay tables (3 seeds x 9 targets x 2 levels)."""
from __future__ import annotations

from oaci.artifacts.canonical_json import canonical_json_bytes
from oaci.mechanism.target_oracle import run_target_oracle


def _cand(h, *, is_erm, tb, tn, te=0.1, sa=0.5, feasible=True):
    return {"model_hash": h, "is_erm": is_erm, "feasible": feasible, "epoch": 100,
            "target_worst_bacc": tb, "target_worst_nll": tn, "target_worst_ece": te, "source_audit_worst_bacc": sa}


def _folds(make_cands):
    out = []
    for s in (0, 1, 2):
        for t in range(1, 10):
            out.append({"seed": s, "target": t, "levels": {str(L): {"candidates": make_cands(s, t, L),
                                                                    "selected": {"ERM": "erm", "OACI": "erm"}}
                                                           for L in (0, 1)}})
    return out


def test_case_C2_trajectory_failure():
    # every OACI candidate is worse than ERM on target -> target oracle can't beat ERM
    def mk(s, t, L):
        return [_cand("erm", is_erm=True, tb=0.50, tn=1.20), _cand("o1", is_erm=False, tb=0.45, tn=1.40, sa=0.6)]
    r = run_target_oracle(_folds(mk))
    assert r["case_label"] == "C2_trajectory_failure"
    assert not r["target_oracle_rescues_bacc"] and not r["source_oracle_rescues"]


def test_case_C1_source_observability_failure():
    # a target-good OACI checkpoint (better bAcc AND NLL) exists but has LOW source_audit bAcc (invisible to
    # the source oracle); a decoy has high source_audit bAcc but bad target
    def mk(s, t, L):
        return [_cand("erm", is_erm=True, tb=0.50, tn=1.20, sa=0.50),
                _cand("good", is_erm=False, tb=0.60, tn=1.00, sa=0.40),      # great target, poor source signal
                _cand("decoy", is_erm=False, tb=0.44, tn=1.50, sa=0.90)]     # best source signal, bad target
    r = run_target_oracle(_folds(mk))
    assert r["target_oracle_joint_k2"] == "reproducible_gain"
    assert not r["source_oracle_rescues"] and r["case_label"] == "C1_source_observability_failure"


def test_case_C3_calibration_not_discrimination():
    # target-accuracy-good checkpoint exists (bAcc up) but its NLL is worse -> joint fails
    def mk(s, t, L):
        return [_cand("erm", is_erm=True, tb=0.50, tn=1.20, sa=0.50),
                _cand("acc", is_erm=False, tb=0.56, tn=1.45, sa=0.40)]       # bAcc up, NLL up
    r = run_target_oracle(_folds(mk))
    assert r["target_oracle_rescues_bacc"] and not r["target_oracle_rescues_joint"]
    assert r["case_label"] == "C3_calibration_not_discrimination"


def test_target_oracle_is_labeled_non_deployable():
    def mk(s, t, L):
        return [_cand("erm", is_erm=True, tb=0.5, tn=1.2), _cand("o", is_erm=False, tb=0.55, tn=1.1, sa=0.4)]
    r = run_target_oracle(_folds(mk))
    for name, sv in r["selectors"].items():
        assert sv["non_deployable"] == name.startswith("target_oracle")
    assert canonical_json_bytes(r)                                          # serializes


def test_target_oracle_is_order_invariant():
    def mk(s, t, L):
        return [_cand("erm", is_erm=True, tb=0.5, tn=1.2), _cand("a", is_erm=False, tb=0.55, tn=1.0, sa=0.6),
                _cand("b", is_erm=False, tb=0.55, tn=1.0, sa=0.4)]
    a = run_target_oracle(_folds(mk))["case_label"]
    fr = _folds(mk)
    rev = [{**f, "levels": {L: {**lv, "candidates": list(reversed(lv["candidates"]))} for L, lv in f["levels"].items()}}
           for f in reversed(fr)]
    assert run_target_oracle(rev)["case_label"] == a


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} target-oracle tests")


if __name__ == "__main__":
    _run_all()
