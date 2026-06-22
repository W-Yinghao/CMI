"""Phase 1.2.4/1.2.5 -- calibrate the learned conditional-task-safety gate against the EXACT
Bayes conditional task delta, computed from the TRUE mixture params on an INDEPENDENT MC draw
(with CI), and classified SAFE / UNSAFE / BAYES_AMBIGUOUS vs delta_Y.

Settles the 'safe-span bias' question: the synergy 'safe' span is CONDITIONALLY UNSAFE (shared
domain factor -> explaining-away), so the gate's nonzero value is real information, not bias --
de-biasing would be anti-conservative. The factorized case is genuinely safe. Headline safety
check: the gate must NOT ACCEPT a deletion the Bayes oracle classifies UNSAFE (no slack)."""
from dataclasses import replace
import numpy as np
import torch

torch.set_num_threads(1)

from tos_cmi.data.synthetic import make_partial_synergy, make_partial_factorized
from tos_cmi.score_fisher import (ScoreFisherConfig, select_score_fisher, _metric,
                                  task_protected_projector)
from tos_cmi.eval.bayes_oracle import bayes_conditional_task_delta, classify_safety


def _cfg():
    return ScoreFisherConfig(epochs=200, hidden=64, gate_boot=200, n_perm_null=2, task_protect=True)


def _bayes(data, P, delta_Y):
    t = data["truth"]
    r = bayes_conditional_task_delta(t["mu_yd"], t["sigma"], t["py"], t["pdy"], P, n_mc=20000)
    r["verdict"] = classify_safety(r["ci_lo"], r["ci_hi"], delta_Y)
    return r


def test_bayes_oracle_distinguishes_synergy_from_factorized():
    """Deleting the geometrically task-orthogonal safe span: synergy is conditionally UNSAFE,
    factorized is genuinely safe -- on an independent MC draw with a CI (no same-sample plug-in)."""
    cfg = _cfg()
    for name, mk, want in [("synergy", make_partial_synergy, "UNSAFE"),
                           ("factorized", make_partial_factorized, "SAFE")]:
        data = mk(n=8000, seed=0); s = data["spec"]
        Z = data["Z"].astype(np.float64); M = _metric(Z, data["y"], s.n_cls, cfg)
        P, _ = task_protected_projector(data["nuisance_basis"], data["task_overlap_basis"], M)
        r = _bayes(data, P, cfg.delta_Y)
        print("Bayes [%s]:" % name, {"delta": round(r["delta"], 4),
              "ci": (round(r["ci_lo"], 4), round(r["ci_hi"], 4)), "verdict": r["verdict"]})
        assert r["verdict"] == want, (name, r)
    print("test_bayes_oracle_distinguishes_synergy_from_factorized: OK")


def _gate_vs_bayes(mk, seed=0):
    data = mk(n=6000, seed=seed); s = data["spec"]; cfg = _cfg()
    rep = select_score_fisher(data["Z"], data["y"], data["d"], s.n_cls, s.n_dom, cfg, seed=seed)
    r = _bayes(data, rep.P, cfg.delta_Y)
    return rep, r, cfg


def test_gate_no_unsafe_acceptance():
    """SAFETY: when the gate ACCEPTS a deletion (k>=1), the deployed projector must NOT be Bayes
    UNSAFE (no slack). The deployed-projector Bayes delta is evaluated against delta_Y."""
    for name, mk in [("factorized", make_partial_factorized), ("synergy", make_partial_synergy)]:
        rep, r, cfg = _gate_vs_bayes(mk)
        print("gate vs bayes [%s]:" % name, {"k": rep.k_star, "reason": rep.decision_reason,
              "bayes_delta": round(r["delta"], 4), "verdict": r["verdict"]})
        if rep.k_star >= 1:
            assert r["verdict"] != "UNSAFE", ("UNSAFE ACCEPTANCE", name, rep.k_star, r)
    print("test_gate_no_unsafe_acceptance: OK")


def test_gate_accepts_factorized_safe():
    rep, r, cfg = _gate_vs_bayes(make_partial_factorized)
    print("factorized gate:", {"k": rep.k_star, "reason": rep.decision_reason,
          "bayes_delta": round(r["delta"], 4), "verdict": r["verdict"]})
    assert rep.k_star >= 1, (rep.k_star, rep.decision_reason)
    assert r["verdict"] == "SAFE", r
    print("test_gate_accepts_factorized_safe: OK")


if __name__ == "__main__":
    test_bayes_oracle_distinguishes_synergy_from_factorized()
    test_gate_accepts_factorized_safe()
    test_gate_no_unsafe_acceptance()
    print("ALL BAYES-CALIBRATION TESTS PASSED")
