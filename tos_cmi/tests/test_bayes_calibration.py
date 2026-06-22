"""Phase 1.2.4 -- calibrate the learned conditional-task-safety gate against the EXACT Bayes
conditional task delta (the synthetic is a known Gaussian mixture, so I(Y;deleted|kept) is
computable with no probe training).

Settles the Phase 1.2.3 "bias" question: the synergy "safe" span is in fact CONDITIONALLY
UNSAFE (shared domain factor -> explaining-away), so the gate's nonzero value is real
information, not estimator bias -- de-biasing would be anti-conservative. The factorized case
is genuinely safe. The headline safety check: the gate must NOT ACCEPT a deletion that Bayes
says loses conditional task info (no unsafe acceptance).
"""
from dataclasses import replace
import numpy as np
import torch

torch.set_num_threads(1)

from tos_cmi.data.synthetic import make_partial_synergy, make_partial_factorized
from tos_cmi.score_fisher import (ScoreFisherConfig, select_score_fisher, _metric,
                                  task_protected_projector)
from tos_cmi.eval.bayes_oracle import bayes_conditional_task_delta


def _cfg():
    return ScoreFisherConfig(epochs=200, hidden=64, gate_boot=200, n_perm_null=2, task_protect=True)


def test_bayes_oracle_distinguishes_synergy_from_factorized():
    """Deleting the geometrically task-orthogonal safe span: synergy is conditionally UNSAFE
    (Delta* >> delta_Y) while factorized is genuinely safe (Delta* ~ 0)."""
    cfg = _cfg()
    out = {}
    for name, mk in [("synergy", make_partial_synergy), ("factorized", make_partial_factorized)]:
        data = mk(n=8000, seed=0); s = data["spec"]
        Z = data["Z"].astype(np.float64); M = _metric(Z, data["y"], s.n_cls, cfg)
        P, _ = task_protected_projector(data["nuisance_basis"], data["task_overlap_basis"], M)
        out[name] = bayes_conditional_task_delta(Z, data["y"], data["d"], s.n_cls, s.n_dom, P)["delta"]
    print("Bayes Delta_Y* (delete safe span):", {k: round(v, 4) for k, v in out.items()})
    assert out["synergy"] > 0.05, out          # explaining-away -> conditionally UNSAFE
    assert out["factorized"] < 0.02, out       # independent factors -> genuinely safe
    print("test_bayes_oracle_distinguishes_synergy_from_factorized: OK")


def _gate_vs_bayes(mk, seed=0):
    data = mk(n=6000, seed=seed); s = data["spec"]; cfg = _cfg()
    rep = select_score_fisher(data["Z"], data["y"], data["d"], s.n_cls, s.n_dom, cfg, seed=seed)
    Z = data["Z"].astype(np.float64)
    bayes = bayes_conditional_task_delta(Z, data["y"], data["d"], s.n_cls, s.n_dom, rep.P)["delta"]
    return rep, bayes, cfg


def test_gate_no_unsafe_acceptance():
    """SAFETY CALIBRATION: whenever the gate ACCEPTS a deletion (k>=1), the deployed projector's
    Bayes conditional task loss must be within delta_Y (a small margin). I.e. the gate never
    accepts a deletion that ground-truth Bayes says is conditionally unsafe."""
    for name, mk in [("factorized", make_partial_factorized), ("synergy", make_partial_synergy)]:
        rep, bayes, cfg = _gate_vs_bayes(mk)
        gate_info = rep.rank_records[rep.k_star - 1]["task_info_delta_mean"] if rep.k_star else 0.0
        print("gate vs bayes [%s]:" % name, {"k": rep.k_star, "reason": rep.decision_reason,
              "gate_task_info": round(gate_info, 4), "bayes_delta": round(bayes, 4)})
        if rep.k_star >= 1:                     # accepted a deletion -> it must be Bayes-safe
            assert bayes <= cfg.delta_Y + 0.02, ("UNSAFE ACCEPTANCE", name, rep.k_star, bayes)
    print("test_gate_no_unsafe_acceptance: OK")


def test_gate_accepts_factorized_safe():
    """On the genuinely-safe factorized case the gate should ACCEPT (k>=1) and Bayes confirms
    the deployed deletion is safe."""
    rep, bayes, cfg = _gate_vs_bayes(make_partial_factorized)
    print("factorized gate:", {"k": rep.k_star, "reason": rep.decision_reason, "bayes": round(bayes, 4)})
    assert rep.k_star >= 1, (rep.k_star, rep.decision_reason)
    assert bayes < 0.03, bayes
    print("test_gate_accepts_factorized_safe: OK")


if __name__ == "__main__":
    test_bayes_oracle_distinguishes_synergy_from_factorized()
    test_gate_accepts_factorized_safe()
    test_gate_no_unsafe_acceptance()
    print("ALL BAYES-CALIBRATION TESTS PASSED")
