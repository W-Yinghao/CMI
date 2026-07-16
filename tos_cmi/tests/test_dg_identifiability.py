"""Tests for the HARDENED DG-identifiability protocol (Phase-0/Phase-1 rescue).

Encodes the scientific structure discovered in calibration:
  * cross-fitted target oracle finds a REAL hindsight ticket (selected on T_select, confirmed on disjoint
    T_query, beats matched-rank random) -> no selection optimism;
  * a MAJORITY-sign harmful shortcut (most source subjects benefit) is NOT source-identifiable: the nested
    source-meta selector correctly REFUSES (k*=0) -> TARGET_HINDSIGHT_ONLY, the honest Result-B analog;
  * a near-BALANCED (visibly unstable) shortcut IS source-identifiable: source-meta recovers a positive
    target-query gain -> the machinery can say "yes" when the ticket really is source-visible.
"""
import numpy as np
import pytest

from tos_cmi.data.spurious_task_dgp import make_spurious_task_dgp
from tos_cmi.eval.dg_identifiability import (get_candidate_basis, build_basis, contested_basis,
    crossfit_target_oracle, nested_source_meta, apply_rule_to_target_full, recovery_verdict, delete_topk)


def _split(dgp):
    Z, y, d, t = dgp["Z"], dgp["y"], dgp["d"], dgp["target_dom"]
    src = d != t
    return Z[src], y[src].astype(int), d[src], Z[d == t], y[d == t].astype(int)


def test_bases_orthonormal_and_contested_bounded():
    dgp = make_spurious_task_dgp(n_domains=10, per_domain=200, seed=0, n_minority_source=3)
    Zs, ys, ds, Zt, yt = _split(dgp)
    n_cls = dgp["n_cls"]
    for fam in ("marg", "cond", "rule", "grad"):
        B = build_basis(fam, Zs, ys, ds, seed=0)
        assert B.ndim == 2 and B.shape[1] == Zs.shape[1] and B.shape[0] >= 1
        assert np.allclose(B @ B.T, np.eye(B.shape[0]), atol=1e-6)          # orthonormal rows
        Bc = contested_basis(B, Zs, ys, seed=0)
        assert Bc.shape[0] <= n_cls - 1 + 1e-9                              # contested dim <= C-1 (head rowspace)


def test_delete_topk_removes_span():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((50, 8))
    Q, _ = np.linalg.qr(rng.standard_normal((8, 3)))
    B = Q.T                                                                 # 3 orthonormal rows
    Zr = delete_topk(Z, B, 3)
    assert np.abs(Zr @ B.T).max() < 1e-8                                    # no energy left along B


def test_crossfit_oracle_finds_real_ticket_no_optimism():
    """Majority-sign shortcut: a beneficial deletion EXISTS and survives the T_select->T_query cross-fit."""
    dgp = make_spurious_task_dgp(n_domains=12, per_domain=250, seed=1, n_minority_source=3,
                                 inv_strength=0.5, spur_strength=2.5, id_strength=3.0)
    Zs, ys, ds, Zt, yt = _split(dgp)
    B = get_candidate_basis("cond", False, Zs, ys, ds, seed=0)
    orc = crossfit_target_oracle(Zs, ys, Zt, yt, B, seed=0)
    assert orc["delta_query"] > 0.05                                       # real gain on DISJOINT query trials
    assert orc["delta_query"] > orc["delta_query_random"] + 0.02           # specific, not generic dim-reduction


def test_majority_shortcut_is_not_source_identifiable():
    """The honest Result-B analog: when the shortcut helps the source majority, source-only selection must
    REFUSE (k*=0) under both robust objectives -> it cannot recover the minority-target ticket."""
    dgp = make_spurious_task_dgp(n_domains=12, per_domain=250, seed=1, n_minority_source=3,
                                 inv_strength=0.5, spur_strength=2.5, id_strength=3.0)
    Zs, ys, ds, Zt, yt = _split(dgp)
    for obj in ("mean_1se", "cvar25"):
        sm = nested_source_meta(Zs, ys, ds, "cond", False, seed=0, objective=obj, eps=0.01)
        ev = apply_rule_to_target_full(Zs, ys, ds, Zt, yt, "cond", False, sm["k_star"], seed=0)
        assert sm["k_star"] == 0
        assert abs(ev["delta_query"]) < 1e-9


def test_balanced_shortcut_is_source_identifiable():
    """When the shortcut is near sign-balanced (visibly unstable) across source subjects, the nested
    source-meta selector recovers a positive target-query gain (machinery CAN say yes)."""
    dgp = make_spurious_task_dgp(n_domains=12, per_domain=250, seed=1, n_minority_source=5,
                                 inv_strength=0.5, spur_strength=2.5, id_strength=3.0)
    Zs, ys, ds, Zt, yt = _split(dgp)
    best = -1.0
    for obj in ("mean_1se", "cvar25"):
        sm = nested_source_meta(Zs, ys, ds, "cond", False, seed=0, objective=obj, eps=0.01)
        if sm["k_star"] >= 1:
            ev = apply_rule_to_target_full(Zs, ys, ds, Zt, yt, "cond", False, sm["k_star"], seed=0)
            best = max(best, ev["delta_query"])
    assert best > 0.01                                                     # source-only recovered real utility


def test_recovery_verdict_states():
    # oracle ticket exists + meta recovers well -> practical
    v = recovery_verdict(oracle_delta=0.10, oracle_lcb=0.04, meta_delta=0.04, meta_lcb=0.01,
                         meta_random_delta=0.0)
    assert v["state"] == "SOURCE_IDENTIFIABLE_PRACTICAL"
    # oracle exists, meta significant but tiny (<25% recovery, <0.5pp) -> detectable tiny
    v = recovery_verdict(oracle_delta=0.10, oracle_lcb=0.04, meta_delta=0.002, meta_lcb=0.001,
                         meta_random_delta=-0.005)
    assert v["state"] == "SOURCE_DETECTABLE_TINY"
    # oracle exists but meta indistinguishable from random -> hindsight only
    v = recovery_verdict(oracle_delta=0.10, oracle_lcb=0.04, meta_delta=-0.001, meta_lcb=-0.005,
                         meta_random_delta=0.0)
    assert v["state"] == "TARGET_HINDSIGHT_ONLY"
    # oracle LCB not > 0 -> no confirmed ticket even with hindsight
    v = recovery_verdict(oracle_delta=0.02, oracle_lcb=-0.01, meta_delta=0.0, meta_lcb=-0.01,
                         meta_random_delta=0.0)
    assert v["state"] == "NO_CONFIRMED_TICKET"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
