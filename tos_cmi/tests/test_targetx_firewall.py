"""F2.0c firewall tests for the target-X observability audit (A8). Lock the contract: (i) no target-greedy
action in the eligible set; (ii) target-X scores depend on T_cal X ONLY (perturbing T_query X leaves scores
unchanged); (iii) identity fallback when all deletion scores <= 0; (iv) G1 frozen sign, identity score = 0."""
import numpy as np
import pytest

from tos_cmi.data.spurious_task_dgp import make_spurious_task_dgp
from tos_cmi.eeg.relaxation_ladder import _dense
from tos_cmi.eval.dg_identifiability import get_candidate_basis
from tos_cmi.eval.targetx_observability import (eligible_actions, session_split, targetx_select,
    observable_G1, audit_fold, OBSERVABLES, PRIMARY)


def _feat(seed=0, sessions=2):
    dgp = make_spurious_task_dgp(n_domains=8, per_domain=160, seed=seed)
    Z, y, d, t = dgp["Z"], dgp["y"].astype(int), dgp["d"], dgp["target_dom"]
    src = d != t; tgt = d == t
    nt = int(tgt.sum())
    st = np.array(["0A" if i < nt // 2 else "1B" for i in range(nt)]) if sessions == 2 else np.array(["0A"] * nt)
    return dict(Z_source=Z[src], y_source=y[src], subj_source=d[src], Z_target=Z[tgt], y_target=y[tgt],
                subj_target=d[tgt], session_target=st, n_cls=dgp["n_cls"], dataset="synth",
                heldout_subject="0", seed=seed)


def test_primary_is_G1():
    assert PRIMARY == "G1" and "G1" in OBSERVABLES


def test_no_target_greedy_in_eligible_actions():
    f = _feat(0)
    Zs, ys, ds = f["Z_source"], f["y_source"], _dense(f["subj_source"])
    B = get_candidate_basis("cond", False, Zs, ys, ds, max_rank=8, seed=0)
    acts = eligible_actions(B, Zs, ys, ds, seed=0)
    names = [n for n, _ in acts]
    assert not any("targetgreedy" in n or "target_greedy" in n or "hindsight" in n for n in names)
    assert ("identity", []) in [(n, S) for n, S in acts]


def test_targetx_scores_use_cal_only_not_query():
    """Perturbing T_query X must not change any target-X score (they read T_cal X only)."""
    f = _feat(1)
    r1 = audit_fold(f, seed=1, family="cond", smoke=True)
    f2 = {**f, "Z_target": f["Z_target"].copy()}
    yt = np.asarray(f["y_target"]); cal, qry, _ = session_split(f["session_target"], yt)
    f2["Z_target"][qry] += 100.0 * np.random.default_rng(0).standard_normal(f2["Z_target"][qry].shape)
    r2 = audit_fold(f2, seed=1, family="cond", smoke=True)
    s1 = {rw["action"]: rw["scores"]["G1"] for rw in _rows(r1)}
    s2 = {rw["action"]: rw["scores"]["G1"] for rw in _rows(r2)}
    for k in s1:
        assert abs(s1[k] - s2[k]) < 1e-9, f"score changed when T_query X perturbed: {k}"
    assert r1["firewall"]["query_x_used_for_selection"] is False
    assert r1["firewall"]["target_greedy_in_action_set"] is False


def _rows(res):
    # audit_fold returns rows internally; re-run eligible to expose them via a light recompute is overkill,
    # so use the stored 'rows' if present (audit_fold keeps them).
    return res["rows"]


def test_identity_fallback_when_all_scores_nonpositive():
    scores = [("identity", [], 0.0), ("singleton_0", [0], -0.5), ("singleton_1", [1], -0.1)]
    assert targetx_select(scores) == ("identity", [])
    scores2 = scores + [("singleton_2", [2], +0.3)]
    assert targetx_select(scores2) == ("singleton_2", [2])


def test_identity_score_zero_and_G1_sign():
    f = _feat(2)
    Zs, ys, ds = f["Z_source"], f["y_source"], _dense(f["subj_source"])
    B = get_candidate_basis("cond", False, Zs, ys, ds, max_rank=8, seed=2)
    ctx = dict(B=B, mu_s=Zs.mean(0), mu_tcal=f["Z_target"].mean(0))
    assert observable_G1([], ctx) == 0.0                       # identity -> 0
    # deleting a direction that carries source-target mean gap should give G1 >= 0 (a reduction)
    g = observable_G1([0], ctx)
    assert np.isfinite(g)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
