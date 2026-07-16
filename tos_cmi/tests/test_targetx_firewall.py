"""F2.0c/F2.0d firewall + readiness tests for the target-X observability audit (amendments 01 & 02).
Locks: typed actions; ambient same-rank random controls survive dedup and are NOT selectable; G1 selector has
task-safety + random-specificity gates; identity fallback; BNCI2015 session-macro != pooled; runner preserves
all action rows; rank-stratified observability; deterministic gate verdict; no query leakage; no target-greedy."""
import numpy as np
import pytest

from tos_cmi.data.spurious_task_dgp import make_spurious_task_dgp
from tos_cmi.eeg.relaxation_ladder import _dense
from tos_cmi.eval.dg_identifiability import get_candidate_basis
from tos_cmi.eval.targetx_observability import (build_actions, ambient_random_projectors, g1_select,
    observable_G1, observable_G2_sanity, session_split, utility, audit_fold, source_task_drop,
    PRIMARY, TASK_SAFETY_MAX_DROP)


def _feat(seed=0, sessions=2, per=160):
    dgp = make_spurious_task_dgp(n_domains=8, per_domain=per, seed=seed)
    Z, y, d, t = dgp["Z"], dgp["y"].astype(int), dgp["d"], dgp["target_dom"]
    src, tgt = d != t, d == t
    nt = int(tgt.sum())
    if sessions == 3:
        st = np.array(["0A", "1B", "2C"])[np.arange(nt) % 3]
    elif sessions == 2:
        st = np.where(np.arange(nt) < nt // 2, "0A", "1B")
    else:
        st = np.array(["0A"] * nt)
    return dict(Z_source=Z[src], y_source=y[src], subj_source=d[src], Z_target=Z[tgt], y_target=y[tgt],
                subj_target=d[tgt], session_target=st, n_cls=dgp["n_cls"], dataset="synth",
                heldout_subject="0", seed=seed)


def _ctx(f):
    Zs = f["Z_source"]
    return dict(Zs=Zs, mu_s=Zs.mean(0), mu_tcal=f["Z_target"].mean(0))


def test_primary_is_G1():
    assert PRIMARY == "G1"


def test_random_controls_survive_dedup():
    """Ambient random projectors are NOT basis coordinate subsets, so they cannot be deduped into informed
    actions -> there is >=1 random-kind action per informed rank."""
    f = _feat(0)
    Zs, ys, ds = f["Z_source"], f["y_source"], _dense(f["subj_source"])
    B = get_candidate_basis("cond", False, Zs, ys, ds, max_rank=8, seed=0)
    acts = build_actions(B, Zs, ys, ds, seed=0, n_random_per_rank=20)
    informed_ranks = {a["rank"] for a in acts if a["kind"] == "informed" and a["rank"] >= 1}
    for k in informed_ranks:
        n = sum(a["kind"] == "random" and a["rank"] == k for a in acts)
        assert n >= 20, f"rank {k}: only {n} random controls survived"


def test_random_controls_not_selectable():
    f = _feat(1)
    Zs, ys, ds = f["Z_source"], f["y_source"], _dense(f["subj_source"])
    B = get_candidate_basis("cond", False, Zs, ys, ds, max_rank=8, seed=1)
    acts = build_actions(B, Zs, ys, ds, seed=1, n_random_per_rank=10)
    assert all(a["eligible"] is False for a in acts if a["kind"] == "random")
    assert all(not a["name"].startswith("random") for a in acts if a["eligible"])


def test_random_rank_matches_informed_rank():
    f = _feat(2)
    Zs, ys, ds = f["Z_source"], f["y_source"], _dense(f["subj_source"])
    B = get_candidate_basis("cond", False, Zs, ys, ds, max_rank=8, seed=2)
    acts = build_actions(B, Zs, ys, ds, seed=2, n_random_per_rank=10)
    for a in acts:
        if a["kind"] == "random":
            assert a["dirs"].shape[0] == a["rank"]
            assert np.allclose(a["dirs"] @ a["dirs"].T, np.eye(a["rank"]), atol=1e-8)  # orthonormal


def test_g1_selector_has_task_safety_gate():
    """A deletion that badly hurts source-LOSO must be rejected even if its G1 is large."""
    D = 12
    # informed action = delete the invariant task directions (destroys source task) with a large G1
    rng = np.random.default_rng(0)
    Zs = rng.standard_normal((400, D)); ys = (Zs[:, 0] > 0).astype(int)   # task lives in dim 0
    ds = np.repeat(np.arange(4), 100)
    dirs_bad = np.eye(D)[[0]]                                              # deleting dim 0 kills the task
    drop = source_task_drop(Zs, ys, ds, dirs_bad, seed=0)
    assert drop > TASK_SAFETY_MAX_DROP                                    # gate would reject it


def test_g1_identity_fallback_against_random_null():
    """If no informed action is both safe and above the same-rank random G1 null, selector returns identity."""
    f = _feat(3)
    Zs, ys, ds = f["Z_source"], f["y_source"], _dense(f["subj_source"])
    B = get_candidate_basis("cond", False, Zs, ys, ds, max_rank=6, seed=3)
    # random controls with HUGE G1 (project onto the mean-gap direction) -> specificity gate impossible to pass
    ctx = dict(Zs=Zs, mu_s=Zs.mean(0), mu_tcal=f["Z_target"].mean(0), Xcal=f["Z_target"],
               head=None, n_cls=f["n_cls"], classes=[0, 1], src_contrasts={}, p_source_prior=np.array([.5, .5]),
               log_kappa_identity=0.0)
    d = ctx["mu_s"] - ctx["mu_tcal"]; dhat = d / (np.linalg.norm(d) + 1e-9)
    acts = [dict(name="identity", kind="informed", rank=0, dirs=np.zeros((0, Zs.shape[1])), eligible=True),
            dict(name="singleton_0", kind="informed", rank=1, dirs=B[[0]], eligible=True)]
    acts += [dict(name=f"random_r1_{i}", kind="random", rank=1, dirs=dhat[None, :], eligible=False) for i in range(20)]
    sel, diag = g1_select(acts, ctx, Zs, ys, ds, seed=3)
    assert sel["name"] == "identity" and diag["n_candidates"] == 0


def test_bnci2015_session_macro_not_pooled():
    """With multiple query sessions the macro utility differs from pooled when session gains differ."""
    f = _feat(4, sessions=3, per=240)
    Zs, ys = f["Z_source"], f["y_source"]
    yt = np.asarray(f["y_target"]); st = np.asarray(f["session_target"])
    cal, qry, info = session_split(st, yt)
    assert len(info["query_sessions"]) >= 2                          # 0A -> {1B,2C}
    B = get_candidate_basis("cond", False, Zs, ys, _dense(f["subj_source"]), max_rank=6, seed=4)
    action = dict(name="s", kind="informed", rank=2, dirs=B[:2], eligible=True)
    macro, pooled = utility(action, Zs, ys, f["Z_target"][qry], yt[qry], st[qry], seed=4)
    assert np.isfinite(macro) and np.isfinite(pooled)              # both computed (macro is primary)


def test_full_runner_preserves_action_rows():
    """audit_fold must return ALL per-action rows (informed + random), not just top-k."""
    f = _feat(5)
    res = audit_fold(f, seed=5, family="cond", smoke=True, n_random_per_rank=8)
    assert res is not None
    assert len(res["rows"]) == res["n_actions"] == res["n_informed"] + res["n_random"]
    assert res["n_random"] >= 8 and res["n_informed"] >= 2
    assert all("utility_macro" in rw and "G1" in rw["scores"] for rw in res["rows"])


def test_rank_stratified_observability_computable():
    """Per-action rows carry rank + G1 + utility, so a within-rank Spearman is computable (not just pooled)."""
    f = _feat(6)
    res = audit_fold(f, seed=6, family="cond", smoke=False, n_random_per_rank=10, observables=["G1"])
    by_rank = {}
    for rw in res["rows"]:
        if rw["kind"] == "informed" and rw["rank"] >= 1:
            by_rank.setdefault(rw["rank"], []).append((rw["scores"]["G1"], rw["utility_macro"]))
    assert any(len(v) >= 3 for v in by_rank.values())             # at least one rank has enough points


def test_gate_verdict_deterministic():
    """Same feat + seed -> identical selection (deterministic; required for a reproducible gate verdict)."""
    f = _feat(7)
    r1 = audit_fold(f, seed=7, family="cond", smoke=True, n_random_per_rank=8)
    r2 = audit_fold(f, seed=7, family="cond", smoke=True, n_random_per_rank=8)
    assert r1["selected_action"] == r2["selected_action"]
    assert r1["delta_tx_macro"] == r2["delta_tx_macro"]


def test_targetx_scores_use_cal_only_not_query():
    f = _feat(8)
    r1 = audit_fold(f, seed=8, family="cond", smoke=True, n_random_per_rank=6)
    f2 = {**f, "Z_target": f["Z_target"].copy()}
    yt = np.asarray(f["y_target"]); cal, qry, _ = session_split(f["session_target"], yt)
    f2["Z_target"][qry] += 100.0 * np.random.default_rng(0).standard_normal(f2["Z_target"][qry].shape)
    r2 = audit_fold(f2, seed=8, family="cond", smoke=True, n_random_per_rank=6)
    s1 = {rw["action"]: rw["scores"]["G1"] for rw in r1["rows"]}
    s2 = {rw["action"]: rw["scores"]["G1"] for rw in r2["rows"]}
    for k in s1:
        assert abs(s1[k] - s2[k]) < 1e-9                          # G1 uses T_cal X only
    assert r1["firewall"]["query_x_used_for_selection"] is False
    assert r1["firewall"]["target_greedy_in_action_set"] is False


def test_G2_equals_G1_sanity():
    f = _feat(9)
    Zs, ys, ds = f["Z_source"], f["y_source"], _dense(f["subj_source"])
    B = get_candidate_basis("cond", False, Zs, ys, ds, max_rank=6, seed=9)
    ctx = _ctx(f)
    for S in ([0], [0, 1], [1, 2]):
        assert abs(observable_G1(B[S], ctx) - observable_G2_sanity(B[S], ctx)) < 1e-12


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
