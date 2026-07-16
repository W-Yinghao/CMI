"""Firewall + F2.1b readiness tests: source-whitened metric + task-contested basis, selected-rank random,
source-greedy parity in smoke, shared hashed rule, session-macro, audit trail, no query/target-greedy leakage."""
import numpy as np
import pytest

from tos_cmi.data.spurious_task_dgp import make_spurious_task_dgp
from tos_cmi.eeg.relaxation_ladder import _dense
from tos_cmi.eval import targetx_metric as M
from tos_cmi.eval.targetx_observability import (build_actions, g1_select, observable_G1, observable_G2_sanity,
    session_split, utility, audit_fold, PRIMARY, TASK_SAFETY_MAX_DROP, PRIMARY_MAX_RANK)


def _feat(seed=0, sessions=2, per=160):
    dgp = make_spurious_task_dgp(n_domains=8, per_domain=per, seed=seed)
    Z, y, d, t = dgp["Z"], dgp["y"].astype(int), dgp["d"], dgp["target_dom"]
    src, tgt = d != t, d == t
    nt = int(tgt.sum())
    st = (np.array(["0A", "1B", "2C"])[np.arange(nt) % 3] if sessions == 3
          else np.where(np.arange(nt) < nt // 2, "0A", "1B") if sessions == 2 else np.array(["0A"] * nt))
    return dict(Z_source=Z[src], y_source=y[src], subj_source=d[src], Z_target=Z[tgt], y_target=y[tgt],
                subj_target=d[tgt], session_target=st, n_cls=dgp["n_cls"], dataset="synth", heldout_subject="0", seed=seed)


def test_primary_rank_cap_and_rule_hash_stable():
    assert PRIMARY == "G1" and PRIMARY_MAX_RANK == 3
    assert M.rule_hash() == M.rule_hash() and len(M.rule_hash()) == 12


def test_whitened_metric_and_contested_basis():
    f = _feat(0); res = audit_fold(f, seed=0, family="cond", smoke=True, n_random_per_rank=8)
    fw = res["fold"]["firewall"]
    assert fw["metric"] == "source_ledoitwolf_whitened" and fw["primary_basis"] == "cond_contested"
    assert fw["contested_rank"] <= fw["full_cond_rank"]                     # contested is a restriction
    assert fw["contested_rank"] + fw["free_rank"] <= fw["full_cond_rank"] + 1e-9
    # random controls are whitened-orthonormal at their rank
    rc = [rw for rw in res["rows"] if rw["kind"] == "random"]
    assert rc and all(rw["basis_label"] == "ambient_whitened" for rw in rc)


def test_contested_basis_inside_head_rowspace():
    f = _feat(1); Zs, ys, ds = f["Z_source"], f["y_source"], _dense(f["subj_source"])
    W = M.source_whitener(Zs); Zs_w = M.to_whitened(Zs, W)
    row_w, null_w = M.whitened_head_rowspace(Zs_w, ys, 1)
    B = M.whitened_cond_basis(Zs_w, ys, ds, max_rank=8)
    Bc = M.project_basis(B, row_w)
    if Bc.shape[0]:                                                        # contested dirs lie in row(W_c)
        resid = Bc - (Bc @ row_w.T) @ row_w
        assert np.abs(resid).max() < 1e-6


def test_delta_random_matched_to_selected_rank():
    f = _feat(2); res = audit_fold(f, seed=2, family="cond", smoke=True, n_random_per_rank=8)
    fold = res["fold"]; k = fold["selected_rank"]
    rand_k = [rw["utility_macro"] for rw in res["rows"] if rw["kind"] == "random" and rw["rank"] == k]
    if k >= 1:
        assert abs(fold["delta_random_selected_rank"] - float(np.mean(rand_k))) < 1e-9   # ONLY selected-rank
    else:
        assert fold["delta_random_selected_rank"] == 0.0


def test_source_greedy_present_in_smoke():
    f = _feat(3); res = audit_fold(f, seed=3, family="cond", smoke=True, n_random_per_rank=8)
    names = {rw["action"] for rw in res["rows"]}
    assert "srcgreedy_standalone" in names                                 # comparator NOT dropped in smoke
    assert np.isfinite(res["fold"]["delta_source_greedy"])                 # not NaN


def test_baselines_and_hindsight_present():
    f = _feat(4); res = audit_fold(f, seed=4, family="cond", smoke=True, n_random_per_rank=8)
    names = {rw["action"] for rw in res["rows"]}
    assert {"whitening", "mean_centering", "hindsight_constrained", "hindsight_unconstrained"} <= names
    fold = res["fold"]
    assert np.isfinite(fold["delta_whitening"]) and np.isfinite(fold["delta_mean_centering"])
    assert np.isfinite(fold["delta_hindsight_constrained"]) and np.isfinite(fold["delta_hindsight_unconstrained"])
    assert all(rw["eligible"] is False for rw in res["rows"] if rw["action"].startswith("hindsight"))


def test_shared_rule_hash_stamped():
    f = _feat(5); res = audit_fold(f, seed=5, family="cond", smoke=True, n_random_per_rank=8)
    assert res["fold"]["rule_hash"] == M.rule_hash()
    assert all(rw["rule_hash"] == M.rule_hash() for rw in res["rows"])


def test_g1_uses_whitened_discrepancy():
    f = _feat(6); Zs = f["Z_source"]; W = M.source_whitener(Zs)
    d_white = -M.to_whitened(f["Z_target"].mean(0)[None, :], W)[0]
    U = M.ambient_random_projectors_whitened(Zs.shape[1], 2, 1, 0)[0]
    assert abs(observable_G1(U, {"d_white": d_white}) - float(np.sum((U @ d_white) ** 2))) < 1e-12
    assert observable_G1(np.zeros((0, Zs.shape[1])), {"d_white": d_white}) == 0.0


def test_session_macro_differs_from_pooled():
    D = 6; rng = np.random.default_rng(0)
    Zs = rng.standard_normal((300, D)); ys = np.repeat([0, 1], 150)
    action = dict(apply_source=lambda Z: Z - (Z @ np.eye(D)[[0]].T) @ np.eye(D)[[0]],
                  apply_target_query=lambda Z: Z - (Z @ np.eye(D)[[0]].T) @ np.eye(D)[[0]])
    na, nb = 40, 200
    Za = np.hstack([np.where(np.repeat([0, 1], na // 2) > 0, 3.0, -3.0)[:, None] + 0.1 * rng.standard_normal((na, 1)), rng.standard_normal((na, D - 1))])
    Zb = np.hstack([0.05 * rng.standard_normal((nb, 1)), rng.standard_normal((nb, D - 1))])
    Zq = np.vstack([Za, Zb]); yq = np.concatenate([np.repeat([0, 1], na // 2), np.repeat([0, 1], nb // 2)])
    sq = np.array(["A"] * na + ["B"] * nb)
    macro, pooled = utility(action, Zs, ys, Zq, yq, sq, seed=0)
    assert abs(macro - pooled) > 1e-6


def test_no_query_or_target_greedy_leak():
    f = _feat(7); r1 = audit_fold(f, seed=7, family="cond", smoke=True, n_random_per_rank=6)
    f2 = {**f, "Z_target": f["Z_target"].copy()}
    yt = np.asarray(f["y_target"]); cal, qry, _ = session_split(f["session_target"], yt)
    f2["Z_target"][qry] += 100.0 * np.random.default_rng(0).standard_normal(f2["Z_target"][qry].shape)
    r2 = audit_fold(f2, seed=7, family="cond", smoke=True, n_random_per_rank=6)
    s1 = {rw["action"]: rw["G1"] for rw in r1["rows"] if rw["G1"] is not None}
    s2 = {rw["action"]: rw["G1"] for rw in r2["rows"] if rw["G1"] is not None}
    for k in s1:
        assert abs(s1[k] - s2[k]) < 1e-9                                    # T_query X never enters scores
    assert r1["fold"]["firewall"]["query_x_used_for_selection"] is False
    assert r1["fold"]["firewall"]["target_greedy_in_action_set"] is False


def test_phase_primary_g1_only_and_deterministic():
    f = _feat(8)
    r1 = audit_fold(f, seed=8, family="cond", smoke=False, phase="primary", n_random_per_rank=8)
    info = next(rw for rw in r1["rows"] if rw["kind"] == "informed" and rw["rank"] >= 1)
    assert set(info["scores"].keys()) == {"G1"}
    r2 = audit_fold(f, seed=8, family="cond", smoke=False, phase="primary", n_random_per_rank=8)
    assert r1["fold"]["selected_action"] == r2["fold"]["selected_action"]


def test_G2_equals_G1_sanity():
    f = _feat(9); W = M.source_whitener(f["Z_source"])
    d_white = -M.to_whitened(f["Z_target"].mean(0)[None, :], W)[0]; ctx = {"d_white": d_white}
    for k in (1, 2):
        U = M.ambient_random_projectors_whitened(f["Z_source"].shape[1], k, 1, 0)[0]
        assert abs(observable_G1(U, ctx) - observable_G2_sanity(U, ctx)) < 1e-12


def test_stability_metrics_not_jaccard():
    rng = np.random.default_rng(0)
    Q1, _ = np.linalg.qr(rng.standard_normal((8, 2))); Q2, _ = np.linalg.qr(rng.standard_normal((8, 2)))
    A, B = Q1[:, :2].T, Q2[:, :2].T
    assert abs(M.normalized_projector_overlap(A, A) - 1.0) < 1e-6         # identical subspace -> 1.0
    assert M.chordal_distance(A, A) < 1e-6 and M.chordal_distance(A, B) > 0
    assert len(M.principal_angles_cos(A, A)) == 2 and abs(M.principal_angles_cos(A, A)[0] - 1.0) < 1e-6


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
