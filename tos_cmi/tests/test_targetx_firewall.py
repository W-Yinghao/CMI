"""Firewall + readiness tests for the target-X observability audit (amendments 01/02/03). Locks typed actions,
ambient random controls (survive dedup, not selectable, rank-matched), G1 gates + identity fallback, baselines
(whitening + mean-centering with per-domain transforms), target-hindsight denominator (not selectable),
session-macro, full audit trail + projector reconstruction, phase enforcement, no query/target-greedy leakage."""
import numpy as np
import pytest

from tos_cmi.data.spurious_task_dgp import make_spurious_task_dgp
from tos_cmi.eeg.relaxation_ladder import _dense
from tos_cmi.eval.dg_identifiability import get_candidate_basis
from tos_cmi.eval.targetx_observability import (build_actions, ambient_random_projectors, g1_select,
    observable_G1, observable_G2_sanity, session_split, utility, audit_fold, source_task_drop, make_action,
    _hash, _delete_fn, PRIMARY, TASK_SAFETY_MAX_DROP, PRIMARY_MAX_RANK)


def _feat(seed=0, sessions=2, per=160):
    dgp = make_spurious_task_dgp(n_domains=8, per_domain=per, seed=seed)
    Z, y, d, t = dgp["Z"], dgp["y"].astype(int), dgp["d"], dgp["target_dom"]
    src, tgt = d != t, d == t
    nt = int(tgt.sum())
    st = (np.array(["0A", "1B", "2C"])[np.arange(nt) % 3] if sessions == 3
          else np.where(np.arange(nt) < nt // 2, "0A", "1B") if sessions == 2 else np.array(["0A"] * nt))
    return dict(Z_source=Z[src], y_source=y[src], subj_source=d[src], Z_target=Z[tgt], y_target=y[tgt],
                subj_target=d[tgt], session_target=st, n_cls=dgp["n_cls"], dataset="synth", heldout_subject="0", seed=seed)


def _B(f, mr=8):
    return get_candidate_basis("cond", False, f["Z_source"], f["y_source"], _dense(f["subj_source"]), max_rank=mr, seed=0)


def test_primary_is_G1_and_rank_cap():
    assert PRIMARY == "G1" and PRIMARY_MAX_RANK == 3


def test_random_controls_survive_dedup_and_not_selectable():
    f = _feat(0); B = _B(f)
    acts = build_actions(B, f["Z_source"], f["y_source"], _dense(f["subj_source"]), f["Z_target"], seed=0, n_random_per_rank=20)
    for k in {a["rank"] for a in acts if a["kind"] == "informed" and a["rank"] >= 1}:
        assert sum(a["kind"] == "random" and a["rank"] == k for a in acts) >= 20
    assert all(a["eligible"] is False for a in acts if a["kind"] != "informed")
    assert all(not a["name"].startswith("random") for a in acts if a["eligible"])
    assert all(a["dirs"].shape[0] == a["rank"] for a in acts if a["kind"] == "random")


def test_baselines_present_with_per_domain_transforms():
    f = _feat(1); B = _B(f)
    acts = build_actions(B, f["Z_source"], f["y_source"], _dense(f["subj_source"]), f["Z_target"], seed=1, n_random_per_rank=8)
    names = {a["name"]: a for a in acts}
    assert "whitening" in names and "mean_centering" in names
    mc = names["mean_centering"]; Zs = f["Z_source"]
    assert np.allclose(mc["apply_source"](Zs), Zs - Zs.mean(0))                 # source-centering
    assert np.allclose(mc["apply_target_cal"](f["Z_target"]), f["Z_target"] - f["Z_target"].mean(0))  # cal-centering
    assert names["whitening"]["eligible"] is False and names["mean_centering"]["eligible"] is False


def test_target_hindsight_not_selectable():
    f = _feat(2)
    res = audit_fold(f, seed=2, family="cond", smoke=True, n_random_per_rank=8)
    names = {rw["action"]: rw for rw in res["rows"]}
    assert "target_hindsight" in names and names["target_hindsight"]["eligible"] is False
    assert res["fold"]["selected_action"] != "target_hindsight"
    assert np.isfinite(res["fold"]["delta_target_hindsight"])                   # denominator computed


def test_g1_task_safety_gate():
    D = 12; rng = np.random.default_rng(0)
    Zs = rng.standard_normal((400, D)); ys = (Zs[:, 0] > 0).astype(int); ds = np.repeat(np.arange(4), 100)
    assert source_task_drop(Zs, ys, ds, np.eye(D)[[0]], seed=0) > TASK_SAFETY_MAX_DROP


def test_g1_identity_fallback_against_random_null():
    f = _feat(3); B = _B(f, 6)
    ctx = dict(Zs=f["Z_source"], mu_s=f["Z_source"].mean(0), mu_tcal=f["Z_target"].mean(0), Xcal=f["Z_target"],
               head=None, n_cls=f["n_cls"], classes=[0, 1], src_contrasts={}, p_source_prior=np.array([.5, .5]),
               log_kappa_identity=0.0)
    d = ctx["mu_s"] - ctx["mu_tcal"]; dhat = (d / (np.linalg.norm(d) + 1e-9))[None, :]
    acts = [make_action("identity", "informed", 0, dirs=np.zeros((0, f["Z_source"].shape[1])), eligible=True),
            make_action("singleton_0", "informed", 1, dirs=B[[0]], eligible=True)]
    acts += [make_action(f"random_r1_{i}", "random", 1, dirs=dhat, eligible=False) for i in range(20)]  # null >= any G1
    sel, diag = g1_select(acts, ctx, f["Z_source"], f["y_source"], _dense(f["subj_source"]), 3)
    assert sel["name"] == "identity" and diag["n_candidates"] == 0


def test_session_macro_differs_from_pooled():
    """Fixture with two query sessions of different sizes AND different per-session gains -> macro != pooled."""
    D = 6; rng = np.random.default_rng(0)
    Zs = rng.standard_normal((300, D)); ys = np.repeat([0, 1], 150); ds = np.repeat(np.arange(3), 100)
    B = np.eye(D)[[0]]
    action = make_action("del0", "informed", 1, dirs=B, eligible=True)
    # session A (small, deletion helps) vs session B (large, deletion hurts) -> pooled weights toward B, macro equal
    na, nb = 40, 200
    Za = np.hstack([np.where(np.repeat([0, 1], na // 2) > 0, 3.0, -3.0)[:, None] + 0.1 * rng.standard_normal((na, 1)),
                    rng.standard_normal((na, D - 1))])
    ya = np.repeat([0, 1], na // 2)
    Zb = np.hstack([0.05 * rng.standard_normal((nb, 1)), rng.standard_normal((nb, D - 1))]); yb = np.repeat([0, 1], nb // 2)
    Zq = np.vstack([Za, Zb]); yq = np.concatenate([ya, yb]); sq = np.array(["A"] * na + ["B"] * nb)
    macro, pooled = utility(action, Zs, ys, Zq, yq, sq, seed=0)
    assert abs(macro - pooled) > 1e-6                                          # equal-weight macro != trial-weighted pooled


def test_full_audit_trail_and_projector_reconstruction():
    f = _feat(5); res = audit_fold(f, seed=5, family="cond", smoke=True, n_random_per_rank=8)
    rw = next(r for r in res["rows"] if r["kind"] == "informed" and r["rank"] >= 1)
    for key in ("basis_hash", "projector_hash", "basis_indices", "source_task_drop", "random_q95_same_rank",
                "safe_gate_pass", "specificity_gate_pass", "utility_macro", "utility_pooled"):
        assert key in rw
    # reconstruct the SELECTED projector from basis_hash + indices
    fold = res["fold"]; B = _B(f, 10)
    assert _hash(B) == fold["selected_basis_hash"]
    if fold["selected_basis_indices"]:
        dirs = B[fold["selected_basis_indices"]]
        assert dirs.shape[0] == len(fold["selected_basis_indices"])


def test_phase_primary_runs_g1_only():
    f = _feat(6)
    res = audit_fold(f, seed=6, family="cond", smoke=False, phase="primary", n_random_per_rank=8)
    info_row = next(r for r in res["rows"] if r["kind"] == "informed" and r["rank"] >= 1)
    assert set(info_row["scores"].keys()) == {"G1"}                            # secondaries NOT computed in primary


def test_deterministic_selection():
    f = _feat(7)
    r1 = audit_fold(f, seed=7, family="cond", smoke=True, n_random_per_rank=8)
    r2 = audit_fold(f, seed=7, family="cond", smoke=True, n_random_per_rank=8)
    assert r1["fold"]["selected_action"] == r2["fold"]["selected_action"]
    assert r1["fold"]["delta_tx"] == r2["fold"]["delta_tx"]


def test_scores_use_cal_only_not_query():
    f = _feat(8); r1 = audit_fold(f, seed=8, family="cond", smoke=True, n_random_per_rank=6)
    f2 = {**f, "Z_target": f["Z_target"].copy()}
    yt = np.asarray(f["y_target"]); cal, qry, _ = session_split(f["session_target"], yt)
    f2["Z_target"][qry] += 100.0 * np.random.default_rng(0).standard_normal(f2["Z_target"][qry].shape)
    r2 = audit_fold(f2, seed=8, family="cond", smoke=True, n_random_per_rank=6)
    s1 = {rw["action"]: rw["G1"] for rw in r1["rows"] if rw["G1"] is not None}
    s2 = {rw["action"]: rw["G1"] for rw in r2["rows"] if rw["G1"] is not None}
    for k in s1:
        assert abs(s1[k] - s2[k]) < 1e-9
    assert r1["fold"]["firewall"]["query_x_used_for_selection"] is False
    assert r1["fold"]["firewall"]["target_greedy_in_action_set"] is False


def test_G2_equals_G1_sanity():
    f = _feat(9); B = _B(f, 6); ctx = dict(mu_s=f["Z_source"].mean(0), mu_tcal=f["Z_target"].mean(0))
    for S in ([0], [0, 1], [1, 2]):
        assert abs(observable_G1(B[S], ctx) - observable_G2_sanity(B[S], ctx)) < 1e-12


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
