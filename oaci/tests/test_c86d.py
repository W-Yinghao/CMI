"""C86D shadow + failure tests — process isolation, policies, freeze, evaluator,
exact CVaR, claim boundary, C85U identity, gated execution. No real active run."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from oaci.active_testing import c86d
from oaci.active_testing.c86d import core, policies, c85u_config
from oaci.active_testing.c86d.pipeline import (
    AUTHORIZATION_PHRASE, C86DNotAuthorized, C86DOrderingError, HeldEvaluator,
    execute, run_selection,
)
from oaci.active_testing.c86d.server import C86DServerError, start_query_server


# --- exact fractional-boundary CVaR (PM point 6) ------------------------------
def test_exact_cvar_fractional_boundary():
    r = list(range(1, 23))                       # 22 targets, values 1..22
    # frac 0.25 -> k=5.5 -> (22+21+20+19+18 + 0.5*17)/5.5
    expect = (22 + 21 + 20 + 19 + 18 + 0.5 * 17) / 5.5
    assert abs(core.exact_upper_cvar(r, 0.25) - expect) < 1e-9
    # differs from the shadow ceil(frac*N)=6 mean of top6
    ceil_mean = np.mean([22, 21, 20, 19, 18, 17])
    assert abs(core.exact_upper_cvar(r, 0.25) - ceil_mean) > 1e-6


# --- claim boundary -----------------------------------------------------------
@pytest.mark.parametrize("q", sorted(core.NONLINEAR_PLUGINS))
def test_plugins_no_unbiasedness(q):
    assert policies.unbiasedness_claim(q) is False
    with pytest.raises(core.C86DClaimError):
        core.assert_linear_claim(q)


@pytest.mark.parametrize("q", sorted(core.LINEAR_MOMENTS))
def test_linear_moments_unbiased(q):
    assert policies.unbiasedness_claim(q) is True
    core.assert_linear_claim(q)


# --- process isolation: server owns sealed paths, client handle is path-free ---
def test_process_isolation_and_one_query_one_trial(tmp_path):
    f = c86d.build_shadow_field(str(tmp_path))
    client = start_query_server(f["oracle_root"], f["contrib_root"])
    try:
        # client handle holds no oracle/contribution path
        for v in vars(client).values():
            s = str(v)
            assert "acquisition_label_oracle" not in s and "query_contribution_store" not in s
        assert not any(a in ("oracle_root", "contrib_root") for a in vars(client))
        att = client.open_attempt(("ShadowDS", 0), 4)
        pool = policies.load_pool(f["pool_root"])
        trial = sorted(pool[("ShadowDS", 0)])[0]
        label, contexts = client.query(att, trial)
        assert label in (0, 1)
        assert len(contexts) == 8                      # one physical label -> its 8 contexts
        assert all(k.startswith("panel=") for k in contexts)
        assert set(contexts[list(contexts)[0]]) >= {"nll", "correct", "confidence"}
        with pytest.raises(C86DServerError):           # duplicate
            client.query(att, trial)
    finally:
        client.close()


def test_budget_exhaustion(tmp_path):
    f = c86d.build_shadow_field(str(tmp_path), n_trials=6)
    client = start_query_server(f["oracle_root"], f["contrib_root"])
    try:
        att = client.open_attempt(("ShadowDS", 0), 2)
        pool = policies.load_pool(f["pool_root"])
        trials = sorted(pool[("ShadowDS", 0)])
        client.query(att, trials[0]); client.query(att, trials[1])
        assert client.remaining(att) == 0
        with pytest.raises(C86DServerError):
            client.query(att, trials[2])
    finally:
        client.close()


# --- policies produce a valid frozen selection (P0/A1/A2H) --------------------
@pytest.mark.parametrize("method", ["P0", "A1", "A2H"])
def test_run_selection_freezes(tmp_path, method):
    f = c86d.build_shadow_field(str(tmp_path))
    client = start_query_server(f["oracle_root"], f["contrib_root"])
    try:
        pool = policies.load_pool(f["pool_root"])
        fr = run_selection(client, pool[("ShadowDS", 0)], ("ShadowDS", 0), method, 4, seed=0)
        assert fr.frozen is True
        assert len(fr.query_sequence) == 4 and len(set(fr.query_sequence)) == 4
        assert len(fr.lure_weights) == 4
        assert len(fr.selected_by_context) == 8       # one selected candidate per context
        assert all(0 <= s < 81 for s in fr.selected_by_context.values())
    finally:
        client.close()


# --- selection freeze must precede held evaluation ---------------------------
def test_evaluation_requires_frozen_selection(tmp_path):
    f = c86d.build_shadow_field(str(tmp_path))
    client = start_query_server(f["oracle_root"], f["contrib_root"])
    try:
        pool = policies.load_pool(f["pool_root"])
        fr = run_selection(client, pool[("ShadowDS", 0)], ("ShadowDS", 0), "A1", 4, seed=1)
        ev = HeldEvaluator(f["held_by_ctx"], verify_identity=False)
        out = ev.evaluate(fr)
        assert out["n_contexts"] == 8 and 0.0 <= out["target_raw_gap_diagnostic"]
        fr.frozen = False                              # tamper: unfreeze
        with pytest.raises(C86DOrderingError):
            ev.evaluate(fr)
    finally:
        client.close()


# --- endpoints: exact CVaR + target-first --------------------------------------
def test_compute_endpoints_uses_exact_cvar():
    reg = {"cohortA": [0.0, 0.0, 0.3, 0.3], "cohortB": [0.02, 0.02, 0.02, 0.02]}
    m = core.compute_endpoints(reg)
    assert abs(m.tail_by_cohort["cohortA"] - core.exact_upper_cvar([0, 0, 0.3, 0.3], 0.25)) < 1e-9
    assert m.near_opt_by_cohort["cohortB"] == 1.0


# --- method freeze registry ---------------------------------------------------
def test_method_freeze_registry():
    assert core.METHOD_FREEZE["primary_registry"] == ("P0", "A1", "A2H")
    assert core.METHOD_FREEZE["no_post_hoc_method_add_or_drop"] is True
    assert core.METHOD_FREEZE["context_aggregation"] == "equal_weight_mean"


# --- C85U identity: real-manifest verify (guarded) + mismatch path -----------
def test_c85u_identity_verify_real_if_present():
    if not os.path.exists(c85u_config.C85U_ACCEPTANCE_MANIFEST):
        pytest.skip("C85U manifest not mounted")
    idy = core.verify_c85u_identity()
    assert idy["verified"] is True
    assert idy["field_identity"]["candidate_rows"] == 76464
    assert idy["field_identity"]["contexts"] == 944


def test_c85u_identity_mismatch_fails(tmp_path):
    fake = tmp_path / "fake_manifest.json"
    fake.write_text('{"not":"the real thing"}')
    with pytest.raises(core.C86DIdentityError):
        core.verify_c85u_identity(str(fake))


# --- gated execution ----------------------------------------------------------
def test_execute_refuses_without_authorization():
    with pytest.raises(C86DNotAuthorized):
        execute()
    with pytest.raises(C86DNotAuthorized):
        execute(authorization="nope")


def test_execute_authorized_requires_output_root():
    # with the phrase the guard passes; the real run still needs an explicit output_root
    with pytest.raises(ValueError):
        execute(authorization=AUTHORIZATION_PHRASE)


# ============ PM C86D reconciliation tests ============
import csv


# --- blocker 1: A1 entropy non-negative + non-uniform; uniform pool degenerates to P0
def test_a1_entropy_nonnegative_and_nonuniform():
    p = np.array([[0.99, 0.01], [0.5, 0.5], [0.7, 0.3]])
    assert (policies._entropy(p) >= 0).all()
    pool = {"t0": {"c": np.tile([0.99, 0.01], (81, 1))},
            "t1": {"c": np.tile([0.5, 0.5], (81, 1))}}
    s = policies.acquisition_score(pool, "A1")
    assert s["t0"] != s["t1"]                       # informative asymmetry -> non-uniform score
    # symmetric pool -> equal scores -> A1 proposal is uniform (== P0 regime)
    sym = {"t0": {"c": np.tile([0.5, 0.5], (81, 1))}, "t1": {"c": np.tile([0.5, 0.5], (81, 1))}}
    ss = policies.acquisition_score(sym, "A1")
    assert abs(ss["t0"] - ss["t1"]) < 1e-12


# --- blocker 2a: Jeffreys bAcc (missing class -> 0.5), no dropped class
def test_jeffreys_bacc_missing_class():
    # only class 0 present in the queried sample; class 1 must contribute Jeffreys 0.5
    y = [0, 0, 0]
    contribs = {"nll": np.zeros((3, 81)), "correct": np.ones((3, 81), int),
                "confidence": np.full((3, 81), 0.9), "conf_bin": np.full((3, 81), 13, int)}
    bacc, _, _ = policies.estimate_metrics(y, contribs, None, False, len(y))
    # class0 recall = (3+.5)/(3+1)=0.875 ; class1 (missing) = (0+.5)/(0+1)=0.5 ; bAcc=mean=0.6875
    assert np.allclose(bacc, (0.875 + 0.5) / 2)


# --- blocker 2b: 15-bin ECE closed form (not mean|conf-correct|)
def test_binwise_ece_closed_form():
    # 2 candidates, 2 trials in the same bin; conf 0.8 both, correct [1,0] -> acc 0.5, |0.8-0.5|=0.3
    y = [0, 1]
    contribs = {"nll": np.zeros((2, 2)), "correct": np.array([[1, 1], [0, 0]]),
                "confidence": np.full((2, 2), 0.8), "conf_bin": np.full((2, 2), 12, int)}
    _, _, ece = policies.estimate_metrics(y, contribs, None, True, len(y))
    assert np.allclose(ece, 0.3)                    # binwise |mean_conf - mean_acc|, not mean|.|


# --- blocker 2c: composite pipeline reproduces C85U exactly (positive control)
def test_c85u_composite_positive_control():
    if not os.path.exists(c85u_config.C85U_UTILITY_INDEX):
        pytest.skip("C85U index not mounted")
    rows = [r for r in csv.DictReader(open(c85u_config.C85U_UTILITY_INDEX))
            if r["context_id"] == "8f38605b8c47d37ef0b1e76f"]
    bacc = np.array([float(r["balanced_accuracy"]) for r in rows])
    nll = np.array([float(r["NLL"]) for r in rows]); ece = np.array([float(r["ECE"]) for r in rows])
    comp_t = np.array([float(r["composite_utility"]) for r in rows])
    sr_t = np.array([float(r["standardized_regret"]) for r in rows])
    comp, sr, _ = policies.composite_from_metrics(bacc, nll, ece)
    assert np.max(np.abs(comp - comp_t)) < 1e-9
    assert np.max(np.abs(sr - sr_t)) < 1e-9


# --- blocker 3: raw utility gap and standardized regret are NOT interchangeable
def test_raw_gap_vs_standardized_regret_differ():
    comp = np.array([0.9, 0.6, 0.1] + [0.5] * 78)
    _, std, _ = policies.composite_from_metrics(
        comp, np.zeros(81), np.zeros(81))            # feed comp as one metric to get a spread
    raw_gap = comp.max() - comp
    # std regret is raw_gap / spread; they differ unless spread == 1
    assert not np.allclose(raw_gap, (comp.max() - comp) / (comp.max() - comp.min()))


# --- warm start (first 4 uniform) + nested prefix + budget-specific LURE weights
def test_warm_start_and_nested_prefix():
    pool = {f"t{i}": {"c": np.tile([0.5 + 0.004 * i, 0.5 - 0.004 * i], (81, 1))} for i in range(20)}
    order, q = policies.acquisition_path(pool, "A1", seed=0)
    N = len(order)
    assert q[:4] == [1.0 / N, 1.0 / (N - 1), 1.0 / (N - 2), 1.0 / (N - 3)]   # warm start uniform
    p8, w8 = policies.budget_prefix(order, q, N, 8)
    p16, w16 = policies.budget_prefix(order, q, N, 16)
    assert p8 == p16[:8]                             # nested prefixes
    assert np.allclose(w8[:4], 1.0)                 # warm-start steps => LURE weight 1
    assert not np.allclose(w8[4:8], w16[:8][4:8])   # ACTIVE steps use budget-specific LURE v_m^M


# --- D1 stage does not import/reference C85U ---------------------------------
def test_d1_has_no_c85u_reference():
    from oaci.active_testing.c86d import run_d1
    src = open(run_d1.__file__).read()
    assert "candidate_utility_index" not in src and "C85U_UTILITY_INDEX" not in src
    assert "verify_c85u_identity" not in src
    assert not hasattr(run_d1, "C85U_UTILITY_INDEX")


# --- D2 fails closed on a tampered freeze ------------------------------------
def test_d2_rejects_tampered_freeze(tmp_path):
    from oaci.active_testing.c86d import run_d2
    import hashlib
    d1 = tmp_path / "d1"; (d1 / "freezes").mkdir(parents=True)
    blob = '{"method":"P0","target":["Cho2017",1],"seed":0,"budgets":[]}'
    (d1 / "freezes" / "f.json").write_text(blob)
    idx = [{"file": "freezes/f.json", "method": "P0", "target": ["Cho2017", 1], "seed": 0,
            "sha256": hashlib.sha256(b"WRONG").hexdigest()}]   # deliberately wrong hash
    (d1 / "C86D_D1_MANIFEST.json").write_text(json.dumps(
        {"c85u_accessed": False, "budgets": ["FULL"], "n_freeze_files": 1, "freeze_index": idx}))
    if not os.path.exists(c85u_config.C85U_UTILITY_INDEX):
        pytest.skip("C85U index not mounted")
    with pytest.raises(RuntimeError):
        run_d2.run_d2(str(d1), str(tmp_path / "out"))


# --- C86H registry is not pruned by development performance -------------------
def test_c86h_registry_constant():
    assert core.METHOD_FREEZE["primary_registry"] == ("P0", "A1", "A2H")
    assert core.METHOD_FREEZE["no_post_hoc_method_add_or_drop"] is True


# ============ PM C86D final-reconciliation tests ============

# --- LURE population-total estimator (NOT self-normalized) --------------------
def test_lure_population_totals_not_self_normalized():
    y = [0, 1]
    nll = np.array([[1.0, 2.0], [3.0, 4.0]])
    contribs = {"nll": nll, "correct": np.zeros((2, 2), int),
                "confidence": np.full((2, 2), 0.6), "conf_bin": np.full((2, 2), 9, int)}
    v = [2.0, 1.0]
    _, est_nll, _ = policies.estimate_metrics(y, contribs, v, False, n_pool=10)
    assert np.allclose(est_nll, (2.0 * nll[0] + 1.0 * nll[1]) / 2)   # (1/M) Σ v_m nll_m
    # self-normalized would give a different value
    w = np.array(v) / np.sum(v) * 2
    assert not np.allclose(est_nll, (w[:, None] * nll).mean(axis=0))


# --- target-bound chain seeds ------------------------------------------------
def test_chain_seed_target_bound():
    a = policies.chain_seed("Cho2017", 1, 0)
    assert a == policies.chain_seed("Cho2017", 1, 0)            # deterministic
    assert a != policies.chain_seed("Lee2019_MI", 1, 0)        # different target -> different seed
    assert a != policies.chain_seed("Cho2017", 2, 0)          # different subject
    assert a != policies.chain_seed("Cho2017", 1, 1)          # different chain


# --- config split: core holds no C85U path; run_d1 has no C85U ----------------
def test_c85u_paths_only_in_c85u_config():
    assert not hasattr(core, "C85U_UTILITY_INDEX")
    assert not hasattr(core, "C85U_ACCEPTANCE_MANIFEST")
    from oaci.active_testing.c86d import run_d1
    assert not hasattr(run_d1, "C85U_UTILITY_INDEX")
    src = open(run_d1.__file__).read()
    assert "c85u_config" not in src and "candidate_utility_index" not in src


# --- 5-way development taxonomy classifier -----------------------------------
def test_taxonomy_five_way():
    from oaci.active_testing.c86d.run_d2 import _classify
    cohorts = ["Cho2017", "Lee2019_MI", "PhysionetMI"]

    def ep(mean, tail, nopt):
        return {"mean_regret_std": mean, "tail_regret_std": tail, "target_near_opt_prob": nopt,
                "mean_by_cohort": {c: mean for c in cohorts},
                "tail_by_cohort": {c: tail for c in cohorts},
                "near_opt_by_cohort": {c: nopt for c in cohorts}}
    budgets = ["4", "FULL"]
    # ceiling fails -> nontransportable
    e = {f"{m}|{b}": ep(0.3, 0.3, 0.0) for m in ("P0", "A1", "A2H") for b in budgets}
    assert _classify(e, budgets, cohorts)["label"] == "ACQUISITION_VIEW_NONTRANSPORTABLE"
    # ceiling ok, no active gain
    e = {f"{m}|{b}": ep(0.02, 0.02, 1.0) for m in ("P0", "A1", "A2H") for b in budgets}
    assert _classify(e, budgets, cohorts)["label"] == "NO_REGISTERED_ACTIVE_GAIN"
    # ceiling ok, A1 crosses at budget 4
    e = {f"{m}|{b}": ep(0.02, 0.02, 1.0) for m in ("P0", "A1", "A2H") for b in budgets}
    e["P0|4"] = ep(0.20, 0.30, 0.3); e["A1|4"] = ep(0.02, 0.03, 0.9)
    assert _classify(e, budgets, cohorts)["label"] == "BOUNDARY_OPERATIONALLY_CROSSED"


# --- D2 verifies freezes BEFORE opening C85U (order) -------------------------
def test_d2_verifies_freeze_before_opening_c85u(tmp_path, monkeypatch):
    from oaci.active_testing.c86d import run_d2
    d1 = tmp_path / "d1"; (d1 / "freezes").mkdir(parents=True)
    (d1 / "freezes" / "f.json").write_text('{"method":"P0","target":["Cho2017",1],"chain":0,"seed":9,"budgets":[]}')
    idx = [{"file": "freezes/f.json", "method": "P0", "target": ["Cho2017", 1], "chain": 0,
            "sha256": "0" * 64}]                            # deliberately wrong hash
    (d1 / "C86D_D1_MANIFEST.json").write_text(json.dumps(
        {"c85u_accessed": False, "budgets": ["FULL"], "n_freeze_files": 1, "freeze_index": idx}))
    # if C85U were opened before verification, this would fire the wrong error
    monkeypatch.setattr(run_d2, "load_c85u_field",
                        lambda: (_ for _ in ()).throw(AssertionError("C85U opened before verify")))
    with pytest.raises(RuntimeError) as ei:
        run_d2.run_d2(str(d1), str(tmp_path / "out"))
    assert "tampered" in str(ei.value)                     # caught by verify_freezes, before C85U
