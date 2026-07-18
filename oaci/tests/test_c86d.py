"""C86D shadow + failure tests — process isolation, policies, freeze, evaluator,
exact CVaR, claim boundary, C85U identity, gated execution. No real active run."""
from __future__ import annotations

import os

import numpy as np
import pytest

from oaci.active_testing import c86d
from oaci.active_testing.c86d import core, policies
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
        assert out["n_contexts"] == 8 and 0.0 <= out["target_regret"]
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
    if not os.path.exists(core.C85U_ACCEPTANCE_MANIFEST):
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
