"""C86H integrated-runner tests — synthetic production-equivalent end-to-end + failure.

Exercises the full F0->H4 chain on a synthetic production-format field (spawn server +
path-blind worker + verify-before-open-held + two-level classify), the freeze tamper
guard, the gated entrypoint refusals, and the outcome-free resource benchmark. NO real
EEG/label is touched.
"""
import glob
import os

import pytest

from oaci.active_testing.c86h import contract as K
from oaci.active_testing.c86h import benchmark as B
from oaci.active_testing.c86h import field_spec, runner


def test_run_synthetic_e2e(tmp_path):
    m = runner.run_synthetic(str(tmp_path), seed=7, chains=(0, 1), n_trials=88)
    assert m["stage"] == "C86H_H4_TERMINAL_RESULT"
    assert m["confirmatory"] is True
    assert m["held_opened_after_freeze_verification"] is True
    assert m["n_freezes"] == 7 * 3 * 2            # 7 targets x 3 methods x 2 chains
    cls = m["classification"]
    assert cls["formal_gate"] in K.FORMAL_GATE
    assert cls["label_frontier"] in K.LABEL_FRONTIER
    assert cls["interpretive"]["policy_limited"] == "NOT_IDENTIFIABLE_IN_C86H"
    assert cls["interpretive"]["descriptor"] != "POLICY_LIMITED"
    assert m["pooled_dataset_pvalue"] == "FORBIDDEN"
    assert m["maxt_draws"] == 65536 and m["materiality_margin"] == 0.05
    # endpoints exist for every method x budget x cohort
    assert any("|P0|FULL" in k for k in m["endpoints"])
    # FULL ceiling reported per cohort
    assert set(m["full_ceiling"]) == {"SYN_COHORT_A", "SYN_COHORT_B"}


def test_freeze_verification_catches_tamper(tmp_path):
    fr = str(tmp_path / "field")
    field_spec.synthesize_field(
        fr, {"SYN_A": {"dataset": "SYN_A", "subjects": [1, 2], "n_trials": 88}}, seed=3)
    out = str(tmp_path / "run")
    os.makedirs(out, exist_ok=True)
    fdir, index = runner._run_h1_selection(
        fr, out, list(K.METHOD_REGISTRY), list(K.BUDGET_GRID), [0, 1])
    exp_targets = [("SYN_A", 1), ("SYN_A", 2)]
    # clean freezes verify fine
    runner._load_and_verify_freezes(fdir, index, [0, 1], list(K.METHOD_REGISTRY), exp_targets)
    # tamper one freeze blob -> sha in index mismatches -> verification fails closed
    victim = sorted(glob.glob(os.path.join(fdir, "*.json")))[0]
    with open(victim, "a") as fh:
        fh.write(" ")
    with pytest.raises(RuntimeError):
        runner._load_and_verify_freezes(fdir, index, [0, 1], list(K.METHOD_REGISTRY), exp_targets)


def test_execute_gated():
    with pytest.raises(SystemExit):
        runner.execute("授权 C86D")                # wrong token
    with pytest.raises(RuntimeError):
        runner.execute(runner.AUTHORIZATION_TOKEN)  # right token, but no real field


def test_run_confirmation_requires_token_for_real(tmp_path):
    # the code path that opens real data refuses without the token, before opening anything
    with pytest.raises(SystemExit):
        runner.run_confirmation(str(tmp_path), str(tmp_path / "run"), {}, {},
                                authorization="", synthetic=False)


def test_run_confirmation_synthetic_cannot_target_real_root():
    with pytest.raises(RuntimeError):
        runner.run_confirmation(runner.REAL_FIELD_ROOT, "/tmp/c86h_x", {}, {}, synthetic=True)


def test_resource_benchmark_outcome_free():
    out = B.resource_benchmark(pool_size=40, n_chains=6)
    assert out["opened_real_data"] is False and out["synthetic"] is True
    assert out["maxt"]["brandl"]["sign_mode"] == "exhaustive"
    assert out["maxt"]["ds007221"]["sign_mode"] == "monte_carlo"
    assert out["extrapolation"]["n_targets"] == 53
    assert out["decision_rule"].startswith("if infeasible")
