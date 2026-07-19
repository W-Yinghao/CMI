"""C86H integrated-runner tests — synthetic production-equivalent end-to-end + failure.

Exercises the full F0->H4 chain on a synthetic production-format field (spawn server +
path-blind worker + verify-before-open-held + two-level classify), the freeze tamper
guard, the gated entrypoint refusals, and the outcome-free resource benchmark. NO real
EEG/label is touched.
"""
import glob
import json
import os

import numpy as np
import pytest

from oaci.active_testing.c86h import contract as K
from oaci.active_testing.c86h import benchmark as B
from oaci.active_testing.c86h import batch_h1, field_spec, runner


def test_run_synthetic_e2e(tmp_path):
    m = runner.run_synthetic(str(tmp_path), seed=7, chains=(0, 1), n_trials=88)
    assert m["stage"] == "C86H_H4_TERMINAL_RESULT"
    assert m["confirmatory"] is True
    assert m["held_opened_after_freeze_verification"] is True
    assert m["h1_files"] == 7 * 3                  # 7 targets x 3 methods (all chains per file)
    cls = m["classification"]
    assert cls["formal_gate"] in K.FORMAL_GATE
    assert cls["label_frontier"] in K.LABEL_FRONTIER
    assert cls["interpretive"]["policy_limited"] == "NOT_IDENTIFIABLE_IN_C86H"
    assert cls["interpretive"]["descriptor"] != "POLICY_LIMITED"
    assert m["pooled_dataset_pvalue"] == "FORBIDDEN"
    assert m["maxt_draws"] == 65536 and m["materiality_margin"] == 0.05
    assert any("|P0|FULL" in k for k in m["endpoints"])
    assert set(m["full_ceiling"]) == {"SYN_COHORT_A", "SYN_COHORT_B"}
    # full inference detail frozen (not just PASS/FAIL) + atomic result
    det = m["inference_detail"]
    assert len(det) == 2 * 2 * 4                    # cohorts x active methods x finite budgets
    one = det[sorted(det)[0]]
    for k in ("observed_t", "adjusted_maxt_p", "mean_effect", "worst_target",
              "favorable_fraction", "cell_effects", "loto", "tail_effects", "n_targets"):
        assert k in one
    assert len(m["result_sha256"]) == 64
    assert os.path.isfile(os.path.join(str(tmp_path), "run", "C86H_TERMINAL_RESULT.json"))


def test_batch_h1_equals_per_rpc_worker(tmp_path):
    """The label-independent batch path must give byte-identical selections to the frozen
    per-RPC C86D worker."""
    fr = str(tmp_path / "field")
    field_spec.synthesize_field(
        fr, {"SYN_A": {"dataset": "SYN_A", "subjects": [1, 2], "n_trials": 88}}, seed=9)
    methods = list(K.METHOD_REGISTRY); chains = [0, 1, 2]; exp = [("SYN_A", 1), ("SYN_A", 2)]
    # batch
    odir = str(tmp_path / "orders"); h1 = str(tmp_path / "h1")
    batch_h1.run_h1a(os.path.join(fr, "acquisition_unlabeled_pool"), odir, methods, chains)
    batch_h1.run_h1b_sealed(odir, os.path.join(fr, "acquisition_label_oracle"),
                            os.path.join(fr, "query_contribution_store"), h1, methods, chains)
    batch_h1.verify_h1(h1, odir, exp, methods, chains)
    new = batch_h1.load_selections(h1, methods, exp, chains)
    # per-RPC reference
    fdir, index = runner._run_h1_selection(fr, str(tmp_path / "old"), methods,
                                           list(K.BUDGET_GRID), chains)
    old = {}
    for e in index:
        rec = json.load(open(os.path.join(str(tmp_path / "old"), e["file"])))
        for b in rec["budgets"]:
            if b["status"] == "AVAILABLE":
                old[(rec["method"], tuple(rec["target"]), int(rec["chain"]),
                     b["budget"])] = b["selected_by_context"]
    checked = 0
    for (mm, tt, ch), buds in new.items():
        for bstr, fb in buds.items():
            if fb["status"] == "AVAILABLE":
                assert old[(mm, tt, ch, bstr)] == fb["selected_by_context"]
                checked += 1
    assert checked > 0


def test_verify_h1_reconciles_against_labelfree_orders(tmp_path):
    import hashlib
    fr = str(tmp_path / "field")
    field_spec.synthesize_field(
        fr, {"SYN_A": {"dataset": "SYN_A", "subjects": [1], "n_trials": 88}}, seed=3)
    methods = list(K.METHOD_REGISTRY); chains = [0, 1]; exp = [("SYN_A", 1)]
    odir = str(tmp_path / "orders"); h1 = str(tmp_path / "h1")
    batch_h1.run_h1a(os.path.join(fr, "acquisition_unlabeled_pool"), odir, methods, chains)
    batch_h1.run_h1b_sealed(odir, os.path.join(fr, "acquisition_label_oracle"),
                            os.path.join(fr, "query_contribution_store"), h1, methods, chains)
    batch_h1.verify_h1(h1, odir, exp, methods, chains)          # clean reconciles
    # tamper the label-free H1a orders so they no longer match the H1b freeze
    of = glob.glob(os.path.join(odir, "*.npz"))[0]
    z = dict(np.load(of, allow_pickle=True))
    z["orders"] = z["orders"][:, :, ::-1].copy()
    np.savez(of, **z)
    with pytest.raises(RuntimeError):
        batch_h1.verify_h1(h1, odir, exp, methods, chains)


def test_terminal_result_atomic_immutable_self_hashed(tmp_path):
    import hashlib
    fr = str(tmp_path / "field"); run_dir = str(tmp_path / "run")
    field_spec.synthesize_field(
        fr, {"SYN_A": {"dataset": "SYN_A", "subjects": [1, 2], "n_trials": 88}}, seed=5)
    tc = {("SYN_A", 1): "SYN_A", ("SYN_A", 2): "SYN_A"}; cd = {"SYN_A": "SYN_A"}
    runner.run_confirmation(fr, run_dir, tc, cd, chains=[0, 1], synthetic=True)
    rp = os.path.join(run_dir, "C86H_TERMINAL_RESULT.json")
    assert os.path.isfile(rp + ".sha256")                      # persisted integrity digest
    side = open(rp + ".sha256").read().split()[0]
    assert hashlib.sha256(open(rp, "rb").read()).hexdigest() == side
    with pytest.raises(RuntimeError):                          # one-shot: refuses to overwrite
        runner.run_confirmation(fr, run_dir, tc, cd, chains=[0, 1], synthetic=True)


def test_semantics_b_one_label_per_physical_trial(tmp_path):
    fr = str(tmp_path / "field")
    field_spec.synthesize_field(
        fr, {"SYN_A": {"dataset": "SYN_A", "subjects": [1], "n_trials": 88}}, seed=3)
    # every context npz must give the SAME label for a given physical trial id
    import glob as _g
    seen = {}
    for pf in _g.glob(os.path.join(fr, "query_contribution_store", "*.npz")):
        z = np.load(pf, allow_pickle=True)
        for t, y in zip(z["trial_ids"], z["true_label"]):
            key = str(t)
            if key in seen:
                assert seen[key] == int(y)         # one physical trial -> one label
            seen[key] = int(y)
    assert len(seen) > 0


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
    out = B.resource_benchmark(n_trials=88, n_chains=6)
    assert out["opened_real_data"] is False and out["synthetic"] is True
    assert out["maxt"]["brandl"]["sign_mode"] == "exhaustive"
    assert out["maxt"]["ds007221"]["sign_mode"] == "monte_carlo"
    assert out["extrapolation"]["n_targets"] == 53
    assert out["extrapolation"]["compact_freeze_files"] == 159
    assert out["batch_h1_one_target"]["per_target_seconds"] >= 0.0
    assert out["decision_rule"].startswith("if infeasible")
