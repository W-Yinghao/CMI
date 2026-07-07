"""Fork 1 Tier-1 --- hard-gate test suite (dummy arrays; no real EEG; no runs).

Verifies the leak-proofing + execution lock:
  config parse / calibration-audit disjoint / B1 cannot accept / B4 diagnostic-only /
  audit labels structurally unavailable to decisions / UNAVAILABLE-k never reuses audit /
  dry-run task count / execute halts when runs disabled.

  python -m tos_cmi.eeg.test_target_info_tier1        # runs all, prints TARGET_INFO_TIER1_TESTS_PASS
"""
from __future__ import annotations
import dataclasses
import numpy as np

from tos_cmi.eeg.target_info_splits import (make_calibration_audit_splits, select_k_per_class,
                                            target_leak_structural_check, budget_action, b1_triage_action,
                                            DecisionContext, AuditView, SourceContext, CalibrationContext,
                                            UnlabeledTargetContext, B1_ACTIONS, is_deployable_budget,
                                            TARGET_LEAK_TOKEN)
from tos_cmi.eeg.run_target_info_tier1_smoke import (load_cfg, build_plan, expand_tasks, run_cli,
                                                     calibration_delta_bacc, audit_scalar, unlabeled_mismatch,
                                                     compute_decision_row, b3_sequential_decision)


def test_config_parses_target_info_driver():
    cfg = load_cfg()
    assert cfg["driver_status"] == "implementation_only"
    assert cfg["experiments_allowed"] is False and cfg["runs_allowed"] is False
    assert cfg["design_lock_hash"] == "3ad4ef312e325fa6"
    for b in ["B0_source_only", "B1_unlabeled_target", "B2_k_labels_per_class",
              "B3_sequential_calibration", "B4_oracle_selector"]:
        assert b in cfg["budgets"] and "allowed_actions" in cfg["budgets"][b]
    assert "accept" not in cfg["budgets"]["B1_unlabeled_target"]["allowed_actions"]
    assert cfg["budgets"]["B4_oracle_selector"]["diagnostic_only"] is True


def test_calibration_audit_disjoint():
    y = np.array([0] * 20 + [1] * 20)
    splits = make_calibration_audit_splits(y, R=10, seed=0)
    assert len(splits) == 10
    for cal, aud in splits:
        assert set(cal.tolist()).isdisjoint(aud.tolist())
        assert len(cal) > 0 and len(aud) > 0
        assert set(np.unique(y[cal])) == {0, 1} and set(np.unique(y[aud])) == {0, 1}
    # the structural gate must PASS on these
    cfg = load_cfg()
    assert target_leak_structural_check(splits, cfg["budgets"]) == TARGET_LEAK_TOKEN
    # and must FAIL (raise) if calibration and audit overlap
    bad = [(np.array([0, 1, 2]), np.array([2, 3, 4]))]
    try:
        target_leak_structural_check(bad, cfg["budgets"]); raise SystemExit("gate did not catch overlap")
    except (AssertionError, ValueError):
        pass


def test_b1_accept_forbidden():
    # even an extreme target mismatch must NOT yield accept
    for score in [0.99, 1.0, 5.0, 1e9]:
        assert b1_triage_action(score) in B1_ACTIONS
        assert b1_triage_action(score) != "accept"
    # budget_action for B1 ignores any (illegally strong) calibration benefit and never accepts
    ctx = DecisionContext(budget="B1_unlabeled_target", source_safety_pass=True,
                          cal_benefit_lcb=999.0, beats_random=True, unlabeled_triage="request_labels")
    assert budget_action(ctx) == "request_labels"
    # an illegal B1 action is rejected loudly (unconditional raise, survives -O)
    bad = DecisionContext(budget="B1_unlabeled_target", source_safety_pass=True, unlabeled_triage="accept")
    try:
        budget_action(bad); raise SystemExit("B1 accept was not blocked")
    except (AssertionError, ValueError):
        pass


def test_b4_oracle_diagnostic_only():
    ctx = DecisionContext(budget="B4_oracle_selector", source_safety_pass=True,
                          cal_benefit_lcb=999.0, beats_random=True, source_benefit_lcb=999.0)
    assert budget_action(ctx) == "DIAGNOSTIC"            # never 'accept'
    assert not is_deployable_budget("B4_oracle_selector")   # excluded from deployable accept accounting
    assert is_deployable_budget("B2_k_labels_per_class")


def test_audit_labels_unavailable_to_decision():
    # DecisionContext must have NO audit field; audit lives only in AuditView.
    dfields = set(f.name for f in dataclasses.fields(DecisionContext))
    assert not any("audit" in f for f in dfields), dfields
    afields = set(f.name for f in dataclasses.fields(AuditView))
    assert "audit_y" in afields and "audit_idx" in afields
    # config: no budget lists audit labels as a decision input
    cfg = load_cfg()
    for b, spec in cfg["budgets"].items():
        assert "audit_y" not in spec.get("decision_inputs", [])
        assert "target_audit_labels" not in spec.get("decision_inputs", [])


def test_unavailable_k_never_reuses_audit_labels():
    y = np.array([0, 0, 1, 1, 1, 1])                     # class 0 has only 2 trials total
    splits = make_calibration_audit_splits(y, R=1, seed=0)
    cal, aud = splits[0]
    aud_before = aud.copy()
    # request k larger than the calibration pool for class 0
    sel, status, eff = select_k_per_class(cal, y, k=8, seed=0)
    assert status == "UNAVAILABLE" and sel is None
    assert np.array_equal(aud, aud_before)               # audit indices untouched
    # a feasible small k works and draws ONLY from calibration
    sel2, status2, _ = select_k_per_class(cal, y, k=1, seed=0)
    if status2 == "OK":
        assert set(sel2.tolist()).issubset(set(cal.tolist()))
        assert set(sel2.tolist()).isdisjoint(aud.tolist())


def test_dry_run_has_expected_task_count():
    cfg = load_cfg()
    plan = build_plan(cfg)
    # 2 datasets x 1 backbone x 2 worlds x 5 folds x 5 budgets x 7 interventions
    assert len(plan) == 2 * 1 * 2 * 5 * 5 * 7 == 700, len(plan)


def test_execute_halts_when_runs_disabled():
    cfg = load_cfg()
    for argv in ([], ["--execute"]):                     # bare AND --execute must both halt
        code, msg = run_cli(argv, cfg)
        assert code == 1 and msg.startswith("EXPERIMENTS_DISABLED"), (argv, code, msg)
    # only --dry-run is permitted (produces the plan, exit 0)
    code, msg = run_cli(["--dry-run"], cfg)
    assert code == 0 and "DRYRUN_DONE" in msg


# ---------------- hardening tests (from the gate red-team wf_a5f4ad2a-d06) ----------------
def test_gates_reject_optimized_bytecode():
    """R1: the leak layer must not silently no-op under -O; the module refuses to import (asserts stripped)."""
    import subprocess, sys, os
    env = dict(os.environ); env["PYTHONPATH"] = "/home/infres/yinwang/CMI_AAAI_tos"
    r = subprocess.run([sys.executable, "-O", "-c", "import tos_cmi.eeg.target_info_splits"],
                       capture_output=True, text=True, env=env)
    assert r.returncode != 0, "module must REFUSE to import under -O (got exit 0)"
    assert "assertions enabled" in (r.stderr + r.stdout), (r.stderr + r.stdout)[-300:]


def test_audit_alias_rejected():
    """R2: any audit-label alias as a decision input must be rejected by the positive allowlist."""
    cfg = load_cfg()
    for alias in ["audit_y", "target_audit_labels", "target_audit_y", "AUDIT_Y", "aud_labels"]:
        specs = {k: dict(v) for k, v in cfg["budgets"].items()}
        specs["B2_k_labels_per_class"] = dict(specs["B2_k_labels_per_class"],
                                              decision_inputs=["source_safety", alias])
        try:
            target_leak_structural_check([(np.array([0]), np.array([1]))], specs)
            raise SystemExit("audit alias %r slipped past the gate" % alias)
        except (AssertionError, ValueError):
            pass


def test_b4_synonym_and_family_rejected():
    """R3/R5: mis-cased/spaced accept synonyms in B4, and mis-prefixed budget families, must be rejected."""
    cfg = load_cfg()
    for bad_action in ["Accept", "ACCEPT", "accept ", "deployable_accept"]:
        specs = {k: dict(v) for k, v in cfg["budgets"].items()}
        specs["B4_oracle_selector"] = dict(specs["B4_oracle_selector"], allowed_actions=[bad_action])
        try:
            target_leak_structural_check([(np.array([0]), np.array([1]))], specs)
            raise SystemExit("B4 accept synonym %r slipped past the gate" % bad_action)
        except (AssertionError, ValueError):
            pass
    # unknown budget family must raise, not be treated as deployable/plausible
    for bad_budget in ["B1x_thing", "B04_oracle", "B2B4", "Z9_weird"]:
        try:
            is_deployable_budget(bad_budget); raise SystemExit("unknown family %r accepted" % bad_budget)
        except ValueError:
            pass


def test_k_nonpositive_guard():
    """R4: k <= 0 is an error, not a silent 'OK' empty/negative-slice selection."""
    y = np.array([0, 0, 1, 1])
    for bad_k in [0, -1, -8]:
        try:
            select_k_per_class(np.array([0, 1, 2, 3]), y, k=bad_k, seed=0)
            raise SystemExit("k=%r was not rejected" % bad_k)
        except ValueError:
            pass


def test_dry_run_execute_conflict():
    """R6: contradictory --dry-run --execute is rejected (not silently dry-run)."""
    cfg = load_cfg()
    code, msg = run_cli(["--dry-run", "--execute"], cfg)
    assert code == 2 and "CONFLICTING_FLAGS" in msg, (code, msg)


# ---------------- executable-wiring tests (from the PM patch spec) ----------------
def _dummy_world(seed=0, n=60):
    rng = np.random.default_rng(seed)
    Zs = rng.standard_normal((n, 4)); ys = np.array([0, 1] * (n // 2))
    Zt = rng.standard_normal((n, 4)); yt = np.array(([0, 1] * (n // 2)))
    return Zs, ys, Zt, yt


def _decision_and_audit(src, Zt, yt, splits, E, Erand, budget="B2_k_labels_per_class"):
    """Mimic one executor task: calibration deltas -> decision row (no audit) ; audit scalar (audit only)."""
    cal_deltas, cal_rand, clusters, audit_vals = [], [], [], []
    for i, (cal_idx, aud_idx) in enumerate(splits):
        cal = CalibrationContext(Zt[cal_idx], yt[cal_idx], {})
        cal_deltas.append(calibration_delta_bacc(E, src, cal))
        cal_rand.append(calibration_delta_bacc(Erand, src, cal))
        clusters.append(i % 3)
        av = AuditView(aud_idx, yt[aud_idx])
        audit_vals.append(audit_scalar(E, src, av, Zt[aud_idx]))
    row = compute_decision_row(budget, True, None, cal_deltas, cal_rand, clusters, None, boot_seed=0)
    return row, audit_vals


def test_expanded_task_count_matches_schema():
    cfg = load_cfg()
    exp = expand_tasks(cfg)
    # per (ds,bb,world,alpha,fold,interv) = 2*1*2*3*5*7 = 420 combos; budgets B0+B1+B4=1 each, B2=5k*10R=50, B3=10
    per_combo = 1 + 1 + 50 + 10 + 1                             # =63
    assert len(exp) == 420 * per_combo == 26460, len(exp)
    for t in exp:                                               # every expanded task carries the required fields
        for fld in ["dataset", "backbone", "world", "alpha", "fold", "target_subject", "intervention",
                    "budget", "calibration_idx_hash", "audit_idx_hash", "random_seed"]:
            assert fld in t, (fld, t)


def test_calibration_audit_disjoint_all_splits():
    y = np.array([0] * 30 + [1] * 30)
    for seed in range(3):
        splits = make_calibration_audit_splits(y, R=10, seed=seed)
        for cal, aud in splits:
            assert set(cal.tolist()).isdisjoint(aud.tolist())


def test_decision_invariant_to_audit_label_permutation():
    """STRONGEST leak check: permuting AUDIT labels must not change any decision row (only audit metrics may)."""
    Zs, ys, Zt, yt = _dummy_world()
    src = SourceContext(Zs, ys, ys.copy(), 2)
    E = (lambda X: X); Erand = (lambda X: X)
    splits = make_calibration_audit_splits(yt, R=6, seed=0)
    row_a, _ = _decision_and_audit(src, Zt, yt, splits, E, Erand)
    yt2 = yt.copy()
    rng = np.random.default_rng(7)
    for _, aud_idx in splits:                                   # permute ONLY audit labels (cal untouched)
        yt2[aud_idx] = rng.permutation(yt[aud_idx])
    row_b, _ = _decision_and_audit(src, Zt, yt2, splits, E, Erand)
    assert row_a == row_b, (row_a, row_b)                      # decision invariant to audit permutation


def test_decision_changes_or_can_change_under_calibration_label_permutation():
    """The decision DOES depend on calibration content: a positive vs negative calibration signal flips it."""
    clusters = [0, 1, 2, 0, 1, 2]
    hi = compute_decision_row("B2_k_labels_per_class", True, None, [0.5] * 6, [0.0] * 6, clusters, None)
    lo = compute_decision_row("B2_k_labels_per_class", True, None, [-0.5] * 6, [0.0] * 6, clusters, None)
    assert hi["action"] == "accept" and lo["action"] == "abstain" and hi != lo


def test_b1_never_accepts_even_with_large_unlabeled_shift():
    Zs = np.zeros((20, 4)); ys = np.array([0, 1] * 10)
    src = SourceContext(Zs, ys, ys.copy(), 2)
    unl = UnlabeledTargetContext(np.ones((20, 4)) * 1000.0)     # enormous covariate shift
    ms = unlabeled_mismatch(src, unl)
    row = compute_decision_row("B1_unlabeled_target", True, None, None, None, None, ms)
    assert row["action"] in ("reject", "abstain", "request_labels") and row["action"] != "accept"


def test_b3_sequential_stops_without_reading_future_labels():
    clusters = [0, 1, 2]
    deltas = {k: [0.5, 0.5, 0.5] for k in [1, 2, 4, 8, 16]}     # strong benefit already at k=1
    rand = {k: [0.0, 0.0, 0.0] for k in [1, 2, 4, 8, 16]}
    out = b3_sequential_decision([1, 2, 4, 8, 16], deltas, rand, clusters)
    assert out["action"] == "accept" and out["k_used"] == 1 and out["ks_read"] == [1]   # no future-k labels read


def test_b4_excluded_from_deployable_accept_counts():
    rows = [{"budget": "B2_k_labels_per_class", "action": "accept"},
            {"budget": "B4_oracle_selector", "action": "DIAGNOSTIC"},
            {"budget": "B0_source_only", "action": "abstain"}]
    deployable = [r for r in rows if is_deployable_budget(r["budget"])]
    assert all(r["budget"] != "B4_oracle_selector" for r in deployable) and len(deployable) == 2


def test_same_k_random_specificity_flag():
    clusters = [0, 1, 2, 0, 1, 2]
    spec = compute_decision_row("B2_k_labels_per_class", True, None, [0.5] * 6, [0.0] * 6, clusters, None)
    assert spec["action"] == "accept" and spec["specificity"] == "accepted_specific"
    nons = compute_decision_row("B2_k_labels_per_class", True, None, [0.5] * 6, [0.9] * 6, clusters, None)
    assert nons["action"] == "accept" and nons["specificity"] == "accepted_non_specific"   # random reproduces


def test_execute_still_halts_when_runs_disabled():
    cfg = load_cfg()
    for argv in ([], ["--execute"]):
        code, msg = run_cli(argv, cfg)
        assert code == 1 and msg.startswith("EXPERIMENTS_DISABLED"), (argv, code, msg)


ALL = [test_config_parses_target_info_driver, test_calibration_audit_disjoint, test_b1_accept_forbidden,
       test_b4_oracle_diagnostic_only, test_audit_labels_unavailable_to_decision,
       test_unavailable_k_never_reuses_audit_labels, test_dry_run_has_expected_task_count,
       test_execute_halts_when_runs_disabled,
       test_gates_reject_optimized_bytecode, test_audit_alias_rejected, test_b4_synonym_and_family_rejected,
       test_k_nonpositive_guard, test_dry_run_execute_conflict,
       test_expanded_task_count_matches_schema, test_calibration_audit_disjoint_all_splits,
       test_decision_invariant_to_audit_label_permutation,
       test_decision_changes_or_can_change_under_calibration_label_permutation,
       test_b1_never_accepts_even_with_large_unlabeled_shift,
       test_b3_sequential_stops_without_reading_future_labels, test_b4_excluded_from_deployable_accept_counts,
       test_same_k_random_specificity_flag, test_execute_still_halts_when_runs_disabled]


def main():
    for t in ALL:
        t(); print("  PASS", t.__name__)
    print("TARGET_INFO_TIER1_TESTS_PASS (%d/%d)" % (len(ALL), len(ALL)))


if __name__ == "__main__":
    main()
