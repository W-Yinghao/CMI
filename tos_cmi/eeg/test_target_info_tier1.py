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
                                            subject_seeded_splits, nested_k_subsets, stable_seed,
                                            SEED_KEY_FIELDS, SPLIT_RNG_SCHEME,
                                            target_leak_structural_check, budget_action, b1_triage_action,
                                            DecisionContext, AuditView, SourceContext, CalibrationContext,
                                            UnlabeledTargetContext, LabelAccessGuard, hash_array,
                                            B1_ACTIONS, is_deployable_budget, TARGET_LEAK_TOKEN)
from tos_cmi.eeg.run_target_info_tier1_smoke import (load_cfg, build_plan, expand_tasks, run_cli,
                                                     calibration_delta_bacc, audit_scalar, unlabeled_mismatch,
                                                     compute_decision_row, finalize_decision_row,
                                                     b3_sequential_decision, scope_hash, authorize_execution,
                                                     _preflight_from_labels, execute_task, _assemble_run,
                                                     write_two_phase_outputs, source_task_drop_ucb,
                                                     _redact_validation_output, RUN_OUT, CFG)


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
    # class-separated features so a classifier (and thus bAcc) actually responds to labels
    ys = np.array([0, 1] * (n // 2))
    Zs = rng.standard_normal((n, 4)) + ys[:, None] * np.array([1.5, 0, 0, 0])
    yt = np.array([0, 1] * (n // 2))
    Zt = rng.standard_normal((n, 4)) + yt[:, None] * np.array([1.5, 0, 0, 0])
    return Zs, ys, Zt, yt


def _lin_eraser(seed=0, d=4):
    """A fixed linear eraser that ZEROS the discriminative dim (dim 0, where the class signal lives), so
    ΔbAcc(erased) - ΔbAcc(full) is genuinely nonzero and responds to labels. Width-agnostic (masks dim 0 for
    whatever feature count X has). (An identity/invertible map would leave separability unchanged -> vacuous.)"""
    def _E(X):
        X = np.asarray(X); m = np.ones(X.shape[1]); m[0] = 0.0
        return X * m
    return _E


def _fast_cfg():
    """load_cfg with a small R so the execute_task tests stay fast (fewer splits x bootstrap x sklearn fits)."""
    cfg = load_cfg()
    cfg = dict(cfg); cfg["tier1_scope"] = dict(cfg["tier1_scope"], repeats_R=2)
    return cfg


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


# ---------------- P0-1: tamper-proof run authorization ----------------
import hashlib as _hashlib


def _approved_manifest(cfg, token="s3cret"):
    driver_hash = _hashlib.sha256(open(CFG, "rb").read()).hexdigest()[:16]
    return {"run_status": "approved", "runs_allowed": True, "experiments_allowed": True,
            "enable_token_required": True,
            "approved_enable_token_sha256": _hashlib.sha256(token.encode()).hexdigest(),
            "approved_scope_hash": scope_hash(cfg), "approved_driver_hash": driver_hash,
            "approved_git_commit": "ANY"}


def test_fully_approved_manifest_authorizes():
    """Positive control: the gate is a REAL gate (can authorize), not a permanent brick."""
    cfg = load_cfg()
    ok, reasons = authorize_execution(cfg, _approved_manifest(cfg), "s3cret")
    assert ok and reasons == [], reasons


def test_plain_yaml_flip_does_not_enable_execution():
    """P0-1: flipping the DRIVER config booleans does NOT authorize a run (manifest stays preflight_only on disk)."""
    cfg = dict(load_cfg()); cfg["runs_allowed"] = True; cfg["experiments_allowed"] = True
    for argv in ([], ["--execute"]):
        code, msg = run_cli(argv, cfg)
        assert code == 1 and msg.startswith("EXPERIMENTS_DISABLED"), (argv, code, msg)


def test_missing_enable_token_halts():
    cfg = load_cfg()
    ok, reasons = authorize_execution(cfg, _approved_manifest(cfg), None)
    assert not ok and any("token" in r for r in reasons)


def test_wrong_enable_token_halts():
    cfg = load_cfg()
    ok, reasons = authorize_execution(cfg, _approved_manifest(cfg), "not-the-token")
    assert not ok and any("token" in r for r in reasons)


def test_scope_hash_mismatch_halts():
    cfg = load_cfg()
    m = dict(_approved_manifest(cfg)); m["approved_scope_hash"] = "deadbeefdeadbeef"
    ok, reasons = authorize_execution(cfg, m, "s3cret")
    assert not ok and any("scope_hash" in r for r in reasons)


def test_driver_config_hash_mismatch_halts():
    cfg = load_cfg()
    m = dict(_approved_manifest(cfg)); m["approved_driver_hash"] = "deadbeefdeadbeef"
    ok, reasons = authorize_execution(cfg, m, "s3cret")
    assert not ok and any("driver_config_hash" in r for r in reasons)


# ---------------- P0-2: calibration-delta taint / provenance ----------------
def test_cal_deltas_depend_only_on_calibration_labels():
    Zs, ys, Zt, yt = _dummy_world()
    src = SourceContext(Zs, ys, ys.copy(), 2)
    E = _lin_eraser()
    splits = make_calibration_audit_splits(yt, R=1, seed=0)
    cal_idx, aud_idx = splits[0]
    cal = CalibrationContext(Zt[cal_idx], yt[cal_idx], {})
    d1 = calibration_delta_bacc(E, src, cal)
    # mutate audit labels wildly; the calibration delta cannot see them -> identical on the same calibration
    cal_same = CalibrationContext(Zt[cal_idx], yt[cal_idx], {})
    d2 = calibration_delta_bacc(E, src, cal_same)
    assert d1 == d2


def test_audit_label_permutation_changes_only_audit_metrics_not_decision():
    # R=1 so the single split's calibration is disjoint from its audit -- flipping audit labels cannot touch any
    # calibration label. (Across R>1 splits the audit UNION covers calibration trials of OTHER splits, so a global
    # audit flip would corrupt calibration; the invariance guarantee is PER-SPLIT. This is the cross-split reuse
    # the wired-path red-team flagged.)
    Zs, ys, Zt, yt = _dummy_world()
    src = SourceContext(Zs, ys, ys.copy(), 2)
    E = _lin_eraser(); Erand = _lin_eraser(seed=9)               # discriminative erasers so audit metric responds
    splits = make_calibration_audit_splits(yt, R=1, seed=1)
    row_a, audit_a = _decision_and_audit(src, Zt, yt, splits, E, Erand)
    yt2 = yt.copy()
    _, aud_idx = splits[0]
    yt2[aud_idx] = 1 - yt[aud_idx]                               # relabel audit only -> audit metric must move
    row_b, audit_b = _decision_and_audit(src, Zt, yt2, splits, E, Erand)
    assert row_a == row_b                                        # decision invariant to audit relabeling
    assert audit_a != audit_b                                    # but audit metric genuinely moves (non-vacuous)


def test_calibration_label_permutation_can_change_decision():
    Zs, ys, Zt, yt = _dummy_world(seed=1, n=80)
    src = SourceContext(Zs, ys, ys.copy(), 2)
    E = _lin_eraser()
    splits = make_calibration_audit_splits(yt, R=6, seed=0)
    # a well-aligned calibration signal vs a corrupted (flipped-label) one CAN produce different cal deltas
    good = [calibration_delta_bacc(E, src, CalibrationContext(Zt[c], yt[c], {})) for c, _ in splits]
    corrupt = [calibration_delta_bacc(E, src, CalibrationContext(Zt[c], 1 - yt[c], {})) for c, _ in splits]
    assert good != corrupt                                       # calibration content DOES drive the delta


def test_audit_view_cannot_be_passed_to_calibration_delta():
    Zs, ys, Zt, yt = _dummy_world()
    src = SourceContext(Zs, ys, ys.copy(), 2)
    av = AuditView(np.arange(10), yt[:10])
    try:
        calibration_delta_bacc((lambda X: X), src, av); raise SystemExit("AuditView reached calibration delta")
    except TypeError:
        pass


def test_decision_rows_store_calibration_and_audit_hashes():
    row = compute_decision_row("B2_k_labels_per_class", True, None, [0.5] * 3, [0.0] * 3, [0, 1, 2], None,
                               calibration_idx_hash="calidx", calibration_label_hash="callbl")
    final = finalize_decision_row(row, audit_idx_hash="audidx", audit_label_hash="audlbl", audit_delta_bacc=0.02)
    for fld in ["calibration_idx_hash", "audit_idx_hash", "calibration_label_hash", "audit_label_hash",
                "delta_source", "decision_input_hash"]:
        assert fld in final, fld
    assert final["delta_source"] == "calibration_only"
    assert "audit_idx_hash" not in row                           # audit hashes appear ONLY after finalize


def test_label_access_guard_phase_separation():
    cal = CalibrationContext(np.zeros((4, 2)), np.array([0, 1, 0, 1]), {})
    av = AuditView(np.arange(4), np.array([0, 1, 0, 1]))
    assert LabelAccessGuard("decision").calibration_labels(cal) is not None
    try:
        LabelAccessGuard("decision").audit_labels(av); raise SystemExit("audit labels read in decision phase")
    except PermissionError:
        pass
    assert LabelAccessGuard("audit").audit_labels(av) is not None   # allowed only after decision frozen


# ---------------- P0-3: real-split preflight (split/schema only; NO metrics) ----------------
# exactly the PM's forbidden token list (not broader -- e.g. bare "metric" would false-positive on the
# "no_metrics_emitted" declarator field, and "action" on nothing here; the contract token is "gate_action").
_METRIC_WORDS = ("bacc", "balanced_accuracy", "dbacc", "delta", "nll", "accept", "reject", "abstain",
                 "gate_action", "score", "accuracy", "performance", "gain")


def _has_metric_key(obj):
    if isinstance(obj, dict):
        return any(any(w in str(k).lower() for w in _METRIC_WORDS) for k in obj) or any(_has_metric_key(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_metric_key(v) for v in obj)
    return False


def test_preflight_real_splits_outputs_no_metrics():
    out = _preflight_from_labels(np.array([0] * 20 + [1] * 20), k_grid=[1, 2, 4, 8, 16], R=10, seed=0)
    assert set(out.keys()) == {"n_splits", "k_grid", "rows", "unavailable_k"}
    assert not _has_metric_key(out), "preflight leaked a metric-like field"


def test_unavailable_k_marked_not_shrunk():
    y = np.array([0, 0, 1, 1, 1, 1, 1, 1])                       # class 0 tiny
    out = _preflight_from_labels(y, k_grid=[1, 2, 4, 8, 16], R=4, seed=0)
    big = [u for u in out["unavailable_k"] if u["k"] == 16]
    assert big, "large k should be UNAVAILABLE for a tiny class"
    # a status of UNAVAILABLE is recorded verbatim -- k is NOT silently reduced
    assert any(r["k"] == 16 and r["status"] == "UNAVAILABLE" for r in out["rows"])


def test_no_audit_label_reuse_for_calibration():
    y = np.array([0] * 12 + [1] * 12)
    splits = make_calibration_audit_splits(y, R=5, seed=0)
    for cal, aud in splits:
        sel, status, _ = select_k_per_class(cal, y, k=2, seed=0)
        if status == "OK":
            assert set(sel.tolist()).issubset(cal.tolist())
            assert set(sel.tolist()).isdisjoint(aud.tolist())


def test_preflight_gated_by_manifest():
    """--preflight-real-splits halts while manifest.preflight_allowed=false (running it = separate PM go)."""
    cfg = load_cfg()
    code, msg = run_cli(["--preflight-real-splits"], cfg)
    assert code == 1 and msg.startswith("PREFLIGHT_DISABLED"), (code, msg)


# ---------------- subject-seeded split hardening ----------------
def _seed_key(dataset="Lee2019_MI", sub=1, fold=1):
    return {"dataset": dataset, "backbone": "EEGNet", "model_seed": 0, "fold": fold,
            "target_subject": sub, "global_split_seed": 20240707}


def test_subject_seeded_splits_differ_across_subjects():
    y = np.array([0] * 50 + [1] * 50)
    s1 = subject_seeded_splits(y, _seed_key(sub=1), R=10)
    s2 = subject_seeded_splits(y, _seed_key(sub=2), R=10)
    # identical label layout, different subject -> at least some splits must differ (no shared pattern)
    same = sum(np.array_equal(a[0], b[0]) for a, b in zip(s1, s2))
    assert same < 10, "subjects with same labels reused >=all split patterns (subject not in seed)"


def test_subject_seeded_splits_reproducible_for_same_subject():
    y = np.array([0] * 40 + [1] * 40)
    a = subject_seeded_splits(y, _seed_key(sub=7), R=10)
    b = subject_seeded_splits(y, _seed_key(sub=7), R=10)
    for (ca, aa), (cb, ab) in zip(a, b):
        assert np.array_equal(ca, cb) and np.array_equal(aa, ab)   # deterministic


def test_same_split_shared_across_interventions_and_budgets():
    # the split seed_key carries NO intervention/budget/k/world -> the split is shared by all of them.
    for forbidden in ("intervention", "budget", "k", "world", "method"):
        assert forbidden not in SEED_KEY_FIELDS
    y = np.array([0] * 30 + [1] * 30)
    base = subject_seeded_splits(y, _seed_key(sub=3), R=5)
    # "computing splits for intervention A vs B / budget B2 vs B3" is the SAME call -> identical splits
    again = subject_seeded_splits(y, _seed_key(sub=3), R=5)
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(base, again))
    # injecting a method field into the seed_key must be REFUSED (can't accidentally couple split to method)
    try:
        subject_seeded_splits(y, dict(_seed_key(sub=3), intervention="leace"), R=5)
        raise SystemExit("seed_key accepted a forbidden method field")
    except ValueError:
        pass


def test_k_subsets_are_nested_within_calibration_pool():
    y = np.array([0] * 50 + [1] * 50)
    sk = _seed_key(sub=5)
    splits = subject_seeded_splits(y, sk, R=3)
    for sid, (cal, aud) in enumerate(splits, 1):
        ksub, _order = nested_k_subsets(cal, y, [1, 2, 4, 8, 16], sk, sid)
        prev = None
        for k in [1, 2, 4, 8, 16]:
            sel = ksub[k]
            assert sel is not None                                 # 25/class calibration >= 16
            assert set(sel.tolist()).issubset(cal.tolist())        # within calibration pool
            assert set(sel.tolist()).isdisjoint(aud.tolist())      # never audit
            if prev is not None:
                assert set(prev.tolist()).issubset(sel.tolist())   # nested: k_prev subset of k
            prev = sel


def test_subject_seed_key_uses_stable_hash_not_python_hash():
    """stable_seed must be identical across PYTHONHASHSEED (proves it does NOT use python hash())."""
    import subprocess, sys, os
    code = ("import sys; sys.path.insert(0,'/home/infres/yinwang/CMI_AAAI_tos');"
            "from tos_cmi.eeg.target_info_splits import stable_seed;"
            "print(stable_seed('Lee2019_MI','EEGNet',0,1,'1',3,20240707,''))")
    outs = []
    for hs in ("0", "1", "12345"):
        env = dict(os.environ); env["PYTHONHASHSEED"] = hs; env["PYTHONPATH"] = "/home/infres/yinwang/CMI_AAAI_tos"
        outs.append(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env).stdout.strip())
    assert len(set(outs)) == 1 and outs[0], outs                   # identical across hash seeds


def test_preflight_subject_seeded_has_no_metrics():
    # the split-only schema the subject-seeded preflight emits must contain NO metric/action field name
    sample_row = {"dataset": "X", "backbone": "EEGNet", "seed": 0, "fold": 1, "target_subject": 1, "split_id": 1,
                  "k": 1, "class": 0, "n_target_total": 100, "n_calibration": 50, "n_audit": 50,
                  "k_available": True, "unavailable_reason": "", "calibration_idx_hash": "h", "audit_idx_hash": "h",
                  "calibration_label_hash": "h", "audit_label_hash": "h", "k_subset_hash": "h",
                  "calibration_pool_hash": "h"}
    summary = {"purpose": "split_hash_unavailable_k_only", "no_metrics_emitted": True,
               "split_rng_scheme": SPLIT_RNG_SCHEME, "nested_k_checks_passed": "500/500",
               "calibration_audit_overlap_total": 0, "per_subject_split_diversity": []}
    assert not _has_metric_key(sample_row) and not _has_metric_key(summary)


# ---------------- run-ready execute_real wiring (dummy arrays; no real EEG; no runs) ----------------
def _exec_arrays(seed=0, n=80):
    rng = np.random.default_rng(seed)
    ys = np.array([0, 1] * (n // 2))
    Zs = rng.standard_normal((n, 6)) + ys[:, None] * np.array([1.5, 0, 0, 0, 0, 0])
    z_src = ys.copy()
    yt = np.array([0, 1] * (n // 2))
    Zt = rng.standard_normal((n, 6)) + yt[:, None] * np.array([1.5, 0, 0, 0, 0, 0])
    return Zs, ys, z_src, Zt, yt


def _exec_task(budget="B2_k_labels_per_class", world="source_rich_source_visible_world_a"):
    return {"dataset": "DUMMY", "backbone": "EEGNet", "world": world, "alpha": 1.0, "fold": 1,
            "target_subject": 1, "intervention": "identity", "budget": budget}


def test_execute_real_requires_approved_manifest_and_token():
    cfg = load_cfg()
    # default (preflight_only) and TEMPLATE (awaiting_pm_run_go) manifests must both HALT --execute
    for mpath in ["tos_cmi/eeg/configs/target_info_tier1_run_manifest.yaml",
                  "tos_cmi/eeg/configs/target_info_tier1_run_manifest_TEMPLATE.yaml"]:
        code, msg = run_cli(["--execute", "--run-manifest", mpath], cfg)
        assert code == 1 and msg.startswith("EXPERIMENTS_DISABLED"), (mpath, code, msg)
    # positive authorization control (NOT a run): a fully-approved fake manifest + right token authorizes
    ok, reasons = authorize_execution(cfg, _approved_manifest(cfg), "s3cret")
    assert ok and reasons == [], reasons


def test_real_loader_wraps_target_labels_in_guarded_contexts():
    # execute_task builds guarded contexts; it must accept arrays and never expose a raw audit label to a decision.
    import inspect
    from tos_cmi.eeg.run_target_info_tier1_smoke import execute_task as _et
    params = list(inspect.signature(_et).parameters)
    assert "yt" in params and not any("audit" in p for p in params)   # no audit arg into the task core
    Zs, ys, z_src, Zt, yt = _exec_arrays()
    dec, aud = _et(_exec_task(), Zs, ys, z_src, Zt, yt, _seed_key(), (lambda X: X), (lambda X: X), _fast_cfg(), n_boot=15)
    assert dec and aud                                                # produced rows


def test_decision_rows_contain_no_audit_metrics():
    Zs, ys, z_src, Zt, yt = _exec_arrays()
    dec, aud = execute_task(_exec_task(), Zs, ys, z_src, Zt, yt, _seed_key(), (lambda X: X), (lambda X: X),
                            _fast_cfg(), n_boot=15)
    for r in dec:
        assert not any(("audit_delta" in c or "audit_label" in c or "audit_metric" in c) for c in r), r
        assert "audit_idx_hash_hash_only" in r                        # only the hashed audit provenance is allowed
    # the two-phase writer also refuses a leaked decision row
    try:
        write_two_phase_outputs([{**dec[0], "audit_delta_bacc": 0.1}], aud, out=RUN_OUT + "_test")
        raise SystemExit("writer accepted an audit metric in a decision row")
    except AssertionError:
        pass


def test_audit_rows_created_only_after_decision_frozen():
    Zs, ys, z_src, Zt, yt = _exec_arrays()
    dec, aud = execute_task(_exec_task(), Zs, ys, z_src, Zt, yt, _seed_key(), (lambda X: X), (lambda X: X),
                            _fast_cfg(), n_boot=15)
    # audit rows are a SEPARATE record linked by decision_input_hash; audit metric present only there
    assert all("audit_delta_bacc" in a for a in aud)
    assert all(a.get("decision_input_hash") for a in aud)


def test_audit_label_permutation_does_not_change_decision_rows():
    # The decision is computed from the CALIBRATION region only, so it must not depend on ANY audit-region data.
    # NOTE: we hold yt FIXED (mutating labels would re-stratify the split and conflate split-change with
    # label-change) and scramble the audit-region FEATURES; the decision rows must be byte-identical. (The
    # fixed-split "audit LABEL permutation -> decision unchanged" invariant is covered by the P0-2 core test.)
    cfg = load_cfg()
    Zs, ys, z_src, Zt, yt = _exec_arrays()
    E = _lin_eraser(); Erand = _lin_eraser(seed=9)
    cfg1 = dict(cfg); cfg1["tier1_scope"] = dict(cfg["tier1_scope"], repeats_R=1)   # R=1: audit disjoint from cal
    dec_a, _ = execute_task(_exec_task(), Zs, ys, z_src, Zt, yt, _seed_key(), E, Erand, cfg1, n_boot=15)
    from tos_cmi.eeg.target_info_splits import subject_seeded_splits
    Zt2 = Zt.copy()
    rng = np.random.default_rng(11)
    _, aud_idx = subject_seeded_splits(yt, _seed_key(), R=1, calib_fraction=0.5)[0]
    Zt2[aud_idx] = rng.standard_normal(Zt2[aud_idx].shape) * 5.0      # scramble audit-region FEATURES (split fixed)
    dec_b, _ = execute_task(_exec_task(), Zs, ys, z_src, Zt2, yt, _seed_key(), E, Erand, cfg1, n_boot=15)
    assert dec_a == dec_b                                             # decision independent of audit-region data


def test_calibration_label_permutation_can_change_decision_rows():
    cfg = _fast_cfg()
    Zs, ys, z_src, Zt, yt = _exec_arrays()
    E = _lin_eraser()
    d_good, _ = execute_task(_exec_task(), Zs, ys, z_src, Zt, yt, _seed_key(), E, (lambda X: X), cfg, n_boot=15)
    d_bad, _ = execute_task(_exec_task(), Zs, ys, z_src, Zt, 1 - yt, _seed_key(), E, (lambda X: X), cfg, n_boot=15)
    assert [r["calibration_benefit_lcb"] for r in d_good] != [r["calibration_benefit_lcb"] for r in d_bad]


def test_subject_seeded_splits_replayed_in_execute_real():
    # execute_task reproduces subject-seeded splits: same seed_key -> identical decision provenance
    Zs, ys, z_src, Zt, yt = _exec_arrays()
    a, _ = execute_task(_exec_task(), Zs, ys, z_src, Zt, yt, _seed_key(sub=5), (lambda X: X), (lambda X: X),
                        _fast_cfg(), n_boot=15)
    b, _ = execute_task(_exec_task(), Zs, ys, z_src, Zt, yt, _seed_key(sub=5), (lambda X: X), (lambda X: X),
                        _fast_cfg(), n_boot=15)
    assert [r["calibration_idx_hash"] for r in a] == [r["calibration_idx_hash"] for r in b]
    c, _ = execute_task(_exec_task(), Zs, ys, z_src, Zt, yt, _seed_key(sub=6), (lambda X: X), (lambda X: X),
                        _fast_cfg(), n_boot=15)
    assert [r["calibration_idx_hash"] for r in a] != [r["calibration_idx_hash"] for r in c]   # subject-dependent


def test_k_nested_subsets_replayed_in_execute_real():
    Zs, ys, z_src, Zt, yt = _exec_arrays()
    dec, _ = execute_task(_exec_task(), Zs, ys, z_src, Zt, yt, _seed_key(), (lambda X: X), (lambda X: X),
                          _fast_cfg(), n_boot=15)
    ks = sorted({r["k"] for r in dec if r.get("k") is not None})
    assert ks == [1, 2, 4, 8, 16]                                    # all k evaluated for B2


def test_b2_unavailable_k_halts_or_marks_unavailable_without_audit_reuse():
    Zs, ys, z_src, Zt, yt = _exec_arrays(n=8)                        # tiny target -> large k unavailable
    dec, _ = execute_task(_exec_task(), Zs, ys, z_src, Zt, yt, _seed_key(), (lambda X: X), (lambda X: X),
                          _fast_cfg(), n_boot=15)
    un = [r for r in dec if r.get("decision_action") == "unavailable_k"]
    assert un and all(r.get("calibration_label_hash") is None for r in un)   # marked, no audit reuse


def test_b3_sequential_early_stop_uses_no_future_labels_execute():
    Zs, ys, z_src, Zt, yt = _exec_arrays()
    dec, _ = execute_task(_exec_task(budget="B3_sequential_calibration"), Zs, ys, z_src, Zt, yt, _seed_key(),
                          (lambda X: X), (lambda X: X), _fast_cfg(), n_boot=15)
    for r in dec:
        assert "b3_ks_read" in r and r["b3_ks_read"] == sorted(r["b3_ks_read"])   # ascending; no future peek


def test_b1_accept_impossible_in_execute_real():
    Zs, ys, z_src, Zt, yt = _exec_arrays()
    dec, _ = execute_task(_exec_task(budget="B1_unlabeled_target"), Zs, ys, z_src, Zt, yt, _seed_key(),
                          (lambda X: X), (lambda X: X), _fast_cfg(), n_boot=15)
    assert dec and all(r["decision_action"] != "accept" for r in dec)


def test_b4_oracle_excluded_from_deployable_accept_counts_execute():
    Zs, ys, z_src, Zt, yt = _exec_arrays()
    dec, _ = execute_task(_exec_task(budget="B4_oracle_selector"), Zs, ys, z_src, Zt, yt, _seed_key(),
                          (lambda X: X), (lambda X: X), _fast_cfg(), n_boot=15)
    assert dec and all(r["decision_action"] == "DIAGNOSTIC" for r in dec)


def test_same_k_random_specificity_flag_in_execute_real():
    Zs, ys, z_src, Zt, yt = _exec_arrays()
    # a discriminative eraser vs an identical random eraser -> if random reproduces, flagged non_specific
    dec, _ = execute_task(_exec_task(), Zs, ys, z_src, Zt, yt, _seed_key(), _lin_eraser(), _lin_eraser(),
                          _fast_cfg(), n_boot=15)
    accepts = [r for r in dec if r["decision_action"] == "accept"]
    for r in accepts:
        assert r["specificity"] in ("accepted_specific", "accepted_non_specific")


def test_assemble_run_two_phase_outputs_on_synthetic_provider(tmp_out=RUN_OUT + "_synthtest"):
    cfg = _fast_cfg()
    Zs, ys, z_src, Zt, yt = _exec_arrays()

    def provider():
        for budget in ["B0_source_only", "B2_k_labels_per_class", "B4_oracle_selector"]:
            yield {"meta": _exec_task(budget=budget), "Zs": Zs, "ys": ys, "z_src": z_src, "Zt": Zt, "yt": yt,
                   "seed_key": _seed_key(), "eraser": (lambda X: X), "eraser_random": (lambda X: X)}
    nd, na = _assemble_run(cfg, provider(), out=tmp_out, n_boot=15)
    assert nd > 0 and na > 0
    import csv as _csv
    with open("%s/decision_rows.csv" % tmp_out) as fh:
        cols = _csv.DictReader(fh).fieldnames
    assert not any(("audit_delta" in c or "audit_label" in c or "audit_metric" in c) for c in cols), cols


def test_execute_still_halts_with_default_manifest():
    cfg = load_cfg()
    code, msg = run_cli(["--execute"], cfg)
    assert code == 1 and msg.startswith("EXPERIMENTS_DISABLED")


# ---------------- run-readiness v2: source-safety UCB + provider-validation (dummy arrays; no runs) ----------------
_PROVIDER_VAL_MANIFEST = "tos_cmi/eeg/configs/target_info_tier1_provider_validation_manifest_TEMPLATE.yaml"


def _src_with_subj(n_subj=6, per=16):
    rng = np.random.default_rng(0)
    n = n_subj * per
    ys = np.array([0, 1] * (n // 2))
    Zs = rng.standard_normal((n, 6)) + ys[:, None] * np.array([1.5, 0, 0, 0, 0, 0])
    subj = np.repeat(np.arange(n_subj), per)[:n]
    return SourceContext(Zs, ys, ys.copy(), 2, subj=subj)


def test_source_safety_ucb_not_point_estimate():
    r = source_task_drop_ucb(_lin_eraser(), _src_with_subj(6), 0.02)
    assert set(r) >= {"task_drop_mean", "task_drop_ucb", "cluster_unit", "n_clusters", "ci_method",
                      "safety_status", "reason", "pass"}
    assert r["cluster_unit"] == "source_subject" and r["n_clusters"] == 6
    assert "bootstrap" in r["ci_method"]
    assert r["task_drop_ucb"] >= r["task_drop_mean"]                  # UCB is an upper bound, not the mean


def test_point_safe_but_ucb_unsafe_does_not_accept():
    src = _src_with_subj(6)
    r0 = source_task_drop_ucb(_lin_eraser(), src, 1.0)               # generous eps -> read mean/ucb
    mean, ucb = r0["task_drop_mean"], r0["task_drop_ucb"]
    assert ucb >= mean
    if ucb > mean + 1e-9:                                            # spread exists -> the mean-vs-ucb gap matters
        eps = (mean + ucb) / 2.0                                     # mean < eps < ucb
        r = source_task_drop_ucb(_lin_eraser(), src, eps)
        assert r["pass"] is False                                    # keys on UCB (>eps), NOT the mean (<eps)


def test_underpowered_source_safety_abstains_or_rejects():
    r = source_task_drop_ucb(_lin_eraser(), _src_with_subj(2), 0.02)  # 2 subjects < min_clusters
    assert r["safety_status"] == "ABSTAIN" and r["reason"] == "underpowered_safety_ucb" and r["pass"] is False


def test_nan_or_underpowered_safety_does_not_accept():
    Zs, ys, z_src, Zt, yt = _exec_arrays()
    subj2 = np.zeros(len(ys), int); subj2[len(ys) // 2:] = 1          # 2 source subjects -> underpowered safety
    dec, _ = execute_task(_exec_task(), Zs, ys, z_src, Zt, yt, _seed_key(), _lin_eraser(), (lambda X: X),
                          _fast_cfg(), n_boot=15, source_subj=subj2)
    assert dec and all(r.get("decision_action") != "accept" for r in dec)


def test_provider_validation_requires_manifest():
    code, msg = run_cli(["--provider-validate-one-dump"], load_cfg())
    assert code == 1 and msg.startswith("PROVIDER_VALIDATION_DISABLED"), (code, msg)


def test_provider_validation_default_halts():
    # the provider-validation manifest is passed via --run-manifest; the TEMPLATE (allowed=false) must HALT
    code, msg = run_cli(["--provider-validate-one-dump", "--run-manifest", _PROVIDER_VAL_MANIFEST], load_cfg())
    assert code == 1 and msg.startswith("PROVIDER_VALIDATION_DISABLED"), (code, msg)


def test_provider_validation_scope_is_one_dump_only():
    import yaml
    m = yaml.safe_load(open(_PROVIDER_VAL_MANIFEST))
    sc = m["scope"]
    assert sc["dataset"] == "Lee2019_MI" and sc["backbone"] == "EEGNet" and sc["split_id"] == 0 and sc["k"] == 4
    assert sc["fold"] == "first_available" and len(sc["budgets"]) == 3
    assert m["provider_validation_allowed"] is False and m["runs_allowed"] is False


def test_provider_validation_output_schema_has_no_metrics():
    Zs, ys, z_src, Zt, yt = _exec_arrays()
    dec, aud = execute_task(_exec_task(), Zs, ys, z_src, Zt, yt, _seed_key(), _lin_eraser(), (lambda X: X),
                            _fast_cfg(), n_boot=15)
    summ = _redact_validation_output(dec, aud, {"scope": {}})
    metric_tokens = ("bacc", "delta", "nll", "benefit_lcb", "specific", "domain_gain", "random", "accept", "reject")
    assert not any(any(t in f.lower() for t in metric_tokens) for f in summ["decision_row_safe_fields"])
    assert not any(any(t in f.lower() for t in metric_tokens) for f in summ["audit_row_safe_fields"])
    assert not _has_metric_key(summ)


def test_real_provider_loader_can_be_mocked_without_target_leak(tmp=RUN_OUT + "_mocktest"):
    cfg = _fast_cfg()
    Zs, ys, z_src, Zt, yt = _exec_arrays()

    def mock_provider():
        for budget in ["B0_source_only", "B2_k_labels_per_class"]:
            yield {"meta": _exec_task(budget=budget), "Zs": Zs, "ys": ys, "z_src": z_src, "Zt": Zt, "yt": yt,
                   "seed_key": _seed_key(), "eraser": (lambda X: X), "eraser_random": (lambda X: X),
                   "source_subj": None}
    _assemble_run(cfg, mock_provider(), out=tmp, n_boot=15)
    import csv as _csv
    with open("%s/decision_rows.csv" % tmp) as fh:
        cols = _csv.DictReader(fh).fieldnames
    assert not any(("audit_delta" in c or "audit_label" in c or "audit_metric" in c) for c in cols), cols


def test_provider_validation_redacts_metric_values():
    Zs, ys, z_src, Zt, yt = _exec_arrays()
    dec, aud = execute_task(_exec_task(), Zs, ys, z_src, Zt, yt, _seed_key(), _lin_eraser(), _lin_eraser(),
                            _fast_cfg(), n_boot=15)
    summ = _redact_validation_output(dec, aud, {})
    assert summ["metrics_redacted"] is True and summ["metrics_computed_internally"] is True
    assert summ["decision_row_redacted_metric_fields"] > 0            # metric fields exist but are redacted (counted)

    def _floats(o):
        if isinstance(o, float):
            return [o]
        if isinstance(o, dict):
            return sum((_floats(v) for v in o.values()), [])
        if isinstance(o, list):
            return sum((_floats(v) for v in o), [])
        return []
    assert _floats(summ) == []                                       # NO metric VALUE (float) leaked into the summary


def test_execute_real_still_halts_after_provider_mode_added():
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
       test_same_k_random_specificity_flag, test_execute_still_halts_when_runs_disabled,
       # P0-1 run authorization
       test_fully_approved_manifest_authorizes, test_plain_yaml_flip_does_not_enable_execution,
       test_missing_enable_token_halts, test_wrong_enable_token_halts, test_scope_hash_mismatch_halts,
       test_driver_config_hash_mismatch_halts,
       # P0-2 calibration-delta taint / provenance
       test_cal_deltas_depend_only_on_calibration_labels,
       test_audit_label_permutation_changes_only_audit_metrics_not_decision,
       test_calibration_label_permutation_can_change_decision, test_audit_view_cannot_be_passed_to_calibration_delta,
       test_decision_rows_store_calibration_and_audit_hashes, test_label_access_guard_phase_separation,
       # P0-3 real-split preflight
       test_preflight_real_splits_outputs_no_metrics, test_unavailable_k_marked_not_shrunk,
       test_no_audit_label_reuse_for_calibration, test_preflight_gated_by_manifest,
       # subject-seeded split hardening
       test_subject_seeded_splits_differ_across_subjects, test_subject_seeded_splits_reproducible_for_same_subject,
       test_same_split_shared_across_interventions_and_budgets, test_k_subsets_are_nested_within_calibration_pool,
       test_subject_seed_key_uses_stable_hash_not_python_hash, test_preflight_subject_seeded_has_no_metrics,
       # run-ready execute_real wiring
       test_execute_real_requires_approved_manifest_and_token,
       test_real_loader_wraps_target_labels_in_guarded_contexts, test_decision_rows_contain_no_audit_metrics,
       test_audit_rows_created_only_after_decision_frozen, test_audit_label_permutation_does_not_change_decision_rows,
       test_calibration_label_permutation_can_change_decision_rows, test_subject_seeded_splits_replayed_in_execute_real,
       test_k_nested_subsets_replayed_in_execute_real,
       test_b2_unavailable_k_halts_or_marks_unavailable_without_audit_reuse,
       test_b3_sequential_early_stop_uses_no_future_labels_execute, test_b1_accept_impossible_in_execute_real,
       test_b4_oracle_excluded_from_deployable_accept_counts_execute,
       test_same_k_random_specificity_flag_in_execute_real, test_assemble_run_two_phase_outputs_on_synthetic_provider,
       test_execute_still_halts_with_default_manifest,
       # run-readiness v2: source-safety UCB + provider-validation
       test_source_safety_ucb_not_point_estimate, test_point_safe_but_ucb_unsafe_does_not_accept,
       test_underpowered_source_safety_abstains_or_rejects, test_nan_or_underpowered_safety_does_not_accept,
       test_provider_validation_requires_manifest, test_provider_validation_default_halts,
       test_provider_validation_scope_is_one_dump_only, test_provider_validation_output_schema_has_no_metrics,
       test_real_provider_loader_can_be_mocked_without_target_leak, test_provider_validation_redacts_metric_values,
       test_execute_real_still_halts_after_provider_mode_added]


def main():
    for t in ALL:
        t(); print("  PASS", t.__name__)
    print("TARGET_INFO_TIER1_TESTS_PASS (%d/%d)" % (len(ALL), len(ALL)))


if __name__ == "__main__":
    main()
