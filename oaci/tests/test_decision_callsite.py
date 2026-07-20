"""C8a: the native K1/K2 call-site wiring in the runner (audit feature retention, finalize decision, hash
binding, degenerate handling). Runs the REAL small fold from test_runner_finalize through finalize with
decisions enabled. Standalone + pytest-compatible."""
from __future__ import annotations

from oaci.decision.plans import K1Spec, K2Spec
from oaci.runner.decision import K1_DEGENERATE, DecisionContext, compute_level_decision
from oaci.runner.finalize import finalize_level_run
from oaci.tests.test_runner_finalize import _CPU, _audit
from oaci.tests.test_runner_train_select import _exec_cfg, _factory, _MSPEC

_K1_STATUSES = {"leakage_reduction_detected", "stop_no_detectable_heldout_leakage_reduction",
                "estimable_degenerate_same_checkpoint", "skipped_audit_nonestimable_or_oaci_inactive"}


def _specs(n_perm=15):
    k1 = K1Spec("grouped_max_probe_extractable_LQ_ov_OACI_minus_ERM", "source_audit",
                "paired_swap_within_y_recording_group", n_perm, 0.05, "stop_if_within_null_band", 707)
    k2 = K2Spec(("worst_domain_bacc", "worst_domain_nll"), 3, "both_levels",
                {"worst_domain_bacc": 0.0, "worst_domain_nll": 0.0}, "stop_if_no_reproducible_gain")
    return k1, k2


def _finalize(ctx_pack, decision_ctx):
    ai, (fd, maps, ss, lp, fs, plans, rk) = ctx_pack
    return finalize_level_run(ai, fd, fs, ss, lp, plans, _exec_cfg(), _MSPEC, _factory, _CPU,
                              decision_ctx=decision_ctx)


def test_audit_retains_source_audit_features():
    ai, _ctx = _audit()
    af = ai.audit_features
    assert "ERM" in af and "OACI" in af and af["ERM"].features.n > 0          # source-audit features present
    # same selected model hash is feature-extracted once (deduped object identity)
    by_hash = {}
    for _n, mh, feat in ai.audit_feature_items:
        by_hash.setdefault(mh, feat)
        assert by_hash[mh] is feat


def test_k1_callsite_runs_after_lock_and_produces_decision():
    lr = _finalize(_audit(), DecisionContext(enabled=True, k1_spec=_specs()[0], k2_spec=_specs()[1]))
    d = lr.decision
    assert d is not None and d["k1_body"]["k1_status"] in _K1_STATUSES
    assert d["k1_body"]["split_role"] == "source_audit"


def test_k2_single_seed_abstains_in_runner_artifact():
    lr = _finalize(_audit(), DecisionContext(enabled=True, k1_spec=_specs()[0], k2_spec=_specs()[1]))
    k2 = lr.decision["k2_body"]
    assert k2["k2_status"] == "abstain_insufficient_seeds"
    assert k2["available_seeds"] == 1 and k2["required_min_seeds"] == 3


def test_k1_uses_source_audit_population_not_target():
    ai, ctx = _audit(); fd, maps, ss, lp, fs, plans, rk = ctx
    lr = _finalize((ai, ctx), DecisionContext(enabled=True, k1_spec=_specs()[0], k2_spec=_specs()[1]))
    k1b = lr.decision["k1_body"]
    if "audit_population_hash" in k1b:                                        # (absent iff skipped/degenerate)
        assert k1b["audit_population_hash"] == fs.source_audit.design.population_hash


def test_decision_wiring_does_not_change_training_or_selection_hashes():
    lr_off = _finalize(_audit(), None)                                       # decisions disabled
    lr_on = _finalize(_audit(), DecisionContext(enabled=True, k1_spec=_specs()[0], k2_spec=_specs()[1]))
    # training / selection / audit / prediction / metrics identity is IDENTICAL...
    assert [m.method_result_hash for _, m in lr_off.method_items] == [m.method_result_hash for _, m in lr_on.method_items]
    assert lr_off.audit_cache_stats.stats_hash == lr_on.audit_cache_stats.stats_hash
    assert lr_off.erm_stage.stage1_invocation_id == lr_on.erm_stage.stage1_invocation_id
    # ...but the LEVEL hash changes ONLY by the decision binding, and off leaves no decision.
    assert lr_off.decision is None and lr_on.decision is not None
    assert lr_off.level_result_hash != lr_on.level_result_hash


def test_disabled_decisions_leave_level_hash_byte_identical():
    # two independent disabled runs -> identical level hash (the C8a additions are inert when disabled)
    assert _finalize(_audit(), None).level_result_hash == _finalize(_audit(), None).level_result_hash
    assert _finalize(_audit(), DecisionContext(enabled=False)).level_result_hash == _finalize(_audit(), None).level_result_hash


def test_k1_same_checkpoint_degenerate_case_is_handled():
    from oaci.tests.test_decision_k1 import FAST, _paired, _plan
    fe, fo, sg = _paired()
    k1, k2 = _specs()
    out = compute_level_decision(0, feat_by_method={"ERM": fe, "OACI": fo}, audit_support_graph=sg,
                                 audit_fold_plan=_plan(fe, sg), cfg=FAST, k1_spec=k1, k2_spec=k2, k2_units=[],
                                 model_hash_by_method={"ERM": "H", "OACI": "H"})      # OACI == ERM checkpoint
    assert out["k1_body"]["k1_status"] == K1_DEGENERATE and out["k1_body"]["observed_delta"] == 0.0
    assert out["k1_body"]["same_checkpoint"] is True and out["k1_null_arrays"]["null"].shape[0] == 0


def test_decision_artifact_writes_and_deep_verifies_end_to_end():
    """A REAL decision-bearing fold: level_result_hash binds the decision, the writer reconstructs the same
    payload (writer re-hash assert), the indexed decision files deep-verify, and read-back is consistent —
    while training/selection hashes match the no-decision run."""
    import tempfile

    from oaci.artifacts.decision_codec import read_level_decisions, verify_decisions
    from oaci.artifacts.verify import verify_artifact_tree
    from oaci.artifacts.writer import write_artifact_tree_atomic
    from oaci.runner import assemble_fold_run, run_level_complete
    from oaci.tests.test_runner_artifacts import _context
    from oaci.tests.test_runner_finalize import _complete

    lr0, (fd, maps, ss, lp, fs, plans, rk) = _complete()                     # no-decision baseline + inputs
    lr = run_level_complete(rk, fd, ss, lp, fs, plans, _exec_cfg(), _MSPEC, _factory, _CPU,
                            decision_ctx=DecisionContext(enabled=True, k1_spec=_specs()[0], k2_spec=_specs()[1]))
    assert lr.decision is not None and lr.level_result_hash != lr0.level_result_hash
    assert [m.method_result_hash for _, m in lr0.method_items] == [m.method_result_hash for _, m in lr.method_items]
    fr = assemble_fold_run(fs, {int(rk.deletion_level): lr})
    res = write_artifact_tree_atomic(fr, _context(lr), tempfile.mkdtemp(prefix="oaci-decend-"))
    assert verify_artifact_tree(res.artifact_dir, deep=True).ok                # decisions indexed + level hash recomputes
    assert verify_decisions(res.artifact_dir, require=True)["with_decisions"] == [int(rk.deletion_level)]
    d = read_level_decisions(res.artifact_dir, int(rk.deletion_level))
    assert d["k1"]["split_role"] == "source_audit" and d["k1"]["k1_status"] in _K1_STATUSES
    assert d["k2"]["k2_status"] == "abstain_insufficient_seeds" and d["k2"]["available_seeds"] == 1


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} decision-callsite tests")


if __name__ == "__main__":
    _run_all()
