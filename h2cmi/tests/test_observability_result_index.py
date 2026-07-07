"""Project A — tests for the audited result index + validator (Step 8).

Builds fake audited run directories (no training) and checks that the validator accepts a clean
oracle-only report, rejects an identifiable R0 target metric and a forbidden-claim violation, and
that the summary digest carries the claim-boundary flags. Run:

    python -m h2cmi.tests.test_observability_result_index
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from h2cmi.observability import ContractID as C, build_audited_eval_report, report_to_dict
from h2cmi.observability.result_index import build_summary, load_run
from h2cmi.observability.validate_results import validate_all, validate_run

_STRICT = {"balanced_acc": 0.30, "worst_domain_bacc": 0.21}
_OFFLINE = {"delta_adapt": {"d_balanced_acc": -0.01}, "per_domain_pi_T": {"1": [0.25] * 4}}
_ONLINE = {"balanced_acc": 0.28}
_LEAK = {"site": {"I_hat": 0.1}, "subject": {"I_hat": 0.3}}


def _write_run(root, target, seed, report_data, *, status="ok", extra_manifest=None):
    d = Path(root) / f"dataset=BNCI2014_001_target={target}_seed={seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "observability_report.json").write_text(json.dumps(report_data))
    (d / "raw_results.json").write_text(json.dumps(
        {"strict_dg": _STRICT, "offline_tta": _OFFLINE, "online_tta": _ONLINE, "leakage": _LEAK,
         "n_source_trials": 1152, "n_target_trials": 576}))
    manifest = {"status": status, "dataset": "BNCI2014_001", "target_subject": target,
                "seed": seed, "align_factor": "subject", "n_source_trials": 1152,
                "n_target_trials": 576, "alignment_factor_degenerate": False, "n_classes": 4}
    manifest.update(extra_manifest or {})
    (d / "run_manifest.json").write_text(json.dumps(manifest))
    return d


def _clean_report():
    report = build_audited_eval_report("t", strict_dg=_STRICT, offline_tta=_OFFLINE,
                                       online_tta=_ONLINE, leakage=_LEAK,
                                       prior_contracts=None, prior_conclusion=False)
    return report_to_dict(report)


def test_result_validator_accepts_clean_oracle_metrics():
    with tempfile.TemporaryDirectory() as root:
        d = _write_run(root, 1, 0, _clean_report())
        ok, issues = validate_run(d)
        assert ok, f"clean audited run must validate: {issues}"


def test_result_validator_rejects_identifiable_r0_target_metric():
    data = _clean_report()
    # corrupt: mark an R0 target bAcc as identifiable (the overclaim the validator must catch)
    for c in data["claims"]:
        if c["name"] == "strict_dg.target_bacc":
            c["identifiable_estimand"] = "balanced_accuracy"
    with tempfile.TemporaryDirectory() as root:
        d = _write_run(root, 1, 0, data)
        ok, issues = validate_run(d)
        assert not ok and any("identifiable" in i for i in issues)


def test_result_validator_rejects_forbidden_claim_violation():
    data = _clean_report()
    data["forbidden_claims_violated"] = ["offline_tta.target_prior"]
    with tempfile.TemporaryDirectory() as root:
        d = _write_run(root, 1, 0, data)
        ok, issues = validate_run(d)
        assert not ok and any("forbidden_claims_violated" in i for i in issues)


def test_result_index_summary_contains_claim_boundary_flags():
    with tempfile.TemporaryDirectory() as root:
        _write_run(root, 1, 0, _clean_report())
        _write_run(root, 2, 0, _clean_report())
        # a skipped run must not break the digest
        skip_dir = Path(root) / "dataset=BNCI2014_001_target=3_seed=0"
        skip_dir.mkdir(parents=True, exist_ok=True)
        (skip_dir / "run_manifest.json").write_text(json.dumps(
            {"status": "skipped", "skip_reason": "no cache", "dataset": "BNCI2014_001",
             "target_subject": 3, "seed": 0}))
        summary = build_summary(root)
        agg = summary["aggregate"]
        assert agg["n_runs"] == 3 and agg["n_ok"] == 2 and agg["n_skipped"] == 1
        assert agg["all_forbidden_violations_empty"] is True
        assert agg["all_target_metrics_oracle_only"] is True
        assert agg["all_target_metrics_identifiable_null"] is True
        assert "claim_boundary" in summary
        # per-run boundary flags present
        r = load_run(Path(root) / "dataset=BNCI2014_001_target=1_seed=0")
        assert r["all_target_metrics_identifiable_null"] is True
        assert r["target_prior_claim_status"] == "rejected_conclusion_false"
        # the whole grid validates clean
        assert all(v["valid"] for v in validate_all(root).values())


def test_validator_rejects_relabelled_estimand():
    # the blocker: relabelling estimand to escape the target-metric block must be caught
    data = _clean_report()
    for c in data["claims"]:
        if c["name"] == "strict_dg.target_bacc":
            c["estimand"] = "balanced_accuracy_v2"
            c["identifiable_estimand"] = "balanced_accuracy_v2"
    with tempfile.TemporaryDirectory() as root:
        d = _write_run(root, 1, 0, data)
        ok, issues = validate_run(d)
        assert not ok and any("unknown" in i for i in issues)


def test_validator_rejects_bogus_status():
    with tempfile.TemporaryDirectory() as root:
        d = _write_run(root, 1, 0, _clean_report(), status="error")
        ok, issues = validate_run(d)
        assert not ok and any("status" in i for i in issues)
        # a legal skip WITH a reason is fine
        d2 = _write_run(root, 2, 0, _clean_report(), status="skipped",
                        extra_manifest={"skip_reason": "no cache"})
        assert validate_run(d2)[0] is True
        # ...but a skip WITHOUT a reason is not
        d3 = _write_run(root, 3, 0, _clean_report(), status="skipped")
        assert validate_run(d3)[0] is False


def test_validator_rejects_prefix_collision_dir():
    # manifest target=1 but dir token target=11 -> exact-token mismatch must be caught
    with tempfile.TemporaryDirectory() as root:
        d = Path(root) / "dataset=BNCI2014_001_target=11_seed=0"
        d.mkdir(parents=True)
        (d / "observability_report.json").write_text(json.dumps(_clean_report()))
        (d / "run_manifest.json").write_text(json.dumps(
            {"status": "ok", "dataset": "BNCI2014_001", "target_subject": 1, "seed": 0}))
        ok, issues = validate_run(d)
        assert not ok and any("dir token" in i for i in issues)


def test_validator_rejects_rejected_but_identifiable():
    data = _clean_report()
    for c in data["claims"]:
        if c["name"] == "offline_tta.target_prior":     # a rejected claim
            c["identifiable_estimand"] = "target_prior"
    with tempfile.TemporaryDirectory() as root:
        d = _write_run(root, 1, 0, data)
        ok, issues = validate_run(d)
        assert not ok and any("rejected claim marked identifiable" in i for i in issues)


def test_validator_rejects_rejected_prior_conclusion_true():
    data = _clean_report()
    for c in data["claims"]:
        if c["name"] == "offline_tta.target_prior":
            c["conclusion"] = True                      # rejected prior asserted as a conclusion
    with tempfile.TemporaryDirectory() as root:
        d = _write_run(root, 1, 0, data)
        ok, issues = validate_run(d)
        assert not ok and any("conclusion" in i for i in issues)


def test_aggregate_reflects_prior_and_unknown_estimand():
    good = _clean_report()
    bad = _clean_report()
    for c in bad["claims"]:
        if c["name"] == "strict_dg.target_bacc":
            c["estimand"] = "balanced_accuracy_v2"      # unknown estimand
        if c["name"] == "offline_tta.target_prior":
            c["conclusion"] = True                      # non-compliant prior
    with tempfile.TemporaryDirectory() as root:
        _write_run(root, 1, 0, good)
        _write_run(root, 2, 0, bad)
        agg = build_summary(root)["aggregate"]
        assert agg["no_unknown_estimands"] is False
        assert agg["all_prior_claims_compliant"] is False


def test_digest_statistical_layer():
    with tempfile.TemporaryDirectory() as root:
        _write_run(root, 1, 0, _clean_report())
        _write_run(root, 1, 1, _clean_report())
        _write_run(root, 2, 0, _clean_report())
        agg = build_summary(root)["aggregate"]
        # per-target / per-seed descriptive blocks
        assert agg["per_target"]["1"]["n_ok"] == 2 and "2" in agg["per_target"]
        for k in ("n", "mean", "std", "min", "max"):
            assert k in agg["per_target"]["1"]["strict_dg_bacc"]
        assert "0" in agg["per_seed"]
        # overall harm-rate + gain sign counts (offline gain = -0.01 in every clean run)
        o = agg["overall"]
        assert o["offline_tta_harm_rate"] == 1.0
        assert (o["n_offline_tta_gain_negative"] + o["n_offline_tta_gain_positive"]
                + o["n_offline_tta_gain_zero"]) == 3


def test_digest_missing_cells_gate():
    from h2cmi.observability.validate_results import main as vmain
    with tempfile.TemporaryDirectory() as root:
        _write_run(root, 1, 0, _clean_report())     # only 1 of an expected 2x2 grid
        agg = build_summary(root, expected_targets=[1, 2], expected_seeds=[0, 1])["aggregate"]
        assert agg["expected_runs"] == 4
        missing = {(m["target"], m["seed"]) for m in agg["missing_cells"]}
        assert ("1", "0") not in missing and ("2", "1") in missing and len(missing) == 3
        # an incomplete grid fails validation (rc=1) unless --allow-missing
        assert vmain(["--root", root, "--expected-targets", "1", "2",
                      "--expected-seeds", "0", "1"]) == 1
        assert vmain(["--root", root, "--expected-targets", "1", "2",
                      "--expected-seeds", "0", "1", "--allow-missing"]) == 0


def test_digest_write_paths_produce_lf_files():
    # exercises write_summary_md + validate_results file writes (py3.9-safe LF, valid JSON)
    from h2cmi.observability.validate_results import main as vmain
    with tempfile.TemporaryDirectory() as root:
        _write_run(root, 1, 0, _clean_report())
        _write_run(root, 1, 1, _clean_report())
        oj, om = Path(root) / "summary.json", Path(root) / "summary.md"
        rc = vmain(["--root", root, "--expected-targets", "1", "--expected-seeds", "0", "1",
                    "--out-json", str(oj), "--out-md", str(om)])
        assert rc == 0
        jb, mb = oj.read_bytes(), om.read_bytes()
        assert b"\r" not in jb and b"\r" not in mb          # LF only
        data = json.loads(jb)                                # valid JSON
        assert data["validation"]["all_valid"] is True and "per_target" in data["aggregate"]
        assert mb.count(b"\n") >= 5                           # MD is multi-line (not a giant line)


def test_chance_normalized_metrics_for_four_class():
    with tempfile.TemporaryDirectory() as root:
        d = _write_run(root, 1, 0, _clean_report())          # n_classes=4 default
        r = load_run(d)
        assert r["n_classes"] == 4 and r["chance_bacc"] == 0.25
        assert r["strict_dg_bacc_excess"] == round(0.30 - 0.25, 4)          # _STRICT bAcc 0.30
        assert r["strict_dg_bacc_excess_norm"] == round((0.30 - 0.25) / 0.75, 4)
        assert r["offline_tta_gain_bacc_norm"] == round(-0.01 / 0.75, 4)    # _OFFLINE gain -0.01
        agg = build_summary(root)["aggregate"]
        assert agg["n_classes"] == 4 and agg["chance_bacc"] == 0.25
        assert agg["mean_strict_dg_bacc_excess_norm"] == round((0.30 - 0.25) / 0.75, 4)


def test_chance_normalized_metrics_for_binary():
    # SAME raw bAcc 0.30 means BELOW chance for a binary task -> excess/excess_norm go negative
    with tempfile.TemporaryDirectory() as root:
        d = _write_run(root, 1, 0, _clean_report(), extra_manifest={"n_classes": 2})
        r = load_run(d)
        assert r["n_classes"] == 2 and r["chance_bacc"] == 0.5
        assert r["strict_dg_bacc_excess"] == round(0.30 - 0.5, 4)
        assert r["strict_dg_bacc_excess_norm"] == round((0.30 - 0.5) / 0.5, 4)
        assert r["offline_tta_gain_bacc_norm"] == round(-0.01 / 0.5, 4)
        agg = build_summary(root)["aggregate"]
        assert agg["n_classes"] == 2 and agg["chance_bacc"] == 0.5


def test_missing_n_classes_fails_validation_for_ok_run():
    with tempfile.TemporaryDirectory() as root:
        d = _write_run(root, 1, 0, _clean_report())
        mf = d / "run_manifest.json"                          # strip n_classes from an ok manifest
        m = json.loads(mf.read_text()); m.pop("n_classes", None)
        mf.write_text(json.dumps(m))
        ok, issues = validate_run(d)
        assert not ok and any("n_classes" in i for i in issues)


ALL_TESTS = [
    test_result_validator_accepts_clean_oracle_metrics,
    test_result_validator_rejects_identifiable_r0_target_metric,
    test_result_validator_rejects_forbidden_claim_violation,
    test_result_index_summary_contains_claim_boundary_flags,
    test_validator_rejects_relabelled_estimand,
    test_validator_rejects_bogus_status,
    test_validator_rejects_prefix_collision_dir,
    test_validator_rejects_rejected_but_identifiable,
    test_validator_rejects_rejected_prior_conclusion_true,
    test_aggregate_reflects_prior_and_unknown_estimand,
    test_digest_statistical_layer,
    test_digest_missing_cells_gate,
    test_digest_write_paths_produce_lf_files,
    test_chance_normalized_metrics_for_four_class,
    test_chance_normalized_metrics_for_binary,
    test_missing_n_classes_fails_validation_for_ok_run,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} RESULT-INDEX TESTS PASSED")


if __name__ == "__main__":
    run()
