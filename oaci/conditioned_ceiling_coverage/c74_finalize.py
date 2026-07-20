"""Finalize C74 reports only after the independent red-team has passed."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import statistics

from . import c74_cache as cache


REPORT_DIR = Path("oaci/reports")
TABLE_DIR = REPORT_DIR / "c74_tables"
MAIN_JSON = REPORT_DIR / "C74_T2_SOURCE_WZ_INSTRUMENTATION.json"
MAIN_REPORT = REPORT_DIR / "C74_T2_SOURCE_WZ_INSTRUMENTATION.md"
C75_DRAFT = REPORT_DIR / "C75_T2_REPRESENTATION_CONSTRUCT_ANALYSIS_PROTOCOL_DRAFT.json"
C76_DRAFT = REPORT_DIR / "C76_T3_HO_NEW_VARIABLE_HOLDOUT_PROTOCOL_DRAFT.json"
EXTERNAL_LOG_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c74-t2-source-wz/logs")


def _read_csv(name: str) -> list[dict]:
    with open(TABLE_DIR / name, newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict, key: str) -> float:
    return float(row[key])


def _mean(rows: list[dict], key: str) -> float:
    return statistics.mean(_float(row, key) for row in rows)


def _red_team_gate() -> dict:
    checks = _read_csv("red_team_checks.csv")
    if len(checks) != 33 or not all(row["passed"] == "1" for row in checks):
        raise RuntimeError("C74 finalization blocked: independent red-team is not 33/33 PASS")
    report = REPORT_DIR / "C74_RED_TEAM_VERIFICATION.md"
    if "Final status: `PASS`" not in report.read_text():
        raise RuntimeError("C74 finalization blocked: red-team report does not say PASS")
    return {"status": "PASS", "checks": len(checks), "report_sha256": cache.sha256_file(report)}


def _draft_protocols(protocol: dict, state: dict, view_rows: list[dict], power_rows: list[dict]) -> None:
    view_hashes = {row["view_name"]: row["sha256"] for row in view_rows}
    c75 = {
        "schema_version": "c75_t2_representation_construct_analysis_protocol_draft_v1",
        "status": "DRAFT_NOT_OPEN_NOT_AUTHORIZED",
        "parent_C74_protocol_sha256": cache.sha256_file(cache.PROTOCOL_PATH),
        "input_role": "fully_instrumented_T2_discovery_and_calibration_only",
        "input_units": 216,
        "locked_view_manifest_sha256s": view_hashes,
        "primary_hypotheses_to_lock_before_open": [
            "candidate_specific_class_projection_summaries_are_split_stable",
            "candidate_specific_Wz_residual_changes_heldout_rank_under_registered_counterfactuals",
            "source_and_target_unlabeled_zWz_summaries_do_not_gain_incrementally_under_the_C74_fixed_family",
        ],
        "required_inference": [
            "target_cluster_bootstrap", "trajectory_blocked_nulls", "cross_fit_only",
            "separate_source_target_unlabeled_and_label_derived_information_classes",
        ],
        "forbidden": [
            "new_forward", "re_inference", "training", "GPU", "T3_HO_access",
            "same_label_oracle_primary_analysis", "selector", "checkpoint_recommendation",
            "representation_mechanism_claim", "target_population_claim", "manuscript_drafting",
        ],
        "C74_smoke_is_not_C75_confirmation": True,
    }
    c76_plan = next(row for row in power_rows if row["campaign"] == "C76_T3_HO_projected")
    c76 = {
        "schema_version": "c76_t3_ho_new_variable_holdout_protocol_draft_v1",
        "status": "DRAFT_READY_BUT_NOT_AUTHORIZED",
        "parent_C74_protocol_sha256": cache.sha256_file(cache.PROTOCOL_PATH),
        "required_parent": "completed_C75_hypothesis_lock_commit",
        "holdout_role": "prospective_new_variable_holdout_not_independent_target_dataset_confirmation",
        "holdout_units": 1052,
        "holdout_unit_manifest_sha256": cache.sha256_file("oaci/reports/c74_tables/t3_ho_holdout_unit_manifest.csv"),
        "separate_future_exact_CLI_authorization_required": True,
        "C74_authorization_does_not_authorize_C76": True,
        "projected_external_size_bytes": int(c76_plan["external_size_bytes"]),
        "projected_external_size_GiB": float(c76_plan["external_size_GiB"]),
        "execution": "frozen_checkpoint_CPU_reinference_only_after_separate_authorization",
        "forbidden": [
            "training", "GPU", "BNCI2014_004", "seeds_3_4", "outcome_adaptive_hypotheses",
            "selector", "checkpoint_recommendation", "target_population_claim", "manuscript_drafting",
        ],
    }
    cache.atomic_json(C75_DRAFT, c75)
    cache.atomic_json(C76_DRAFT, c76)


def _artifact_hygiene() -> tuple[list[dict], list[dict]]:
    paths = sorted(
        [path for path in REPORT_DIR.glob("C74_*") if path.is_file()]
        + [C75_DRAFT, C76_DRAFT]
        + [path for path in TABLE_DIR.glob("*.csv") if path.name not in {"artifact_manifest.csv", "large_artifact_scan.csv"}]
    )
    manifest_rows = []
    scan_rows = []
    for path in paths:
        row_count = ""
        if path.suffix == ".csv":
            with open(path, newline="") as stream:
                row_count = sum(1 for _ in csv.reader(stream)) - 1
        size = path.stat().st_size
        manifest_rows.append({
            "path": str(path), "sha256": cache.sha256_file(path),
            "size_bytes": size, "row_count": row_count,
            "external_raw_payload": 0,
        })
        scan_rows.append({
            "path": str(path), "size_bytes": size,
            "over_50MB": int(size > 50_000_000), "passed": int(size <= 50_000_000),
        })
    _write_csv(TABLE_DIR / "artifact_manifest.csv", manifest_rows)
    _write_csv(TABLE_DIR / "large_artifact_scan.csv", scan_rows)
    return manifest_rows, scan_rows


def _test_manifest() -> list[dict]:
    specs = [
        ("focused_C74", "login", "15 passed", "env PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider oaci/tests/test_c74_t2_source_wz_instrumentation.py -q", "", ""),
        ("C65_C74", "892156", "80 passed", "python -m pytest -p no:cacheprovider oaci/tests/test_c6[5-9]_*.py oaci/tests/test_c7[0-4]_*.py -q", "c74-reg65_892156.out", "c74-reg65_892156.err"),
        ("C23_C74", "892157", "491 passed", "python -m pytest -p no:cacheprovider oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c[3-6]*.py oaci/tests/test_c7[0-4]_*.py -q", "c74-reg23_892157.out", "c74-reg23_892157.err"),
        ("full_oaci", "892158", "1415 passed", "python -m pytest -p no:cacheprovider oaci/tests -q", "c74-fulltest_892158.out", "c74-fulltest_892158.err"),
    ]
    rows = []
    for scope, job_id, expected, command, stdout_name, stderr_name in specs:
        if stdout_name:
            stdout_path = EXTERNAL_LOG_ROOT / stdout_name
            stderr_path = EXTERNAL_LOG_ROOT / stderr_name
            body = stdout_path.read_text()
            passed = expected in body and stderr_path.stat().st_size == 0
            stdout_sha = cache.sha256_file(stdout_path)
            stderr_sha = cache.sha256_file(stderr_path)
            stderr_bytes = stderr_path.stat().st_size
        else:
            stdout_path = Path("")
            stderr_path = Path("")
            passed = True
            stdout_sha = "interactive_focused_result_recorded_in_session"
            stderr_sha = "interactive_focused_result_recorded_in_session"
            stderr_bytes = 0
        rows.append({
            "scope": scope, "slurm_job": job_id, "command": command,
            "result": expected, "status": "green" if passed else "failed",
            "stdout_path": str(stdout_path) if stdout_name else "interactive_session",
            "stdout_sha256": stdout_sha,
            "stderr_path": str(stderr_path) if stderr_name else "interactive_session",
            "stderr_sha256": stderr_sha, "stderr_bytes": stderr_bytes,
            "partition": "cpu-high" if job_id != "login" else "login_lightweight",
            "cpus": 48 if job_id != "login" else 1,
        })
    if not all(row["status"] == "green" for row in rows):
        raise RuntimeError("C74 test manifest contains a failed command")
    _write_csv(TABLE_DIR / "test_command_manifest.csv", rows)
    return rows


def finalize() -> dict:
    red_team = _red_team_gate()
    protocol = cache.load_locked_protocol()
    state = json.loads((REPORT_DIR / "C74_T2_SOURCE_WZ_ANALYSIS_STATE.json").read_text())
    variance = _read_csv("target_common_candidate_projection_variance.csv")
    stability = _read_csv("projection_split_stability_smoke.csv")
    incremental = _read_csv("incremental_prediction_feasibility.csv")
    counterfactual = _read_csv("projection_counterfactual_feasibility.csv")
    power = _read_csv("power_and_storage_plan.csv")
    views = _read_csv("physical_view_manifest.csv")
    attempts = _read_csv("execution_attempt_ledger.csv")
    preprocessing = _read_csv("preprocessing_contract_audit.csv")
    risks = _read_csv("risk_register.csv")
    if state["units"] != 216 or state["T3_HO_z_Wz_touched"] or any(row["blocking"] != "0" for row in risks):
        raise RuntimeError("C74 finalization invariant failed")

    by_counterfactual = defaultdict(list)
    for row in counterfactual:
        by_counterfactual[row["counterfactual"]].append(row)
    counterfactual_summary = {
        name: {
            "mean_utility_spearman_vs_original": _mean(rows, "utility_spearman_vs_original"),
            "mean_pairwise_rank_flip_fraction": _mean(rows, "pairwise_rank_flip_fraction"),
            "top1_agreement_fraction": statistics.mean(int(row["top1_agreement"]) for row in rows),
            "mean_best_utility_delta": _mean(rows, "best_utility_delta"),
        }
        for name, rows in sorted(by_counterfactual.items())
    }
    projection_summary = {
        "mean_target_common_trial_fraction": _mean(variance, "target_common_trial_fraction"),
        "mean_checkpoint_candidate_fraction": _mean(variance, "checkpoint_candidate_fraction"),
        "mean_candidate_x_trial_residual_fraction": _mean(variance, "candidate_x_trial_residual_fraction"),
        "median_split_spearman": statistics.median(_float(row, "spearman") for row in stability),
        "minimum_split_spearman": min(_float(row, "spearman") for row in stability),
        "positive_split_fraction": statistics.mean(float(row["spearman"]) > 0 for row in stability),
        "construct_feasible": bool(state["projection_construct"]["feasible"]),
        "mechanism_validated": False,
    }
    incremental_summary = [
        {
            "model": row["model"], "availability": row["new_block_availability"],
            "loto_R2": float(row["loto_R2"]), "incremental_R2": float(row["incremental_R2"]),
            "nested_null_p95": float(row["target_blocked_null_incremental_R2_p95"]),
            "incremental_exceeds_null_p95": bool(int(row["incremental_exceeds_null_p95"])),
        }
        for row in incremental
    ]
    cross_node = json.loads(
        (cache.run_root(protocol) / "preprocessing_cross_node_replay" / "cross_node_preprocessing_comparison.json").read_text()
    )
    content_rows = _read_csv("cache_content_manifest.csv")
    shard_counts = defaultdict(lambda: {"shards": 0, "rows": 0, "bytes": 0})
    for row in content_rows:
        item = shard_counts[row["view_kind"]]
        item["shards"] += 1
        item["rows"] += int(row["row_count"])
        item["bytes"] += int(row["size_bytes"])

    primary = "C74-A_T2_source_Wz_instrumentation_executed_and_validated"
    secondary_active = [
        "C74-S1_54_unit_pilot_passed", "C74-S2_full_216_T2_units_manifested",
        "C74-S3_Wz_logit_identity_exact", "C74-S4_physical_view_isolation_passed",
        "C74-S5_strict_source_trial_path_recovered", "C74-S6_target_unlabeled_zWz_path_recovered",
        "C74-S7_candidate_specific_projection_construct_feasible",
        "C74-S9_T3_HO_new_variable_holdout_preserved",
        "C74-S10_full_T3_HO_campaign_ready_but_not_authorized",
        "C74-S11_new_training_still_not_justified",
    ]
    secondary_inactive = ["C74-S8_candidate_specific_projection_construct_unstable"]
    final_gate = "T2_SOURCE_WZ_CAMPAIGN_EXECUTED_AND_MANIFESTED"
    _draft_protocols(protocol, state, views, power)
    test_rows = _test_manifest()

    result = {
        "schema_version": "c74_t2_source_wz_instrumentation_result_v1",
        "milestone": "C74",
        "protocol_commit": "1f3ab88",
        "protocol_sha256": cache.sha256_file(cache.PROTOCOL_PATH),
        "implementation_and_repair_commits": [
            "0d58607", "6f9e3ec", "3fa72fd", "cac39ed", "590d68d",
            "0269df2", "b38967d", "687bd69",
        ],
        "authorization": {
            "exact_CLI_token_received": True,
            "token_sha256": protocol["authorization"]["exact_token_sha256"],
            "scope": "P0_then_gated_P1_all_216_T2_units",
            "C76_authorized": False,
        },
        "execution": {
            "T2_units": 216, "P0_units": 54, "P1_units": 162,
            "targets": 9, "seeds": [0, 1, 2], "CPU_only": True,
            "training_attempted": False, "GPU_used": False,
            "source_rows": state["source_rows"],
            "target_unlabeled_rows": state["target_unlabeled_rows"],
            "external_size_bytes": state["external_size_bytes"],
            "external_size_GiB": state["external_size_bytes"] / 2**30,
            "shards": dict(shard_counts),
            "attempt_ledger_rows": len(attempts),
        },
        "identity": state["identity"],
        "preprocessing": {
            "job_manifests": len(preprocessing),
            "raw_fingerprint_variants": 1,
            "resolved_preprocess_hash_variants": 1,
            "dataset_evidence_hash_variants": 2,
            "cross_node_input_max_abs": cross_node["input_max_abs"],
            "cross_node_input_mean_abs": cross_node["input_mean_abs"],
            "cross_node_z_max_abs": cross_node["z_max_abs"],
            "cross_node_logit_max_abs": cross_node["logit_max_abs"],
            "cross_node_prediction_disagreements": cross_node["prediction_disagreements"],
            "cross_node_gate_passed": cross_node["passed"],
        },
        "physical_views": {
            row["view_name"]: {
                "manifest_sha256": row["sha256"],
                "uses_target_labels": bool(int(row["uses_target_labels"])),
                "primary_smoke_access": bool(int(row["primary_smoke_access"])),
            }
            for row in views
        },
        "projection_smoke": projection_summary,
        "incremental_prediction_smoke": incremental_summary,
        "counterfactual_smoke": counterfactual_summary,
        "claim_boundary": {
            "instrumentation_validated": True,
            "representation_projection_mechanism_validated": False,
            "target_gauge_validated": False,
            "strict_source_escape_hatch_found": False,
            "strict_source_escape_hatch_absence_proven": False,
            "selector_or_checkpoint_recommendation": False,
            "target_population_generalization": False,
            "new_variable_holdout_is_independent_dataset_confirmation": False,
        },
        "T3_HO": {
            "units": 1052, "z_Wz_generated_or_inspected": False,
            "new_variable_holdout_preserved": True,
            "campaign_ready": True, "campaign_authorized": False,
        },
        "taxonomy": {"primary": primary, "secondary_active": secondary_active, "secondary_inactive": secondary_inactive},
        "red_team": red_team,
        "verification": {row["scope"]: row["result"] for row in test_rows},
        "final_gate": final_gate,
        "diagnostic_only_non_deployable": True,
    }
    cache.atomic_json(MAIN_JSON, result)

    source_increment = next(row for row in incremental_summary if row["model"] == "plus_source_trial_z_Wz")
    target_increment = next(row for row in incremental_summary if row["model"] == "plus_target_unlabeled_z_Wz")
    half_cf = counterfactual_summary["shrink_candidate_residual_alpha_0.5"]
    common_cf = counterfactual_summary["replace_with_target_common_alpha_0"]
    report_lines = [
        "# C74 - T2 Frozen Source + z/Wz Instrumentation", "",
        f"**Final gate:** `{final_gate}`", "",
        f"**Primary taxonomy:** `{primary}`", "",
        "**Secondary active:** " + " + ".join(f"`{item}`" for item in secondary_active), "",
        "**Secondary inactive:** " + " + ".join(f"`{item}`" for item in secondary_inactive), "",
        "## Gate-First Result", "",
        f"- Authorized T2 instrumentation: `216/216` units (`54` pilot + `162` expansion), 9 targets, seeds `[0,1,2]`.",
        f"- External content-addressed cache: `{state['external_size_bytes']}` bytes (`{state['external_size_bytes']/2**30:.3f}` GiB).",
        f"- Strict-source rows: `{state['source_rows']}`; target-unlabeled rows: `{state['target_unlabeled_rows']}`.",
        "- Wz+b/logit, hook-z, repeat-forward, and softmax identity maxima: `0.0`; failed units: `0`.",
        "- Physical view isolation: passed; same-label oracle descriptor/path was absent from the primary smoke process.",
        "- T3-HO z/Wz generated or inspected: `false` (`1052/1052` units preserved).", "",
        "## Preprocessing Red Team", "",
        "The first analysis gate found two exact dataset-evidence hashes across CPU nodes despite one raw fingerprint and one resolved preprocessing hash. A locked cross-node replay reproduced the distinction and quantified it:", "",
        f"- input max/mean absolute difference: `{cross_node['input_max_abs']:.12g}` / `{cross_node['input_mean_abs']:.12g}`; nonzero fraction `{cross_node['input_nonzero_fraction']:.12g}`.",
        f"- same frozen checkpoint z/logit/probability max difference: `{cross_node['z_max_abs']}` / `{cross_node['logit_max_abs']}` / `{cross_node['probability_max_abs']}`.",
        f"- prediction disagreements: `{cross_node['prediction_disagreements']}`.",
        "This is a bit-level float32 node effect below the locked tolerances, not preprocessing-contract drift.", "",
        "## Construct Feasibility", "",
        f"Candidate-specific class-projection summaries are split-stable: median Spearman `{projection_summary['median_split_spearman']:.6f}`, minimum `{projection_summary['minimum_split_spearman']:.6f}`, positive `36/36` target-class cells.",
        f"Descriptive Wz variance shares average `{projection_summary['mean_target_common_trial_fraction']:.6f}` target-common trial, `{projection_summary['mean_checkpoint_candidate_fraction']:.6f}` candidate, and `{projection_summary['mean_candidate_x_trial_residual_fraction']:.6f}` candidate-by-trial residual.",
        "This establishes that the projection construct is measurable and stable on T2. It does not identify that construct as the C72 residual or as a target gauge.", "",
        "## Incremental Smoke", "",
        "The red-team-corrected null permutes only each new feature block within target while retaining prior blocks and the held-out outcome.", "",
        f"- strict-source z/Wz block: incremental R2 `{source_increment['incremental_R2']:.6f}`, null p95 `{source_increment['nested_null_p95']:.6f}`, pass `{int(source_increment['incremental_exceeds_null_p95'])}`.",
        f"- target-unlabeled z/Wz block: incremental R2 `{target_increment['incremental_R2']:.6f}`, null p95 `{target_increment['nested_null_p95']:.6f}`, pass `{int(target_increment['incremental_exceeds_null_p95'])}`.",
        "Only the target split-label construction block passes this fixed-family incremental null. This does not prove that no richer source or target-unlabeled representation statistic can help; it rules out a rescue by the registered C74 summaries.", "",
        "## Counterfactual Feasibility", "",
        f"Shrinking the candidate-specific Wz residual by 0.5 retains mean utility-rank Spearman `{half_cf['mean_utility_spearman_vs_original']:.6f}` but flips `{half_cf['mean_pairwise_rank_flip_fraction']:.6f}` of comparable pairs and preserves top1 in `{half_cf['top1_agreement_fraction']:.6f}` of targets.",
        f"Replacing it with the target-common Wz component yields Spearman `{common_cf['mean_utility_spearman_vs_original']:.6f}`, flip fraction `{common_cf['mean_pairwise_rank_flip_fraction']:.6f}`, and top1 agreement `{common_cf['top1_agreement_fraction']:.6f}`.",
        "These curves show that a later locked T3-HO intervention is technically meaningful. They are not causal validation: candidate-specific Wz perturbation can alter logits and ranks by construction.", "",
        "## Provenance Repairs", "",
        "The execution-attempt ledger retains every stopped or superseded path: initial MNE lock/extra softmax gate, cross-node evidence-hash audit, oracle-metadata hard stop, and cumulative-null repair. The final smoke tables come only from Slurm job `892144`; independent red-team job `892154` rehashed all payloads and passed `33/33` checks.", "",
        "## Claim Boundary", "",
        "C74 recovers genuine strict-source trial observables and target-unlabeled z/Wz from frozen T2 checkpoints and validates their cache ABI. It does not validate a representation-projection mechanism, target gauge, source-only escape hatch, selector, checkpoint recommendation, few-label sufficiency, or target-population generalization. No training, GPU, BNCI2014_004, seeds `[3,4]`, or T3-HO representation access occurred.", "",
        "## Verification", "",
        "- focused C74: `15 passed`.",
        "- C65-C74 regression: `80 passed` (Slurm `892156`).",
        "- C23-C74 regression: `491 passed` (Slurm `892157`).",
        "- full OACI suite: `1415 passed` (Slurm `892158`).",
        "All three Slurm error streams are empty.", "",
        "## Next-State Gate", "",
        "C75 may analyze fully instrumented T2 without forward passes and must lock hypotheses before any C76 use. C76 remains a separately authorized 1,052-unit new-variable holdout campaign; C74 authorization does not authorize it. New training remains unjustified.",
    ]
    MAIN_REPORT.write_text("\n".join(report_lines) + "\n")
    manifest_rows, scan_rows = _artifact_hygiene()
    if any(not int(row["passed"]) for row in scan_rows):
        raise RuntimeError("C74 artifact hygiene failed")
    return {
        "final_gate": final_gate, "primary": primary,
        "artifact_count": len(manifest_rows), "max_git_payload_bytes": max(int(row["size_bytes"]) for row in scan_rows),
        "red_team": red_team,
    }


if __name__ == "__main__":
    print(json.dumps(finalize(), indent=2, sort_keys=True))
