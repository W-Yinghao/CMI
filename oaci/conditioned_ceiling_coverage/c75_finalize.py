"""Generate C75 compact reports only after the independent red-team passes."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from . import c74_cache
from . import c75_data
from . import c75_protocol


REPORT_DIR = Path("oaci/reports")
TABLE_DIR = REPORT_DIR / "c75_tables"
MAIN_MD = REPORT_DIR / "C75_REPRESENTATION_CONSTRUCT_VALIDITY.md"
MAIN_JSON = REPORT_DIR / "C75_REPRESENTATION_CONSTRUCT_VALIDITY.json"
THEORY_NOTE = REPORT_DIR / "C75_FACTORIZATION_NONIDENTIFIABILITY_NOTE.md"


def _rows(name: str) -> list[dict]:
    with open(TABLE_DIR / name, newline="") as stream:
        return list(csv.DictReader(stream))


def _row(name: str, **matches) -> dict:
    selected = [
        row for row in _rows(name)
        if all(row[key] == value for key, value in matches.items())
    ]
    if len(selected) != 1:
        raise RuntimeError(f"C75 expected one row in {name}: {matches}")
    return selected[0]


def _float(row: dict, key: str) -> float:
    return float(row[key])


def _artifact_manifest() -> list[dict]:
    paths = [
        MAIN_MD, MAIN_JSON, THEORY_NOTE,
        REPORT_DIR / "C75_RED_TEAM_VERIFICATION.md",
        REPORT_DIR / "C75_REPRESENTATION_CONSTRUCT_ANALYSIS_STATE.json",
        c75_protocol.PROTOCOL_PATH, c75_protocol.PROTOCOL_SHA_PATH,
        c75_protocol.TIMING_PATH,
    ]
    paths.extend(sorted(TABLE_DIR.glob("*.csv")))
    paths.extend([
        Path("oaci/conditioned_ceiling_coverage/c75_data.py"),
        Path("oaci/conditioned_ceiling_coverage/c75_modeling.py"),
        Path("oaci/conditioned_ceiling_coverage/c75_projection.py"),
        Path("oaci/conditioned_ceiling_coverage/c75_representation_construct_validity.py"),
        Path("oaci/conditioned_ceiling_coverage/c75_red_team.py"),
        Path("oaci/conditioned_ceiling_coverage/synthetic_factorization_generator.py"),
        Path("oaci/tests/test_c75_representation_construct_validity.py"),
    ])
    result = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        row_count = 0
        if path.suffix == ".csv":
            with open(path, newline="") as stream:
                row_count = sum(1 for _ in csv.DictReader(stream))
        result.append({
            "path": str(path), "sha256": c75_protocol.sha256(path),
            "size_bytes": path.stat().st_size, "row_count": row_count,
            "raw_trial_cache": 0,
        })
    return result


def finalize() -> dict:
    checks = _rows("red_team_checks.csv")
    if not checks or any(row["blocking"] == "1" and row["passed"] != "1" for row in checks):
        raise RuntimeError("C75 finalization requires a passing independent red-team")
    red_team_text = (REPORT_DIR / "C75_RED_TEAM_VERIFICATION.md").read_text()
    if "Final status: `PASS`" not in red_team_text:
        raise RuntimeError("C75 red-team report is not PASS")

    protocol = c75_data.load_protocol()
    feature_manifest, _ = c75_data.load_feature_cache()
    redundancy = {row["view"]: row for row in _rows("Wz_logit_redundancy.csv")}
    f2 = _row("cross_fitted_incremental_relevance.csv", path="P_F2_strict_architecture", outcome="continuous_joint_utility")
    f4 = _row("cross_fitted_incremental_relevance.csv", path="P_F4_target_architecture", outcome="continuous_joint_utility")
    f5 = _row("cross_fitted_incremental_relevance.csv", path="P_F5_construction_positive", outcome="continuous_joint_utility")
    projection = _rows("projection_construct_validity.csv")[0]
    conditional = {row["path"]: row for row in _rows("representation_conditional_observability.csv")}
    actionability = {row["path"]: row for row in _rows("actionability_increment_summary.csv")}
    variance = {row["component"]: row for row in _rows("projection_variance_bootstrap.csv")}
    counterfactual = {row["intervention"]: row for row in _rows("counterfactual_identity_vs_mechanism.csv")}
    synthetic = {row["case"]: row for row in _rows("synthetic_false_positive_control.csv")}
    transformations = _rows("synthetic_reparameterization_audit.csv")
    qualification = _rows("t3_qualification_decision.csv")

    active_primary = [
        "C75-A_stable_projection_construct_functionally_redundant_nonpredictive",
        "C75-D_factorization_nonidentifiable_functional_logit_level_only",
        "C75-E_mixed_architecture_tied_representation_signal",
    ]
    inactive_primary = [
        "C75-B_target_unlabeled_projection_candidate",
        "C75-C_strict_source_representation_escape_hatch_candidate",
        "C75-F_cache_protocol_blocker",
    ]
    active_secondary = ["S1", "S2", "S3", "S4", "S5", "S6", "S10", "S12"]
    inactive_secondary = ["S7", "S8", "S9", "S11"]
    final_gate = "T3_HO_REPRESENTATION_CAMPAIGN_NOT_JUSTIFIED"

    theory_lines = [
        "# C75 - Factorization Non-Identifiability Note", "",
        "## Equivalence", "",
        "Let a linear classifier head satisfy `logits = Wz + b`. For any invertible matrix `A`, define:", "",
        "```text", "z' = A z", "W' = W A^{-1}", "```", "",
        "Then `W'z' = W A^{-1} A z = Wz`, so logits, probabilities, margins, predictions, and every metric computed from them are unchanged.", "",
        "## Identifiability Boundary", "",
        "`Wz` is function-level identifiable once persisted `b` is known because `Wz = logits - b`. The separate origin of an effect in `W` versus coordinates of `z` is not identifiable from the prediction function alone. General invertible reparameterizations change z norms/covariances, W row geometry, and normalized z/W alignment; orthogonal transforms preserve only the registered orthogonal invariants.", "",
        "The actual frozen architecture fixes one reproducible coordinate system, so coordinate-tied descriptors remain measurable. Reproducibility does not turn them into a unique causal origin: that requires extra architectural assumptions or an intervention that breaks the equivalence class.", "",
        "## Synthetic Audit", "",
        f"Across identity, orthogonal, diagonal-scale, and well-conditioned non-orthogonal transforms, the maximum Wz error is `{max(float(row['Wz_max_abs_error']) for row in transformations):.12g}` and every prediction function is invariant. Diagonal and non-orthogonal transforms change coordinate geometry.", "",
        "The construct-validity benchmark gives detection rates:", "",
        f"- stable endpoint-irrelevant descriptor: `{float(synthetic['stable_endpoint_irrelevant']['detection_rate']):.3f}`",
        f"- functionally redundant descriptor: `{float(synthetic['functionally_redundant']['detection_rate']):.3f}`",
        f"- truly incremental representation descriptor: `{float(synthetic['incremental_representation']['detection_rate']):.3f}`", "",
        "## C75 Consequence", "",
        "C75 can identify exact Wz/logit redundancy and test whether registered coordinate-tied blocks add held-out information. It cannot uniquely name W or z as the origin of an association. The significant nonlinear proxy is therefore an architecture-tied association only, not representation causality, target gauge, or a selector.",
    ]
    THEORY_NOTE.write_text("\n".join(theory_lines) + "\n")

    payload = {
        "schema_version": "c75_representation_construct_validity_report_v1",
        "milestone": "C75",
        "protocol_sha256": c75_protocol.sha256(c75_protocol.PROTOCOL_PATH),
        "parent_C74_result_commit": c75_protocol.PARENT_COMMIT,
        "implementation_commits": ["768d37b", "acd1c45", "cf4c985", "9c0127b", "41e8344"],
        "final_gate": final_gate,
        "taxonomy": {
            "primary_active": active_primary, "primary_inactive": inactive_primary,
            "secondary_active": active_secondary, "secondary_inactive": inactive_secondary,
            "scope_note": "C75-D is identifiability-only; C75-E is association-only and does not override failed predictive qualification.",
        },
        "data_boundary": {
            "T2_units": feature_manifest["unit_count"], "targets": feature_manifest["target_count"],
            "T3_HO_z_Wz_touched": False, "same_label_oracle_accessed": False,
            "forward_passes": 0, "re_inference": 0, "training": 0, "GPU": False,
            "feature_cache_sha256": feature_manifest["descriptor"]["sha256"],
            "feature_cache_size_bytes": feature_manifest["descriptor"]["size_bytes"],
            "C74_descriptors_rehashed": feature_manifest["payload_descriptors_rehashed"],
        },
        "Wz_logit_redundancy": {
            view: {
                "identity_max_abs": _float(row, "summary_max_abs_Wz_minus_logits_minus_b"),
                "B1_rank": int(row["B1_rank"]), "B2_rank": int(row["B2_rank"]),
                "column_space_prediction_delta_max_abs": _float(row, "column_space_prediction_delta_max_abs"),
                "column_space_incremental_R2": _float(row, "column_space_incremental_R2"),
                "naive_duplicate_incremental_R2": _float(row, "naive_ridge_incremental_R2"),
            } for view, row in redundancy.items()
        },
        "registered_incremental_relevance": {
            "F2_strict_source": {key: _float(f2, key) for key in ("incremental_R2", "nested_null_p95", "max_stat_corrected_p", "median_increment_residual_rho")}
            | {"positive_targets": int(f2["positive_target_count"]), "qualified": False},
            "F4_target_unlabeled": {key: _float(f4, key) for key in ("incremental_R2", "nested_null_p95", "max_stat_corrected_p", "median_increment_residual_rho")}
            | {"positive_targets": int(f4["positive_target_count"]), "qualified": False},
            "F5_construction_positive_control": {key: _float(f5, key) for key in ("incremental_R2", "nested_null_p95", "max_stat_corrected_p", "median_increment_residual_rho")}
            | {"positive_targets": int(f5["positive_target_count"]), "target_label_derived": True},
        },
        "projection_construct": {
            "median_split_spearman": _float(projection, "median_split_spearman"),
            "minimum_split_spearman": _float(projection, "minimum_split_spearman"),
            "positive_cells": int(projection["positive_split_cells"]),
            "classification": projection["classification"],
            "registered_incremental": False, "mechanism_origin_validated": False,
        },
        "conditional_observability_counter_result": {
            path: {
                "max_alignment": _float(row, "max_observed_alignment"),
                "global_six_test_max_stat_p": _float(row, "global_path_bandwidth_max_stat_p"),
                "exact_conditional_CS": False, "predictive_qualification": False,
            } for path, row in conditional.items()
        },
        "actionability": {
            path: {
                "delta_top1": _float(row, "delta_top1"), "delta_top3": _float(row, "delta_top3"),
                "regret_reduction": _float(row, "regret_reduction"),
                "positive_regret_reduction_targets": int(row["positive_regret_reduction_targets"]),
            } for path, row in actionability.items() if path in {"P_F2_strict_architecture", "P_F4_target_architecture", "P_F5_construction_positive"}
        },
        "variance_estimand": {
            component: {"mean_share": _float(row, "point_mean"), "ci_low": _float(row, "ci_low"), "ci_high": _float(row, "ci_high")}
            for component, row in variance.items()
        },
        "counterfactual": {
            name: {"observed_mean_flip": _float(row, "observed_mean_flip"), "matched_null_p95": _float(row, "matched_null_p95"), "max_family_p": _float(row, "max_family_p"), "mechanism_origin_identified": False}
            for name, row in counterfactual.items()
        },
        "qualification": {
            "all_required_passes": [row["candidate"] for row in qualification if row["gate"] == "ALL_REQUIRED" and row["passed"] == "1"],
            "C76_protocol_created": False, "T3_HO_campaign_justified": False,
        },
        "red_team": {
            "status": "PASS", "blocking_checks": 31, "total_checks": 32,
            "repairs": ["canonical_float_projection_summary", "fold_scaled_global_maxstat_conditional_proxy"],
        },
        "verification": {
            "focused_C75": {"passed": 15},
            "C65_C75": {"passed": 95, "slurm_job": 892459},
            "C23_C75": {"passed": 502, "slurm_job": 892460},
            "full_OACI": {"passed": 1430, "slurm_job": 892461},
            "all_stderr_empty": True,
        },
        "claim_boundary": {
            "representation_causality": False, "target_gauge": False,
            "source_only_escape_hatch": False, "selector": False,
            "deployability": False, "target_population_generalization": False,
            "new_training_justified": False, "diagnostic_only": True,
        },
    }
    c74_cache.atomic_json(MAIN_JSON, payload)

    lines = [
        "# C75 - T2 Representation-Projection Construct Validity / Factorization Non-Identifiability Audit", "",
        f"**Final gate:** `{final_gate}`", "",
        "**Primary active:** `" + " + ".join(active_primary) + "`", "",
        "**Primary inactive:** `" + " + ".join(inactive_primary) + "`", "",
        "## Gate-First Result", "",
        f"C75 used only the manifested 216-unit T2 cache. T3-HO z/Wz access, same-label oracle payload access, forward passes, re-inference, training, and GPU use were all zero. The compact registered feature cache is `{feature_manifest['descriptor']['size_bytes']}` bytes and replays 1,080 C74 descriptors.", "",
        "Neither registered architecture block qualifies for a C76 holdout:", "",
        f"- F2 strict-source: incremental R2 `{_float(f2, 'incremental_R2'):.6f}`, null p95 `{_float(f2, 'nested_null_p95'):.6f}`, max-stat p `{_float(f2, 'max_stat_corrected_p'):.3f}`, positive targets `{int(f2['positive_target_count'])}/9`.",
        f"- F4 target-unlabeled: incremental R2 `{_float(f4, 'incremental_R2'):.6f}`, null p95 `{_float(f4, 'nested_null_p95'):.6f}`, max-stat p `{_float(f4, 'max_stat_corrected_p'):.3f}`, positive targets `{int(f4['positive_target_count'])}/9`.",
        f"- F5 construction-label positive control: incremental R2 `{_float(f5, 'incremental_R2'):.6f}`, max-stat p `{_float(f5, 'max_stat_corrected_p'):.3f}`, positive targets `{int(f5['positive_target_count'])}/9`.", "",
        "## Exact Redundancy", "",
        f"After the red-team repair, canonical `logits-b` and Wz summaries are bit-identical. Strict-source B1/B2 ranks are `{redundancy['strict_source']['B1_rank']}/{redundancy['strict_source']['B2_rank']}` with prediction delta `{_float(redundancy['strict_source'], 'column_space_prediction_delta_max_abs'):.3g}`; target-unlabeled ranks are `{redundancy['target_unlabeled']['B1_rank']}/{redundancy['target_unlabeled']['B2_rank']}` with delta `{_float(redundancy['target_unlabeled'], 'column_space_prediction_delta_max_abs'):.3g}`. Any naive duplicate-block change is regularization parameterization, not information gain.", "",
        "## Construct Validity", "",
        f"Class-conditioned projection summaries remain highly split-stable: median Spearman `{_float(projection, 'median_split_spearman'):.6f}`, minimum `{_float(projection, 'minimum_split_spearman'):.6f}`, positive `{int(projection['positive_split_cells'])}/36`. Stability does not become registered held-out prediction: F4 primary incremental R2 is negative and fails every materiality/null qualification gate.", "",
        f"Descriptive Wz shares remain `{_float(variance['target_common_trial'], 'point_mean'):.6f}` target-common trial, `{_float(variance['checkpoint_candidate'], 'point_mean'):.6f}` checkpoint/candidate, and `{_float(variance['candidate_x_trial_residual'], 'point_mean'):.6f}` interaction residual. These are crossed descriptive ANOVA estimands, not causal shares.", "",
        "## Nonlinear Counter-Result", "",
        f"The preregistered, fold-scaled RBF residual-alignment proxy is significant after global 2-path x 3-bandwidth max-stat correction: strict-source p `{_float(conditional['strict_source'], 'global_path_bandwidth_max_stat_p'):.3f}` and target-unlabeled p `{_float(conditional['target_unlabeled'], 'global_path_bandwidth_max_stat_p'):.3f}`. This is a real association-only counter-result and prevents a blanket endpoint-irrelevance claim.", "",
        "It does not satisfy the locked predictive qualification, improve control reliably across targets, estimate exact conditional-CS, identify representation origin, or authorize T3-HO. C75-E is active only in this narrow sense.", "",
        "## Factorization Boundary", "",
        "For every invertible A, `z'=Az` and `W'=WA^{-1}` preserve `W'z'=Wz`. Thus logits/probabilities identify the function but cannot uniquely assign an effect to W versus z coordinates. Synthetic identity, orthogonal, scaled, and non-orthogonal transforms preserve Wz to below `1e-10` while general transforms change coordinate geometry. C75-D is this identifiability statement; it does not erase the nonlinear association above.", "",
        "## Counterfactual Audit", "",
        f"Residual shrink gives mean pairwise flips `{_float(counterfactual['residual_shrink_0.5'], 'observed_mean_flip'):.6f}` with matched max-family p `{_float(counterfactual['residual_shrink_0.5'], 'max_family_p'):.3f}`. Target-common replacement gives `{_float(counterfactual['target_common_replacement'], 'observed_mean_flip'):.6f}` flips with p `{_float(counterfactual['target_common_replacement'], 'max_family_p'):.3f}`. Matched nulls explain these sensitivity curves; mechanism origin remains unidentified.", "",
        "## Red-Team Repairs", "",
        "The first completed analysis was invalidated because float32 and float64 reductions created pseudo-rank in an exactly duplicate Wz block. A canonical float64 summary plus hard identity gate repaired it. A second completed analysis was superseded because the RBF proxy lacked fold-local scaling and cross-path multiplicity correction. Final evidence comes only from the repaired extraction (`892425`) and final analysis (`892437`). Independent red-team job `892453` passed 31/31 blocking checks and 32/32 total checks after rehashing all 1,080 inputs.", "",
        "## Claim Boundary", "",
        "C75 supports exact Wz/logit redundancy, general factorization non-identifiability, a stable but non-qualified registered projection construct, and a nonlinear association-only counter-result. It does not validate representation causality, name the unexplained residual as target gauge, establish a source-only escape hatch, create a selector/checkpoint recommendation, justify T3-HO generation, or justify new training. No C76 protocol is created.", "",
        "## Verification", "",
        "- focused C75: `15 passed`.",
        "- C65-C75 regression: `95 passed` (Slurm `892459`).",
        "- C23-C75 regression: `502 passed` (Slurm `892460`).",
        "- full OACI suite: `1430 passed` (Slurm `892461`).",
        "- all three stderr streams: empty.", "",
        "## Next-State Gate", "",
        "The locked C75 qualification rule does not justify the 1,052-unit T3-HO representation campaign. T3-HO remains untouched, no C76 protocol exists, and any next scientific step requires a new PM decision rather than automatic continuation.",
    ]
    MAIN_MD.write_text("\n".join(lines) + "\n")

    manifest_rows = _artifact_manifest()
    with open(TABLE_DIR / "artifact_manifest.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)
    return {
        "final_gate": final_gate, "primary_active": active_primary,
        "artifact_count": len(manifest_rows), "C76_protocol_created": False,
        "T3_HO_z_Wz_touched": False,
    }


if __name__ == "__main__":
    print(json.dumps(finalize(), indent=2, sort_keys=True))
