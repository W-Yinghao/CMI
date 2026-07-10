"""Metadata-only C75 protocol and registered feature/inference contracts."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


MILESTONE = "C75"
PARENT_COMMIT = "fe467b9826c458f0626b775704a60c55ec31b6f8"
REPORT_DIR = Path("oaci/reports")
TABLE_DIR = REPORT_DIR / "c75_tables"
PROTOCOL_PATH = REPORT_DIR / "C75_REPRESENTATION_CONSTRUCT_PROTOCOL.json"
PROTOCOL_SHA_PATH = REPORT_DIR / "C75_REPRESENTATION_CONSTRUCT_PROTOCOL.sha256"
TIMING_PATH = REPORT_DIR / "C75_PROTOCOL_TIMING_AUDIT.md"
C74_PROTOCOL = REPORT_DIR / "C74_T2_SOURCE_WZ_INSTRUMENTATION_PROTOCOL.json"
C74_PROTOCOL_SHA = REPORT_DIR / "C74_T2_SOURCE_WZ_INSTRUMENTATION_PROTOCOL.sha256"
C74_RESULT = REPORT_DIR / "C74_T2_SOURCE_WZ_INSTRUMENTATION.json"
C74_ARTIFACT_MANIFEST = REPORT_DIR / "c74_tables/artifact_manifest.csv"
C74_VIEW_MANIFEST = REPORT_DIR / "c74_tables/physical_view_manifest.csv"
C74_CACHE_MANIFEST = REPORT_DIR / "c74_tables/cache_content_manifest.csv"
C74_T2_UNITS = REPORT_DIR / "c74_tables/full_t2_unit_manifest.csv"
C74_T3_UNITS = REPORT_DIR / "c74_tables/t3_ho_holdout_unit_manifest.csv"
C74_ATTEMPTS = REPORT_DIR / "c74_tables/execution_attempt_ledger.csv"
C74_CROSS_NODE = "/projects/EEG-foundation-model/yinghao/oaci-c74-t2-source-wz/protocol_76d0f4d9d96d0128/preprocessing_cross_node_replay/cross_node_preprocessing_comparison.json"

NULL_REPLICATES = 499
BOOTSTRAP_REPLICATES = 2000
RNG_SEED = 7501
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
SVD_RANK_TOLERANCE = 1e-10


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(name: str, rows: list[dict]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(TABLE_DIR / name, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def feature_registry() -> list[dict]:
    return [
        {
            "block": "F0", "name": "existing_source_metadata_baseline", "view": "strict_source+unit_metadata",
            "dimension": 9,
            "features": "seed_onehot[3];level_onehot[2];candidate_order_scaled;source_bAcc;source_NLL;source_ECE",
            "formula": "metadata plus 15-bin source-label endpoint summaries; regime constant omitted",
            "factorization_status": "functional_or_metadata", "target_labels": 0, "qualification_candidate": 0,
        },
        {
            "block": "F1", "name": "strict_source_functional", "view": "strict_source_trial_view",
            "dimension": 25,
            "features": "mean_confidence;mean_entropy;mean_top1_margin;mean_logit_norm;occupancy[4];predicted_class_confidence[4];true_class_recall[4];true_class_probability[4];mean_probability[4];ensemble_disagreement",
            "formula": "fixed class-conditioned logits/probabilities summaries on all 8 source domains",
            "factorization_status": "prediction_function_invariant", "target_labels": 0, "qualification_candidate": 0,
        },
        {
            "block": "F2", "name": "strict_source_architecture_tied", "view": "strict_source_trial_view+checkpoint_Wb",
            "dimension": 25,
            "features": "z_norm_moments[4];z_spectrum[6];W_geometry[10];normalized_zW_alignment[5]",
            "formula": "z moments on all rows; covariance spectrum on 8 SHA-smallest trials per source-domain x class (256 rows); W row/SVD geometry; normalized projection alignment",
            "factorization_status": "orthogonal_invariant_but_general_GL_nonidentifiable", "target_labels": 0, "qualification_candidate": 1,
        },
        {
            "block": "F3", "name": "target_unlabeled_functional", "view": "target_unlabeled_representation_view",
            "dimension": 18,
            "features": "mean_confidence;mean_entropy;mean_top1_margin;mean_logit_norm;occupancy[4];mean_probability[4];ensemble_disagreement;probability_shift_norm_mean_std;logit_shift_norm_mean_std;target_common_logit_residual_rms",
            "formula": "label-free summaries and candidate-vs-24-candidate target-common functional contrasts",
            "factorization_status": "prediction_function_invariant", "target_labels": 0, "qualification_candidate": 0,
        },
        {
            "block": "F4", "name": "target_unlabeled_architecture_tied", "view": "target_unlabeled_representation_view+checkpoint_Wb",
            "dimension": 35,
            "features": "z_norm_moments[4];z_spectrum[6];W_geometry[10];Wz_mean_std[8];candidate_common_Wz_residual[7]",
            "formula": "target z spectrum on 256 SHA-smallest trials; W geometry; Wz and candidate-vs-target-common projection summaries",
            "factorization_status": "mixed_Wz_GL_invariant_zW_geometry_GL_nonidentifiable", "target_labels": 0, "qualification_candidate": 1,
        },
        {
            "block": "F5", "name": "construction_label_positive_control", "view": "target_construction_view+target_unlabeled_predictions",
            "dimension": 15,
            "features": "construct_bAcc;construct_NLL;construct_ECE;class_recall[4];class_true_probability[4];mean_confidence;mean_entropy;mean_margin;construct_joint_utility",
            "formula": "physically separated construction labels only; construction utility uses within-target oriented midranks",
            "factorization_status": "target_label_derived_diagnostic", "target_labels": 1, "qualification_candidate": 0,
        },
    ]


def outcome_registry() -> list[dict]:
    return [
        {"outcome": "continuous_joint_utility", "orientation": "higher_better", "primary": 1, "definition": "within-target mean of midrank percentiles for eval bAcc,-NLL,-ECE"},
        {"outcome": "bAcc", "orientation": "higher_better", "primary": 0, "definition": "held-out evaluation balanced accuracy over four classes"},
        {"outcome": "negNLL", "orientation": "higher_better", "primary": 0, "definition": "negative held-out evaluation NLL"},
        {"outcome": "negECE", "orientation": "higher_better", "primary": 0, "definition": "negative 15-bin held-out evaluation ECE"},
        {"outcome": "primary_joint_good", "orientation": "higher_better", "primary": 0, "definition": "all three held-out oriented within-target percentiles >=0.75"},
        {"outcome": "unexplained_candidate_specific_residual", "orientation": "signed", "primary": 0, "definition": "outer-LOTO target-centered continuous utility residual after fixed F0+F1+F3+F5 functional/label-diagnostic baseline; never called gauge"},
    ]


def availability_registry() -> list[dict]:
    return [
        {"block": "F0", "information_class": "strict_source+registered_metadata", "available_strict_DG": 1, "target_unlabeled": 0, "target_label_derived": 0, "diagnostic_only": 0},
        {"block": "F1", "information_class": "strict_source_trial", "available_strict_DG": 1, "target_unlabeled": 0, "target_label_derived": 0, "diagnostic_only": 0},
        {"block": "F2", "information_class": "strict_source_architecture", "available_strict_DG": 1, "target_unlabeled": 0, "target_label_derived": 0, "diagnostic_only": 0},
        {"block": "F3", "information_class": "target_unlabeled_functional", "available_strict_DG": 0, "target_unlabeled": 1, "target_label_derived": 0, "diagnostic_only": 1},
        {"block": "F4", "information_class": "target_unlabeled_architecture", "available_strict_DG": 0, "target_unlabeled": 1, "target_label_derived": 0, "diagnostic_only": 1},
        {"block": "F5", "information_class": "target_construction_label", "available_strict_DG": 0, "target_unlabeled": 0, "target_label_derived": 1, "diagnostic_only": 1},
    ]


def factorization_registry() -> list[dict]:
    return [
        {"quantity": "logits=Wz+b", "general_invertible_A_invariant": 1, "orthogonal_A_invariant": 1, "directly_function_identifiable": 1, "status": "function_level"},
        {"quantity": "probabilities/margins/predictions", "general_invertible_A_invariant": 1, "orthogonal_A_invariant": 1, "directly_function_identifiable": 1, "status": "function_level"},
        {"quantity": "Wz", "general_invertible_A_invariant": 1, "orthogonal_A_invariant": 1, "directly_function_identifiable": 1, "status": "projection_level_given_b"},
        {"quantity": "head_bias_b", "general_invertible_A_invariant": 1, "orthogonal_A_invariant": 1, "directly_function_identifiable": 1, "status": "parameter_level"},
        {"quantity": "z_coordinate_mean/covariance", "general_invertible_A_invariant": 0, "orthogonal_A_invariant": 0, "directly_function_identifiable": 0, "status": "coordinate_dependent"},
        {"quantity": "z_norm/covariance_eigenvalues", "general_invertible_A_invariant": 0, "orthogonal_A_invariant": 1, "directly_function_identifiable": 0, "status": "orthogonal_only"},
        {"quantity": "W_row_norms/cosines/singular_values", "general_invertible_A_invariant": 0, "orthogonal_A_invariant": 1, "directly_function_identifiable": 0, "status": "orthogonal_only"},
        {"quantity": "normalized_zW_alignment", "general_invertible_A_invariant": 0, "orthogonal_A_invariant": 1, "directly_function_identifiable": 0, "status": "orthogonal_only"},
    ]


def qualification_rows() -> list[dict]:
    common = [
        ("incremental_R2", ">=", 0.02),
        ("observed_incremental_R2", ">", "primary_nested_null_p95"),
        ("leave_target_out_median_target_rho", ">", 0),
        ("positive_target_count", ">=", 7),
        ("max_stat_corrected_p", "<", 0.05),
        ("nonredundant_with_logits_probabilities", "==", 1),
        ("target_label_leakage", "==", 0),
    ]
    rows = []
    for candidate, prior in (("F2_strict_source", "F0+F1"), ("F4_target_unlabeled", "F0+F1+F3")):
        for gate, operator, threshold in common:
            rows.append({"candidate": candidate, "prior": prior, "primary_outcome": "continuous_joint_utility", "gate": gate, "operator": operator, "threshold": threshold, "all_required": 1})
    return rows


def prepare_protocol() -> dict:
    timestamp = _now()
    views = _read_csv(C74_VIEW_MANIFEST)
    t2 = _read_csv(C74_T2_UNITS)
    t3 = _read_csv(C74_T3_UNITS)
    if len(t2) != 216 or len(t3) != 1052 or any(row["z_Wz_generated_in_C74"] != "0" for row in t3):
        raise RuntimeError("C75 C74 unit-role replay failed")
    if not Path(C74_CROSS_NODE).is_file():
        raise RuntimeError("C75 C74 cross-node replay missing")

    registries = {
        "feature_block_registry.csv": feature_registry(),
        "feature_availability_ledger.csv": availability_registry(),
        "factorization_status_ledger.csv": factorization_registry(),
        "outcome_registry.csv": outcome_registry(),
        "t3_qualification_gates.csv": qualification_rows(),
    }
    replay = [
        {"artifact": "C74_protocol", "path": str(C74_PROTOCOL), "sha256": sha256(C74_PROTOCOL), "expected": Path(C74_PROTOCOL_SHA).read_text().strip(), "passed": int(sha256(C74_PROTOCOL) == Path(C74_PROTOCOL_SHA).read_text().strip())},
        {"artifact": "C74_result", "path": str(C74_RESULT), "sha256": sha256(C74_RESULT), "expected": sha256(C74_RESULT), "passed": 1},
        {"artifact": "C74_artifact_manifest", "path": str(C74_ARTIFACT_MANIFEST), "sha256": sha256(C74_ARTIFACT_MANIFEST), "expected": sha256(C74_ARTIFACT_MANIFEST), "passed": 1},
        {"artifact": "C74_view_manifest", "path": str(C74_VIEW_MANIFEST), "sha256": sha256(C74_VIEW_MANIFEST), "expected": sha256(C74_VIEW_MANIFEST), "passed": 1},
        {"artifact": "C74_cache_manifest", "path": str(C74_CACHE_MANIFEST), "sha256": sha256(C74_CACHE_MANIFEST), "expected": sha256(C74_CACHE_MANIFEST), "passed": 1},
        {"artifact": "C74_execution_attempt_ledger", "path": str(C74_ATTEMPTS), "sha256": sha256(C74_ATTEMPTS), "expected": sha256(C74_ATTEMPTS), "passed": 1},
        {"artifact": "C74_cross_node_replay", "path": C74_CROSS_NODE, "sha256": sha256(C74_CROSS_NODE), "expected": sha256(C74_CROSS_NODE), "passed": 1},
    ]
    registries["c74_identity_replay.csv"] = replay
    for name, rows in registries.items():
        _write_csv(name, rows)

    table_hashes = {
        name: {"path": str(TABLE_DIR / name), "sha256": sha256(TABLE_DIR / name), "rows": len(rows), "size_bytes": (TABLE_DIR / name).stat().st_size}
        for name, rows in registries.items()
    }
    view_hashes = {row["view_name"]: {"manifest_sha256": row["sha256"], "primary_access": int(row["primary_smoke_access"])} for row in views}
    if view_hashes["same_label_oracle_view"]["primary_access"] != 0:
        raise RuntimeError("C75 oracle view unexpectedly available")

    protocol = {
        "schema_version": "c75_representation_construct_protocol_v1",
        "milestone": MILESTONE,
        "protocol_lock_timestamp_utc": timestamp,
        "parent_C74_result_commit": PARENT_COMMIT,
        "parent_C74_protocol_sha256": sha256(C74_PROTOCOL),
        "parent_C74_result_sha256": sha256(C74_RESULT),
        "parent_C74_artifact_manifest_sha256": sha256(C74_ARTIFACT_MANIFEST),
        "parent_C74_view_manifest_sha256": sha256(C74_VIEW_MANIFEST),
        "parent_C74_cache_manifest_sha256": sha256(C74_CACHE_MANIFEST),
        "C74_view_manifest_hashes": view_hashes,
        "locked_registry_tables": table_hashes,
        "execution_boundary": {
            "forward_passes": False, "re_inference": False, "training": False, "GPU": False,
            "T3_HO_z_Wz_access": False, "BNCI2014_004": False, "seeds_3_4": False,
            "selector_or_checkpoint_artifacts": False, "raw_cache_in_git": False,
            "manuscript_drafting": False, "high_dimensional_feature_search": False,
        },
        "data_role": {
            "T2_units": 216, "role": "retrospective construct validity/discovery only",
            "T3_HO_units": 1052, "role": "untouched new-variable holdout",
            "T3_HO_access_in_C75": False, "independent_target_dataset_confirmation": False,
            "same_label_oracle_primary_access": False,
        },
        "endpoint_contract": {
            "evaluation_labels": "target_evaluation_view only",
            "construction_labels": "F5 positive control only",
            "ECE_bins": 15,
            "continuous_joint_utility": "within-target mean midrank percentile of eval bAcc,-NLL,-ECE",
            "primary_joint_good": "all eval oriented percentiles >=0.75",
            "unexplained_residual": "outer-LOTO target-centered utility residual after F0+F1+F3+F5; never gauge",
        },
        "feature_contract": {
            "source_spectral_support": "8 SHA256-smallest trial IDs per source_domain x source_class = 256 rows",
            "target_spectral_support": "256 SHA256-smallest target trial IDs",
            "covariance": "center selected z rows; eigenvalues=singular_values^2/(n-1)",
            "effective_rank": "exp(-sum(p*log(p))) with p=eigenvalue/trace",
            "stable_rank": "trace/max_eigenvalue",
            "target_common": "mean over all 24 T2 candidates within target on aligned trial IDs",
            "no_feature_selection": True,
        },
        "factorization_contract": {
            "equivalence": "z_prime=A z; W_prime=W A^{-1}; W_prime z_prime=Wz for invertible A",
            "synthetic_transforms": ["orthogonal_QR", "diagonal_scale_0.5_to_2", "nonorthogonal_condition_le_4"],
            "directly_identifiable": ["logits", "probabilities", "margins", "Wz_given_b"],
            "not_uniquely_identifiable": ["z_coordinates", "W_vs_z_origin", "general_GL_coordinate_geometry"],
        },
        "redundancy_contract": {
            "paths": {
                "strict_source": ["B0=F0", "B1=B0+F1+matched(logits-b)", "B2=B1+Wz_duplicate", "B3=B2+source_z", "B4=B3+W_geometry"],
                "target_unlabeled": ["B0=F0+F1", "B1=B0+F3+matched(logits-b)", "B2=B1+Wz_duplicate", "B3=B2+target_z", "B4=B3+W_geometry"],
            },
            "column_space": "training-fold standardized SVD basis",
            "rank_tolerance": SVD_RANK_TOLERANCE,
            "naive_duplicate_ridge_audit": True,
            "Wz_nonredundant_only_if_column_rank_increases_beyond_matched_logits_minus_b": False,
        },
        "inference_contract": {
            "outer": "leave-one-target-out",
            "secondary": "leave-one-target-x-trajectory-cell-out and leave-one-trajectory-template-out",
            "within_target_centering": True,
            "model": "ridge on registered low-dimensional blocks",
            "inner_alpha_grid": list(RIDGE_ALPHAS),
            "inner_selection": "leave-one-training-target-out mean squared error; deterministic smallest-alpha tie break",
            "feature_scaling": "training-fold mean/std only",
            "null_replicates": NULL_REPLICATES,
            "primary_null": "permute new block within each target x trajectory cell; keep prior blocks and outcome fixed",
            "secondary_null": "permute new block within target",
            "max_stat_family": "F1,F2,F3,F4,F5 increments x six outcomes",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap": "resample targets then checkpoint units within sampled target",
            "rng_seed": RNG_SEED,
        },
        "actionability_contract": {
            "field": "24-candidate target universe",
            "metrics": ["target_centered_spearman", "pairwise_ordering", "top1", "top3", "regret", "joint_good_coverage"],
            "checkpoint_ids_emitted": False,
        },
        "qualification_contract": {
            "primary_outcome": "continuous_joint_utility",
            "candidate_blocks": {"strict_source": "F2|F0+F1", "target_unlabeled": "F4|F0+F1+F3"},
            "all_gates_required": True,
            "gates_table": str(TABLE_DIR / "t3_qualification_gates.csv"),
            "if_none_pass": "do not create C76 protocol; T3-HO campaign not justified",
            "if_pass": "lock exact feature formula/outcomes/nulls/actionability/failure rules in C76 protocol without touching T3-HO",
        },
        "variance_contract": {
            "estimand": "per-target-class two-way descriptive ANOVA of Wz[candidate,trial,class] into candidate main + target-common trial main + interaction residual",
            "normalization": "sum of squares divided by total within target-class sum of squares",
            "causal_interpretation": False,
            "bootstrap": "target-cluster mean-share intervals",
        },
        "counterfactual_contract": {
            "registered": ["residual_shrink_0.5", "target_common_replacement", "candidate_permutation", "trajectory_preserving_shuffle", "magnitude_matched_random", "random_target_common_replacement"],
            "primary_effect": "pairwise rank flip fraction",
            "mechanism_origin_claim_allowed": False,
        },
        "conditional_observability_contract": {
            "primary_estimator": "nested cross-fitted prediction contrast; proxy not exact conditional-CS divergence",
            "RBF_bandwidths": [0.5, 1.0, 2.0],
            "bandwidth_scale": "training-fold median pairwise distance",
            "null_calibration": "same blocked new-block permutations with bandwidth max-stat",
            "iid_guarantee_claim_allowed": False,
        },
        "synthetic_contract": {
            "seed": RNG_SEED + 100,
            "replicates": 500,
            "n": 512, "z_dim": 16, "classes": 4,
            "cases": ["factorization_equivalent", "stable_endpoint_irrelevant", "incremental_representation", "Wz_logit_redundant"],
            "false_positive_target": 0.05,
        },
        "taxonomy": {
            "primary": [
                "C75-A_stable_projection_construct_functionally_redundant_nonpredictive",
                "C75-B_target_unlabeled_projection_candidate", "C75-C_strict_source_representation_escape_hatch_candidate",
                "C75-D_factorization_nonidentifiable_functional_logit_level_only",
                "C75-E_mixed_architecture_tied_representation_signal", "C75-F_cache_protocol_blocker",
            ],
            "secondary": [f"S{i}" for i in range(1, 13)],
        },
        "final_gates": [
            "REPRESENTATION_CONSTRUCT_STABLE_BUT_ENDPOINT_IRRELEVANT",
            "TARGET_UNLABELED_REPRESENTATION_CANDIDATE_READY_FOR_T3_HO",
            "STRICT_SOURCE_REPRESENTATION_ESCAPE_HATCH_READY_FOR_T3_HO",
            "FUNCTIONAL_LOGIT_LEVEL_MECHANISM_ONLY", "T3_HO_REPRESENTATION_CAMPAIGN_NOT_JUSTIFIED",
            "T3_HO_REPRESENTATION_CAMPAIGN_READY_BUT_NOT_AUTHORIZED", "CACHE_OR_PROTOCOL_REPAIR_REQUIRED",
        ],
        "forbidden_claims": [
            "stable construct is predictive", "unexplained residual is target gauge", "all source functions fail",
            "representation mechanism validated", "selector", "checkpoint recommendation", "deployable",
            "few-label sufficiency", "target-population generalization", "manuscript drafting",
        ],
        "diagnostic_only_non_deployable": True,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROTOCOL_PATH, "w") as stream:
        json.dump(protocol, stream, indent=2, sort_keys=True)
        stream.write("\n")
    digest = sha256(PROTOCOL_PATH)
    PROTOCOL_SHA_PATH.write_text(digest + "\n")
    TIMING_PATH.write_text("\n".join([
        "# C75 - Protocol Timing Audit", "",
        f"- Protocol lock timestamp: `{timestamp}`",
        f"- Parent C74 result commit: `{PARENT_COMMIT[:7]}`",
        "- New C75 endpoint analysis before lock: `false`",
        "- C74 external z/Wz payload read before lock: `false`",
        "- T3-HO z/Wz accessed: `false`",
        "- Feature formulas, outcomes, nulls, multiplicity family, qualification gates, and stop rules locked: `true`",
        "", "C75 remains retrospective T2 construct analysis. This protocol does not make it an independent confirmation.",
    ]) + "\n")
    return {"protocol_sha256": digest, "timestamp": timestamp, "registry_tables": table_hashes}


if __name__ == "__main__":
    print(json.dumps(prepare_protocol(), indent=2, sort_keys=True))
