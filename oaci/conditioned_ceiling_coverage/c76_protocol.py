"""Metadata-only C76 protocol and registered orbit/association contracts."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


MILESTONE = "C76"
PARENT_COMMIT = "fb8a412695ab9df391622c1694bb9bd8cbc567a7"
REPORT_DIR = Path("oaci/reports")
TABLE_DIR = REPORT_DIR / "c76_tables"
PROTOCOL_PATH = REPORT_DIR / "C76_REPRESENTATION_ASSOCIATION_PROTOCOL.json"
PROTOCOL_SHA_PATH = REPORT_DIR / "C76_REPRESENTATION_ASSOCIATION_PROTOCOL.sha256"
TIMING_PATH = REPORT_DIR / "C76_PROTOCOL_TIMING_AUDIT.md"

C75_PROTOCOL = REPORT_DIR / "C75_REPRESENTATION_CONSTRUCT_PROTOCOL.json"
C75_PROTOCOL_SHA = REPORT_DIR / "C75_REPRESENTATION_CONSTRUCT_PROTOCOL.sha256"
C75_RESULT = REPORT_DIR / "C75_REPRESENTATION_CONSTRUCT_VALIDITY.json"
C75_REPORT = REPORT_DIR / "C75_REPRESENTATION_CONSTRUCT_VALIDITY.md"
C75_ARTIFACT_MANIFEST = REPORT_DIR / "c75_tables/artifact_manifest.csv"
C75_RED_TEAM = REPORT_DIR / "C75_RED_TEAM_VERIFICATION.md"
C75_RELEVANCE = REPORT_DIR / "c75_tables/cross_fitted_incremental_relevance.csv"
C75_RBF = REPORT_DIR / "c75_tables/representation_conditional_observability.csv"
C75_REDUNDANCY = REPORT_DIR / "c75_tables/Wz_logit_redundancy.csv"
C75_PROJECTION = REPORT_DIR / "c75_tables/projection_construct_validity.csv"
C75_T3_DECISION = REPORT_DIR / "c75_tables/t3_ho_decision.csv"

C74_PROTOCOL = REPORT_DIR / "C74_T2_SOURCE_WZ_INSTRUMENTATION_PROTOCOL.json"
C74_VIEW_MANIFEST = REPORT_DIR / "c74_tables/physical_view_manifest.csv"
C74_T2_UNITS = REPORT_DIR / "c74_tables/full_t2_unit_manifest.csv"
C74_T3_UNITS = REPORT_DIR / "c74_tables/t3_ho_holdout_unit_manifest.csv"

NULL_REPLICATES = 499
BOOTSTRAP_REPLICATES = 2000
SYNTHETIC_REPLICATES = 500
ORBIT_REPLICATES = 4
RNG_SEED = 7601
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
KRR_ALPHAS = (0.01, 0.1, 1.0, 10.0)
BANDWIDTH_FACTORS = (0.5, 1.0, 2.0)
KERNEL_FAMILIES = ("rbf", "laplacian")
ASSOCIATION_STATISTICS = ("normalized_alignment", "centered_hsic")
FUNCTIONAL_IDENTITY_TOLERANCE = 1e-8


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
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def orbit_registry() -> list[dict]:
    return [
        {"orbit": "O0", "name": "identity", "scope": "global", "family": "identity", "replicates": 1, "orthogonal": 1, "general_GL": 1, "efficient_structure": "identity", "condition_bound": 1.0, "multiplicity_family": 0},
        {"orbit": "O1", "name": "global_orthogonal", "scope": "global", "family": "orthogonal", "replicates": ORBIT_REPLICATES, "orthogonal": 1, "general_GL": 1, "efficient_structure": "2x2_block_rotations", "condition_bound": 1.0, "multiplicity_family": 1},
        {"orbit": "O2", "name": "checkpoint_specific_orthogonal", "scope": "checkpoint", "family": "orthogonal", "replicates": ORBIT_REPLICATES, "orthogonal": 1, "general_GL": 1, "efficient_structure": "2x2_block_rotations", "condition_bound": 1.0, "multiplicity_family": 1},
        {"orbit": "O3", "name": "global_diagonal_scale", "scope": "global", "family": "diagonal", "replicates": ORBIT_REPLICATES, "orthogonal": 0, "general_GL": 1, "efficient_structure": "elementwise_scale_0.7_to_1.4", "condition_bound": 2.0, "multiplicity_family": 1},
        {"orbit": "O4", "name": "checkpoint_specific_diagonal_scale", "scope": "checkpoint", "family": "diagonal", "replicates": ORBIT_REPLICATES, "orthogonal": 0, "general_GL": 1, "efficient_structure": "elementwise_scale_0.7_to_1.4", "condition_bound": 2.0, "multiplicity_family": 1},
        {"orbit": "O5", "name": "global_well_conditioned_nonorthogonal", "scope": "global", "family": "nonorthogonal", "replicates": ORBIT_REPLICATES, "orthogonal": 0, "general_GL": 1, "efficient_structure": "invertible_2x2_upper_triangular_blocks", "condition_bound": 3.0, "multiplicity_family": 1},
        {"orbit": "O6", "name": "checkpoint_specific_well_conditioned_nonorthogonal", "scope": "checkpoint", "family": "nonorthogonal", "replicates": ORBIT_REPLICATES, "orthogonal": 0, "general_GL": 1, "efficient_structure": "invertible_2x2_upper_triangular_blocks", "condition_bound": 3.0, "multiplicity_family": 1},
        {"orbit": "O7", "name": "checkpoint_specific_signed_latent_permutation", "scope": "checkpoint", "family": "signed_permutation", "replicates": ORBIT_REPLICATES, "orthogonal": 1, "general_GL": 1, "efficient_structure": "latent_coordinate_permutation_and_sign", "condition_bound": 1.0, "multiplicity_family": 1},
    ]


def feature_registry() -> list[dict]:
    return [
        {"group": "G0", "name": "metadata_source_baseline", "source": "C75_F0", "strict_source": 1, "target_unlabeled": 0, "target_labels": 0, "function_invariant": 1, "orthogonal_invariant": 1, "coordinate_dependent": 0, "redundant_with_logits": 0, "qualification_candidate": 0},
        {"group": "G1S", "name": "strict_source_functional", "source": "C75_F1", "strict_source": 1, "target_unlabeled": 0, "target_labels": 0, "function_invariant": 1, "orthogonal_invariant": 1, "coordinate_dependent": 0, "redundant_with_logits": 0, "qualification_candidate": 0},
        {"group": "G1T", "name": "target_unlabeled_functional", "source": "C75_F3", "strict_source": 0, "target_unlabeled": 1, "target_labels": 0, "function_invariant": 1, "orthogonal_invariant": 1, "coordinate_dependent": 0, "redundant_with_logits": 0, "qualification_candidate": 0},
        {"group": "G2S", "name": "strict_source_Wz_functional", "source": "C75_source_Wz_summary", "strict_source": 1, "target_unlabeled": 0, "target_labels": 0, "function_invariant": 1, "orthogonal_invariant": 1, "coordinate_dependent": 0, "redundant_with_logits": 1, "qualification_candidate": 0},
        {"group": "G2T", "name": "target_unlabeled_Wz_functional", "source": "C75_target_Wz_summary", "strict_source": 0, "target_unlabeled": 1, "target_labels": 0, "function_invariant": 1, "orthogonal_invariant": 1, "coordinate_dependent": 0, "redundant_with_logits": 1, "qualification_candidate": 0},
        {"group": "G3S", "name": "strict_source_orthogonal_geometry", "source": "orbit_recomputed_C75_F2", "strict_source": 1, "target_unlabeled": 0, "target_labels": 0, "function_invariant": 0, "orthogonal_invariant": 1, "coordinate_dependent": 0, "redundant_with_logits": 0, "qualification_candidate": 1},
        {"group": "G3T", "name": "target_unlabeled_orthogonal_geometry", "source": "orbit_recomputed_C75_F4", "strict_source": 0, "target_unlabeled": 1, "target_labels": 0, "function_invariant": 0, "orthogonal_invariant": 1, "coordinate_dependent": 0, "redundant_with_logits": 0, "qualification_candidate": 1},
        {"group": "G4S", "name": "strict_source_fixed_coordinate_probes", "source": "8_SHA_locked_coordinate_probes", "strict_source": 1, "target_unlabeled": 0, "target_labels": 0, "function_invariant": 0, "orthogonal_invariant": 0, "coordinate_dependent": 1, "redundant_with_logits": 0, "qualification_candidate": 0},
        {"group": "G4T", "name": "target_unlabeled_fixed_coordinate_probes", "source": "8_SHA_locked_coordinate_probes", "strict_source": 0, "target_unlabeled": 1, "target_labels": 0, "function_invariant": 0, "orthogonal_invariant": 0, "coordinate_dependent": 1, "redundant_with_logits": 0, "qualification_candidate": 0},
        {"group": "G6", "name": "construction_label_positive_control", "source": "C75_F5", "strict_source": 0, "target_unlabeled": 0, "target_labels": 1, "function_invariant": 1, "orthogonal_invariant": 1, "coordinate_dependent": 0, "redundant_with_logits": 0, "qualification_candidate": 0},
    ]


def kernel_registry() -> list[dict]:
    rows = []
    for path in ("strict_source", "target_unlabeled"):
        for kernel in KERNEL_FAMILIES:
            for factor in BANDWIDTH_FACTORS:
                for statistic in ASSOCIATION_STATISTICS:
                    rows.append({
                        "path": path, "kernel": kernel, "bandwidth_factor": factor,
                        "statistic": statistic, "fold_scaling": "outer_training_fold_zscore",
                        "bandwidth_selection": "outer_training_fold_median_pairwise_distance",
                        "null_reselects_bandwidth": 1, "global_family": 1,
                    })
    return rows


def null_registry() -> list[dict]:
    return [
        {"null": "N1_target_block", "operation": "permute complete 24-candidate target feature blocks after locked candidate-key alignment", "tests": "pooled_identity", "required_for_candidate": 1},
        {"null": "N2_checkpoint_block", "operation": "permute checkpoint feature rows across targets within seed x level x candidate_order key", "tests": "checkpoint_identity", "required_for_candidate": 1},
        {"null": "N3_trajectory_preserving", "operation": "permute candidate feature rows within target x trajectory four-candidate cells", "tests": "C75_local_association", "required_for_candidate": 1},
        {"null": "N4_candidate_within_target", "operation": "permute candidate feature rows within each 24-candidate target", "tests": "within_target_association", "required_for_candidate": 1},
        {"null": "N5_identity_matched", "operation": "replace architecture block by covariance/dimension-matched Gaussian smooth features conditional on registered metadata", "tests": "nuisance_geometry", "required_for_candidate": 1},
        {"null": "N6_orbit_transformed", "operation": "checkpoint-specific O6 transform then N3 trajectory-preserving permutation", "tests": "orbit_conditioned_association", "required_for_candidate": 1},
    ]


def qualification_registry() -> list[dict]:
    gates = [
        ("functional_identity", "==", 1),
        ("orbit_robustness", "==", 1),
        ("association_effect", ">=", 0.02),
        ("association_bootstrap_lower", ">", 0.0),
        ("incremental_R2", ">=", 0.02),
        ("observed_above_nested_null_p95", "==", 1),
        ("global_max_stat_p", "<", 0.05),
        ("leave_target_median_increment", ">", 0.0),
        ("positive_targets", ">=", 7),
        ("material_actionability", "==", 1),
        ("not_redundant_with_logits", "==", 1),
        ("target_label_leakage", "==", 0),
    ]
    rows = []
    for candidate, path in (("G3S_strict_source", "strict_source"), ("G3T_target_unlabeled", "target_unlabeled")):
        for gate, operator, threshold in gates:
            rows.append({"candidate": candidate, "path": path, "gate": gate, "operator": operator, "threshold": threshold, "all_required": 1})
    return rows


def prepare_protocol() -> dict:
    timestamp = _now()
    if sha256(C75_PROTOCOL) != C75_PROTOCOL_SHA.read_text().strip():
        raise RuntimeError("C76 parent C75 protocol hash mismatch")
    c75_result = json.loads(C75_RESULT.read_text())
    if c75_result["final_gate"] != "T3_HO_REPRESENTATION_CAMPAIGN_NOT_JUSTIFIED":
        raise RuntimeError("C76 parent C75 final gate mismatch")
    t2 = _read_csv(C74_T2_UNITS)
    t3 = _read_csv(C74_T3_UNITS)
    if len(t2) != 216 or len(t3) != 1052 or any(row["z_Wz_generated_in_C74"] != "0" for row in t3):
        raise RuntimeError("C76 T2/T3 role replay failed")

    registries = {
        "orbit_transform_registry.csv": orbit_registry(),
        "functional_architecture_feature_registry.csv": feature_registry(),
        "kernel_bandwidth_registry.csv": kernel_registry(),
        "null_registry.csv": null_registry(),
        "t3_candidate_gate_registry.csv": qualification_registry(),
    }
    replay_paths = [
        C75_PROTOCOL, C75_RESULT, C75_REPORT, C75_ARTIFACT_MANIFEST,
        C75_RED_TEAM, C75_RELEVANCE, C75_RBF, C75_REDUNDANCY,
        C75_PROJECTION, C75_T3_DECISION, C74_PROTOCOL, C74_VIEW_MANIFEST,
        C74_T2_UNITS, C74_T3_UNITS,
    ]
    registries["c75_artifact_hash_lock.csv"] = [
        {"artifact": path.name, "path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in replay_paths
    ]
    for name, rows in registries.items():
        _write_csv(name, rows)
    locked_tables = {
        name: {"path": str(TABLE_DIR / name), "sha256": sha256(TABLE_DIR / name), "rows": len(rows), "size_bytes": (TABLE_DIR / name).stat().st_size}
        for name, rows in registries.items()
    }

    protocol = {
        "schema_version": "c76_representation_association_protocol_v1",
        "milestone": MILESTONE, "protocol_lock_timestamp_utc": timestamp,
        "parent_C75_result_commit": PARENT_COMMIT,
        "parent_C75_protocol_sha256": sha256(C75_PROTOCOL),
        "parent_C75_result_sha256": sha256(C75_RESULT),
        "locked_registry_tables": locked_tables,
        "status": "prospective_derived_outcome_protocol_over_known_T2;exploratory_not_independent_confirmation",
        "execution_boundary": {
            "forward_passes": False, "re_inference": False, "training": False,
            "GPU": False, "T3_HO_z_Wz_access": False, "BNCI2014_004": False,
            "seeds_3_4": False, "selector_or_checkpoint_artifacts": False,
            "raw_cache_in_git": False, "manuscript_drafting": False,
            "unregistered_high_dimensional_search": False,
        },
        "data_role": {
            "T2_units": 216, "T2_role": "exploratory_orbit_and_transportability_analysis",
            "T3_HO_units": 1052, "T3_HO_role": "untouched_new_variable_holdout",
            "T3_HO_access_in_C76": False, "same_label_oracle_access": False,
            "target_population_claim_allowed": False,
        },
        "orbit_contract": {
            "dimension": 800, "replicates_per_nonidentity_family": ORBIT_REPLICATES,
            "registry": str(TABLE_DIR / "orbit_transform_registry.csv"),
            "parameter_seed": RNG_SEED + 100, "checkpoint_seed_key": "SHA256(unit_id|orbit|replicate)",
            "row_convention": "z_prime=z@A.T;W_prime=W@inv(A)",
            "functional_tolerance_max_abs": FUNCTIONAL_IDENTITY_TOLERANCE,
            "prediction_disagreement_allowed": 0, "probability_tolerance_max_abs": FUNCTIONAL_IDENTITY_TOLERANCE,
            "global_transform_shared_exactly": True, "checkpoint_transform_independent_hash_locked": True,
            "orbit_robustness_gate": {
                "all_family_median_effect_signs_match_baseline": True,
                "all_family_median_absolute_effect_retention_min": 0.80,
                "all_family_median_absolute_effect_retention_max": 1.25,
                "all_family_median_candidate_kernel_order_spearman_min": 0.95,
            },
        },
        "feature_contract": {
            "registry": str(TABLE_DIR / "functional_architecture_feature_registry.csv"),
            "C75_F2_F4_recomputed_for_every_orbit": True,
            "coordinate_probes": "8 fixed SHA-seeded Rademacher directions; mean-z and W-row projections; no selection",
            "no_feature_selection": True, "no_outcome_adaptive_transform": True,
        },
        "association_contract": {
            "paths": {"strict_source": "G3S|G0+G1S", "target_unlabeled": "G3T|G0+G1S+G1T"},
            "outcome": "C75 continuous_joint_utility; unexplained residual is never called gauge",
            "kernel_registry": str(TABLE_DIR / "kernel_bandwidth_registry.csv"),
            "kernel_families": list(KERNEL_FAMILIES), "bandwidth_factors": list(BANDWIDTH_FACTORS),
            "statistics": list(ASSOCIATION_STATISTICS),
            "fold_scaling": "outer-training-fold mean/std only",
            "bandwidth": "outer-training-fold median pairwise L2 distance; recomputed inside every null",
            "global_max_stat_family_size": 24,
            "effect_materiality_min": 0.02,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "null_contract": {
            "registry": str(TABLE_DIR / "null_registry.csv"), "replicates": NULL_REPLICATES,
            "all_six_required_for_positive_candidate": True,
            "max_stat": "all 2 paths x 2 kernels x 3 bandwidths x 2 statistics within each null; candidate must pass every required null",
            "nested_scaling_bandwidth": True, "rng_seed": RNG_SEED + 200,
        },
        "topology_contract": {
            "levels": ["pooled", "within_target", "within_target_x_trajectory", "within_regime", "leave_target_out", "leave_trajectory_out", "cross_target", "cross_regime"],
            "conditioning": ["target_identity", "checkpoint_identity_kernel", "seed", "regime", "level", "candidate_order", "source_performance", "construction_competence"],
            "pooled_only_rule": "pooled significant and within-target nonsignificant -> identity_or_heterogeneity",
            "local_nontransport_rule": "within-target significant but nonlinear LOTO qualification fails",
        },
        "prediction_contract": {
            "outer": "leave-one-target-out", "secondary": "leave-one-trajectory-template-out",
            "model": "kernel ridge residual correction beyond path functional baseline",
            "kernels": list(KERNEL_FAMILIES), "bandwidth_factors": list(BANDWIDTH_FACTORS),
            "alpha_grid": list(KRR_ALPHAS),
            "inner_selection": "leave-one-training-target-out MSE; kernel/bandwidth/alpha nested",
            "primary_null": "N3 trajectory-preserving; full inner selection repeated inside null",
            "null_replicates": NULL_REPLICATES,
            "bootstrap": "target cluster then candidate units", "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "actionability_materiality": {
                "regret_route": "mean_regret_reduction>=0.02 AND target-bootstrap_CI_low>0 AND positive_targets>=7",
                "topk_route": "top3_increment>=2/9 AND joint_good_coverage_increment>=2/9 AND exact_target_sign_permutation_p<0.05",
                "either_route_passes": True,
            },
        },
        "qualification_contract": {
            "registry": str(TABLE_DIR / "t3_candidate_gate_registry.csv"),
            "all_gates_required": True,
            "if_none_pass": "do_not_create_C77;close_representation_branch_under_current_architecture_frozen_universe",
            "if_pass": "create_locked_C77_protocol_without_T3_access;separate_authorization_still_required",
        },
        "synthetic_contract": {
            "replicates": SYNTHETIC_REPLICATES, "targets": 9, "candidates_per_target": 24,
            "cases": ["S0_no_association", "S1_coordinate_artifact", "S2_pooled_identity", "S3_local_nonlinear_nontransport", "S4_factorization_invariant_endpoint", "S5_association_no_extreme_action", "S6_predictive_actionable"],
            "false_positive_target": 0.05, "seed": RNG_SEED + 300,
        },
        "taxonomy": {
            "primary": ["C76-A_RBF_association_collapses_under_blocked_orbit_controls", "C76-B_architecture_tied_coordinate_association_only", "C76-C_identity_or_heterogeneity_explains_association", "C76-D_local_nonlinear_measurement_nontransportable_nonactionable", "C76-E_factorization_invariant_incremental_candidate_for_T3_HO", "C76-F_protocol_cache_or_claim_blocker"],
            "secondary": [f"C76-S{index}" for index in range(1, 12)],
        },
        "final_gates": ["RBF_ASSOCIATION_COLLAPSES_UNDER_STRICT_CONTROLS", "ARCHITECTURE_TIED_ASSOCIATION_ONLY", "LOCAL_NONLINEAR_MEASUREMENT_NONTRANSPORTABLE", "FACTOR_INVARIANT_CANDIDATE_READY_FOR_T3_HO", "REPRESENTATION_BRANCH_SATURATED_NO_T3_HO", "CACHE_OR_PROTOCOL_REPAIR_REQUIRED"],
        "forbidden_claims": ["association_is_prediction", "association_is_actionability", "representation_origin", "target_gauge", "selector", "checkpoint_recommendation", "deployable", "target_population_generalization", "EEG_minimax_theorem", "manuscript_drafting"],
        "diagnostic_only_non_deployable": True,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROTOCOL_PATH.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    digest = sha256(PROTOCOL_PATH)
    PROTOCOL_SHA_PATH.write_text(digest + "\n")
    TIMING_PATH.write_text("\n".join([
        "# C76 - Protocol Timing Audit", "",
        f"- Protocol lock timestamp: `{timestamp}`",
        f"- Parent C75 result commit: `{PARENT_COMMIT[:7]}`",
        "- C76 association outcome computed before lock: `false`",
        "- C74/C75 external payload read for C76 before lock: `false`",
        "- T3-HO z/Wz accessed: `false`",
        "- Orbit families, kernels, statistics, nulls, materiality, transportability, multiplicity, and stop rules locked: `true`",
        "", "C76 remains exploratory over known T2 outcomes. This timing lock does not make it independent confirmation.",
    ]) + "\n")
    return {"protocol_sha256": digest, "timestamp": timestamp, "locked_tables": locked_tables}


if __name__ == "__main__":
    print(json.dumps(prepare_protocol(), indent=2, sort_keys=True))
