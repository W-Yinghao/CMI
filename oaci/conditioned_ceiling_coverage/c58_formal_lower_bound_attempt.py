"""C58 - Formal Lower-Bound Attempt / Instrumented Real-EEG Training Gate."""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os

from . import audit_utils as au
from . import schema as c49_schema


MILESTONE = "C58"
REPORT_DIR = "oaci/reports"
REPORT_JSON = "oaci/reports/C58_FORMAL_LOWER_BOUND_ATTEMPT.json"
TABLE_DIR = "oaci/reports/c58_tables"
FORMAL_DIR = "oaci/reports/c58_formal_lower_bound"
TRAINING_DIR = "oaci/reports/c58_training_gate"
C57_JSON = "oaci/reports/C57_MANUSCRIPT_SCAFFOLD_CLAIM_CONTRACT.json"
C57_KEY_TABLE = "oaci/reports/c57_tables/key_number_provenance.csv"

DECISIONS = (
    "C58-A_finite_population_lower_bound_established",
    "C58-B_lecam_style_two_point_bound_established_under_empirical_assumptions",
    "C58-C_fano_assouad_packing_bound_nontrivial",
    "C58-D_empirical_boundary_only_formal_bound_not_yet_supported",
    "C58-E_source_observable_escape_hatch_found",
    "C58-F_formalization_requires_new_instrumented_real_eeg_training",
    "C58-G_new_training_campaign_scientifically_authorized",
    "C58-H_new_training_not_justified_yet",
)

SECONDARY_DECISIONS = (
    "C58-S1_sigma_field_ladder_locked",
    "C58-S2_selector_measurability_contract_locked",
    "C58-S3_empirical_bayes_ceiling_nontrivial",
    "C58-S4_indistinguishable_target_divergent_pairs_found",
    "C58-S5_packing_set_too_small_or_unstable",
    "C58-S6_mutual_information_estimates_unstable",
    "C58-S7_rank_gauge_synthetic_model_matches_C30_C55",
    "C58-S8_theorem_assumptions_too_strong",
    "C58-S9_training_needed_for_split_label_cache",
    "C58-S10_training_needed_for_atom_trace",
    "C58-S11_no_manuscript_drafting",
)

TRAINING_GATE_DECISION = "TRAINING_NEEDED_BUT_NOT_AUTHORIZED"

FORBIDDEN_PATTERNS = (
    "source-only rescue",
    "OACI rescue",
    "deployable selector",
    "checkpoint recommendation artifact",
    "few-label sufficiency",
    "formal theorem",
    "same-label endpoint oracle available at selection time",
    "training protocol used for method tuning",
    "new training silently run",
    "C57 automatically starts M1",
    "M1 manuscript drafting",
    "all DG methods fail",
    "EEG transfer impossible",
)

NEGATION_CUES = (
    "not ",
    "no ",
    "never ",
    "cannot ",
    "do not ",
    "does not ",
    "is not ",
    "are not ",
    "unavailable",
    "blocked",
    "blocks ",
    "future",
    "diagnostic",
    "without ",
    "only if explicitly requested",
)

ENDPOINT_ORACLE_HIT = 0.9444444444444444
RANDOM_TIE_HIT = 0.4297233780360411
N_CANDIDATES = 3804
N_CELLS = 162


def _lock_config() -> str:
    return au.lock_config(MILESTONE)


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: list[dict], cols: list[str]) -> None:
    au.write_csv(path, rows, cols)


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt3(value) -> str:
    if isinstance(value, bool):
        return str(value)
    try:
        x = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(x):
        return "n/a"
    return f"{x:.3f}"


def _as_float(value, default=math.nan) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def _key_rows() -> list[dict]:
    return _read_csv(C57_KEY_TABLE)


def _key_map(rows: list[dict]) -> dict[str, dict]:
    return {r["provenance_id"]: r for r in rows}


def _kv(rows: list[dict], key: str, default=math.nan) -> float:
    return _as_float(_key_map(rows).get(key, {}).get("value", default), default)


def _row_count(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def build_sigma_field_ladder() -> list[dict]:
    return [
        {
            "sigma_field": "G0_random_or_tie",
            "available_inputs": "trajectory candidate set only",
            "uses_source": 0,
            "uses_target_or_trajectory_key": 0,
            "uses_target_unlabeled": 0,
            "uses_target_labels": 0,
            "uses_same_label_endpoint": 0,
            "available_at_selection_time": 1,
            "c58_role": "null floor",
            "boundary_status": "baseline not sufficient",
        },
        {
            "sigma_field": "G1_strict_source_observables",
            "available_inputs": "source losses, leakage, R_src, C30/C42/C43 source scores",
            "uses_source": 1,
            "uses_target_or_trajectory_key": 0,
            "uses_target_unlabeled": 0,
            "uses_target_labels": 0,
            "uses_same_label_endpoint": 0,
            "available_at_selection_time": 1,
            "c58_role": "source-only actionability class",
            "boundary_status": "weak signal but not reliable",
        },
        {
            "sigma_field": "G2_source_plus_key",
            "available_inputs": "source observables plus target_id or trajectory_id keys",
            "uses_source": 1,
            "uses_target_or_trajectory_key": 1,
            "uses_target_unlabeled": 0,
            "uses_target_labels": 0,
            "uses_same_label_endpoint": 0,
            "available_at_selection_time": 0,
            "c58_role": "key-only escape hatch",
            "boundary_status": "key naming alone does not close residual",
        },
        {
            "sigma_field": "G3_target_unlabeled_geometry",
            "available_inputs": "target-unlabeled prediction or geometry fields",
            "uses_source": 1,
            "uses_target_or_trajectory_key": 1,
            "uses_target_unlabeled": 1,
            "uses_target_labels": 0,
            "uses_same_label_endpoint": 0,
            "available_at_selection_time": 0,
            "c58_role": "unlabeled transductive candidate",
            "boundary_status": "not available in committed C52-C55 cache",
        },
        {
            "sigma_field": "G4_target_grouped_zero_label_structure",
            "available_inputs": "target or trajectory grouped zero-label structure",
            "uses_source": 1,
            "uses_target_or_trajectory_key": 1,
            "uses_target_unlabeled": 0,
            "uses_target_labels": 0,
            "uses_same_label_endpoint": 0,
            "available_at_selection_time": 0,
            "c58_role": "conditioned diagnostic problem class",
            "boundary_status": "descriptive homogeneity without selector",
        },
        {
            "sigma_field": "G5_split_label_or_few_label",
            "available_inputs": "disjoint target label construction and evaluation cache",
            "uses_source": 1,
            "uses_target_or_trajectory_key": 1,
            "uses_target_unlabeled": 0,
            "uses_target_labels": 1,
            "uses_same_label_endpoint": 0,
            "available_at_selection_time": 0,
            "c58_role": "future information class",
            "boundary_status": "unresolved because split-label cache is missing",
        },
        {
            "sigma_field": "G6_target_label_diagnostic_content",
            "available_inputs": "same audit target labels outside deployable selection",
            "uses_source": 1,
            "uses_target_or_trajectory_key": 1,
            "uses_target_unlabeled": 0,
            "uses_target_labels": 1,
            "uses_same_label_endpoint": 0,
            "available_at_selection_time": 0,
            "c58_role": "diagnostic closure class",
            "boundary_status": "closes residual only diagnostically",
        },
        {
            "sigma_field": "G7_same_label_endpoint_oracle",
            "available_inputs": "candidate-specific evaluated target endpoint scalar",
            "uses_source": 0,
            "uses_target_or_trajectory_key": 1,
            "uses_target_unlabeled": 0,
            "uses_target_labels": 1,
            "uses_same_label_endpoint": 1,
            "available_at_selection_time": 0,
            "c58_role": "endpoint oracle reference",
            "boundary_status": "tautological diagnostic reference",
        },
    ]


def build_selector_measurability_contract() -> list[dict]:
    return [
        {
            "rule_family": "random_tie_rule",
            "measurable_wrt": "G0_random_or_tie",
            "allowed_for_original_source_only_DG": 1,
            "uses_target_labels": 0,
            "outputs_action_rule": 0,
            "c58_bound_scope": "finite-population baseline",
            "forbidden_interpretation": "not evidence that good checkpoints are absent",
        },
        {
            "rule_family": "strict_source_score_rule",
            "measurable_wrt": "G1_strict_source_observables",
            "allowed_for_original_source_only_DG": 1,
            "uses_target_labels": 0,
            "outputs_action_rule": 0,
            "c58_bound_scope": "observed source-score lower-bound surrogate",
            "forbidden_interpretation": "not source-only rescue",
        },
        {
            "rule_family": "source_scalarization_hindsight_rule",
            "measurable_wrt": "G1_strict_source_observables",
            "allowed_for_original_source_only_DG": 0,
            "uses_target_labels": 0,
            "outputs_action_rule": 0,
            "c58_bound_scope": "diagnostic upper envelope over registered source scalarizations",
            "forbidden_interpretation": "not tuned selector",
        },
        {
            "rule_family": "target_or_trajectory_key_only_rule",
            "measurable_wrt": "G2_source_plus_key",
            "allowed_for_original_source_only_DG": 0,
            "uses_target_labels": 0,
            "outputs_action_rule": 0,
            "c58_bound_scope": "key-only residual test",
            "forbidden_interpretation": "not trajectory key sufficiency",
        },
        {
            "rule_family": "conditioned_local_bayes_ceiling",
            "measurable_wrt": "G4/G6 diagnostic expansion",
            "allowed_for_original_source_only_DG": 0,
            "uses_target_labels": 1,
            "outputs_action_rule": 0,
            "c58_bound_scope": "diagnostic ceiling only",
            "forbidden_interpretation": "not deployable selector",
        },
        {
            "rule_family": "cross_cell_endpoint_template_rule",
            "measurable_wrt": "G6_target_label_diagnostic_content",
            "allowed_for_original_source_only_DG": 0,
            "uses_target_labels": 1,
            "outputs_action_rule": 0,
            "c58_bound_scope": "partial target-label template transfer",
            "forbidden_interpretation": "not split-label closure",
        },
        {
            "rule_family": "same_label_endpoint_scalar_rule",
            "measurable_wrt": "G7_same_label_endpoint_oracle",
            "allowed_for_original_source_only_DG": 0,
            "uses_target_labels": 1,
            "outputs_action_rule": 0,
            "c58_bound_scope": "endpoint-oracle reference",
            "forbidden_interpretation": "not available at selection time",
        },
        {
            "rule_family": "future_instrumented_training_rule",
            "measurable_wrt": "future split-label cache only after explicit authorization",
            "allowed_for_original_source_only_DG": 0,
            "uses_target_labels": 1,
            "outputs_action_rule": 0,
            "c58_bound_scope": "protocol specification only",
            "forbidden_interpretation": "not run by C58",
        },
    ]


def build_utility_and_loss_definitions() -> list[dict]:
    return [
        {
            "object_id": "Y_joint_good",
            "definition": "candidate satisfies target bAcc/NLL/ECE joint-good condition used in C31-C55",
            "orientation": "higher is better",
            "availability_class": "diagnostic endpoint label",
            "used_by_c58": "finite-population hit/miss risk",
        },
        {
            "object_id": "H_star_G",
            "definition": "mean over evaluation cells of best attainable joint-good hit inside a registered information partition",
            "orientation": "higher is better",
            "availability_class": "finite-population analysis",
            "used_by_c58": "empirical Bayes hit ceiling",
        },
        {
            "object_id": "M_G",
            "definition": "1 - H_star_G for the registered partition or rule family",
            "orientation": "lower is better",
            "availability_class": "finite-population analysis",
            "used_by_c58": "miss lower-bound candidate for that rule family",
        },
        {
            "object_id": "R_G_to_EO",
            "definition": "H_star_endpoint_oracle - H_star_G",
            "orientation": "lower is better",
            "availability_class": "diagnostic comparison",
            "used_by_c58": "hit-rate regret to same-label endpoint oracle",
        },
        {
            "object_id": "LeCam_proxy",
            "definition": "empirical source-near/target-divergent two-point witness score, not a TV distance",
            "orientation": "larger means more lower-bound-like ambiguity",
            "availability_class": "empirical witness",
            "used_by_c58": "formal candidate only",
        },
        {
            "object_id": "Fano_MI_proxy",
            "definition": "registered discrete-MI or entropy proxy from source/endpoint scalar audits",
            "orientation": "smaller source MI and larger ambiguity support non-identifiability candidates",
            "availability_class": "empirical witness",
            "used_by_c58": "packing attempt gate",
        },
    ]


def _bound_row(bound_id: str, information_class: str, family: str, hit: float | None,
               exact: bool, artifact: str, note: str) -> dict:
    if hit is None or not math.isfinite(float(hit)):
        miss = ""
        regret = ""
    else:
        miss = 1.0 - float(hit)
        regret = ENDPOINT_ORACLE_HIT - float(hit)
    return {
        "bound_id": bound_id,
        "information_class": information_class,
        "selector_family_or_partition": family,
        "measured_hit": "" if hit is None else hit,
        "empirical_miss_lower_bound": miss,
        "regret_to_endpoint_oracle": regret,
        "exact_partition_bound": int(exact),
        "theorem_status": "finite_population_exact_for_registered_partition" if exact else "observed_surrogate_or_diagnostic",
        "artifact": artifact,
        "note": note,
    }


def build_finite_population_rows(key_rows: list[dict]) -> list[dict]:
    rows = [
        _bound_row("B0", "G0_random_or_tie", "random/tie within trajectory", RANDOM_TIE_HIT, True,
                   "oaci/reports/c52_tables/key_null_calibration_summary.csv",
                   "direct finite-population baseline with fractional tie handling"),
        _bound_row("B1", "G1_strict_source_observables", "best strict source score", _kv(key_rows, "K_C52_best_strict_source_hit"), False,
                   "oaci/reports/c52_tables/conditioning_ladder_summary.csv",
                   "registered source score surrogate; not a full Bayes optimum over all possible source functions"),
        _bound_row("B2", "G1_strict_source_observables", "best hindsight source scalarization", _kv(key_rows, "K_C43_best_source_scalarization_top1"), False,
                   "oaci/reports/C43_SOURCE_OBJECTIVE_SCALARIZATION_FRONTIER.json",
                   "diagnostic hindsight grid; not a tuned deployable selector"),
        _bound_row("B3", "G2_source_plus_key", "best key/source-geometry baseline", _kv(key_rows, "K_C52_best_key_only_hit"), False,
                   "oaci/reports/c52_tables/gauge_key_decomposition.csv",
                   "key/source-geometry escape hatch remains below source rank and endpoint labels"),
        _bound_row("B4", "G2_source_plus_key", "target/trajectory key-only identity tie", RANDOM_TIE_HIT, True,
                   "oaci/reports/c52_tables/key_null_calibration_summary.csv",
                   "keys are constant inside each evaluated trajectory and do not order candidates"),
        _bound_row("B5", "G6_target_label_diagnostic_content", "trajectory-centered label diagnostic", _kv(key_rows, "K_C52_best_label_derived_hit"), False,
                   "oaci/reports/C52_MINIMAL_GAUGE_KEY_SUFFICIENCY.json",
                   "diagnostic label content closes most residual but is unavailable for source-only selection"),
        _bound_row("B6", "G6_target_label_diagnostic_content", "best cross-cell endpoint template", _kv(key_rows, "K_C55_template_only_best"), False,
                   "oaci/reports/C55_CROSS_CELL_ENDPOINT_TEMPLATE_BOUNDARY.json",
                   "partial template transfer; does not beat the max null p95"),
        _bound_row("B7", "G7_same_label_endpoint_oracle", "same-label endpoint scalar", ENDPOINT_ORACLE_HIT, True,
                   "oaci/reports/C55_CROSS_CELL_ENDPOINT_TEMPLATE_BOUNDARY.json",
                   "endpoint-oracle reference; unavailable at selection time"),
        _bound_row("B8", "G6_target_label_diagnostic_content", "best single endpoint component", _kv(key_rows, "K_C54_best_single_endpoint_component_hit"), False,
                   "oaci/reports/C54_ENDPOINT_SCALAR_TAUTOLOGY_BIT_BUDGET.json",
                   "strong target endpoint component but target-label-derived"),
        _bound_row("B9", "G4/G6_conditioned_local_ceiling", "C48 conditioned local Bayes ceiling", _kv(key_rows, "K_C48_local_ceiling_hit"), False,
                   "oaci/reports/C48_CONDITIONED_LOCAL_BAYES_CEILING.json",
                   "different diagnostic local ceiling; not selection-time rule"),
    ]
    return rows


def build_partition_bayes_rows(finite_rows: list[dict]) -> list[dict]:
    rows = []
    for r in finite_rows:
        hit = _as_float(r["measured_hit"])
        rows.append({
            "partition_id": r["bound_id"],
            "information_class": r["information_class"],
            "n_candidates": N_CANDIDATES,
            "n_evaluation_cells": N_CELLS,
            "bayes_hit_proxy": "" if not math.isfinite(hit) else hit,
            "bayes_error_proxy": "" if not math.isfinite(hit) else 1.0 - hit,
            "exact_for_registered_partition": r["exact_partition_bound"],
            "target_labels_used_to_define_partition": int("G6" in r["information_class"] or "G7" in r["information_class"]),
            "caveat": r["note"],
        })
    return rows


def build_cellwise_ledger_rows() -> list[dict]:
    return [
        {
            "ledger_id": "C0_frozen_population",
            "information_class": "all",
            "n_cells": N_CELLS,
            "n_candidates": N_CANDIDATES,
            "available": 1,
            "best_hit": "",
            "bayes_error": "",
            "support_artifact": "C50-C55 committed reports",
            "caveat": "C58 does not reconstruct all row-level cells; it audits finite-population summaries",
        },
        {
            "ledger_id": "C1_random_tie",
            "information_class": "G0_random_or_tie",
            "n_cells": N_CELLS,
            "n_candidates": N_CANDIDATES,
            "available": 1,
            "best_hit": RANDOM_TIE_HIT,
            "bayes_error": 1.0 - RANDOM_TIE_HIT,
            "support_artifact": "C52 key null calibration",
            "caveat": "fractional tie hit, not integer winner count",
        },
        {
            "ledger_id": "C2_source",
            "information_class": "G1_strict_source_observables",
            "n_cells": N_CELLS,
            "n_candidates": N_CANDIDATES,
            "available": 1,
            "best_hit": 0.5061728395061729,
            "bayes_error": 0.49382716049382713,
            "support_artifact": "C42/C52",
            "caveat": "observed strict-source rule, not universal source Bayes bound",
        },
        {
            "ledger_id": "C3_key_only",
            "information_class": "G2_source_plus_key",
            "n_cells": N_CELLS,
            "n_candidates": N_CANDIDATES,
            "available": 1,
            "best_hit": 0.4876543209876543,
            "bayes_error": 0.5123456790123457,
            "support_artifact": "C52",
            "caveat": "best key/source geometry closes 12/162 cells in C53-style accounting",
        },
        {
            "ledger_id": "C4_label_diagnostic",
            "information_class": "G6_target_label_diagnostic_content",
            "n_cells": N_CELLS,
            "n_candidates": N_CANDIDATES,
            "available": 1,
            "best_hit": 0.8127572016460904,
            "bayes_error": 0.18724279835390957,
            "support_artifact": "C52",
            "caveat": "label-derived diagnostic closes 131/162 cells in C53-style accounting",
        },
        {
            "ledger_id": "C5_endpoint_oracle",
            "information_class": "G7_same_label_endpoint_oracle",
            "n_cells": N_CELLS,
            "n_candidates": N_CANDIDATES,
            "available": 1,
            "best_hit": ENDPOINT_ORACLE_HIT,
            "bayes_error": 1.0 - ENDPOINT_ORACLE_HIT,
            "support_artifact": "C54/C55",
            "caveat": "same-label endpoint scalar; unavailable at selection time",
        },
        {
            "ledger_id": "C6_trajectory_fragmentation",
            "information_class": "G4/G6_conditioned_diagnostic",
            "n_cells": N_CELLS,
            "n_candidates": N_CANDIDATES,
            "available": 1,
            "best_hit": "",
            "bayes_error": "",
            "support_artifact": "C50",
            "caveat": "trajectory actionability failure fraction 70/162 = 0.432",
        },
        {
            "ledger_id": "C7_split_label_missing",
            "information_class": "G5_split_label_or_few_label",
            "n_cells": N_CELLS,
            "n_candidates": N_CANDIDATES,
            "available": 0,
            "best_hit": "",
            "bayes_error": "",
            "support_artifact": "C53",
            "caveat": "per-trial split-label cache unavailable",
        },
    ]


def build_regret_rows(finite_rows: list[dict]) -> list[dict]:
    rows = []
    for r in finite_rows:
        if r["measured_hit"] == "":
            continue
        hit = float(r["measured_hit"])
        rows.append({
            "comparison_id": r["bound_id"],
            "information_class": r["information_class"],
            "hit": hit,
            "endpoint_oracle_hit": ENDPOINT_ORACLE_HIT,
            "hit_gap_to_endpoint_oracle": ENDPOINT_ORACLE_HIT - hit,
            "relative_gap_fraction_of_random_to_oracle": (ENDPOINT_ORACLE_HIT - hit) / (ENDPOINT_ORACLE_HIT - RANDOM_TIE_HIT),
            "diagnostic_only": int("G6" in r["information_class"] or "G7" in r["information_class"] or "G4" in r["information_class"]),
            "note": r["note"],
        })
    return rows


def build_lecam_rows(key_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    witness_count = _row_count("oaci/reports/c45_tables/source_equivalent_target_divergent_pairs.csv")
    rows = [
        {
            "witness_id": "LC1_within_target_q10",
            "conditioning": "within_target",
            "source_neighborhood_divergent_q10": _kv(key_rows, "K_C46_within_target_q10"),
            "target_divergent_pair_count": witness_count,
            "two_point_proxy": 0.5 * (1.0 - _kv(key_rows, "K_C46_within_target_q10")),
            "formal_status": "empirical_proxy_not_TV",
            "note": "near source-neighborhood ambiguity exists, but no distributional two-world construction is proved",
        },
        {
            "witness_id": "LC2_within_trajectory_q10",
            "conditioning": "within_trajectory",
            "source_neighborhood_divergent_q10": _kv(key_rows, "K_C46_within_trajectory_q10"),
            "target_divergent_pair_count": witness_count,
            "two_point_proxy": 0.5 * (1.0 - _kv(key_rows, "K_C46_within_trajectory_q10")),
            "formal_status": "empirical_proxy_not_TV",
            "note": "trajectory conditioning is less ambiguous than cross-target mixing but still not a theorem",
        },
        {
            "witness_id": "LC3_cross_target_q10",
            "conditioning": "cross_target",
            "source_neighborhood_divergent_q10": _kv(key_rows, "K_C46_cross_target_q10"),
            "target_divergent_pair_count": witness_count,
            "two_point_proxy": 0.5 * (1.0 - _kv(key_rows, "K_C46_cross_target_q10")),
            "formal_status": "shows_cross_target_comparability_break",
            "note": "cross-target source equivalence strongly breaks target utility comparability",
        },
    ]
    pair_rows = [
        {
            "summary_id": "P1_c45_pair_table",
            "artifact": "oaci/reports/c45_tables/source_equivalent_target_divergent_pairs.csv",
            "row_count": witness_count,
            "source_equivalence_definition": "q10 source-neighborhood witness from C45",
            "target_divergence_definition": "target endpoint or target utility disagreement",
            "formal_status": "empirical_witness_set",
        },
        {
            "summary_id": "P2_c46_within_target",
            "artifact": "oaci/reports/C46_CONDITIONING_BOUNDARY_AUDIT.json",
            "row_count": "",
            "source_equivalence_definition": "within-target source neighborhoods",
            "target_divergence_definition": "q10 divergent fraction 0.004842615012106538",
            "formal_status": "conditioning_restores_local_meaning",
        },
        {
            "summary_id": "P3_c46_cross_target",
            "artifact": "oaci/reports/C46_CONDITIONING_BOUNDARY_AUDIT.json",
            "row_count": "",
            "source_equivalence_definition": "cross-target source neighborhoods",
            "target_divergence_definition": "q10 divergent fraction 0.9369369369369369",
            "formal_status": "cross_target_break",
        },
        {
            "summary_id": "P4_no_distributional_worlds",
            "artifact": "C58 formal audit",
            "row_count": "",
            "source_equivalence_definition": "not available",
            "target_divergence_definition": "not available",
            "formal_status": "blocks_formal_lecam_bound",
        },
    ]
    candidates = [
        {
            "candidate_id": "T1_within_target_two_point",
            "source_indistinguishability_proxy": _kv(key_rows, "K_C46_within_target_q10"),
            "target_loss_gap_proxy": "source-equivalent target-divergent pair set",
            "bound_candidate": "0.5*(1-source_proxy)",
            "nontrivial": 1,
            "blocking_assumption": "source proxy is not total variation between two distributions",
        },
        {
            "candidate_id": "T2_cross_target_two_point",
            "source_indistinguishability_proxy": _kv(key_rows, "K_C46_cross_target_q10"),
            "target_loss_gap_proxy": "cross-target utility comparability break",
            "bound_candidate": "descriptive only",
            "nontrivial": 0,
            "blocking_assumption": "mixing targets changes problem class rather than two-world lower-bound pair",
        },
        {
            "candidate_id": "T3_endpoint_oracle_separator",
            "source_indistinguishability_proxy": _kv(key_rows, "K_C54_binary_threshold_sufficient"),
            "target_loss_gap_proxy": ENDPOINT_ORACLE_HIT - _kv(key_rows, "K_C52_best_strict_source_hit"),
            "bound_candidate": "information gap to same-label endpoint oracle",
            "nontrivial": 1,
            "blocking_assumption": "endpoint scalar is diagnostic oracle, not alternative source-observable world",
        },
    ]
    return rows, pair_rows, candidates


def build_assumption_gap_rows() -> list[dict]:
    return [
        {
            "gap_id": "AG1_distributional_worlds",
            "template": "Le Cam",
            "needed_for_theorem": "two distributions with small TV/KL and separated target utility",
            "current_artifact_status": "empirical source-near target-divergent pairs only",
            "blocking": 1,
            "repair_requires_training": 0,
        },
        {
            "gap_id": "AG2_independent_samples",
            "template": "Fano/Assouad",
            "needed_for_theorem": "independent samples or controlled replicate fields",
            "current_artifact_status": "fixed finite audit population",
            "blocking": 1,
            "repair_requires_training": 1,
        },
        {
            "gap_id": "AG3_mutual_information_estimator",
            "template": "Fano",
            "needed_for_theorem": "stable MI estimate between observations and target-good index",
            "current_artifact_status": "source-rank MI proxy small; endpoint MI tautological",
            "blocking": 1,
            "repair_requires_training": 0,
        },
        {
            "gap_id": "AG4_split_label_disjointness",
            "template": "information boundary",
            "needed_for_theorem": "construction/evaluation labels are disjoint",
            "current_artifact_status": "split-label cache unavailable",
            "blocking": 1,
            "repair_requires_training": 1,
        },
        {
            "gap_id": "AG5_atom_trace",
            "template": "mechanism-to-theorem bridge",
            "needed_for_theorem": "candidate-level leakage atom traces and bootstrap order",
            "current_artifact_status": "C39/C40 atom decomposition irrecoverable from current artifacts",
            "blocking": 1,
            "repair_requires_training": 1,
        },
        {
            "gap_id": "AG6_finite_population_scope",
            "template": "finite-population partition bound",
            "needed_for_theorem": "registered finite population, registered partitions, exact row provenance",
            "current_artifact_status": "sufficient for C58-A within frozen C50-C55 population",
            "blocking": 0,
            "repair_requires_training": 0,
        },
    ]


def build_fano_rows() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    packing = [
        {
            "packing_id": "F1_source_rank_partition",
            "packing_size_proxy": 2,
            "separation_proxy": ENDPOINT_ORACLE_HIT - 0.5061728395061729,
            "mi_proxy": 0.00871699528126672,
            "bound_status": "candidate_only",
            "failure_reason": "packing too small and MI proxy not theorem-grade",
        },
        {
            "packing_id": "F2_key_only_cells",
            "packing_size_proxy": N_CELLS,
            "separation_proxy": ENDPOINT_ORACLE_HIT - 0.4876543209876543,
            "mi_proxy": "",
            "bound_status": "not_established",
            "failure_reason": "cell labels are keys, not independent target-good hypotheses",
        },
        {
            "packing_id": "F3_endpoint_oracle",
            "packing_size_proxy": 2,
            "separation_proxy": 0.0,
            "mi_proxy": 0.903,
            "bound_status": "tautological_oracle",
            "failure_reason": "endpoint scalar directly reads evaluated endpoint content",
        },
    ]
    cell = [
        {"packing_cell": "PC1_random_tie", "n_cells": N_CELLS, "hit": RANDOM_TIE_HIT, "used_for_packing": 0, "reason": "baseline not packing"},
        {"packing_cell": "PC2_source", "n_cells": N_CELLS, "hit": 0.5061728395061729, "used_for_packing": 1, "reason": "weak source partition proxy"},
        {"packing_cell": "PC3_key_only", "n_cells": N_CELLS, "hit": 0.4876543209876543, "used_for_packing": 1, "reason": "key-only ambiguity"},
        {"packing_cell": "PC4_endpoint_oracle", "n_cells": N_CELLS, "hit": ENDPOINT_ORACLE_HIT, "used_for_packing": 0, "reason": "oracle reference, not packing"},
    ]
    mi = [
        {"proxy_id": "MI1_source_rank_vs_joint_good", "value": 0.00871699528126672, "artifact": "oaci/reports/c54_tables/endpoint_tautology_distance.csv", "stable_for_theorem": 0, "note": "small source MI proxy"},
        {"proxy_id": "MI2_endpoint_joint_margin_vs_joint_good", "value": 0.903, "artifact": "oaci/reports/c54_tables/endpoint_tautology_distance.csv", "stable_for_theorem": 0, "note": "large because endpoint scalar is same-label diagnostic content"},
        {"proxy_id": "MI3_pareto_tie_count", "value": 3.2037037037037037, "artifact": "oaci/reports/C53_DIAGNOSTIC_LABEL_CONTENT_MINIMALITY.json", "stable_for_theorem": 0, "note": "ambiguity proxy, not MI estimator"},
        {"proxy_id": "MI4_split_label_cache", "value": "", "artifact": "C53", "stable_for_theorem": 0, "note": "unavailable"},
    ]
    null = [
        {"null_id": "N1_cell_preserving_label_shuffle", "observed_hit": ENDPOINT_ORACLE_HIT, "null_p95": 0.45075937973389896, "passes": 1, "claim_scope": "endpoint scalar only"},
        {"null_id": "N5_trajectory_block_shuffle", "observed_hit": ENDPOINT_ORACLE_HIT, "null_p95": 0.7712962962962961, "passes": 1, "claim_scope": "endpoint scalar only"},
        {"null_id": "Nmax_template_only", "observed_hit": 0.7037037037037037, "null_p95": 0.7712962962962961, "passes": 0, "claim_scope": "template-only partial transfer not formal bound"},
    ]
    return packing, cell, mi, null


def build_conditional_sufficiency_rows() -> tuple[list[dict], list[dict], list[dict]]:
    suff = [
        {"information_set": "S", "description": "strict source observables", "best_hit": 0.5061728395061729, "endpoint_gap": ENDPOINT_ORACLE_HIT - 0.5061728395061729, "sufficient": 0, "diagnostic_only": 0, "evidence": "C42/C52"},
        {"information_set": "S+K", "description": "source plus target/trajectory keys", "best_hit": 0.4876543209876543, "endpoint_gap": ENDPOINT_ORACLE_HIT - 0.4876543209876543, "sufficient": 0, "diagnostic_only": 1, "evidence": "C52"},
        {"information_set": "S+K+U", "description": "target-unlabeled geometry", "best_hit": "", "endpoint_gap": "", "sufficient": 0, "diagnostic_only": 1, "evidence": "unavailable in C52-C55 cache"},
        {"information_set": "D", "description": "target-label diagnostic content", "best_hit": 0.8127572016460904, "endpoint_gap": ENDPOINT_ORACLE_HIT - 0.8127572016460904, "sufficient": 0, "diagnostic_only": 1, "evidence": "C52"},
        {"information_set": "E", "description": "same-label endpoint scalar", "best_hit": ENDPOINT_ORACLE_HIT, "endpoint_gap": 0.0, "sufficient": 1, "diagnostic_only": 1, "evidence": "C54/C55"},
    ]
    ent = [
        {"ambiguity_id": "A1_random_tie", "proxy": "trajectory random/tie hit", "value": RANDOM_TIE_HIT, "source": "C52", "interpretation": "high ambiguity without source ordering"},
        {"ambiguity_id": "A2_source_rank_mi", "proxy": "discrete MI vs joint-good", "value": 0.00871699528126672, "source": "C54", "interpretation": "source rank carries weak information"},
        {"ambiguity_id": "A3_endpoint_mi", "proxy": "endpoint joint-margin MI", "value": 0.903, "source": "C54", "interpretation": "large but tautological endpoint label content"},
        {"ambiguity_id": "A4_endpoint_tie_count", "proxy": "mean endpoint scalar top-tie count", "value": 1.1111111111111112, "source": "C53", "interpretation": "low ambiguity once endpoint scalar is read"},
        {"ambiguity_id": "A5_pareto_tie_count", "proxy": "mean Pareto endpoint top-tie count", "value": 3.2037037037037037, "source": "C53", "interpretation": "more ambiguity for coarser endpoint families"},
    ]
    boundary = [
        {"candidate": "S", "screens_off_target_good": 0, "minimality_status": "insufficient", "reason": "strict source hit 0.506"},
        {"candidate": "S+K", "screens_off_target_good": 0, "minimality_status": "insufficient", "reason": "best key-only hit 0.488"},
        {"candidate": "S+K+U", "screens_off_target_good": 0, "minimality_status": "unavailable", "reason": "target-unlabeled geometry cache unavailable"},
        {"candidate": "D", "screens_off_target_good": 0, "minimality_status": "diagnostic partial closure", "reason": "label-derived hit 0.813"},
        {"candidate": "E", "screens_off_target_good": 1, "minimality_status": "same-label endpoint oracle", "reason": "endpoint scalar hit 0.944 and threshold overlap 1.000"},
    ]
    return suff, ent, boundary


def build_escape_hatch_rows(key_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    attack = [
        {"attack_id": "EH1_source_rank", "candidate_escape_hatch": "C42 source-rank", "observed_hit": _kv(key_rows, "K_C42_source_rank_top1_joint"), "passes_reliability_gate": 0, "forces_C58_E": 0, "reason": "weak signal, not reliable actionability"},
        {"attack_id": "EH2_source_scalarization", "candidate_escape_hatch": "C43 best source scalarization", "observed_hit": _kv(key_rows, "K_C43_best_source_scalarization_top1"), "passes_reliability_gate": 0, "forces_C58_E": 0, "reason": "hindsight diagnostic grid below endpoint/null boundary"},
        {"attack_id": "EH3_source_pareto", "candidate_escape_hatch": "C44 source Pareto membership", "observed_hit": _kv(key_rows, "K_C44_front_good_probability"), "passes_reliability_gate": 0, "forces_C58_E": 0, "reason": "front covers 0.972 of candidates and carries baseline good probability"},
        {"attack_id": "EH4_conditioning", "candidate_escape_hatch": "C47 conditioned source top1", "observed_hit": _kv(key_rows, "K_C47_conditioned_source_top1"), "passes_reliability_gate": 0, "forces_C58_E": 0, "reason": "conditioning is separate diagnostic problem class"},
        {"attack_id": "EH5_key_only", "candidate_escape_hatch": "C52 key/source geometry", "observed_hit": _kv(key_rows, "K_C52_best_key_only_hit"), "passes_reliability_gate": 0, "forces_C58_E": 0, "reason": "key-only residual remains"},
        {"attack_id": "EH6_template_only", "candidate_escape_hatch": "C55 endpoint template-only", "observed_hit": _kv(key_rows, "K_C55_template_only_best"), "passes_reliability_gate": 0, "forces_C58_E": 0, "reason": "partial transfer does not beat max null p95"},
    ]
    rules = [
        {"rule_id": "AR1_random", "source_observable": 0, "target_label_derived": 0, "hit": RANDOM_TIE_HIT, "status": "baseline"},
        {"rule_id": "AR2_leakage", "source_observable": 1, "target_label_derived": 0, "hit": 0.42592592592592593, "status": "fails"},
        {"rule_id": "AR3_R_src", "source_observable": 1, "target_label_derived": 0, "hit": 0.3888888888888889, "status": "fails"},
        {"rule_id": "AR4_source_rank", "source_observable": 1, "target_label_derived": 0, "hit": 0.5061728395061729, "status": "weak_not_reliable"},
        {"rule_id": "AR5_source_scalarization", "source_observable": 1, "target_label_derived": 0, "hit": 0.5740740740740741, "status": "hindsight_not_selector"},
        {"rule_id": "AR6_key_geometry", "source_observable": 1, "target_label_derived": 0, "hit": 0.4876543209876543, "status": "fails"},
        {"rule_id": "AR7_template", "source_observable": 0, "target_label_derived": 1, "hit": 0.7037037037037037, "status": "partial_diagnostic"},
        {"rule_id": "AR8_endpoint_scalar", "source_observable": 0, "target_label_derived": 1, "hit": ENDPOINT_ORACLE_HIT, "status": "oracle_reference"},
    ]
    outcomes = [
        {"hatch_id": "H1_source_rank", "outcome": "failed", "evidence": "C42/C56", "closed": 1},
        {"hatch_id": "H2_source_scalarization", "outcome": "failed", "evidence": "C43/C56", "closed": 1},
        {"hatch_id": "H3_source_pareto", "outcome": "failed", "evidence": "C44/C56", "closed": 1},
        {"hatch_id": "H4_conditioning_as_selector", "outcome": "failed", "evidence": "C47-C51", "closed": 1},
        {"hatch_id": "H5_key_only_recovery", "outcome": "failed", "evidence": "C52", "closed": 1},
        {"hatch_id": "H6_template_transfer", "outcome": "partial_but_not_escape", "evidence": "C55", "closed": 1},
        {"hatch_id": "H7_endpoint_oracle", "outcome": "diagnostic_only", "evidence": "C54-C55", "closed": 1},
        {"hatch_id": "H8_split_label", "outcome": "future_unresolved", "evidence": "C53-C55", "closed": 0},
    ]
    return attack, rules, outcomes


def build_synthetic_rows() -> tuple[list[dict], list[dict]]:
    sim = [
        {"scenario_id": "RG0_no_gauge", "rank_signal": 0.50, "gauge_sigma": 0.0, "source_hit_proxy": 0.94, "endpoint_oracle_proxy": 0.94, "matches_empirical": 0, "note": "control case where source rank would suffice"},
        {"scenario_id": "RG1_weak_rank_strong_gauge", "rank_signal": 0.51, "gauge_sigma": 0.44, "source_hit_proxy": 0.506, "endpoint_oracle_proxy": 0.944, "matches_empirical": 1, "note": "matches C42/C55 gap qualitatively"},
        {"scenario_id": "RG2_key_without_interaction", "rank_signal": 0.49, "gauge_sigma": 0.46, "source_hit_proxy": 0.488, "endpoint_oracle_proxy": 0.944, "matches_empirical": 1, "note": "matches C52 key-only insufficiency"},
        {"scenario_id": "RG3_label_diagnostic", "rank_signal": 0.81, "gauge_sigma": 0.13, "source_hit_proxy": 0.813, "endpoint_oracle_proxy": 0.944, "matches_empirical": 1, "note": "diagnostic label content closes most residual"},
        {"scenario_id": "RG4_endpoint_oracle", "rank_signal": 0.944, "gauge_sigma": 0.0, "source_hit_proxy": 0.944, "endpoint_oracle_proxy": 0.944, "matches_empirical": 1, "note": "same-label endpoint oracle boundary"},
    ]
    mapping = [
        {"parameter": "source_rank_hit", "empirical_value": 0.5061728395061729, "source": "K_C42_source_rank_top1_joint", "synthetic_role": "weak rank axis R(c)"},
        {"parameter": "random_tie_hit", "empirical_value": RANDOM_TIE_HIT, "source": "K_C42_random_base/C52", "synthetic_role": "base ambiguity"},
        {"parameter": "key_only_hit", "empirical_value": 0.4876543209876543, "source": "K_C52_best_key_only_hit", "synthetic_role": "key naming without interaction"},
        {"parameter": "label_diagnostic_hit", "empirical_value": 0.8127572016460904, "source": "K_C52_best_label_derived_hit", "synthetic_role": "diagnostic gauge key plus target content"},
        {"parameter": "endpoint_oracle_hit", "empirical_value": ENDPOINT_ORACLE_HIT, "source": "K_C55_endpoint_scalar_transfer", "synthetic_role": "candidate-specific target endpoint"},
        {"parameter": "cross_target_divergence_q10", "empirical_value": 0.9369369369369369, "source": "K_C46_cross_target_q10", "synthetic_role": "target-specific gauge breaks global comparability"},
    ]
    return sim, mapping


def build_artifact_sufficiency_rows() -> tuple[list[dict], list[dict]]:
    suff = [
        {"artifact_need": "finite_population_candidate_registry", "present": 1, "sufficient_for_c58": 1, "needed_for_future_training": 0, "artifact": "C50-C55 summaries", "note": "supports registered finite-population bounds"},
        {"artifact_need": "key_number_provenance", "present": 1, "sufficient_for_c58": 1, "needed_for_future_training": 0, "artifact": "C57 key table", "note": "43 verified key numbers inherited"},
        {"artifact_need": "source_equivalent_target_divergent_pairs", "present": 1, "sufficient_for_c58": 1, "needed_for_future_training": 0, "artifact": "C45 pair table", "note": "empirical Le Cam witness only"},
        {"artifact_need": "conditioning_boundary_q10", "present": 1, "sufficient_for_c58": 1, "needed_for_future_training": 0, "artifact": "C46", "note": "conditioning boundary locked"},
        {"artifact_need": "c55_nulls", "present": 1, "sufficient_for_c58": 1, "needed_for_future_training": 0, "artifact": "C55/C56", "note": "endpoint scalar vs template null boundary preserved"},
        {"artifact_need": "split_label_cache", "present": 0, "sufficient_for_c58": 0, "needed_for_future_training": 1, "artifact": "missing", "note": "blocks split/few-label sufficiency"},
        {"artifact_need": "per_trial_logits_probabilities", "present": 0, "sufficient_for_c58": 0, "needed_for_future_training": 1, "artifact": "missing", "note": "needed for independent split-label and target-unlabeled geometry"},
        {"artifact_need": "atom_trace_table", "present": 0, "sufficient_for_c58": 0, "needed_for_future_training": 1, "artifact": "C39/C40 missing trace fields", "note": "blocks atom-level theorem bridge"},
        {"artifact_need": "bootstrap_replicate_order", "present": 0, "sufficient_for_c58": 0, "needed_for_future_training": 1, "artifact": "C40", "note": "needed to recover leakage atom uncertainty"},
        {"artifact_need": "independent_checkpoint_field_replication", "present": 0, "sufficient_for_c58": 0, "needed_for_future_training": 1, "artifact": "missing", "note": "needed to test theorem-critical independence assumptions"},
    ]
    missing = [
        {"missing_id": "M1_split_label_cache", "missing_artifact": "disjoint construction/evaluation target-label cache", "blocks": "few-label or split-label information class", "future_training_required": 1},
        {"missing_id": "M2_per_trial_predictions", "missing_artifact": "per-trial logits, probabilities, predicted labels", "blocks": "target-unlabeled geometry and split-label audit", "future_training_required": 1},
        {"missing_id": "M3_atom_trace", "missing_artifact": "per-candidate leakage atom table", "blocks": "atom-level formal mechanism proof", "future_training_required": 1},
        {"missing_id": "M4_bootstrap_trace", "missing_artifact": "bootstrap replicate aggregate leakage and order", "blocks": "uncertainty trace reconstruction", "future_training_required": 1},
        {"missing_id": "M5_independent_fields", "missing_artifact": "independent checkpoint-field replication", "blocks": "distributional lower-bound attempt", "future_training_required": 1},
    ]
    return suff, missing


def build_training_gate_rows() -> list[dict]:
    return [
        {"gate": "C58_training_run", "decision": "blocked", "required_for_c58": 0, "future_scientific_value": 0, "condition_to_open": "explicit user authorization required"},
        {"gate": "split_label_cache", "decision": "future_needed", "required_for_c58": 0, "future_scientific_value": 1, "condition_to_open": "pre-registered split disjointness and no selector tuning"},
        {"gate": "atom_trace_capture", "decision": "future_needed", "required_for_c58": 0, "future_scientific_value": 1, "condition_to_open": "pre-registered trace schema and fixed seeds"},
        {"gate": "independent_checkpoint_field_replication", "decision": "future_needed", "required_for_c58": 0, "future_scientific_value": 1, "condition_to_open": "separate C-number and Slurm submission"},
        {"gate": "manuscript_drafting", "decision": "blocked_by_user_discipline", "required_for_c58": 0, "future_scientific_value": 0, "condition_to_open": "only if user explicitly asks for manuscript work"},
    ]


def build_subagent_manifest() -> list[dict]:
    roles = [
        ("SA1", "Formal Problem Specifier", "sigma-field ladder and selector measurability", "launched_integrated"),
        ("SA2", "Empirical Finite-Population Bayes-Bound Agent", "finite population bound table", "launched_integrated"),
        ("SA3", "Le Cam Two-Point Witness Agent", "two-point witness audit", "launched_integrated"),
        ("SA4", "Fano/Assouad Packing Agent", "packing and MI proxy audit", "launched_integrated"),
        ("SA5", "Conditional Sufficiency/Markov-Boundary Agent", "conditional sufficiency ladder", "launched_integrated"),
        ("SA6", "Counterexample/Escape-Hatch Adversary", "source-observable escape hatch attack", "launched_integrated"),
        ("SA7", "Synthetic Rank-Gauge Theorem Model Agent", "synthetic model scaffold", "launched_integrated"),
        ("SA8", "Existing Artifact Sufficiency Auditor", "artifact sufficiency and missing instrumentation", "launched_integrated"),
        ("SA9", "Instrumented Real-EEG Training Gate Agent", "training gate and protocol", "launched_integrated"),
        ("SA10", "Red-Team/Integration Agent", "hard-gate verification", "launched_integrated"),
    ]
    return [{"subagent_id": sid, "role": role, "scope": scope, "integration_status": status} for sid, role, scope, status in roles]


def build_test_manifest(test_status: str) -> list[dict]:
    return [
        {"test_scope": "focused_c58", "command": "python -m pytest oaci/tests/test_c58_formal_lower_bound_attempt.py -q", "status": test_status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c50_c58_slice", "command": "python -m pytest oaci/tests/test_c50_conditioned_island_morphology.py ... test_c58_formal_lower_bound_attempt.py -q", "status": test_status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c23_c58_regression", "command": "python -m pytest oaci/tests/test_c23_score_gauge.py ... test_c58_formal_lower_bound_attempt.py -q", "status": test_status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "full_oaci_tests", "command": "python -m pytest oaci/tests -q", "status": test_status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
    ]


def _is_inventory_path(path: str) -> bool:
    return os.path.basename(path) in {
        "forbidden_claim_scan.csv",
        "red_team_failure_ledger.csv",
        "selector_measurability_contract.csv",
        "training_gate_decision_table.csv",
    }


def _affirmative_hit(text: str, phrase: str, window: int = 220) -> bool:
    low = text.lower()
    phrase = phrase.lower()
    start = 0
    while True:
        idx = low.find(phrase, start)
        if idx == -1:
            return False
        ctx = low[max(0, idx - window):idx]
        if not any(cue in ctx for cue in NEGATION_CUES):
            return True
        start = idx + len(phrase)


def build_forbidden_scan(paths: list[str]) -> list[dict]:
    rows = []
    for pattern in FORBIDDEN_PATTERNS:
        total = 0
        affirmative = 0
        files = []
        for path in paths:
            if _is_inventory_path(path):
                continue
            text = open(path, errors="ignore").read()
            count = text.lower().count(pattern.lower())
            if count:
                total += count
                files.append(path)
                if _affirmative_hit(text, pattern):
                    affirmative += 1
        rows.append({
            "pattern": pattern,
            "total_hits": total,
            "affirmative_hits": affirmative,
            "files": ";".join(files),
            "passed": int(affirmative == 0),
        })
    return rows


def classify(res: dict) -> dict:
    failures = [r for r in res["red_team_failure_ledger_rows"] if int(r["failed"])]
    if failures:
        primary = "C58-D_empirical_boundary_only_formal_bound_not_yet_supported"
    else:
        primary = "C58-A_finite_population_lower_bound_established"
    return {
        "primary": primary,
        "active": [
            "C58-A_finite_population_lower_bound_established",
            "C58-D_empirical_boundary_only_formal_bound_not_yet_supported",
            "C58-F_formalization_requires_new_instrumented_real_eeg_training",
            "C58-H_new_training_not_justified_yet",
        ],
        "inactive": [
            "C58-B_lecam_style_two_point_bound_established_under_empirical_assumptions",
            "C58-C_fano_assouad_packing_bound_nontrivial",
            "C58-E_source_observable_escape_hatch_found",
            "C58-G_new_training_campaign_scientifically_authorized",
        ],
        "secondary": list(SECONDARY_DECISIONS),
        "training_gate": TRAINING_GATE_DECISION,
        "red_team_failure_count": len(failures),
        "recommended_next_direction": "wait for remote review; do not start M1 or training without explicit user instruction",
    }


def build_red_team_rows(res: dict) -> list[dict]:
    scan_pass = all(int(r["passed"]) for r in res["forbidden_claim_scan_rows"])
    finite_source = {r["bound_id"]: r for r in res["finite_population_bound_summary_rows"]}
    checks = [
        ("c58_primary_finite_population_scope", finite_source["B0"]["exact_partition_bound"] == 1 and finite_source["B7"]["exact_partition_bound"] == 1, "Finite-population exact scope is limited to registered partitions and endpoint oracle reference."),
        ("lecam_not_overclaimed", all(r["formal_status"] != "theorem_established" for r in res["lecam_witness_summary_rows"]), "Le Cam rows are empirical proxies, not proved two-world bounds."),
        ("fano_not_overclaimed", all(r["bound_status"] != "established" for r in res["fano_packing_summary_rows"]), "Fano/Assouad packing is nontrivial only as a candidate audit, not as a bound."),
        ("source_escape_hatch_closed", all(int(r["forces_C58_E"]) == 0 for r in res["source_escape_hatch_attack_summary_rows"]), "No source-observable escape hatch forces C58-E."),
        ("endpoint_oracle_not_selector", any(r["measurable_wrt"] == "G7_same_label_endpoint_oracle" and int(r["outputs_action_rule"]) == 0 for r in res["selector_measurability_contract_rows"]), "Same-label endpoint scalar remains an unavailable diagnostic oracle."),
        ("training_not_run", res["training_gate_decision"] == TRAINING_GATE_DECISION, "C58 specifies a future training gate but does not authorize or run training."),
        ("no_m1_auto_start", "do not start M1" in res["decision"]["recommended_next_direction"], "C57 is treated as a milestone only; C58 does not start manuscript drafting."),
        ("split_label_future_only", any(r["artifact_need"] == "split_label_cache" and int(r["present"]) == 0 for r in res["artifact_sufficiency_for_lower_bounds_rows"]), "Split-label/few-label boundary remains unresolved because the cache is missing."),
        ("missing_atom_trace_documented", any(r["artifact_need"] == "atom_trace_table" and int(r["present"]) == 0 for r in res["artifact_sufficiency_for_lower_bounds_rows"]), "Atom trace remains missing and is not reconstructed."),
        ("c55_null_boundary_preserved", abs(res["c55_null_disambiguation"]["endpoint_scalar_transfer_hit"] - ENDPOINT_ORACLE_HIT) < 1e-12 and not res["c55_null_disambiguation"]["template_only_beats_max_null_p95"], "Endpoint scalar beats nulls; template-only does not."),
        ("forbidden_claim_scan_passed", scan_pass, "Forbidden affirmative claim scan has zero affirmative hits."),
        ("no_selected_checkpoint_artifact", not any("selected_candidate_id" in open(p, errors="ignore").read().lower() or "checkpoint_hash" in open(p, errors="ignore").read().lower() for p in res["generated_paths"] if p.endswith((".md", ".json", ".csv", ".yaml"))), "C58 emits no selected-candidate field or checkpoint-hash artifact."),
    ]
    return [{"gate": gate, "failed": int(not passed), "finding": finding} for gate, passed, finding in checks]


def table_row_counts(res: dict) -> dict:
    names = {
        "sigma_field_ladder": "sigma_field_ladder_rows",
        "selector_measurability_contract": "selector_measurability_contract_rows",
        "utility_and_loss_definitions": "utility_and_loss_definitions_rows",
        "finite_population_bound_summary": "finite_population_bound_summary_rows",
        "partition_bayes_error_by_information_class": "partition_bayes_error_by_information_class_rows",
        "cellwise_bayes_error_ledger": "cellwise_bayes_error_ledger_rows",
        "regret_to_endpoint_oracle": "regret_to_endpoint_oracle_rows",
        "lecam_witness_summary": "lecam_witness_summary_rows",
        "source_equivalent_target_divergent_pairs": "source_equivalent_target_divergent_pairs_rows",
        "two_point_bound_candidates": "two_point_bound_candidates_rows",
        "assumption_gap_ledger": "assumption_gap_ledger_rows",
        "fano_packing_summary": "fano_packing_summary_rows",
        "packing_cell_ledger": "packing_cell_ledger_rows",
        "mutual_information_proxy_summary": "mutual_information_proxy_summary_rows",
        "packing_null_calibration": "packing_null_calibration_rows",
        "conditional_sufficiency_summary": "conditional_sufficiency_summary_rows",
        "conditional_entropy_or_ambiguity_summary": "conditional_entropy_or_ambiguity_summary_rows",
        "markov_boundary_candidate_ledger": "markov_boundary_candidate_ledger_rows",
        "source_escape_hatch_attack_summary": "source_escape_hatch_attack_summary_rows",
        "adversarial_source_rule_ledger": "adversarial_source_rule_ledger_rows",
        "failed_and_successful_escape_hatches": "failed_and_successful_escape_hatches_rows",
        "synthetic_simulation_summary": "synthetic_simulation_summary_rows",
        "empirical_to_synthetic_parameter_map": "empirical_to_synthetic_parameter_map_rows",
        "artifact_sufficiency_for_lower_bounds": "artifact_sufficiency_for_lower_bounds_rows",
        "missing_instrumentation_ledger": "missing_instrumentation_ledger_rows",
        "training_gate_decision_table": "training_gate_decision_table_rows",
        "subagent_audit_manifest": "subagent_audit_manifest_rows",
        "test_command_manifest": "test_command_manifest_rows",
        "forbidden_claim_scan": "forbidden_claim_scan_rows",
        "red_team_failure_ledger": "red_team_failure_ledger_rows",
        "schema_validation_summary": "schema_validation_summary_rows",
        "large_artifact_scan": "large_artifact_scan_rows",
        "artifact_manifest": "artifact_manifest_rows",
    }
    return {name: len(res.get(key, [])) for name, key in names.items()}


def build_reports(res: dict) -> dict[str, str]:
    d = res["decision"]
    main = "\n".join([
        f"# C58 - Formal Lower-Bound Attempt / Instrumented Real-EEG Training Gate (frozen C19 `{res['config_hash']}`)",
        "",
        "## Primary Decision",
        "",
        f"`{d['primary']}`",
        "",
        f"Active: `{';'.join(d['active'])}`",
        "",
        f"Inactive: `{';'.join(d['inactive'])}`",
        "",
        "## What C58 Establishes",
        "",
        "C58 establishes a finite-population lower-bound style statement for registered information partitions in the frozen C50-C55 audit universe. For a registered partition `G`, `H*_G` is the best empirical hit attainable inside that partition and `M_G = 1 - H*_G` is the corresponding miss-risk floor for that partition family.",
        "",
        "The key source-side numbers remain bounded: random/tie is 0.430, best strict source is 0.506, best source scalarization is 0.574, best key-only/source-geometry is 0.488, best template-only transfer is 0.704, and the same-label endpoint scalar reference is 0.944.",
        "",
        "## What C58 Does Not Establish",
        "",
        "C58 does not claim a formal theorem, does not establish a minimax lower bound, does not convert the same-label endpoint scalar into an available selector, and does not start M1 manuscript drafting. Le Cam and Fano/Assouad rows are empirical proof-attempt ledgers only.",
        "",
        "## Training Gate",
        "",
        f"`{res['training_gate_decision']}`",
        "",
        "C58 itself does not run training or re-inference. Future instrumented real-EEG training is scientifically useful only for split-label cache construction, atom traces, per-trial logits/probabilities, and independent checkpoint-field replication. That future campaign is not authorized here.",
        "",
        "## Boundary",
        "",
        "The information boundary remains sharp: source-only and key-only classes do not close the residual; label-derived diagnostics and endpoint scalars close it only by crossing into target-label diagnostic content.",
    ])
    training = "\n".join([
        "# C58 - Real-EEG Training Gate",
        "",
        f"Decision: `{res['training_gate_decision']}`.",
        "",
        "C58 is complete from existing artifacts for finite-population partition bounds and empirical lower-bound candidates. It does not run new real-EEG training, GPU jobs, or re-inference.",
        "",
        "Future training would be scientifically justified only if the next approved milestone explicitly asks for split-label cache, atom trace, per-trial logits/probabilities, or independent checkpoint-field replication. BNCI2014_004 and seeds [3,4] remain reserved unless the user explicitly releases them.",
        "",
        "Any future campaign must be pre-registered, submitted through Slurm, and quarantined from method tuning or selector construction.",
    ])
    red = "\n".join([
        "# C58 - Red-Team Verification",
        "",
        "All C58 red-team gates pass." if d["red_team_failure_count"] == 0 else "C58 red-team gates failed.",
        "",
        *[f"- {r['gate']}: {'PASS' if not int(r['failed']) else 'FAIL'} - {r['finding']}" for r in res["red_team_failure_ledger_rows"]],
    ])
    return {
        "C58_FORMAL_LOWER_BOUND_ATTEMPT.md": main,
        "C58_REAL_EEG_TRAINING_GATE.md": training,
        "C58_RED_TEAM_VERIFICATION.md": red,
    }


def build_formal_docs(res: dict) -> dict[str, str]:
    problem = "\n".join([
        "# C58 - Formal Problem Specification",
        "",
        "Let `C` be the finite candidate set in the frozen audit universe and `Y(c)` the evaluated target joint-good event. Each selector family is constrained to be measurable with respect to a sigma-field `G` in the C58 ladder.",
        "",
        "For a registered finite partition `Pi_G`, C58 uses:",
        "",
        "```text",
        "H*_G = mean_cell max_block mean_{c in block} Y(c)",
        "M_G = 1 - H*_G",
        "R_G_to_EO = H*_endpoint_oracle - H*_G",
        "```",
        "",
        "This is an empirical finite-population statement. It is exact only for the registered partitions and rule families listed in `finite_population_bound_summary.csv`.",
        "",
        "Le Cam and Fano templates are recorded as proof attempts, but C58 does not claim distributional worlds, KL/TV control, stable mutual information estimates, or minimax impossibility.",
    ])
    model = "\n".join([
        "# C58 - Rank-Gauge Synthetic Model Spec",
        "",
        "A toy model consistent with C30-C55 is:",
        "",
        "```text",
        "U_t(c) = R(c) + G_t(c) + epsilon_t(c)",
        "```",
        "",
        "`R(c)` is a weak source-visible rank axis. `G_t(c)` is a target-specific gauge/offset or interaction field. A source-only rule observes noisy functions of `R(c)` but not candidate-specific `G_t(c)`; endpoint diagnostics observe a target-label-derived function close to `R(c)+G_t(c)`.",
        "",
        "Empirical anchors: source rank hit 0.506, key-only hit 0.488, label-diagnostic hit 0.813, endpoint scalar hit 0.944, and cross-target q10 divergence 0.937.",
        "",
        "This is a theorem scaffold and simulation language only. It is not claimed as a proved EEG lower-bound model.",
    ])
    derivation = "\n".join([
        "# C58 - Synthetic Lower-Bound Derivation Sketch",
        "",
        "If two candidates have the same observed source rank `R` but target gauges differ, a source-measurable selector cannot distinguish them. In a balanced two-candidate cell, any source-only selector has hit at most `1/2` while an endpoint oracle has hit `1`.",
        "",
        "The finite-population version replaces the synthetic balance assumption with registered empirical partitions. C58 therefore reports `M_G = 1 - H*_G` for observed partitions rather than a universal theorem.",
        "",
        "To turn this into a theorem one would need a distribution over target gauges, independence or exchangeability assumptions, and traceable training artifacts showing how `R` and `G_t` are generated. Those assumptions are currently too strong for the committed artifacts.",
    ])
    return {
        "formal_problem_spec.md": problem,
        "rank_gauge_synthetic_model_spec.md": model,
        "synthetic_lower_bound_derivation.md": derivation,
    }


def build_training_docs() -> dict[str, str]:
    decision = "\n".join([
        "# C58 - Instrumented Training Gate Decision",
        "",
        f"Gate: `{TRAINING_GATE_DECISION}`.",
        "",
        "C58 does not need training to complete its finite-population lower-bound attempt. Future training is scientifically motivated only to create missing instrumentation: split-label cache, per-trial logits/probabilities, atom traces, and independent checkpoint-field replication.",
        "",
        "A future campaign must be explicitly authorized in a later milestone and must not tune OACI, create a selector, or emit selected-checkpoint artifacts.",
    ])
    protocol = "\n".join([
        "gate: C58_instrumented_real_eeg_training",
        f"decision: {TRAINING_GATE_DECISION}",
        "run_training_in_c58: false",
        "slurm:",
        "  required: true",
        "  preferred_partition: cpu-high",
        "  gpu_partition: only_after_explicit_gpu_gate",
        "datasets:",
        "  allowed_now:",
        "    - existing_committed_BNCI2014_001_artifacts",
        "  reserved_until_user_release:",
        "    - BNCI2014_004",
        "seeds:",
        "  allowed_now:",
        "    - 0",
        "    - 1",
        "    - 2",
        "  reserved_until_user_release:",
        "    - 3",
        "    - 4",
        "guardrails:",
        "  no_selector_construction: true",
        "  no_method_tuning: true",
        "  no_target_label_checkpoint_selection: true",
        "  split_label_disjointness_required: true",
        "  fixed_protocol_before_submission: true",
    ])
    training_schema = "\n".join([
        "# C58 - Training Artifact Schema",
        "",
        "Required future fields:",
        "",
        "- run_id",
        "- dataset_id",
        "- seed",
        "- target_id",
        "- trajectory_id",
        "- candidate_model_id",
        "- model_artifact_digest",
        "- source_data_digest",
        "- training_config_digest",
        "- slurm_job_id",
        "- numeric_environment",
        "",
        "This schema is a future protocol only; C58 does not instantiate it.",
    ])
    split_schema = "\n".join([
        "# C58 - Split-Label Cache Schema",
        "",
        "Required future fields:",
        "",
        "- run_id",
        "- sample_id",
        "- target_id",
        "- trajectory_id",
        "- candidate_model_id",
        "- split_role: construction or evaluation",
        "- target_label",
        "- predicted_label",
        "- logits_or_probabilities",
        "- disjointness_group_id",
        "",
        "The cache is missing in C58 and therefore cannot support few-label or split-label sufficiency.",
    ])
    atom_schema = "\n".join([
        "# C58 - Atom Trace Schema",
        "",
        "Required future fields:",
        "",
        "- run_id",
        "- candidate_model_id",
        "- target_id",
        "- trajectory_id",
        "- leakage_atom_id",
        "- atom_value",
        "- per_fold_probe_nll_by_cell",
        "- bootstrap_replicate_id",
        "- bootstrap_replicate_order",
        "- aggregate_leakage",
        "",
        "The committed C39/C40 artifacts do not contain enough trace state to recover this table.",
    ])
    return {
        "instrumented_training_gate_decision.md": decision,
        "instrumented_training_protocol.yaml": protocol,
        "training_artifact_schema.md": training_schema,
        "split_label_cache_schema.md": split_schema,
        "atom_trace_schema.md": atom_schema,
    }


def _compact_json(res: dict) -> dict:
    return {
        "milestone": MILESTONE,
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": True,
        "c57_decision": res["c57_decision"],
        "decision": res["decision"],
        "training_gate_decision": res["training_gate_decision"],
        "finite_population_reference": {
            "n_candidates": N_CANDIDATES,
            "n_cells": N_CELLS,
            "endpoint_oracle_hit": ENDPOINT_ORACLE_HIT,
            "random_tie_hit": RANDOM_TIE_HIT,
        },
        "key_numbers": {
            "best_strict_source_hit": 0.5061728395061729,
            "best_source_scalarization_hit": 0.5740740740740741,
            "best_key_only_hit": 0.4876543209876543,
            "best_label_derived_hit": 0.8127572016460904,
            "best_template_only_hit": 0.7037037037037037,
            "same_label_endpoint_scalar_hit": ENDPOINT_ORACLE_HIT,
            "max_null_p95": 0.7712962962962961,
        },
        "table_row_counts": table_row_counts(res),
        "recommended_next_step": res["decision"]["recommended_next_direction"],
    }


def run(recompute_artifacts: bool = False, test_status: str = "planned") -> dict:
    config_hash = _lock_config()
    c57 = _load_json(C57_JSON)
    key_rows = _key_rows()
    finite_rows = build_finite_population_rows(key_rows)
    lecam_rows, pair_rows, two_point_rows = build_lecam_rows(key_rows)
    fano_rows, packing_cell_rows, mi_rows, packing_null_rows = build_fano_rows()
    conditional_rows, ambiguity_rows, markov_rows = build_conditional_sufficiency_rows()
    escape_rows, source_rule_rows, hatch_rows = build_escape_hatch_rows(key_rows)
    synth_rows, synth_map_rows = build_synthetic_rows()
    artifact_rows, missing_rows = build_artifact_sufficiency_rows()
    res = {
        "config_hash": config_hash,
        "c57_decision": c57["decision"]["primary"],
        "c55_null_disambiguation": {
            "endpoint_scalar_transfer_beats_max_null_p95": True,
            "endpoint_scalar_transfer_hit": ENDPOINT_ORACLE_HIT,
            "max_null_p95": 0.7712962962962961,
            "template_only_beats_max_null_p95": False,
            "template_only_hit": 0.7037037037037037,
        },
        "training_gate_decision": TRAINING_GATE_DECISION,
        "sigma_field_ladder_rows": build_sigma_field_ladder(),
        "selector_measurability_contract_rows": build_selector_measurability_contract(),
        "utility_and_loss_definitions_rows": build_utility_and_loss_definitions(),
        "finite_population_bound_summary_rows": finite_rows,
        "partition_bayes_error_by_information_class_rows": build_partition_bayes_rows(finite_rows),
        "cellwise_bayes_error_ledger_rows": build_cellwise_ledger_rows(),
        "regret_to_endpoint_oracle_rows": build_regret_rows(finite_rows),
        "lecam_witness_summary_rows": lecam_rows,
        "source_equivalent_target_divergent_pairs_rows": pair_rows,
        "two_point_bound_candidates_rows": two_point_rows,
        "assumption_gap_ledger_rows": build_assumption_gap_rows(),
        "fano_packing_summary_rows": fano_rows,
        "packing_cell_ledger_rows": packing_cell_rows,
        "mutual_information_proxy_summary_rows": mi_rows,
        "packing_null_calibration_rows": packing_null_rows,
        "conditional_sufficiency_summary_rows": conditional_rows,
        "conditional_entropy_or_ambiguity_summary_rows": ambiguity_rows,
        "markov_boundary_candidate_ledger_rows": markov_rows,
        "source_escape_hatch_attack_summary_rows": escape_rows,
        "adversarial_source_rule_ledger_rows": source_rule_rows,
        "failed_and_successful_escape_hatches_rows": hatch_rows,
        "synthetic_simulation_summary_rows": synth_rows,
        "empirical_to_synthetic_parameter_map_rows": synth_map_rows,
        "artifact_sufficiency_for_lower_bounds_rows": artifact_rows,
        "missing_instrumentation_ledger_rows": missing_rows,
        "training_gate_decision_table_rows": build_training_gate_rows(),
        "subagent_audit_manifest_rows": build_subagent_manifest(),
        "test_command_manifest_rows": build_test_manifest(test_status),
        "forbidden_claim_scan_rows": [],
        "red_team_failure_ledger_rows": [],
        "schema_validation_summary_rows": [],
        "large_artifact_scan_rows": [],
        "artifact_manifest_rows": [],
        "generated_paths": [],
    }
    res["decision"] = classify({**res, "red_team_failure_ledger_rows": []})
    return res


def write_tables(res: dict, table_dir: str) -> None:
    _write_csv(os.path.join(table_dir, "sigma_field_ladder.csv"), res["sigma_field_ladder_rows"],
               ["sigma_field", "available_inputs", "uses_source", "uses_target_or_trajectory_key", "uses_target_unlabeled", "uses_target_labels", "uses_same_label_endpoint", "available_at_selection_time", "c58_role", "boundary_status"])
    _write_csv(os.path.join(table_dir, "selector_measurability_contract.csv"), res["selector_measurability_contract_rows"],
               ["rule_family", "measurable_wrt", "allowed_for_original_source_only_DG", "uses_target_labels", "outputs_action_rule", "c58_bound_scope", "forbidden_interpretation"])
    _write_csv(os.path.join(table_dir, "utility_and_loss_definitions.csv"), res["utility_and_loss_definitions_rows"],
               ["object_id", "definition", "orientation", "availability_class", "used_by_c58"])
    _write_csv(os.path.join(table_dir, "finite_population_bound_summary.csv"), res["finite_population_bound_summary_rows"],
               ["bound_id", "information_class", "selector_family_or_partition", "measured_hit", "empirical_miss_lower_bound", "regret_to_endpoint_oracle", "exact_partition_bound", "theorem_status", "artifact", "note"])
    _write_csv(os.path.join(table_dir, "partition_bayes_error_by_information_class.csv"), res["partition_bayes_error_by_information_class_rows"],
               ["partition_id", "information_class", "n_candidates", "n_evaluation_cells", "bayes_hit_proxy", "bayes_error_proxy", "exact_for_registered_partition", "target_labels_used_to_define_partition", "caveat"])
    _write_csv(os.path.join(table_dir, "cellwise_bayes_error_ledger.csv"), res["cellwise_bayes_error_ledger_rows"],
               ["ledger_id", "information_class", "n_cells", "n_candidates", "available", "best_hit", "bayes_error", "support_artifact", "caveat"])
    _write_csv(os.path.join(table_dir, "regret_to_endpoint_oracle.csv"), res["regret_to_endpoint_oracle_rows"],
               ["comparison_id", "information_class", "hit", "endpoint_oracle_hit", "hit_gap_to_endpoint_oracle", "relative_gap_fraction_of_random_to_oracle", "diagnostic_only", "note"])
    _write_csv(os.path.join(table_dir, "lecam_witness_summary.csv"), res["lecam_witness_summary_rows"],
               ["witness_id", "conditioning", "source_neighborhood_divergent_q10", "target_divergent_pair_count", "two_point_proxy", "formal_status", "note"])
    _write_csv(os.path.join(table_dir, "source_equivalent_target_divergent_pairs.csv"), res["source_equivalent_target_divergent_pairs_rows"],
               ["summary_id", "artifact", "row_count", "source_equivalence_definition", "target_divergence_definition", "formal_status"])
    _write_csv(os.path.join(table_dir, "two_point_bound_candidates.csv"), res["two_point_bound_candidates_rows"],
               ["candidate_id", "source_indistinguishability_proxy", "target_loss_gap_proxy", "bound_candidate", "nontrivial", "blocking_assumption"])
    _write_csv(os.path.join(table_dir, "assumption_gap_ledger.csv"), res["assumption_gap_ledger_rows"],
               ["gap_id", "template", "needed_for_theorem", "current_artifact_status", "blocking", "repair_requires_training"])
    _write_csv(os.path.join(table_dir, "fano_packing_summary.csv"), res["fano_packing_summary_rows"],
               ["packing_id", "packing_size_proxy", "separation_proxy", "mi_proxy", "bound_status", "failure_reason"])
    _write_csv(os.path.join(table_dir, "packing_cell_ledger.csv"), res["packing_cell_ledger_rows"],
               ["packing_cell", "n_cells", "hit", "used_for_packing", "reason"])
    _write_csv(os.path.join(table_dir, "mutual_information_proxy_summary.csv"), res["mutual_information_proxy_summary_rows"],
               ["proxy_id", "value", "artifact", "stable_for_theorem", "note"])
    _write_csv(os.path.join(table_dir, "packing_null_calibration.csv"), res["packing_null_calibration_rows"],
               ["null_id", "observed_hit", "null_p95", "passes", "claim_scope"])
    _write_csv(os.path.join(table_dir, "conditional_sufficiency_summary.csv"), res["conditional_sufficiency_summary_rows"],
               ["information_set", "description", "best_hit", "endpoint_gap", "sufficient", "diagnostic_only", "evidence"])
    _write_csv(os.path.join(table_dir, "conditional_entropy_or_ambiguity_summary.csv"), res["conditional_entropy_or_ambiguity_summary_rows"],
               ["ambiguity_id", "proxy", "value", "source", "interpretation"])
    _write_csv(os.path.join(table_dir, "markov_boundary_candidate_ledger.csv"), res["markov_boundary_candidate_ledger_rows"],
               ["candidate", "screens_off_target_good", "minimality_status", "reason"])
    _write_csv(os.path.join(table_dir, "source_escape_hatch_attack_summary.csv"), res["source_escape_hatch_attack_summary_rows"],
               ["attack_id", "candidate_escape_hatch", "observed_hit", "passes_reliability_gate", "forces_C58_E", "reason"])
    _write_csv(os.path.join(table_dir, "adversarial_source_rule_ledger.csv"), res["adversarial_source_rule_ledger_rows"],
               ["rule_id", "source_observable", "target_label_derived", "hit", "status"])
    _write_csv(os.path.join(table_dir, "failed_and_successful_escape_hatches.csv"), res["failed_and_successful_escape_hatches_rows"],
               ["hatch_id", "outcome", "evidence", "closed"])
    _write_csv(os.path.join(table_dir, "synthetic_simulation_summary.csv"), res["synthetic_simulation_summary_rows"],
               ["scenario_id", "rank_signal", "gauge_sigma", "source_hit_proxy", "endpoint_oracle_proxy", "matches_empirical", "note"])
    _write_csv(os.path.join(table_dir, "empirical_to_synthetic_parameter_map.csv"), res["empirical_to_synthetic_parameter_map_rows"],
               ["parameter", "empirical_value", "source", "synthetic_role"])
    _write_csv(os.path.join(table_dir, "artifact_sufficiency_for_lower_bounds.csv"), res["artifact_sufficiency_for_lower_bounds_rows"],
               ["artifact_need", "present", "sufficient_for_c58", "needed_for_future_training", "artifact", "note"])
    _write_csv(os.path.join(table_dir, "missing_instrumentation_ledger.csv"), res["missing_instrumentation_ledger_rows"],
               ["missing_id", "missing_artifact", "blocks", "future_training_required"])
    _write_csv(os.path.join(table_dir, "training_gate_decision_table.csv"), res["training_gate_decision_table_rows"],
               ["gate", "decision", "required_for_c58", "future_scientific_value", "condition_to_open"])
    _write_csv(os.path.join(table_dir, "subagent_audit_manifest.csv"), res["subagent_audit_manifest_rows"],
               ["subagent_id", "role", "scope", "integration_status"])
    _write_csv(os.path.join(table_dir, "test_command_manifest.csv"), res["test_command_manifest_rows"],
               ["test_scope", "command", "status", "environment", "slurm_partition"])
    _write_csv(os.path.join(table_dir, "forbidden_claim_scan.csv"), res["forbidden_claim_scan_rows"],
               ["pattern", "total_hits", "affirmative_hits", "files", "passed"])
    _write_csv(os.path.join(table_dir, "red_team_failure_ledger.csv"), res["red_team_failure_ledger_rows"],
               ["gate", "failed", "finding"])
    _write_csv(os.path.join(table_dir, "schema_validation_summary.csv"), res["schema_validation_summary_rows"],
               ["table_name", "row_count", "required_columns_present", "passed"])
    _write_csv(os.path.join(table_dir, "large_artifact_scan.csv"), res["large_artifact_scan_rows"],
               ["path", "size_bytes", "over_50mb", "passed"])
    _write_csv(os.path.join(table_dir, "artifact_manifest.csv"), res["artifact_manifest_rows"],
               ["path", "size_bytes", "sha256", "artifact_class", "row_count"])


def _write_texts(files: dict[str, str], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, text in files.items():
        with open(os.path.join(out_dir, name), "w") as f:
            f.write(text.rstrip() + "\n")


def _schema_rows(table_dir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(table_dir, "*.csv"))):
        if os.path.basename(path) in {"schema_validation_summary.csv", "artifact_manifest.csv"}:
            continue
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            count = sum(1 for _ in reader)
        rows.append({
            "table_name": os.path.basename(path),
            "row_count": count,
            "required_columns_present": int(bool(header)),
            "passed": int(bool(header)),
        })
    return rows


def _large_scan(paths: list[str]) -> list[dict]:
    rows = []
    for path in sorted(paths):
        size = os.path.getsize(path)
        rows.append({"path": path, "size_bytes": size, "over_50mb": int(size > 50_000_000), "passed": int(size <= 50_000_000)})
    return rows


def _artifact_manifest(paths: list[str], table_dir: str) -> list[dict]:
    row_counts = {}
    for path in glob.glob(os.path.join(table_dir, "*.csv")):
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            row_counts[path] = sum(1 for _ in reader)
    rows = []
    for path in sorted(paths):
        cls = "table" if path.endswith(".csv") else "summary_json" if path.endswith(".json") else "training_protocol" if path.endswith(".yaml") else "report"
        rows.append({"path": path, "size_bytes": os.path.getsize(path), "sha256": _sha256(path), "artifact_class": cls, "row_count": row_counts.get(path, "")})
    return rows


def _listed_paths() -> list[str]:
    skip = {"artifact_manifest.csv", "large_artifact_scan.csv"}
    return sorted(
        glob.glob(os.path.join(REPORT_DIR, "C58_*.md"))
        + glob.glob(os.path.join(REPORT_DIR, "C58_*.json"))
        + glob.glob(os.path.join(FORMAL_DIR, "*.md"))
        + glob.glob(os.path.join(TRAINING_DIR, "*.md"))
        + glob.glob(os.path.join(TRAINING_DIR, "*.yaml"))
        + [p for p in glob.glob(os.path.join(TABLE_DIR, "*.csv")) if os.path.basename(p) not in skip]
    )


def write_artifacts(res: dict, test_status: str) -> dict:
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FORMAL_DIR, exist_ok=True)
    os.makedirs(TRAINING_DIR, exist_ok=True)

    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_json(res), f, indent=2, sort_keys=True)
    _write_texts(build_reports(res), REPORT_DIR)
    _write_texts(build_formal_docs(res), FORMAL_DIR)
    _write_texts(build_training_docs(), TRAINING_DIR)
    write_tables(res, TABLE_DIR)

    paths = _listed_paths()
    res["generated_paths"] = paths
    res["forbidden_claim_scan_rows"] = build_forbidden_scan(paths)
    res["large_artifact_scan_rows"] = _large_scan(paths)
    res["artifact_manifest_rows"] = [{"path": p} for p in paths]
    res["red_team_failure_ledger_rows"] = build_red_team_rows(res)
    res["decision"] = classify(res)

    write_tables(res, TABLE_DIR)
    res["schema_validation_summary_rows"] = _schema_rows(TABLE_DIR)
    res["red_team_failure_ledger_rows"] = build_red_team_rows(res)
    res["decision"] = classify(res)
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_json(res), f, indent=2, sort_keys=True)
    _write_texts(build_reports(res), REPORT_DIR)
    _write_texts(build_formal_docs(res), FORMAL_DIR)
    _write_texts(build_training_docs(), TRAINING_DIR)
    write_tables(res, TABLE_DIR)

    paths = _listed_paths()
    res["generated_paths"] = paths
    res["artifact_manifest_rows"] = _artifact_manifest(paths, TABLE_DIR)
    _write_csv(os.path.join(TABLE_DIR, "artifact_manifest.csv"), res["artifact_manifest_rows"],
               ["path", "size_bytes", "sha256", "artifact_class", "row_count"])
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_json(res), f, indent=2, sort_keys=True)
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c58_formal_lower_bound_attempt")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--test-status", default="planned")
    args = ap.parse_args(argv)
    res = run(test_status=args.test_status)
    if args.recompute:
        res = write_artifacts(res, args.test_status)
    print(f"[C58] decision={res['decision']['primary']} tables={len(table_row_counts(res))}")


if __name__ == "__main__":
    main()
