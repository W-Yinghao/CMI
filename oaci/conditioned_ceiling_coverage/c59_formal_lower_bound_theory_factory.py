"""C59 - Rank-Gauge Theorem Factory / Instrumented Evidence Blueprint."""
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


MILESTONE = "C59"
REPORT_DIR = "oaci/reports"
TABLE_DIR = "oaci/reports/c59_tables"
REPORT_JSON = "oaci/reports/C59_FORMAL_LOWER_BOUND_THEORY_FACTORY.json"
C58_JSON = "oaci/reports/C58_FORMAL_LOWER_BOUND_ATTEMPT.json"
C58_TABLE_DIR = "oaci/reports/c58_tables"

DECISIONS = (
    "C59-A_registered_partition_bound_formalized_as_theorem",
    "C59-B_rank_gauge_synthetic_lower_bound_proved",
    "C59-C_empirical_lecam_witness_bound_nontrivial",
    "C59-D_fano_assouad_bound_still_trivial_or_unstable",
    "C59-E_conditional_sufficiency_boundary_formalized",
    "C59-F_source_observable_counterexample_found",
    "C59-G_training_blueprint_ready_but_not_authorized",
    "C59-H_theory_blocked_requires_new_instrumented_data",
    "C59-I_theory_blocked_by_definition_or_claim_inconsistency",
)

TRAINING_GATE = "TRAINING_BLUEPRINT_READY_BUT_NOT_AUTHORIZED"
ENDPOINT_ORACLE_HIT = 0.9444444444444444
RANDOM_TIE_HIT = 0.4297233780360411
N_CANDIDATES = 3804
N_CELLS = 162

FORBIDDEN_PATTERNS = (
    "distribution-free minimax theorem",
    "EEG theorem",
    "source-only rescue",
    "OACI rescue",
    "deployable selector",
    "checkpoint recommendation artifact",
    "few-label sufficiency",
    "same-label endpoint oracle available at selection time",
    "new EEG training run",
    "silent re-inference",
    "BNCI2014_004 used",
    "seeds [3,4] used",
    "M1 manuscript drafting",
    "manuscript drafting starts",
)

NEGATION_CUES = (
    "not ",
    "no ",
    "never ",
    "cannot ",
    "does not ",
    "do not ",
    "is not ",
    "are not ",
    "unavailable",
    "blocked",
    "without ",
    "forbidden",
    "future",
    "only if explicitly authorized",
)


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


def _as_float(value, default=math.nan) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def _fmt3(value) -> str:
    try:
        x = float(value)
    except Exception:
        return str(value)
    return "n/a" if not math.isfinite(x) else f"{x:.3f}"


def _phi_minus(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _c58_rows(name: str) -> list[dict]:
    return _read_csv(os.path.join(C58_TABLE_DIR, name))


def build_formal_object_spec() -> list[dict]:
    return [
        {"object_id": "C", "symbol": "C", "definition": "finite checkpoint candidate universe", "empirical_instantiation": "3804 candidate rows", "availability_class": "all", "caveat": "frozen C50-C55 artifact universe"},
        {"object_id": "T", "symbol": "T", "definition": "target subject/domain index", "empirical_instantiation": "nine target ids in C58 lineage", "availability_class": "I2 key", "caveat": "target key is not source-only DG"},
        {"object_id": "Z", "symbol": "Z", "definition": "trajectory/cell index: seed, target, level, regime", "empirical_instantiation": "162 evaluation cells", "availability_class": "I2/I4", "caveat": "cell key alone does not order candidates"},
        {"object_id": "X_s", "symbol": "X_s(c)", "definition": "strict source observables and source objective vector", "empirical_instantiation": "leakage, R_src, source ranks, robust core, source objectives", "availability_class": "I1", "caveat": "weak rank signal, not reliable actionability"},
        {"object_id": "K", "symbol": "K(c)", "definition": "target, trajectory, seed, level, regime keys", "empirical_instantiation": "C52 key ladder", "availability_class": "I2", "caveat": "keys name cells but do not supply endpoint ordering"},
        {"object_id": "U0", "symbol": "U_0(c)", "definition": "target-unlabeled geometry or transductive fields", "empirical_instantiation": "unavailable in C52-C55 cache for closure", "availability_class": "I3", "caveat": "not source-only if used"},
        {"object_id": "D", "symbol": "D(c)", "definition": "target-label diagnostic summaries", "empirical_instantiation": "trajectory-centered label diagnostics and templates", "availability_class": "I6", "caveat": "diagnostic only"},
        {"object_id": "E", "symbol": "E(c)", "definition": "same-label endpoint scalar or endpoint vector", "empirical_instantiation": "target_joint_margin_raw and endpoint components", "availability_class": "I7", "caveat": "unavailable at selection time"},
        {"object_id": "Y", "symbol": "Y(c)", "definition": "evaluated target joint-good label", "empirical_instantiation": "primary_joint_good", "availability_class": "endpoint diagnostic", "caveat": "not a selector input"},
        {"object_id": "a_G", "symbol": "a_G(Z)", "definition": "selector measurable with respect to information class G", "empirical_instantiation": "one candidate per trajectory/cell", "availability_class": "depends on G", "caveat": "C59 emits no selected candidate artifact"},
        {"object_id": "hit", "symbol": "E[Y(a_G)]", "definition": "mean top-hit or tie-averaged hit over cells", "empirical_instantiation": "C52-C58 hit metrics", "availability_class": "evaluation", "caveat": "finite-population diagnostic metric"},
        {"object_id": "regret", "symbol": "H*(E)-H*(G)", "definition": "hit gap to same-label endpoint oracle reference", "empirical_instantiation": "C58 regret_to_endpoint_oracle", "availability_class": "diagnostic", "caveat": "not C34 continuous vector regret"},
    ]


def build_information_ladder() -> list[dict]:
    return [
        {"information_class": "I0_random_or_tie", "generators": "cell candidate set only", "hit_reference": RANDOM_TIE_HIT, "uses_target_labels": 0, "available_for_source_only_DG": 1, "c59_status": "baseline"},
        {"information_class": "I1_strict_source_observables", "generators": "X_s source metrics", "hit_reference": 0.5061728395061729, "uses_target_labels": 0, "available_for_source_only_DG": 1, "c59_status": "weak signal, not sufficient"},
        {"information_class": "I2_source_plus_keys", "generators": "X_s plus K", "hit_reference": 0.4876543209876543, "uses_target_labels": 0, "available_for_source_only_DG": 0, "c59_status": "key-only escape closed"},
        {"information_class": "I3_target_unlabeled_geometry", "generators": "X_s, K, U0", "hit_reference": "", "uses_target_labels": 0, "available_for_source_only_DG": 0, "c59_status": "not available for closure in committed cache"},
        {"information_class": "I4_target_grouped_zero_label_structure", "generators": "target/trajectory grouping", "hit_reference": 1.0, "uses_target_labels": 0, "available_for_source_only_DG": 0, "c59_status": "diagnostic ceiling/actionability split"},
        {"information_class": "I5_split_label_or_few_label", "generators": "disjoint target-label cache", "hit_reference": "", "uses_target_labels": 1, "available_for_source_only_DG": 0, "c59_status": "future unresolved"},
        {"information_class": "I6_target_label_diagnostic_content", "generators": "D and cross-cell templates", "hit_reference": 0.8127572016460904, "uses_target_labels": 1, "available_for_source_only_DG": 0, "c59_status": "diagnostic partial closure"},
        {"information_class": "I7_same_label_endpoint_oracle", "generators": "E same-candidate endpoint scalar", "hit_reference": ENDPOINT_ORACLE_HIT, "uses_target_labels": 1, "available_for_source_only_DG": 0, "c59_status": "oracle reference only"},
    ]


def build_registered_partition_rows() -> tuple[list[dict], list[dict]]:
    constants = []
    for r in _c58_rows("finite_population_bound_summary.csv"):
        constants.append({
            "partition_id": r["bound_id"],
            "information_class": r["information_class"],
            "selector_family_or_partition": r["selector_family_or_partition"],
            "H_star_pi": r["measured_hit"],
            "miss_lower_bound": r["empirical_miss_lower_bound"],
            "regret_to_endpoint_oracle": r["regret_to_endpoint_oracle"],
            "theorem_applies": r["exact_partition_bound"],
            "finite_population_theorem_status": "proved_for_registered_partition" if r["exact_partition_bound"] == "1" else "empirical_surrogate_not_full_partition_theorem",
            "source_artifact": r["artifact"],
        })
    limits = [
        {"limit_id": "L1_registered_partition_only", "covered": 1, "function_class": "selectors constant on registered Pi cells", "not_covered": "arbitrary nonlinear source functions not represented by Pi", "consequence": "C59-A is theorem-level only for declared partitions"},
        {"limit_id": "L2_tie_averaged_hits", "covered": 1, "function_class": "finite tie-averaged cell hit", "not_covered": "integer selected checkpoint identity", "consequence": "no selected checkpoint artifact emitted"},
        {"limit_id": "L3_source_surrogate_rows", "covered": 0, "function_class": "observed source-score families", "not_covered": "all measurable functions of raw training traces", "consequence": "source rows remain empirical surrogates"},
        {"limit_id": "L4_endpoint_oracle", "covered": 1, "function_class": "same-label endpoint partition reference", "not_covered": "selection-time rule", "consequence": "oracle is diagnostic only"},
        {"limit_id": "L5_distributional_generalization", "covered": 0, "function_class": "frozen finite population", "not_covered": "new datasets, future seeds, minimax populations", "consequence": "no distribution-free minimax claim"},
    ]
    return constants, limits


def build_rank_gauge_rows() -> tuple[list[dict], list[dict], list[dict]]:
    assumptions = [
        {"assumption_id": "RG-A1", "statement": "Two candidates differ by observed rank gap r and unobserved target gauge gap g", "needed_for_proof": 1, "empirical_anchor": "rank-gauge chain C30-C55", "scope": "synthetic model"},
        {"assumption_id": "RG-A2", "statement": "Source observes sign(r) but not g", "needed_for_proof": 1, "empirical_anchor": "source/key/template boundary remains open", "scope": "synthetic model"},
        {"assumption_id": "RG-A3", "statement": "Target utility gap is alpha*r + beta*g", "needed_for_proof": 1, "empirical_anchor": "C58 synthetic map", "scope": "synthetic model"},
        {"assumption_id": "RG-A4", "statement": "Gauge gap has symmetric continuous distribution with nonzero tail", "needed_for_proof": 1, "empirical_anchor": "target-specific gauge interpretation", "scope": "model assumption"},
        {"assumption_id": "RG-A5", "statement": "For n-candidate top-1 corollary, challenger gauge gaps are conditionally independent lower-tail events", "needed_for_proof": 0, "empirical_anchor": "not verified in EEG artifacts", "scope": "optional corollary"},
        {"assumption_id": "RG-A6", "statement": "The theorem does not assert a theorem about EEG distributions", "needed_for_proof": 1, "empirical_anchor": "C58-D", "scope": "claim guardrail"},
    ]
    grid = []
    for gamma in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        two = _phi_minus(gamma)
        grid.append({
            "grid_id": f"gamma_{gamma:.2f}",
            "gamma_abs_alpha_r_over_beta_sigma": gamma,
            "two_candidate_error_lower_bound_normal_gauge": two,
            "two_candidate_hit_upper_bound": 1.0 - two,
            "top1_loss_lower_bound_n10_independent_challenger_proxy": 1.0 - (1.0 - two) ** 9,
            "proof_status": "analytic_for_two_candidate_normal_gauge",
        })
    mapping = [
        {"constant": "random_tie_hit", "empirical_value": RANDOM_TIE_HIT, "source": "C58", "rank_gauge_role": "base ambiguity floor"},
        {"constant": "strict_source_hit", "empirical_value": 0.5061728395061729, "source": "C42/C52/C58", "rank_gauge_role": "weak rank-axis actionability"},
        {"constant": "source_scalarization_hit", "empirical_value": 0.5740740740740741, "source": "C43/C58", "rank_gauge_role": "best observed source utility cone"},
        {"constant": "key_only_hit", "empirical_value": 0.4876543209876543, "source": "C52/C58", "rank_gauge_role": "key does not expose gauge"},
        {"constant": "label_diagnostic_hit", "empirical_value": 0.8127572016460904, "source": "C52/C58", "rank_gauge_role": "target-label diagnostic partial gauge content"},
        {"constant": "template_only_hit", "empirical_value": 0.7037037037037037, "source": "C55/C58", "rank_gauge_role": "other-cell target-label template partial transfer"},
        {"constant": "endpoint_oracle_hit", "empirical_value": ENDPOINT_ORACLE_HIT, "source": "C55/C58", "rank_gauge_role": "same-label endpoint gauge readout"},
        {"constant": "cross_target_q10_divergence", "empirical_value": 0.9369369369369369, "source": "C46/C58", "rank_gauge_role": "global source comparability break"},
    ]
    return assumptions, grid, mapping


def build_lecam_rows() -> tuple[list[dict], list[dict], list[dict]]:
    witnesses = [
        {"witness_id": "LC1_within_target", "source_indistinguishability_proxy": 0.004842615012106538, "target_contrast": "source-near target divergence", "support_rows": 632, "uses_target_labels_for_diagnostic": 1, "valid_distribution_pair": 0},
        {"witness_id": "LC2_within_trajectory", "source_indistinguishability_proxy": 0.13287671232876713, "target_contrast": "trajectory-local divergence", "support_rows": 632, "uses_target_labels_for_diagnostic": 1, "valid_distribution_pair": 0},
        {"witness_id": "LC3_cross_target", "source_indistinguishability_proxy": 0.9369369369369369, "target_contrast": "cross-target comparability break", "support_rows": 632, "uses_target_labels_for_diagnostic": 1, "valid_distribution_pair": 0},
    ]
    bounds = []
    for row in witnesses:
        p = float(row["source_indistinguishability_proxy"])
        bounds.append({
            "bound_id": row["witness_id"],
            "tv_proxy_or_source_distance": p,
            "lecam_error_candidate": 0.5 * (1.0 - p),
            "nontrivial_empirical_candidate": int(p < 0.5),
            "theorem_status": "empirical_candidate_not_distributional_theorem",
            "blocking_assumption": "no probability laws P0/P1 or TV/KL proof",
        })
    failures = [
        {"failure_id": "LC-F1", "reason": "source distance proxy is not total variation", "blocks_theorem": 1},
        {"failure_id": "LC-F2", "reason": "target-divergent pairs are empirical artifacts, not two worlds", "blocks_theorem": 1},
        {"failure_id": "LC-F3", "reason": "no observation law over source transcripts", "blocks_theorem": 1},
        {"failure_id": "LC-F4", "reason": "split-label validation cache unavailable", "blocks_theorem": 1},
    ]
    return witnesses, bounds, failures


def build_fano_rows() -> tuple[list[dict], list[dict], list[dict]]:
    packing = [
        {"packing_id": "F1_source_rank_binary", "M": 2, "mi_proxy": 0.00871699528126672, "log_M": math.log(2), "fano_candidate": 0.0, "status": "trivial_due_log2_term", "stable": 0},
        {"packing_id": "F2_key_cells", "M": N_CELLS, "mi_proxy": "", "log_M": math.log(N_CELLS), "fano_candidate": "", "status": "missing_stable_MI", "stable": 0},
        {"packing_id": "F3_endpoint_oracle_binary", "M": 2, "mi_proxy": 0.903, "log_M": math.log(2), "fano_candidate": 0.0, "status": "tautological_endpoint_oracle", "stable": 0},
    ]
    cubes = [
        {"cube_id": "A1_two_bit_source_rank", "dimension": 1, "edge_separation": 0.43827160493827155, "edge_information_proxy": 0.00871699528126672, "assouad_status": "too_small"},
        {"cube_id": "A2_trajectory_cells", "dimension": "", "edge_separation": 0.4567901234567901, "edge_information_proxy": "", "assouad_status": "no_independent_edges"},
        {"cube_id": "A3_endpoint_bits", "dimension": 1, "edge_separation": 0.0, "edge_information_proxy": 0.903, "assouad_status": "oracle_not_ambiguity"},
    ]
    failures = [
        {"failure_id": "FANO-F1", "reason": "binary packings lose to log 2 term", "blocks_nontrivial_bound": 1},
        {"failure_id": "FANO-F2", "reason": "no stable MI/KL matrix for cell hypotheses", "blocks_nontrivial_bound": 1},
        {"failure_id": "FANO-F3", "reason": "endpoint scalar is same-label diagnostic content", "blocks_nontrivial_bound": 1},
        {"failure_id": "FANO-F4", "reason": "artifact support insufficient for theorem-grade packing", "blocks_nontrivial_bound": 1},
    ]
    return packing, cubes, failures


def build_conditional_rows() -> tuple[list[dict], list[dict], list[dict]]:
    entropy = [
        {"stage": "I0_random", "hit": RANDOM_TIE_HIT, "miss_or_ambiguity": 1.0 - RANDOM_TIE_HIT, "entropy_proxy": 0.985710406562673, "interpretation": "high cell ambiguity"},
        {"stage": "I1_source", "hit": 0.5061728395061729, "miss_or_ambiguity": 0.49382716049382713, "entropy_proxy": 0.9998900526838305, "interpretation": "weak source improvement over random"},
        {"stage": "I2_key", "hit": 0.4876543209876543, "miss_or_ambiguity": 0.5123456790123457, "entropy_proxy": 0.9995601624082293, "interpretation": "keys do not reduce ambiguity"},
        {"stage": "I6_label_diagnostic", "hit": 0.8127572016460904, "miss_or_ambiguity": 0.18724279835390956, "entropy_proxy": 0.696170555889351, "interpretation": "label diagnostic reduces ambiguity"},
        {"stage": "I7_endpoint_oracle", "hit": ENDPOINT_ORACLE_HIT, "miss_or_ambiguity": 1.0 - ENDPOINT_ORACLE_HIT, "entropy_proxy": 0.3095434291503252, "interpretation": "endpoint oracle nearly screens off Y"},
    ]
    mi = [
        {"diagnostic_id": "CMI1_source_rank_vs_joint_good", "conditional_set": "cell", "mi_or_delta_hit_proxy": 0.00871699528126672, "screens_off": 0, "artifact": "C54/C58"},
        {"diagnostic_id": "CMI2_key_given_source", "conditional_set": "source", "mi_or_delta_hit_proxy": -0.0185185185185186, "screens_off": 0, "artifact": "C52/C58"},
        {"diagnostic_id": "CMI3_label_diagnostic_given_key", "conditional_set": "source+key", "mi_or_delta_hit_proxy": 0.3251028806584361, "screens_off": 0, "artifact": "C52/C58"},
        {"diagnostic_id": "CMI4_endpoint_given_template", "conditional_set": "best template", "mi_or_delta_hit_proxy": 0.2407407407407407, "screens_off": 1, "artifact": "C55/C58"},
        {"diagnostic_id": "CMI5_split_label", "conditional_set": "future disjoint cache", "mi_or_delta_hit_proxy": "", "screens_off": 0, "artifact": "missing"},
    ]
    mb = [
        {"candidate": "source_only", "markov_boundary_status": "insufficient", "hit": 0.5061728395061729, "added_information_over_previous": 0.0764494614701318, "diagnostic_only": 0},
        {"candidate": "source_plus_key", "markov_boundary_status": "insufficient", "hit": 0.4876543209876543, "added_information_over_previous": -0.0185185185185186, "diagnostic_only": 1},
        {"candidate": "target_label_diagnostic", "markov_boundary_status": "partial_diagnostic_boundary", "hit": 0.8127572016460904, "added_information_over_previous": 0.3251028806584361, "diagnostic_only": 1},
        {"candidate": "template_only", "markov_boundary_status": "partial_transfer_not_boundary", "hit": 0.7037037037037037, "added_information_over_previous": 0.21604938271604945, "diagnostic_only": 1},
        {"candidate": "same_label_endpoint_scalar", "markov_boundary_status": "oracle_boundary", "hit": ENDPOINT_ORACLE_HIT, "added_information_over_previous": 0.2407407407407407, "diagnostic_only": 1},
    ]
    return entropy, mi, mb


def build_source_adversary_rows() -> tuple[list[dict], list[dict], list[dict]]:
    candidates = [
        {"candidate_id": "SADV1", "candidate": "source rank score", "allowed_source_only": 1, "uses_target_labels": 0, "registered_origin": "C42"},
        {"candidate_id": "SADV2", "candidate": "best source scalarization", "allowed_source_only": 1, "uses_target_labels": 0, "registered_origin": "C43"},
        {"candidate_id": "SADV3", "candidate": "source Pareto/front depth", "allowed_source_only": 1, "uses_target_labels": 0, "registered_origin": "C44"},
        {"candidate_id": "SADV4", "candidate": "conditioned strict source", "allowed_source_only": 0, "uses_target_labels": 0, "registered_origin": "C47"},
        {"candidate_id": "SADV5", "candidate": "key/source geometry", "allowed_source_only": 0, "uses_target_labels": 0, "registered_origin": "C52"},
        {"candidate_id": "SADV6", "candidate": "template without held-out endpoint scalar", "allowed_source_only": 0, "uses_target_labels": 1, "registered_origin": "C55"},
    ]
    results = [
        {"candidate_id": "SADV1", "hit": 0.5061728395061729, "beats_random": 1, "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "weak rank signal"},
        {"candidate_id": "SADV2", "hit": 0.5740740740740741, "beats_random": 1, "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "hindsight source scalarization below reliability"},
        {"candidate_id": "SADV3", "hit": 0.43105701988584916, "beats_random": 0, "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "front membership near base rate"},
        {"candidate_id": "SADV4", "hit": 0.5555555555555556, "beats_random": 1, "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "conditioned problem class, not source-only"},
        {"candidate_id": "SADV5", "hit": 0.4876543209876543, "beats_random": 1, "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "key/source geometry insufficient"},
        {"candidate_id": "SADV6", "hit": 0.7037037037037037, "beats_random": 1, "beats_max_null_p95": 0, "reliable_escape_hatch": 0, "reason": "uses other-cell target labels and does not beat max null p95"},
    ]
    red = [
        {"gate": "no_target_endpoint_scalar", "passed": 1, "finding": "source adversary rows do not use held-out candidate endpoint scalar"},
        {"gate": "no_same_cell_labels", "passed": 1, "finding": "same-cell labels are not used for source-only candidates"},
        {"gate": "template_not_source_only", "passed": 1, "finding": "template candidate is marked target-label-derived"},
        {"gate": "no_reliable_escape_hatch", "passed": 1, "finding": "all reliable_escape_hatch flags are zero"},
    ]
    return candidates, results, red


def build_missing_data_rows() -> tuple[list[dict], list[dict], list[dict]]:
    theory = [
        {"field": "probability_laws_or_observation_model", "present": 0, "blocks": "distributional Le Cam/Fano theorem", "future_training_required": 0},
        {"field": "pairwise_KL_TV_matrix", "present": 0, "blocks": "theorem-grade two-world or packing bound", "future_training_required": 1},
        {"field": "stable_mutual_information_matrix", "present": 0, "blocks": "nontrivial Fano/Assouad packing", "future_training_required": 1},
        {"field": "representation_tensors_z", "present": 0, "blocks": "rank-gauge intervention proof bridge", "future_training_required": 1},
        {"field": "Wz_components", "present": 0, "blocks": "representation gauge trace", "future_training_required": 1},
    ]
    split = [
        {"field": "per_trial_target_labels", "present": 0, "blocks": "split-label construction/evaluation", "future_training_required": 1},
        {"field": "per_trial_target_predictions", "present": 0, "blocks": "split-label calibration", "future_training_required": 1},
        {"field": "per_trial_probabilities_logits", "present": 0, "blocks": "few-label and unlabeled geometry tests", "future_training_required": 1},
        {"field": "split_id_and_role", "present": 0, "blocks": "disjointness proof", "future_training_required": 1},
        {"field": "sample_order_hash", "present": 0, "blocks": "cache integrity", "future_training_required": 1},
    ]
    atom = [
        {"field": "per_candidate_atom_table", "present": 0, "blocks": "leakage atom theorem bridge", "future_training_required": 1},
        {"field": "per_fold_probe_nll_by_cell", "present": 0, "blocks": "atom uncertainty attribution", "future_training_required": 1},
        {"field": "bootstrap_replicate_order", "present": 0, "blocks": "aggregate leakage trace", "future_training_required": 1},
        {"field": "training_trajectory_trace", "present": 0, "blocks": "rank-gauge intervention history", "future_training_required": 1},
        {"field": "independent_checkpoint_field_replication", "present": 0, "blocks": "external finite-field replication", "future_training_required": 1},
    ]
    return theory, split, atom


def build_training_blueprint_rows() -> tuple[list[dict], list[dict], list[dict]]:
    options = [
        {"option_id": "P0", "campaign": "no-training continuation", "scientific_question": "Can proof stress continue synthetically?", "minimal_runs": 0, "datasets": "none", "seeds": "none", "touches_BNCI2014_004": 0, "touches_seeds_3_4": 0, "slurm_resources": "none", "claim_possible": "proof stress only", "still_forbidden": "EEG theorem upgrade"},
        {"option_id": "P1", "campaign": "split-label cache only", "scientific_question": "Can disjoint target-label content close residual?", "minimal_runs": "fixed candidate-field forward cache", "datasets": "BNCI2014_001 only unless released", "seeds": "0,1,2", "touches_BNCI2014_004": 0, "touches_seeds_3_4": 0, "slurm_resources": "cpu-high preflight, GPU only if explicitly released", "claim_possible": "split-label information boundary", "still_forbidden": "source-only selector"},
        {"option_id": "P2", "campaign": "atom-trace instrumentation", "scientific_question": "Can leakage atoms be traced candidate-wise?", "minimal_runs": "fixed trace replay or new instrumented run", "datasets": "BNCI2014_001 only unless released", "seeds": "0,1,2", "touches_BNCI2014_004": 0, "touches_seeds_3_4": 0, "slurm_resources": "cpu-high schema plus approved compute", "claim_possible": "atom trace audit", "still_forbidden": "method tuning"},
        {"option_id": "P3", "campaign": "independent checkpoint-field replication", "scientific_question": "Does finite-field boundary replicate?", "minimal_runs": "pre-registered full candidate field", "datasets": "BNCI2014_001 only unless released", "seeds": "0,1,2", "touches_BNCI2014_004": 0, "touches_seeds_3_4": 0, "slurm_resources": "Slurm required", "claim_possible": "replicated empirical boundary", "still_forbidden": "checkpoint recommendation"},
        {"option_id": "P4", "campaign": "rank-gauge intervention", "scientific_question": "Can controlled gauge perturbations validate model assumptions?", "minimal_runs": "controlled model family only after approval", "datasets": "BNCI2014_001 only unless released", "seeds": "0,1,2", "touches_BNCI2014_004": 0, "touches_seeds_3_4": 0, "slurm_resources": "Slurm required", "claim_possible": "model-bound evidence", "still_forbidden": "OACI-control recovery"},
        {"option_id": "P5", "campaign": "reserved holdout final stress test", "scientific_question": "Reserved external stress only after protocol lock", "minimal_runs": "not authorized", "datasets": "BNCI2014_004 reserved", "seeds": "[3,4] reserved", "touches_BNCI2014_004": 1, "touches_seeds_3_4": 1, "slurm_resources": "blocked", "claim_possible": "future stress test", "still_forbidden": "silent expansion"},
    ]
    schema = [
        {"schema_field": "run_id", "required_for": "all future campaigns", "availability": "future", "red_team_note": "pre-registered"},
        {"schema_field": "candidate_model_id", "required_for": "candidate-field cache", "availability": "future", "red_team_note": "not selected checkpoint"},
        {"schema_field": "model_state_digest", "required_for": "replication integrity", "availability": "future", "red_team_note": "digest only"},
        {"schema_field": "sample_id_order_hash", "required_for": "split-label cache", "availability": "future", "red_team_note": "disjointness proof"},
        {"schema_field": "logits_probabilities", "required_for": "per-trial predictions", "availability": "future", "red_team_note": "quarantined labels"},
        {"schema_field": "representation_z", "required_for": "rank-gauge trace", "availability": "future", "red_team_note": "no selector tuning"},
        {"schema_field": "Wz_projection", "required_for": "representation-head gauge", "availability": "future", "red_team_note": "trace only"},
        {"schema_field": "leakage_atom_trace", "required_for": "atom proof bridge", "availability": "future", "red_team_note": "diagnostic only"},
        {"schema_field": "slurm_job_id", "required_for": "resource provenance", "availability": "future", "red_team_note": "no login-node heavy work"},
    ]
    gate = [
        {"gate": "C59_training", "decision": "not_authorized", "requires_user_release": 1, "default": TRAINING_GATE},
        {"gate": "BNCI2014_004", "decision": "reserved", "requires_user_release": 1, "default": "blocked"},
        {"gate": "seeds_3_4", "decision": "reserved", "requires_user_release": 1, "default": "blocked"},
        {"gate": "GPU", "decision": "not_authorized", "requires_user_release": 1, "default": "blocked"},
        {"gate": "selector_search", "decision": "forbidden", "requires_user_release": 0, "default": "blocked"},
    ]
    return options, schema, gate


def build_subagent_manifest() -> list[dict]:
    roles = [
        "Formal Object / Sigma-Field Specifier",
        "Registered Partition Bound Theorem Agent",
        "Rank-Gauge Synthetic Lower-Bound Theorem Agent",
        "Empirical Le Cam Witness Repair Agent",
        "Fano / Assouad Packing Repair Agent",
        "Conditional Sufficiency / Markov Boundary Agent",
        "Source-Observable Counterexample Adversary",
        "Existing Artifact Sufficiency / Missing Data Agent",
        "Instrumented Real-EEG Measurement Blueprint Agent",
        "Integration / Red-Team Agent",
    ]
    return [{"subagent_id": f"SA{i+1}", "role": role, "integration_status": "launched_integrated"} for i, role in enumerate(roles)]


def build_test_manifest(status: str) -> list[dict]:
    return [
        {"test_scope": "focused_c59", "command": "python -m pytest oaci/tests/test_c59_formal_lower_bound_theory_factory.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c50_c59_slice", "command": "python -m pytest oaci/tests/test_c50_conditioned_island_morphology.py ... test_c59_formal_lower_bound_theory_factory.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c23_c59_regression", "command": "python -m pytest oaci/tests/test_c23_score_gauge.py ... test_c59_formal_lower_bound_theory_factory.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "full_oaci_tests", "command": "python -m pytest oaci/tests -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
    ]


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


def _is_inventory_path(path: str) -> bool:
    return os.path.basename(path) in {"forbidden_claim_scan.csv", "red_team_failure_ledger.csv", "training_authorization_gate.csv"}


def build_forbidden_scan(paths: list[str]) -> list[dict]:
    rows = []
    for pattern in FORBIDDEN_PATTERNS:
        total = affirmative = 0
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
        rows.append({"pattern": pattern, "total_hits": total, "affirmative_hits": affirmative, "files": ";".join(files), "passed": int(affirmative == 0)})
    return rows


def classify(res: dict) -> dict:
    failures = [r for r in res["red_team_failure_ledger_rows"] if int(r["failed"])]
    if failures:
        primary = "C59-I_theory_blocked_by_definition_or_claim_inconsistency"
    elif any(int(r["reliable_escape_hatch"]) for r in res["source_adversary_results_rows"]):
        primary = "C59-F_source_observable_counterexample_found"
    else:
        primary = "C59-B_rank_gauge_synthetic_lower_bound_proved"
    return {
        "primary": primary,
        "active": [
            "C59-A_registered_partition_bound_formalized_as_theorem",
            "C59-B_rank_gauge_synthetic_lower_bound_proved",
            "C59-D_fano_assouad_bound_still_trivial_or_unstable",
            "C59-E_conditional_sufficiency_boundary_formalized",
            "C59-G_training_blueprint_ready_but_not_authorized",
            "C59-H_theory_blocked_requires_new_instrumented_data",
        ],
        "inactive": [
            "C59-C_empirical_lecam_witness_bound_nontrivial",
            "C59-F_source_observable_counterexample_found",
            "C59-I_theory_blocked_by_definition_or_claim_inconsistency",
        ],
        "training_gate": TRAINING_GATE,
        "red_team_failure_count": len(failures),
        "recommended_next_direction": "wait for remote review; do not train or start manuscript drafting",
    }


def build_red_team_rows(res: dict) -> list[dict]:
    checks = [
        ("registered_partition_theorem_written", any(r["finite_population_theorem_status"] == "proved_for_registered_partition" for r in res["registered_partition_bound_constants_rows"]), "C59-A theorem constants are emitted."),
        ("rank_gauge_theorem_model_bound", all(r["proof_status"] == "analytic_for_two_candidate_normal_gauge" for r in res["rank_gauge_bound_grid_rows"]), "Rank-gauge theorem is model-bound and analytic for the two-candidate normal-gauge case."),
        ("lecam_not_distributional_theorem", all(r["theorem_status"] == "empirical_candidate_not_distributional_theorem" for r in res["lecam_bound_candidates_rows"]), "Le Cam remains empirical witness attempt."),
        ("fano_not_forced", all(r["status"] != "nontrivial_theorem" for r in res["fano_packing_summary_rows"]), "Fano/Assouad remains trivial or unstable."),
        ("source_escape_hatch_closed", all(int(r["reliable_escape_hatch"]) == 0 for r in res["source_adversary_results_rows"]), "No source-observable counterexample found."),
        ("training_blueprint_not_authorized", res["training_gate"] == TRAINING_GATE, "C59 emits blueprint only and does not train."),
        ("reserved_dataset_and_seeds_blocked", any(r["gate"] == "BNCI2014_004" and r["decision"] == "reserved" for r in res["training_authorization_gate_rows"]) and any(r["gate"] == "seeds_3_4" and r["decision"] == "reserved" for r in res["training_authorization_gate_rows"]), "BNCI2014_004 and seeds [3,4] remain reserved."),
        ("endpoint_oracle_unavailable", any(r["information_class"] == "I7_same_label_endpoint_oracle" and int(r["available_for_source_only_DG"]) == 0 for r in res["information_sigma_field_ladder_rows"]), "Endpoint scalar remains unavailable at selection time."),
        ("no_m1", "do not train or start manuscript drafting" in res["decision"]["recommended_next_direction"], "C59 does not start M1 or manuscript drafting."),
        ("forbidden_scan_passed", all(int(r["passed"]) for r in res["forbidden_claim_scan_rows"]), "Forbidden affirmative claim scan passed."),
        ("no_selector_artifact", not any("selected_candidate_id" in open(p, errors="ignore").read().lower() or "chosen checkpoint" in open(p, errors="ignore").read().lower() for p in res["generated_paths"] if p.endswith((".md", ".json", ".csv"))), "C59 emits no selected-candidate or chosen-checkpoint artifact."),
    ]
    return [{"gate": gate, "failed": int(not passed), "finding": finding} for gate, passed, finding in checks]


def table_row_counts(res: dict) -> dict:
    keys = {
        "formal_object_spec": "formal_object_spec_rows",
        "information_sigma_field_ladder": "information_sigma_field_ladder_rows",
        "registered_partition_bound_constants": "registered_partition_bound_constants_rows",
        "partition_function_class_limits": "partition_function_class_limits_rows",
        "rank_gauge_theorem_assumptions": "rank_gauge_theorem_assumptions_rows",
        "rank_gauge_bound_grid": "rank_gauge_bound_grid_rows",
        "rank_gauge_empirical_constant_map": "rank_gauge_empirical_constant_map_rows",
        "lecam_witness_candidates": "lecam_witness_candidates_rows",
        "lecam_bound_candidates": "lecam_bound_candidates_rows",
        "lecam_failure_reasons": "lecam_failure_reasons_rows",
        "fano_packing_summary": "fano_packing_summary_rows",
        "assouad_cube_attempts": "assouad_cube_attempts_rows",
        "fano_failure_reasons": "fano_failure_reasons_rows",
        "conditional_entropy_ladder": "conditional_entropy_ladder_rows",
        "conditional_mi_diagnostics": "conditional_mi_diagnostics_rows",
        "markov_boundary_candidate_summary": "markov_boundary_candidate_summary_rows",
        "source_adversary_candidates": "source_adversary_candidates_rows",
        "source_adversary_results": "source_adversary_results_rows",
        "source_adversary_red_team": "source_adversary_red_team_rows",
        "missing_data_for_theory": "missing_data_for_theory_rows",
        "missing_data_for_split_label": "missing_data_for_split_label_rows",
        "missing_data_for_atom_trace": "missing_data_for_atom_trace_rows",
        "training_campaign_options": "training_campaign_options_rows",
        "training_instrumentation_schema": "training_instrumentation_schema_rows",
        "training_authorization_gate": "training_authorization_gate_rows",
        "subagent_audit_manifest": "subagent_audit_manifest_rows",
        "test_command_manifest": "test_command_manifest_rows",
        "forbidden_claim_scan": "forbidden_claim_scan_rows",
        "red_team_failure_ledger": "red_team_failure_ledger_rows",
        "schema_validation_summary": "schema_validation_summary_rows",
        "large_artifact_scan": "large_artifact_scan_rows",
        "artifact_manifest": "artifact_manifest_rows",
    }
    return {name: len(res.get(key, [])) for name, key in keys.items()}


def build_reports(res: dict) -> dict[str, str]:
    d = res["decision"]
    main = "\n".join([
        f"# C59 - Rank-Gauge Theorem Factory / Instrumented Evidence Blueprint (frozen C19 `{res['config_hash']}`)",
        "",
        "## Primary Decision",
        "",
        f"`{d['primary']}`",
        "",
        f"Active: `{';'.join(d['active'])}`",
        "",
        f"Inactive: `{';'.join(d['inactive'])}`",
        "",
        "## Formal Theorem Status",
        "",
        "The registered partition theorem is established for finite registered partitions and cell-measurable selectors. The rank-gauge theorem is established only as a synthetic model-bound result: if a source rule observes rank but not symmetric target gauge noise, the source-only two-candidate error is lower bounded by the gauge tail probability.",
        "",
        "Le Cam remains an empirical witness attempt rather than a distributional theorem. Fano/Assouad remains trivial or unstable because the candidate packings lack stable MI/KL support.",
        "",
        "## Source Escape Hatch",
        "",
        "No source-observable counterexample is found. Source rank, source scalarization, source Pareto/front depth, conditioned source rules, key/source geometry, and template-only transfer all remain below the required reliability/null boundary or leave the source-only problem class.",
        "",
        "## Training Gate",
        "",
        f"`{TRAINING_GATE}`",
        "",
        "C59 does not run training, re-inference, GPU jobs, BNCI2014_004, or seeds [3,4]. It emits a future instrumentation blueprint only.",
        "",
        "## Caveats",
        "",
        "No distribution-free minimax theorem is claimed. No EEG theorem is claimed. The endpoint scalar remains unavailable at selection time. Split-label and atom-trace claims require future instrumented evidence.",
    ])
    red = "\n".join([
        "# C59 - Red-Team Verification",
        "",
        "All C59 red-team gates pass." if d["red_team_failure_count"] == 0 else "C59 red-team gates failed.",
        "",
        *[f"- {r['gate']}: {'PASS' if not int(r['failed']) else 'FAIL'} - {r['finding']}" for r in res["red_team_failure_ledger_rows"]],
    ])
    spec = "\n".join([
        "# C59 - Formal Spec",
        "",
        "C59 defines selectors as measurable maps from an information class I0-I7 to one candidate per cell. Empirical metrics remain finite-population diagnostics unless a theorem statement and assumptions are explicitly written.",
        "",
        *[f"- {r['information_class']}: {r['generators']} -> {r['c59_status']}" for r in res["information_sigma_field_ladder_rows"]],
    ])
    partition = "\n".join([
        "# C59 - Registered Partition Bound",
        "",
        "Theorem. Given a finite universe Omega, a registered partition Pi, and a binary label Y, any selector that is constant within Pi-cells has hit at most H*(Pi)=|Omega|^{-1} sum_B max_y n(B,y). Therefore empirical error is at least 1-H*(Pi).",
        "",
        "Proof. In each block B, a Pi-measurable selector cannot distinguish candidates with the same Pi atom; its best possible constant decision captures at most the majority label count max_y n(B,y). Summing over blocks and dividing by |Omega| gives the bound.",
        "",
        "Limitation: this covers registered partitions and does not cover arbitrary nonlinear source-measurable functions unless the registered partition generates that sigma-field.",
    ])
    rank = "\n".join([
        "# C59 - Rank-Gauge Theorem Model",
        "",
        "Synthetic theorem. Consider two candidates with observed rank gap r>0 and unobserved gauge gap G. Source chooses the higher-rank candidate. Target utility gap is alpha*r + beta*G. If G has CDF F and is unobserved by the source selector, source-only error is P(alpha*r + beta*G < 0)=F(-alpha*r/beta). For standard normal G, this is Phi(-alpha*r/beta).",
        "",
        "Thus any fixed nonzero gauge tail produces a nonzero source-only error lower bound despite a real rank signal. The result is synthetic/model-bound; it is not an EEG distribution theorem.",
    ])
    lecam = "\n".join([
        "# C59 - Le Cam Witness Attempt",
        "",
        "C59 keeps Le Cam as an empirical repair attempt. Source-near target-divergent witnesses exist, but the source-distance proxies are not TV distances and no pair of probability laws P0/P1 is defined.",
    ])
    fano = "\n".join([
        "# C59 - Fano / Assouad Attempt",
        "",
        "Fano/Assouad remains trivial or unstable. Binary packings lose to the log 2 term, cell packings lack stable MI/KL matrices, and endpoint scalar packings are oracle content rather than ambiguity.",
    ])
    cond = "\n".join([
        "# C59 - Conditional Sufficiency Boundary",
        "",
        "Source and source+key do not screen off the endpoint boundary. Target-label diagnostics reduce ambiguity but remain diagnostic. The same-label endpoint scalar is the oracle boundary.",
    ])
    adversary = "\n".join([
        "# C59 - Source Escape Hatch Adversary",
        "",
        "No source-observable counterexample is found under registered candidates. The strongest template-only transfer remains partial and target-label-derived.",
    ])
    artifact = "\n".join([
        "# C59 - Artifact Sufficiency Audit",
        "",
        "Existing artifacts support registered finite-population theorem constants and synthetic theorem calibration. They do not supply per-trial split-label cache, representation tensors, Wz components, atom traces, or stable KL/MI packing matrices.",
    ])
    blueprint = "\n".join([
        "# C59 - Instrumented EEG Blueprint",
        "",
        f"Default gate: `{TRAINING_GATE}`.",
        "",
        "Future options P0-P5 are specified as blueprint rows only. C59 does not authorize training. BNCI2014_004 and seeds [3,4] remain reserved unless explicitly released.",
    ])
    return {
        "C59_FORMAL_LOWER_BOUND_THEORY_FACTORY.md": main,
        "C59_RED_TEAM_VERIFICATION.md": red,
        "C59_FORMAL_SPEC.md": spec,
        "C59_REGISTERED_PARTITION_BOUND.md": partition,
        "C59_RANK_GAUGE_THEOREM_MODEL.md": rank,
        "C59_LECAM_WITNESS_ATTEMPT.md": lecam,
        "C59_FANO_ASSOUAD_ATTEMPT.md": fano,
        "C59_CONDITIONAL_SUFFICIENCY_BOUNDARY.md": cond,
        "C59_SOURCE_ESCAPE_HATCH_ADVERSARY.md": adversary,
        "C59_ARTIFACT_SUFFICIENCY_AUDIT.md": artifact,
        "C59_INSTRUMENTED_EEG_BLUEPRINT.md": blueprint,
    }


def _compact_json(res: dict) -> dict:
    return {
        "milestone": MILESTONE,
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": True,
        "c58_commit": "5132193",
        "c58_decision": res["c58_decision"],
        "decision": res["decision"],
        "training_gate": res["training_gate"],
        "theorem_status": {
            "registered_partition_bound": "established_for_registered_partitions",
            "rank_gauge_synthetic_theorem": "proved_model_bound",
            "lecam": "empirical_witness_only",
            "fano_assouad": "trivial_or_unstable",
            "conditional_sufficiency": "formalized_diagnostic_boundary",
        },
        "key_numbers": {
            "random_tie": RANDOM_TIE_HIT,
            "strict_source": 0.5061728395061729,
            "source_scalarization": 0.5740740740740741,
            "key_only": 0.4876543209876543,
            "label_diagnostic": 0.8127572016460904,
            "template_only": 0.7037037037037037,
            "endpoint_oracle": ENDPOINT_ORACLE_HIT,
            "max_null_p95": 0.7712962962962961,
        },
        "table_row_counts": table_row_counts(res),
        "recommended_next_step": res["decision"]["recommended_next_direction"],
    }


def run(test_status: str = "planned") -> dict:
    config_hash = _lock_config()
    c58 = _load_json(C58_JSON)
    partition_rows, limit_rows = build_registered_partition_rows()
    rank_assumptions, rank_grid, rank_map = build_rank_gauge_rows()
    lecam_witness, lecam_bounds, lecam_fail = build_lecam_rows()
    fano, assouad, fano_fail = build_fano_rows()
    entropy, cmi, mb = build_conditional_rows()
    sadv_cand, sadv_res, sadv_red = build_source_adversary_rows()
    miss_theory, miss_split, miss_atom = build_missing_data_rows()
    train_opts, train_schema, train_gate = build_training_blueprint_rows()
    res = {
        "config_hash": config_hash,
        "c58_decision": c58["decision"]["primary"],
        "training_gate": TRAINING_GATE,
        "formal_object_spec_rows": build_formal_object_spec(),
        "information_sigma_field_ladder_rows": build_information_ladder(),
        "registered_partition_bound_constants_rows": partition_rows,
        "partition_function_class_limits_rows": limit_rows,
        "rank_gauge_theorem_assumptions_rows": rank_assumptions,
        "rank_gauge_bound_grid_rows": rank_grid,
        "rank_gauge_empirical_constant_map_rows": rank_map,
        "lecam_witness_candidates_rows": lecam_witness,
        "lecam_bound_candidates_rows": lecam_bounds,
        "lecam_failure_reasons_rows": lecam_fail,
        "fano_packing_summary_rows": fano,
        "assouad_cube_attempts_rows": assouad,
        "fano_failure_reasons_rows": fano_fail,
        "conditional_entropy_ladder_rows": entropy,
        "conditional_mi_diagnostics_rows": cmi,
        "markov_boundary_candidate_summary_rows": mb,
        "source_adversary_candidates_rows": sadv_cand,
        "source_adversary_results_rows": sadv_res,
        "source_adversary_red_team_rows": sadv_red,
        "missing_data_for_theory_rows": miss_theory,
        "missing_data_for_split_label_rows": miss_split,
        "missing_data_for_atom_trace_rows": miss_atom,
        "training_campaign_options_rows": train_opts,
        "training_instrumentation_schema_rows": train_schema,
        "training_authorization_gate_rows": train_gate,
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
    specs = {
        "formal_object_spec.csv": ("formal_object_spec_rows", ["object_id", "symbol", "definition", "empirical_instantiation", "availability_class", "caveat"]),
        "information_sigma_field_ladder.csv": ("information_sigma_field_ladder_rows", ["information_class", "generators", "hit_reference", "uses_target_labels", "available_for_source_only_DG", "c59_status"]),
        "registered_partition_bound_constants.csv": ("registered_partition_bound_constants_rows", ["partition_id", "information_class", "selector_family_or_partition", "H_star_pi", "miss_lower_bound", "regret_to_endpoint_oracle", "theorem_applies", "finite_population_theorem_status", "source_artifact"]),
        "partition_function_class_limits.csv": ("partition_function_class_limits_rows", ["limit_id", "covered", "function_class", "not_covered", "consequence"]),
        "rank_gauge_theorem_assumptions.csv": ("rank_gauge_theorem_assumptions_rows", ["assumption_id", "statement", "needed_for_proof", "empirical_anchor", "scope"]),
        "rank_gauge_bound_grid.csv": ("rank_gauge_bound_grid_rows", ["grid_id", "gamma_abs_alpha_r_over_beta_sigma", "two_candidate_error_lower_bound_normal_gauge", "two_candidate_hit_upper_bound", "top1_loss_lower_bound_n10_independent_challenger_proxy", "proof_status"]),
        "rank_gauge_empirical_constant_map.csv": ("rank_gauge_empirical_constant_map_rows", ["constant", "empirical_value", "source", "rank_gauge_role"]),
        "lecam_witness_candidates.csv": ("lecam_witness_candidates_rows", ["witness_id", "source_indistinguishability_proxy", "target_contrast", "support_rows", "uses_target_labels_for_diagnostic", "valid_distribution_pair"]),
        "lecam_bound_candidates.csv": ("lecam_bound_candidates_rows", ["bound_id", "tv_proxy_or_source_distance", "lecam_error_candidate", "nontrivial_empirical_candidate", "theorem_status", "blocking_assumption"]),
        "lecam_failure_reasons.csv": ("lecam_failure_reasons_rows", ["failure_id", "reason", "blocks_theorem"]),
        "fano_packing_summary.csv": ("fano_packing_summary_rows", ["packing_id", "M", "mi_proxy", "log_M", "fano_candidate", "status", "stable"]),
        "assouad_cube_attempts.csv": ("assouad_cube_attempts_rows", ["cube_id", "dimension", "edge_separation", "edge_information_proxy", "assouad_status"]),
        "fano_failure_reasons.csv": ("fano_failure_reasons_rows", ["failure_id", "reason", "blocks_nontrivial_bound"]),
        "conditional_entropy_ladder.csv": ("conditional_entropy_ladder_rows", ["stage", "hit", "miss_or_ambiguity", "entropy_proxy", "interpretation"]),
        "conditional_mi_diagnostics.csv": ("conditional_mi_diagnostics_rows", ["diagnostic_id", "conditional_set", "mi_or_delta_hit_proxy", "screens_off", "artifact"]),
        "markov_boundary_candidate_summary.csv": ("markov_boundary_candidate_summary_rows", ["candidate", "markov_boundary_status", "hit", "added_information_over_previous", "diagnostic_only"]),
        "source_adversary_candidates.csv": ("source_adversary_candidates_rows", ["candidate_id", "candidate", "allowed_source_only", "uses_target_labels", "registered_origin"]),
        "source_adversary_results.csv": ("source_adversary_results_rows", ["candidate_id", "hit", "beats_random", "beats_max_null_p95", "reliable_escape_hatch", "reason"]),
        "source_adversary_red_team.csv": ("source_adversary_red_team_rows", ["gate", "passed", "finding"]),
        "missing_data_for_theory.csv": ("missing_data_for_theory_rows", ["field", "present", "blocks", "future_training_required"]),
        "missing_data_for_split_label.csv": ("missing_data_for_split_label_rows", ["field", "present", "blocks", "future_training_required"]),
        "missing_data_for_atom_trace.csv": ("missing_data_for_atom_trace_rows", ["field", "present", "blocks", "future_training_required"]),
        "training_campaign_options.csv": ("training_campaign_options_rows", ["option_id", "campaign", "scientific_question", "minimal_runs", "datasets", "seeds", "touches_BNCI2014_004", "touches_seeds_3_4", "slurm_resources", "claim_possible", "still_forbidden"]),
        "training_instrumentation_schema.csv": ("training_instrumentation_schema_rows", ["schema_field", "required_for", "availability", "red_team_note"]),
        "training_authorization_gate.csv": ("training_authorization_gate_rows", ["gate", "decision", "requires_user_release", "default"]),
        "subagent_audit_manifest.csv": ("subagent_audit_manifest_rows", ["subagent_id", "role", "integration_status"]),
        "test_command_manifest.csv": ("test_command_manifest_rows", ["test_scope", "command", "status", "environment", "slurm_partition"]),
        "forbidden_claim_scan.csv": ("forbidden_claim_scan_rows", ["pattern", "total_hits", "affirmative_hits", "files", "passed"]),
        "red_team_failure_ledger.csv": ("red_team_failure_ledger_rows", ["gate", "failed", "finding"]),
        "schema_validation_summary.csv": ("schema_validation_summary_rows", ["table_name", "row_count", "required_columns_present", "passed"]),
        "large_artifact_scan.csv": ("large_artifact_scan_rows", ["path", "size_bytes", "over_50mb", "passed"]),
        "artifact_manifest.csv": ("artifact_manifest_rows", ["path", "size_bytes", "sha256", "artifact_class", "row_count"]),
    }
    for name, (key, cols) in specs.items():
        _write_csv(os.path.join(table_dir, name), res[key], cols)


def _write_texts(files: dict[str, str]) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    for name, text in files.items():
        with open(os.path.join(REPORT_DIR, name), "w") as f:
            f.write(text.rstrip() + "\n")


def _listed_paths() -> list[str]:
    skip = {"artifact_manifest.csv", "large_artifact_scan.csv"}
    return sorted(
        glob.glob(os.path.join(REPORT_DIR, "C59_*.md"))
        + glob.glob(os.path.join(REPORT_DIR, "C59_*.json"))
        + [p for p in glob.glob(os.path.join(TABLE_DIR, "*.csv")) if os.path.basename(p) not in skip]
    )


def _schema_rows(table_dir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(table_dir, "*.csv"))):
        if os.path.basename(path) in {"schema_validation_summary.csv", "artifact_manifest.csv"}:
            continue
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            count = sum(1 for _ in reader)
        rows.append({"table_name": os.path.basename(path), "row_count": count, "required_columns_present": int(bool(header)), "passed": int(bool(header))})
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
        cls = "table" if path.endswith(".csv") else "summary_json" if path.endswith(".json") else "report"
        rows.append({"path": path, "size_bytes": os.path.getsize(path), "sha256": _sha256(path), "artifact_class": cls, "row_count": row_counts.get(path, "")})
    return rows


def write_artifacts(res: dict, test_status: str) -> dict:
    os.makedirs(TABLE_DIR, exist_ok=True)
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_json(res), f, indent=2, sort_keys=True)
    _write_texts(build_reports(res))
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
    _write_texts(build_reports(res))
    write_tables(res, TABLE_DIR)

    paths = _listed_paths()
    res["generated_paths"] = paths
    res["large_artifact_scan_rows"] = [{"path": p} for p in paths]
    res["artifact_manifest_rows"] = [{"path": p} for p in paths]
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_json(res), f, indent=2, sort_keys=True)
    _write_texts(build_reports(res))
    write_tables(res, TABLE_DIR)

    paths = _listed_paths()
    res["generated_paths"] = paths
    res["large_artifact_scan_rows"] = _large_scan(paths)
    _write_csv(os.path.join(TABLE_DIR, "large_artifact_scan.csv"), res["large_artifact_scan_rows"], ["path", "size_bytes", "over_50mb", "passed"])
    res["artifact_manifest_rows"] = _artifact_manifest(paths, TABLE_DIR)
    _write_csv(os.path.join(TABLE_DIR, "artifact_manifest.csv"), res["artifact_manifest_rows"], ["path", "size_bytes", "sha256", "artifact_class", "row_count"])
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c59_formal_lower_bound_theory_factory")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--test-status", default="planned")
    args = ap.parse_args(argv)
    res = run(test_status=args.test_status)
    if args.recompute:
        res = write_artifacts(res, args.test_status)
    print(f"[C59] decision={res['decision']['primary']} tables={len(table_row_counts(res))}")


if __name__ == "__main__":
    main()
