"""C85P protocol registry and locked synthetic-contract materializer.

The module validates and renders prospective contracts.  It deliberately does
not execute the locked S0-S10 scenarios or complete any proof obligation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .c85_decision_experiments import DecisionContractError
from .c85_lower_bound_contracts import TheoremStatus


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci" / "reports"
TABLE_DIR = REPORT_DIR / "c85p_tables"
PROTOCOL_PATH = REPORT_DIR / "C85_TPAMI_DECISION_THEORY_PROTOCOL.json"
PROTOCOL_SIDECAR = REPORT_DIR / "C85_TPAMI_DECISION_THEORY_PROTOCOL.sha256"
GENERATOR_PATH = TABLE_DIR / "synthetic_generator_contract.json"
ADDENDUM_PATH = REPORT_DIR / "C84A_PM_REALIZED_POLICY_USE_ADDENDUM.json"
ADDENDUM_SIDECAR = REPORT_DIR / "C84A_PM_REALIZED_POLICY_USE_ADDENDUM.sha256"

EXPECTED_PROTOCOL_SHA256 = "af4c2cb35a6b6555d6c9ded3105eb7ad4f061ba237d3e8cc3ed6f5a18aede006"
EXPECTED_GENERATOR_SHA256 = "c87fec6a6572291fad8849f6c08bea2cb3f49467e243ded1d44c1f38e3d0b297"
EXPECTED_ADDENDUM_JSON_SHA256 = "6dd1d03d9f15f3cf45ba594e6946ebfc556bd3902559bd36f995cd10145734c5"
EXPECTED_SCENARIOS = tuple(f"S{index}" for index in range(11))
SUCCESS_GATE = "C85_TPAMI_DECISION_THEORY_PROTOCOL_LOCKED_READY_FOR_PROOF_AND_SYNTHETIC_EXECUTION"
FAILURE_GATE = "C85_DECISION_THEORY_ASSUMPTION_THEOREM_STATUS_OR_EMPIRICAL_BRIDGE_RECONCILIATION_REQUIRED"
PROTOCOL_ONLY = "LOCKED_PROTOCOL_ONLY_NOT_EXECUTED"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _sidecar_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        entries[name.strip()] = digest
    return entries


def validate_locked_inputs() -> dict[str, Any]:
    protocol_sha = sha256_file(PROTOCOL_PATH)
    generator_sha = sha256_file(GENERATOR_PATH)
    addendum_sha = sha256_file(ADDENDUM_PATH)
    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise DecisionContractError("C85 protocol bytes drifted after chronology lock")
    if generator_sha != EXPECTED_GENERATOR_SHA256:
        raise DecisionContractError("C85 synthetic generator contract drifted")
    if addendum_sha != EXPECTED_ADDENDUM_JSON_SHA256:
        raise DecisionContractError("C84A PM addendum bytes drifted")
    protocol_sidecar = _sidecar_entries(PROTOCOL_SIDECAR)
    addendum_sidecar = _sidecar_entries(ADDENDUM_SIDECAR)
    if protocol_sidecar.get(PROTOCOL_PATH.name) != protocol_sha:
        raise DecisionContractError("C85 protocol sidecar mismatch")
    if addendum_sidecar.get(ADDENDUM_PATH.name) != addendum_sha:
        raise DecisionContractError("C84A addendum sidecar mismatch")

    protocol = read_json(PROTOCOL_PATH)
    contract = read_json(GENERATOR_PATH)
    addendum = read_json(ADDENDUM_PATH)
    if protocol["protocol_status"] != "LOCKED_PROSPECTIVE_TO_C85T_PROOFS_AND_SYNTHETIC_RESULTS":
        raise DecisionContractError("C85 protocol is not prospectively locked")
    if contract["status"] != "LOCKED_IN_C85P_NOT_EXECUTED_UNTIL_C85T":
        raise DecisionContractError("synthetic contract execution boundary drifted")
    scenario_ids = tuple(row["id"] for row in contract["scenarios"])
    if scenario_ids != EXPECTED_SCENARIOS or len(set(scenario_ids)) != 11:
        raise DecisionContractError("S0-S10 coverage or order drifted")
    if contract.get("outcome_informed_design") is not False:
        raise DecisionContractError("synthetic parameters must be outcome independent")
    if any(row["status_at_C85P"] != TheoremStatus.OPEN.value for row in protocol["theorem_targets"]):
        raise DecisionContractError("C85P may not mark a theorem target proved")
    if len(protocol["theorem_targets"]) != 7:
        raise DecisionContractError("T1-T7 theorem target registry is incomplete")
    if addendum["frozen_science"]["taxonomy_changed"] is not False:
        raise DecisionContractError("C84 PM addendum cannot change taxonomy")
    if addendum["protected_counters"] != {key: 0 for key in addendum["protected_counters"]}:
        raise DecisionContractError("C84A addendum protected counter is nonzero")
    return {
        "protocol": protocol,
        "generator": contract,
        "addendum": addendum,
        "protocol_sha256": protocol_sha,
        "generator_sha256": generator_sha,
        "addendum_sha256": addendum_sha,
    }


def deterministic_seed(scenario_id: str, replicate_id: int) -> int:
    if scenario_id not in EXPECTED_SCENARIOS or replicate_id < 0:
        raise DecisionContractError("invalid synthetic seed key")
    payload = f"C85_SYNTHETIC_V1|{scenario_id}|{replicate_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _rows(*rows: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = [dict(row) for row in rows]
    if not result:
        raise DecisionContractError("registry cannot be empty")
    fields = tuple(result[0])
    if any(tuple(row) != fields for row in result):
        raise DecisionContractError("registry row schemas differ")
    return result


def _terminology_rows() -> list[dict[str, Any]]:
    values = [
        ("statistical_experiment", "A measurable observation space and state-indexed family of laws.", "E=(Z,{P_theta^E})", "L01|L02"),
        ("Blackwell_order", "One experiment is obtainable from another by an observation-only randomization; equivalently under the theorem conditions it is no better in every decision problem.", "information comparison, not frozen-policy comparison", "L01"),
        ("Le_Cam_deficiency", "Directed distance measuring how closely one experiment can simulate another through randomization.", "future approximate experiment comparison", "L02"),
        ("Bayes_risk", "Prior-averaged expected loss of a decision rule.", "R_mean under a state prior", "L01|L02"),
        ("minimax_regret", "Choose a rule minimizing the maximum loss relative to the statewise best action over an ambiguity set.", "finite identified-set LP", "L07|L08"),
        ("partial_identification", "Observables and assumptions restrict a target to a set rather than a singleton.", "utility identified set U(z)", "L07"),
        ("distributionally_robust_optimization", "Optimize risk against a registered set of distributions or groups.", "future robust selector objective", "L13"),
        ("worst_group_risk", "Maximum conditional expected loss over predeclared groups.", "R_worst", "L13"),
        ("CVaR_expected_shortfall", "Average upper-tail loss represented by the Rockafellar-Uryasev variational form.", "R_CVaR,alpha", "L05|L06"),
        ("best_arm_identification", "Pure-exploration identification of the action with best mean under arm-wise observations.", "contrast class; not the C85 full-information query", "L11"),
        ("top_k_identification", "Identify a set containing the best actions under an explicit error criterion.", "near-optimal/top-k endpoint", "L11"),
        ("near_optimal_action_set", "Actions within epsilon utility of the statewise optimum.", "A_epsilon(theta)", "PROJECT_DEFINITION"),
        ("Rashomon_set", "A prespecified class of models with near-optimal predictive performance.", "analogy for candidate multiplicity, not an empirical C84 geometry claim", "L09"),
        ("predictive_multiplicity", "Multiple similarly performing models can make different predictions or decisions.", "future action-geometry audit", "L09"),
        ("active_testing", "Select test examples to label for sample-efficient model evaluation with design correction.", "future full-information label acquisition", "L14"),
        ("active_model_selection", "Allocate a budget of model probes to identify the best model.", "related but observation structure differs from C85 labels", "L12"),
        ("experimental_design", "Choose observations or allocations to optimize a registered inferential objective.", "future pairwise-difference variance design", "L14"),
        ("Hill_effective_size", "Effective number associated with a diversity order; order two is inverse squared mass.", "gap-softmax action multiplicity", "L10"),
    ]
    return _rows(*(
        {"term": term, "standard_definition": definition, "c85_mapping": mapping, "source_ids": sources, "status": "STANDARD_TERM_NO_NEW_FIELD"}
        for term, definition, mapping, sources in values
    ))


def _empirical_motivation_rows() -> list[dict[str, Any]]:
    return _rows(
        {"motivation_id": "M1", "frozen_label": "COTT_like_average_tail_separation", "compact_observation": "positive mean and Q2 pass with registered target-tail floor failures", "allowed_role": "motivate mean/worst/CVaR separation", "forbidden_use": "choose alpha or robust threshold", "confirmatory_gate_changed": 0},
        {"motivation_id": "M2", "frozen_label": "MaNo_like_policy_collapse", "compact_observation": "Cho MaNo is exactly B1-equivalent in 160/160 contexts", "allowed_role": "motivate realized action dependence", "forbidden_use": "claim experiment contains no information", "confirmatory_gate_changed": 0},
        {"motivation_id": "M3", "frozen_label": "Cho_like_cohort_specific_qualification", "compact_observation": "qualified fixed policy in one cohort only", "allowed_role": "motivate environment-indexed risk", "forbidden_use": "set synthetic state probabilities", "confirmatory_gate_changed": 0},
        {"motivation_id": "M4", "frozen_label": "label_frontier_heterogeneity", "compact_observation": "B* absent/8/absent under registered Q0", "allowed_role": "motivate information-policy separation", "forbidden_use": "infer label information order", "confirmatory_gate_changed": 0},
    )


def _state_action_rows() -> list[dict[str, Any]]:
    values = [
        ("SA1", "state", "theta in Theta", "latent environment determining observation law and utilities"),
        ("SA2", "action", "a in {1,...,M}", "one frozen candidate choice"),
        ("SA3", "utility", "u_theta(a) in [0,1]", "common bounded action utility"),
        ("SA4", "optimal_action", "argmax_a u_theta(a)", "ties retained as a set"),
        ("SA5", "selection_loss", "max_j u_theta(j)-u_theta(a)", "nonnegative regret loss"),
        ("SA6", "experiment", "E=(Z,{P_theta^E})", "state-indexed observation law"),
        ("SA7", "decision_rule", "delta:Z->P(A)", "randomization allowed"),
        ("SA8", "dataset", "d", "highest empirical cohort grouping"),
        ("SA9", "target_group", "g", "robust-risk group"),
        ("SA10", "environment_distribution", "Pi", "state/group population law"),
    ]
    return _rows(*({"contract_id": i, "object": o, "symbol": s, "definition": d, "status": "LOCKED"} for i, o, s, d in values))


def _experiment_rows() -> list[dict[str, Any]]:
    return _rows(
        {"experiment_id": "E0", "standard_class": "NO_INFORMATION", "observation": "constant sigma-field", "comparison_role": "baseline", "real_data_instantiated": 0},
        {"experiment_id": "EU", "standard_class": "TARGET_UNLABELED", "observation": "registered unlabeled target summary", "comparison_role": "algorithmic input class", "real_data_instantiated": 0},
        {"experiment_id": "EL_B", "standard_class": "PASSIVE_FULL_INFORMATION_LABEL_QUERY", "observation": "B queried trial loss vectors", "comparison_role": "costly-label experiment", "real_data_instantiated": 0},
        {"experiment_id": "EFULL", "standard_class": "FULL_INFORMATION", "observation": "registered complete information idealization", "comparison_role": "theory reference only", "real_data_instantiated": 0},
        {"experiment_id": "EGARBLED", "standard_class": "MARKOV_GARBLING", "observation": "randomized transform of a richer experiment", "comparison_role": "Blackwell witness", "real_data_instantiated": 0},
    )


def _policy_rows() -> list[dict[str, Any]]:
    return _rows(
        {"class_id": "D_ALL", "definition": "all measurable randomized rules under E", "nested_requirement": "unrestricted", "optimization_status": "population infimum", "empirical_equivalent": "NONE"},
        {"class_id": "DELTA_REGISTERED", "definition": "prospectively registered rule class", "nested_requirement": "not assumed across experiments", "optimization_status": "may be restricted or imperfect", "empirical_equivalent": "frozen method family analogy only"},
        {"class_id": "DELTA_FIXED", "definition": "one fixed action rule", "nested_requirement": "singleton", "optimization_status": "no optimization", "empirical_equivalent": "B1-like reference"},
        {"class_id": "DELTA_PURE", "definition": "deterministic measurable rules", "nested_requirement": "subset of randomized rules", "optimization_status": "finite extreme points", "empirical_equivalent": "NONE"},
        {"class_id": "DELTA_RANDOMIZED", "definition": "Markov kernels from observation to action", "nested_requirement": "contains pure rules", "optimization_status": "simplex optimization", "empirical_equivalent": "NONE"},
    )


def _realized_policy_rows() -> list[dict[str, Any]]:
    return _rows(
        {"quantity_id": "RP1", "name": "action_divergence", "definition": "P[delta_E(Z_E)!=delta_0] under an explicit coupling/realized-action convention", "interpretation": "realized departure from reference", "information_content_claim": 0},
        {"quantity_id": "RP2", "name": "action_entropy", "definition": "H(delta_E(Z_E))", "interpretation": "realized action diversity", "information_content_claim": 0},
        {"quantity_id": "RP3", "name": "incremental_fixed_policy_risk_value", "definition": "R(delta_0)-R(delta_E)", "interpretation": "value realized through one fixed policy", "information_content_claim": 0},
        {"quantity_id": "RP4", "name": "policy_collapse", "definition": "delta_E(Z_E)=delta_0 almost surely", "interpretation": "identical realized action losses under common evaluation distribution", "information_content_claim": 0},
    )


def _policy_collapse_rows() -> list[dict[str, Any]]:
    return _rows(
        {"theorem_id": "T3", "premise": "delta_E(Z_E)=delta_0 almost surely under the evaluation law", "conclusion": "fixed policies have identical action loss and risk; incremental realized action value is zero", "does_not_imply": "experiment equivalence, absence of information, or inability of another rule to use E", "randomization_obligation": "state coupling convention explicitly", "status": "OPEN"},
    )


def _robust_risk_rows() -> list[dict[str, Any]]:
    return _rows(
        {"risk_id": "R_MEAN", "name": "mean_risk", "definition": "E_Pi[L]", "parameter": "Pi", "orientation": "lower_is_better", "c84_threshold_reused": 0},
        {"risk_id": "R_WORST", "name": "worst_group_risk", "definition": "sup_g E[L|g]", "parameter": "registered group set G", "orientation": "lower_is_better", "c84_threshold_reused": 0},
        {"risk_id": "R_CVAR", "name": "upper_tail_CVaR", "definition": "inf_eta eta+E[(L-eta)_+]/(1-alpha)", "parameter": "symbolic alpha in (0,1)", "orientation": "lower_is_better", "c84_threshold_reused": 0},
        {"risk_id": "R_DRO", "name": "distributionally_robust_risk", "definition": "sup_{Q in registered ambiguity set} E_Q[L]", "parameter": "prospectively registered ambiguity set", "orientation": "lower_is_better", "c84_threshold_reused": 0},
    )


def _partial_identification_rows() -> list[dict[str, Any]]:
    return _rows(
        {"contract_id": "PI1", "component": "identified_set", "definition": "U(z)={u compatible with observation and assumptions}", "condition": "closed nonempty subset of [0,1]^M", "empirical_instantiation": "NONE"},
        {"contract_id": "PI2", "component": "compatibility", "definition": "observation-law, moment, support and structural restrictions named in assumption lattice", "condition": "no hidden C84 utility intervals", "empirical_instantiation": "NONE"},
        {"contract_id": "PI3", "component": "point_identification", "definition": "U(z) is a singleton", "condition": "diameter zero", "empirical_instantiation": "NONE"},
        {"contract_id": "PI4", "component": "partial_identification", "definition": "U(z) contains multiple utility vectors", "condition": "positive infinity-norm diameter", "empirical_instantiation": "NONE"},
        {"contract_id": "PI5", "component": "randomization", "definition": "q in action simplex allowed", "condition": "pure restriction reported separately", "empirical_instantiation": "NONE"},
    )


def _minimax_rows() -> list[dict[str, Any]]:
    return _rows(
        {"problem_id": "MMR1", "action_class": "pure", "objective": "min_a sup_{u in U(z)} max_j u_j-u_a", "finite_form": "enumerate actions and registered extreme points", "solver_status": "CONTRACT_ONLY"},
        {"problem_id": "MMR2", "action_class": "randomized", "objective": "min_q sup_u sum_a q_a(max_j u_j-u_a)", "finite_form": "LP: min t; q>=0; sum q=1; scenario regret constraints", "solver_status": "CONTRACT_ONLY"},
        {"problem_id": "MMR3", "action_class": "point_identified", "objective": "choose argmax utility", "finite_form": "zero minimax regret when optimum is known", "solver_status": "CONTRACT_ONLY"},
    )


def _assumption_rows() -> list[dict[str, Any]]:
    return _rows(
        {"assumption_id": "A0", "parent": "NONE", "statement": "finite nonempty action set and bounded utilities", "used_by": "T1|T2|T3|T4|T5|T6|T7", "empirically_verified": 0},
        {"assumption_id": "A1", "parent": "A0", "statement": "common state/action/loss spaces across compared experiments", "used_by": "T1|T2", "empirically_verified": 0},
        {"assumption_id": "A2", "parent": "A1", "statement": "coarser experiment is an observation-only Markov garbling", "used_by": "T1|T2", "empirically_verified": 0},
        {"assumption_id": "A3", "parent": "A0", "statement": "equal prior on two states with distinct optimal actions and wrong-action loss at least Delta", "used_by": "T4", "empirically_verified": 0},
        {"assumption_id": "A4", "parent": "A0", "statement": "finite separated multi-state packing and bounded information", "used_by": "T5", "empirically_verified": 0},
        {"assumption_id": "A5", "parent": "A0", "statement": "registered groups and population masses", "used_by": "T6", "empirically_verified": 0},
        {"assumption_id": "A6", "parent": "A0", "statement": "pairwise utility-estimation errors obey explicit sub-Gaussian tail bounds", "used_by": "T7", "empirically_verified": 0},
        {"assumption_id": "A7", "parent": "A6", "statement": "dependence handled by valid union bound; independence not silently assumed", "used_by": "T7", "empirically_verified": 0},
        {"assumption_id": "A8", "parent": "A0", "statement": "identified set is nonempty and represented by registered extreme points for finite LP", "used_by": "MMR2", "empirically_verified": 0},
    )


def _theorem_rows(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _rows(*(
        {"theorem_id": row["id"], "name": row["name"], "target": row["target"], "literature_class": row["literature_class"], "status": row["status_at_C85P"], "c85t_obligation": row["c85t_obligation"], "proof_executed_C85P": 0}
        for row in protocol["theorem_targets"]
    ))


def _lower_bound_rows() -> list[dict[str, Any]]:
    return _rows(
        {"theorem_id": "T4", "method": "two_state_Le_Cam", "candidate_statement": "equal-prior average regret >= (Delta/2)(1-TV(P0,P1))", "required_constant_audit": "derive testing reduction and randomization convention", "status": "OPEN", "may_remain_open_after_C85T": 0},
        {"theorem_id": "T5", "method": "finite_Fano_packing", "candidate_statement": "packing error lower bound times registered regret separation", "required_constant_audit": "define packing, mutual information bound and log-cardinality convention", "status": "OPEN", "may_remain_open_after_C85T": 1},
    )


def _proof_obligation_rows() -> list[dict[str, Any]]:
    obligations = [
        ("PO1", "T1", "common spaces|Markov kernel|all measurable randomized rules", "construct composed rule and compare risks", "OPEN"),
        ("PO2", "T2", "finite exact channels|non-nested registered classes", "enumerate all states/observations/actions and risks", "OPEN"),
        ("PO3", "T3", "almost-sure action equality|common loss", "prove pointwise loss equality then integrate", "OPEN"),
        ("PO4", "T4", "two states|equal prior|action separation|TV convention", "prove reduction to binary testing and exact constant", "OPEN"),
        ("PO5", "T5", "finite packing|loss separation|information control", "prove Fano specialization or leave open", "OPEN"),
        ("PO6", "T6", "finite groups|common group masses|symbolic alpha intervals", "enumerate mean/worst/CVaR inequalities", "OPEN"),
        ("PO7", "T7", "sub-Gaussian pairwise errors|gap definitions", "prove outside-set event inclusion and valid probability bound", "OPEN"),
    ]
    return _rows(*({"obligation_id": i, "theorem_id": t, "assumptions": a, "required_work": w, "status": s, "proof_artifact": "C85T_REQUIRED"} for i, t, a, w, s in obligations))


def _theorem_status_rows() -> list[dict[str, Any]]:
    return _rows(*(
        {"status": status.value, "meaning": {
            "PROVED": "complete proof with audited assumptions",
            "PROVED_FINITE_MODEL_ONLY": "complete proof restricted to declared finite model",
            "COUNTEREXAMPLE": "finite counterexample exhaustively verified",
            "CONJECTURE": "plausible target without proof",
            "OPEN": "registered obligation not completed",
            "INVALIDATED": "registered target disproved or internally inconsistent",
        }[status.value], "allowed_at_C85P_for_T1_T7": int(status is TheoremStatus.OPEN)}
        for status in TheoremStatus
    ))


def _counterexample_rows() -> list[dict[str, Any]]:
    return _rows(
        {"counterexample_id": "CE_T2", "theorem_id": "T2", "scenario_id": "S1", "locked_object": "binary perfect experiment versus constant garbling with non-nested fixed policies", "parameter_source": "prospective simple rational construction", "status": "OPEN", "executed": 0},
        {"counterexample_id": "CE_T2_GAP", "theorem_id": "T2", "scenario_id": "S10", "locked_object": "three-state policy-approximation reversal", "parameter_source": "prospective simple rational construction", "status": "OPEN", "executed": 0},
        {"counterexample_id": "CE_T6", "theorem_id": "T6", "scenario_id": "S5", "locked_object": "nine favorable groups and one adverse group", "parameter_source": "prospective simple rational construction", "status": "OPEN", "executed": 0},
    )


def _near_optimal_rows() -> list[dict[str, Any]]:
    return _rows(
        {"geometry_id": "NG1", "object": "gap", "definition": "Delta_a=u_star-u_a", "parameter_status": "state dependent", "raw_M_substitute_allowed": 0},
        {"geometry_id": "NG2", "object": "epsilon_near_optimal_set", "definition": "{a:Delta_a<=epsilon}", "parameter_status": "epsilon symbolic in theory", "raw_M_substitute_allowed": 0},
        {"geometry_id": "NG3", "object": "soft_gap_weights", "definition": "exp(-Delta_a/tau)/normalizer", "parameter_status": "tau symbolic in theory", "raw_M_substitute_allowed": 0},
        {"geometry_id": "NG4", "object": "outside_set_error", "definition": "P[selected action not in A_epsilon]", "parameter_status": "bound target T7", "raw_M_substitute_allowed": 0},
    )


def _multiplicity_rows() -> list[dict[str, Any]]:
    return _rows(
        {"summary_id": "EM1", "name": "near_optimal_count", "formula": "|A_epsilon|", "origin": "near-optimal action set", "parameter": "epsilon>0", "status": "LOCKED"},
        {"summary_id": "EM2", "name": "Hill_2_effective_size", "formula": "1/sum_a w_a(tau)^2", "origin": "Hill number order 2", "parameter": "tau>0", "status": "LOCKED"},
        {"summary_id": "EM3", "name": "entropy_effective_size", "formula": "exp(-sum_a w_a(tau)log w_a(tau))", "origin": "Hill number order 1", "parameter": "tau>0", "status": "LOCKED"},
    )


def _near_tie_rows() -> list[dict[str, Any]]:
    return _rows(
        {"bound_id": "NT1", "theorem_id": "T7", "error_assumption": "pairwise estimation error is sub-Gaussian with registered scale sigma_a", "event": "selected outside A_epsilon", "candidate_bound": "min(1,sum_{Delta_a>epsilon} exp(-(Delta_a-epsilon)^2/(2 sigma_a^2)))", "dependence_handling": "union bound; no independence required", "status": "OPEN"},
    )


def _tail_counterexample_rows() -> list[dict[str, Any]]:
    return _rows(
        {"counterexample_id": "TAIL_T6", "scenario_id": "S5", "groups": 10, "favorable_groups": 9, "reference_losses": "0.5 repeated 10", "policy_losses": "0.3 repeated 9|1.0", "alpha_policy": "locked intervals; no C84 alpha", "expected_relation": "mean improves; worst worsens; upper-tail relation audited in C85T", "status": "OPEN"},
    )


def _robust_objective_rows() -> list[dict[str, Any]]:
    return _rows(
        {"objective_id": "RO1", "name": "mean_regret", "formula": "E[L]", "prospective_use": "baseline", "replaces_C84_Q1_floor": 0, "execution_authorized": 0},
        {"objective_id": "RO2", "name": "worst_group_regret", "formula": "sup_g E[L|g]", "prospective_use": "tail-robust policy class", "replaces_C84_Q1_floor": 0, "execution_authorized": 0},
        {"objective_id": "RO3", "name": "CVaR_regret", "formula": "CVaR_alpha(L)", "prospective_use": "upper-tail policy class", "replaces_C84_Q1_floor": 0, "execution_authorized": 0},
        {"objective_id": "RO4", "name": "DRO_regret", "formula": "sup_Q E_Q[L]", "prospective_use": "distributional ambiguity", "replaces_C84_Q1_floor": 0, "execution_authorized": 0},
    )


def _costly_label_rows() -> list[dict[str, Any]]:
    return _rows(
        {"contract_id": "CL1", "query_unit": "one target trial", "observation": "full loss vector L_t for all M frozen candidates", "cost": "one label", "standard_bandit_equivalent": 0, "training_allowed": 0},
        {"contract_id": "CL2", "query_unit": "one arm pull", "observation": "one selected arm loss", "cost": "one pull", "standard_bandit_equivalent": 1, "training_allowed": 0},
        {"contract_id": "CL3", "query_unit": "batch passive labels", "observation": "stratified full-information loss vectors", "cost": "number of labels", "standard_bandit_equivalent": 0, "training_allowed": 0},
    )


def _active_method_rows() -> list[dict[str, Any]]:
    return _rows(
        {"method_id": "AT0", "standard_name": "passive_stratified_sampling", "acquisition_signal": "fixed stratum allocation", "estimation_correction": "design weights where required", "endpoint": "selection regret|top-k", "authorized": 0},
        {"method_id": "AT1", "standard_name": "active_testing", "acquisition_signal": "prospectively registered evaluation uncertainty", "estimation_correction": "importance weighting", "endpoint": "model evaluation and selection", "authorized": 0},
        {"method_id": "AT2", "standard_name": "pairwise_difference_variance_design", "acquisition_signal": "variance of candidate loss differences", "estimation_correction": "design-aware", "endpoint": "best/top-k identification", "authorized": 0},
        {"method_id": "AT3", "standard_name": "consensus_disagreement_acquisition", "acquisition_signal": "candidate prediction disagreement", "estimation_correction": "prospectively specified", "endpoint": "selection regret", "authorized": 0},
        {"method_id": "AT4", "standard_name": "sequential_stopping", "acquisition_signal": "registered confidence or regret certificate", "estimation_correction": "anytime-valid requirement", "endpoint": "stopping cost and selection regret", "authorized": 0},
    )


def _future_active_rows() -> list[dict[str, Any]]:
    requirements = [
        ("FA1", "untouched target population", "avoid outcome-informed acquisition design"),
        ("FA2", "fixed candidate zoo", "query value must not alter training or retention"),
        ("FA3", "query-cost unit", "one label reveals all candidate losses"),
        ("FA4", "acquisition information set", "no held-evaluation leakage"),
        ("FA5", "importance/design weighting", "unbiased or explicitly targeted estimand"),
        ("FA6", "stopping rule", "prospectively bounded error/cost"),
        ("FA7", "passive comparator", "same labels, population and endpoint"),
        ("FA8", "authorization", "separate protocol, lock and direct PI statement"),
    ]
    return _rows(*({"requirement_id": i, "requirement": r, "reason": reason, "satisfied_in_C85P": 0, "blocks_execution_now": 1} for i, r, reason in requirements))


def _scenario_rows(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _rows(*(
        {"scenario_id": row["id"], "name": row["name"], "state_or_group_count": len(row.get("states", row.get("groups", []))), "action_count": len(row.get("actions", row.get("utilities", [[None]])[0])), "sample_size": row["sample_size"], "risk_functionals": "|".join(row["risk_functionals"]), "theorem_targets": "|".join(row["theorem_targets"]), "success_criterion": row["success_criterion"], "seed0": deterministic_seed(row["id"], 0), "status": "LOCKED_NOT_EXECUTED"}
        for row in contract["scenarios"]
    ))


def _synthetic_validation_rows(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _rows(*(
        {"scenario_id": row["id"], "schema_valid": 1, "seed_rule_bound": 1, "parameters_locked": 1, "outcome_informed": 0, "scientific_execution": 0, "result_status": "CONTRACT_VALIDATED_NOT_EXECUTED", "next_stage": "C85T_AFTER_PM_REVIEW"}
        for row in contract["scenarios"]
    ))


def _guard_rows() -> list[dict[str, Any]]:
    objects = ["theorem_constants", "CVaR_alpha", "epsilon", "tau", "synthetic_gaps", "state_probabilities", "sample_sizes", "active_policy_hyperparameters", "new_empirical_thresholds"]
    return _rows(*({"guard_id": f"OG{index}", "design_object": value, "C84_outcome_may_determine": 0, "prospective_source": "C85P_protocol_or_future_untouched_protocol", "status": "PASS"} for index, value in enumerate(objects, 1)))


def _literature_rows() -> list[dict[str, Any]]:
    values = [
        ("L01", "Blackwell", 1953, "Equivalent Comparisons of Experiments", "Annals of Mathematical Statistics", "10.1214/aoms/1177729032", "https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-24/issue-2/Equivalent-Comparisons-of-Experiments/10.1214/aoms/1177729032.full", "Blackwell comparison and decision-problem equivalence"),
        ("L02", "Le Cam", 1986, "Asymptotic Methods in Statistical Decision Theory", "Springer Series in Statistics", "10.1007/978-1-4612-4946-7", "https://link.springer.com/book/10.1007/978-1-4612-4946-7", "deficiency and statistical experiments"),
        ("L03", "Yu", 1997, "Assouad, Fano, and Le Cam", "Festschrift for Lucien Le Cam", "10.1007/978-1-4612-1880-7_29", "https://doi.org/10.1007/978-1-4612-1880-7_29", "relationships among minimax lower-bound methods"),
        ("L04", "Yang and Barron", 1999, "Information-Theoretic Determination of Minimax Rates of Convergence", "Annals of Statistics", "10.1214/aos/1017939142", "https://projecteuclid.org/journals/annals-of-statistics/volume-27/issue-5/Information-theoretic-determination-of-minimax-rates-of-convergence/10.1214/aos/1017939142.full", "Fano and information-theoretic minimax bounds"),
        ("L05", "Rockafellar and Uryasev", 2000, "Optimization of Conditional Value-at-Risk", "Journal of Risk", "10.21314/JOR.2000.038", "https://doi.org/10.21314/JOR.2000.038", "CVaR optimization representation"),
        ("L06", "Rockafellar and Uryasev", 2002, "Conditional Value-at-Risk for General Loss Distributions", "Journal of Banking and Finance", "10.1016/S0378-4266(02)00271-6", "https://doi.org/10.1016/S0378-4266(02)00271-6", "CVaR for discrete/general loss laws"),
        ("L07", "Manski", 2000, "Identification Problems and Decisions under Ambiguity", "Journal of Econometrics", "10.1016/S0304-4076(99)00045-7", "https://doi.org/10.1016/S0304-4076(99)00045-7", "partial identification and decisions under ambiguity"),
        ("L08", "Manski", 2004, "Statistical Treatment Rules for Heterogeneous Populations", "Econometrica", "10.1111/j.1468-0262.2004.00530.x", "https://doi.org/10.1111/j.1468-0262.2004.00530.x", "minimax-regret statistical treatment rules"),
        ("L09", "Fisher, Rudin, and Dominici", 2019, "All Models are Wrong, but Many are Useful", "JMLR", "NONE", "https://jmlr.org/papers/v20/18-760.html", "Rashomon set and model-class multiplicity"),
        ("L10", "Hill", 1973, "Diversity and Evenness: A Unifying Notation and Its Consequences", "Ecology", "10.2307/1934352", "https://doi.org/10.2307/1934352", "effective diversity/Hill numbers"),
        ("L11", "Kaufmann, Cappe, and Garivier", 2016, "On the Complexity of Best-Arm Identification in Multi-Armed Bandit Models", "JMLR", "NONE", "https://jmlr.org/papers/v17/kaufman16a.html", "best-arm/top-m identification under arm pulls"),
        ("L12", "Madani, Lizotte, and Greiner", 2004, "Active Model Selection", "UAI", "NONE", "https://www.csd.uwo.ca/~dlizotte/publications/madani04active.pdf", "budgeted active model probes"),
        ("L13", "Sagawa, Koh, Hashimoto, and Liang", 2020, "Distributionally Robust Neural Networks for Group Shifts", "ICLR", "NONE", "https://openreview.net/forum?id=ryxGuJrFvS", "worst-group risk and group DRO"),
        ("L14", "Kossen, Farquhar, Gal, and Rainforth", 2021, "Active Testing: Sample-Efficient Model Evaluation", "ICML/PMLR", "NONE", "https://proceedings.mlr.press/v139/kossen21a.html", "active test-label acquisition and bias correction"),
    ]
    return _rows(*(
        {"source_id": source_id, "authors": authors, "year": year, "title": title, "venue": venue, "doi": doi, "primary_url": url, "verified_contribution": contribution, "verification_status": "VERIFIED_PRIMARY_OR_CANONICAL", "conditions_imported_without_audit": 0}
        for source_id, authors, year, title, venue, doi, url, contribution in values
    ))


def _proof_audit_rows() -> list[dict[str, Any]]:
    return _rows(
        {"theorem_id": "T1", "classification": "CLASSICAL_THEOREM_PROJECT_PROOF_PENDING", "source_ids": "L01", "assumptions_registered": 1, "constants_verified": "NA", "project_proof_complete": 0, "status": "OPEN", "overclaim": 0},
        {"theorem_id": "T2", "classification": "PROJECT_COUNTEREXAMPLE_TARGET", "source_ids": "L01", "assumptions_registered": 1, "constants_verified": 0, "project_proof_complete": 0, "status": "OPEN", "overclaim": 0},
        {"theorem_id": "T3", "classification": "ELEMENTARY_PROPOSITION_TARGET", "source_ids": "NONE", "assumptions_registered": 1, "constants_verified": "NA", "project_proof_complete": 0, "status": "OPEN", "overclaim": 0},
        {"theorem_id": "T4", "classification": "CLASSICAL_METHOD_PROJECT_SPECIALIZATION", "source_ids": "L02|L03", "assumptions_registered": 1, "constants_verified": 0, "project_proof_complete": 0, "status": "OPEN", "overclaim": 0},
        {"theorem_id": "T5", "classification": "CLASSICAL_METHOD_PROJECT_SPECIALIZATION", "source_ids": "L03|L04", "assumptions_registered": 1, "constants_verified": 0, "project_proof_complete": 0, "status": "OPEN", "overclaim": 0},
        {"theorem_id": "T6", "classification": "PROJECT_COUNTEREXAMPLE_TARGET", "source_ids": "L05|L06|L13", "assumptions_registered": 1, "constants_verified": 0, "project_proof_complete": 0, "status": "OPEN", "overclaim": 0},
        {"theorem_id": "T7", "classification": "PROJECT_BOUND_TARGET", "source_ids": "L11", "assumptions_registered": 1, "constants_verified": 0, "project_proof_complete": 0, "status": "OPEN", "overclaim": 0},
    )


def _counterexample_check_rows() -> list[dict[str, Any]]:
    return _rows(
        {"counterexample_id": "CE_T2", "scenario_id": "S1", "enumeration_space_locked": 1, "exhaustive_execution": 0, "algebraic_proof": 0, "status": "NOT_EXECUTED_C85T_REQUIRED"},
        {"counterexample_id": "CE_T2_GAP", "scenario_id": "S10", "enumeration_space_locked": 1, "exhaustive_execution": 0, "algebraic_proof": 0, "status": "NOT_EXECUTED_C85T_REQUIRED"},
        {"counterexample_id": "CE_T6", "scenario_id": "S5", "enumeration_space_locked": 1, "exhaustive_execution": 0, "algebraic_proof": 0, "status": "NOT_EXECUTED_C85T_REQUIRED"},
    )


def _risk_rows() -> list[dict[str, Any]]:
    risks = [
        ("R01", "classical theorem attribution exceeds verified conditions", "require source and project proof separation"),
        ("R02", "OPEN proof called proved", "status validator rejects PROVED during C85P"),
        ("R03", "C84 outcome selects theory parameter", "outcome-informed design guard"),
        ("R04", "restricted-policy reversal called information reversal", "claim and terminology registry"),
        ("R05", "policy collapse called no-information theorem", "T3 non-implication contract"),
        ("R06", "CVaR tail convention ambiguous", "upper-loss variational definition"),
        ("R07", "worst target substituted for CVaR", "separate risk functional registry"),
        ("R08", "raw candidate count called effective multiplicity", "gap-weighted registry"),
        ("R09", "Le Cam proxy treated as TV", "T4 requires probability laws and exact TV"),
        ("R10", "Fano sketch called theorem", "T5 may remain OPEN"),
        ("R11", "full-information label query treated as bandit pull", "costly-label distinction"),
        ("R12", "active method silently authorized", "all active rows authorized=0"),
        ("R13", "real project arrays imported", "static import and path audit"),
        ("R14", "synthetic contract executed during C85P", "validation status contract-only"),
        ("R15", "manuscript prose modified", "artifact scope and Git audit"),
    ]
    return _rows(*({"risk_id": i, "risk": risk, "mitigation": mitigation, "blocking_if_open": 1, "status": "CLOSED_AT_READINESS"} for i, risk, mitigation in risks))


def _failure_rows() -> list[dict[str, Any]]:
    return _rows(
        {"reason_id": "NONE_OPEN", "category": "READINESS", "status": "CLOSED", "blocking": 0, "detail": "No open C85P protocol, theorem-status, empirical-bridge, or artifact blocker after validation."},
    )


def build_tables() -> dict[str, list[dict[str, Any]]]:
    locked = validate_locked_inputs()
    protocol = locked["protocol"]
    contract = locked["generator"]
    return {
        "terminology_registry.csv": _terminology_rows(),
        "empirical_motivation_registry.csv": _empirical_motivation_rows(),
        "state_action_loss_contract.csv": _state_action_rows(),
        "information_experiment_registry.csv": _experiment_rows(),
        "policy_class_registry.csv": _policy_rows(),
        "realized_policy_use_registry.csv": _realized_policy_rows(),
        "policy_collapse_theorem_contract.csv": _policy_collapse_rows(),
        "robust_risk_functional_registry.csv": _robust_risk_rows(),
        "partial_identification_contract.csv": _partial_identification_rows(),
        "minimax_regret_problem_registry.csv": _minimax_rows(),
        "assumption_lattice.csv": _assumption_rows(),
        "theorem_registry.csv": _theorem_rows(protocol),
        "lower_bound_theorem_registry.csv": _lower_bound_rows(),
        "proof_obligation_registry.csv": _proof_obligation_rows(),
        "theorem_status_registry.csv": _theorem_status_rows(),
        "counterexample_registry.csv": _counterexample_rows(),
        "near_optimal_geometry_contract.csv": _near_optimal_rows(),
        "effective_multiplicity_registry.csv": _multiplicity_rows(),
        "near_tie_bound_contract.csv": _near_tie_rows(),
        "tail_risk_counterexample_registry.csv": _tail_counterexample_rows(),
        "robust_selector_objective_registry.csv": _robust_objective_rows(),
        "costly_label_experiment_contract.csv": _costly_label_rows(),
        "active_testing_method_registry.csv": _active_method_rows(),
        "future_active_protocol_requirements.csv": _future_active_rows(),
        "synthetic_scenario_registry.csv": _scenario_rows(contract),
        "synthetic_validation_matrix.csv": _synthetic_validation_rows(contract),
        "outcome_informed_design_guard.csv": _guard_rows(),
        "literature_source_registry.csv": _literature_rows(),
        "proof_audit.csv": _proof_audit_rows(),
        "counterexample_exhaustive_check.csv": _counterexample_check_rows(),
        "risk_register.csv": _risk_rows(),
        "failure_reason_ledger.csv": _failure_rows(),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise DecisionContractError(f"refusing to write empty table {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise DecisionContractError(f"schema drift in {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def materialize_tables() -> dict[str, int]:
    tables = build_tables()
    for name, rows in tables.items():
        write_csv(TABLE_DIR / name, rows)
    return {name: len(rows) for name, rows in tables.items()}


def validate_materialized_tables() -> dict[str, int]:
    expected = build_tables()
    observed: dict[str, int] = {}
    for name, rows in expected.items():
        path = TABLE_DIR / name
        if not path.is_file():
            raise DecisionContractError(f"missing materialized C85P table {name}")
        with path.open(newline="") as handle:
            actual = list(csv.DictReader(handle))
        canonical_expected = [{key: str(value) for key, value in row.items()} for row in rows]
        if actual != canonical_expected:
            raise DecisionContractError(f"materialized C85P table drifted: {name}")
        observed[name] = len(actual)
    return observed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-contract")
    subparsers.add_parser("materialize-tables")
    args = parser.parse_args(argv)
    if args.command == "validate-contract":
        locked = validate_locked_inputs()
        print(json.dumps({
            "protocol_sha256": locked["protocol_sha256"],
            "generator_sha256": locked["generator_sha256"],
            "scenarios": len(locked["generator"]["scenarios"]),
            "status": PROTOCOL_ONLY,
        }, sort_keys=True))
        return 0
    counts = materialize_tables()
    print(json.dumps({"tables": counts, "status": PROTOCOL_ONLY}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
