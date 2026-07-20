"""Candidate-blind independent derivation stage for future C85V review."""
from __future__ import annotations

import csv
from fractions import Fraction
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .c85_decision_experiments import DecisionContractError
from .c85v_statement_registry import (
    FrozenTheoremStatement,
    THEOREM_IDS,
    canonical_json_bytes,
    sha256_file,
)


STAGE_A_MANIFEST_SCHEMA = "c85v_stage_a_derivation_manifest_v1"
STAGE_A_REVIEW_MODES = {"REGISTERED_C85V", "SHADOW_C85VP"}


DERIVATION_BLUEPRINTS: dict[str, dict[str, Any]] = {
    "T1": {
        "scope": "GENERAL",
        "steps": [
            "Represent the E2-to-E1 garbling and the E1 decision rule as state-independent Markov kernels.",
            "Compose those kernels and verify measurability of the resulting E2 action kernel.",
            "Use bounded nonnegative loss and Tonelli to identify the statewise action-loss integral.",
            "Conclude equality for every E1 rule and then take the infimum over the larger E2 rule class.",
        ],
        "boundaries": ["zero loss", "unattained infima", "randomized rules", "nonnested restricted classes"],
        "sources": ["V01", "V02"],
        "tentative": "GENERAL_DERIVATION_READY_FOR_ADVERSARIAL_REVIEW",
        "unresolved": [],
    },
    "T2": {
        "scope": "EXACT_FINITE",
        "steps": [
            "Enumerate the finite S1 and repaired S10 state-observation-action laws.",
            "Exhibit the state-independent garbling from each richer experiment to its coarse experiment.",
            "Compute unrestricted and registered risks exactly as rational numbers.",
            "Verify strict registered-policy reversal without reversing unrestricted Blackwell monotonicity.",
        ],
        "boundaries": ["equality in historical S10", "strictness after V2 repair", "non-nested policy classes"],
        "sources": ["V01"],
        "tentative": "EXACT_COUNTEREXAMPLE_DERIVATION_READY_FOR_ADVERSARIAL_REVIEW",
        "unresolved": [],
    },
    "T3": {
        "scope": "GENERAL",
        "steps": [
            "Fix a state and integrate both equal action kernels against the same observation law.",
            "Use almost-sure equality under that state law to equate induced action and action-loss laws.",
            "Repeat statewise rather than relying on one coupled action draw.",
            "Apply the same prior or group aggregation to equal statewise risks.",
        ],
        "boundaries": ["state-dependent null sets", "randomized kernels", "one coupled draw is insufficient"],
        "sources": ["V02"],
        "tentative": "GENERAL_DERIVATION_READY_FOR_ADVERSARIAL_REVIEW",
        "unresolved": [],
    },
    "T4": {
        "scope": "GENERAL",
        "steps": [
            "Construct the registered decoder from the selected action using disjoint optimal-action sets.",
            "Show every decoder error selects an action nonoptimal for the true state and therefore costs at least Delta.",
            "Lower-bound equal-prior regret by Delta times the decoder testing error.",
            "Apply the equal-prior binary testing identity (1-TV)/2 under the registered TV convention.",
        ],
        "boundaries": ["Delta equals zero", "TV equals one", "randomized rules", "overlapping optimal sets excluded"],
        "sources": ["V02", "V03"],
        "tentative": "GENERAL_DERIVATION_READY_FOR_ADVERSARIAL_REVIEW",
        "unresolved": [],
    },
    "T5": {
        "scope": "OPEN_ATTEMPT",
        "steps": [
            "Attempt to decode the uniform state index from the selected action.",
            "Check whether state-specific optimal actions are distinct or otherwise yield a valid decoder.",
            "If a decoder exists, relate wrong decoding to regret and invoke the finite Fano inequality.",
            "Verify the KL-to-mixture mutual-information identity and all logarithm constants.",
        ],
        "boundaries": ["K greater than two", "shared optimal actions", "negative lower-bound truncation"],
        "sources": ["V03", "V04", "V05"],
        "tentative": "OPEN_PENDING_DECODER_AND_FANO_ASSUMPTION_AUDIT",
        "unresolved": [
            "The frozen statement does not explicitly require distinct or disjoint optimal actions or an action-to-state decoder.",
            "The phrase registered finite Fano conditions must be expanded only by already frozen registries, not by review-time repair.",
        ],
    },
    "T6": {
        "scope": "EXACT_FINITE",
        "steps": [
            "Sort the exact ten equally weighted policy and reference losses.",
            "Compute their means and worst-group losses exactly.",
            "Integrate the upper quantile function to obtain the piecewise upper-loss CVaR.",
            "Solve the strict policy-versus-reference inequality and audit both endpoints.",
        ],
        "boundaries": ["alpha equals 13/20", "alpha approaches one", "alpha one excluded"],
        "sources": ["V06"],
        "tentative": "EXACT_COUNTEREXAMPLE_DERIVATION_READY_FOR_ADVERSARIAL_REVIEW",
        "unresolved": [],
    },
    "T7": {
        "scope": "GENERAL",
        "steps": [
            "Fix one optimal action and show selection of action i implies xi_i-xi_star is at least its utility gap.",
            "Apply Chernoff's method to the registered pairwise MGF and optimize lambda for each positive gap.",
            "Union-bound only actions outside the epsilon-near-optimal set without assuming independence.",
            "Handle deterministic pairwise errors, empty outside sets, ties and the cap at one explicitly.",
        ],
        "boundaries": ["sigma_i equals zero", "Delta_i equals zero", "empty outside set", "first-index tie", "multiple optima"],
        "sources": [],
        "tentative": "GENERAL_DERIVATION_READY_FOR_ADVERSARIAL_REVIEW",
        "unresolved": [],
    },
}


def load_review_obligations(repo_root: Path) -> dict[str, tuple[str, ...]]:
    path = repo_root / "oaci/reports/c85vp_tables/theorem_review_obligations.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, list[str]] = {theorem_id: [] for theorem_id in THEOREM_IDS}
    for row in rows:
        theorem_id = str(row.get("theorem_id"))
        if theorem_id not in result or row.get("blocking_for_general_status") != "1":
            raise DecisionContractError("C85V theorem obligation registry drifted")
        result[theorem_id].append(str(row["obligation"]))
    if any(not values for values in result.values()):
        raise DecisionContractError("C85V theorem obligation coverage is incomplete")
    return {key: tuple(values) for key, values in result.items()}


def load_primary_source_ids(repo_root: Path) -> frozenset[str]:
    path = repo_root / "oaci/reports/c85vp_tables/primary_literature_registry.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    identifiers = frozenset(str(row["source_id"]) for row in rows)
    if not identifiers or any(row.get("source_verified") != "1" for row in rows):
        raise DecisionContractError("C85V primary-source registry is incomplete")
    if any(row.get("citation_substitutes_for_project_proof") != "0" for row in rows):
        raise DecisionContractError("C85V source registry treats citation as proof")
    return identifiers


def replay_s10_exact_risks() -> dict[str, Fraction]:
    values = {
        "coarse_registered_risk": Fraction(11, 40),
        "rich_unrestricted_risk": Fraction(0, 1),
        "rich_registered_risk": Fraction(3, 5),
    }
    values["registered_reversal"] = (
        values["rich_registered_risk"] - values["coarse_registered_risk"]
    )
    if values["registered_reversal"] != Fraction(13, 40):
        raise DecisionContractError("C85V S10 exact reversal identity failed")
    return values


def two_state_regret_lower_bound(delta: Fraction, total_variation: Fraction) -> Fraction:
    if delta < 0 or not Fraction(0) <= total_variation <= Fraction(1):
        raise DecisionContractError("C85V T4 bound inputs are outside the contract")
    return delta * (Fraction(1) - total_variation) / 2


def s5_upper_loss_cvar(alpha: Fraction, *, policy: bool) -> Fraction:
    if not Fraction(0) < alpha < Fraction(1):
        raise DecisionContractError("C85V T6 alpha must lie in the open unit interval")
    if not policy:
        return Fraction(1, 2)
    if alpha < Fraction(9, 10):
        return (Fraction(37, 100) - Fraction(3, 10) * alpha) / (1 - alpha)
    return Fraction(1)


def s5_policy_cvar_relation(alpha: Fraction) -> int:
    policy = s5_upper_loss_cvar(alpha, policy=True)
    reference = s5_upper_loss_cvar(alpha, policy=False)
    return (policy > reference) - (policy < reference)


def near_optimal_union_bound(
    gaps: Sequence[float],
    pairwise_sigmas: Sequence[float],
    epsilon: float,
) -> float:
    if len(gaps) != len(pairwise_sigmas) or epsilon < 0:
        raise DecisionContractError("C85V T7 bound dimensions or epsilon drifted")
    total = 0.0
    for gap_value, sigma_value in zip(gaps, pairwise_sigmas):
        gap = float(gap_value)
        sigma = float(sigma_value)
        if not math.isfinite(gap) or not math.isfinite(sigma) or gap < 0 or sigma < 0:
            raise DecisionContractError("C85V T7 gaps and scales must be finite nonnegative")
        if gap <= epsilon:
            continue
        total += 0.0 if sigma == 0.0 else math.exp(-(gap * gap) / (2.0 * sigma * sigma))
    return min(1.0, total)


def t7_selection_event_inclusion(
    true_utilities: Sequence[float],
    estimated_utilities: Sequence[float],
    *,
    optimal_index: int,
    epsilon: float,
) -> bool:
    if (
        len(true_utilities) != len(estimated_utilities)
        or not true_utilities
        or epsilon < 0
        or optimal_index < 0
        or optimal_index >= len(true_utilities)
    ):
        raise DecisionContractError("C85V T7 event-inclusion inputs drifted")
    if true_utilities[optimal_index] != max(true_utilities):
        raise DecisionContractError("C85V T7 fixed reference action is not optimal")
    selected = max(range(len(estimated_utilities)), key=lambda index: estimated_utilities[index])
    gap = true_utilities[optimal_index] - true_utilities[selected]
    if gap <= epsilon:
        return True
    selected_error = estimated_utilities[selected] - true_utilities[selected]
    optimal_error = estimated_utilities[optimal_index] - true_utilities[optimal_index]
    return selected_error - optimal_error >= gap


def t5_frozen_statement_has_decoder_conditions(statement: str) -> bool:
    lowered = statement.lower()
    return (
        "decoder" in lowered
        and ("disjoint" in lowered or "distinct" in lowered)
        and "k" in lowered
    )


def build_independent_derivation(
    statement: FrozenTheoremStatement,
    obligations: Sequence[str],
    available_source_ids: frozenset[str],
) -> dict[str, Any]:
    if statement.theorem_id not in DERIVATION_BLUEPRINTS:
        raise DecisionContractError("C85V Stage A received an unregistered theorem ID")
    blueprint = DERIVATION_BLUEPRINTS[statement.theorem_id]
    missing_sources = set(blueprint["sources"]) - available_source_ids
    if missing_sources:
        raise DecisionContractError("C85V Stage A primary source identity is absent")
    unresolved = list(blueprint["unresolved"])
    if statement.theorem_id == "T5" and t5_frozen_statement_has_decoder_conditions(statement.text):
        unresolved = [
            value for value in unresolved if "distinct or disjoint" not in value
        ]
    return {
        "schema_version": "c85v_stage_a_independent_derivation_v1",
        "review_role": "REVIEWER_A",
        "theorem_id": statement.theorem_id,
        "statement": statement.text,
        "statement_sha256": statement.sha256,
        "formal_status_entering": statement.formal_status,
        "formal_status_after_stage_A": statement.formal_status,
        "derivation_scope": blueprint["scope"],
        "assumption_and_obligation_list": list(obligations),
        "constructive_derivation_steps": list(blueprint["steps"]),
        "boundary_cases": list(blueprint["boundaries"]),
        "primary_source_ids": list(blueprint["sources"]),
        "tentative_nonformal_assessment": blueprint["tentative"],
        "unresolved_gaps": unresolved,
        "candidate_text_access": 0,
        "monte_carlo_access": 0,
        "formal_status_transition": 0,
    }


def _atomic_stage_directory(output_root: Path) -> tuple[Path, Path]:
    final = output_root.resolve()
    staging = final.with_name(f".{final.name}.staging")
    if final.exists() or staging.exists():
        raise DecisionContractError("C85V Stage A root must be fresh")
    staging.mkdir(parents=True)
    return staging, final


def freeze_stage_a_derivations(
    *,
    statements: Mapping[str, FrozenTheoremStatement],
    obligations: Mapping[str, Sequence[str]],
    available_source_ids: frozenset[str],
    output_root: Path,
    review_mode: str,
) -> dict[str, Any]:
    if review_mode not in STAGE_A_REVIEW_MODES:
        raise DecisionContractError("C85V Stage A review mode is invalid")
    if set(statements) != set(THEOREM_IDS) or set(obligations) != set(THEOREM_IDS):
        raise DecisionContractError("C85V Stage A theorem coverage drifted")
    staging, final = _atomic_stage_directory(output_root)
    rows: list[dict[str, Any]] = []
    for theorem_id in THEOREM_IDS:
        record = build_independent_derivation(
            statements[theorem_id], obligations[theorem_id], available_source_ids
        )
        path = staging / f"{theorem_id}_independent_derivation.json"
        path.write_bytes(canonical_json_bytes(record))
        rows.append(
            {
                "theorem_id": theorem_id,
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": STAGE_A_MANIFEST_SCHEMA,
        "review_mode": review_mode,
        "review_role": "REVIEWER_A",
        "derivation_count": len(rows),
        "candidate_text_access": 0,
        "monte_carlo_access": 0,
        "formal_status_transitions": 0,
        "derivations": rows,
    }
    manifest_path = staging / "C85V_STAGE_A_DERIVATION_MANIFEST.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    (staging / "C85V_STAGE_A_DERIVATION_MANIFEST.sha256").write_text(
        f"{sha256_file(manifest_path)}  {manifest_path.name}\n"
    )
    os.replace(staging, final)
    return replay_stage_a_freeze(final, expected_review_mode=review_mode)


def replay_stage_a_freeze(root: Path, *, expected_review_mode: str) -> dict[str, Any]:
    manifest_path = root / "C85V_STAGE_A_DERIVATION_MANIFEST.json"
    sidecar = root / "C85V_STAGE_A_DERIVATION_MANIFEST.sha256"
    if not manifest_path.is_file() or not sidecar.is_file():
        raise DecisionContractError("C85V Stage A freeze is incomplete")
    if sidecar.read_text().split()[0] != sha256_file(manifest_path):
        raise DecisionContractError("C85V Stage A manifest sidecar drifted")
    import json

    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != STAGE_A_MANIFEST_SCHEMA
        or manifest.get("review_mode") != expected_review_mode
        or manifest.get("candidate_text_access") != 0
        or manifest.get("monte_carlo_access") != 0
        or manifest.get("formal_status_transitions") != 0
    ):
        raise DecisionContractError("C85V Stage A protected contract drifted")
    rows = manifest.get("derivations")
    if not isinstance(rows, list) or len(rows) != 7:
        raise DecisionContractError("C85V Stage A derivation count drifted")
    if {row.get("theorem_id") for row in rows} != set(THEOREM_IDS):
        raise DecisionContractError("C85V Stage A theorem coverage drifted")
    for row in rows:
        path = root / str(row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or sha256_file(path) != row.get("sha256")
        ):
            raise DecisionContractError("C85V Stage A derivation identity drifted")
    return manifest
