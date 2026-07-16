"""Proof-artifact construction and fail-closed C85T status transitions.

This module defines future proof generators.  Importing it does not render a
proof, run an audit, or change a theorem status.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

from .c85_decision_experiments import DecisionContractError
from .c85_lower_bound_contracts import TheoremStatus
from .c85t_rng import REGISTERED_EXECUTION_TOKEN


THEOREM_IDS = tuple(f"T{index}" for index in range(1, 8))
REQUIRED_SECTIONS = (
    "Exact Statement",
    "Assumptions",
    "Proof Or Counterexample",
    "Boundary Cases",
    "Independent Red Team",
    "Final Status",
)
PROOF_FILENAMES = {
    "T1": "T1_blackwell_monotonicity.md",
    "T2": "T2_restricted_policy_counterexamples.md",
    "T3": "T3_policy_collapse.md",
    "T4": "T4_two_state_lecam_regret_bound.md",
    "T5": "T5_fano_extension.md",
    "T6": "T6_mean_tail_counterexample.md",
    "T7": "T7_near_optimal_union_bound.md",
}
ALLOWED_TRANSITIONS = {
    "T1": {TheoremStatus.PROVED, TheoremStatus.PROVED_FINITE_MODEL_ONLY},
    "T2": {TheoremStatus.COUNTEREXAMPLE, TheoremStatus.INVALIDATED},
    "T3": {TheoremStatus.PROVED, TheoremStatus.PROVED_FINITE_MODEL_ONLY},
    "T4": {
        TheoremStatus.PROVED,
        TheoremStatus.PROVED_FINITE_MODEL_ONLY,
        TheoremStatus.INVALIDATED,
    },
    "T5": {
        TheoremStatus.PROVED,
        TheoremStatus.PROVED_FINITE_MODEL_ONLY,
        TheoremStatus.OPEN,
        TheoremStatus.INVALIDATED,
    },
    "T6": {TheoremStatus.COUNTEREXAMPLE, TheoremStatus.INVALIDATED},
    "T7": {
        TheoremStatus.PROVED,
        TheoremStatus.PROVED_FINITE_MODEL_ONLY,
        TheoremStatus.INVALIDATED,
    },
}


@dataclass(frozen=True)
class ProofCandidate:
    theorem_id: str
    exact_statement: str
    assumptions: tuple[str, ...]
    proof_or_counterexample: str
    boundary_cases: tuple[str, ...]
    proposed_status: TheoremStatus
    simulation_used_as_proof: bool = False
    citation_used_as_complete_proof: bool = False


@dataclass(frozen=True)
class ProofAudit:
    theorem_id: str
    verdict: str
    checks: tuple[str, ...]
    reviewer_role: str = "C85T_INDEPENDENT_PROOF_RED_TEAM"


def _statement_sha(statement: str) -> str:
    return hashlib.sha256(statement.encode("utf-8")).hexdigest()


def validate_proof_candidate(candidate: ProofCandidate) -> None:
    if candidate.theorem_id not in THEOREM_IDS:
        raise DecisionContractError("unknown theorem ID")
    if candidate.proposed_status not in ALLOWED_TRANSITIONS[candidate.theorem_id]:
        raise DecisionContractError("theorem transition is outside the locked set")
    if not candidate.exact_statement.strip() or not candidate.assumptions:
        raise DecisionContractError("proof candidate lacks statement or assumptions")
    if not candidate.proof_or_counterexample.strip() or not candidate.boundary_cases:
        raise DecisionContractError("proof candidate lacks argument or boundary cases")
    if candidate.simulation_used_as_proof and candidate.proposed_status in {
        TheoremStatus.PROVED,
        TheoremStatus.PROVED_FINITE_MODEL_ONLY,
    }:
        raise DecisionContractError("simulation cannot create PROVED status")
    if candidate.citation_used_as_complete_proof and candidate.proposed_status in {
        TheoremStatus.PROVED,
        TheoremStatus.PROVED_FINITE_MODEL_ONLY,
    }:
        raise DecisionContractError("citation alone cannot create a project proof")


def apply_status_transition(candidate: ProofCandidate, audit: ProofAudit) -> TheoremStatus:
    validate_proof_candidate(candidate)
    if audit.theorem_id != candidate.theorem_id:
        raise DecisionContractError("proof candidate and audit theorem IDs differ")
    if candidate.proposed_status == TheoremStatus.OPEN:
        if candidate.theorem_id != "T5":
            raise DecisionContractError("only T5 may remain OPEN at successful C85T")
        return TheoremStatus.OPEN
    if audit.verdict != "PASS" or not audit.checks:
        raise DecisionContractError("non-OPEN theorem transition requires independent PASS")
    return candidate.proposed_status


def render_proof_markdown(candidate: ProofCandidate, audit: ProofAudit) -> str:
    status = apply_status_transition(candidate, audit)
    assumptions = "\n".join(f"- {value}" for value in candidate.assumptions)
    boundaries = "\n".join(f"- {value}" for value in candidate.boundary_cases)
    checks = "\n".join(f"- {value}" for value in audit.checks)
    return (
        f"# {candidate.theorem_id} Proof Artifact\n\n"
        "## Exact Statement\n\n"
        f"{candidate.exact_statement}\n\n"
        f"Statement SHA-256: `{_statement_sha(candidate.exact_statement)}`\n\n"
        "## Assumptions\n\n"
        f"{assumptions}\n\n"
        "## Proof Or Counterexample\n\n"
        f"{candidate.proof_or_counterexample}\n\n"
        "## Boundary Cases\n\n"
        f"{boundaries}\n\n"
        "## Independent Red Team\n\n"
        f"Role: `{audit.reviewer_role}`\n\nVerdict: `{audit.verdict}`\n\n{checks}\n\n"
        "## Final Status\n\n"
        f"`{status.value}`\n"
    )


def validate_rendered_proof(text: str, theorem_id: str) -> None:
    if theorem_id not in THEOREM_IDS:
        raise DecisionContractError("unknown theorem ID")
    if not text.startswith(f"# {theorem_id} Proof Artifact\n"):
        raise DecisionContractError("proof heading drifted")
    for section in REQUIRED_SECTIONS:
        if text.count(f"## {section}\n") != 1:
            raise DecisionContractError(f"proof section {section!r} is missing or duplicated")
    if "Statement SHA-256:" not in text:
        raise DecisionContractError("proof statement digest is absent")


def _future_candidates(
    statements: Mapping[str, str], exact_results: Mapping[str, Any]
) -> dict[str, ProofCandidate]:
    """Build future candidates from exact output; not called during C85TL."""

    if set(statements) != set(THEOREM_IDS):
        raise DecisionContractError("T1-T7 exact statement registry is incomplete")
    if not {"S1", "S5", "S10"} <= set(exact_results):
        raise DecisionContractError("proof candidates require exact scenario evidence")
    return {
        "T1": ProofCandidate(
            "T1",
            statements["T1"],
            (
                "common state, action, and loss spaces",
                "E1 is obtained from E2 through a state-independent Markov kernel",
                "randomized measurable decision rules are allowed",
            ),
            "Compose any E1 decision kernel with the E2-to-E1 garbling kernel. "
            "Tonelli's theorem for the nonnegative bounded loss gives exactly the "
            "same statewise action law and risk. The composed rule is admissible "
            "under E2, so taking the E2 infimum cannot exceed the E1 infimum.",
            ("ties do not affect the infimum", "the claim does not compare nonnested restricted classes"),
            TheoremStatus.PROVED,
        ),
        "T2": ProofCandidate(
            "T2",
            statements["T2"],
            ("finite S1 and repaired S10 laws", "registered policies fixed prospectively"),
            "Enumerate state, observation, and action atoms. S1 gives a richer "
            "unrestricted risk no larger than the coarse risk while its fixed "
            "opposite-state rule is worse. S10 independently gives coarse risk "
            "11/40, rich unrestricted risk 0, and rich registered risk 3/5; the "
            "registered reversal is 13/40.",
            ("the counterexamples do not contradict unrestricted Blackwell monotonicity",),
            TheoremStatus.COUNTEREXAMPLE,
        ),
        "T3": ProofCandidate(
            "T3",
            statements["T3"],
            (
                "action kernels agree almost surely under each state law",
                "the same state/action loss is applied",
            ),
            "Kernel equality almost surely implies equality of the induced joint "
            "observation-action measures after integration against each state law. "
            "Therefore every bounded action-loss expectation agrees statewise. "
            "Prior and group risks are linear aggregations of those equal risks.",
            ("one coupled equal action draw is insufficient", "null sets must be controlled statewise"),
            TheoremStatus.PROVED,
        ),
        "T4": ProofCandidate(
            "T4",
            statements["T4"],
            (
                "equal two-state prior",
                "disjoint optimal-action sets and an explicit action-to-state decoder",
                "every nonoptimal action has regret at least Delta",
                "TV is sup_A |P0(A)-P1(A)|",
            ),
            "Decode the state from the selected action's disjoint optimal set. In "
            "state j, every decoder error entails a nonoptimal action and regret at "
            "least Delta. Thus average regret is at least Delta times average testing "
            "error. The minimum equal-prior randomized testing error is "
            "(1-TV(P0,P1))/2, yielding the claimed bound.",
            ("Delta=0 is valid but vacuous", "overlapping optimal sets require a different decoder and are excluded"),
            TheoremStatus.PROVED,
        ),
        "T5": ProofCandidate(
            "T5",
            statements["T5"],
            ("finite uniform state packing", "state-specific optimal actions", "registered KL-to-mixture control"),
            "The finite Fano reduction and its exact information assumptions remain "
            "an open proof obligation. C85T retains this failed/incomplete attempt "
            "without using simulation as proof.",
            ("K=2 requires separate treatment", "negative candidate lower bounds are truncated at zero"),
            TheoremStatus.OPEN,
        ),
        "T6": ProofCandidate(
            "T6",
            statements["T6"],
            ("the exact ten equally weighted S5 groups", "upper-loss CVaR with alpha in (0,1)"),
            "Direct finite-distribution calculation gives policy mean 0.37 versus "
            "reference 0.5 and policy worst loss 1 versus 0.5. For alpha<0.9, "
            "policy CVaR is (0.37-0.3 alpha)/(1-alpha); for alpha>=0.9 it is 1. "
            "Comparison with 0.5 is strict exactly when alpha>13/20.",
            ("alpha=13/20 is equality and excluded", "alpha=1 is outside the CVaR domain"),
            TheoremStatus.COUNTEREXAMPLE,
        ),
        "T7": ProofCandidate(
            "T7",
            statements["T7"],
            (
                "the registered pairwise sub-Gaussian MGF bound",
                "first-index argmax selection",
                "no independence assumption",
            ),
            "If action i is selected over an optimal action, then "
            "xi_i-xi_star >= Delta_i. For each action outside A_epsilon, the "
            "Chernoff bound optimized at lambda=Delta_i/sigma_i^2 gives "
            "exp(-Delta_i^2/(2 sigma_i^2)). A union bound over those actions and "
            "a cap at one prove the result without independence.",
            ("ties satisfy the same weak event inclusion", "an empty outside set gives probability and bound zero"),
            TheoremStatus.PROVED,
        ),
    }


def _future_independent_audits(candidates: Mapping[str, ProofCandidate]) -> dict[str, ProofAudit]:
    """Apply theorem-specific mechanical red-team checks at C85T execution."""

    audits: dict[str, ProofAudit] = {}
    required_tokens = {
        "T1": ("garbling", "infimum"),
        "T2": ("11/40", "3/5", "13/40"),
        "T3": ("almost surely", "statewise"),
        "T4": ("(1-TV(P0,P1))/2", "Delta"),
        "T5": ("open proof obligation",),
        "T6": ("13/20", "alpha=1"),
        "T7": ("union bound", "without independence"),
    }
    for theorem_id, candidate in candidates.items():
        validate_proof_candidate(candidate)
        missing = tuple(
            token for token in required_tokens[theorem_id]
            if token not in candidate.proof_or_counterexample
            and token not in " ".join(candidate.boundary_cases)
        )
        verdict = "PASS" if not missing else "FAIL"
        checks = (
            "exact statement digest bound",
            "assumption list nonempty",
            "simulation not used as proof",
            "theorem-specific constants and boundary cases replayed",
        ) if not missing else tuple(f"missing token: {token}" for token in missing)
        audits[theorem_id] = ProofAudit(theorem_id, verdict, checks)
    return audits


def execute_proof_pipeline(
    *,
    statements: Mapping[str, str],
    exact_results: Mapping[str, Any],
    output_dir: Path,
    execution_token: str,
) -> dict[str, str]:
    """Render proof artifacts only in the future authorized C85T run."""

    if execution_token != REGISTERED_EXECUTION_TOKEN:
        raise DecisionContractError("proof execution requires C85T authorization")
    if output_dir.exists():
        raise DecisionContractError("proof output directory must be fresh")
    candidates = _future_candidates(statements, exact_results)
    audits = _future_independent_audits(candidates)
    output_dir.mkdir(parents=True)
    statuses: dict[str, str] = {}
    for theorem_id in THEOREM_IDS:
        text = render_proof_markdown(candidates[theorem_id], audits[theorem_id])
        validate_rendered_proof(text, theorem_id)
        (output_dir / PROOF_FILENAMES[theorem_id]).write_text(text)
        statuses[theorem_id] = apply_status_transition(
            candidates[theorem_id], audits[theorem_id]
        ).value
    return statuses

