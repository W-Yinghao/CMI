"""Theorem-status and lower-bound proof obligations locked by C85P."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

from .c85_decision_experiments import DecisionContractError


class TheoremStatus(str, Enum):
    PROVED = "PROVED"
    PROVED_FINITE_MODEL_ONLY = "PROVED_FINITE_MODEL_ONLY"
    COUNTEREXAMPLE = "COUNTEREXAMPLE"
    CONJECTURE = "CONJECTURE"
    OPEN = "OPEN"
    INVALIDATED = "INVALIDATED"


@dataclass(frozen=True)
class ProofObligation:
    theorem_id: str
    status: TheoremStatus
    assumptions: tuple[str, ...]
    exact_claim: str
    required_checks: tuple[str, ...]

    def assert_not_overclaimed(self) -> None:
        if self.status in {TheoremStatus.PROVED, TheoremStatus.PROVED_FINITE_MODEL_ONLY}:
            raise DecisionContractError(
                f"{self.theorem_id} cannot be marked proved during C85P"
            )


def validate_c85p_statuses(obligations: Sequence[ProofObligation]) -> None:
    if not obligations or len({item.theorem_id for item in obligations}) != len(obligations):
        raise DecisionContractError("proof obligations must be nonempty and uniquely keyed")
    for obligation in obligations:
        obligation.assert_not_overclaimed()
        if not obligation.assumptions or not obligation.required_checks:
            raise DecisionContractError("every proof obligation needs assumptions and checks")


def fano_candidate_error_expression(
    mutual_information: float, packing_size: int,
) -> float:
    """A registered candidate expression, not a C85P theorem statement."""

    information = float(mutual_information)
    if not math.isfinite(information) or information < 0.0:
        raise DecisionContractError("mutual information must be finite and nonnegative")
    if packing_size < 2:
        raise DecisionContractError("Fano packing requires at least two states")
    return max(0.0, 1.0 - (information + math.log(2.0)) / math.log(packing_size))


def require_symbolic_robust_parameters(alpha: str, epsilon: str, tau: str) -> None:
    expected = {"alpha": "SYMBOLIC_(0,1)", "epsilon": "SYMBOLIC_POSITIVE", "tau": "SYMBOLIC_POSITIVE"}
    observed = {"alpha": alpha, "epsilon": epsilon, "tau": tau}
    if observed != expected:
        raise DecisionContractError("robust-risk theory parameters must remain symbolic in C85P")
