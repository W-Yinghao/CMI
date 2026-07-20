"""Prospective statistical-decision contracts for C85.

This package is intentionally independent of the empirical, training, selector,
and multidataset stacks.  C85P exposes finite-model definitions and protocol
validators only; C85T is responsible for proofs and synthetic results.
"""

from .c85_decision_experiments import (
    DecisionContractError,
    FiniteDecisionProblem,
    FiniteExperiment,
)

__all__ = [
    "DecisionContractError",
    "FiniteDecisionProblem",
    "FiniteExperiment",
]
