"""Hierarchical conditional mutual-information estimation and control (review 5.4, P0-2).

The corrected estimator (P0-2): instead of the posterior-KL "upper bound"
E KL(q(d|z,y) || p(d|y)) -- which is NOT an upper bound for arbitrary q -- we use the
identity

    I(Z; D | Y) = H(D | Y) - H(D | Z, Y)

where H(D|Y) is encoder-independent (empirical) and H(D|Z,Y) is the optimal conditional
cross-entropy of a neural domain critic.  The encoder MAXIMISES that optimal conditional
cross-entropy (conditional GRL / min-max), so the estimator is a profile objective with a
clean envelope-theorem interpretation.  The hierarchical chain rule

    I(Z; D_{1:K} | Y) = sum_j I(Z; D_j | Y, Pa(D_j))

is realised by one conditional critic per factor j conditioned on its DAG parents.
"""
from __future__ import annotations

from h2cmi.cmi.hierarchical import (
    HierarchicalCMI,
    DualBudget,
    reference_conditional_entropy,
    grad_reverse,
)

__all__ = ["HierarchicalCMI", "DualBudget", "reference_conditional_entropy", "grad_reverse"]
