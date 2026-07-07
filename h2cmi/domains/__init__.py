"""Domain-factor DAG: the structured replacement for a flat cartesian domain ID.

The review's central modelling point (section 5.1/5.4/5.5): EEG nuisance is NOT one
categorical ``site x device x subject x session x medication`` label.  It is a DAG of
factors, each with its own causal meaning, its own *handling policy* (hard
canonicalisation vs random effect vs conditioning vs label mechanism) and its own
leakage budget.  The exact chain-rule decomposition

    I(Z; D_{1:K} | Y) = sum_j I(Z; D_j | Y, Pa(D_j))

is computed against this DAG (see ``h2cmi.cmi.hierarchical``).
"""
from __future__ import annotations

from h2cmi.domains.dag import (
    DomainFactor, DomainDAG, DomainLabels, HANDLING, compact_domain_labels,
)

__all__ = ["DomainFactor", "DomainDAG", "DomainLabels", "HANDLING", "compact_domain_labels"]
