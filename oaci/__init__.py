"""OACI — Overlap-Aware Risk-Feasible Conditional Invariance for strict-DG EEG.

Working title: *Partial Conditional Invariance under Domain–Class Support Mismatch*.

The premise the global objective silently assumes — that every ``p(z|y,d)`` we want to
compare has enough support to estimate — fails on clinical EEG: some sites lack a class,
some classes are near-empty in some domains, subject≈label, the sampler distorts
``p(D|Y)``. OACI only **measures / enforces** conditional invariance on the ``(d,y)``
cells the data can actually estimate (the *support graph*); unsupported cells are marked
**non-identifiable**, never smoothed into existence. It then minimises an *upper
confidence bound* on the overlap-aware conditional leakage ``I_ov(Z;D|Y)`` subject to
source risk staying within ``ε`` of ERM (risk-feasible noninferiority).

Isolated from ``cmi/`` (the CLOSED LPC line) and ``h2cmi/`` — nothing here imports either.

Docs: ``README.md`` (overview), ``THEORY.md`` (the three results: support / no-excess-risk
/ label-preservation), ``EXPERIMENTS.md`` (protocol + missing-cell stress test + kill
criteria).
"""
from __future__ import annotations

from .support_graph import (
    SupportGraph,
    build_support_graph,
    counts_from_labels,
    empirical_class_prior,
)

__all__ = [
    "SupportGraph",
    "build_support_graph",
    "counts_from_labels",
    "empirical_class_prior",
]
