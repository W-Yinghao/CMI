"""OACI extractable-leakage estimation: the probe-class-extractable overlap leakage
``L_Q^ov`` (a LOWER bound on ``I_ov``; THEORY §4) with strict grouped cross-fit and a
recording-clustered bootstrap UCB.

Pipeline:
    feat = FrozenFeatures(Z, y, d, group)              # frozen representation + labels
    sg   = build_support_graph(counts, m, p_ref)       # FIXED support / S_y / p_ref
    plan = make_fold_plan(feat, sg, n_folds, seed)     # grouped, domain-stratified, feasible
    est  = estimate_extractable_leakage(feat, sg, plan, cfg)   # point L_abs / L_cond
    ucb  = bootstrap_ucb(feat, sg, plan, cfg, alpha, B, seed)  # L̂ + bootstrap_ucl
"""
from __future__ import annotations

from .critic import CriticConfig, DomainProbe
from .crossfit import FoldPlan, FrozenFeatures, make_fold_plan, oof_nll_by_class
from .estimate import estimate_extractable_leakage, reference_conditional_entropy
from .ucb import bootstrap_ucb, within_domain_group_bootstrap

__all__ = [
    "CriticConfig",
    "DomainProbe",
    "FrozenFeatures",
    "FoldPlan",
    "make_fold_plan",
    "oof_nll_by_class",
    "estimate_extractable_leakage",
    "reference_conditional_entropy",
    "bootstrap_ucb",
    "within_domain_group_bootstrap",
]
