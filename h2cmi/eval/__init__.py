"""Evaluation: metric panel, cross-fitted leakage, and the three-setting harness.

The review insists DG, offline transductive TTA and online streaming TTA must NOT share
one "DG accuracy" header, and that every round report a full panel (not just mean bAcc)
with the statistical unit being the domain/site, not the trial.
"""
from __future__ import annotations

from h2cmi.eval.metrics import (
    classification_metrics,
    worst_domain_balanced_acc,
    domain_cvar,
    cluster_bootstrap_ci,
    metric_panel,
)
from h2cmi.eval.leakage import crossfit_conditional_leakage
from h2cmi.eval.harness import (
    evaluate_strict_dg,
    evaluate_offline_tta,
    evaluate_online_tta,
    run_three_settings,
)

__all__ = [
    "classification_metrics", "worst_domain_balanced_acc", "domain_cvar",
    "cluster_bootstrap_ci", "metric_panel", "crossfit_conditional_leakage",
    "evaluate_strict_dg", "evaluate_offline_tta", "evaluate_online_tta",
    "run_three_settings",
]
