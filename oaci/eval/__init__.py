"""OACI evaluation — fixed-estimand accuracy/calibration metrics, paired clustered bootstrap,
noninferiority decisions, and the missing-cell sweep scalar. numpy-only (operates on saved
prediction artifacts; no torch, no retraining)."""
from __future__ import annotations

from .artifacts import PredictionBundle, align_pair, population_hash
from .metrics import (
    domain_bacc,
    domain_baccs,
    mean_domain_bacc,
    pooled_bacc,
    worst_domain_bacc,
    worst_paired_delta_bacc,
)
from .calibration import (
    fit_temperature,
    fixed_bin_edges,
    mean_domain_nll,
    pooled_nll,
    top_label_ece,
    worst_domain_nll,
)
from .bootstrap import (
    BootstrapPlan,
    is_whole_group_resample,
    make_bootstrap_plan,
    paired_ci,
    point_delta_over_seeds,
)
from .noninferiority import noninferiority, source_risk_noninferiority, superiority
from .sweep import (
    assert_fixed_audit_population,
    post_fragmentation_curve_average,
    simultaneous_band,
)

__all__ = [
    "PredictionBundle", "align_pair", "population_hash",
    "domain_bacc", "domain_baccs", "mean_domain_bacc", "pooled_bacc",
    "worst_domain_bacc", "worst_paired_delta_bacc",
    "fit_temperature", "fixed_bin_edges", "mean_domain_nll", "pooled_nll",
    "top_label_ece", "worst_domain_nll",
    "BootstrapPlan", "is_whole_group_resample", "make_bootstrap_plan", "paired_ci",
    "point_delta_over_seeds",
    "noninferiority", "source_risk_noninferiority", "superiority",
    "assert_fixed_audit_population", "post_fragmentation_curve_average", "simultaneous_band",
]
