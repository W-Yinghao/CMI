"""Eval-unit metrics for one PredictionBundle (A2b-1b-ii-b).

Strict-only: reference bAcc is NaN whenever a domain misses a pre-registered class (never silently
averaged over present classes); the observed companion and class coverage are always reported. NLL
and ECE are computed on the eval-unit bundle with ONE fixed bin-edge array. No temperature fitting,
no non-inferiority / efficacy inference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..eval.calibration import (domain_eces, mean_domain_ece, mean_domain_nll, pooled_nll,
                                top_label_ece, worst_domain_ece, worst_domain_nll)
from ..eval.metrics import domain_bacc, mean_domain_bacc, pooled_bacc_summary, worst_domain_bacc
from .scientific_hash import scientific_value_hash


@dataclass(frozen=True)
class EvaluationMetrics:
    pooled_reference_bacc: float
    pooled_reference_status: str
    pooled_observed_bacc: float
    pooled_class_coverage: float
    mean_domain_reference_bacc: float
    worst_domain_reference_bacc: float
    domain_reference_status: str
    mean_domain_observed_bacc: float
    worst_domain_observed_bacc: float
    domain_class_coverage_items: tuple
    pooled_nll: float
    mean_domain_nll: float
    worst_domain_nll: float
    pooled_ece: float
    mean_domain_ece: float
    worst_domain_ece: float
    metrics_hash: str


def evaluate_prediction_bundle(bundle, *, bin_edges) -> EvaluationMetrics:
    y, pred, dom, logits = bundle.y, bundle.pred, bundle.domain, bundle.logits
    classes = list(range(bundle.n_classes))
    ps = pooled_bacc_summary(y, pred, classes)
    mref = mean_domain_bacc(y, pred, dom, classes, "reference")
    wref = worst_domain_bacc(y, pred, dom, classes, "reference")
    mobs = mean_domain_bacc(y, pred, dom, classes, "observed")
    wobs = worst_domain_bacc(y, pred, dom, classes, "observed")
    dom_status = "estimable" if not np.isnan(mref) else "nonestimable_missing_class_domain"
    cov = []
    for d in sorted(set(int(x) for x in dom.tolist())):
        m = dom == d
        cov.append((d, domain_bacc(y[m], pred[m], classes)[2]))
    edges = np.asarray(bin_edges, float)
    m = dict(pooled_reference_bacc=ps["reference"], pooled_reference_status=ps["reference_status"],
             pooled_observed_bacc=ps["observed"], pooled_class_coverage=ps["class_coverage"],
             mean_domain_reference_bacc=mref, worst_domain_reference_bacc=wref, domain_reference_status=dom_status,
             mean_domain_observed_bacc=mobs, worst_domain_observed_bacc=wobs,
             domain_class_coverage_items=tuple(cov),
             pooled_nll=pooled_nll(logits, y), mean_domain_nll=mean_domain_nll(logits, y, dom),
             worst_domain_nll=worst_domain_nll(logits, y, dom),
             pooled_ece=top_label_ece(logits, y, bin_edges=edges),
             mean_domain_ece=mean_domain_ece(logits, y, dom, bin_edges=edges),
             worst_domain_ece=worst_domain_ece(logits, y, dom, bin_edges=edges))
    return EvaluationMetrics(**m, metrics_hash=scientific_value_hash(m))
