"""TALOS-L logit-bias and temperature adapter."""

from __future__ import annotations

import numpy as np

from talos_eeg.adapters.trust_region import (
    AdapterState,
    TrustRegionBounds,
    clip_l2,
    clip_scalar,
    identity_adapter,
    predict_proba,
)
from talos_eeg.data.source_state import SourceState


def fit_logit_bias_temperature(
    state: SourceState,
    z_target: np.ndarray,
    *,
    bounds: TrustRegionBounds,
    variant: str = "TALOS_L",
) -> AdapterState:
    """Fit a deterministic label-free logit adapter from target features only."""

    base = identity_adapter(variant, state.n_features, state.n_classes)
    proba = predict_proba(state, z_target, base)
    predicted_prior = np.maximum(proba.mean(axis=0), 1e-8)
    target_entropy = -np.sum(proba * np.log(np.maximum(proba, 1e-8)), axis=1)
    mean_conf = float(np.max(proba, axis=1).mean())

    raw_beta = 0.5 * (np.log(np.maximum(state.source_prior, 1e-8)) - np.log(predicted_prior))
    raw_beta = raw_beta - raw_beta.mean()
    beta, beta_hit = clip_l2(raw_beta, bounds.tau_beta)

    max_entropy = np.log(float(state.n_classes))
    entropy_ratio = float(target_entropy.mean() / max(max_entropy, 1e-8))
    raw_log_t = 0.20 * (mean_conf - 0.75) - 0.10 * (entropy_ratio - 0.50)
    log_t, log_t_hit = clip_scalar(raw_log_t, bounds.tau_log_t)

    hits = []
    if beta_hit:
        hits.append("beta")
    if log_t_hit:
        hits.append("log_t")
    return AdapterState(
        variant=variant,
        diag=base.diag,
        shift=base.shift,
        beta=beta,
        log_t=log_t,
        boundary_hits=tuple(hits),
    )
