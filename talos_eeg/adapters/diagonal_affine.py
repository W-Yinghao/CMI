"""TALOS-D and TALOS-LD diagonal feature-affine adapters."""

from __future__ import annotations

import numpy as np

from talos_eeg.adapters.logit_bias import fit_logit_bias_temperature
from talos_eeg.adapters.trust_region import (
    AdapterState,
    TrustRegionBounds,
    adapted_features,
    clip_abs,
    clip_l2,
    identity_adapter,
)
from talos_eeg.data.source_state import SourceState


def fit_diagonal_affine(
    state: SourceState,
    z_target: np.ndarray,
    *,
    bounds: TrustRegionBounds,
    variant: str = "TALOS_D",
) -> AdapterState:
    """Fit a deterministic target-unlabeled diagonal affine correction."""

    z = np.asarray(z_target, dtype=np.float64)
    base = identity_adapter(variant, state.n_features, state.n_classes)
    target_mean = z.mean(axis=0)
    target_std = np.maximum(z.std(axis=0), 1e-6)

    raw_diag_delta = state.feature_std / target_std - 1.0
    diag_delta, diag_hit = clip_abs(raw_diag_delta, bounds.tau_a)
    diag = 1.0 + diag_delta
    raw_shift = state.feature_mean - target_mean * diag
    shift, shift_hit = clip_l2(raw_shift, bounds.tau_c)

    hits = []
    if diag_hit:
        hits.append("diag")
    if shift_hit:
        hits.append("shift")
    return AdapterState(
        variant=variant,
        diag=diag,
        shift=shift,
        beta=base.beta,
        log_t=0.0,
        boundary_hits=tuple(hits),
    )


def fit_diagonal_logit_affine(
    state: SourceState,
    z_target: np.ndarray,
    *,
    bounds: TrustRegionBounds,
) -> AdapterState:
    """Fit TALOS-LD by composing TALOS-D feature correction with TALOS-L logits."""

    d_state = fit_diagonal_affine(state, z_target, bounds=bounds, variant="TALOS_LD")
    z_prime = adapted_features(z_target, d_state)
    l_state = fit_logit_bias_temperature(state, z_prime, bounds=bounds, variant="TALOS_LD")
    hits = tuple(sorted(set(d_state.boundary_hits + l_state.boundary_hits)))
    return AdapterState(
        variant="TALOS_LD",
        diag=d_state.diag,
        shift=d_state.shift,
        beta=l_state.beta,
        log_t=l_state.log_t,
        boundary_hits=hits,
    )
