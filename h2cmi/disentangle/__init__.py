"""Task / nuisance latent disentanglement penalties (review section 5.6).

Cheap, differentiable surrogates (HSIC / cross-covariance / small classifier
probes) for the composite objective

    L_disentangle = I(Z_c ; Z_n | Y) + eta * I(Z_n ; Y | D) - kappa * I(Z_n ; D)
"""
from __future__ import annotations

from .penalties import (
    DisentangleLoss,
    cross_covariance_penalty,
    hsic,
    orthogonality_penalty,
)

__all__ = [
    "hsic",
    "cross_covariance_penalty",
    "orthogonality_penalty",
    "DisentangleLoss",
]
