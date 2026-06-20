"""Reference-prior marginal alignment distances for H2-CMI.

Differentiable, numerically-stable distribution-distance functions used to align each
domain's latent marginal towards a shared reference prior.  See
:mod:`h2cmi.align.distances` for the implementations.
"""
from __future__ import annotations

from h2cmi.align.distances import (
    energy_distance,
    gauss_w2,
    ledoit_wolf_cov,
    sliced_wasserstein,
)
from h2cmi.align.reference_marginal import (
    ReferenceMarginalAlignment,
    gls_weights,
    gls_reference_domain_marginal,
    class_given_domain,
)

__all__ = [
    "sliced_wasserstein",
    "energy_distance",
    "gauss_w2",
    "ledoit_wolf_cov",
    "ReferenceMarginalAlignment",
    "gls_weights",
    "gls_reference_domain_marginal",
    "class_given_domain",
]
