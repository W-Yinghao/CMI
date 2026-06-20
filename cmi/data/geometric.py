"""LogCov / tangent-space features (pyRiemann) — a deterministic, NON-neural representation.

Covariances (OAS shrinkage) -> Riemannian tangent space (reference mean fit on the source
training set) -> a fixed C(C+1)/2 vector per trial. Used as the 'LogCov' carrier: a frozen
geometric feature with a small trainable MLP head, to show the LPC-CMI regularizer is not
specific to neural encoders (it works on a classical covariance representation too).
"""
from __future__ import annotations
import numpy as np


def tangent_features(Xtrain, *others, estimator="oas"):
    """Fit Covariances+TangentSpace on Xtrain ([n,C,T]); return tangent vectors ([n, C(C+1)/2])
    for Xtrain and each array in `others`, transformed with the SAME fitted space (no target leakage)."""
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    cov = Covariances(estimator=estimator)
    ts = TangentSpace(metric="riemann")
    Ctr = cov.transform(Xtrain.astype("float64"))
    ts.fit(Ctr)
    out = [ts.transform(Ctr).astype("float32")]
    for X in others:
        out.append(ts.transform(cov.transform(X.astype("float64"))).astype("float32"))
    return out
