"""Shared label-free scoring path. SINGLE source of φ_a so deploy.py and run_gonogo.py cannot diverge.

`score_actions(state, z, actions)` takes NO y argument: every output is a deterministic function of (state, z).
This is the object the metamorphic guard proves invariant to label permutation.
"""
from __future__ import annotations
import numpy as np

from .actions import apply_action
from .features import paired_features, context_features, feature_vector


def score_actions(state, z, actions) -> dict:
    """{action: dict(p, ztil, phi, fvec)} for identity + each requested non-identity action. Label-free."""
    z = np.asarray(z, float)
    p0, z0 = apply_action("identity", state, z)
    out = {"identity": dict(p=p0, ztil=z0, phi=None, fvec=None)}
    for a in actions:
        if a == "identity":
            continue
        pa, za = apply_action(a, state, z)
        phi = paired_features(p0, pa, z0, za)
        ctx = context_features(state, za, pa)
        out[a] = dict(p=pa, ztil=za, phi={**phi, **ctx}, fvec=feature_vector(phi, ctx))
    return out
