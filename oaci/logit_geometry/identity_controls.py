"""C27 — identity controls carried forward. C26 showed the occupancy pattern IS the target fingerprint; C27
reports target-id predictability from each FROZEN factor family so offset recovery and identity fingerprinting
are read jointly (they are entangled)."""
from __future__ import annotations

import numpy as np

from ..score_gauge.identity_leakage_audit import _nearest_centroid_cv
from . import factor_registry, schema


def id_accuracy(logit_cands, *families) -> float:
    names = factor_registry.family_feature_names(*families)
    X = np.array([[(c.get("feats") if c.get("feats") is not None else factor_registry.candidate_features(c["L"]))[n]
                   for n in names] for c in logit_cands], dtype=np.float64)
    y = np.array([c["target"] for c in logit_cands])
    return _nearest_centroid_cv(X, y)
