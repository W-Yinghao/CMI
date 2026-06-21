"""Per-action regressor ĝ_a: φ_a(B) ↦ ΔR̂_a(B). HistGBM when enough batches, Ridge fallback on tiny LOCO folds.

A DeepSets set-encoder over per-sample features is the planned upgrade (pre-registration §6); for the GPU-free
go/no-go we regress on the batch-summary vector, which is sufficient to test whether the signal exists at all.
We deliberately do NOT impose monotonicity: A0 proved the sign of these observables vs harm is unreliable, so the
direction must be learned, not asserted.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    _HAS_HGB = True
except Exception:                                            # pragma: no cover
    _HAS_HGB = False


class ActionRegressor:
    def __init__(self, seed=0, min_for_gbm=40):
        self.seed = seed
        self.min_for_gbm = min_for_gbm
        self.scaler = None
        self.model = None
        self.const = 0.0

    def fit(self, X, dr):
        X = np.asarray(X, float); dr = np.asarray(dr, float)
        if len(X) < 8 or np.std(dr) < 1e-12:
            self.const = float(dr.mean()) if len(dr) else 0.0
            self.model = None
            return self
        if _HAS_HGB and len(X) >= self.min_for_gbm:
            self.scaler = None
            self.model = HistGradientBoostingRegressor(
                max_iter=200, max_depth=3, learning_rate=0.05, min_samples_leaf=10,
                l2_regularization=1.0, random_state=self.seed).fit(X, dr)
        else:
            self.scaler = StandardScaler().fit(X)
            self.model = Ridge(alpha=1.0).fit(self.scaler.transform(X), dr)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        if self.model is None:
            return np.full(len(X), self.const, float)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return np.asarray(self.model.predict(X), float)
