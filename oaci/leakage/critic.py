"""The conditional-domain probe ``q(D | Z, Y=y)`` for ONE class ``y``.

Properties enforced here (the rest of the no-leakage discipline lives in ``crossfit.py``):

* operates on a **frozen** representation ``Z`` (the encoder is not trained here);
* **per-class, independent** — one probe per class ``y``, never a single ``q(D|Z)``;
* the label space is exactly ``S_y`` (the eligible domains for ``y``); samples in
  unsupported cells (``d ∉ S_y``) are excluded by the caller and never appear in training,
  in the label space, or in scoring;
* **all preprocessing is fit on the training rows only** — feature standardisation here, and
  (in ``crossfit``) the train/test fold split. The random feature map is data-INDEPENDENT
  (seeded by capacity), so it leaks nothing.

The probe family ``Q`` is indexed by ``capacity``: ``0`` = multinomial logistic on the
standardised ``Z`` (linear probe); ``c>0`` = logistic on ``[Z_std, ReLU(Z_std·R_c)]`` with a
fixed random ``R_c`` of width ``c`` (a larger, strictly more expressive probe). Convex + L2,
so no early stopping is needed; the NLL is reported in **nats** to match the reference
entropy. Negative-leakage estimates are never clipped — that happens downstream by *keeping*
them, here we just return honest NLLs.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn.linear_model import LogisticRegression


@dataclass
class CriticConfig:
    """Probe family + fit hyper-parameters (all data-independent)."""
    capacities: tuple[int, ...] = (0, 64)   # the probe family Q (capacity sup is taken downstream)
    l2_C: float = 1.0                       # sklearn inverse-L2 strength (1/λ)
    max_iter: int = 500
    prob_floor: float = 1e-6                # floor predicted prob before log (finite NLL)
    feature_seed_base: int = 10_000         # R_c seed = base + capacity (fixed, not data-driven)
    solver: str = "lbfgs"                   # frozen sklearn defaults (explicit, not implicit)
    tol: float = 1e-4
    fit_intercept: bool = True


class DomainProbe:
    """A single ``q(D|Z, Y=y)`` over the fixed label space ``{0..n_labels-1}`` = ``S_y``."""

    def __init__(self, capacity: int, n_labels: int, cfg: CriticConfig):
        self.capacity = int(capacity)
        self.n_labels = int(n_labels)
        self.cfg = cfg
        self._mean = self._std = self._R = self._model = None
        self._only_label: int | None = None  # set when the train fold has a single domain

    # ---- feature map (standardise on train; random ReLU lift is data-independent) ----
    def _features(self, Z: np.ndarray) -> np.ndarray:
        Zs = (Z - self._mean) / self._std
        if self.capacity > 0:
            return np.concatenate([Zs, np.maximum(Zs @ self._R, 0.0)], axis=1)
        return Zs

    def fit(self, Z: np.ndarray, labels: np.ndarray, sample_weight=None) -> "DomainProbe":
        Z = np.asarray(Z, dtype=np.float64)
        labels = np.asarray(labels, dtype=int)
        w = np.ones(Z.shape[0]) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).ravel()
        ws = w.sum()
        # MASS-weighted standardisation (so duplicating a row with split mass is a no-op)
        self._mean = (w[:, None] * Z).sum(axis=0) / ws
        var = (w[:, None] * (Z - self._mean) ** 2).sum(axis=0) / ws
        self._std = np.sqrt(var) + 1e-8
        if self.capacity > 0:
            rng = np.random.default_rng(self.cfg.feature_seed_base + self.capacity)
            self._R = rng.standard_normal((Z.shape[1], self.capacity)) / np.sqrt(Z.shape[1])
        uniq = np.unique(labels)
        if uniq.size < 2:
            self._only_label = int(uniq[0])
            self._model = None
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = LogisticRegression(C=self.cfg.l2_C, max_iter=self.cfg.max_iter,
                                                 solver=self.cfg.solver, tol=self.cfg.tol,
                                                 fit_intercept=self.cfg.fit_intercept)
                self._model.fit(self._features(Z), labels, sample_weight=w)   # weighted MLE
        return self

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        """Probabilities over the FULL ``S_y`` label space (unseen-in-train domains get the
        floor), each row renormalised."""
        Z = np.asarray(Z, dtype=np.float64)
        P = np.full((Z.shape[0], self.n_labels), self.cfg.prob_floor, dtype=np.float64)
        if self._model is None:
            P[:, self._only_label] = 1.0
        else:
            p = self._model.predict_proba(self._features(Z))
            for j, cls in enumerate(self._model.classes_):
                P[:, int(cls)] = p[:, j]
        P = np.clip(P, self.cfg.prob_floor, None)
        P /= P.sum(axis=1, keepdims=True)
        return P

    def nll(self, Z: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Per-sample negative log-likelihood in **nats** (natural log)."""
        labels = np.asarray(labels, dtype=int)
        P = self.predict_proba(Z)
        return -np.log(P[np.arange(labels.shape[0]), labels])
