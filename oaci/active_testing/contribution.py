"""The queried-information object: one trial's true label plus its per-candidate
LINEAR loss/contribution vector.

A single queried binary label releases only *this trial's* contribution row.  The
row holds the registered linear moments (per-candidate NLL, correctness, class
numerator, signed calibration contribution).  It does NOT hold balanced accuracy,
ECE, midranks, composite utility, the selected action, or target regret — those
are nonlinear plugins assembled downstream, with no unbiasedness claim.

Pairwise NLL differences are DERIVED on demand from the 81-candidate NLL vector;
the C(81,2) = 3,240 differences per trial are never persisted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import CONFIDENCE_BINS, NONLINEAR_PLUGINS, PROB_FLOOR


class C86LPClaimError(RuntimeError):
    """Raised on an unbiasedness claim over a nonlinear plugin quantity."""


@dataclass(frozen=True)
class ContributionRow:
    """Per-trial linear contribution vector for all K candidates (one queried trial)."""

    trial_id: str
    true_label: int
    nll: np.ndarray                         # (K,) -log p(candidate assigns true class)
    correct: np.ndarray                     # (K,) int {0,1}, first-index argmax tie rule
    hard_pred: np.ndarray                   # (K,) argmax class
    confidence: np.ndarray                  # (K,) max class prob
    conf_bin: np.ndarray                    # (K,) int in [0, CONFIDENCE_BINS)
    signed_calibration: np.ndarray          # (K,) confidence - correct

    def __post_init__(self) -> None:
        k = self.nll.shape[0]
        for name in ("correct", "hard_pred", "confidence", "conf_bin", "signed_calibration"):
            arr = getattr(self, name)
            if arr.shape[0] != k:
                raise ValueError(f"contribution field {name!r} length {arr.shape[0]} != K {k}")
        for arr in (self.nll, self.correct, self.hard_pred, self.confidence,
                    self.conf_bin, self.signed_calibration):
            arr.setflags(write=False)


def compute_contribution(trial_id: str, true_label: int, probs: np.ndarray) -> ContributionRow:
    """Build the linear contribution row from an [K, 2] candidate probability matrix.

    ``probs[k]`` is candidate k's (p(class0), p(class1)); rows should sum to ~1.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != 2:
        raise ValueError(f"probs must be [K, 2]; got {probs.shape}")
    if true_label not in (0, 1):
        raise ValueError(f"true_label must be binary; got {true_label!r}")

    p_true = np.clip(probs[:, true_label], PROB_FLOOR, 1.0)
    nll = -np.log(p_true)
    hard_pred = np.argmax(probs, axis=1)            # numpy returns FIRST max index on ties
    confidence = np.max(probs, axis=1)
    correct = (hard_pred == true_label).astype(np.int64)
    conf_bin = np.minimum((confidence * CONFIDENCE_BINS).astype(np.int64), CONFIDENCE_BINS - 1)
    signed_calibration = confidence - correct.astype(np.float64)
    return ContributionRow(
        trial_id=trial_id, true_label=int(true_label), nll=nll, correct=correct,
        hard_pred=hard_pred, confidence=confidence, conf_bin=conf_bin,
        signed_calibration=signed_calibration,
    )


def pairwise_nll_differences(nll: np.ndarray) -> np.ndarray:
    """Derive the C(K,2) vector of |nll_k - nll_k'| on demand (never persisted)."""
    nll = np.asarray(nll, dtype=np.float64)
    iu = np.triu_indices(nll.shape[0], k=1)
    return np.abs(nll[iu[0]] - nll[iu[1]])


def unbiasedness_claim(quantity: str) -> bool:
    """True iff LURE gives a registered-linear-moment unbiasedness claim for ``quantity``."""
    if quantity in NONLINEAR_PLUGINS:
        return False
    from .constants import LINEAR_MOMENTS
    if quantity in LINEAR_MOMENTS:
        return True
    raise ValueError(f"unknown estimand quantity {quantity!r}")


def assert_linear_claim(quantity: str) -> None:
    """Guard: forbid asserting unbiasedness on a nonlinear plugin."""
    if quantity in NONLINEAR_PLUGINS:
        raise C86LPClaimError(
            f"{quantity!r} is a nonlinear plugin; no LURE unbiasedness claim is registered"
        )
