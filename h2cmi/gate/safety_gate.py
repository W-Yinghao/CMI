"""Source-only LEARNED safety gate (review section 7).

The gate decides, *per target domain*, whether test-time adaptation (TTA) should
run or whether the model should fall back to the identity transform.  The decision
is made from cheap, source-computable diagnostics only -- no target labels are ever
used.  The gate is *trained* on inner leave-one-source-domain-out splits, where the
true adaptation gain ``bAcc_adapt - bAcc_identity`` is observable for every held-out
pseudo-target.  At deployment we predict, from the same diagnostic vector, whether
adapting would *harm* the held-out domain; if so we abstain (identity).

Everything here depends only on :mod:`numpy`, :mod:`sklearn` and the standard
library, and is fully self-contained / degenerate-safe (it must not crash when only
one harm-class is observed, when groups are empty, or when a feature has zero
variance).

The diagnostic feature vector ``g`` is built from a dict of named scalars whose keys
are listed in :data:`GATE_FEATURE_KEYS` (review section 7):

    delta_density_nll : improvement (>0 better) of the class-conditional density NLL
                        on the target under the adapted transform.
    transform_norm    : magnitude of the learned alignment transform (deviation from
                        identity); large == aggressive adaptation.
    condition_number  : conditioning of the alignment / whitening transform; large ==
                        numerically unstable adaptation.
    prior_shift       : magnitude of estimated label-prior shift source->target.
    pred_disagreement : disagreement between identity and adapted predictions.
    cmi_residual      : leftover hierarchical class-conditional MI (leakage) after
                        adaptation; large == adaptation failed to remove shift.
    ood_score         : out-of-distribution score of the target w.r.t. source support.
    ess               : effective sample size of the (pseudo-)target.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

__all__ = ["GATE_FEATURE_KEYS", "gate_features", "SafetyGate"]

#: Fixed ordering of the diagnostic features (review section 7).
GATE_FEATURE_KEYS: tuple[str, ...] = (
    "delta_density_nll",
    "transform_norm",
    "condition_number",
    "prior_shift",
    "pred_disagreement",
    "cmi_residual",
    "ood_score",
    "ess",
)

_N_FEATS = len(GATE_FEATURE_KEYS)


def _finite(x: float) -> float:
    """Coerce ``x`` to a finite float; non-finite / unparseable -> 0.0."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return v


def gate_features(diag: dict) -> np.ndarray:
    """Build a length-8 float feature vector from a diagnostics dict.

    Values are taken in :data:`GATE_FEATURE_KEYS` order; missing keys default to
    ``0.0`` and any non-finite value is replaced by ``0.0``.
    """
    diag = diag or {}
    return np.array(
        [_finite(diag.get(k, 0.0)) for k in GATE_FEATURE_KEYS],
        dtype=np.float64,
    )


class SafetyGate:
    """A learned, source-only gate that predicts whether TTA will harm a target.

    Parameters
    ----------
    model : {"logistic", "gbt"}
        Backing sklearn classifier.  ``"logistic"`` ->
        :class:`~sklearn.linear_model.LogisticRegression`, anything else ->
        :class:`~sklearn.ensemble.GradientBoostingClassifier`.
    harm_delta : float
        A pseudo-target is labelled *harmful* iff ``gain < -harm_delta``.  Using a
        positive ``harm_delta`` only flags adaptation that hurts by more than a
        margin (tolerates tiny regressions).
    risk_threshold : float
        Adapt only when ``P(harm) < risk_threshold``.
    min_evidence : float
        Additional guard at decision time: adapt only when the supplied evidence
        (density-NLL improvement) is ``>= min_evidence``.
    """

    def __init__(
        self,
        model: str = "logistic",
        harm_delta: float = 0.0,
        risk_threshold: float = 0.5,
        min_evidence: float = 0.0,
    ) -> None:
        self.model = str(model)
        self.harm_delta = float(harm_delta)
        self.risk_threshold = float(risk_threshold)
        self.min_evidence = float(min_evidence)

        # learned state (set in ``fit``)
        self._clf = None
        self._mean = np.zeros(_N_FEATS, dtype=np.float64)
        self._std = np.ones(_N_FEATS, dtype=np.float64)
        self._const_prob: Optional[float] = None  # set when only one class present
        self._fitted = False
        self.train_feats_: Optional[np.ndarray] = None
        self.train_gains_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ utils
    def _harm_labels(self, gains: np.ndarray) -> np.ndarray:
        return (np.asarray(gains, dtype=np.float64) < -self.harm_delta).astype(int)

    def _standardise(self, feats: np.ndarray) -> np.ndarray:
        feats = np.atleast_2d(np.asarray(feats, dtype=np.float64))
        return (feats - self._mean) / self._std

    def _make_clf(self):
        if self.model == "logistic":
            return LogisticRegression(max_iter=1000)
        return GradientBoostingClassifier()

    # -------------------------------------------------------------------- fit
    def fit(self, feats: np.ndarray, gains: np.ndarray) -> "SafetyGate":
        """Fit the gate on inner pseudo-target diagnostics and observed gains.

        Parameters
        ----------
        feats : array_like, shape (M, 8)
            Diagnostic vectors from inner pseudo-targets.
        gains : array_like, shape (M,)
            ``bAcc_adapt - bAcc_identity`` for each pseudo-target.
        """
        feats = np.atleast_2d(np.asarray(feats, dtype=np.float64))
        gains = np.asarray(gains, dtype=np.float64).ravel()
        if feats.shape[1] != _N_FEATS:
            raise ValueError(
                f"expected {_N_FEATS} features, got {feats.shape[1]}"
            )
        if feats.shape[0] != gains.shape[0]:
            raise ValueError("feats and gains must have the same length")

        # keep raw training data for threshold calibration / metrics
        self.train_feats_ = feats.copy()
        self.train_gains_ = gains.copy()

        # standardisation stats (guard zero / non-finite std)
        self._mean = np.nanmean(feats, axis=0)
        std = np.nanstd(feats, axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
        self._std = std
        self._mean = np.where(np.isfinite(self._mean), self._mean, 0.0)

        y = self._harm_labels(gains)
        Xs = self._standardise(feats)

        if np.unique(y).size < 2:
            # only one harm-class observed -> constant predictor
            self._clf = None
            # if everyone is harmful -> P(harm)=1, else P(harm)=0
            self._const_prob = 1.0 if (y.size and y[0] == 1) else 0.0
        else:
            self._const_prob = None
            clf = self._make_clf()
            clf.fit(Xs, y)
            self._clf = clf

        self._fitted = True
        return self

    # ----------------------------------------------------------- predict harm
    def predict_harm_prob(self, g: np.ndarray) -> float:
        """Return ``P(harm)`` for a single diagnostic vector ``g``."""
        if not self._fitted:
            raise RuntimeError("SafetyGate must be fitted before prediction")
        if self._const_prob is not None:
            return float(self._const_prob)
        g = np.asarray(g, dtype=np.float64).ravel()
        g = np.where(np.isfinite(g), g, 0.0)
        Xs = self._standardise(g)
        # locate the column for the positive (harm == 1) class
        classes = list(self._clf.classes_)
        if 1 in classes:
            pos = classes.index(1)
        else:  # pragma: no cover - defensive
            pos = len(classes) - 1
        prob = self._clf.predict_proba(Xs)[0, pos]
        return float(prob)

    # --------------------------------------------------------------- decision
    def should_adapt(
        self, g: np.ndarray, evidence: Optional[float] = None
    ) -> bool:
        """Decide whether to adapt the given (pseudo-)target.

        Returns ``True`` iff ``predict_harm_prob(g) < risk_threshold`` AND
        (``evidence is None`` OR ``evidence >= min_evidence``).
        """
        risk_ok = self.predict_harm_prob(g) < self.risk_threshold
        if evidence is None:
            evidence_ok = True
        else:
            ev = _finite(evidence)
            evidence_ok = ev >= self.min_evidence
        return bool(risk_ok and evidence_ok)

    # ---------------------------------------------------------------- metrics
    def harm_detection_metrics(
        self, feats: np.ndarray, gains: np.ndarray
    ) -> dict:
        """Evaluate harm detection / selective adaptation on a held-out set.

        Returns a dict with::

            auroc          : AUROC of P(harm) vs true harm labels (nan if 1 class)
            auprc          : average precision of P(harm) vs harm labels
            coverage       : fraction of targets the gate chose to adapt
            avoided_harm   : mean over NOT-adapted of max(0, -gain)  (harm avoided)
            missed_benefit : mean over NOT-adapted of max(0,  gain)  (benefit skipped)
            selective_gain : mean gain over ADAPTED targets
        """
        feats = np.atleast_2d(np.asarray(feats, dtype=np.float64))
        gains = np.asarray(gains, dtype=np.float64).ravel()
        m = feats.shape[0]

        y_true = self._harm_labels(gains)
        probs = np.array(
            [self.predict_harm_prob(feats[i]) for i in range(m)],
            dtype=np.float64,
        )
        adapt = np.array(
            [self.should_adapt(feats[i]) for i in range(m)], dtype=bool
        )

        # AUROC / AUPRC -- nan when a single class is present
        if np.unique(y_true).size < 2 or m == 0:
            auroc = float("nan")
            auprc = float("nan")
        else:
            try:
                auroc = float(roc_auc_score(y_true, probs))
            except ValueError:  # pragma: no cover - defensive
                auroc = float("nan")
            try:
                auprc = float(average_precision_score(y_true, probs))
            except ValueError:  # pragma: no cover - defensive
                auprc = float("nan")

        coverage = float(adapt.mean()) if m else float("nan")

        not_adapt = ~adapt
        if not_adapt.any():
            g_skip = gains[not_adapt]
            avoided_harm = float(np.maximum(0.0, -g_skip).mean())
            missed_benefit = float(np.maximum(0.0, g_skip).mean())
        else:
            avoided_harm = float("nan")
            missed_benefit = float("nan")

        if adapt.any():
            selective_gain = float(gains[adapt].mean())
        else:
            selective_gain = float("nan")

        return {
            "auroc": auroc,
            "auprc": auprc,
            "coverage": coverage,
            "avoided_harm": avoided_harm,
            "missed_benefit": missed_benefit,
            "selective_gain": selective_gain,
        }


# ============================================================== self-test ===
def _self_test() -> None:
    rng = np.random.default_rng(0)
    M = 200

    # Synthesize 8-dim diagnostics. Harm correlates with high condition_number
    # (index 2) and *negative* prior alignment, which we encode as large
    # prior_shift (index 3). We craft a latent harm-score and derive a gain.
    feats = rng.normal(0.0, 1.0, size=(M, _N_FEATS))
    # make condition_number and prior_shift non-negative-ish positive features
    feats[:, 2] = np.abs(feats[:, 2]) + rng.uniform(1.0, 3.0, size=M)  # cond num
    feats[:, 3] = np.abs(feats[:, 3])  # prior_shift magnitude
    feats[:, 7] = rng.uniform(20.0, 200.0, size=M)  # ess

    # latent harm signal: high condition number + high prior shift -> harm
    cond_z = (feats[:, 2] - feats[:, 2].mean()) / (feats[:, 2].std() + 1e-9)
    prior_z = (feats[:, 3] - feats[:, 3].mean()) / (feats[:, 3].std() + 1e-9)
    harm_score = 1.2 * cond_z + 1.0 * prior_z + rng.normal(0.0, 0.5, size=M)

    # gain is anti-correlated with harm_score (high harm_score -> negative gain)
    gains = -0.05 * harm_score + rng.normal(0.0, 0.01, size=M)

    # split train / held-out
    idx = rng.permutation(M)
    tr, te = idx[: M // 2], idx[M // 2:]

    gate = SafetyGate(model="logistic", harm_delta=0.0, risk_threshold=0.5)
    gate.fit(feats[tr], gains[tr])

    metrics = gate.harm_detection_metrics(feats[te], gains[te])
    print("harm_detection_metrics (held-out):")
    for k, v in metrics.items():
        print(f"  {k:14s} = {v:.4f}")

    auroc = metrics["auroc"]
    assert math.isfinite(auroc) and auroc > 0.6, (
        f"expected AUROC > 0.6 on easy synthetic signal, got {auroc}"
    )

    decision = gate.should_adapt(feats[te[0]], evidence=0.1)
    assert isinstance(decision, bool), "should_adapt must return a bool"
    print(f"should_adapt(example, evidence=0.1) = {decision} (type ok)")

    # also exercise the GBT backend + degenerate single-class path
    gate_gbt = SafetyGate(model="gbt").fit(feats[tr], gains[tr])
    assert isinstance(gate_gbt.should_adapt(feats[te[0]]), bool)

    all_safe = SafetyGate().fit(feats[:10], np.abs(gains[:10]) + 1.0)
    assert all_safe.predict_harm_prob(feats[0]) == 0.0
    assert all_safe.should_adapt(feats[0]) is True
    all_harm = SafetyGate().fit(feats[:10], -np.abs(gains[:10]) - 1.0)
    assert all_harm.predict_harm_prob(feats[0]) == 1.0
    assert all_harm.should_adapt(feats[0]) is False

    # gate_features helper: missing + non-finite handling
    gv = gate_features({"condition_number": 5.0, "ess": float("nan")})
    assert gv.shape == (_N_FEATS,)
    assert gv[GATE_FEATURE_KEYS.index("condition_number")] == 5.0
    assert gv[GATE_FEATURE_KEYS.index("ess")] == 0.0
    assert gv[GATE_FEATURE_KEYS.index("ood_score")] == 0.0

    print("\nAll SafetyGate self-tests passed.")


if __name__ == "__main__":
    _self_test()
