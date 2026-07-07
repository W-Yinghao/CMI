"""
csc.certificate.residual_test — source-side concept-evidence test.

Implements the cross-fitted, permutation-calibrated residual-decoder statistic

    T = CE( h0(Y | Z, D) ) - CE( h(Y | Z, D) )

  h0 : domain-INTERCEPT only  -> features [Z, onehot(D)]
       (shared Z->Y boundary; D may only change the per-class prior == label shift)
  h  : domain-DEPENDENT boundary -> features [Z, onehot(D), Z (x) onehot(D)]
       (the interaction lets the Z->Y boundary move per domain == concept variation)

T > 0 (out-of-fold) means: even after absorbing per-domain LABEL shift, the source
domains still carry domain-dependent BOUNDARY structure -> evidence of genuine concept
variation. The cross-fit removes the in-sample bias that a richer model would otherwise
get; the within-Y permutation null gives a calibrated threshold for the residual positive
fluctuation.

CRITICAL validity gate (the "single-class subject-domain" case): if any domain is not
class-spanning, or classes do not overlap across domains, the residual decoder degenerates
(I(Y;D|Z) collapses onto label predictability H(Y|Z)) and T is confounded. We then return
status=INVALID and the certifier must abstain rather than emit a (false) concept reading.

This is the falsifiable core. The H2-CMI P0-4 note is explicit that I(Y;D|Z) is a
*predictive-insufficiency diagnostic*, not "genuine concept shift"; the intercept/boundary
split + the support-graph gate are exactly what make the residual interpretable as concept.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss


@dataclass
class SupportGraph:
    valid: bool
    reasons: list
    n_domains: int
    n_classes: int
    min_classes_per_domain: int
    min_domains_per_class: int


@dataclass
class ResidualTestResult:
    status: str               # "VALID" | "INVALID"
    T: float                  # cross-fitted CE(h0) - CE(h)
    p_value: float            # permutation p (within-Y null); 1.0 if INVALID
    significant: bool
    null_mean: float
    null_q: float             # (1-alpha) quantile of the null
    support: SupportGraph
    ce_h0: float = float("nan")
    ce_h: float = float("nan")


def check_support_graph(Y, D, min_classes_per_domain: int = 2,
                        min_domains_per_class: int = 2) -> SupportGraph:
    Y = np.asarray(Y); D = np.asarray(D)
    classes = np.unique(Y); domains = np.unique(D)
    reasons = []
    cpd = min(int(np.unique(Y[D == d]).size) for d in domains)
    if cpd < min_classes_per_domain:
        reasons.append(f"a domain spans only {cpd} class(es) "
                       f"(need >= {min_classes_per_domain}); residual decoder degenerates")
    dpc = min(int(np.unique(D[Y == c]).size) for c in classes)
    if dpc < min_domains_per_class:
        reasons.append(f"a class appears in only {dpc} domain(s) "
                       f"(need >= {min_domains_per_class}); boundary not comparable")
    return SupportGraph(valid=len(reasons) == 0, reasons=reasons,
                        n_domains=int(domains.size), n_classes=int(classes.size),
                        min_classes_per_domain=cpd, min_domains_per_class=dpc)


def _onehot(D, domains) -> np.ndarray:
    idx = {d: i for i, d in enumerate(domains)}
    O = np.zeros((len(D), len(domains)))
    for i, d in enumerate(D):
        O[i, idx[d]] = 1.0
    return O


def _features(Z, D, domains, interaction: bool) -> np.ndarray:
    O = _onehot(D, domains)
    if not interaction:
        return np.concatenate([Z, O], axis=1)
    # Z (x) onehot(D): block of Z scaled by each domain indicator (per-domain Z weights)
    inter = (Z[:, :, None] * O[:, None, :]).reshape(len(Z), -1)
    return np.concatenate([Z, O, inter], axis=1)


def _xfit_ce(Z, Y, D, domains, interaction, classes, n_folds, C, rng_seed) -> float:
    """Out-of-fold cross-entropy of the (h0 or h) decoder."""
    X = _features(Z, D, domains, interaction)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rng_seed)
    ce_sum, n = 0.0, 0
    for tr, te in skf.split(X, Y):
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(X[tr], Y[tr])
        proba = clf.predict_proba(X[te])
        # align proba columns to the global class set (a fold may miss a class)
        full = np.full((len(te), len(classes)), 1e-12)
        for j, c in enumerate(clf.classes_):
            full[:, list(classes).index(c)] = proba[:, j]
        full /= full.sum(1, keepdims=True)
        ce_sum += log_loss(Y[te], full, labels=list(classes)) * len(te)
        n += len(te)
    return ce_sum / n


def residual_decoder_test(Z, Y, D,
                          n_folds: int = 4,
                          n_perm: int = 100,
                          alpha: float = 0.05,
                          C: float = 1.0,
                          seed: int = 0) -> ResidualTestResult:
    Z = np.asarray(Z, float); Y = np.asarray(Y); D = np.asarray(D)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)   # standardise -> even ridge over interactions
    classes = list(np.unique(Y)); domains = list(np.unique(D))

    support = check_support_graph(Y, D)
    if not support.valid:
        return ResidualTestResult(status="INVALID", T=float("nan"), p_value=1.0,
                                  significant=False, null_mean=float("nan"),
                                  null_q=float("nan"), support=support)

    ce_h0 = _xfit_ce(Z, Y, D, domains, False, classes, n_folds, C, seed)
    ce_h = _xfit_ce(Z, Y, D, domains, True, classes, n_folds, C, seed)
    T = ce_h0 - ce_h

    # within-Y permutation null: shuffle D inside each class -> destroys boundary-by-domain
    # while preserving class counts. Recompute T under each permutation.
    rng = np.random.default_rng(seed + 1)
    null = np.empty(n_perm)
    for p in range(n_perm):
        Dp = D.copy()
        for c in classes:
            m = np.where(Y == c)[0]
            Dp[m] = D[m][rng.permutation(len(m))]
        ce_h0_p = _xfit_ce(Z, Y, Dp, domains, False, classes, n_folds, C, seed)
        ce_h_p = _xfit_ce(Z, Y, Dp, domains, True, classes, n_folds, C, seed)
        null[p] = ce_h0_p - ce_h_p

    p_value = (1.0 + np.sum(null >= T)) / (1.0 + n_perm)
    null_q = float(np.quantile(null, 1.0 - alpha))
    return ResidualTestResult(status="VALID", T=float(T), p_value=float(p_value),
                              significant=bool(p_value <= alpha),
                              null_mean=float(null.mean()), null_q=null_q,
                              support=support, ce_h0=float(ce_h0), ce_h=float(ce_h))


if __name__ == "__main__":
    from csc.sim.shift_simulator import SimConfig, make_source
    cfg = SimConfig(seed=1)
    src = make_source(cfg, n_domains=8, concept_domains=3)
    r = residual_decoder_test(src.Z, src.Y, src.D, n_perm=60)
    print(f"status={r.status}  T={r.T:+.4f}  p={r.p_value:.3f}  "
          f"sig={r.significant}  null_mean={r.null_mean:+.4f}  null_q={r.null_q:+.4f}")
    print(f"  CE(h0)={r.ce_h0:.4f}  CE(h)={r.ce_h:.4f}  support={r.support.valid}")
