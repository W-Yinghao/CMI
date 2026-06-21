"""
csc.calibration.lodo — nested, oracle-labeled leave-one-domain-out calibration (CSC-P1).

The v0 PREREGISTRATION asked that a held-out source domain "never be CONCEPT_SUSPECT".
That is WRONG: the source CONTAINS genuine concept domains, so forcing a held-out concept
domain to be null calibrates away the very power we need. This module instead:

  for each held-out source domain d:
    1. build the atlas + residual evidence on the OTHER domains only  (NO leakage);
    2. certify d as an UNLABELED pseudo-target;
    3. compute an ORACLE boundary-effect on d WITH d's labels (calibration only), as an
       EQUIVALENCE test with a bootstrap CI:
           oracle_lb(d) > eps_concept   -> d genuinely moved its boundary (VISIBLE_CONCEPT)
           oracle_ub(d) < eps_stable    -> d's boundary == pooled (COVARIATE_STABLE)
           otherwise                    -> AMBIGUOUS (excluded from the forced binary)
    4. score the (label-blind) certificate against the (label-aware) oracle verdict.

The oracle isolates BOUNDARY movement from LABEL shift: the pooled (shared-boundary) model
is prior-corrected to d using d's ORACLE class frequencies before its CE is compared to a
d-specific refit. So a pure label shift contributes ~0 to the oracle effect.

tau_detect is calibrated from BLOCK-RESAMPLED pseudo-targets within the training domains
(the finite-sample fluctuation of a held-out-domain mean), NOT hand-set.

"Not rejecting a boundary shift is NOT proving stability" -> two-sided equivalence bands.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from ..certificate import analyze_source, certify, CertifierConfig
from ..certificate.certifier import CONCEPT_SUSPECT, COVARIATE_COMPATIBLE, UNIDENTIFIABLE


VISIBLE_CONCEPT = "VISIBLE_CONCEPT"
COVARIATE_STABLE = "COVARIATE_STABLE"
AMBIGUOUS = "AMBIGUOUS"


@dataclass
class OracleEffect:
    point: float              # mean boundary-effect (nats), label-shift-corrected
    lb: float                 # bootstrap lower CI
    ub: float                 # bootstrap upper CI
    verdict: str              # VISIBLE_CONCEPT | COVARIATE_STABLE | AMBIGUOUS


@dataclass
class LodoRecord:
    domain: object
    cert_state: str
    n_label: float
    n_cov: float
    n_concept: float
    oracle: OracleEffect


@dataclass
class LodoResult:
    records: list
    tau_detect: float                 # calibrated detection floor (between-domain fluct.)
    agreement: float                  # cert vs oracle on non-ambiguous domains
    n_nonambiguous: int
    detail: dict = field(default_factory=dict)


def _aligned_proba(clf, Z, classes):
    proba = clf.predict_proba(Z)
    cl = list(classes)
    full = np.full((len(Z), len(cl)), 1e-12)
    for j, c in enumerate(clf.classes_):
        full[:, cl.index(c)] = proba[:, j]
    return full / full.sum(1, keepdims=True)


def _persample_xfit_loss(Z, Y, classes, n_folds, C, seed):
    """Per-sample cross-fitted -log p(y|z) for a d-specific (Z-only) boundary."""
    cl = list(classes)
    loss = np.zeros(len(Y))
    counts = np.array([(Y == c).sum() for c in cl])
    nf = int(min(n_folds, counts[counts > 0].min())) if (counts > 0).any() else 2
    nf = max(nf, 2)
    skf = StratifiedKFold(n_splits=nf, shuffle=True, random_state=seed)
    for tr, te in skf.split(Z, Y):
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(Z[tr], Y[tr])
        p = _aligned_proba(clf, Z[te], cl)
        yi = np.searchsorted(cl, Y[te])
        loss[te] = -np.log(p[np.arange(len(te)), yi])
    return loss


def oracle_boundary_effect(Z_tr, Y_tr, Z_d, Y_d, classes,
                           n_boot=300, n_folds=4, C=1.0, alpha=0.05,
                           eps_concept=0.02, eps_stable=0.02, seed=0) -> OracleEffect:
    """Boundary movement of domain d vs the training pooled boundary, label-shift corrected,
    with a bootstrap CI and an equivalence verdict."""
    cl = list(classes)
    mu, sd = Z_tr.mean(0), Z_tr.std(0) + 1e-8
    Ztr = (Z_tr - mu) / sd
    Zd = (Z_d - mu) / sd

    M = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(Ztr, Y_tr)
    pi_tr = np.array([(Y_tr == c).mean() for c in cl]) + 1e-9
    pi_d = np.array([(Y_d == c).mean() for c in cl]) + 1e-9     # ORACLE prior (calibration)

    p_share = _aligned_proba(M, Zd, cl)
    adj = p_share * (pi_d / pi_tr)[None, :]
    adj /= adj.sum(1, keepdims=True)
    yi = np.searchsorted(cl, Y_d)
    loss_pooled = -np.log(adj[np.arange(len(Y_d)), yi])         # pooled boundary @ d's prior
    loss_dspec = _persample_xfit_loss(Zd, Y_d, cl, n_folds, C, seed)
    Ti = loss_pooled - loss_dspec                              # >0 => boundary genuinely moved

    rng = np.random.default_rng(seed + 3)
    boots = np.array([Ti[rng.integers(0, len(Ti), len(Ti))].mean() for _ in range(n_boot)])
    lb, ub = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    point = float(Ti.mean())
    if lb > eps_concept:
        verdict = VISIBLE_CONCEPT
    elif ub < eps_stable:
        verdict = COVARIATE_STABLE
    else:
        verdict = AMBIGUOUS
    return OracleEffect(point=point, lb=float(lb), ub=float(ub), verdict=verdict)


def calibrate_tau_detect(Z, Y, D, alpha=0.05, n_block=200, seed=0) -> float:
    """Detection floor = high quantile of the (normalised) between-domain mean fluctuation,
    estimated by block-resampling each training domain. A target shift below this is within
    normal source wobble and must NOT count as 'visible'."""
    Z = np.asarray(Z, float); D = np.asarray(D)
    domains = list(np.unique(D))
    pooled = Z.mean(0)
    # scale = RMS domain-mean deviation (robust normaliser)
    dm = np.stack([Z[D == d].mean(0) for d in domains])
    scale = float(np.sqrt(((dm - pooled) ** 2).sum(1).mean())) + 1e-8
    rng = np.random.default_rng(seed + 5)
    fluct = []
    for d in domains:
        idx = np.where(D == d)[0]
        for _ in range(max(1, n_block // len(domains))):
            bs = idx[rng.integers(0, len(idx), len(idx))]
            fluct.append(np.linalg.norm(Z[bs].mean(0) - pooled) / scale)
    return float(np.quantile(fluct, 1 - alpha))


def nested_lodo(Z, Y, D,
                n_boot=80, n_dir_boot=150, oracle_boot=300,
                cert_cfg: Optional[CertifierConfig] = None,
                eps_concept=0.02, eps_stable=0.02, alpha=0.05, seed=0) -> LodoResult:
    Z = np.asarray(Z, float); Y = np.asarray(Y); D = np.asarray(D)
    classes = list(np.unique(Y)); domains = list(np.unique(D))
    cert_cfg = cert_cfg or CertifierConfig()

    tau = calibrate_tau_detect(Z, Y, D, alpha=alpha, seed=seed)

    records = []
    for d in domains:
        tr = D != d
        te = D == d
        sa = analyze_source(Z[tr], Y[tr], D[tr], n_boot=n_boot, n_dir_boot=n_dir_boot,
                            alpha=alpha, seed=seed)
        cert = certify(sa, Z[te], cert_cfg)
        oracle = oracle_boundary_effect(Z[tr], Y[tr], Z[te], Y[te], classes,
                                        n_boot=oracle_boot, alpha=alpha,
                                        eps_concept=eps_concept, eps_stable=eps_stable,
                                        seed=seed)
        records.append(LodoRecord(domain=d, cert_state=cert.state, n_label=cert.n_label,
                                  n_cov=cert.n_cov, n_concept=cert.n_concept, oracle=oracle))

    # agreement on NON-ambiguous oracle verdicts only (the forced-binary cases)
    agree, n_na = 0, 0
    for r in records:
        if r.oracle.verdict == VISIBLE_CONCEPT:
            n_na += 1
            agree += int(r.cert_state == CONCEPT_SUSPECT)
        elif r.oracle.verdict == COVARIATE_STABLE:
            n_na += 1
            # stable => certificate must NOT cry concept (abstain or compatible both ok)
            agree += int(r.cert_state != CONCEPT_SUSPECT)
    agreement = agree / n_na if n_na else float("nan")
    return LodoResult(records=records, tau_detect=tau, agreement=agreement,
                      n_nonambiguous=n_na,
                      detail=dict(eps_concept=eps_concept, eps_stable=eps_stable))


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    from csc.sim.shift_simulator import SimConfig, make_source
    src = make_source(SimConfig(seed=4), n_domains=8, concept_domains=3, seed=4)
    res = nested_lodo(src.Z, src.Y, src.D, n_boot=50, n_dir_boot=100, oracle_boot=200, seed=4)
    print(f"calibrated tau_detect = {res.tau_detect:.3f}")
    print(f"{'domain':>6} {'cert':>22} {'oracle_pt':>10} {'oracle_CI':>18} {'verdict':>16}")
    for r in res.records:
        o = r.oracle
        print(f"{str(r.domain):>6} {r.cert_state:>22} {o.point:>10.3f} "
              f"[{o.lb:+.3f},{o.ub:+.3f}]   {o.verdict:>16}")
    print(f"cert-vs-oracle agreement on {res.n_nonambiguous} non-ambiguous domains "
          f"= {res.agreement:.2f}")
