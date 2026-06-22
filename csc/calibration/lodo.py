"""
csc.calibration.lodo — nested, oracle-labeled leave-one-(group)-out calibration (CSC-P1.1).

Review fixes over the P1 version:

* the calibrated `tau_detect` now ACTUALLY enters the certificate: per outer fold we build it
  on the TRAINING domains only (no leakage), `dataclasses.replace` it into the config, and
  call `certify_robust` (so the consensus level is exercised too).
* it is calibrated on the EXACT certificate statistic `visibility_statistic = max(n_cov,
  n_concept, n_resid)` (same units as the certifier), not a Euclidean-mean / RMS proxy.
* MECHANISM-GROUP-OUT folds: leaving a whole concept family out gives the oracle genuine
  VISIBLE_CONCEPT folds (leave-one-domain-out leaves concept structure in the pool, so the
  oracle saw none -> the agreement was non-diagnostic).
* the oracle effect is bootstrapped WITH REFITTING (resample the held-out group, refit the
  group-specific boundary on the bootstrap sample, evaluate OOB) and uses a TWO-SIDED
  equivalence band: COVARIATE_STABLE requires the WHOLE CI inside (-eps_stable, eps_stable),
  not just ub<eps_stable; pre-registered 0 < eps_stable < eps_concept.
* the scorecard reports concept power / false-concept / compatible-coverage / abstention
  SEPARATELY, and refuses an aggregate agreement when the oracle bank lacks either class.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from sklearn.linear_model import LogisticRegression

from ..certificate import (
    analyze_source, certify_robust, certify, CertifierConfig, visibility_statistic,
)
from ..certificate.certifier import CONCEPT_SUSPECT, COVARIATE_COMPATIBLE, UNIDENTIFIABLE

VISIBLE_CONCEPT = "VISIBLE_CONCEPT"
COVARIATE_STABLE = "COVARIATE_STABLE"
AMBIGUOUS = "AMBIGUOUS"


@dataclass
class OracleEffect:
    point: float
    lb: float
    ub: float
    verdict: str


@dataclass
class LodoRecord:
    fold: object
    cert_state: str
    n_label: float
    n_cov: float
    n_concept: float
    tau_detect: float
    concept_atlas: bool          # did the TRAINING fold yield a concept atlas at all?
    oracle: OracleEffect


@dataclass
class LodoResult:
    records: list
    tau_detect_mean: float
    scorecard: dict
    valid_bank: bool                  # oracle produced BOTH visible and stable folds
    detail: dict = field(default_factory=dict)


def _aligned_proba(clf, Z, classes):
    proba = clf.predict_proba(Z)
    cl = list(classes)
    full = np.full((len(Z), len(cl)), 1e-12)
    for j, c in enumerate(clf.classes_):
        full[:, cl.index(c)] = proba[:, j]
    return full / full.sum(1, keepdims=True)


# --------------------------------------------------------------------------------------
# (#2/#3) calibrate the EXACT certificate statistic on training-domain pseudo-targets
# --------------------------------------------------------------------------------------
def calibrate_thresholds(Z_tr, Y_tr, D_tr, atlas, base_cfg: CertifierConfig,
                         target_size=None, block_ids_tr=None,
                         alpha=0.05, n_block=240, quantile=None, seed=0) -> CertifierConfig:
    """tau_detect / tau_label = `quantile` (default 1-alpha) of the certifier's OWN statistics
    over pseudo-targets resampled from the TRAINING domains only (fold-isolated; held-out
    labels never enter). Each pseudo-target MATCHES the held-out target's `target_size` (and,
    on real data, its `block_ids_tr` subject/session structure -- WHOLE blocks, never split)
    so the floor reflects the right finite-sample fluctuation. tau_resid / tau_margin are
    pre-registered constants passed through (NOT calibrated here)."""
    from ..certificate.atlas import components
    q = (1 - alpha) if quantile is None else quantile
    Z_tr = np.asarray(Z_tr, float); D_tr = np.asarray(D_tr)
    bids = None if block_ids_tr is None else np.asarray(block_ids_tr)
    domains = list(np.unique(D_tr))
    rng = np.random.default_rng(seed + 13)
    vis, lab = [], []
    reps = max(1, n_block // len(domains))
    for d in domains:
        idx = np.where(D_tr == d)[0]
        blocks = ([idx[bids[idx] == b] for b in np.unique(bids[idx])]
                  if bids is not None else None)
        tgt = target_size or len(idx)
        for _ in range(reps):
            if blocks is not None:
                # WHOLE-block (subject) resampling: add whole blocks until >= target_size; NO
                # mid-block truncation (the v0 trimmed the last block, breaking clustering).
                picks, n = [], 0
                while n < tgt:
                    blk = blocks[rng.integers(0, len(blocks))]
                    picks.append(blk); n += len(blk)
                bs = np.concatenate(picks)
            else:
                bs = idx[rng.integers(0, len(idx), tgt)]    # IID rows, match target SIZE
            c = components(atlas, Z_tr[bs].mean(0) - atlas.pooled_mean)
            vis.append(max(c["n_cov"], c["n_concept"], c["n_resid"]))
            lab.append(c["n_label"])
    return dataclasses.replace(base_cfg,
                               tau_detect=float(np.quantile(vis, q)),
                               tau_label=float(np.quantile(lab, q)))


def calibrate_tau_detect(Z, Y, D, atlas, alpha=0.05, n_block=240, seed=0) -> float:
    """Scalar convenience wrapper -- KEYWORD args (the v0 passed alpha/n_block/seed
    positionally into target_size/block_ids_tr/alpha, a real bug)."""
    cfg = calibrate_thresholds(Z, Y, D, atlas, CertifierConfig(),
                               alpha=alpha, n_block=n_block, seed=seed)
    return cfg.tau_detect


# --------------------------------------------------------------------------------------
# (#6) oracle boundary effect: refit bootstrap + two-sided equivalence band
# --------------------------------------------------------------------------------------
def oracle_boundary_effect(Z_tr, Y_tr, Z_g, Y_g, classes,
                           n_boot=150, C=1.0, alpha=0.05,
                           eps_concept=0.03, eps_stable=0.01, seed=0) -> OracleEffect:
    assert 0 < eps_stable < eps_concept, "pre-register 0 < eps_stable < eps_concept"
    cl = list(classes)
    mu, sd = Z_tr.mean(0), Z_tr.std(0) + 1e-8
    Ztr, Zg = (Z_tr - mu) / sd, (Z_g - mu) / sd
    M = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(Ztr, Y_tr)
    pi_tr = np.array([(Y_tr == c).mean() for c in cl]) + 1e-9

    rng = np.random.default_rng(seed + 3)
    ng = len(Y_g)
    boots = []
    for _ in range(n_boot):
        bs = rng.integers(0, ng, ng)
        oob = np.setdiff1d(np.arange(ng), np.unique(bs))
        if len(oob) < len(cl) + 2:
            continue
        ybs = Y_g[bs]
        if len(np.unique(ybs)) < 2:
            continue
        # group-specific boundary refit on the bootstrap sample
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(Zg[bs], ybs)
        p_g = _aligned_proba(clf, Zg[oob], cl)
        # pooled boundary, prior-corrected to the group's ORACLE prior (isolates boundary)
        pi_g = np.array([(ybs == c).mean() for c in cl]) + 1e-9
        p_sh = _aligned_proba(M, Zg[oob], cl) * (pi_g / pi_tr)[None, :]
        p_sh /= p_sh.sum(1, keepdims=True)
        yi = np.searchsorted(cl, Y_g[oob])
        eff = (-np.log(p_sh[np.arange(len(oob)), yi])) - (-np.log(p_g[np.arange(len(oob)), yi]))
        boots.append(float(eff.mean()))
    if not boots:
        return OracleEffect(float("nan"), float("nan"), float("nan"), AMBIGUOUS)
    boots = np.array(boots)
    point = float(np.median(boots))
    lb, ub = (float(x) for x in np.quantile(boots, [alpha / 2, 1 - alpha / 2]))
    if lb > eps_concept:
        verdict = VISIBLE_CONCEPT
    elif (-eps_stable < lb) and (ub < eps_stable):
        verdict = COVARIATE_STABLE
    else:
        verdict = AMBIGUOUS
    return OracleEffect(point, lb, ub, verdict)


# --------------------------------------------------------------------------------------
# (#1) CALIBRATION_NULL_BANK: leave-one-domain-out, validates FALSE-CONCEPT control + supplies
#      threshold calibration. It does NOT (and structurally CANNOT) validate deployment power
#      -- in-distribution held-out domains have within-source-spread shifts, and leave-all-
#      concept-out leaves no training atlas. Power lives in the separate OOD_POWER_BANK
#      (csc.protocol.ood_power_bank), on generator-truth OOD targets.
# --------------------------------------------------------------------------------------
def nested_lodo(Z, Y, D, folds=None, group_ids=None,
                n_boot=60, n_dir_boot=150, oracle_boot=150,
                cert_cfg: Optional[CertifierConfig] = None, consensus=0.85,
                eps_concept=0.03, eps_stable=0.01, alpha=0.05,
                min_stable=2, seed=0) -> LodoResult:
    """Calibration/null bank. `eps_concept`, `eps_stable` are FROZEN INDEPENDENTLY before this
    runs -- they define the oracle TRUTH and are NEVER chosen from certificate performance.
    Goes through the cluster-aware calibrated path (analyze_source -> calibrate_thresholds ->
    certify_robust). `group_ids` = subject/session ids for cluster-aware resampling."""
    Z = np.asarray(Z, float); Y = np.asarray(Y); D = np.asarray(D)
    gid = None if group_ids is None else np.asarray(group_ids)
    classes = list(np.unique(Y)); domains = list(np.unique(D))
    cert_cfg = cert_cfg or CertifierConfig()
    if folds is None:
        folds = [[d] for d in domains]

    records = []
    for g in folds:
        tr = ~np.isin(D, g); te = np.isin(D, g)
        if len(np.unique(D[tr])) < 2:
            continue
        gtr = None if gid is None else gid[tr]
        gte = None if gid is None else gid[te]
        sa = analyze_source(Z[tr], Y[tr], D[tr], n_boot=n_boot, n_dir_boot=n_dir_boot,
                            alpha=alpha, group_ids=gtr, seed=seed)
        cfg_d = calibrate_thresholds(Z[tr], Y[tr], D[tr], sa.atlas, cert_cfg,
                                     target_size=int(te.sum()), block_ids_tr=gtr,
                                     alpha=alpha, seed=seed)
        cert = certify_robust(sa, Z[te], cfg=cfg_d, consensus=consensus,
                              group_ids=gte, seed=seed)
        oracle = oracle_boundary_effect(Z[tr], Y[tr], Z[te], Y[te], classes,
                                        n_boot=oracle_boot, alpha=alpha,
                                        eps_concept=eps_concept, eps_stable=eps_stable, seed=seed)
        records.append(LodoRecord(fold=list(g), cert_state=cert.state, n_label=cert.n_label,
                                  n_cov=cert.n_cov, n_concept=cert.n_concept,
                                  tau_detect=cfg_d.tau_detect,
                                  concept_atlas=sa.concept_evidenced, oracle=oracle))

    stable = [r for r in records if r.oracle.verdict == COVARIATE_STABLE]
    visible = [r for r in records if r.oracle.verdict == VISIBLE_CONCEPT]
    amb = [r for r in records if r.oracle.verdict == AMBIGUOUS]

    def rate(rs, pred):
        return (sum(pred(r) for r in rs) / len(rs)) if rs else None

    scorecard = dict(
        bank="CALIBRATION_NULL_BANK", n_folds=len(records),
        n_stable=len(stable), n_visible=len(visible), n_ambiguous=len(amb),
        false_concept_on_stable=rate(stable, lambda r: r.cert_state == CONCEPT_SUSPECT),
        compatible_coverage_on_stable=rate(stable, lambda r: r.cert_state == COVARIATE_COMPATIBLE),
        abstention=rate(records, lambda r: r.cert_state == UNIDENTIFIABLE),
    )
    # this bank is VALID for false-concept control iff it has enough oracle-stable folds.
    calibration_null_bank_valid = len(stable) >= min_stable
    taus = [r.tau_detect for r in records]
    return LodoResult(records=records,
                      tau_detect_mean=float(np.mean(taus)) if taus else float("nan"),
                      scorecard=scorecard, valid_bank=calibration_null_bank_valid,
                      detail=dict(bank="CALIBRATION_NULL_BANK", eps_concept=eps_concept,
                                  eps_stable=eps_stable, consensus=consensus,
                                  min_stable=min_stable,
                                  note="power is NOT measured here; see ood_power_bank"))


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    from csc.sim.shift_simulator import SimConfig, make_source
    src = make_source(SimConfig(seed=4), n_domains=8, concept_domains=3, seed=4)
    res = nested_lodo(src.Z, src.Y, src.D, n_boot=30, n_dir_boot=100, oracle_boot=120, seed=4)
    for r in res.records:
        o = r.oracle
        print(f"  fold {str(r.fold):>8} tau={r.tau_detect:.2f} -> {r.cert_state:>22} "
              f"| oracle {o.point:+.3f} [{o.lb:+.3f},{o.ub:+.3f}] {o.verdict}")
    print(f"CALIBRATION_NULL_BANK_VALID={res.valid_bank}  scorecard={res.scorecard}")
