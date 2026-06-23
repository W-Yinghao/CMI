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
                         target_n_subjects=None, block_ids_tr=None,
                         alpha=0.05, n_block=240, quantile=None, seed=0,
                         quantile_method="linear") -> CertifierConfig:
    """tau_detect / tau_label = `quantile` of the certifier's EXACT statistic over pseudo-targets
    drawn from the TRAINING domains only (fold-isolated). Each pseudo-target draws
    `target_n_subjects` WHOLE subjects (matching the held-out target's CLUSTER count) and its
    visibility statistic is the SUBJECT-VOTE (cluster_mean) delta -- the SAME statistic the
    certifier thresholds (the v0 used a row mean, a mismatch). tau_resid / tau_margin are
    pre-registered constants passed through."""
    from ..certificate.atlas import components, cluster_mean
    from ..certificate.residual_test import stage_seed
    q = (1 - alpha) if quantile is None else quantile
    Z_tr = np.asarray(Z_tr, float); D_tr = np.asarray(D_tr)
    bids = (np.asarray(block_ids_tr) if block_ids_tr is not None
            else np.arange(len(Z_tr)))                       # each row its own cluster otherwise
    domains = list(np.unique(D_tr))
    rng = np.random.default_rng(stage_seed(seed, "calibration"))
    vis, lab = [], []
    reps = max(1, n_block // len(domains))
    for d in domains:
        idx = np.where(D_tr == d)[0]
        subs = np.unique(bids[idx])
        k = target_n_subjects or len(subs)
        for _ in range(reps):
            pick = rng.choice(subs, size=k, replace=True)    # match target SUBJECT count
            # new cluster id per drawn copy (correct resampling-with-replacement multiplicity)
            bs = np.concatenate([idx[bids[idx] == s] for s in pick])
            gid = np.concatenate([np.full((bids[idx] == s).sum(), i) for i, s in enumerate(pick)])
            delta = cluster_mean(Z_tr[bs], gid) - atlas.pooled_mean   # SUBJECT-VOTE, like certifier
            c = components(atlas, delta)
            vis.append(max(c["n_cov"], c["n_concept"], c["n_resid"]))
            lab.append(c["n_label"])
    return dataclasses.replace(base_cfg,
                               tau_detect=float(np.quantile(vis, q, method=quantile_method)),
                               tau_label=float(np.quantile(lab, q, method=quantile_method)))


def calibrate_tau_detect(Z, Y, D, atlas, alpha=0.05, n_block=240, seed=0) -> float:
    """Scalar convenience wrapper -- KEYWORD args (the v0 passed alpha/n_block/seed
    positionally into target_size/block_ids_tr/alpha, a real bug)."""
    cfg = calibrate_thresholds(Z, Y, D, atlas, CertifierConfig(),
                               alpha=alpha, n_block=n_block, seed=seed)
    return cfg.tau_detect


# --------------------------------------------------------------------------------------
# (#6) oracle boundary effect: refit bootstrap + two-sided equivalence band
# --------------------------------------------------------------------------------------
def _subj_w(groups):
    """1/n_s epoch weights, mean 1 (one vote per subject in a fit)."""
    uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    return w * (len(w) / w.sum())


def _subject_mean(values, groups):
    """Average `values` per BIOLOGICAL subject, then over subjects (one vote/subject)."""
    g = np.asarray(groups)
    return float(np.mean([values[g == u].mean() for u in np.unique(g)]))


def oracle_boundary_effect(Z_tr, Y_tr, Z_g, Y_g, classes,
                           n_boot=150, C=1.0, alpha=0.05,
                           eps_concept=0.03, eps_stable=0.01,
                           group_tr=None, group_g=None, seed=0) -> OracleEffect:
    """CSC-P1.4.2 #3: SUBJECT-level oracle, matching the decoder's estimand. The pooled and
    group-specific logistic fits are SUBJECT-weighted (1/n_s); the boundary effect is aggregated
    per BIOLOGICAL subject then averaged over subjects (not row-weighted); the OOB bootstrap
    resamples WHOLE subjects (paired conditions stay together)."""
    from ..certificate.residual_test import stage_seed
    assert 0 < eps_stable < eps_concept, "pre-register 0 < eps_stable < eps_concept"
    cl = list(classes)
    mu, sd = Z_tr.mean(0), Z_tr.std(0) + 1e-8
    Ztr, Zg = (Z_tr - mu) / sd, (Z_g - mu) / sd
    w_tr = None if group_tr is None else _subj_w(np.asarray(group_tr))
    M = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(Ztr, Y_tr, sample_weight=w_tr)
    pi_tr = np.array([(Y_tr == c).mean() for c in cl]) + 1e-9

    rng = np.random.default_rng(stage_seed(seed, "oracle"))
    ng = len(Y_g)
    gg = None if group_g is None else np.asarray(group_g)
    uniq = None if gg is None else np.unique(gg)
    idx_by = None if gg is None else {u: np.where(gg == u)[0] for u in uniq}
    boots = []
    for _ in range(n_boot):
        if gg is not None:                                  # whole-SUBJECT resample + OOB
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            # fresh subject id per drawn copy (correct multiplicity); paired conditions intact
            bs = np.concatenate([idx_by[u] for u in pick])
            gid_bs = np.concatenate([np.full(len(idx_by[u]), i) for i, u in enumerate(pick)])
            oob = np.concatenate([idx_by[u] for u in uniq if u not in set(pick.tolist())]) \
                if set(uniq.tolist()) - set(pick.tolist()) else np.array([], dtype=int)
        else:
            bs = rng.integers(0, ng, ng); gid_bs = bs
            oob = np.setdiff1d(np.arange(ng), np.unique(bs))
        if len(oob) < len(cl) + 2:
            continue
        ybs = Y_g[bs]
        if len(np.unique(ybs)) < 2 or len(np.unique(Y_g[oob])) < 1:
            continue
        # group-specific boundary refit on the bootstrap sample (SUBJECT-weighted)
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(
            Zg[bs], ybs, sample_weight=_subj_w(gid_bs))
        p_g = _aligned_proba(clf, Zg[oob], cl)
        # pooled boundary, prior-corrected to the group's ORACLE prior (isolates boundary)
        pi_g = np.array([(ybs == c).mean() for c in cl]) + 1e-9
        p_sh = _aligned_proba(M, Zg[oob], cl) * (pi_g / pi_tr)[None, :]
        p_sh /= p_sh.sum(1, keepdims=True)
        yi = np.searchsorted(cl, Y_g[oob])
        eff = (-np.log(p_sh[np.arange(len(oob)), yi])) - (-np.log(p_g[np.arange(len(oob)), yi]))
        boots.append(_subject_mean(eff, gg[oob] if gg is not None else np.arange(len(oob))))
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
def nested_lodo(Z, Y, D, cfg=None, folds=None, group_ids=None, min_stable=2, seed=0) -> LodoResult:
    """CALIBRATION/null bank via the SINGLE executor (csc.protocol.execute_protocol) -- so the
    LODO path uses the EXACT same parameters as deployment (no drift). `cfg` is a ProtocolConfig;
    its oracle bands (`oracle_eps_*`) are FROZEN INDEPENDENTLY and define the oracle TRUTH, never
    chosen from certificate performance. The oracle CI is subject-OOB when group_ids are given."""
    from ..protocol import execute_protocol, ProtocolConfig
    Z = np.asarray(Z, float); Y = np.asarray(Y); D = np.asarray(D)
    gid = None if group_ids is None else np.asarray(group_ids)
    classes = list(np.unique(Y)); domains = list(np.unique(D))
    cfg = cfg or ProtocolConfig(group_aware=(gid is not None))
    if folds is None:
        folds = [[d] for d in domains]

    records = []
    for g in folds:
        tr = ~np.isin(D, g); te = np.isin(D, g)
        if len(np.unique(D[tr])) < 2:
            continue
        gtr = None if gid is None else gid[tr]
        gte = None if gid is None else gid[te]
        out = execute_protocol(Z[tr], Y[tr], D[tr], Z[te], cfg,
                               src_group_ids=gtr, tgt_group_ids=gte, seed=seed)
        cert, sa = out["certificate"], out["analysis"]
        oracle = oracle_boundary_effect(Z[tr], Y[tr], Z[te], Y[te], classes,
                                        n_boot=cfg.oracle_boot, alpha=cfg.alpha,
                                        eps_concept=cfg.oracle_eps_concept_ce,
                                        eps_stable=cfg.oracle_eps_stable_ce,
                                        group_tr=gtr, group_g=gte, seed=seed)
        records.append(LodoRecord(fold=list(g), cert_state=cert.state, n_label=cert.n_label,
                                  n_cov=cert.n_cov, n_concept=cert.n_concept,
                                  tau_detect=out["tau_detect"],
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
                      detail=dict(bank="CALIBRATION_NULL_BANK", min_stable=min_stable,
                                  note="estimator sanity only; false-concept control is "
                                       "validated by synthetic_null_bank (generator truth)"))


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    from csc.sim.shift_simulator import SimConfig, make_source
    from csc.protocol import ProtocolConfig
    src = make_source(SimConfig(seed=4), n_domains=8, concept_domains=3, seed=4)
    cfg = ProtocolConfig(n_boot=30, n_dir_boot=100, oracle_boot=120)
    res = nested_lodo(src.Z, src.Y, src.D, cfg=cfg, group_ids=src.group_ids, seed=4)
    for r in res.records:
        o = r.oracle
        print(f"  fold {str(r.fold):>8} tau={r.tau_detect:.2f} -> {r.cert_state:>22} "
              f"| oracle {o.point:+.3f} [{o.lb:+.3f},{o.ub:+.3f}] {o.verdict}")
    print(f"CALIBRATION_NULL_BANK_VALID={res.valid_bank}  scorecard={res.scorecard}")
