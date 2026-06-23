"""
csc.protocol — executable manifest + the ONE internal executor (CSC-P1.4).

`ProtocolConfig` is an EXECUTABLE spec, not just a hashed record:
  * `validate()` rejects unsupported field values and FAILS CLOSED (group_aware ⇒ group ids
    are mandatory; degrading silently to IID is forbidden);
  * every field DRIVES behaviour (quantile_convention, tau_target_size_matched,
    tau_group_resampling, analysis_unit, rng/seed derivation), or it is not present;
  * the FULL canonical manifest (including rng algorithm + seed-derivation rule) hashes to a
    complete SHA-256 -> a method id that cannot hide algorithm differences.

`execute_protocol` is the SINGLE path. run_frozen_protocol, nested_lodo, ood_power_bank, the
sweep and any confirmatory runner ALL call it -> no parameter drift (the v0 LODO hand-built
the pipeline and silently used certify_robust defaults).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import hashlib
import json
from typing import Optional

import numpy as np

from .certificate import (
    analyze_source, certify_robust, CertifierConfig,
    COVARIATE_COMPATIBLE, CONCEPT_SUSPECT, UNIDENTIFIABLE,
)
from .calibration.lodo import calibrate_thresholds

from math import comb


def _cp_bound(failures, n, conf=0.95, side="upper"):
    """Exact one-sided Clopper-Pearson bound on a Bernoulli rate."""
    if n == 0:
        return 1.0 if side == "upper" else 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if side == "upper":              # P(X <= failures) = 1-conf
            cdf = sum(comb(n, k) * mid ** k * (1 - mid) ** (n - k) for k in range(failures + 1))
            hi, lo = (mid, lo) if cdf < 1 - conf else (hi, mid)
        else:                            # lower: P(X >= successes) = 1-conf
            succ = failures                      # here `failures` = #successes for the lower bound
            cdf = sum(comb(n, k) * mid ** k * (1 - mid) ** (n - k) for k in range(succ, n + 1))
            hi, lo = (hi, mid) if cdf < 1 - conf else (mid, lo)
    return (lo + hi) / 2


def _concept_failure_reason(out, alpha):
    """Binding-failure decomposition for a (generator-visible) target that did NOT fire
    CONCEPT_SUSPECT -- distinguishes the gates that abstained."""
    cert, sa = out["certificate"], out["analysis"]
    if cert.state == CONCEPT_SUSPECT:
        return "FIRED"
    if sa.test.status != "VALID":
        return "support_invalid"
    if sa.signature_overlap:
        return "signature_overlap"
    if sa.detail.get("p_global", 1.0) > alpha:
        return "geometric_maxstat_not_sig"                # geometric gate (type-I controlled)
    if not sa.test.significant:
        return "residual_T_not_sig"                       # cross-fitted decoder gate
    if not sa.concept_evidenced:
        return "concept_not_evidenced_other"
    return "not_dominant_or_robust_consensus_abstain"   # evidenced but certifier abstained


_SUPPORTED_TAU_METHOD = {"training_only_pseudotarget_quantile"}
_SUPPORTED_QUANTILE = {"linear"}
_SUPPORTED_UNIT = {"epoch", "subject"}
_SUPPORTED_RNG = {"numpy.default_rng(PCG64)"}


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class ProtocolConfig:
    # certifier dominance margins -- FIXED pre-registered constants
    tau_resid: float = 0.6
    tau_margin: float = 2.0
    # tau_detect / tau_label CALIBRATION RULE (drives calibrate_thresholds; never hard-set)
    tau_calibration_method: str = "training_only_pseudotarget_quantile"
    tau_quantile: float = 0.95
    tau_target_size_matched: bool = True
    tau_group_resampling: bool = True
    tau_n_pseudotargets: int = 240
    # source analysis
    alpha: float = 0.05
    var_keep: float = 0.95
    C: float = 0.5            # stronger L2 -> the high-dim Z(x)D interaction decoder does not
                             # overfit at the subject level (epoch fit, subject-aggregated loss)
    invalid_null_frac_max: float = 0.20   # CSC-P1.4.2 #1: > this fraction of invalid null
                                          # replicates (residual or geometry) -> SOURCE INVALID.
                                          # A METHOD parameter -> part of the manifest hash.
    source_cv_folds: int = 4
    n_boot: int = 80
    n_dir_boot: int = 200
    min_principal_angle_deg: float = 20.0
    # covariate-loading stability gate
    cov_loading_margin_kappa: float = 1.5
    # robust certificate
    consensus: float = 0.85
    target_n_boot: int = 200
    # oracle truth bands -- FROZEN INDEPENDENTLY of certificate performance
    oracle_eps_concept_ce: float = 0.03
    oracle_eps_stable_ce: float = 0.01
    oracle_boot: int = 150
    # inference unit + conventions  (these DRIVE execution)
    analysis_unit: str = "subject"          # "subject" -> cluster-vote gates/bootstraps
    group_aware: bool = True                 # True -> group ids MANDATORY (fail closed)
    quantile_convention: str = "linear"
    rng_algorithm: str = "numpy.default_rng(PCG64)"   # method-level RNG choice (the runtime
                                                      # root seed lives in ExecutionContext, NOT
                                                      # in the method manifest)

    def validate(self):
        if self.tau_calibration_method not in _SUPPORTED_TAU_METHOD:
            raise ProtocolError(f"tau_calibration_method {self.tau_calibration_method!r}")
        if self.quantile_convention not in _SUPPORTED_QUANTILE:
            raise ProtocolError(f"quantile_convention {self.quantile_convention!r}")
        if self.analysis_unit not in _SUPPORTED_UNIT:
            raise ProtocolError(f"analysis_unit {self.analysis_unit!r}")
        if self.rng_algorithm not in _SUPPORTED_RNG:
            raise ProtocolError(f"rng_algorithm {self.rng_algorithm!r}")
        if not (0 < self.oracle_eps_stable_ce < self.oracle_eps_concept_ce):
            raise ProtocolError("require 0 < oracle_eps_stable_ce < oracle_eps_concept_ce")
        if not (0 < self.alpha < 1):
            raise ProtocolError(f"alpha {self.alpha} out of (0,1)")
        if not (0 < self.tau_quantile < 1):
            raise ProtocolError(f"tau_quantile {self.tau_quantile} out of (0,1)")
        if not (0 < self.consensus <= 1):
            raise ProtocolError(f"consensus {self.consensus} out of (0,1]")
        if not (0 < self.var_keep <= 1):
            raise ProtocolError(f"var_keep {self.var_keep} out of (0,1]")
        if self.C <= 0:
            raise ProtocolError(f"C must be > 0, got {self.C}")
        if self.tau_resid < 0 or self.tau_margin < 1:
            raise ProtocolError("require tau_resid >= 0 and tau_margin >= 1")
        if self.cov_loading_margin_kappa <= 0:
            raise ProtocolError("cov_loading_margin_kappa must be > 0")
        if not (0 < self.invalid_null_frac_max < 1):
            raise ProtocolError(f"invalid_null_frac_max {self.invalid_null_frac_max} out of (0,1)")
        if self.source_cv_folds < 2:
            raise ProtocolError("source_cv_folds must be >= 2")
        for nm in ("n_boot", "n_dir_boot", "target_n_boot", "oracle_boot", "tau_n_pseudotargets"):
            if getattr(self, nm) < 1:
                raise ProtocolError(f"{nm} must be >= 1, got {getattr(self, nm)}")
        # CSC-P1.4.2 #6: this paper FIXES the subject-level estimand, so ONLY the cluster-valid
        # manifest is supported; every degenerate / unit-mismatched combination fails closed
        # (no silent row-level fallback, no calibrator/certifier unit mismatch).
        if not (self.analysis_unit == "subject" and self.group_aware and self.tau_group_resampling):
            raise ProtocolError(
                "unsupported manifest combination: this method REQUIRES "
                "analysis_unit='subject' AND group_aware=True AND tau_group_resampling=True "
                f"(got unit={self.analysis_unit!r}, group_aware={self.group_aware}, "
                f"tau_group_resampling={self.tau_group_resampling})")
        # CSC-P1.4.2 #1: a bootstrap p-value has minimum 1/(B+1); the null budget must be able to
        # REACH alpha, else the test can never reject regardless of evidence.
        import math
        b_min = math.ceil(1.0 / self.alpha) - 1
        for nm in ("n_boot", "n_dir_boot"):
            if getattr(self, nm) < b_min:
                raise ProtocolError(f"{nm}={getattr(self, nm)} < ceil(1/alpha)-1={b_min}: "
                                    f"bootstrap p-value cannot reach alpha={self.alpha}")
        return self

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def manifest(self) -> dict:
        d = self.to_dict()
        rule = dict(method=self.tau_calibration_method, quantile=self.tau_quantile,
                    target_size_matched=self.tau_target_size_matched,
                    group_resampling=self.tau_group_resampling,
                    n_pseudotargets=self.tau_n_pseudotargets)
        d["tau_detect"] = rule
        d["tau_label"] = rule
        return d

    def canonical_json(self) -> str:
        return json.dumps(self.manifest(), sort_keys=True, separators=(",", ":"))

    def hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()   # FULL sha256


from .certificate.residual_test import stage_seed as _stage_seed   # ONE shared derivation


@dataclass(frozen=True)
class ExecutionContext:
    root_seed: int = 0
    def seed(self, stage: str) -> int:
        return _stage_seed(self.root_seed, stage)


# --------------------------------------------------------------------------------------
# the ONE executor
# --------------------------------------------------------------------------------------
def execute_protocol(Z_src, Y_src, D_src, Z_tgt, cfg: ProtocolConfig,
                     src_group_ids=None, tgt_group_ids=None, seed: int = 0) -> dict:
    cfg.validate()
    if cfg.group_aware and (src_group_ids is None or tgt_group_ids is None):
        raise ProtocolError("group_aware=True requires BOTH src_group_ids and tgt_group_ids "
                            "(refusing to silently degrade to IID)")
    ctx = ExecutionContext(root_seed=seed)
    quantile = cfg.tau_quantile if cfg.tau_calibration_method == \
        "training_only_pseudotarget_quantile" else None
    cal_groups = src_group_ids if cfg.tau_group_resampling else None
    unit_groups_tgt = tgt_group_ids if cfg.analysis_unit == "subject" else None
    unit_groups_src = src_group_ids if cfg.analysis_unit == "subject" else None
    # match the target's CLUSTER (subject) count, not its row count
    tgt_n_subj = (len(np.unique(tgt_group_ids)) if (tgt_group_ids is not None and
                  cfg.tau_target_size_matched) else None)

    sa = analyze_source(Z_src, Y_src, D_src, n_boot=cfg.n_boot, n_dir_boot=cfg.n_dir_boot,
                        alpha=cfg.alpha, var_keep=cfg.var_keep, C=cfg.C,
                        min_angle_deg=cfg.min_principal_angle_deg,
                        cov_loading_margin_kappa=cfg.cov_loading_margin_kappa,
                        n_folds=cfg.source_cv_folds, invalid_frac_max=cfg.invalid_null_frac_max,
                        group_ids=unit_groups_src, seed=ctx.seed("analyze_source"))
    base = CertifierConfig(tau_resid=cfg.tau_resid, tau_margin=cfg.tau_margin)
    cal = calibrate_thresholds(Z_src, Y_src, D_src, sa.atlas, base,
                               target_n_subjects=tgt_n_subj, block_ids_tr=cal_groups,
                               alpha=cfg.alpha, n_block=cfg.tau_n_pseudotargets,
                               quantile=quantile, seed=ctx.seed("calibrate_thresholds"),
                               quantile_method=cfg.quantile_convention)
    cert = certify_robust(sa, Z_tgt, cfg=cal, n_boot=cfg.target_n_boot,
                          consensus=cfg.consensus, group_ids=unit_groups_tgt,
                          seed=ctx.seed("certify_robust"))
    return dict(certificate=cert, analysis=sa, calibrated_cfg=cal,
                tau_detect=cal.tau_detect, tau_label=cal.tau_label)


def run_frozen_protocol(Z_src, Y_src, D_src, Z_tgt, cfg: ProtocolConfig,
                        src_group_ids=None, tgt_group_ids=None, seed: int = 0) -> dict:
    """Public alias for the single executor (kept for readability)."""
    return execute_protocol(Z_src, Y_src, D_src, Z_tgt, cfg,
                            src_group_ids=src_group_ids, tgt_group_ids=tgt_group_ids, seed=seed)


# --------------------------------------------------------------------------------------
# OOD_POWER_BANK -- power on GENERATOR-TRUTH OOD targets, UNCONDITIONAL denominator (#6)
# --------------------------------------------------------------------------------------
def ood_power_bank(cfg: ProtocolConfig, seeds, n_domains: int = 8, concept_domains: int = 3,
                   min_visible: int = 5) -> dict:
    from .sim.shift_simulator import SimConfig, make_source, make_target
    recs = []
    for s in seeds:
        scfg = SimConfig(seed=s)
        src = make_source(scfg, n_domains=n_domains, concept_domains=concept_domains, seed=s)
        for kind, truth in [("covariate", "COVARIATE"), ("boundary_coupled", "CONCEPT_VISIBLE")]:
            tb = make_target(kind, scfg, geom=src.geom, seed=1000 + s)
            out = execute_protocol(src.Z, src.Y, src.D, tb.Z, cfg,
                                   src_group_ids=src.group_ids, tgt_group_ids=tb.group_ids,
                                   seed=s)
            sa = out["analysis"]
            recs.append(dict(seed=s, kind=kind, gen_truth=truth,
                             cert=out["certificate"].state,
                             concept_atlas=bool(sa.concept_evidenced),
                             fail_reason=_concept_failure_reason(out, cfg.alpha),
                             # separate evidence fields (not collapsed into one boolean)
                             support_valid=(sa.test.status == "VALID"),
                             signature_separable=(not sa.signature_overlap),
                             geometry_maxstat_significant=(sa.detail.get("p_global", 1.0) <= cfg.alpha),
                             residual_T_significant=bool(sa.test.significant),
                             joint_concept_evidence=bool(sa.concept_evidenced),
                             target_certificate_fired=(out["certificate"].state == CONCEPT_SUSPECT)))
    # UNCONDITIONAL: power denominator = ALL generator-visible clusters (atlas failures count
    # as power MISSES, reported separately as atlas_availability).
    vis = [r for r in recs if r["gen_truth"] == "CONCEPT_VISIBLE"]
    cov = [r for r in recs if r["gen_truth"] == "COVARIATE"]

    def rate(rs, pred):
        return (sum(pred(r) for r in rs) / len(rs)) if rs else None

    n_fired = sum(r["cert"] == CONCEPT_SUSPECT for r in vis)
    decomp = {}
    for r in vis:
        decomp[r["fail_reason"]] = decomp.get(r["fail_reason"], 0) + 1
    return dict(bank="OOD_POWER_BANK", n_records=len(recs), n_visible_total=len(vis),
                concept_power=rate(vis, lambda r: r["cert"] == CONCEPT_SUSPECT),
                # one-sided 95% LOWER Clopper-Pearson bound on concept power (per cluster)
                concept_power_cp_lower=_cp_bound(n_fired, len(vis), side="lower"),
                binding_failure_decomposition=decomp,
                # separate availabilities (NOT one collapsed flag)
                support_valid_rate=rate(vis, lambda r: r["support_valid"]),
                signature_separable_rate=rate(vis, lambda r: r["signature_separable"]),
                geometry_maxstat_sig_rate=rate(vis, lambda r: r["geometry_maxstat_significant"]),
                residual_T_sig_rate=rate(vis, lambda r: r["residual_T_significant"]),
                atlas_availability=rate(vis, lambda r: r["concept_atlas"]),   # = joint evidence
                covariate_compatible_coverage=rate(cov, lambda r: r["cert"] == COVARIATE_COMPATIBLE),
                false_concept_on_covariate=rate(cov, lambda r: r["cert"] == CONCEPT_SUSPECT),
                evaluable=(len(vis) >= min_visible), records=recs)


# --------------------------------------------------------------------------------------
# SYNTHETIC_NULL_BANK -- false-concept control on GENERATOR-TRUTH-stable targets (#5b)
# --------------------------------------------------------------------------------------
def synthetic_null_bank(cfg: ProtocolConfig, seeds, n_domains: int = 8, concept_domains: int = 3,
                        min_stable: int = 5) -> dict:
    """Generator KNOWS these targets have a stable boundary (covariate / clean), so this bank
    validates false-concept control DIRECTLY (no oracle needed). The LODO oracle bank is then
    only an estimator sanity-check."""
    from .sim.shift_simulator import SimConfig, make_source, make_target
    recs = []
    seed_cluster_fail = 0          # per INDEPENDENT source seed: any false-concept on a stable target
    seeds = list(seeds)
    for s in seeds:
        scfg = SimConfig(seed=s)
        src = make_source(scfg, n_domains=n_domains, concept_domains=concept_domains, seed=s)
        any_fc = False
        for kind in ("covariate", "clean"):
            tb = make_target(kind, scfg, geom=src.geom, seed=2000 + s)
            out = execute_protocol(src.Z, src.Y, src.D, tb.Z, cfg,
                                   src_group_ids=src.group_ids, tgt_group_ids=tb.group_ids,
                                   seed=s)
            st = out["certificate"].state
            recs.append(dict(seed=s, kind=kind, cert=st))
            if st == CONCEPT_SUSPECT:
                any_fc = True
        seed_cluster_fail += int(any_fc)
    n = len(recs)
    false_concept = sum(r["cert"] == CONCEPT_SUSPECT for r in recs)
    cp_ub = _cp_bound(seed_cluster_fail, len(seeds), side="upper")
    return dict(bank="SYNTHETIC_NULL_BANK", n_stable_total=n, n_source_clusters=len(seeds),
                false_concept_count=false_concept,
                false_concept_rate=(false_concept / n if n else None),
                seed_cluster_failures=seed_cluster_fail,
                false_concept_cp_upper_cluster=cp_ub,
                # evaluable = enough INDEPENDENT clusters to even compute a bound; control_PASS =
                # the cluster-level CP upper bound is at/below the pre-registered alpha. The two
                # are distinct: an evaluable bank with CP_ub > alpha has NOT validated control.
                evaluable=(len(seeds) >= min_stable),
                control_pass=(cp_ub <= cfg.alpha),
                records=recs)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    from csc.sim.shift_simulator import SimConfig, make_source, make_target
    cfg = ProtocolConfig(n_boot=20, n_dir_boot=80, target_n_boot=80, tau_n_pseudotargets=120)
    print("FULL manifest hash:", cfg.hash())
    src = make_source(SimConfig(seed=0), n_domains=8, concept_domains=3, seed=0)
    for kind in ("clean", "covariate", "boundary_coupled", "label_shift"):
        tb = make_target(kind, SimConfig(seed=0), geom=src.geom, seed=1)
        out = execute_protocol(src.Z, src.Y, src.D, tb.Z, cfg,
                               src_group_ids=src.group_ids, tgt_group_ids=tb.group_ids, seed=0)
        print(f"  {kind:16s} -> {out['certificate'].state:20s} (tau_detect={out['tau_detect']:.2f})")
    try:
        execute_protocol(src.Z, src.Y, src.D, tb.Z, cfg, seed=0)   # group_aware, no ids
    except ProtocolError as e:
        print("  fail-closed OK:", str(e)[:50])
