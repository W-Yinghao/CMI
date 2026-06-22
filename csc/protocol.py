"""
csc.protocol — the SINGLE frozen-path entrypoint + serializable manifest (CSC-P1.3).

The whole point of this module is that development sweeps, unit tests, the audit, and any
future confirmatory run ALL go through ONE function, `run_frozen_protocol`, parameterised by
ONE fully-serializable `ProtocolConfig`. The v0 audit measured `certify(...)` directly --
NOT the calibrated `certify_robust` path that would be frozen -- so its numbers did not
characterise the deployable algorithm. That is fixed here: there is no other way to certify.

`ProtocolConfig`
  * records tau_detect / tau_label as a CALIBRATION RULE (method + quantile + matching), NOT
    as default numbers (those are computed per source);
  * records every knob that affects the result (CV folds, bootstrap counts, kappa, oracle
    bands, consensus, quantile convention, group-awareness);
  * hashes to a canonical JSON -> a manifest hash that uniquely identifies the method.

Two validity banks (separate objects -- see csc.calibration.lodo for the NULL bank):
  * CALIBRATION_NULL_BANK  -> false-concept control + threshold calibration (in-distribution
    LODO; NOT power).
  * OOD_POWER_BANK (here)   -> deployment POWER on GENERATOR-TRUTH out-of-distribution targets
    through the SAME frozen path, with the training source retaining its concept atlas.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import hashlib
import json
from typing import Optional

import numpy as np

from .certificate import (
    analyze_source, certify_robust, CertifierConfig,
    COVARIATE_COMPATIBLE, CONCEPT_SUSPECT, UNIDENTIFIABLE,
)
from .calibration.lodo import calibrate_thresholds


@dataclass(frozen=True)
class ProtocolConfig:
    # certifier dominance margins -- FIXED pre-registered constants
    tau_resid: float = 0.6
    tau_margin: float = 2.0
    # tau_detect / tau_label CALIBRATION RULE (values computed per source, never hard-set)
    tau_calibration_method: str = "training_only_pseudotarget_quantile"
    tau_quantile: float = 0.95
    tau_target_size_matched: bool = True
    tau_group_resampling: bool = True
    tau_n_pseudotargets: int = 240
    # source analysis
    alpha: float = 0.05
    var_keep: float = 0.95
    C: float = 1.0
    source_cv_folds: int = 4
    n_boot: int = 80                 # residual-test h0 null bootstrap (reporting)
    n_dir_boot: int = 200            # concept/cov h0 null + cov cluster bootstrap
    min_principal_angle_deg: float = 20.0
    # covariate equivalence-stability
    cov_loading_margin_kappa: float = 1.5
    # robust certificate (confidence-region consensus)
    consensus: float = 0.85
    target_n_boot: int = 200
    # oracle TRUTH bands -- FROZEN INDEPENDENTLY of any certificate performance
    oracle_eps_concept_ce: float = 0.03
    oracle_eps_stable_ce: float = 0.01
    oracle_boot: int = 150
    # conventions
    quantile_convention: str = "linear"
    group_aware: bool = True

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def tau_rule(self) -> dict:
        return dict(method=self.tau_calibration_method, quantile=self.tau_quantile,
                    target_size_matched=self.tau_target_size_matched,
                    group_resampling=self.tau_group_resampling,
                    n_pseudotargets=self.tau_n_pseudotargets)

    def manifest(self) -> dict:
        """Human/machine-readable manifest: tau_detect/tau_label appear as RULES, not numbers."""
        d = self.to_dict()
        d["tau_detect"] = self.tau_rule()
        d["tau_label"] = self.tau_rule()
        return d

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]


def run_frozen_protocol(Z_src, Y_src, D_src, Z_tgt, cfg: ProtocolConfig,
                        src_group_ids=None, tgt_group_ids=None, seed: int = 0) -> dict:
    """THE single certification path. Source-only calibration -> calibrated robust certificate.
    Everything (sweep / tests / audit / confirmatory) must call THIS."""
    sa = analyze_source(Z_src, Y_src, D_src, n_boot=cfg.n_boot, n_dir_boot=cfg.n_dir_boot,
                        alpha=cfg.alpha, var_keep=cfg.var_keep, C=cfg.C,
                        min_angle_deg=cfg.min_principal_angle_deg,
                        cov_loading_margin_kappa=cfg.cov_loading_margin_kappa,
                        n_folds=cfg.source_cv_folds, group_ids=src_group_ids, seed=seed)
    base = CertifierConfig(tau_resid=cfg.tau_resid, tau_margin=cfg.tau_margin)
    cal = calibrate_thresholds(Z_src, Y_src, D_src, sa.atlas, base,
                               target_size=len(np.asarray(Z_tgt)),
                               block_ids_tr=src_group_ids, alpha=cfg.alpha,
                               n_block=cfg.tau_n_pseudotargets, quantile=cfg.tau_quantile,
                               seed=seed)
    cert = certify_robust(sa, Z_tgt, cfg=cal, n_boot=cfg.target_n_boot,
                          consensus=cfg.consensus, group_ids=tgt_group_ids, seed=seed)
    return dict(certificate=cert, analysis=sa, calibrated_cfg=cal,
                tau_detect=cal.tau_detect, tau_label=cal.tau_label)


def ood_power_bank(cfg: ProtocolConfig, seeds, n_domains: int = 8, concept_domains: int = 3,
                   min_visible: int = 5) -> dict:
    """OOD_POWER_BANK: deployment power on GENERATOR-TRUTH out-of-distribution targets, through
    the SAME frozen path. The FULL source is used (training retains its concept atlas), so a
    boundary_coupled target is a fair power test. Truth is the GENERATOR (we built it), not the
    oracle -- the oracle only sanity-checks the estimator in the null bank."""
    from .sim.shift_simulator import SimConfig, make_source, make_target
    recs = []
    for s in seeds:
        scfg = SimConfig(seed=s)
        src = make_source(scfg, n_domains=n_domains, concept_domains=concept_domains, seed=s)
        for kind, truth in [("covariate", "COVARIATE"), ("boundary_coupled", "CONCEPT_VISIBLE")]:
            tb = make_target(kind, scfg, geom=src.geom, seed=1000 + s)
            out = run_frozen_protocol(src.Z, src.Y, src.D, tb.Z, cfg, seed=s)
            recs.append(dict(seed=s, kind=kind, gen_truth=truth,
                             cert=out["certificate"].state,
                             concept_atlas=bool(out["analysis"].concept_evidenced)))
    vis = [r for r in recs if r["gen_truth"] == "CONCEPT_VISIBLE" and r["concept_atlas"]]
    cov = [r for r in recs if r["gen_truth"] == "COVARIATE"]

    def rate(rs, pred):
        return (sum(pred(r) for r in rs) / len(rs)) if rs else None

    return dict(bank="OOD_POWER_BANK", n_records=len(recs), n_fair_visible=len(vis),
                concept_power=rate(vis, lambda r: r["cert"] == CONCEPT_SUSPECT),
                covariate_compatible_coverage=rate(cov, lambda r: r["cert"] == COVARIATE_COMPATIBLE),
                false_concept_on_covariate=rate(cov, lambda r: r["cert"] == CONCEPT_SUSPECT),
                ood_power_bank_valid=(len(vis) >= min_visible),
                records=recs)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    from csc.sim.shift_simulator import SimConfig, make_source, make_target
    cfg = ProtocolConfig(n_boot=20, n_dir_boot=80, target_n_boot=80, tau_n_pseudotargets=120)
    print("manifest hash:", cfg.hash())
    src = make_source(SimConfig(seed=0), n_domains=8, concept_domains=3, seed=0)
    for kind in ("clean", "covariate", "boundary_coupled", "label_shift"):
        tb = make_target(kind, SimConfig(seed=0), geom=src.geom, seed=1)
        out = run_frozen_protocol(src.Z, src.Y, src.D, tb.Z, cfg, seed=0)
        print(f"  {kind:16s} -> {out['certificate'].state:20s} "
              f"(tau_detect={out['tau_detect']:.2f} tau_label={out['tau_label']:.2f})")
