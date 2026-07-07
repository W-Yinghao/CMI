"""
csc.certificate.certifier — the three-state concept-shift certificate with abstention.

Given (a) the source-side residual test (is there *any* identifiable concept structure to
calibrate against?) and (b) the source shift atlas, decide, from UNLABELED target Z only,
one of three states:

  COVARIATE_ADAPTABLE   the target's marginal shift lies in the covariate (nuisance)
                        atlas and the source shows no domain-dependent boundary there:
                        adaptation is in-scope.
  CONCEPT_SUSPECT       the target's marginal shift aligns with directions where the
                        source DID exhibit boundary movement (and the residual test is
                        significant): a concept change left a visible signature.
  UNIDENTIFIABLE        the certificate ABSTAINS. Either there is no visible shift
                        (a pure conditional shift cannot be excluded -> the impossibility
                        result), the shift is in a direction the source never spanned
                        (out of the identifiable atlas), or the source support graph is
                        invalid (no concept atlas can be built).

The whole point: the certifier NEVER reads target labels and NEVER returns "safe" for a
shift it cannot see. A naive low-marginal-shift = safe rule would FALSE-CERTIFY a pure
conditional shift; this returns UNIDENTIFIABLE instead. That abstention is the product.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .atlas import ShiftAtlas
from .residual_test import ResidualTestResult


COVARIATE_ADAPTABLE = "COVARIATE_ADAPTABLE"
CONCEPT_SUSPECT = "CONCEPT_SUSPECT"
UNIDENTIFIABLE = "UNIDENTIFIABLE"


@dataclass
class CertifierConfig:
    tau_detect: float = 1.5       # shift must exceed this * source spread to count as visible
    tau_resid: float = 0.6        # if out-of-atlas component dominates by this margin -> abstain
    tau_margin: float = 1.15      # dominance margin between concept vs covariate components


@dataclass
class Certificate:
    state: str
    reason: str
    n_cov: float                  # covariate component, in units of source covariate spread
    n_concept: float              # concept component, in units of source concept spread
    n_resid: float                # out-of-atlas component, in covariate-spread units
    visible: bool
    source_significant: bool
    detail: dict


def _proj_norm(delta, basis) -> float:
    if basis.shape[1] == 0:
        return 0.0
    return float(np.linalg.norm(basis.T @ delta))


def certify(atlas: ShiftAtlas,
            source_test: ResidualTestResult,
            Z_target: np.ndarray,
            cfg: Optional[CertifierConfig] = None) -> Certificate:
    cfg = cfg or CertifierConfig()
    Z_target = np.asarray(Z_target, float)
    delta = Z_target.mean(0) - atlas.pooled_mean

    # --- decompose the observed marginal shift onto the atlas ---------------------------
    proj_cov = atlas.cov_dirs @ (atlas.cov_dirs.T @ delta) if atlas.cov_dirs.shape[1] else np.zeros_like(delta)
    proj_con = atlas.concept_dirs @ (atlas.concept_dirs.T @ delta) if atlas.concept_dirs.shape[1] else np.zeros_like(delta)
    resid = delta - proj_cov - proj_con

    c_cov = float(np.linalg.norm(proj_cov))
    c_con = float(np.linalg.norm(proj_con))
    c_res = float(np.linalg.norm(resid))

    sc = atlas.sigma_cov if atlas.sigma_cov > 1e-8 else 1.0
    scon = atlas.sigma_concept if atlas.sigma_concept > 1e-8 else 1.0
    n_cov, n_con, n_res = c_cov / sc, c_con / scon, c_res / sc
    detail = dict(delta_norm=float(np.linalg.norm(delta)),
                  c_cov=c_cov, c_concept=c_con, c_resid=c_res,
                  sigma_cov=atlas.sigma_cov, sigma_concept=atlas.sigma_concept)

    # --- (0) no valid concept atlas: cannot calibrate -> abstain ------------------------
    if source_test.status != "VALID":
        return Certificate(UNIDENTIFIABLE,
                           "source support graph invalid -> no concept atlas: "
                           + "; ".join(source_test.support.reasons),
                           n_cov, n_con, n_res, False, False, detail)

    sig = source_test.significant

    # --- (1) no visible marginal shift: pure conditional shift cannot be excluded -------
    visible = max(n_cov, n_con, n_res) >= cfg.tau_detect
    if not visible:
        return Certificate(UNIDENTIFIABLE,
                           "no marginal signature above source between-domain spread; "
                           "a pure conditional (invisible) shift cannot be excluded",
                           n_cov, n_con, n_res, False, sig, detail)

    # --- (2) shift in a direction the source never spanned: out of identifiable range ---
    if n_res >= cfg.tau_resid * max(n_cov, n_con) and n_res >= cfg.tau_detect:
        return Certificate(UNIDENTIFIABLE,
                           "marginal shift lies outside the source atlas (novel direction); "
                           "identifiability of its label effect is not established",
                           n_cov, n_con, n_res, True, sig, detail)

    # --- (3) classify the visible, in-atlas shift --------------------------------------
    if n_con >= cfg.tau_margin * n_cov:
        if sig:
            return Certificate(CONCEPT_SUSPECT,
                               "marginal shift aligns with source concept directions and "
                               "the residual test is significant (T>0)",
                               n_cov, n_con, n_res, True, sig, detail)
        return Certificate(UNIDENTIFIABLE,
                           "shift aligns with concept directions but the source residual "
                           "test is not significant -> concept atlas not trustworthy",
                           n_cov, n_con, n_res, True, sig, detail)

    if n_cov >= cfg.tau_margin * n_con:
        return Certificate(COVARIATE_ADAPTABLE,
                           "marginal shift lies in the covariate (nuisance) atlas where the "
                           "source boundary did not move",
                           n_cov, n_con, n_res, True, sig, detail)

    # ambiguous mix of covariate and concept components -> abstain
    return Certificate(UNIDENTIFIABLE,
                       "shift mixes covariate and concept directions without a clear "
                       "dominant component -> cannot attribute the marginal change",
                       n_cov, n_con, n_res, True, sig, detail)


# scoring: map ground-truth shift class -> the acceptable certificate(s) --------------
ACCEPTABLE = {
    "NONE": {UNIDENTIFIABLE, COVARIATE_ADAPTABLE},   # clean: abstain or "safe", never CONCEPT_SUSPECT
    "COVARIATE": {COVARIATE_ADAPTABLE},
    "CONCEPT_VISIBLE": {CONCEPT_SUSPECT},
    "CONCEPT_INVISIBLE": {UNIDENTIFIABLE},           # the false-certification guard
}

# the one outcome that must NEVER happen per truth class (a *false certification*)
FORBIDDEN = {
    "NONE": {CONCEPT_SUSPECT},
    "COVARIATE": {CONCEPT_SUSPECT},                  # crying wolf on a benign covariate shift
    "CONCEPT_VISIBLE": {COVARIATE_ADAPTABLE},        # certifying a real concept shift as safe
    "CONCEPT_INVISIBLE": {COVARIATE_ADAPTABLE, CONCEPT_SUSPECT},  # MUST abstain
}


if __name__ == "__main__":
    from csc.sim.shift_simulator import SimConfig, make_source, make_target, _TRUTH
    from .atlas import build_atlas
    from .residual_test import residual_decoder_test
    cfg = SimConfig(seed=3)
    src = make_source(cfg, n_domains=8, concept_domains=3)
    atl = build_atlas(src.Z, src.Y, src.D)
    rt = residual_decoder_test(src.Z, src.Y, src.D, n_perm=60)
    print(f"source residual test: T={rt.T:+.3f} p={rt.p_value:.3f} sig={rt.significant}")
    for kind in _TRUTH:
        tb = make_target(kind, cfg, geom=src.geom)
        cert = certify(atl, rt, tb.Z)
        print(f"  {kind:18s} truth={tb.truth:18s} -> {cert.state:20s} "
              f"(cov={cert.n_cov:.2f} con={cert.n_concept:.2f} res={cert.n_resid:.2f})")
