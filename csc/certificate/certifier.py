"""
csc.certificate.certifier — the three-state concept-shift certificate with abstention
(CSC-P0 rewrite).

From a SourceAnalysis (atlas + residual evidence) and UNLABELED target Z, return one of:

  COVARIATE_COMPATIBLE   visible shift that lies in the covariate (nuisance) atlas, with no
                         label-shift signature and no boundary movement there. NOTE: this is
                         a *compatibility* statement (the shift is of a kind the source
                         showed leaves the boundary fixed), NOT a guarantee that any specific
                         adaptation lowers risk -- that would require naming the operator and
                         bounding R_T(A(h)) - R_T(h) (see THEORY §5 / PREREGISTRATION).
  CONCEPT_SUSPECT        visible shift aligned with DIRECTION-LINKED concept evidence
                         (a source boundary actually moved along that direction).
  UNIDENTIFIABLE         abstain: no visible shift (pure conditional not excludable), a
                         label-shift signature (not separable without a label-shift model),
                         an out-of-atlas direction, an ambiguous mix, or no valid atlas.

Hard guarantees by construction:
  * the certifier never reads target labels;
  * it never emits a positive verdict for a shift it cannot see (clean / pure-conditional
    -> UNIDENTIFIABLE);
  * a pure label shift -> UNIDENTIFIABLE (it is not concept, and not separable here).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .atlas import SourceAnalysis


COVARIATE_COMPATIBLE = "COVARIATE_COMPATIBLE"
CONCEPT_SUSPECT = "CONCEPT_SUSPECT"
UNIDENTIFIABLE = "UNIDENTIFIABLE"


@dataclass
class CertifierConfig:
    tau_detect: float = 1.5       # shift must exceed this * source spread to count as visible
    tau_label: float = 1.0        # label-shift signature beyond this * source label spread -> abstain
    tau_resid: float = 0.6        # out-of-atlas dominance margin -> abstain
    tau_margin: float = 1.15      # dominance margin between concept and covariate


@dataclass
class Certificate:
    state: str
    reason: str
    n_label: float
    n_cov: float
    n_concept: float
    n_resid: float
    visible: bool
    concept_evidenced: bool
    detail: dict


def _proj(delta, basis):
    if basis.shape[1] == 0:
        return np.zeros_like(delta)
    return basis @ (basis.T @ delta)


def certify(analysis: SourceAnalysis,
            Z_target: np.ndarray,
            cfg: Optional[CertifierConfig] = None) -> Certificate:
    cfg = cfg or CertifierConfig()
    atlas = analysis.atlas
    Z_target = np.asarray(Z_target, float)
    delta = Z_target.mean(0) - atlas.pooled_mean

    p_lab = _proj(delta, atlas.label_dirs)
    p_cov = _proj(delta, atlas.cov_dirs)
    p_con = _proj(delta, atlas.concept_dirs)
    resid = delta - p_lab - p_cov - p_con

    c_lab, c_cov, c_con, c_res = (float(np.linalg.norm(v)) for v in (p_lab, p_cov, p_con, resid))
    s_lab = atlas.sigma_label if atlas.sigma_label > 1e-8 else 1.0
    s_cov = atlas.sigma_cov if atlas.sigma_cov > 1e-8 else 1.0
    s_con = atlas.sigma_concept if atlas.sigma_concept > 1e-8 else 1.0
    n_lab, n_cov, n_con, n_res = c_lab / s_lab, c_cov / s_cov, c_con / s_con, c_res / s_cov

    detail = dict(delta_norm=float(np.linalg.norm(delta)),
                  c_label=c_lab, c_cov=c_cov, c_concept=c_con, c_resid=c_res,
                  sigma_label=atlas.sigma_label, sigma_cov=atlas.sigma_cov,
                  sigma_concept=atlas.sigma_concept)
    ev = analysis.concept_evidenced

    def cert(state, reason, visible):
        return Certificate(state, reason, n_lab, n_cov, n_con, n_res, visible, ev, detail)

    # (0) no valid concept atlas -> abstain
    if analysis.test.status != "VALID":
        return cert(UNIDENTIFIABLE,
                    "source support graph invalid -> no concept atlas: "
                    + "; ".join(analysis.test.support.reasons), False)

    # (1) label-shift signature -> abstain (not concept; not separable without a label model)
    if n_lab >= cfg.tau_label:
        return cert(UNIDENTIFIABLE,
                    "marginal shift carries a LABEL-shift signature (moves along the "
                    "class-mean subspace); not separable from concept without an "
                    "identifiable label-shift model", True)

    # (2) no visible shift -> abstain (pure conditional / clean cannot be excluded)
    visible = max(n_cov, n_con, n_res) >= cfg.tau_detect
    if not visible:
        return cert(UNIDENTIFIABLE,
                    "no marginal signature above the source between-domain spread; a pure "
                    "conditional (invisible) shift cannot be excluded", False)

    # (3) out-of-atlas direction -> abstain
    if n_res >= cfg.tau_resid * max(n_cov, n_con) and n_res >= cfg.tau_detect:
        return cert(UNIDENTIFIABLE,
                    "marginal shift lies outside the source atlas (novel direction); its "
                    "label effect is not identifiable from the source", True)

    # (4) classify the visible, in-atlas, label-free shift
    if n_con >= cfg.tau_margin * n_cov:
        if ev:
            return cert(CONCEPT_SUSPECT,
                        "shift aligns with a direction carrying significant boundary "
                        "evidence (direction-linked, not global)", True)
        return cert(UNIDENTIFIABLE,
                    "shift aligns with concept directions but none carry significant "
                    "boundary evidence -> concept claim not supported", True)

    if n_cov >= cfg.tau_margin * n_con:
        return cert(COVARIATE_COMPATIBLE,
                    "shift lies in the covariate (nuisance) atlas where the source boundary "
                    "did not move", True)

    return cert(UNIDENTIFIABLE,
                "shift mixes covariate and concept components without a dominant one -> "
                "cannot attribute the marginal change", True)


def certify_robust(analysis: SourceAnalysis,
                   Z_target: np.ndarray,
                   cfg: Optional[CertifierConfig] = None,
                   n_boot: int = 200,
                   consensus: float = 0.9,
                   seed: int = 0) -> Certificate:
    """Confidence-region decision (CSC-P1). Instead of trusting a single target-mean point,
    block-bootstrap the target rows, certify each replicate, and emit a DEFINITE state only
    if a `consensus` fraction of the replicates agree on it (the bootstrap region C_T maps
    to a single atlas attribution, i.e. Gamma_T is a singleton). Otherwise abstain.

    This makes COVARIATE_COMPATIBLE / CONCEPT_SUSPECT robust to finite-target sampling: a
    shift whose attribution flips under resampling is, honestly, UNIDENTIFIABLE."""
    cfg = cfg or CertifierConfig()
    base = certify(analysis, Z_target, cfg)
    Z = np.asarray(Z_target, float)
    rng = np.random.default_rng(seed)
    counts = {}
    for _ in range(n_boot):
        bs = rng.integers(0, len(Z), len(Z))
        st = certify(analysis, Z[bs], cfg).state
        counts[st] = counts.get(st, 0) + 1
    top = max(counts, key=counts.get)
    frac = counts[top] / n_boot
    base.detail["consensus"] = frac
    base.detail["state_counts"] = counts
    if base.state != UNIDENTIFIABLE and base.state == top and frac >= consensus:
        return base
    if base.state == UNIDENTIFIABLE:
        return base
    return Certificate(UNIDENTIFIABLE,
                       f"bootstrap confidence region is not a single definite attribution "
                       f"(top={top}:{frac:.2f} < consensus {consensus}); abstain",
                       base.n_label, base.n_cov, base.n_concept, base.n_resid,
                       base.visible, base.concept_evidenced, base.detail)


# scoring: ground-truth shift class -> acceptable / forbidden certificate(s) ------------
ACCEPTABLE = {
    "NONE": {UNIDENTIFIABLE},                       # clean: MUST abstain (impossibility result)
    "COVARIATE": {COVARIATE_COMPATIBLE},
    "CONCEPT_VISIBLE": {CONCEPT_SUSPECT},
    "CONCEPT_INVISIBLE": {UNIDENTIFIABLE},
    "LABEL_SHIFT": {UNIDENTIFIABLE},
    "LABEL_COVARIATE": {UNIDENTIFIABLE},
}

# outcomes that must NEVER happen per truth class (a *false certification*)
FORBIDDEN = {
    "NONE": {COVARIATE_COMPATIBLE, CONCEPT_SUSPECT},
    "COVARIATE": {CONCEPT_SUSPECT},
    "CONCEPT_VISIBLE": {COVARIATE_COMPATIBLE},
    "CONCEPT_INVISIBLE": {COVARIATE_COMPATIBLE, CONCEPT_SUSPECT},
    "LABEL_SHIFT": {COVARIATE_COMPATIBLE, CONCEPT_SUSPECT},
    "LABEL_COVARIATE": {COVARIATE_COMPATIBLE, CONCEPT_SUSPECT},
}


if __name__ == "__main__":
    from csc.sim.shift_simulator import SimConfig, make_source, make_target, _TRUTH
    from .atlas import analyze_source
    cfg = SimConfig(seed=3)
    src = make_source(cfg, n_domains=8, concept_domains=3)
    sa = analyze_source(src.Z, src.Y, src.D, n_boot=60, n_dir_boot=120)
    print(f"source: T={sa.test.T:+.3f} p={sa.test.p_value:.3f} "
          f"concept_evidenced={sa.concept_evidenced}")
    for kind in _TRUTH:
        tb = make_target(kind, cfg, geom=src.geom)
        c = certify(sa, tb.Z)
        print(f"  {kind:22s} truth={tb.truth:16s} -> {c.state:20s} "
              f"(lab={c.n_label:.2f} cov={c.n_cov:.2f} con={c.n_concept:.2f} res={c.n_resid:.2f})")
