"""
CSC Route B3 — paired (within-subject ON/OFF) TARGET simulator. DEVELOPMENT only, self-contained in
`csc/mininfo/` (does not modify the shared simulator or touch the frozen A tag). Reuses only the
generative primitives `make_geom` / `_draw_labels` / `_relabel_invisible`.

Each biological subject is measured under TWO conditions A=0, B=1, sharing the subject random effect
(`eps_s`), with UNEQUAL epochs per condition. The kinds span the B3 control/positive zoo:

  clean                : A == B, no condition difference                          (truth NO_CONCEPT)
  paired_covariate     : B has a class-INDEPENDENT nuisance offset, P(Y|Z) fixed  (truth NO_CONCEPT)
  paired_label         : B has a skewed class prior, P(Y|Z) per-z unchanged       (truth NO_CONCEPT)
  paired_concept       : B's class-conditional boundary moves along w_concept      (truth CONCEPT)
  paired_pure_conditional : P(Z_B)==P(Z_A) but near-boundary B labels flipped      (truth CONCEPT)
  paired_concept_plus_cov : B has BOTH boundary move AND nuisance offset           (truth CONCEPT)
  random_label         : labels randomized in BOTH conditions (degenerate control) (truth NO_CONCEPT)

`missing_frac` drops one condition for a fraction of subjects (→ unpaired subjects the certifier must
detect as invalid pair structure).
"""
from __future__ import annotations

import numpy as np

from csc.sim.shift_simulator import SimConfig, make_geom, _draw_labels, _relabel_invisible

PAIRED_TRUTH = {
    "clean": "NO_CONCEPT", "paired_covariate": "NO_CONCEPT", "paired_label": "NO_CONCEPT",
    "random_label": "NO_CONCEPT", "paired_covariate_plus_label": "NO_CONCEPT",
    "missing_pair": "NO_CONCEPT", "unequal_epochs_extreme": "NO_CONCEPT",
    "paired_concept": "CONCEPT", "paired_pure_conditional": "CONCEPT",
    "paired_concept_plus_cov": "CONCEPT",
}


def _skewed(K, peak=0.8):
    p = np.full(K, (1.0 - peak) / (K - 1)); p[0] = peak
    return p


def make_paired_target(kind, geom, cfg: SimConfig = None, n_subjects: int = 30,
                       concept_scale: float = 14.0, cov_scale: float = 10.0, label_peak: float = 0.8,
                       missing_frac: float = 0.0, base_prior=None, label_noise: float = 0.0,
                       pure_cond_frac: float = 0.35, seed: int = 0):
    """Return (Z, Y, D, G, truth): D = condition (0/1), G = biological subject id (shared across
    conditions). Y is oracle (the certifier may only query a few subjects' labels).

    B3-P2.3 stress knobs: base_prior (CLASS IMBALANCE applied to BOTH conditions, != label shift which
    moves only B); label_noise (symmetric label flips in BOTH conditions = nuisance, not concept);
    pure_cond_frac (invisible-relabel strength). Structural-control kinds: missing_pair (forces a high
    unpaired fraction), unequal_epochs_extreme (condition A few epochs, B many)."""
    if kind not in PAIRED_TRUTH:
        raise ValueError(f"unknown paired kind {kind!r}; choose {list(PAIRED_TRUTH)}")
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(seed)
    K, d, cov_dim = cfg.K, cfg.d, cfg.cov_dim
    u_cov = np.zeros(cov_dim); u_cov[0] = 1.0
    base = np.full(K, 1.0 / K) if base_prior is None else np.asarray(base_prior, float)
    # per-condition generative knobs; both start at the (possibly imbalanced) baseline prior
    bA, cA, priorA = np.zeros(cov_dim), 0.0, base
    bB, cB, priorB = np.zeros(cov_dim), 0.0, base
    if kind == "paired_covariate":
        bB = cov_scale * u_cov
    elif kind == "paired_label":
        priorB = _skewed(K, label_peak)                       # B differs from A -> label shift
    elif kind == "paired_covariate_plus_label":
        bB = cov_scale * u_cov; priorB = _skewed(K, label_peak)
    elif kind == "paired_concept":
        cB = concept_scale
    elif kind == "paired_concept_plus_cov":
        cB = concept_scale; bB = cov_scale * u_cov
    # clean / paired_pure_conditional / random_label / missing_pair / unequal_epochs_extreme: A==B base
    miss = max(missing_frac, 0.4) if kind == "missing_pair" else missing_frac
    uneq = (kind == "unequal_epochs_extreme")

    Zs, Ys, Ds, Gs = [], [], [], []
    for s in range(n_subjects):
        eps_s = cfg.subject_tau * rng.standard_normal(d)             # shared across conditions
        conds = [0, 1]
        if miss and rng.random() < miss:                            # drop one condition -> unpaired
            conds = [int(rng.integers(0, 2))]
        for c in conds:
            b, cc, prior = (bA, cA, priorA) if c == 0 else (bB, cB, priorB)
            if uneq:                                                # extreme per-condition epoch imbalance
                n_e = int(rng.integers(4, 7)) if c == 0 else int(rng.integers(25, 36))
            else:
                n_e = int(rng.integers(cfg.epochs_min, cfg.epochs_max + 1))
            y = _draw_labels(rng, prior, n_e, K, min(2, n_e))
            z = geom.mu[y].copy()
            z = z + cc * geom.s_y[y][:, None] * geom.w_concept[None, :]   # condition boundary move
            z = z + (geom.U_cov @ b)[None, :]                            # condition nuisance offset
            z = z + eps_s[None, :] + geom.noise * rng.standard_normal((n_e, d))
            if kind == "paired_pure_conditional" and c == 1:            # invisible conditional flip in B
                y = _relabel_invisible(z, y, geom, pure_cond_frac, rng)
            if kind == "random_label":
                y = rng.integers(0, K, size=n_e)
            elif label_noise > 0:                                       # symmetric nuisance label noise
                flip = rng.random(n_e) < label_noise
                y = y.copy()
                y[flip] = (y[flip] + rng.integers(1, K, size=int(flip.sum()))) % K
            Zs.append(z); Ys.append(y); Ds.append(np.full(n_e, c)); Gs.append(np.full(n_e, s))
    return (np.concatenate(Zs), np.concatenate(Ys), np.concatenate(Ds), np.concatenate(Gs),
            PAIRED_TRUTH[kind])
