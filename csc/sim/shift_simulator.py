"""
csc.sim.shift_simulator — controllable generator for the concept-shift taxonomy.

Produces a multi-domain *source* (Z, Y, D) with class-spanning domains, plus a *target*
(Z observed; Y kept only for oracle/eval — the certifier never reads it) under each
shift type in the taxonomy:

  clean              P(Z) ~ source,  P(Y|Z) ~ source                   false-alarm control
  covariate          P(Z) shifts along a NUISANCE direction,           -> COVARIATE_COMPATIBLE
                     P(Y|Z) invariant
  boundary_coupled   P(Y|Z) shifts WITH a visible marginal signature   -> CONCEPT_SUSPECT
  pure_conditional   P(Y|Z) shifts, Z statistically identical to clean  -> UNIDENTIFIABLE
                     (relabel-only; certifier sees no marginal signature)
  label_shift        P(Y) shifts, P(Z|Y) fixed (target shift)          -> UNIDENTIFIABLE
                     (moves the pooled mean along the class-mean subspace; NOT separable
                      from concept without an identifiable label-shift model)
  label_covariate_mixed   label shift + a covariate offset             -> UNIDENTIFIABLE
                     (confounded: the label component blocks attribution)

Generative model (the key design is an explicit nuisance/discriminative split):

    z = mu_y                      (class signal, in the DISCRIMINATIVE subspace, fixed)
      + c_d * s_y * w_concept     (per-domain boundary movement, also discriminative)
      + U_cov @ b_d               (per-domain covariate offset, in the NUISANCE subspace)
      + noise

The labels depend ONLY on the discriminative-subspace coordinates, so:
  * moving along U_cov changes P(Z) but NOT P(Y|Z)            -> covariate shift
  * moving class means along w_concept changes P(Y|Z)         -> concept shift
This separation is what lets the certifier decide, from unlabeled target Z alone,
*which kind* of shift it is looking at — and abstain when it cannot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class SimConfig:
    d: int = 12               # ambient dimension
    K: int = 3                # number of classes
    n_per_domain: int = 400
    sep: float = 2.5          # class separation in the discriminative subspace
    noise: float = 1.0
    cov_dim: int = 3          # dimension of the nuisance (covariate) subspace
    prior_alpha: float = 4.0  # Dirichlet conc. for per-domain label prior (label shift)
    min_per_class: int = 8    # guarantee class-spanning domains
    seed: int = 0


@dataclass
class GenGeom:
    """Fixed geometry shared by source and every target (so atlases are comparable)."""
    mu: np.ndarray            # (K, d) class means in the discriminative subspace
    U_cov: np.ndarray         # (d, cov_dim) orthonormal nuisance basis
    w_concept: np.ndarray     # (d,) unit concept direction (discriminative, != mu span)
    s_y: np.ndarray           # (K,) per-class concept sign (asymmetric -> visible signature)
    noise: float


@dataclass
class DomainSpec:
    b: np.ndarray             # (cov_dim,) covariate offset coords
    c: float                  # concept magnitude
    prior: np.ndarray         # (K,) label prior


@dataclass
class SourceBundle:
    Z: np.ndarray
    Y: np.ndarray
    D: np.ndarray
    geom: GenGeom
    domains: list             # list[DomainSpec]
    pooled_mean: np.ndarray   # overall mean of Z (deployment reference)
    group_ids: np.ndarray = None   # subject id per epoch (subjects nested within domains)


def _orthonormal(rng: np.random.Generator, d: int, k: int) -> np.ndarray:
    A = rng.standard_normal((d, k))
    Q, _ = np.linalg.qr(A)
    return Q[:, :k]


def make_geom(cfg: SimConfig, rng: np.random.Generator) -> GenGeom:
    # Discriminative subspace: first build an orthonormal frame, carve task + concept dirs
    frame = _orthonormal(rng, cfg.d, cfg.K + 1 + cfg.cov_dim)
    task_dirs = frame[:, : cfg.K]                       # class-mean directions
    w_concept = frame[:, cfg.K]                         # concept direction
    U_cov = frame[:, cfg.K + 1 : cfg.K + 1 + cfg.cov_dim]  # nuisance subspace (orthogonal)
    mu = cfg.sep * task_dirs.T                          # (K, d)
    s_y = (np.arange(cfg.K) - (cfg.K - 1) / 2.0)        # symmetric core...
    s_y = s_y + 0.6                                     # ...made asymmetric -> mean signature
    return GenGeom(mu=mu, U_cov=U_cov, w_concept=w_concept, s_y=s_y, noise=cfg.noise)


def _draw_labels(rng, prior, n, K, min_per_class) -> np.ndarray:
    y = rng.choice(K, size=n, p=prior)
    # enforce class-spanning (each class present at least min_per_class times)
    for k in range(K):
        have = int((y == k).sum())
        if have < min_per_class:
            idx = rng.choice(n, size=min_per_class - have, replace=False)
            y[idx] = k
    return y


def _sample_domain(cfg, geom, spec: DomainSpec, n, rng) -> tuple:
    y = _draw_labels(rng, spec.prior, n, cfg.K, cfg.min_per_class)
    z = geom.mu[y].copy()                                   # class signal
    z = z + spec.c * geom.s_y[y][:, None] * geom.w_concept  # concept (boundary) move
    z = z + (geom.U_cov @ spec.b)[None, :]                  # covariate offset (nuisance)
    z = z + geom.noise * rng.standard_normal(z.shape)       # noise
    return z, y


def make_source(cfg: Optional[SimConfig] = None,
                n_domains: int = 8,
                concept_domains: int = 3,
                cov_scale: float = 2.0,
                concept_scale: float = 1.4,
                subjects_per_domain: int = 5,
                geom: Optional[GenGeom] = None,
                seed: Optional[int] = None) -> SourceBundle:
    """Class-spanning multi-domain source. `concept_domains` of `n_domains` carry genuine
    boundary movement (the source concept *atlas*); the rest are covariate-only. Epochs are
    grouped into SUBJECTS nested within domains (`group_ids`) so cluster-aware inference can be
    exercised. (Synthetic epochs are i.i.d. within subject, so the clustering exercises the
    CODE PATH; genuine within-subject correlation only matters on real EEG.)"""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    geom = geom or make_geom(cfg, rng)

    domains = []
    concept_set = set(rng.choice(n_domains, size=min(concept_domains, n_domains),
                                 replace=False).tolist())
    for d in range(n_domains):
        b = cov_scale * rng.standard_normal(cfg.cov_dim)
        c = concept_scale * (1.0 + 0.3 * rng.standard_normal()) if d in concept_set else 0.0
        prior = rng.dirichlet(np.full(cfg.K, cfg.prior_alpha))
        domains.append(DomainSpec(b=b, c=float(c), prior=prior))

    Zs, Ys, Ds, Gs = [], [], [], []
    for d, spec in enumerate(domains):
        z, y = _sample_domain(cfg, geom, spec, cfg.n_per_domain, rng)
        Zs.append(z); Ys.append(y); Ds.append(np.full(len(y), d))
        # subjects nested within domain: global id = d*1000 + subject index
        subj = d * 1000 + (np.arange(len(y)) % subjects_per_domain)
        Gs.append(subj)
    Z = np.concatenate(Zs); Y = np.concatenate(Ys); D = np.concatenate(Ds)
    G = np.concatenate(Gs)
    return SourceBundle(Z=Z, Y=Y, D=D, geom=geom, domains=domains,
                        pooled_mean=Z.mean(0), group_ids=G)


@dataclass
class TargetBundle:
    Z: np.ndarray
    Y: np.ndarray             # ground-truth labels (oracle/eval ONLY; certifier ignores)
    kind: str
    truth: str                # ground-truth certificate class for scoring
    group_ids: np.ndarray = None   # subject id per epoch (target clusters)


_TRUTH = {
    "clean": "NONE",
    "covariate": "COVARIATE",
    "boundary_coupled": "CONCEPT_VISIBLE",
    "pure_conditional": "CONCEPT_INVISIBLE",
    "label_shift": "LABEL_SHIFT",
    "label_covariate_mixed": "LABEL_COVARIATE",
}


def _skewed_prior(K: int, peak: float = 0.8) -> np.ndarray:
    p = np.full(K, (1.0 - peak) / (K - 1))
    p[0] = peak
    return p


def make_target(kind: str,
                cfg: Optional[SimConfig] = None,
                geom: GenGeom = None,
                n: int = 1500,
                cov_target_scale: float = 8.0,
                concept_target_scale: float = 4.0,
                relabel_frac: float = 0.35,
                label_peak: float = 0.8,
                subjects: int = 8,
                seed: int = 123) -> TargetBundle:
    """Generate a target deployment batch under one shift type. Requires the SAME `geom`
    used for the source (pass `source.geom`). Epochs are grouped into `subjects` target
    clusters (`group_ids`) for cluster-aware certification."""
    cfg = cfg or SimConfig()
    assert geom is not None, "pass geom=source.geom"
    rng = np.random.default_rng(seed)
    if kind not in _TRUTH:
        raise ValueError(f"unknown shift kind {kind!r}; choose from {list(_TRUTH)}")
    unif = np.full(cfg.K, 1.0 / cfg.K)

    if kind == "clean":
        spec = DomainSpec(b=np.zeros(cfg.cov_dim), c=0.0, prior=unif)
        Z, Y = _sample_domain(cfg, geom, spec, n, rng)

    elif kind == "covariate":
        # move along a NUISANCE direction: P(Z) changes, P(Y|Z) invariant
        u = rng.standard_normal(cfg.cov_dim); u /= np.linalg.norm(u)
        spec = DomainSpec(b=cov_target_scale * u, c=0.0, prior=unif)
        Z, Y = _sample_domain(cfg, geom, spec, n, rng)

    elif kind == "boundary_coupled":
        # move class means along the concept direction: P(Y|Z) changes, visible signature
        spec = DomainSpec(b=np.zeros(cfg.cov_dim), c=concept_target_scale, prior=unif)
        Z, Y = _sample_domain(cfg, geom, spec, n, rng)

    elif kind == "pure_conditional":
        # generate CLEAN Z, then relabel within a near-boundary band: P(Z) identical,
        # P(Y|Z) changed. The certifier sees Z only -> indistinguishable from clean.
        spec = DomainSpec(b=np.zeros(cfg.cov_dim), c=0.0, prior=unif)
        Z, Y = _sample_domain(cfg, geom, spec, n, rng)
        Y = _relabel_invisible(Z, Y, geom, relabel_frac, rng)

    elif kind == "label_shift":
        # P(Y) skewed, P(Z|Y) unchanged: pooled mean moves along the class-mean subspace.
        # This is a TARGET shift, NOT a boundary change -- but it leaves a marginal trace
        # the certifier must NOT mistake for concept (it must abstain).
        spec = DomainSpec(b=np.zeros(cfg.cov_dim), c=0.0,
                          prior=_skewed_prior(cfg.K, label_peak))
        Z, Y = _sample_domain(cfg, geom, spec, n, rng)

    else:  # label_covariate_mixed
        u = rng.standard_normal(cfg.cov_dim); u /= np.linalg.norm(u)
        spec = DomainSpec(b=0.5 * cov_target_scale * u, c=0.0,
                          prior=_skewed_prior(cfg.K, label_peak))
        Z, Y = _sample_domain(cfg, geom, spec, n, rng)

    G = np.arange(len(Y)) % subjects                       # target subject clusters
    return TargetBundle(Z=Z, Y=Y, kind=kind, truth=_TRUTH[kind], group_ids=G)


def make_paired_clean_pure(cfg: Optional[SimConfig] = None,
                           geom: GenGeom = None,
                           n: int = 1500,
                           relabel_frac: float = 0.35,
                           seed: int = 123) -> tuple:
    """Return (clean, pure_conditional) targets that share BYTE-IDENTICAL Z and differ
    ONLY in labels. The certifier (Z-only) MUST return the same state for both -- this is
    the operational proof of the impossibility result (THEORY §1.1)."""
    cfg = cfg or SimConfig()
    assert geom is not None, "pass geom=source.geom"
    rng = np.random.default_rng(seed)
    spec = DomainSpec(b=np.zeros(cfg.cov_dim), c=0.0, prior=np.full(cfg.K, 1.0 / cfg.K))
    Z, Y = _sample_domain(cfg, geom, spec, n, rng)
    Y_relabel = _relabel_invisible(Z, Y, geom, relabel_frac, rng)
    G = np.arange(len(Y)) % 8
    clean = TargetBundle(Z=Z, Y=Y, kind="clean", truth=_TRUTH["clean"], group_ids=G)
    pure = TargetBundle(Z=Z.copy(), Y=Y_relabel, kind="pure_conditional",
                        truth=_TRUTH["pure_conditional"], group_ids=G.copy())
    return clean, pure


def _relabel_invisible(Z, Y, geom: GenGeom, frac, rng) -> np.ndarray:
    """Flip labels of points near the decision boundary to the next class (mod K). Z is
    untouched, so P(Z) is exactly preserved while P(Y|Z) changes."""
    K = geom.mu.shape[0]
    # distance to each class mean -> margin between best and 2nd-best
    d2 = ((Z[:, None, :] - geom.mu[None, :, :]) ** 2).sum(-1)  # (n, K)
    srt = np.sort(d2, axis=1)
    margin = srt[:, 1] - srt[:, 0]
    n_flip = int(frac * len(Y))
    near = np.argsort(margin)[:max(n_flip, 1)]
    Y = Y.copy()
    Y[near] = (Y[near] + 1) % K
    return Y


if __name__ == "__main__":
    cfg = SimConfig()
    src = make_source(cfg)
    print(f"source: Z{src.Z.shape} domains={len(src.domains)} "
          f"concept_doms={[i for i,s in enumerate(src.domains) if s.c!=0]}")
    for kind in _TRUTH:
        tb = make_target(kind, cfg, geom=src.geom)
        shift = np.linalg.norm(tb.Z.mean(0) - src.pooled_mean)
        print(f"  target {kind:18s} truth={tb.truth:18s} |Δmean|={shift:6.3f}")
