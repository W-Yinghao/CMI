"""Controllable EEG mechanism simulator with a hierarchical site->subject->session DAG.

Generative model (per trial i with latent state y* and session/site domain d):

  sources   s_k(t) = a_k(t)   band-limited oscillation, k = 1..n_sources
            class-discriminative sources have amplitude modulated by the class via a
            site-specific class->power map  mu_{y, site}  (concept shift = a per-site
            rotation of this map -> the source->label rule differs across sites).
  channels  x(t) = M_d s(t) + e(t)
            M_d = M0 + cov * (dM_site + dM_subject)         (covariance shift)
                  scaled per-channel by a montage gain g_site  (montage shift)
            e(t) ~ session-dependent pink-ish noise           (noise shift)

Orthogonal knobs (``ShiftSpec``):
  cov      magnitude of the per-site/-subject spatial-mixing perturbation (the
           CORRECTABLE shift -- SPD/covariance alignment should help here).
  prior    per-site label-prior skew  p_site(y)              (label-shift).
  concept  rotation angle of the per-site class->source-power map for the
           ``concept_site_frac`` fraction of "concept-shifted" sites (the shift that
           pooled covariance alignment CANNOT fix -- adaptation should ABSTAIN here).
  montage  per-site channel-gain / partial-dropout magnitude  (device shift).
  noise    session noise scale.
  label_mechanism_rho  site-dependent corruption of the OBSERVED label given the true
           state y* (review section 8): y != y* at noisy sites.

Everything is numpy float32; the hierarchical structure is a real ``DomainDAG`` so the
hierarchical CMI decomposition has a ground-truth nesting to operate on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from h2cmi.domains import DomainDAG, DomainLabels
from h2cmi.label.site_mechanism import SiteLabelMechanism


@dataclass
class ShiftSpec:
    cov: float = 1.0
    prior: float = 0.0
    concept: float = 0.0                 # rotation angle (radians) at concept-shifted sites
    concept_site_frac: float = 0.0       # fraction of sites that are concept-shifted
    montage: float = 0.0
    noise: float = 0.3
    label_mechanism_rho: float = 0.0     # site label-corruption strength in [0,1]


@dataclass
class SimulatedEEG:
    X: np.ndarray                        # [N, n_chans, n_times] float32
    y: np.ndarray                        # [N] OBSERVED labels (== ystar unless label mech)
    ystar: np.ndarray                    # [N] true latent state
    domains: DomainLabels                # hierarchical site/subject/session levels
    dag: DomainDAG
    n_classes: int
    fs: float
    meta: dict = field(default_factory=dict)   # oracle per-domain generative params

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def site(self) -> np.ndarray:
        return self.domains.factor("site")

    @property
    def subject(self) -> np.ndarray:
        return self.domains.factor("subject")

    @property
    def session(self) -> np.ndarray:
        return self.domains.factor("session")


class EEGSimulator:
    """Sample hierarchical, mechanism-controlled synthetic EEG."""

    def __init__(self, n_classes: int = 2, n_chans: int = 16, n_times: int = 256,
                 fs: float = 128.0, n_sources: int = 8, n_disc: int | None = None,
                 bands: tuple[tuple[float, float], ...] = ((4, 8), (8, 13), (13, 30), (30, 45)),
                 shift: ShiftSpec | None = None, seed: int = 0):
        self.n_classes = int(n_classes)
        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.fs = float(fs)
        self.n_sources = int(n_sources)
        self.n_disc = int(n_disc) if n_disc is not None else min(n_sources, max(2, n_classes + 2))
        self.bands = bands
        self.shift = shift or ShiftSpec()
        self.seed = int(seed)
        rng = np.random.default_rng(seed)

        # canonical spatial mixing (shared) and per-source centre frequency
        self.M0 = rng.standard_normal((self.n_chans, self.n_sources)).astype(np.float32)
        self.M0 /= np.linalg.norm(self.M0, axis=0, keepdims=True) + 1e-8
        self.src_freq = np.array(
            [rng.uniform(*bands[k % len(bands)]) for k in range(self.n_sources)], dtype=np.float32)

        # canonical class -> discriminative-source-power map (rows=class, cols=disc source)
        base = rng.standard_normal((self.n_classes, self.n_disc)).astype(np.float32)
        # make classes well separated: orthogonalise-ish by adding a class-indexed bump
        for c in range(self.n_classes):
            base[c, c % self.n_disc] += 2.5
        self.class_power = base

    # -- helpers -----------------------------------------------------------------
    def _rotation(self, rng, angle):
        """A rotation acting on the discriminative-source subspace (concept shift)."""
        R = np.eye(self.n_disc, dtype=np.float32)
        if angle == 0 or self.n_disc < 2:
            return R
        # compose a few Givens rotations to get a generic rotation of magnitude ~angle
        for _ in range(self.n_disc // 2):
            i, j = rng.choice(self.n_disc, size=2, replace=False)
            c, s = np.cos(angle), np.sin(angle)
            G = np.eye(self.n_disc, dtype=np.float32)
            G[i, i] = c; G[j, j] = c; G[i, j] = -s; G[j, i] = s
            R = R @ G
        return R

    def _site_prior(self, rng):
        """Per-site class prior with magnitude controlled by ShiftSpec.prior."""
        if self.shift.prior <= 0:
            return np.full(self.n_classes, 1.0 / self.n_classes, dtype=np.float64)
        logits = self.shift.prior * rng.standard_normal(self.n_classes)
        p = np.exp(logits - logits.max())
        return (p / p.sum()).astype(np.float64)

    def _gen_sources(self, rng, power, n_times):
        """Generate [n_trials, n_sources, n_times] band-limited source signals.

        ``power`` is [n_trials, n_sources] non-negative amplitude scaling.
        Each source is a sinusoid at its centre freq (random phase) + small broadband
        wiggle, so band power in the assigned band encodes the (class-driven) amplitude.
        """
        n_tr = power.shape[0]
        t = np.arange(n_times, dtype=np.float32) / self.fs
        phase = rng.uniform(0, 2 * np.pi, size=(n_tr, self.n_sources, 1)).astype(np.float32)
        w = (2 * np.pi * self.src_freq).reshape(1, self.n_sources, 1)
        osc = np.sin(w * t.reshape(1, 1, n_times) + phase)
        wiggle = 0.15 * rng.standard_normal((n_tr, self.n_sources, n_times)).astype(np.float32)
        amp = np.sqrt(np.clip(power, 1e-4, None)).astype(np.float32)[:, :, None]
        return amp * osc + wiggle

    # -- main sampler ------------------------------------------------------------
    def sample(self, n_sites: int = 4, subjects_per_site: int = 3,
               sessions_per_subject: int = 2, trials_per_session: int = 40,
               *, subject_determines_label: bool = False) -> SimulatedEEG:
        rng = np.random.default_rng(self.seed + 1)
        dag = DomainDAG.hierarchical_site_subject_session(
            n_sites, subjects_per_site, sessions_per_subject,
            subject_determines_label=subject_determines_label)

        n_subjects = n_sites * subjects_per_site
        n_sessions = n_subjects * sessions_per_subject

        # per-site generative params
        n_concept_sites = int(round(self.shift.concept_site_frac * n_sites))
        concept_sites = set(rng.choice(n_sites, size=n_concept_sites, replace=False).tolist()) \
            if n_concept_sites > 0 else set()
        site_prior = [self._site_prior(rng) for _ in range(n_sites)]
        site_rot = [self._rotation(rng, self.shift.concept) if s in concept_sites
                    else np.eye(self.n_disc, dtype=np.float32) for s in range(n_sites)]
        site_dM = [self.shift.cov * 0.6 * rng.standard_normal((self.n_chans, self.n_sources)).astype(np.float32)
                   for _ in range(n_sites)]
        site_gain = []
        for _ in range(n_sites):
            g = np.ones(self.n_chans, dtype=np.float32)
            if self.shift.montage > 0:
                g = g * (1.0 + self.shift.montage * rng.standard_normal(self.n_chans).astype(np.float32))
                drop = rng.random(self.n_chans) < (0.15 * self.shift.montage)
                g[drop] *= 0.1                                   # partial electrode dropout
            site_gain.append(g)

        # optional site-dependent label mechanism
        label_mech = None
        if self.shift.label_mechanism_rho > 0:
            label_mech = SiteLabelMechanism(self.n_classes, n_sites, rho=self.shift.label_mechanism_rho)
            C_global = 0.85 * np.eye(self.n_classes) + 0.15 / self.n_classes
            C_global /= C_global.sum(1, keepdims=True)
            C_local = np.stack([rng.dirichlet(np.ones(self.n_classes), size=self.n_classes)
                                for _ in range(n_sites)])
            label_mech.set_matrices(C_global, C_local)

        # per-subject mixing perturbation (nested in site)
        subj_dM = [self.shift.cov * 0.3 * rng.standard_normal((self.n_chans, self.n_sources)).astype(np.float32)
                   for _ in range(n_subjects)]

        Xs, ystar_all, yobs_all, lvl_site, lvl_subj, lvl_sess = [], [], [], [], [], []
        sess_id = 0
        for site in range(n_sites):
            for sj in range(subjects_per_site):
                subject = site * subjects_per_site + sj
                for ss in range(sessions_per_subject):
                    n_tr = trials_per_session
                    # latent true state from the site prior (or fixed per subject for SCPS)
                    if subject_determines_label:
                        ystar = np.full(n_tr, subject % self.n_classes, dtype=np.int64)
                    else:
                        ystar = rng.choice(self.n_classes, size=n_tr, p=site_prior[site]).astype(np.int64)
                    # class -> disc-source power, with per-site concept rotation
                    cp = self.class_power[ystar]                              # [n_tr, n_disc]
                    cp = cp @ site_rot[site].T                                # concept shift
                    power = np.full((n_tr, self.n_sources), 0.5, dtype=np.float32)
                    power[:, :self.n_disc] = np.clip(0.5 + cp, 1e-3, None)
                    src = self._gen_sources(rng, power, self.n_times)         # [n_tr, n_src, T]

                    M_d = self.M0 + site_dM[site] + subj_dM[subject]          # covariance shift
                    x = np.einsum("cs,nst->nct", M_d, src)                    # [n_tr, n_chans, T]
                    x = x * site_gain[site][None, :, None]                    # montage gain
                    noise_scale = self.shift.noise * rng.uniform(0.5, 1.5)
                    x = x + noise_scale * rng.standard_normal(x.shape).astype(np.float32)
                    # per-trial z-score per channel (matches the AAAI loaders' convention)
                    x = (x - x.mean(2, keepdims=True)) / (x.std(2, keepdims=True) + 1e-6)

                    if label_mech is not None:
                        yobs = label_mech.corrupt(ystar, np.full(n_tr, site), rng)
                    else:
                        yobs = ystar.copy()

                    Xs.append(x.astype(np.float32))
                    ystar_all.append(ystar); yobs_all.append(yobs)
                    lvl_site.append(np.full(n_tr, site)); lvl_subj.append(np.full(n_tr, subject))
                    lvl_sess.append(np.full(n_tr, sess_id))
                    sess_id += 1

        X = np.concatenate(Xs, 0)
        ystar = np.concatenate(ystar_all, 0)
        yobs = np.concatenate(yobs_all, 0)
        levels = np.stack([np.concatenate(lvl_site), np.concatenate(lvl_subj),
                           np.concatenate(lvl_sess)], axis=1)
        domains = DomainLabels(dag, levels)
        meta = dict(concept_sites=sorted(concept_sites), site_prior=site_prior,
                    n_sessions=n_sessions, label_mechanism=label_mech is not None,
                    shift=self.shift)
        return SimulatedEEG(X, yobs, ystar, domains, dag, self.n_classes, self.fs, meta)


def train_target_split(sim: SimulatedEEG, n_target_sites: int = 1, seed: int = 0):
    """Split a SimulatedEEG into source (training) and held-out TARGET sites (unseen DG).

    Returns (src_idx, tgt_idx) over trials.  Target sites are entirely unseen during
    training -- the strict-DG / TTA evaluation unit is the site.
    """
    rng = np.random.default_rng(seed)
    sites = np.unique(sim.site)
    tgt_sites = set(rng.choice(sites, size=min(n_target_sites, len(sites) - 1), replace=False).tolist())
    tgt_mask = np.isin(sim.site, list(tgt_sites))
    return np.where(~tgt_mask)[0], np.where(tgt_mask)[0]


if __name__ == "__main__":
    sim = EEGSimulator(n_classes=3, n_chans=16, n_times=256,
                       shift=ShiftSpec(cov=1.0, prior=0.5, concept=0.4, concept_site_frac=0.5,
                                       montage=0.3, noise=0.3, label_mechanism_rho=0.2),
                       seed=1).sample(n_sites=4, subjects_per_site=3, sessions_per_subject=2,
                                      trials_per_session=30)
    print("X", sim.X.shape, "y", sim.y.shape, "classes", np.bincount(sim.y))
    print("DAG:", sim.dag)
    print("sites", np.unique(sim.site), "subjects", len(np.unique(sim.subject)),
          "sessions", len(np.unique(sim.session)))
    print("concept sites:", sim.meta["concept_sites"])
    print("label flips (y!=ystar):", int((sim.y != sim.ystar).sum()), "/", sim.n)
    s, t = train_target_split(sim, 1, seed=0)
    print("src/tgt trials:", len(s), len(t), "finite:", bool(np.isfinite(sim.X).all()))
