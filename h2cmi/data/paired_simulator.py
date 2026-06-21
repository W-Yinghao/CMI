"""Paired, mechanism-orthogonal EEG simulator for the shift-grid (review §"模拟器修改").

Requirements this design satisfies:

1. A scenario shift is applied ONLY to the held-out ``target_site``; every source site uses
   canonical (shared) site-level parameters.
2. For a fixed ``(data_seed, target_site)`` the SOURCE data is element-wise identical across
   ALL scenarios (so one Source-A / Source-B pair is trained once and reused for M0..M3).
3. Every mechanism (labels, oscillation phase, mixing, sensor noise, prior, the target
   cov/concept/montage knobs) draws from an INDEPENDENT RNG stream keyed by a stable
   SeedSequence, so toggling one knob never perturbs another's draws.
4. The ``concept`` scenario sets the concept rotation on the target site explicitly (no
   random ``concept_site_frac``).

Site-level params are SHARED-canonical across sites (so ``no_shift`` means "unseen subjects,
no systematic site gap"); subject-level anatomy perturbations are canonical and present in
every scenario; the target site additionally receives the scenario's mechanism shift.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from h2cmi.domains import DomainDAG, DomainLabels
from h2cmi.data.eeg_simulator import SimulatedEEG

# Independent RNG streams (stable integer ids -> reproducible, non-interfering SeedSequences)
_STREAMS = {name: i for i, name in enumerate([
    "struct", "subj_perturb", "labels", "phase", "noise", "session_noise",
    "target_cov", "target_concept", "target_prior", "target_montage",
])}


def _rng(data_seed: int, name: str, *ints: int) -> np.random.Generator:
    seq = np.random.SeedSequence([int(data_seed), _STREAMS[name], *[int(x) for x in ints]])
    return np.random.default_rng(seq)


@dataclass
class ScenarioSpec:
    name: str
    target_cov: float = 0.0
    target_prior: float = 0.0
    target_concept: float = 0.0           # shared rotation of the class->source-power map
    target_montage: float = 0.0
    target_noise_delta: float = 0.0
    matched_domain: bool = False          # target subjects reuse SOURCE anatomy (identity-null)


# canonical scenario names (review Stage-A renames). ``concept`` is a conditional GEOMETRY
# rotation, not a label mechanism, so it is named ``conditional_rotation``. ``no_shift`` keeps
# unseen-subject anatomy (a real random effect), so it is the ``population_null``;
# ``matched_domain_null`` is the true identity-null for calibrating the rollback threshold.
PRESET_SCENARIOS = {
    "population_null":          ScenarioSpec("population_null"),
    "matched_domain_null":      ScenarioSpec("matched_domain_null", matched_domain=True),
    "cov":                      ScenarioSpec("cov", target_cov=1.0),
    "prior":                    ScenarioSpec("prior", target_prior=1.0),
    "conditional_rotation":     ScenarioSpec("conditional_rotation", target_concept=0.6),
    "cov_prior":                ScenarioSpec("cov_prior", target_cov=1.0, target_prior=1.0),
    "cov_conditional_rotation": ScenarioSpec("cov_conditional_rotation",
                                             target_cov=1.0, target_concept=0.6),
}
# back-compat: old names resolve to the SAME canonical spec (so output is always canonical)
_ALIASES = {"no_shift": "population_null", "concept": "conditional_rotation",
            "cov_concept": "cov_conditional_rotation"}
PRESET_SCENARIOS.update({old: PRESET_SCENARIOS[new] for old, new in _ALIASES.items()})


class PairedEEGSimulator:
    def __init__(self, n_classes=3, n_chans=16, n_times=128, fs=128.0, n_sources=8,
                 n_disc=None, bands=((4, 8), (8, 13), (13, 30), (30, 45)),
                 base_noise=0.3, subj_anatomy=0.3, data_seed=0):
        self.n_classes = int(n_classes)
        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.fs = float(fs)
        self.n_sources = int(n_sources)
        self.n_disc = int(n_disc) if n_disc is not None else min(n_sources, max(2, n_classes + 2))
        self.bands = bands
        self.base_noise = float(base_noise)
        self.subj_anatomy = float(subj_anatomy)
        self.data_seed = int(data_seed)

        rs = _rng(data_seed, "struct")
        self.M0 = rs.standard_normal((self.n_chans, self.n_sources)).astype(np.float32)
        self.M0 /= np.linalg.norm(self.M0, axis=0, keepdims=True) + 1e-8
        self.src_freq = np.array(
            [rs.uniform(*bands[k % len(bands)]) for k in range(self.n_sources)], dtype=np.float32)
        base = rs.standard_normal((self.n_classes, self.n_disc)).astype(np.float32)
        for c in range(self.n_classes):
            base[c, c % self.n_disc] += 2.5
        self.class_power = base
        self.uniform_prior = np.full(self.n_classes, 1.0 / self.n_classes)

    # -- helpers -----------------------------------------------------------------
    def _rotation(self, rng, angle):
        R = np.eye(self.n_disc, dtype=np.float32)
        if angle == 0 or self.n_disc < 2:
            return R
        for _ in range(self.n_disc // 2):
            i, j = rng.choice(self.n_disc, size=2, replace=False)
            c, s = np.cos(angle), np.sin(angle)
            G = np.eye(self.n_disc, dtype=np.float32)
            G[i, i] = c; G[j, j] = c; G[i, j] = -s; G[j, i] = s
            R = R @ G
        return R

    def _subj_dM(self, subject):
        rng = _rng(self.data_seed, "subj_perturb", subject)
        return self.subj_anatomy * rng.standard_normal((self.n_chans, self.n_sources)).astype(np.float32)

    def _gen_sources(self, rng, power):
        n_tr = power.shape[0]
        t = np.arange(self.n_times, dtype=np.float32) / self.fs
        phase = rng.uniform(0, 2 * np.pi, size=(n_tr, self.n_sources, 1)).astype(np.float32)
        w = (2 * np.pi * self.src_freq).reshape(1, self.n_sources, 1)
        osc = np.sin(w * t.reshape(1, 1, self.n_times) + phase)
        wiggle = 0.15 * rng.standard_normal((n_tr, self.n_sources, self.n_times)).astype(np.float32)
        amp = np.sqrt(np.clip(power, 1e-4, None)).astype(np.float32)[:, :, None]
        return amp * osc + wiggle

    def _site_params(self, site, target_site, scenario: ScenarioSpec):
        """Resolve (prior, rotation, site_dM, gain, noise_delta) for a site.

        Source sites: canonical (uniform prior, identity rotation, no extra mixing, unit
        gain, no noise delta). Target site: canonical + the scenario's mechanism shift,
        each drawn from its own stream.
        """
        prior = self.uniform_prior.copy()
        rot = np.eye(self.n_disc, dtype=np.float32)
        site_dM = np.zeros((self.n_chans, self.n_sources), dtype=np.float32)
        gain = np.ones(self.n_chans, dtype=np.float32)
        noise_delta = 0.0
        if site == target_site:
            if scenario.target_cov > 0:
                r = _rng(self.data_seed, "target_cov", site)
                site_dM = scenario.target_cov * 0.6 * r.standard_normal(
                    (self.n_chans, self.n_sources)).astype(np.float32)
            if scenario.target_prior > 0:
                r = _rng(self.data_seed, "target_prior", site)
                logits = scenario.target_prior * r.standard_normal(self.n_classes)
                p = np.exp(logits - logits.max()); prior = (p / p.sum()).astype(np.float64)
            if scenario.target_concept > 0:
                r = _rng(self.data_seed, "target_concept", site)
                rot = self._rotation(r, scenario.target_concept)
            if scenario.target_montage > 0:
                r = _rng(self.data_seed, "target_montage", site)
                gain = (1.0 + scenario.target_montage * r.standard_normal(self.n_chans)).astype(np.float32)
                drop = r.random(self.n_chans) < (0.15 * scenario.target_montage)
                gain[drop] *= 0.1
            noise_delta = scenario.target_noise_delta
        return prior, rot, site_dM, gain, noise_delta

    def _gen_session(self, site, subject, session, prior, rot, M_d, gain, noise_delta):
        rl = _rng(self.data_seed, "labels", site, subject, session)
        ystar = rl.choice(self.n_classes, size=self.trials, p=prior).astype(np.int64)
        cp = (self.class_power[ystar] @ rot.T)
        power = np.full((self.trials, self.n_sources), 0.5, dtype=np.float32)
        power[:, :self.n_disc] = np.clip(0.5 + cp, 1e-3, None)
        src = self._gen_sources(_rng(self.data_seed, "phase", site, subject, session), power)
        x = np.einsum("cs,nst->nct", M_d, src) * gain[None, :, None]
        rs = _rng(self.data_seed, "session_noise", site, subject, session)
        scale = self.base_noise * rs.uniform(0.5, 1.5) + noise_delta
        x = x + scale * _rng(self.data_seed, "noise", site, subject, session).standard_normal(
            x.shape).astype(np.float32)
        x = (x - x.mean(2, keepdims=True)) / (x.std(2, keepdims=True) + 1e-6)
        return x.astype(np.float32), ystar

    # -- full sample -------------------------------------------------------------
    def sample(self, n_sites, subjects_per_site, sessions_per_subject, trials_per_session,
               target_site, scenario: ScenarioSpec | str) -> SimulatedEEG:
        if isinstance(scenario, str):
            scenario = PRESET_SCENARIOS[scenario]
        self.trials = int(trials_per_session)
        dag = DomainDAG.hierarchical_site_subject_session(
            n_sites, subjects_per_site, sessions_per_subject)
        # source subjects (for matched_domain_null: target subjects reuse these anatomies)
        src_subjects = [s for s in range(n_sites * subjects_per_site)
                        if s // subjects_per_site != target_site]
        Xs, ys, lv_site, lv_subj, lv_sess = [], [], [], [], []
        sess_id = 0
        for site in range(n_sites):
            prior, rot, site_dM, gain, ndelta = self._site_params(site, target_site, scenario)
            for sj in range(subjects_per_site):
                subject = site * subjects_per_site + sj
                if scenario.matched_domain and site == target_site:
                    # match a source subject's mixing/anatomy; canonical site params; trial
                    # seeds (phase/noise/labels) stay keyed to THIS subject -> independent resample
                    anatomy_subject = src_subjects[sj % len(src_subjects)]
                    M_d = self.M0 + self._subj_dM(anatomy_subject)
                else:
                    M_d = self.M0 + self._subj_dM(subject) + site_dM
                for ss in range(sessions_per_subject):
                    x, ystar = self._gen_session(site, subject, sess_id, prior, rot, M_d, gain, ndelta)
                    Xs.append(x); ys.append(ystar)
                    lv_site.append(np.full(self.trials, site))
                    lv_subj.append(np.full(self.trials, subject))
                    lv_sess.append(np.full(self.trials, sess_id))
                    sess_id += 1
        X = np.concatenate(Xs, 0); y = np.concatenate(ys, 0)
        levels = np.stack([np.concatenate(lv_site), np.concatenate(lv_subj),
                           np.concatenate(lv_sess)], axis=1)
        domains = DomainLabels(dag, levels)
        meta = dict(scenario=scenario.name, target_site=target_site, data_seed=self.data_seed)
        return SimulatedEEG(X, y, y.copy(), domains, dag, self.n_classes, self.fs, meta)


if __name__ == "__main__":
    sim = PairedEEGSimulator(n_classes=3, n_chans=12, n_times=128, data_seed=0)
    a = sim.sample(3, 2, 2, 16, target_site=0, scenario="population_null")
    b = sim.sample(3, 2, 2, 16, target_site=0, scenario="cov")
    src_a, src_b = a.site != 0, b.site != 0
    same = np.array_equal(a.X[src_a], b.X[src_b]) and np.array_equal(a.y[src_a], b.y[src_b])
    print("source identical across scenarios:", same)
    tgt_diff = not np.array_equal(a.X[a.site == 0], b.X[b.site == 0])
    print("target differs (cov vs population_null):", tgt_diff)
    c = sim.sample(3, 2, 2, 16, target_site=0, scenario="prior")
    print("prior leaves source identical:", np.array_equal(a.X[src_a], c.X[c.site != 0]))
    # back-compat alias resolves to canonical
    print("alias no_shift->population_null:", PRESET_SCENARIOS["no_shift"].name == "population_null")
    # matched_domain_null: source still identical; target uses source anatomy (no site shift)
    m = sim.sample(3, 2, 2, 16, target_site=0, scenario="matched_domain_null")
    print("matched_domain_null source identical:", np.array_equal(a.X[src_a], m.X[m.site != 0]))
    assert same and tgt_diff, "pairing broken"
    print("paired_simulator self-test PASSED")
