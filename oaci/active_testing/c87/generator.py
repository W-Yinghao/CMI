"""C87 synthetic control-world generator (C87P §6.1, v3/B4).

Emits, per (candidate, record), a PREDICTION (pos-class prob) plus a KNOWN true single-task binary label;
the loss is DERIVED from (prediction,label) so the SAME predictions drive the loss estimators (LURE) AND
the label-adaptive selectors (MODEL SELECTOR / CODA). Worlds plant known transport + label-adaptive
structure so the control gate can verify detect(POS) / refuse(NEG) / calibrate(CALIB).

Competence model (per candidate a, record r with true label y_r):
    logit_true_{a,r} = base_a + g_a * D_r + s_p + eta_{a,r}
    prob_true        = sigmoid(logit_true) ;  p_pos = prob_true if y_r==1 else 1-prob_true
    loss             = -log(prob_true)          # binary NLL on the single task
D_r~Bernoulli(phi) marks "informative" records (competence gap g_a shows there); s_p is a shared patient
effect (=> intra-patient loss correlation rho); higher base_a => lower expected held loss (better candidate).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimand import binary_nll, held_view_loss, patient_mean_loss


@dataclass
class Cohort:
    probs: np.ndarray        # (A, n_r) predicted pos-class prob (always visible)
    y: np.ndarray            # (n_r,) true single-task binary label (revealed only on query)
    patient_of: np.ndarray   # (n_r,) patient index
    aC: int                  # acquisition-view pick a*C (deployed at B=0)
    Lpop: np.ndarray         # (A,) realized held-view loss L^H(a) = ground-truth reference
    aHfin: int               # argmin_a L^H(a) (true finite-held best)
    aPopBest: int            # population-optimal candidate (argmax expected ability) — transport target


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _gen_cohort(A, n_pat, rng, *, base, g, phi=0.15, rho=0.3, sigma=1.0,
                pi_pos=0.4, records_per_patient=1):
    # patients + records
    if records_per_patient == 1:
        rpp = np.ones(n_pat, int)
    else:
        rpp = 1 + rng.poisson(records_per_patient - 1, n_pat)
    patient_of = np.repeat(np.arange(n_pat), rpp)
    n_r = patient_of.size
    y = (rng.random(n_r) < pi_pos).astype(int)
    D = (rng.random(n_r) < phi).astype(float)
    s_p = rng.normal(0, np.sqrt(rho) * sigma, n_pat)[patient_of]      # shared per-patient effect
    eta = rng.normal(0, np.sqrt(max(1 - rho, 0)) * sigma, (A, n_r))
    logit_true = base[:, None] + g[:, None] * D[None, :] + s_p[None, :] + eta
    prob_true = np.clip(_sigmoid(logit_true), 1e-6, 1 - 1e-6)
    probs = np.where(y[None, :] == 1, prob_true, 1 - prob_true)
    Lbar, _ = patient_mean_loss(binary_nll(probs, y), patient_of)
    Lpop = held_view_loss(Lbar)
    return probs, y, patient_of, Lpop


# Tuned so POS exhibits DETECTABLE active gain (label-adaptive beats P0) at the tested budgets:
# a competitive TOP CLUSTER whose members are near-tied on non-informative records; the single best is
# distinguished mainly on informative (D=1) records, so uniform P0 confuses the cluster while a
# disagreement/EIG-seeking policy concentrates queries on the distinguishing records and wins.
POS_PARAMS = dict(n_top=8, base_top=1.15, base_rest=(0.25, 0.30), g_best=3.2, phi=0.15, rho=0.30)


def subsample_patients(coh, n_sub, rng):
    """Return a Cohort restricted to a random subset of n_sub patients (coverage checks: fix ONE
    population, then subsample patients from it). aC/aPopBest are population-level (kept)."""
    pat_ids = np.unique(coh.patient_of)
    keep = rng.choice(pat_ids, size=min(n_sub, pat_ids.size), replace=False)
    mask = np.isin(coh.patient_of, keep)
    # remap patient ids to a contiguous range for the subset
    sub_pat = coh.patient_of[mask]
    remap = {p: i for i, p in enumerate(np.unique(sub_pat))}
    new_pat = np.array([remap[p] for p in sub_pat])
    probs = coh.probs[:, mask]
    y = coh.y[mask]
    Lbar, _ = patient_mean_loss(binary_nll(probs, y), new_pat)
    Lpop = held_view_loss(Lbar)
    return Cohort(probs=probs, y=y, patient_of=new_pat, aC=coh.aC, Lpop=Lpop,
                  aHfin=int(np.argmin(Lpop)), aPopBest=coh.aPopBest)


def _expected_ability(base, g, phi):
    """Monotone proxy for expected competence (higher => lower expected loss): base + phi*g."""
    return base + phi * g


def make_world(kind, A=120, n_pat=300, E=3, seed=0, records_per_patient=1):
    """Return list of E Cohort objects for a control world in
    {POS, POS_DENSE, NEG_A, NEG_B, CALIB}. Deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    pp = POS_PARAMS
    cohorts = []
    for e in range(E):
        g = np.zeros(A)
        if kind in ("POS", "CALIB", "NEG_A"):
            base = rng.normal(pp["base_rest"][0], pp["base_rest"][1], A)
            top = rng.permutation(A)[:pp["n_top"]]
            base[top] = pp["base_top"]           # competitive top cluster (near-tied off informative recs)
            b = int(top[0])
            g[b] = pp["g_best"]                  # the single best is separable mainly on informative records
            a_pop = int(np.argmax(_expected_ability(base, g, pp["phi"])))
            probs, y, pat, Lpop = _gen_cohort(A, n_pat, rng, base=base, g=g, phi=pp["phi"], rho=pp["rho"],
                                              records_per_patient=records_per_patient)
            if kind == "NEG_A":
                worst_third = np.argsort(-Lpop)[:max(A // 3, 1)]   # anti-transport: aC in worst third
                aC = int(rng.choice(worst_third))
            else:
                aC = a_pop                        # transport: acquisition pick == population best
        elif kind == "POS_DENSE":
            base = rng.normal(pp["base_rest"][0], pp["base_rest"][1], A)
            top = rng.permutation(A)[:max(A // 6, 8)]
            base[top] = pp["base_top"]           # DENSE near-tie plateau (no single well-separated best)
            b = int(top[0]); g[b] = pp["g_best"] * 0.5
            a_pop = int(np.argmax(_expected_ability(base, g, pp["phi"])))
            probs, y, pat, Lpop = _gen_cohort(A, n_pat, rng, base=base, g=g, phi=pp["phi"], rho=pp["rho"],
                                              records_per_patient=records_per_patient)
            aC = a_pop                            # transport target = population best (may != in-sample argmin)
        elif kind == "NEG_B":
            base = np.full(A, 0.6)               # NO separation: all candidates identical distribution
            a_pop = 0
            probs, y, pat, Lpop = _gen_cohort(A, n_pat, rng, base=base, g=g, phi=pp["phi"], rho=pp["rho"],
                                              records_per_patient=records_per_patient)
            aC = int(rng.integers(A))            # no meaningful pick
        else:
            raise ValueError(kind)
        cohorts.append(Cohort(probs=probs, y=y, patient_of=pat, aC=aC,
                              Lpop=Lpop, aHfin=int(np.argmin(Lpop)), aPopBest=a_pop))
    return cohorts
