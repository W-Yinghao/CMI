"""C86H confirmatory inference + two-level output taxonomy (§10-§11).

Pure functions over per-target-subject effect vectors (effect = P0 loss - active loss,
so positive = active improves). No real data is read here; the C86H entrypoint feeds
these functions with the frozen-dispatcher outputs once a field exists. Everything is
seed-deterministic so a confirmation is exactly replayable.

The registered decision object is TWO LEVELS:
  * Level 1 (the gate)  : formal C86-A..E + label-complexity C86-L1..L4
  * Level 2 (a reading) : the interpretive descriptor; POLICY_LIMITED is fixed to
                          NOT_IDENTIFIABLE_IN_C86H (no oracle-acquisition diagnostic).
"""
from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import numpy as np

from oaci.theory.c86_active_program import empirical_upper_cvar
from . import contract as K


# --------------------------------------------------------------------------- seeds
def maxt_seed(cohort: str, salt: str = "C86H_MAXT_NULL_V1") -> int:
    """Target-bound deterministic seed for a cohort's within-cohort max-T null."""
    digest = hashlib.sha256(f"{salt}|{cohort}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


# ------------------------------------------------------------------- point statistics
def _t_stat(effect: np.ndarray) -> float:
    e = np.asarray(effect, dtype=np.float64)
    n = e.shape[0]
    m = float(e.mean())
    sd = float(e.std(ddof=1)) if n > 1 else 0.0
    if sd == 0.0:
        return 0.0 if m == 0.0 else float(np.sign(m) * np.inf)
    return m / (sd / np.sqrt(n))


def favorable_fraction(effect: Sequence[float]) -> float:
    return float(np.mean(np.asarray(effect, dtype=np.float64) > 0.0))


def worst_target(effect: Sequence[float]) -> float:
    return float(np.min(np.asarray(effect, dtype=np.float64)))


def positive_cells(cell_mean_effects: Sequence[float]) -> int:
    """Count panel x seed x level cells (8) whose mean effect is positive."""
    return int(np.sum(np.asarray(cell_mean_effects, dtype=np.float64) > 0.0))


def loto_preservation(effect: Sequence[float], margin: float = K.MATERIALITY_MARGIN) -> float:
    """Leave-one-target-out fraction whose remaining mean effect still meets the margin."""
    e = np.asarray(effect, dtype=np.float64)
    n = e.shape[0]
    if n <= 1:
        return 0.0
    holds = sum(1 for i in range(n) if np.delete(e, i).mean() >= margin)
    return holds / n


def tail_effects(p0_losses: Sequence[float], active_losses: Sequence[float],
                 alphas: Sequence[float] = K.CVAR_ALPHA_GRID) -> dict:
    """Upper-tail CVaR reduction (P0 - active) at each registered alpha."""
    return {a: empirical_upper_cvar(p0_losses, a) - empirical_upper_cvar(active_losses, a)
            for a in alphas}


# ------------------------------------------------------------------ familywise max-T
def maxt_familywise(family_effects: Mapping[tuple, Sequence[float]], seed: int,
                    alpha: float = K.FAMILYWISE_ALPHA,
                    draws: int = K.MAXT_DRAWS) -> dict:
    """Single-step sign-flip max-T over a within-cohort family (target-subject cluster).

    ``family_effects`` maps each (method, budget) hypothesis to a per-target effect
    vector, all sharing the SAME target ordering. A single sign vector is drawn per
    replicate and applied across the whole family, respecting the target-subject cluster.
    Returns per-hypothesis familywise significance and the critical value.
    """
    keys = list(family_effects)
    E = np.stack([np.asarray(family_effects[k], dtype=np.float64) for k in keys])  # (H, n)
    n = E.shape[1]
    if E.shape[0] == 0 or n < 2:
        return {"significant": {k: False for k in keys}, "critical": float("nan")}
    obs = np.array([_t_stat(E[i]) for i in range(E.shape[0])])
    rng = np.random.default_rng(seed)
    null_max = np.empty(draws, dtype=np.float64)
    root_n = np.sqrt(n)
    for d in range(draws):
        s = rng.choice(np.array([-1.0, 1.0]), size=n)
        Es = E * s
        m = Es.mean(axis=1)
        sd = Es.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(sd > 0.0, m / (sd / root_n), 0.0)
        null_max[d] = float(np.max(t))
    critical = float(np.quantile(null_max, 1.0 - alpha))
    return {"significant": {k: bool(obs[i] > critical) for i, k in enumerate(keys)},
            "critical": critical,
            "observed": {k: float(obs[i]) for i, k in enumerate(keys)}}


# --------------------------------------------------------------------- qualifications
def mean_qualification(effect: Sequence[float], cell_mean_effects: Sequence[float],
                       familywise_significant: bool) -> dict:
    """All registered mean-side gates for one (cohort, method, budget)."""
    e = np.asarray(effect, dtype=np.float64)
    checks = {
        "familywise_significant": bool(familywise_significant),
        "mean_material": float(e.mean()) >= K.MATERIALITY_MARGIN,
        "favorable_fraction": favorable_fraction(e) >= K.FAVORABLE_TARGET_FRACTION,
        "worst_target": worst_target(e) >= K.WORST_TARGET_EFFECT_FLOOR,
        "positive_cells": positive_cells(cell_mean_effects) >= K.POSITIVE_CELLS_MIN,
        "loto": loto_preservation(e) >= K.LOTO_PRESERVATION_MIN,
    }
    return {"qualified": all(checks.values()), "checks": checks}


def tail_qualification(p0_losses: Sequence[float], active_losses: Sequence[float]) -> dict:
    te = tail_effects(p0_losses, active_losses)
    primary = te[K.PRIMARY_CVAR_ALPHA] >= K.TAIL_CVAR90_MARGIN
    nonneg = all(v >= 0.0 for v in te.values())
    return {"qualified": bool(primary and nonneg), "tail_effects": te,
            "primary_ok": bool(primary), "all_alpha_nonnegative": bool(nonneg)}


# ---------------------------------------------------------------------- budget status
def budget_status(budget, pool_size: int) -> str:
    """Registered support rule: a finite budget above the pool is INPUT_UNAVAILABLE and is
    NEVER substituted with FULL."""
    if budget == "FULL":
        return "SUPPORTED"
    return "SUPPORTED" if int(budget) <= int(pool_size) else "INPUT_UNAVAILABLE"


# --------------------------------------------------------------- Level-1 formal gate
def _passes_all(per_cohort, kind, mb) -> bool:
    return all(per_cohort[c][kind].get(mb, False) for c in per_cohort)


def formal_gate(per_cohort: Mapping[str, Mapping[str, Mapping[tuple, bool]]],
                blocker: bool = False) -> str:
    """Precedence E -> A -> B -> C -> D over the registered definitions.

    ``per_cohort[cohort] = {"mean": {(method,budget): bool}, "tail": {...: bool}}``.
    """
    if blocker:
        return "C86-E"
    family = [(m, b) for m in K.ACTIVE_METHODS for b in K.FINITE_BUDGETS]
    if any(_passes_all(per_cohort, "mean", mb) and _passes_all(per_cohort, "tail", mb)
           for mb in family):
        return "C86-A"
    if any(_passes_all(per_cohort, "mean", mb) for mb in family):
        return "C86-B"
    any_mean = any(per_cohort[c]["mean"].get(mb, False)
                   for c in per_cohort for mb in family)
    if not any_mean:
        return "C86-C"
    return "C86-D"


def _cohort_method_frontier(per_cohort, cohort, method):
    """Smallest finite budget that qualifies at that budget and every larger one."""
    for i, b in enumerate(K.FINITE_BUDGETS):
        if all(per_cohort[cohort]["mean"].get((method, bb), False)
               for bb in K.FINITE_BUDGETS[i:]):
            return b
    return None


def label_frontier(per_cohort: Mapping[str, Mapping[str, Mapping[tuple, bool]]]) -> str:
    cohorts = list(per_cohort)
    for m in K.ACTIVE_METHODS:
        fr = {c: _cohort_method_frontier(per_cohort, c, m) for c in cohorts}
        if all(v is not None for v in fr.values()):
            idxs = [K.FINITE_BUDGETS.index(v) for v in fr.values()]
            if max(idxs) - min(idxs) <= 1:
                maxb = max(fr.values())
                if maxb <= 8:
                    return "C86-L1"
                if maxb in (16, 32):
                    return "C86-L2"
    # frontiers absent in some cohort -> L4; else heterogeneous -> L3
    def has_any(c):
        return any(_cohort_method_frontier(per_cohort, c, m) is not None
                   for m in K.ACTIVE_METHODS)
    if not all(has_any(c) for c in cohorts):
        return "C86-L4"
    return "C86-L3"


# --------------------------------------------------------- Level-2 interpretive reading
def interpretive_descriptor(gate: str) -> dict:
    """Secondary mechanism reading. Reported alongside the gate, never replacing it.

    POLICY_LIMITED is fixed to NOT_IDENTIFIABLE_IN_C86H and is never emitted from
    ordinary results, because this contract defines no oracle-acquisition diagnostic.
    """
    reading = {
        "C86-A": "BOUNDARY_OPERATIONALLY_CROSSED",
        "C86-B": "BOUNDARY_WEAKENED_NOT_ROBUST",
        "C86-C": "NO_REGISTERED_ACTIVE_GAIN",
        "C86-D": "ACQUISITION_VIEW_NONTRANSPORTABLE",
        "C86-E": None,  # blocker: no interpretive reading
    }[gate]
    return {"descriptor": reading,
            "policy_limited": K.POLICY_LIMITED_RESOLUTION,
            "oracle_acquisition_diagnostic": K.ORACLE_ACQUISITION_DIAGNOSTIC}


def classify(per_cohort, blocker: bool = False) -> dict:
    """Full two-level classification for a completed C86H confirmation."""
    gate = formal_gate(per_cohort, blocker=blocker)
    out = {"formal_gate": gate,
           "label_frontier": None if blocker else label_frontier(per_cohort),
           "interpretive": interpretive_descriptor(gate),
           "pooled_dataset_pvalue": K.POOLED_DATASET_PVALUE}
    return out
