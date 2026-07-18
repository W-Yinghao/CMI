"""C86H confirmatory inference + two-level output taxonomy (§10-§11).

Pure functions over per-target-subject effect vectors (effect = P0 loss - active loss,
so positive = active improves). No real data is read here; the C86H runner feeds these
with the frozen-dispatcher outputs once a field exists. Everything is seed-deterministic
so a confirmation is exactly replayable.

The registered decision object is TWO LEVELS:
  * Level 1 (the gate)  : formal C86-A..E + label-complexity C86-L1..L4
  * Level 2 (a reading) : the interpretive descriptor, computed from the REAL secondary
                          objects (FULL construction-view ceiling + cross-cohort
                          robustness), NOT by table-lookup from the Level-1 gate.
                          POLICY_LIMITED is fixed to NOT_IDENTIFIABLE_IN_C86H (no
                          oracle-acquisition diagnostic).

Within-cohort max-T identity is bound to the registered program
(``C86_ACTIVE_TESTING_PROGRAM_PROTOCOL.json`` line 248):
    seed = low64( SHA256("C86_MAXT_V1" | dataset | registered_family) )
Brandl (16 targets, 2**16 == 65536 registered draws) enumerates the FULL sign group
exactly; ds007221 (2**37 >> 65536) uses the 65536 deterministic Monte-Carlo draws.
Both report the registered plus-one adjusted max-T p (min 1/65537).
"""
from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import numpy as np

from oaci.theory.c86_active_program import empirical_upper_cvar
from . import contract as K


# ------------------------------------------------------------------ family / seeds
def registered_family() -> str:
    """Canonical within-cohort max-T family string: A1@4|A1@8|...|A2H@32."""
    return "|".join(f"{m}@{b}" for m in K.ACTIVE_METHODS for b in K.FINITE_BUDGETS)


def family_digest(family: str | None = None) -> str:
    fam = registered_family() if family is None else family
    return hashlib.sha256(fam.encode("utf-8")).hexdigest()


def maxt_seed(dataset: str, family: str | None = None, salt: str = "C86_MAXT_V1") -> int:
    """Registered within-cohort max-T seed: low64(SHA256(C86_MAXT_V1|dataset|family))."""
    fam = registered_family() if family is None else family
    digest = hashlib.sha256(f"{salt}|{dataset}|{fam}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


# ------------------------------------------------------------------- point statistics
def _t_vec(m: np.ndarray, sd: np.ndarray, root_n: float) -> np.ndarray:
    """Shared degenerate-variance convention used by BOTH the observed and the null t."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0.0, m / (sd / root_n),
                        np.where(m == 0.0, 0.0, np.sign(m) * np.inf))


def _t_stat_rows(E: np.ndarray) -> np.ndarray:
    """Per-hypothesis one-sample t over the target-subject cluster; (H,) from (H,n)."""
    root_n = np.sqrt(E.shape[1])
    return _t_vec(E.mean(axis=1), E.std(axis=1, ddof=1), root_n)


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
def _null_max_distribution(E: np.ndarray, seed: int, draws: int) -> tuple:
    """Return (null_max, sign_mode, n_signs). Exhaustive full-enumeration when 2**n <= draws
    (Brandl 2**16 == 65536); the plus-one estimator is retained for both branches
    (conservative: the identity draw makes the exhaustive floor 2/(N+1))."""
    n = E.shape[1]
    root_n = np.sqrt(n)

    def stat_max(sign: np.ndarray) -> float:
        Es = E * sign
        return float(np.max(_t_vec(Es.mean(axis=1), Es.std(axis=1, ddof=1), root_n)))

    if 2 ** n <= draws:                                   # Brandl: 2**16 == 65536 exact
        n_signs = 2 ** n
        bits = np.arange(n)
        null_max = np.empty(n_signs, dtype=np.float64)
        for idx in range(n_signs):
            sign = 1.0 - 2.0 * ((idx >> bits) & 1).astype(np.float64)
            null_max[idx] = stat_max(sign)
        return null_max, "exhaustive", n_signs

    rng = np.random.default_rng(seed)                     # ds007221: MC 65536 draws
    null_max = np.empty(draws, dtype=np.float64)
    two = np.array([-1.0, 1.0])
    for d in range(draws):
        null_max[d] = stat_max(rng.choice(two, size=n))
    return null_max, "monte_carlo", draws


def maxt_familywise(family_effects: Mapping[tuple, Sequence[float]], dataset: str,
                    alpha: float = K.FAMILYWISE_ALPHA, draws: int = K.MAXT_DRAWS) -> dict:
    """Single-step sign-flip max-T over a within-cohort family (target-subject cluster).

    Seed is the registered dataset-bound family seed. Emits, per (method,budget)
    hypothesis: observed statistic, single-step plus-one adjusted max-T p-value
    ((1+#{null>=obs})/(N+1)), and familywise significance (adjusted p <= alpha).
    """
    keys = list(family_effects)
    E = np.stack([np.asarray(family_effects[k], dtype=np.float64) for k in keys])  # (H,n)
    n = E.shape[1]
    fam = registered_family()
    seed = maxt_seed(dataset, fam)
    if E.shape[0] == 0 or n < 2:
        return {"dataset": dataset, "family": fam, "family_sha256": family_digest(fam),
                "seed": seed, "sign_mode": None, "n_signs": 0, "critical": float("nan"),
                "alpha": alpha, "hypotheses": {}, "significant": {k: False for k in keys}}
    obs = _t_stat_rows(E)
    null_max, sign_mode, n_signs = _null_max_distribution(E, seed, draws)
    critical = float(np.quantile(null_max, 1.0 - alpha))
    hyp, sig = {}, {}
    for i, k in enumerate(keys):
        adj_p = (1 + int(np.sum(null_max >= obs[i]))) / (n_signs + 1)
        significant = adj_p <= alpha
        hyp[k] = {"observed": float(obs[i]), "adjusted_p": float(adj_p),
                  "significant": bool(significant)}
        sig[k] = bool(significant)
    return {"dataset": dataset, "family": fam, "family_sha256": family_digest(fam),
            "seed": seed, "sign_mode": sign_mode, "n_signs": n_signs,
            "critical": critical, "alpha": alpha, "hypotheses": hyp, "significant": sig}


# --------------------------------------------------------------------- qualifications
def mean_qualification(effect: Sequence[float], cell_mean_effects: Sequence[float],
                       familywise_significant: bool) -> dict:
    """Registered MEAN-side gates for one (cohort, method, budget).

    Per the registered inference contract, mean qualification is
    familywise + mean_material + favorable + worst + positive_cells. LOTO is a SEPARATE
    stability check scoped to C86-A only (see ``stability_qualification``); folding it in
    here would wrongly demote a true C86-B / label frontier to C86-C / C86-L4.
    """
    e = np.asarray(effect, dtype=np.float64)
    checks = {
        "familywise_significant": bool(familywise_significant),
        "mean_material": float(e.mean()) >= K.MATERIALITY_MARGIN,
        "favorable_fraction": favorable_fraction(e) >= K.FAVORABLE_TARGET_FRACTION,
        "worst_target": worst_target(e) >= K.WORST_TARGET_EFFECT_FLOOR,
        "positive_cells": positive_cells(cell_mean_effects) >= K.POSITIVE_CELLS_MIN,
    }
    return {"qualified": all(checks.values()), "checks": checks}


def stability_qualification(effect: Sequence[float]) -> dict:
    """C86-A-only registered stability check: LOTO preservation >= 0.75."""
    loto = loto_preservation(np.asarray(effect, dtype=np.float64))
    return {"qualified": loto >= K.LOTO_PRESERVATION_MIN, "loto": loto}


def tail_qualification(p0_losses: Sequence[float], active_losses: Sequence[float]) -> dict:
    te = tail_effects(p0_losses, active_losses)
    primary = te[K.PRIMARY_CVAR_ALPHA] >= K.TAIL_CVAR90_MARGIN
    nonneg = all(v >= 0.0 for v in te.values())
    return {"qualified": bool(primary and nonneg), "tail_effects": te,
            "primary_ok": bool(primary), "all_alpha_nonnegative": bool(nonneg)}


# ---------------------------------------------------------------------- budget status
def budget_status(budget, pool_size: int) -> str:
    """Registered support rule: a finite budget above the pool is INPUT_UNAVAILABLE and
    is NEVER substituted with FULL."""
    if budget == "FULL":
        return "SUPPORTED"
    return "SUPPORTED" if int(budget) <= int(pool_size) else "INPUT_UNAVAILABLE"


# --------------------------------------------------------------- Level-1 formal gate
def _passes_all(per_cohort, kind, mb) -> bool:
    return all(per_cohort[c][kind].get(mb, False) for c in per_cohort)


def formal_gate(per_cohort: Mapping[str, Mapping[str, Mapping[tuple, bool]]],
                blocker: bool = False) -> str:
    """Precedence E -> A -> B -> C -> D over the registered definitions.

    ``per_cohort[cohort] = {"mean": {...}, "tail": {...}, "stability": {...}}``.
    C86-A requires mean AND tail AND the registered stability (LOTO) universally; C86-B
    requires only mean universally (no tail, no stability).
    """
    if blocker:
        return "C86-E"
    family = [(m, b) for m in K.ACTIVE_METHODS for b in K.FINITE_BUDGETS]
    if any(_passes_all(per_cohort, "mean", mb) and _passes_all(per_cohort, "tail", mb)
           and _passes_all(per_cohort, "stability", mb) for mb in family):
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


_FRONTIER_RANK = {"C86-L1": 1, "C86-L2": 2, "C86-L3": 3, "C86-L4": 4}


def label_frontier(per_cohort: Mapping[str, Mapping[str, Mapping[tuple, bool]]]) -> str:
    """Registered label-complexity frontier. L1/L2 require a SINGLE method with all-cohort
    frontiers, ordinal distance <=1, AND homogeneous registered stability; heterogeneous
    method / ordinal / stability -> L3; a missing cohort frontier -> L4. The best (lowest
    complexity) label across methods is returned (L1 is existential over methods)."""
    cohorts = list(per_cohort)
    best = None
    for m in K.ACTIVE_METHODS:
        fr = {c: _cohort_method_frontier(per_cohort, c, m) for c in cohorts}
        if not all(v is not None for v in fr.values()):
            continue
        idxs = [K.FINITE_BUDGETS.index(v) for v in fr.values()]
        ordinal_ok = (max(idxs) - min(idxs)) <= 1
        stab = [per_cohort[c].get("stability", {}).get((m, fr[c]), False) for c in cohorts]
        stab_homog = all(stab) or not any(stab)     # heterogeneous stability -> L3
        maxb = max(fr.values())
        if ordinal_ok and stab_homog and maxb <= 8:
            lbl = "C86-L1"
        elif ordinal_ok and stab_homog and maxb in (16, 32):
            lbl = "C86-L2"
        else:
            lbl = "C86-L3"
        if best is None or _FRONTIER_RANK[lbl] < _FRONTIER_RANK[best]:
            best = lbl
    if best is not None:
        return best

    def has_any(c):
        return any(_cohort_method_frontier(per_cohort, c, m) is not None
                   for m in K.ACTIVE_METHODS)

    if not all(has_any(c) for c in cohorts):
        return "C86-L4"
    return "C86-L3"


# --------------------------------------------------- Level-2 interpretive reading (decoupled)
def full_ceiling_usable(mean: float, tail: float, near_opt: float) -> bool:
    """Is the FULL construction-view a usable near-optimal ceiling for this cohort?

    Registered C86D disposition constants: mean/tail <= CEIL_MAX and near-opt >= CEIL_NEAROPT_MIN.
    """
    return (mean <= K.CEIL_MAX and tail <= K.CEIL_MAX and near_opt >= K.CEIL_NEAROPT_MIN)


def interpretive_descriptor(full_ceiling_per_cohort: Mapping[str, Mapping[str, float]],
                            active_gain: Mapping[str, bool]) -> dict:
    """Level-2 reading computed from the REAL secondary objects, decoupled from the gate.

    * FULL ceiling not usable in ANY cohort            -> ACQUISITION_VIEW_NONTRANSPORTABLE
    * FULL ceiling usable + robust same-method x-cohort -> BOUNDARY_OPERATIONALLY_CROSSED
    * material gain but not universal (cohort/tail/stability) -> BOUNDARY_WEAKENED_NOT_ROBUST
    * no registered active gain                         -> NO_REGISTERED_ACTIVE_GAIN
    POLICY_LIMITED is fixed to NOT_IDENTIFIABLE_IN_C86H (no oracle-acquisition diagnostic).
    """
    usable = {c: full_ceiling_usable(v["mean"], v["tail"], v["near_opt"])
              for c, v in full_ceiling_per_cohort.items()}
    if not any(usable.values()):
        descriptor = "ACQUISITION_VIEW_NONTRANSPORTABLE"
    elif active_gain.get("robust_same_method_cross_cohort"):
        descriptor = "BOUNDARY_OPERATIONALLY_CROSSED"
    elif active_gain.get("any_material"):
        descriptor = "BOUNDARY_WEAKENED_NOT_ROBUST"
    else:
        descriptor = "NO_REGISTERED_ACTIVE_GAIN"
    return {"descriptor": descriptor,
            "full_ceiling_usable": usable,
            "policy_limited": K.POLICY_LIMITED_RESOLUTION,
            "oracle_acquisition_diagnostic": K.ORACLE_ACQUISITION_DIAGNOSTIC}


def classify(per_cohort, full_ceiling_per_cohort, active_gain, blocker: bool = False) -> dict:
    """Full two-level classification for a completed C86H confirmation.

    Level 1 (formal gate + frontier) and Level 2 (interpretive descriptor) are computed
    from SEPARATE inputs -- the descriptor never reads the gate.
    """
    gate = formal_gate(per_cohort, blocker=blocker)
    frontier = None if blocker else label_frontier(per_cohort)
    descriptor = ({"descriptor": None, "policy_limited": K.POLICY_LIMITED_RESOLUTION,
                   "oracle_acquisition_diagnostic": K.ORACLE_ACQUISITION_DIAGNOSTIC}
                  if blocker
                  else interpretive_descriptor(full_ceiling_per_cohort, active_gain))
    return {"formal_gate": gate, "label_frontier": frontier,
            "interpretive": descriptor, "pooled_dataset_pvalue": K.POOLED_DATASET_PVALUE}
