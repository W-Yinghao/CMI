"""Reusable paired grouped permutation engine (the K1 null; see ``oaci/decision/k1_permutation.py``).

The K1 statistic is a PAIRED difference between two representations of the SAME held-out audit rows
(``Z_ERM`` and ``Z_OACI`` on identical ``sample_id``/``y``/``d``/``group``/``mass``). The pre-registered
null exchanges the two representations WITHIN each ``(Y, recording_group)`` stratum as a BLOCK (a paired
sign-flip at the stratum level): for a permutation's per-stratum swap bit ``b_s``, every row in stratum
``s`` either keeps its ``(ERM->ERM arm, OACI->OACI arm)`` assignment (``b_s=0``) or swaps it (``b_s=1``).

This module owns only the PLAN (deterministic per-stratum swap bits + a plan hash) and the row-level arm
construction. It never shuffles rows independently, never touches ``Y``/``D``, never rebuilds the support
or the fold/probe plans, and never reads target data — those disciplines are enforced by the caller
(``k1_permutation.py``) which keeps the support graph, fold plan and probe config FIXED across the null.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .design import hash_strings


def strata_of_rows(y, group) -> tuple:
    """Map each row to its ``(Y, recording_group)`` stratum. Returns ``(stratum_index[N] int, keys)`` where
    ``keys`` is the deterministically-sorted tuple of unique ``(int(y), str(group))`` stratum keys and
    ``stratum_index[i]`` indexes into it. Row-order sensitive in the OUTPUT array only (a row keeps its own
    stratum); the KEY SET + order are canonical (sorted), so the plan hash is row-order invariant."""
    y = np.asarray(y, dtype=int).ravel()
    grp = [str(g) for g in np.asarray(group).ravel().tolist()]
    if y.shape[0] != len(grp):
        raise ValueError("y and group length mismatch")
    keys = sorted({(int(y[i]), grp[i]) for i in range(len(grp))})
    index = {k: i for i, k in enumerate(keys)}
    stratum_index = np.array([index[(int(y[i]), grp[i])] for i in range(len(grp))], dtype=int)
    return stratum_index, tuple(keys)


def _plan_hash(seed, n_permutations, strata_keys, bits) -> str:
    h = hashlib.sha256()
    h.update(f"paired_swap_within_y_recording_group|seed={int(seed)}|n={int(n_permutations)}".encode())
    for (yv, gv) in strata_keys:
        h.update(int(yv).to_bytes(8, "little", signed=True))
        hash_strings(h, [gv])
    b = np.ascontiguousarray(np.asarray(bits, dtype=np.uint8))
    h.update(f"bits{b.shape}".encode()); h.update(b.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class PairedPermutationPlan:
    """Deterministic per-stratum swap bits for the paired grouped permutation null. ``bits`` is
    ``[n_permutations, n_strata]`` boolean (read-only); ``bits[p, s]`` = swap stratum ``s`` in
    permutation ``p``. Bit-for-bit reproducible from ``(seed, n_permutations, strata_keys)``."""
    seed: int
    n_permutations: int
    strata_keys: tuple
    bits: np.ndarray
    plan_hash: str

    @property
    def n_strata(self) -> int:
        return len(self.strata_keys)


def make_paired_permutation_plan(y, group, n_permutations: int, seed: int) -> PairedPermutationPlan:
    """Build the permutation plan for the ``(Y, recording_group)`` strata of these rows. The bits come from
    a single ``default_rng(seed)`` draw over the canonical (sorted) strata, so the plan is deterministic and
    the parallel/sequential nulls are identical. The seed affects ONLY these bits — never training,
    selection, audit leakage or prediction."""
    if int(n_permutations) < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}")
    _, keys = strata_of_rows(y, group)
    rng = np.random.default_rng(int(seed))
    bits = rng.integers(0, 2, size=(int(n_permutations), len(keys))).astype(bool)
    bits.setflags(write=False)
    return PairedPermutationPlan(seed=int(seed), n_permutations=int(n_permutations), strata_keys=keys,
                                 bits=bits, plan_hash=_plan_hash(seed, n_permutations, keys, bits))


def validate_permutation_plan(plan: PairedPermutationPlan) -> None:
    if _plan_hash(plan.seed, plan.n_permutations, plan.strata_keys, plan.bits) != plan.plan_hash:
        raise ValueError("permutation plan hash does not recompute")


def swap_row_mask(plan: PairedPermutationPlan, stratum_index: np.ndarray, p: int) -> np.ndarray:
    """The per-ROW boolean swap mask for permutation ``p`` (a row swaps iff its stratum's bit is set)."""
    return plan.bits[int(p)][stratum_index]


def build_paired_arms(Z_a: np.ndarray, Z_b: np.ndarray, swap_row: np.ndarray) -> tuple:
    """Given the two paired representations and a per-row swap mask, return ``(Z_a_arm, Z_b_arm)``: rows
    where ``swap_row`` is True exchange their ``a``/``b`` representation as a block. ``swap_row`` all-False
    reproduces ``(Z_a, Z_b)`` exactly (the observed statistic)."""
    Z_a = np.asarray(Z_a); Z_b = np.asarray(Z_b)
    if Z_a.shape != Z_b.shape:
        raise ValueError("paired representations must have identical shape")
    m = np.asarray(swap_row, dtype=bool).reshape(-1, 1)
    return np.where(m, Z_b, Z_a), np.where(m, Z_a, Z_b)


def permutation_p_values(observed: float, null: np.ndarray, n_permutations: int) -> dict:
    """Monte-Carlo permutation p-values (the observed counts as one realisation, so ``+1`` num/denom). The
    paired sign-flip null is symmetric about 0, so the two-sided test compares ``|Δ|`` against 0."""
    null = np.asarray(null, dtype=np.float64)
    denom = int(n_permutations) + 1
    p_lower = (1 + int(np.sum(null <= observed))) / denom            # Δ<0 = OACI leaks less
    p_upper = (1 + int(np.sum(null >= observed))) / denom
    p_two = (1 + int(np.sum(np.abs(null) >= abs(observed)))) / denom
    return {"p_lower": float(p_lower), "p_upper": float(p_upper), "p_two_sided": float(p_two)}
