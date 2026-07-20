"""LURE — Levelled Unbiased Risk Estimator (Farquhar, Gal & Rainforth 2021; Kossen et al. 2021).

Frozen formula (C87P SPEC / §6.4):
    R_LURE = (1/M) sum_{m=1..M} v_m * L_m
    v_m    = 1 + ((N - M)/(N - m)) * ( 1/((N - m + 1) * q(i_m)) - 1 )
where N = pool size, M = number acquired, and q(i_m) is the proposal mass of the m-th acquired point
among the N-m+1 points REMAINING at step m (without-replacement acquisition). Key identities verified
in tests / CALIB:
    * uniform proposal => q(i_m) = 1/(N-m+1) => every v_m = 1 => R_LURE == naive mean of acquired losses;
    * M == N          => v_m = 1 for all m (full pool);
    * E[v_m] = 1      => R_LURE is unbiased for the full-pool mean under any proposal.
"""
from __future__ import annotations

import numpy as np


def lure_weights(N: int, q_seq: np.ndarray) -> np.ndarray:
    """v_m for m=1..M. q_seq[m-1] = proposal mass of the m-th acquired point among remaining points."""
    q = np.asarray(q_seq, float)
    M = q.size
    m = np.arange(1, M + 1)
    v = np.ones(M, float)
    # last acquisition m=M has (N-M)/(N-m)=0 factor -> v_M = 1 exactly; guard the division.
    denom = (N - m).astype(float)
    factor = np.where(denom != 0.0, (N - M) / np.where(denom == 0.0, 1.0, denom), 0.0)
    v = 1.0 + factor * (1.0 / ((N - m + 1) * q) - 1.0)
    return v


def lure_risk(loss_seq: np.ndarray, q_seq: np.ndarray, N: int) -> float:
    """R_LURE for a single candidate. loss_seq[m-1] = that candidate's loss at the m-th acquired point."""
    v = lure_weights(N, q_seq)
    return float(np.mean(v * np.asarray(loss_seq, float)))


def without_replacement_proposal_sequence(weights: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Given fixed positive proposal weights over the N pool points and the acquisition ORDER (indices
    acquired, in order), return q_seq: the proposal mass of each acquired point among the remaining set
    at its step. This is what LURE consumes; it makes the estimator unbiased for any `weights`."""
    w = np.asarray(weights, float).copy()
    remaining = np.ones(w.size, bool)
    q_seq = np.empty(order.size, float)
    for m, i in enumerate(order):
        tot = w[remaining].sum()
        q_seq[m] = w[i] / tot
        remaining[i] = False
    return q_seq
