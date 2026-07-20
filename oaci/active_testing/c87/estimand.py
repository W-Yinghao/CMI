"""C87 estimand module — cross-fit held-SELECTION reference + T^CF / R^CF / G / s_e / D^fin.

This is PRODUCTION code (C87E reuses it with the real 648-model zoo); the synthetic control gate
(controls.py) exercises exactly these functions. It is faithful to C87P_FROZEN.md SPEC 4.A as
CORRECTED by the v3 PM addendum:

  * s_e = candidate-loss DISPERSION standardizer (SD over the full fixed candidate set of held
    patient-mean losses). NOT "robust". Computed on the complete candidate set, no cross-fit
    (each candidate is a fixed function; winner's curse enters only at argmin/argmax).
  * The 5-fold patient-level cross-fit estimates the generalization loss of a held-SELECTION
    PROCEDURE ("select on 4/5 held patients, evaluate on 1/5"), NOT an unbiased oracle minimum.
        a_hat^H_{-f}       = argmin_a  L_{H\f}(a)                      (select on the other folds)
        L^{H,CF}_ref       = (1/F) sum_f L_{H_f}( a_hat^H_{-f} )       (cross-fit reference)
        T^CF               = (1/F) sum_f [ L_{H_f}(aC) - L_{H_f}(a_hat^H_{-f}) ]
        R^CF(a_pick)       = (1/F) sum_f [ L_{H_f}(a_pick) - L_{H_f}(a_hat^H_{-f}) ]   (MAY be < 0)
        G(pi)              = R^CF(P0 pick) - R^CF(pi pick)    (reference cancels; paired)
  * SECONDARY finite-held oracle: a^{H,fin} = argmin_a L^H(a), realized regret >= 0 but optimistic;
    D^fin = max_a L^H(a) - min_a L^H(a) = secondary range / degeneracy diagnostic.

All losses are patient-level means of per-record binary NLL on the single selected task (v3 pin).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

EPS = 1e-7
XFIT_FOLDS = 5
XFIT_SALT = "C87_HELD_XFIT_V1"


def binary_nll(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-record binary NLL for every candidate. probs:(A,n_r) pos-class prob; y:(n_r,) in {0,1}.

    Returns (A, n_r). Clipped for numerical safety (proper score; no metric shopping)."""
    p = np.clip(np.asarray(probs, float), EPS, 1.0 - EPS)
    yy = np.asarray(y, float)[None, :]
    return -(yy * np.log(p) + (1.0 - yy) * np.log(1.0 - p))


def patient_mean_loss(per_record_loss: np.ndarray, patient_of: np.ndarray):
    """Collapse per-record loss (A,n_r) to per-patient mean loss (A,n_pat). Returns (Lbar, pat_ids).

    pat_ids is the sorted unique patient id array; column j of Lbar corresponds to pat_ids[j]."""
    pat_ids = np.unique(patient_of)
    A = per_record_loss.shape[0]
    Lbar = np.empty((A, pat_ids.size), float)
    for j, pid in enumerate(pat_ids):
        m = patient_of == pid
        Lbar[:, j] = per_record_loss[:, m].mean(axis=1)
    return Lbar, pat_ids


def held_view_loss(Lbar: np.ndarray, cols: np.ndarray | None = None) -> np.ndarray:
    """L^H(a) = mean over (selected) held patients of the patient-mean loss. Returns (A,)."""
    if cols is None:
        return Lbar.mean(axis=1)
    return Lbar[:, cols].mean(axis=1)


def dispersion_s_e(Lbar: np.ndarray) -> float:
    """s_e = SD over candidates of L^H(a), computed on the FULL fixed candidate set (no cross-fit)."""
    return float(np.std(held_view_loss(Lbar), ddof=0))


def d_fin(Lbar: np.ndarray) -> float:
    """D^fin = max_a L^H(a) - min_a L^H(a): SECONDARY range / degeneracy diagnostic."""
    Lh = held_view_loss(Lbar)
    return float(Lh.max() - Lh.min())


def finite_oracle_index(Lbar: np.ndarray) -> int:
    """a^{H,fin} = argmin_a L^H(a): SECONDARY in-sample finite-held argmin (optimistic; not primary)."""
    return int(np.argmin(held_view_loss(Lbar)))


def _fold_assignment(pat_ids: np.ndarray, folds: int = XFIT_FOLDS, salt: str = XFIT_SALT) -> np.ndarray:
    """Deterministic patient->fold in [0,folds) via SHA-256(salt|patient_id) (mirrors SPEC 4.A)."""
    out = np.empty(pat_ids.size, int)
    for i, pid in enumerate(pat_ids):
        h = hashlib.sha256(f"{salt}|{int(pid)}".encode()).hexdigest()
        out[i] = int(h, 16) % folds
    return out


@dataclass
class CrossFit:
    """Cross-fit held-selection reference and its per-fold machinery for one cohort."""

    ref: float                 # L^{H,CF}_ref = mean_f L_{H_f}(a_hat_{-f})
    fold_of: np.ndarray        # (n_pat,) fold id per patient column
    sel_per_fold: np.ndarray   # (F,) selected candidate index a_hat_{-f} for each held-out fold
    Lfold: np.ndarray          # (A, F) L_{H_f}(a) = mean loss over patients in fold f (per candidate)


def cross_fit(Lbar: np.ndarray, pat_ids: np.ndarray, folds: int = XFIT_FOLDS,
              salt: str = XFIT_SALT) -> CrossFit:
    """Build the held-SELECTION cross-fit reference. Lbar:(A,n_pat). Selection metric = held loss.

    a_hat_{-f} = argmin over candidates of mean loss on patients NOT in fold f; the reference is the
    mean over folds of that selector's loss ON fold f. This is the generalization loss of the
    select-on-4/5-eval-on-1/5 procedure (NOT an oracle)."""
    A, n_pat = Lbar.shape
    fold_of = _fold_assignment(pat_ids, folds, salt)
    Lfold = np.empty((A, folds), float)
    for f in range(folds):
        cols = np.where(fold_of == f)[0]
        # guard: a fold with no patients cannot be evaluated
        Lfold[:, f] = Lbar[:, cols].mean(axis=1) if cols.size else np.nan
    sel = np.empty(folds, int)
    ref_terms = np.empty(folds, float)
    for f in range(folds):
        other = [g for g in range(folds) if g != f and np.isfinite(Lfold[0, g])]
        # select on the OTHER folds (their combined patient-mean loss)
        cols_other = np.where(np.isin(fold_of, other))[0]
        a_hat = int(np.argmin(Lbar[:, cols_other].mean(axis=1)))
        sel[f] = a_hat
        ref_terms[f] = Lfold[a_hat, f]
    return CrossFit(ref=float(np.nanmean(ref_terms)), fold_of=fold_of, sel_per_fold=sel, Lfold=Lfold)


def excess_loss_cf(Lbar: np.ndarray, a_pick: int, cf: CrossFit) -> float:
    """R^CF(a_pick) = mean_f [ L_{H_f}(a_pick) - L_{H_f}(a_hat_{-f}) ].  MAY be negative."""
    folds = cf.sel_per_fold.size
    terms = [cf.Lfold[a_pick, f] - cf.Lfold[cf.sel_per_fold[f], f] for f in range(folds)
             if np.isfinite(cf.Lfold[0, f])]
    return float(np.mean(terms))


def transport_gap_cf(Lbar: np.ndarray, a_C: int, cf: CrossFit) -> float:
    """T^CF = R^CF(a_C): cross-fitted excess loss of the acquisition-view pick (= R at B=0)."""
    return excess_loss_cf(Lbar, a_C, cf)


def active_gain(Lbar: np.ndarray, a_pick_pi: int, a_pick_p0: int, cf: CrossFit) -> float:
    """G(pi) = R^CF(P0 pick) - R^CF(pi pick). Reference term cancels => paired difference."""
    return excess_loss_cf(Lbar, a_pick_p0, cf) - excess_loss_cf(Lbar, a_pick_pi, cf)


def is_vacuous(Lbar: np.ndarray, cluster_se: float, k: float = 1.0) -> bool:
    """Degeneracy guard: s_e (and D^fin) inside patient-cluster bootstrap noise => vacuous cohort.

    A vacuous cohort counts as NON-PASS for the IUT and may only DOWNGRADE the verdict (SPEC 4.F)."""
    return dispersion_s_e(Lbar) <= k * cluster_se
