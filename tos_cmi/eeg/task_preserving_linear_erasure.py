"""Phase 2 -- task-carrier-preserving LEACE (TP-LEACE). A simple, auditable version (NOT full SPLINCE):

  1. estimate a linear task carrier T on SOURCE only (logistic-regression class directions);
  2. split Z = Z_T (projection onto span T) + Z_perp (task-orthogonal complement);
  3. apply plain LEACE to erase subject D on Z_perp ONLY;
  4. reconstruct RZ = Z_T + LEACE(Z_perp).

Intent: never break the task carrier to erase subject -- it would rather LEAVE subject information that lives
inside the task carrier than damage the task. Directly probes task<->subject entanglement in a compact latent:
if subject sits mostly inside the task carrier, TP-LEACE preserves the task but erases little subject; if it
sits mostly in the complement, TP-LEACE both preserves task and erases subject.

Also provides alpha-LEACE (soft interpolation toward full LEACE) for a later round -- not in the dry-run set.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from tos_cmi.eeg.erasure_baselines import leace_eraser, _ids


def _task_carrier(Zf, yf):
    """Orthonormal basis Q (d x r) of the linear task-discriminative subspace (logreg class directions)."""
    R = LogisticRegression(max_iter=200, C=1.0).fit(Zf, yf)
    T = np.atleast_2d(R.coef_).T                    # (d, n_cls) or (d, 1) for binary
    Q, _ = np.linalg.qr(T)                          # orthonormal columns spanning span(T)
    return Q


def tp_leace_factory(Zf, yf, subjf, n_cls, seed=0):
    """DEPLOYABLE: returns apply(X) = Z_T + LEACE_perp(Z_perp), all fit on the given source subset."""
    subj01, ns = _ids(subjf)
    Q = _task_carrier(Zf, yf)
    Pt = Q @ Q.T                                    # projector onto task carrier
    Zperp = Zf - Zf @ Pt.T
    if ns < 2:
        E_perp = (lambda X: X)
    else:
        E_perp = leace_eraser(Zperp, np.eye(ns)[subj01])
    def apply(X):
        Xt = X @ Pt.T
        return Xt + E_perp(X - Xt)
    return apply


def alpha_leace_factory(alpha):
    """Soft path (round 2, not dry-run): RZ = (1-alpha) Z + alpha LEACE(Z). Returns a factory."""
    def factory(Zf, yf, subjf, n_cls, seed=0):
        subj01, ns = _ids(subjf)
        E = (lambda X: X) if ns < 2 else leace_eraser(Zf, np.eye(ns)[subj01])
        return lambda X: (1.0 - alpha) * X + alpha * E(X)
    return factory
