"""CMI-Trace E3 (Proposition 2) — K*_subj on beneficial vs legitimate-use deployment worlds.

Under squared loss the incremental gain of an eraser T is EXACT:
    Gain*(T) = E[delta_T^2] (1 - K*),   K* = 2 E[r_T delta_T] / E[delta_T^2],
with delta_T = h(Z) - h_T(Z),  r_T = Y - h_T(Z). Removal helps iff K* < 1. CMI amount does NOT enter the sign.

Two worlds share SOURCE data, target-X, and the candidate subspace T (removes the Z_spur coordinates of the
spurious-task DGP); they differ ONLY in whether the source shortcut relation holds at deployment:
  * beneficial     — target spurious sign FLIPPED (shortcut breaks at deployment)  -> expect K*<1, Gain*>0
  * legitimate-use — target spurious sign PRESERVED (relation holds at deployment) -> expect K*>1, Gain*<0
Heads h (full) and h_T (through the eraser) are fit on the SHARED source and reused on both worlds.

The identity is algebraically exact, so |Gain* - Gain_direct| must be ~0 (QC gate); a mismatch is a bug.
Reuses tos_cmi.data.spurious_task_dgp.make_spurious_task_dgp.
"""
from __future__ import annotations
import numpy as np

from tos_cmi.data.spurious_task_dgp import make_spurious_task_dgp

EPS = 1e-12


def _ysign(y, n_cls):
    """Map integer labels to a squared-loss regression target in [-1,1]."""
    return (2.0 * np.asarray(y, float) - (n_cls - 1)) / max(1, n_cls - 1)


def _ridge_fit(X, t, alpha=1e-3):
    """Closed-form ridge readout t ~ X w + c. Returns predict(X') = X' w + c."""
    X = np.asarray(X, float); t = np.asarray(t, float)
    Xa = np.concatenate([X, np.ones((len(X), 1))], 1)
    A = Xa.T @ Xa + alpha * np.eye(Xa.shape[1]); A[-1, -1] = A[-1, -1] - alpha  # do not penalise the bias
    w = np.linalg.solve(A, Xa.T @ t)
    return lambda Xp: np.concatenate([np.asarray(Xp, float), np.ones((len(Xp), 1))], 1) @ w


def make_two_worlds(spur_strength=3.0, seed=0, **dgp_kwargs):
    """Build the beneficial + legitimate-use worlds from one shared DGP draw. The legitimate world is the
    beneficial one with the TARGET spur block sign-flipped back to the source sign (shares source + target-X).
    Returns dict with 'source' (shared) and 'beneficial'/'legitimate' target deployment blocks + block idx."""
    g = make_spurious_task_dgp(spur_strength=spur_strength, seed=seed, **dgp_kwargs)
    Z, y, d = g["Z"], g["y"], g["d"]
    spur = g["spur"]; tdom = g["target_dom"]; n_cls = g["n_cls"]
    s_target = float(g["spurious_sign"][tdom])                 # -1 in the beneficial (flipped) world
    tmask = d == tdom; smask = ~tmask
    ys = _ysign(y, n_cls)

    Z_benef = Z.copy()
    Z_legit = Z.copy()
    # legitimate = beneficial + spur_strength*(1 - s_target)*ysign on the target spur columns (flips s to +1)
    add = spur_strength * (1.0 - s_target) * ys[tmask]
    for j in spur:
        Z_legit[np.where(tmask)[0], j] = Z_benef[tmask][:, j] + add

    src = {"Z": Z[smask], "y": y[smask], "ysign": ys[smask], "d": d[smask]}
    return {
        "source": src, "spur": spur, "target_dom": tdom, "n_cls": n_cls, "spur_strength": spur_strength,
        "beneficial": {"Z": Z_benef[tmask], "y": y[tmask], "ysign": ys[tmask]},
        "legitimate": {"Z": Z_legit[tmask], "y": y[tmask], "ysign": ys[tmask]},
    }


def eraser_projector(dz, drop_cols):
    """(I - P_T): axis-aligned projector that ZEROES the drop_cols (the Z_spur coordinates)."""
    keep = np.ones(dz); keep[list(drop_cols)] = 0.0
    return np.diag(keep)


def kstar_on_world(h, h_T, Z_dep, ysign_dep):
    """Compute (E[delta^2], E[r*delta], K*, Gain*, Gain_direct, identity_residual) on a deployment set with
    the SHARED source-fit heads. Identity Gain* == Gain_direct is exact for squared loss."""
    hZ = h(Z_dep); hTZ = h_T(Z_dep)
    delta = hZ - hTZ
    r = ysign_dep - hTZ
    Ed2 = float(np.mean(delta ** 2))
    Erd = float(np.mean(r * delta))
    Kstar = 2.0 * Erd / max(Ed2, EPS) if Ed2 > EPS else float("nan")
    gain_star = Ed2 * (1.0 - Kstar) if Ed2 > EPS else 0.0
    risk_h = float(np.mean((ysign_dep - hZ) ** 2))
    risk_hT = float(np.mean((ysign_dep - hTZ) ** 2))
    gain_direct = risk_h - risk_hT                            # risk reduction from removing T
    return {
        "E_delta2": Ed2, "E_r_delta": Erd, "K_star": Kstar, "gain_star": gain_star,
        "gain_direct": gain_direct, "identity_residual": abs(gain_star - gain_direct),
        "risk_full": risk_h, "risk_erased": risk_hT,
        "helps": bool(gain_direct > 0), "K_star_lt_1": bool(Kstar < 1.0) if Kstar == Kstar else None,
    }


def run_worlds(spur_strength=3.0, seed=0, ridge_alpha=1e-3, identity_tol=1e-8, **dgp_kwargs):
    """Full E3: build both worlds, fit shared source heads (h full, h_T through the eraser), evaluate the K*
    identity on each deployment world, and check the exact-identity QC gate on BOTH."""
    W = make_two_worlds(spur_strength=spur_strength, seed=seed, **dgp_kwargs)
    src = W["source"]; dz = src["Z"].shape[1]
    ImP = eraser_projector(dz, W["spur"])                     # I - P_T
    h = _ridge_fit(src["Z"], src["ysign"], alpha=ridge_alpha)
    h_T = _ridge_fit(src["Z"] @ ImP, src["ysign"], alpha=ridge_alpha)
    h_full = lambda Z: h(Z)
    h_eras = lambda Z: h_T(Z @ ImP)

    out = {"seed": int(seed), "spur_strength": float(spur_strength), "target_dom": int(W["target_dom"]),
           "spur_cols": list(W["spur"]), "d_z": int(dz), "n_cls": int(W["n_cls"])}
    for world in ("beneficial", "legitimate"):
        b = W[world]
        out[world] = kstar_on_world(h_full, h_eras, b["Z"], b["ysign"])
    out["identity_ok"] = bool(max(out["beneficial"]["identity_residual"],
                                  out["legitimate"]["identity_residual"]) <= identity_tol)
    # semantic separation (the Prop-2 prediction): beneficial K*<1 & helps; legitimate K*>1 & hurts
    out["worlds_separate"] = bool(out["beneficial"]["K_star"] < 1.0 < out["legitimate"]["K_star"]
                                  and out["beneficial"]["gain_direct"] > 0 > out["legitimate"]["gain_direct"])
    return out
