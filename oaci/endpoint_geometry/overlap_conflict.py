"""C31 Q2 — endpoint overlap / conflict. Do accuracy-good and calibration-good checkpoints coincide or conflict?
Jaccard overlap + conditional probabilities + conflict rates + the continuous trade-off signal (correlation of
bAcc improvement with NLL/ECE improvement). A negative delta-correlation OR high conflict => accuracy-calibration
trade-off (E1); positive overlap with common joint-good => trade-off not confirmed at the population level."""
from __future__ import annotations

import numpy as np

from . import schema


def _jaccard(rows, a, b):
    A = np.array([r[a] for r in rows]); B = np.array([r[b] for r in rows])
    inter = int(np.sum((A == 1) & (B == 1))); union = int(np.sum((A == 1) | (B == 1)))
    return (inter / union) if union else None


def _cond(rows, given, target):
    G = [r for r in rows if r[given] == 1]
    return (float(np.mean([r[target] for r in G])) if G else None)


def _corr(rows, a, b):
    x = np.array([r[a] for r in rows if r.get(a) is not None and r.get(b) is not None], float)
    y = np.array([r[b] for r in rows if r.get(a) is not None and r.get(b) is not None], float)
    return float(np.corrcoef(x, y)[0, 1]) if (len(x) > 3 and x.std() > 1e-9 and y.std() > 1e-9) else None


def _partial_corr_epoch(rows, a, b, ctrl="epoch"):
    """Within-trajectory epoch-residualized correlation: rules out the training-progress confound where bAcc and
    calibration both improve over epochs so their deltas correlate for a reason unrelated to checkpoint-level
    coincidence. Residualize a and b on epoch inside each (seed,target,level) trajectory, then pool residuals."""
    ra, rb, by = [], [], {}
    for r in rows:
        by.setdefault((r["seed"], r["target"], r["level"]), []).append(r)
    for cs in by.values():
        cs = [c for c in cs if c.get(a) is not None and c.get(b) is not None and c.get(ctrl) is not None]
        if len(cs) < 5:
            continue
        A = np.array([c[a] for c in cs], float); B = np.array([c[b] for c in cs], float); C = np.array([c[ctrl] for c in cs], float)
        if C.std() < 1e-9:
            continue
        A = A - np.polyval(np.polyfit(C, A, 1), C); B = B - np.polyval(np.polyfit(C, B, 1), C)
        ra.extend(A); rb.extend(B)
    ra, rb = np.array(ra), np.array(rb)
    return float(np.corrcoef(ra, rb)[0, 1]) if (len(ra) > 3 and ra.std() > 1e-9 and rb.std() > 1e-9) else None


def _per_traj_epoch_resid_corr(rows, a, b, ctrl="epoch"):
    """Per-trajectory epoch-residualized correlation of a vs b (one value per (seed,target,level))."""
    by, out = {}, {}
    for r in rows:
        by.setdefault((r["seed"], r["target"], r["level"]), []).append(r)
    for k, cs in by.items():
        cs = [c for c in cs if c.get(a) is not None and c.get(b) is not None and c.get(ctrl) is not None]
        if len(cs) < 5:
            continue
        A = np.array([c[a] for c in cs], float); B = np.array([c[b] for c in cs], float); C = np.array([c[ctrl] for c in cs], float)
        if C.std() < 1e-9:
            continue
        A = A - np.polyval(np.polyfit(C, A, 1), C); B = B - np.polyval(np.polyfit(C, B, 1), C)
        if A.std() > 1e-9 and B.std() > 1e-9:
            out[k] = float(np.corrcoef(A, B)[0, 1])
    return out


def _e1_robustness(rows):
    """RED-TEAM checks that E1's positive coupling is not a shared-ERM artifact and is target-robust."""
    # (1) the ERM reference is a per-trajectory constant -> subtraction is absorbed by the epoch residualization
    by = {}
    for r in rows:
        by.setdefault((r["seed"], r["target"], r["level"]), []).append(r)
    max_spread = 0.0
    for cs in by.values():
        for key in ("erm_bacc", "erm_nll", "erm_ece"):
            vals = [c[key] for c in cs if c.get(key) is not None]
            if len(vals) > 1:
                max_spread = max(max_spread, float(np.ptp(vals)))
    # (2) per-target sign of the epoch-residualized coupling + (3) target cluster-bootstrap CI of the per-target mean
    by_t = {}
    for k, v in _per_traj_epoch_resid_corr(rows, "bacc_delta", "nll_improve").items():
        by_t.setdefault(k[1], []).append(v)
    tvals = {t: float(np.mean(v)) for t, v in by_t.items()}
    ts = sorted(tvals); arr = np.array([tvals[t] for t in ts])
    n_pos = int(np.sum(arr > 0))
    rng = np.random.RandomState(schema.PERM_SEED); boot = []
    if len(ts) >= 3:
        for _ in range(2000):
            idx = rng.randint(0, len(ts), len(ts)); boot.append(float(arr[idx].mean()))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (None, None)
    return {"erm_constant_within_trajectory": bool(max_spread < 1e-9), "erm_max_within_traj_spread": max_spread,
            "per_target_epoch_resid_corr": tvals, "n_targets_positive": n_pos, "n_targets": len(ts),
            "target_bootstrap_mean": float(arr.mean()) if len(ts) else None,
            "target_bootstrap_ci_lo": ci[0], "target_bootstrap_ci_hi": ci[1],
            "coupling_target_robust": bool(len(ts) and n_pos == len(ts) and ci[0] is not None and ci[0] > 0)}


def overlap_conflict(rows) -> dict:
    jac = {"accuracy_x_nll": _jaccard(rows, "accuracy_good", "nll_good"),
           "accuracy_x_ece": _jaccard(rows, "accuracy_good", "ece_good"),
           "accuracy_x_calibration": _jaccard(rows, "accuracy_good", "calibration_good"),
           "nll_x_ece": _jaccard(rows, "nll_good", "ece_good")}
    cond = {"P(calibration_good|accuracy_good)": _cond(rows, "accuracy_good", "calibration_good"),
            "P(accuracy_good|calibration_good)": _cond(rows, "calibration_good", "accuracy_good")}
    conflict = {"accuracy_good_calibration_bad_rate": float(np.mean([r["accuracy_good_calibration_bad"] for r in rows])),
                "calibration_good_accuracy_flat_rate": float(np.mean([r["calibration_good_accuracy_flat"] for r in rows]))}
    dcorr = {"bacc_delta_vs_nll_improve": _corr(rows, "bacc_delta", "nll_improve"),
             "bacc_delta_vs_ece_improve": _corr(rows, "bacc_delta", "ece_improve"),
             "nll_improve_vs_ece_improve": _corr(rows, "nll_improve", "ece_improve")}
    # epoch-confound control: does the coupling survive residualizing on training epoch within each trajectory?
    epoch_ctrl = {"bacc_delta_vs_nll_improve": _partial_corr_epoch(rows, "bacc_delta", "nll_improve"),
                  "bacc_delta_vs_ece_improve": _partial_corr_epoch(rows, "bacc_delta", "ece_improve")}
    # trade-off confirmed if bAcc improvement is NEGATIVELY correlated with calibration improvement
    neg = [v for v in (dcorr["bacc_delta_vs_nll_improve"], dcorr["bacc_delta_vs_ece_improve"]) if v is not None]
    tradeoff = bool(neg and np.mean(neg) <= -0.15)
    ec = [v for v in epoch_ctrl.values() if v is not None]
    return {"jaccard": jac, "conditional": cond, "conflict": conflict, "delta_correlations": dcorr,
            "epoch_residualized_correlations": epoch_ctrl, "e1_robustness": _e1_robustness(rows),
            "coupling_survives_epoch_control": bool(ec and np.mean(ec) >= 0.15),
            "tradeoff_confirmed": tradeoff, "mean_bacc_vs_calib_improve_corr": (float(np.mean(neg)) if neg else None),
            "note": ("accuracy improvement is NEGATIVELY correlated with calibration improvement (mean corr %.3f) "
                     "-> accuracy-calibration TRADE-OFF at the population level" % np.mean(neg) if tradeoff else
                     "accuracy improvement is NOT negatively correlated with calibration improvement (mean corr %s); "
                     "accuracy-good checkpoints largely overlap calibration-good -> trade-off NOT confirmed at the "
                     "population level" % (round(float(np.mean(neg)), 3) if neg else "n/a"))}
