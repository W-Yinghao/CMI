"""C22 Q2 (CRITICAL, reported BEFORE any rescue claim) — is the C19/C20 within-target signal really source
COMPETENCE, or a trajectory-position (epoch / candidate-order) or training-log proxy? Compares the frozen-probe
within-target AUC against epoch / order / train_surrogate / R_src baselines, and measures the RESIDUAL signal
(partial Spearman of probe-score vs label CONTROLLING epoch). If the probe does not beat the epoch baseline or
the residual vanishes, the C19 positive must be downgraded to a trajectory-position diagnostic (T2)."""
from __future__ import annotations

import numpy as np

from ..identifiability.multivariate_probe import _auc
from . import schema


def _within_target_auc(rows, value_key):
    """Mean per-target AUC using rows[value_key] as the score (oriented: report as-is; strength = |auc-0.5|)."""
    per = []
    for t in sorted({r["target"] for r in rows}):
        g = [r for r in rows if r["target"] == t and r.get(value_key) is not None]
        y = np.array([r["label"] for r in g]); v = np.array([r[value_key] for r in g], dtype=float)
        if 0 < y.sum() < len(y) and len(y) >= 4:
            per.append(_auc(y, v))
    per = [p for p in per if p is not None]
    if not per:
        return None, None
    # orient each per-target auc to its informative side, then average the STRENGTH
    strength = float(np.mean([max(p, 1 - p) for p in per]))
    return float(np.mean(per)), strength


def _rank(v):
    order = np.argsort(np.argsort(v)); return order.astype(float)


def _spearman(a, b):
    if len(a) < 4:
        return None
    ra, rb = _rank(np.asarray(a, float)), _rank(np.asarray(b, float))
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _partial(rows, control_key):
    """Mean over targets of partial Spearman(score, label | control_key). ~0 => score adds nothing beyond it."""
    vals = []
    for t in sorted({r["target"] for r in rows}):
        g = [r for r in rows if r["target"] == t and r.get(control_key) is not None]
        if len(g) < 5:
            continue
        s = [r["score"] for r in g]; y = [r["label"] for r in g]; c = [r[control_key] for r in g]
        r_sy, r_sc, r_yc = _spearman(s, y), _spearman(s, c), _spearman(y, c)
        if None in (r_sy, r_sc, r_yc):
            continue
        denom = ((1 - r_sc ** 2) * (1 - r_yc ** 2)) ** 0.5
        if denom > 1e-9:
            vals.append((r_sy - r_sc * r_yc) / denom)
    return (float(np.mean(vals)) if vals else None), len(vals)


def epoch_confound(rows) -> dict:
    """On the IN-REGIME rows (where the C19 signal lives). T2 gate = TRAJECTORY (epoch/order) ONLY: the probe
    must beat the trajectory baselines AND retain a residual controlling epoch. Training-log baselines (R_src,
    train_surrogate) are a SEPARATE source-observable overlap check -- they do NOT trigger a trajectory downgrade."""
    inr = [r for r in rows if r["mode"] == "in_regime"]
    probe_auc, probe_str = _within_target_auc(inr, "score")
    base_str = {b: _within_target_auc(inr, b)[1] for b in schema.EPOCH_BASELINES}
    traj_str = {b: base_str[b] for b in schema.TRAJECTORY_BASELINES if base_str.get(b) is not None}
    log_str = {b: base_str[b] for b in schema.TRAINING_LOG_BASELINES if base_str.get(b) is not None}
    best_traj = max(traj_str.values(), default=None)
    best_log = max(log_str.values(), default=None)
    partial_epoch, n_t = _partial(inr, "epoch")
    partial_rsrc, _ = _partial(inr, "R_src")
    beats_trajectory = bool(probe_str is not None and best_traj is not None and probe_str > best_traj + 0.01)
    residual_vs_epoch = bool(partial_epoch is not None and abs(partial_epoch) >= 0.10)
    # T2 (trajectory-position confound) ONLY:
    epoch_confounded = bool(not beats_trajectory or not residual_vs_epoch)
    # separate source-observable overlap (NOT a trajectory downgrade):
    source_risk_overlap = bool(probe_str is not None and best_log is not None and probe_str <= best_log + 0.01)
    probe_adds_over_source_risk = bool(partial_rsrc is not None and abs(partial_rsrc) >= 0.10)
    return {"probe_within_target_auc": probe_auc, "probe_within_target_strength": probe_str,
            "baseline_within_target_strength": base_str, "trajectory_baseline_strength": traj_str,
            "training_log_baseline_strength": log_str, "best_trajectory_baseline_strength": best_traj,
            "best_training_log_baseline_strength": best_log, "probe_beats_trajectory": beats_trajectory,
            "partial_spearman_score_label_given_epoch": partial_epoch, "n_targets_partial": n_t,
            "residual_signal_present": residual_vs_epoch, "epoch_confounded": epoch_confounded,
            "partial_spearman_score_label_given_R_src": partial_rsrc,
            "source_risk_overlap": source_risk_overlap, "probe_adds_over_source_risk": probe_adds_over_source_risk,
            "note": ("T2 (trajectory confound) fires ONLY if the probe fails to beat the epoch/order baselines OR "
                     "the residual controlling epoch vanishes. Training-log baselines (R_src/train_surrogate) are "
                     "a SEPARATE source-observable overlap: source_risk_overlap means a single source-risk scalar "
                     "matches the probe (a low-dimensional / risk-family finding, echoing C17), NOT a trajectory "
                     "proxy. Reported BEFORE any normalization-rescue interpretation.")}
