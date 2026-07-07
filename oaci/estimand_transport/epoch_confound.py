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


def _partial_score_label_given_epoch(rows):
    """Mean over targets of partial Spearman(score, label | epoch). ~0 => the score adds nothing beyond epoch."""
    vals = []
    for t in sorted({r["target"] for r in rows}):
        g = [r for r in rows if r["target"] == t and r.get("epoch") is not None]
        if len(g) < 5:
            continue
        s = [r["score"] for r in g]; y = [r["label"] for r in g]; e = [r["epoch"] for r in g]
        r_sy, r_se, r_ye = _spearman(s, y), _spearman(s, e), _spearman(y, e)
        if None in (r_sy, r_se, r_ye):
            continue
        denom = ((1 - r_se ** 2) * (1 - r_ye ** 2)) ** 0.5
        if denom > 1e-9:
            vals.append((r_sy - r_se * r_ye) / denom)
    return (float(np.mean(vals)) if vals else None), len(vals)


def epoch_confound(rows) -> dict:
    """On the IN-REGIME rows (where the C19 signal lives): probe vs epoch-family baselines + residual."""
    inr = [r for r in rows if r["mode"] == "in_regime"]
    probe_auc, probe_str = _within_target_auc(inr, "score")
    baselines = {b: _within_target_auc(inr, b) for b in schema.EPOCH_BASELINES}
    base_str = {b: baselines[b][1] for b in baselines}
    best_base = max((v for v in base_str.values() if v is not None), default=None)
    partial, n_t = _partial_score_label_given_epoch(inr)
    beats_epoch = bool(probe_str is not None and best_base is not None and probe_str > best_base + 0.01)
    residual_present = bool(partial is not None and abs(partial) >= 0.10)
    return {"probe_within_target_auc": probe_auc, "probe_within_target_strength": probe_str,
            "baseline_within_target_strength": base_str, "best_epoch_family_baseline_strength": best_base,
            "probe_beats_epoch_family": beats_epoch,
            "partial_spearman_score_label_given_epoch": partial, "n_targets_partial": n_t,
            "residual_signal_present": residual_present,
            "epoch_confounded": bool(not beats_epoch or not residual_present),
            "note": ("If the probe does not beat the epoch/order/training-log baselines OR the residual "
                     "(partial Spearman controlling epoch) vanishes, the within-target signal is a trajectory-"
                     "position / training-log proxy, not source competence -> taxonomy T2. Reported BEFORE any "
                     "normalization-rescue interpretation.")}
