"""C20-A — cross-regime leave-one-target-out validation of the frozen C19 robust-core probe. For each held-out
target: fit the fixed probe on the DEVELOPMENT regimes (S0/S2/S3) from the OTHER targets, evaluate on the
held-out target in a HELD-OUT regime (S4/S5/S6/S7). Tests target generalization + regime generalization +
support-deletion robustness simultaneously. Success = pooled held-out AUC beats the cross-regime permutation
null (p<0.05) AND margin vs STRICT chance 0.5 >= 0.03. Target labels are diagnostic-only; no selector."""
from __future__ import annotations

import numpy as np

from ..competence_probe import schema as c19
from . import frozen_probe, permutation, schema


def _pool(dev_by_regime) -> list:
    return [r for regime in schema.DEVELOPMENT_REGIMES for r in dev_by_regime[regime]]


def cross_regime_loto(dev_by_regime, val_rows, cols, *, n_perm=schema.N_PERM, perm_seed=schema.PERM_SEED) -> dict:
    label = c19.DIAGNOSTIC_LABEL
    dev_pool = _pool(dev_by_regime)
    targets = sorted({r["target"] for r in dev_pool})
    scores, ys, per_target, n_test_by_t = [], [], {}, {}
    for t in targets:
        train = [r for r in dev_pool if r["target"] != t]
        test = [r for r in val_rows if r["target"] == t]
        s, y, ntr, nte = frozen_probe.fit_predict(train, test, cols)
        n_test_by_t[t] = int(nte)
        if s is not None:
            scores.extend(s.tolist()); ys.extend(y.tolist())
            per_target[str(t)] = frozen_probe.auc(y, s)
    obs = frozen_probe.auc(ys, scores)
    if obs is None:
        return {"loto_auc": None, "n_used": len(ys), "per_target_auc": per_target, "permutation_p": None,
                "permutation_mean_auc": None, "beats_permutation": False, "meets_margin_vs_chance": False,
                "passes": False, "note": "no scorable held-out candidates"}
    null = permutation.cross_regime_null(dev_pool, val_rows, cols, targets, n_perm=n_perm, perm_seed=perm_seed,
                                         label=label)
    p = float((np.sum(null >= obs) + 1) / (len(null) + 1)) if len(null) else None
    pm = float(null.mean()) if len(null) else None
    beats = bool(p is not None and p < schema.SUCCESS_P)
    margin = bool(obs - 0.5 >= schema.SUCCESS_AUC_MARGIN_VS_CHANCE)
    pt_vals = [v for v in per_target.values() if v is not None]
    spread = (max(pt_vals) - min(pt_vals)) if len(pt_vals) >= 2 else None
    return {"loto_auc": obs, "n_used": len(ys), "n_test_by_target": n_test_by_t, "per_target_auc": per_target,
            "per_target_spread": spread, "permutation_mean_auc": pm, "permutation_p": p,
            "margin_vs_chance": float(obs - 0.5), "beats_permutation": beats, "meets_margin_vs_chance": margin,
            "passes": bool(beats and margin), "non_deployable": schema.NON_DEPLOYABLE,
            "note": "DIAGNOSTIC-ONLY cross-regime LOTO of the frozen C19 probe; no selector emitted."}
