"""C25 Q3 — why do source features DESTROY the R3 recovery in R4 (source + target-unlabeled collapses)? Three
diagnostics: (1) ridge coefficient-norm allocation source-vs-target (do source dims hijack the fit?), (2)
design condition number R3 vs R4, (3) the DECISIVE random-dim control -- add as many RANDOM noise dims as
source features to R3 and refit: if that ALSO collapses the LOTO recovery, the mechanism is small-N / high-dim
noise (source acts as generic nuisance), not source-specific anti-alignment; plus source-vs-target offset-
prediction alignment (do the two gauges point in opposite directions?)."""
from __future__ import annotations

import numpy as np

from ..score_gauge import offset_model
from ..score_gauge.ceiling_ladder import _pooled_auc
from . import schema


def _ridge_weights(X, y, l2):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd; ym = y.mean()
    A = Xs.T @ Xs + l2 * np.eye(Xs.shape[1])
    w = np.linalg.solve(A, Xs.T @ (y - ym))
    return w, float(np.linalg.cond(A))


def _design(gauge_table, names):
    targets = sorted(gauge_table)
    X = np.array([[gauge_table[t]["gauge"][n] for n in names] for t in targets], dtype=np.float64)
    y = np.array([gauge_table[t]["offset"] for t in targets], dtype=np.float64)
    return targets, X, y


def _gap(mr, gauge_table, names, raw, oracle):
    fit = offset_model.fit_offsets(gauge_table, names=list(names))
    oh = fit["offset_hat_loto"]
    auc = _pooled_auc(mr, subtract=lambda r: oh.get(r["target"], 0.0))
    return (((auc - raw) / (oracle - raw)) if (auc is not None and (oracle - raw) > 1e-6) else None), fit["loto_r2"]


def interference_audit(rows, source_gt, source_names, r3_gt, r3_names, mode, raw, oracle) -> dict:
    mr = [r for r in rows if r["mode"] == mode]
    targets = sorted(set(source_gt) & set(r3_gt))
    combined = {t: {"gauge": {**source_gt[t]["gauge"], **r3_gt[t]["gauge"]}, "offset": r3_gt[t]["offset"]}
                for t in targets}
    cnames = list(source_names) + list(r3_names)
    # (1) coefficient-norm allocation (in-sample ridge on the combined design)
    _, X, y = _design(combined, cnames)
    w, cond_r4 = _ridge_weights(X, y, schema.RIDGE_L2)
    ns = len(source_names)
    src_norm = float(np.linalg.norm(w[:ns])); tgt_norm = float(np.linalg.norm(w[ns:]))
    src_share = src_norm / (src_norm + tgt_norm + 1e-12)
    # (2) condition numbers
    _, Xr3, yr3 = _design(r3_gt, r3_names)
    _, cond_r3 = _ridge_weights(Xr3, yr3, schema.RIDGE_L2)
    # (3) random-dim control: add ns random noise dims to R3, refit LOTO gap (avg over trials)
    r3_gap, _ = _gap(mr, r3_gt, r3_names, raw, oracle)
    r4_gap, _ = _gap(mr, combined, cnames, raw, oracle)
    rng = np.random.RandomState(schema.PERM_SEED); rand_gaps = []
    for _t in range(schema.R4_RANDOM_DIM_TRIALS):
        rnames = [f"rand_{i}" for i in range(ns)]
        rmat = rng.randn(len(targets), ns)
        aug = {t: {"gauge": {**r3_gt[t]["gauge"], **{rnames[i]: float(rmat[j, i]) for i in range(ns)}},
                   "offset": r3_gt[t]["offset"]} for j, t in enumerate(targets)}
        g, _ = _gap(mr, aug, list(r3_names) + rnames, raw, oracle)
        if g is not None:
            rand_gaps.append(g)
    rand_mean = float(np.mean(rand_gaps)) if rand_gaps else None
    random_dims_also_collapse = bool(rand_mean is not None and r3_gap is not None and rand_mean < r3_gap - schema.SUCCESS_GAP_CLOSED)
    # (4) source-vs-target offset-prediction alignment (LOTO offset_hat from each gauge, correlated across targets)
    fs = offset_model.fit_offsets(source_gt, names=list(source_names))["offset_hat_loto"]
    ft = offset_model.fit_offsets(r3_gt, names=list(r3_names))["offset_hat_loto"]
    ks = [t for t in targets if t in fs and t in ft]
    a = np.array([fs[t] for t in ks]); b = np.array([ft[t] for t in ks])
    align = float(np.corrcoef(a, b)[0, 1]) if (a.std() > 1e-9 and b.std() > 1e-9) else None
    return {"r3_gap": r3_gap, "r4_gap": r4_gap, "n_source_features": ns, "n_target_features": len(r3_names),
            "source_coef_norm": src_norm, "target_coef_norm": tgt_norm, "source_coef_norm_share": src_share,
            "source_hijacks_ridge": bool(src_share >= schema.R4_SOURCE_COEF_DOMINATION),
            "condition_number_r3": cond_r3, "condition_number_r4": cond_r4,
            "random_dim_control_mean_gap": rand_mean, "random_dims_also_collapse": random_dims_also_collapse,
            "source_target_offset_alignment": align,
            "mechanism": ("small_N_high_dim_noise" if random_dims_also_collapse else
                          ("source_specific_anti_alignment" if (align is not None and align < 0) else
                           "source_nuisance_dominates_ridge" if src_share >= schema.R4_SOURCE_COEF_DOMINATION else
                           "source_dilutes_weak_target_signal")),
            "note": ("R4 collapse reproduced by RANDOM noise dims of the same count -> the mechanism is small-N / "
                     "high-dimensional nuisance: at 9 targets, adding ~%d non-informative dims overfits LOTO and "
                     "drowns the 12-dim target-unlabeled signal; source features are generic nuisance for the "
                     "offset (not a specific anti-aligned direction)." % ns) if random_dims_also_collapse else
                     "R4 collapse is NOT reproduced by random dims -> source-specific interference."}
