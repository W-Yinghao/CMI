"""C26 Q4 — is predicted-class-mix a MAIN effect or does it work only through a confidence/margin SCAFFOLD?
Fixed 2-family interaction diagnostics (NOT feature search): predmix-only / confmargin-only / both recovery;
predmix residualized against confmargin (and vice versa); and an exact 2-family Shapley main/interaction
decomposition of the gap-closure value function."""
from __future__ import annotations

import numpy as np

from ..information_ladder import target_unlabeled_features as tuf
from . import artifact_loader, schema


def _predmix_fn(c):
    return {k: float(c[k]) for k in schema.PRED_PROP}


def _confmargin_fn(c):
    return {k: float(c[k]) for k in schema.CONF_MARGIN}


def _recover(joined, rows, mode, raw, oracle, feature_fn):
    table, names = artifact_loader.build_gauge(joined, rows, mode, feature_fn)
    perm = tuf.r3_loto_permutation(rows, table, names, mode, raw, oracle)
    return perm["gap_closed"], perm["survives_permutation"], perm["auc_improve_perm_p"]


def _residualize(joined, rows, mode, target_feats, on_feats):
    """Residualize each `target_feats` per-target gauge column against the `on_feats` gauge columns (least-
    squares across the 9 targets), returning a gauge table of the residual target features."""
    tt, tnames = artifact_loader.build_gauge(joined, rows, mode, lambda c: {k: float(c[k]) for k in target_feats})
    ot, onames = artifact_loader.build_gauge(joined, rows, mode, lambda c: {k: float(c[k]) for k in on_feats})
    targets = sorted(tt)
    Y = np.array([[tt[t]["gauge"][n] for n in tnames] for t in targets])
    Xo = np.array([[ot[t]["gauge"][n] for n in onames] for t in targets])
    Xo = np.column_stack([np.ones(len(targets)), (Xo - Xo.mean(0)) / (Xo.std(0) + 1e-9)])
    resid = {}
    for j, n in enumerate(tnames):
        beta, *_ = np.linalg.lstsq(Xo, Y[:, j], rcond=None)
        r = Y[:, j] - Xo @ beta
        for i, t in enumerate(targets):
            resid.setdefault(t, {})[n + "_resid"] = float(r[i])
    table = {t: {"gauge": resid[t], "offset": tt[t]["offset"]} for t in targets}
    return table, [n + "_resid" for n in tnames]


def interaction_diagnostics(joined, rows, mode, raw, oracle) -> dict:
    v_pm, pm_surv, pm_p = _recover(joined, rows, mode, raw, oracle, _predmix_fn)
    v_cm, cm_surv, cm_p = _recover(joined, rows, mode, raw, oracle, _confmargin_fn)
    v_both, both_surv, both_p = _recover(joined, rows, mode, raw, oracle, lambda c: {**_predmix_fn(c), **_confmargin_fn(c)})
    a = v_pm or 0.0; b = v_cm or 0.0; ab = v_both or 0.0
    main_pm = 0.5 * (a - 0.0) + 0.5 * (ab - b)               # 2-player Shapley main effects
    main_cm = 0.5 * (b - 0.0) + 0.5 * (ab - a)
    interaction = ab - a - b                                 # v(both) - v(pm) - v(cm) + v(empty=0)
    # residualized recovery: does predmix add offset info BEYOND confmargin (and vice versa)?
    pr_table, pr_names = _residualize(joined, rows, mode, schema.PRED_PROP, schema.CONF_MARGIN)
    pm_resid = tuf.r3_loto_permutation(rows, pr_table, pr_names, mode, raw, oracle)
    cr_table, cr_names = _residualize(joined, rows, mode, schema.CONF_MARGIN, schema.PRED_PROP)
    cm_resid = tuf.r3_loto_permutation(rows, cr_table, cr_names, mode, raw, oracle)
    predmix_needs_scaffold = bool((v_pm or -9) < schema.SUCCESS_GAP_CLOSED and (v_both or -9) >= schema.SUCCESS_GAP_CLOSED)
    interaction_dominant = bool(abs(interaction) >= max(abs(main_pm), abs(main_cm)))
    return {"predmix_only_gap": v_pm, "confmargin_only_gap": v_cm, "both_gap": v_both,
            "predmix_survives": pm_surv, "confmargin_survives": cm_surv, "both_survives": both_surv,
            "shapley_main_predmix": main_pm, "shapley_main_confmargin": main_cm, "shapley_interaction": interaction,
            "predmix_residualized_gap": pm_resid["gap_closed"], "predmix_residualized_survives": pm_resid["survives_permutation"],
            "confmargin_residualized_gap": cm_resid["gap_closed"], "confmargin_residualized_survives": cm_resid["survives_permutation"],
            "predmix_needs_confidence_scaffold": predmix_needs_scaffold, "interaction_dominant": interaction_dominant,
            "note": ("predmix alone gap %.3f, both %.3f -> predmix needs a confidence/margin scaffold; "
                     "Shapley interaction %.3f vs mains (pm %.3f, cm %.3f)."
                     % ((v_pm or 0), (v_both or 0), interaction, main_pm, main_cm))}
