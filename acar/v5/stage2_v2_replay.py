"""ACAR V5 Stage-2 v2-REPLAY comparator seam (numpy lazy; acar.features numpy-only; acar.regressor/sklearn lazy). Recomputes the
bit-for-bit v2 recipe on the SAME substrate to produce a subject-macro `v2_replay_red` per disease — the G2 comparator baseline
(`red − v2_replay_red ≥ 0.02`). The recipe: acar.features.feature_vector (11-D = 7 paired + 4 context) as the regressor input,
per-action acar.regressor.ActionRegressor(seed=0) trained on FIT, a one-sided threshold calibrated on CAL, routed on EVAL,
subject-macro `red` (negated mean chosen ΔR).

This comparator READS LABELS — ONLY here, to form ΔR = NLL(f_a) − NLL(f_0). It NEVER alters candidate thresholds, candidate action
choices, or candidate selection; it only supplies `v2_replay_red`. Fail-closed (V2ReplayNotEvaluable). Stage-2B1: SYNTHETIC
fixtures only — never run on the real admitted package until a real Stage-2B authorization. The exact recipe fidelity is
validated at real-run time.
"""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_feature_loader as FL
from acar.v5 import stage2_policy_eval as PE
from acar.v5.stage2_gates import V2ReplayNotEvaluable   # reuse so a real run wired into the engine gets DEV_STOP


def default_regressor_factory():
    """The pinned v2 regressor: acar.regressor.ActionRegressor(seed=0) (HGB≥40 / Ridge≥8 / constant). Lazy sklearn import."""
    from acar.regressor import ActionRegressor
    return ActionRegressor(seed=0)


def _nll(p, y):
    import numpy as np
    p = np.clip(np.asarray(p, float), 1e-12, 1.0)
    return float(-np.log(p[:, int(y)]).mean())


def _batch_feature_and_dr(outputs, source_lda, y):
    """Per-action (11-D v2 feature_vector, ΔR) for one batch. Label-consumed ONLY for ΔR."""
    import acar.features as AF
    import numpy as np
    p0, z0 = outputs["identity"]
    r0 = _nll(p0, y)
    out = {}
    for a in P.ACTIONS:
        pa, z_post = outputs[a]
        phi = AF.paired_features(p0, pa, z0, z_post)                          # lowercase js/bures (v2 order)
        ctx = AF.context_features(source_lda.old_state, z_post, pa)
        fvec = np.asarray(AF.feature_vector(phi, ctx), float)                 # 11-D, NaN→0 by construction
        if not np.isfinite(fvec).all():
            raise V2ReplayNotEvaluable("non-finite v2 feature_vector")
        out[a] = (fvec, _nll(pa, y) - r0)
    return out


def _resolve_label(label_view, sk):
    try:
        return int(label_view.resolve_label(sk))
    except Exception as e:  # noqa: BLE001 — a missing/unauthorized label ⇒ comparator not evaluable
        raise V2ReplayNotEvaluable(f"label not resolvable for {sk!r}: {e}")


def v2_replay_red_by_disease(disease, folds, *, action_provider=AR.production_action_provider, regressor_factory=None,
                             alpha=P.ALPHA, batch_size=PE.STAGE2_BATCH_SIZE, min_batch=PE.STAGE2_MIN_BATCH):
    """Subject-macro `v2_replay_red` for one disease over its OOF folds. Fail-closed (V2ReplayNotEvaluable) if folds/splits/labels
    are missing, a feature_vector is non-finite, or no EVAL subject is routable."""
    import numpy as np
    if not folds:
        raise V2ReplayNotEvaluable(f"{disease}: no folds")
    reg_factory = regressor_factory or default_regressor_factory
    red_terms = []
    for fold in folds:
        by_subject, source_lda, lv = fold["by_subject"], fold["source_lda"], fold["label_view"]
        fit_keys = FL.subjects_in_group(by_subject, "fit")
        cal_keys = FL.subjects_in_group(by_subject, "cal")
        eval_keys = FL.subjects_in_group(by_subject, "eval")
        if not fit_keys or not cal_keys or not eval_keys:
            raise V2ReplayNotEvaluable(f"{disease}: missing FIT/CAL/EVAL split in a fold")

        def _batches(sk):
            Z = np.asarray(by_subject[sk]["embedding"], float)
            y = _resolve_label(lv, sk)
            for zb, forced in PE.window_batches(Z, batch_size, min_batch):
                if forced:
                    yield None, forced
                else:
                    yield _batch_feature_and_dr(
                        AR.subject_action_outputs(zb, source_lda, action_provider=action_provider), source_lda, y), forced

        # FIT: train per-action regressor ĝ_a: feature_vector -> ΔR
        Xa = {a: [] for a in P.ACTIONS}
        dra = {a: [] for a in P.ACTIONS}
        for sk in fit_keys:
            for feats, forced in _batches(sk):
                if forced:
                    continue
                for a in P.ACTIONS:
                    Xa[a].append(feats[a][0])
                    dra[a].append(feats[a][1])
        regs = {}
        for a in P.ACTIONS:
            if not Xa[a]:
                raise V2ReplayNotEvaluable(f"{disease}: no FIT records for action {a}")
            regs[a] = reg_factory().fit(np.asarray(Xa[a], float), np.asarray(dra[a], float))

        # CAL: one-sided threshold = alpha-quantile of the per-batch best predicted ΔR̂ (adapt only if predicted benefit clears it)
        cal_best = []
        for sk in cal_keys:
            for feats, forced in _batches(sk):
                if forced:
                    continue
                cal_best.append(min(float(regs[a].predict(feats[a][0][None])[0]) for a in P.ACTIONS))
        if not cal_best:
            raise V2ReplayNotEvaluable(f"{disease}: no CAL records to calibrate the v2 threshold")
        threshold = float(np.quantile(cal_best, alpha))

        # EVAL: route (adapt the best-predicted action iff its ΔR̂ < threshold) and accumulate subject-macro red
        for sk in eval_keys:
            drs = []
            for feats, forced in _batches(sk):
                if forced:
                    drs.append(0.0)
                    continue
                preds = {a: float(regs[a].predict(feats[a][0][None])[0]) for a in P.ACTIONS}
                best_a = min(preds, key=lambda a: (preds[a], P.ACTION_ORDER[a]))
                drs.append(feats[best_a][1] if preds[best_a] < threshold else 0.0)
            red_terms.append(float(np.mean(drs)) if drs else 0.0)

    if not red_terms:
        raise V2ReplayNotEvaluable(f"{disease}: no EVAL subjects")
    return -float(np.mean(red_terms))


def make_engine_v2_replay_provider(action_provider=AR.production_action_provider, regressor_factory=None):
    """Adapt v2_replay_red_by_disease to the selection engine's v2_replay_provider(disease, ctx) signature (a REAL run injects
    this; the engine default stays fail-closed). Requires BOTH diseases' folds present in ctx['disease_inputs']."""
    def _provider(disease, ctx):
        di = ctx.get("disease_inputs") or {}
        for d in ("PD", "SCZ"):
            if d not in di:
                raise V2ReplayNotEvaluable(f"v2_replay: disease {d} missing from disease_inputs")
        return v2_replay_red_by_disease(disease, di[disease]["folds"], action_provider=action_provider,
                                        regressor_factory=regressor_factory)
    return _provider
