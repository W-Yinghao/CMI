"""C30 Q3 — does the source rank factor track SOURCE error geometry more than TARGET competence? R_src is the
source risk (a source error proxy); the source calibration/logit features are source-side quantities. We report
their within-target ranking of the TARGET competence label (rank axis) against their identity as source-error
quantities. If they rank target competence only weakly while being strong source-error quantities, the source-
visible competence is a weak shared-trajectory-quality signal, not a calibrated target competence score."""
from __future__ import annotations

import numpy as np

from . import artifact_loader, schema


def source_error_alignment(rows, mode="in_regime") -> dict:
    mr = [r for r in rows if r["mode"] == mode]
    # source risk (R_src) is a source-error quantity; how strongly does it rank TARGET competence within target?
    rsrc_target_rank = artifact_loader.rank_strength(artifact_loader.within_target_auc(rows, "R_src", mode))
    score_rank = artifact_loader.rank_strength(artifact_loader.within_target_auc(rows, schema.SCORE_KEY, mode))
    # correlation of R_src with the source NLL (source error) vs with the target competence label, within target
    def _classwise_corr(a_key, b_key):
        per = []
        for t in sorted({r["target"] for r in mr}):
            g = [r for r in mr if r["target"] == t]
            a = np.array([artifact_loader._val(r, a_key) for r in g], float)
            b = np.array([float(r[b_key]) if b_key == schema.LABEL_KEY else artifact_loader._val(r, b_key) for r in g], float)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 3 and a[ok].std() > 1e-9 and b[ok].std() > 1e-9:
                per.append(float(np.corrcoef(a[ok], b[ok])[0, 1]))
        return float(np.mean(per)) if per else None
    rsrc_vs_source_nll = _classwise_corr("R_src", "feat__source_guard_nll")
    rsrc_vs_target_label = _classwise_corr("R_src", schema.LABEL_KEY)
    # RED-TEAM: "tracks source error" is a TAUTOLOGY (R_src IS the source NLL/CE risk). Show it: residualizing
    # R_src on the source NLL leaves ~chance target-competence rank -> R_src has no target content beyond source risk.
    tautological = bool(rsrc_vs_source_nll is not None and abs(rsrc_vs_source_nll) >= 0.90)
    rsrc_ctrl_nll = artifact_loader.within_target_auc_residualized(rows, "R_src", "feat__source_guard_nll", mode)
    rsrc_ctrl_nll_strength = artifact_loader.rank_strength(rsrc_ctrl_nll)
    # RED-TEAM: does the R_src within-target rank TRANSFER? Per-target AUCs SIGN-FLIP -> the 0.124 strength masks it.
    sc = artifact_loader.sign_consistency(rows, "R_src", mode)
    tracks_source_error = bool(rsrc_vs_source_nll is not None and abs(rsrc_vs_source_nll) >= 0.30)
    return {"R_src_target_competence_rank_strength": rsrc_target_rank, "score_rank_strength": score_rank,
            "R_src_vs_source_nll_corr": rsrc_vs_source_nll, "R_src_vs_target_label_corr": rsrc_vs_target_label,
            "tracks_source_error": tracks_source_error, "tautological_source_error_identity": tautological,
            "R_src_ctrl_source_nll_strength": rsrc_ctrl_nll_strength,
            "R_src_sign_consistency": sc["sign_consistency"], "R_src_rank_transfers": sc["transfers"],
            "R_src_per_target_auc": sc["per_target_auc"], "R_src_n_above_half": sc["n_above_half"], "R_src_n_targets": sc["n_targets"],
            "rank_tracks_source_error_only": bool(tautological),
            "note": ("[RED-TEAM] 'tracks source error' is a TAUTOLOGY: R_src IS the source NLL/CE risk (corr %.3f); "
                     "residualizing R_src on the source NLL leaves strength %.3f (~chance) -> R_src has no target-"
                     "competence content beyond source risk. And the R_src within-target rank does NOT transfer: "
                     "per-target AUC SIGN-FLIPS (%s/%s targets on the majority side; sign_consistency %.2f), so the "
                     "0.124 mean strength MASKS a target-LOCAL, non-transferable signal. What survives: R_src is "
                     "NOT a calibrated/deployable target-competence score."
                     % (rsrc_vs_source_nll or 0, rsrc_ctrl_nll_strength or 0, max(sc["n_above_half"] or 0, (sc["n_targets"] or 0) - (sc["n_above_half"] or 0)),
                        sc["n_targets"], sc["sign_consistency"] or 0) if tautological else
                     "source risk does not cleanly track source error geometry here")}
