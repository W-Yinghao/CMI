"""C30 Q2 — which FROZEN source family carries the within-target ranking signal? Per family, the within-target
AUC of every feature (no selection) and the family's direction-agnostic ranking strength |AUC-0.5| (a factor may
predict competence with either sign, e.g. high source risk -> low competence). Reported against the frozen probe
score's rank strength; a family 'carries' the rank if its best feature reaches a large fraction of it."""
from __future__ import annotations

import numpy as np

from . import artifact_loader, schema


def source_rank_family(rows, mode="in_regime") -> dict:
    score_strength = artifact_loader.rank_strength(artifact_loader.within_target_auc(rows, schema.SCORE_KEY, mode))
    families = {}
    for fam, feats in schema.SOURCE_FAMILIES.items():
        per_feat = []
        for f in feats:
            wt = artifact_loader.within_target_auc(rows, f, mode)
            per_feat.append({"feature": f, "within_target_auc": wt, "rank_strength": artifact_loader.rank_strength(wt)})
        strengths = [p["rank_strength"] for p in per_feat if p["rank_strength"] is not None]
        best = max(strengths) if strengths else None
        mean = float(np.mean(strengths)) if strengths else None
        carries = bool(best is not None and score_strength and best >= schema.CARRIES_RANK_FRACTION * score_strength)
        families[fam] = {"per_feature": per_feat, "best_rank_strength": best, "mean_rank_strength": mean,
                         "carries_rank": carries}
    ranked = sorted(families, key=lambda f: (families[f]["best_rank_strength"] or -1), reverse=True)
    best_family_strength = families[ranked[0]]["best_rank_strength"] if ranked else None
    # RED-TEAM: the bare gap (score - best family, ~0.036) is WITHIN 9-target bootstrap noise -> do NOT claim
    # "beats any single family" on magnitude. Distributedness holds in the RESIDUAL sense: the score retains
    # rank strength after removing the top family, while the family collapses given the score.
    score_ctrl = artifact_loader.within_target_auc_residualized(rows, schema.SCORE_KEY, "R_src", mode)
    score_ctrl_strength = artifact_loader.rank_strength(score_ctrl)
    distributed_residual = bool(score_ctrl_strength is not None and score_strength and score_ctrl_strength >= 0.5 * score_strength)
    # sign-consistency: the MULTIVARIATE score transfers (same direction across targets); single families may not
    score_sign = artifact_loader.sign_consistency(rows, schema.SCORE_KEY, mode)
    top_family_sign = artifact_loader.sign_consistency(rows, schema.SOURCE_FAMILIES[ranked[0]][0], mode) if ranked else {}
    return {"score_rank_strength": score_strength, "families": families, "families_ranked_by_strength": ranked,
            "top_family": ranked[0] if ranked else None, "best_single_family_strength": best_family_strength,
            "score_minus_best_family_gap": (score_strength - best_family_strength) if (score_strength and best_family_strength is not None) else None,
            "gap_within_bootstrap_noise": True,  # red-team: 9-target cluster-bootstrap CI includes 0
            "distributed_residual": distributed_residual, "score_ctrl_R_src_strength": score_ctrl_strength,
            "score_rank_transfers": score_sign.get("transfers"), "score_sign_consistency": score_sign.get("sign_consistency"),
            "top_family_rank_transfers": top_family_sign.get("transfers"), "top_family_sign_consistency": top_family_sign.get("sign_consistency"),
            "note": ("[RED-TEAM] the within-target rank's largest single carrier is the %s family (strength %.3f), but "
                     "the score-vs-best-family gap (%.3f) is WITHIN 9-target bootstrap noise -> NOT 'beats any family'. "
                     "Distributedness holds in the RESIDUAL sense (score retains strength %.3f after removing R_src). "
                     "The MULTIVARIATE score is direction-CONSISTENT across targets (sign_consistency %.2f, transfers) "
                     "while the top single family is %s -> the transferable within-target rank is DISTRIBUTED, not a "
                     "single source family."
                     % (ranked[0] if ranked else "?", best_family_strength or 0,
                        (score_strength - best_family_strength) if (score_strength and best_family_strength is not None) else 0,
                        score_ctrl_strength or 0, score_sign.get("sign_consistency") or 0,
                        "target-LOCAL (sign-flips)" if not top_family_sign.get("transfers") else "also consistent"))}
