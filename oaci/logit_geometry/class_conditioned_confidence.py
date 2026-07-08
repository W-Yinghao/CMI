"""C27-A — class-conditioned confidence decomposition. C26 showed occupancy and global confidence each fail
alone but recover together (synergy). The most natural factorization of that synergy is CLASS-CONDITIONED
confidence: how confidently the model occupies each predicted class. Does occupancy + class-conditioned
confidence (or the occupancy x confidence interaction terms) reproduce the full-R3 recovery?"""
from __future__ import annotations

from . import artifact_loader, factor_registry, schema


def class_conditioned_decomposition(logit_cands, score_rows, mode, raw, oracle) -> dict:
    def rec(*fams):
        return artifact_loader.recover(logit_cands, score_rows, mode, raw, oracle, factor_registry.select(*fams))
    full = rec("occupancy", "global_confidence")                              # C24/C26 full-R3 baseline
    occ_cc = rec("occupancy", "class_conditioned_confidence")
    occ_ccm = rec("occupancy", "class_conditioned_confidence", "class_conditioned_margin")
    occ_x = rec("occ_x_conf_interaction")                                     # occupancy x confidence product alone
    fg = full["gap_closed"]
    cc_explains = bool(occ_cc["gap_closed"] is not None and fg is not None
                       and occ_cc["gap_closed"] >= 0.5 * fg and occ_cc["survives_permutation"])
    interaction_term_carries = bool(occ_x["gap_closed"] is not None and fg is not None
                                    and occ_x["gap_closed"] >= 0.5 * fg and occ_x["survives_permutation"])
    return {"full_r3": full, "occupancy_plus_classcond_confidence": occ_cc,
            "occupancy_plus_classcond_conf_margin": occ_ccm, "occ_x_conf_interaction_only": occ_x,
            "class_conditioned_confidence_explains": cc_explains, "interaction_term_carries": interaction_term_carries,
            "note": ("class-conditioned confidence (occupancy + conf_k) reproduces >=50%% of the full-R3 recovery "
                     "and survives permutation -> the interaction is class-conditioned confidence (how confidently "
                     "the model occupies each class)" if cc_explains else
                     "class-conditioned confidence does NOT reproduce the full-R3 recovery -> the interaction is "
                     "not a simple class-conditioned confidence factorization")}
