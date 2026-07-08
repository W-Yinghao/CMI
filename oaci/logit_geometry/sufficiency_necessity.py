"""C27-C — sufficiency / necessity by FROZEN factor combo. For each pre-declared factor union, report offset
recovery (gap, permutation survival) AND target-identity predictability together -- because C26 proved the two
are entangled, they must be read jointly, not separately."""
from __future__ import annotations

from . import artifact_loader, factor_registry, identity_controls, schema

COMBOS = (
    ("occupancy",),
    ("global_confidence",),
    ("class_conditioned_confidence",),
    ("class_bias",),
    ("occ_x_conf_interaction",),
    ("occupancy", "global_confidence"),                     # full R3 baseline
    ("occupancy", "class_conditioned_confidence"),
    ("occupancy", "class_conditioned_margin"),
)


def sufficiency_necessity(logit_cands, score_rows, mode, raw, oracle) -> list:
    out = []
    for combo in COMBOS:
        r = artifact_loader.recover(logit_cands, score_rows, mode, raw, oracle, factor_registry.select(*combo))
        out.append({"combo": "+".join(combo), "n_features": len(factor_registry.family_feature_names(*combo)),
                    "gap_closed": r["gap_closed"], "perm_p": r["perm_p"], "survives_permutation": r["survives_permutation"],
                    "target_id_accuracy": identity_controls.id_accuracy(logit_cands, *combo)})
    return out
