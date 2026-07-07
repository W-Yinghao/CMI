"""C24 R6 — oracle ceiling. Reuses the C23 ceiling ladder (target-centered / target-rank oracle + within-target
ceiling) and additionally reports the 0-LABEL transductive within-target mean centering. This surfaces the
information-cut precisely: target-centering needs target GROUPING/identity but NO target labels, so it is a
label-free transductive ceiling -- an oracle only in the LOTO-audit sense, not because it needs labels."""
from __future__ import annotations

from ..score_gauge import ceiling_ladder as clad
from . import schema


def oracle_ladder(rows, mode, offset_hat_loto) -> dict:
    lad = clad.ceiling_ladder(rows, mode, offset_hat_loto)
    raw = lad["raw_pooled"]; oracle = lad["target_centered_oracle"]
    return {
        "raw_pooled": raw,
        "regime_centered": lad["regime_centered"],
        "source_gauge_loto": lad["source_gauge_loto"],           # R1 recovery (C23; hurts)
        "target_centered_oracle": oracle,                        # R6: needs grouping, NO labels
        "target_rank_oracle": lad["target_rank_oracle"],
        "within_target_ceiling": lad["within_target_ceiling"],
        "oracle_gap_over_raw": (oracle - raw) if (oracle is not None and raw is not None) else None,
        "note": ("target-centering uses target GROUPING (identity) but no target labels -> a 0-label "
                 "transductive ceiling; it is 'oracle' in the leave-one-target-out audit sense only."),
    }
