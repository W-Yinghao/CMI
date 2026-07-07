"""C25 Q4 — what PROBLEM CLASS is the 0-label target-grouping oracle? Lays the information-assumption ladder
explicitly so the target-centered oracle is neither mistaken for source-only DG (it uses target grouping +ing
the held-out target's OWN candidate scores) nor for a target-LABEL oracle (it uses no labels). Quantifies the
value of GROUPING beyond target-unlabeled marginal geometry (R6 gap - R3 gap)."""
from __future__ import annotations

from . import schema


def grouping_boundary(r1_gap, r3_gap, r6_gap, r5_refine_gap, within_ceiling) -> dict:
    recovers = {"source_only_DG": r1_gap, "target_unlabeled_transductive": r3_gap,
                "target_grouped_transductive_zero_label": r6_gap, "few_label_target_calibration": r5_refine_gap,
                "target_label_oracle": r5_refine_gap}   # full labels >= few-label calibration (upper info rung)
    ladder = []
    for pc in schema.PROBLEM_CLASSES:
        row = dict(pc)
        row["recovers_gap_closed"] = recovers.get(pc["rung"])
        # R6 uses the held-out target's OWN scores' mean (transductive), unlike R3 which uses a cross-target model
        row["uses_held_out_target_scores"] = pc["rung"] in ("target_grouped_transductive_zero_label",
                                                            "few_label_target_calibration", "target_label_oracle")
        row["deployable_transductively"] = pc["rung"] in ("target_unlabeled_transductive",
                                                          "target_grouped_transductive_zero_label")
        ladder.append(row)
    grouping_value_over_marginal = ((r6_gap - r3_gap) if (r6_gap is not None and r3_gap is not None) else None)
    return {"ladder": ladder, "within_target_ceiling": within_ceiling,
            "grouping_value_over_marginal": grouping_value_over_marginal,
            "grouping_is_separate_problem_class": True,
            "boundary": ("The pooled cross-target estimand is recoverable by 0-LABEL transductive within-target "
                         "centering (target grouping + the target's OWN candidate scores) — a distinct problem "
                         "class from source-only DG (C19/C23: offset source-unobservable) and from target-label "
                         "calibration (R5). Target-unlabeled MARGINAL geometry (R3) recovers only a weak part; "
                         "target GROUPING adds the rest (value over marginal = R6 - R3). Target grouping is NOT "
                         "source-only, and the target-centered oracle is NOT a deployable selector."),
            "note": ("R3 (target-unlabeled transductive) uses a CROSS-TARGET model on the held-out target's "
                     "unlabeled geometry (no held-out scores); R6 (target-grouped zero-label) uses the held-out "
                     "target's OWN candidate scores' mean. Both are 0-label; they differ in whether the held-out "
                     "target's own score aggregate is used.")}
