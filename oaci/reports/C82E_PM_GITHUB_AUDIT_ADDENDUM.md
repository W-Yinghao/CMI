# C82E PM GitHub Audit Addendum

## Status

This is an additive interpretation and provenance addendum to
`C82_POST_C81_BASELINE_RECOVERY.md`. It does not rewrite C82 history, alter any
frozen table, recompute a statistic, or change the accepted gate:

```text
C82-D_zero_label_comparison_training_seed_method_identity_or_target_heterogeneous
```

The audit used committed source code, compact result tables, and their hashes
only. It did not open an EEG array, construction/evaluation label view, frozen
selection payload, or same-label oracle.

## 1. LOTO Evidence Semantics

The C82P output registry described `leave_one_target_method_stability.csv` with
the prospective primary key `held_out_target | method_id`. The emitted table is
instead a 16-row seed-by-left-out-target panel with these fields:

```text
seed
left_out_target
full_category
LOTO_category
supporting_same_methods
same_method_preserved
category_preserved
panel_preserved
```

The implementation first derives the full-panel method sets separately for
each seed and then intersects them across seeds. A panel with full category B
can preserve method identity only through the full-panel cross-seed common B
set. In the observed result:

```text
seed 3 full category: B
seed 4 full category: C
full-panel common B-method set: empty
global method-aware LOTO preservation: 7 / 16
registered threshold: 12 / 16
```

Consequently, the eight seed-3 rows are mechanically marked
`same_method_preserved=0` because the cross-seed common B set is empty. The
frozen table does not contain a per-panel, per-method Q1 ledger from which one
could infer that COTT independently lost Q1 in every seed-3 leave-one-target
panel.

The authoritative wording is:

> The registered global method-aware stability rule preserved 7/16 panels. The
> full-panel categories already differed across seeds, and the common B-method
> set was empty.

The following stronger wording is not supported without a new exact per-panel
method ledger and must not be used:

> Every seed-3 LOTO panel independently demonstrated that COTT lost Q1.

The full-panel B/C mismatch and empty cross-seed method intersections already
select C82-D. This clarification therefore does not change the primary gate.

## 2. Q5 Descriptive Best-Method Wording

The information-class membership in `information_class_summary_Q5.csv` was
fixed before C82 execution. Within each fixed class, however, the implementation
selects the displayed method by minimizing the observed mean standardized
regret, with method ID as the tie breaker.

The authoritative description is:

```text
descriptive best registered method within a fixed class
```

It must not be described as either:

```text
a prospectively fixed class representative
an inferential winner across methods
```

Q5 remains a field-specific descriptive summary. Its displayed method choices
do not alter Q1/Q2, max-T, noninferiority, cross-seed intersections, LOTO, or
the C82 taxonomy.

## Identity Replay

```text
C82 result SHA-256:
  d8060e6636adf7fcca7a0ace0e47bb7043676b7681569e09fb8705dcb8d5a8b7

C82 artifact-manifest SHA-256:
  910e2ff1d8445dae262be82d417140cd44fc48be1306f2bbe5a439ec3549f0a2

leave_one_target_method_stability.csv SHA-256:
  8776693481efcb24e87f89fafadd01bbb4e656958a6075970a256da2e882afb3

information_class_summary_Q5.csv SHA-256:
  98a8e3563a5c64c8bf4f4b3404c5a90ce8f6ec2256b7163565f4def84d4d04ed
```

## Final Disposition

```text
C82 gate changed: false
C82 result/table content changed: false
new real-data statistics: 0
new label-view accesses: 0
C81 historical gate changed: false
```

