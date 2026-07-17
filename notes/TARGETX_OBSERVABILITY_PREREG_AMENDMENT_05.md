# Target-X observability audit — PRE-REG AMENDMENT 05 (F2.1c focused closure)

Closes the four remaining Track-A gaps on `406f4a54` per PM review. Branch `agent/cmi-trace-targetx-
observability`. Manuscript FROZEN; no new observables/backbones; F2.1c ends at a 2-subject/dataset engineering
smoke, after which (tests+smoke passing) the full Gates 1-4 run is PRE-AUTHORIZED.

## P0.1 — no empty-contested fallback
When `rank(B_contested) = 0` the selector action set is identity-only (baselines still computed); the fold is
flagged `no_contested_candidate = true`. The full `B_cond` is NEVER re-introduced (it would smuggle head-null
free directions back into the DG selector). The same fallback is removed from the Gate-5 runner.

## P0.2 — constrained hindsight is source-safe + same action pipeline
`hindsight_constrained` (the Gate-3 denominator) is now selected from the SAME informed action objects
(contested span, rank≤3), filtered to source-LOSO task drop ≤ 0.02, chosen by max T_cal bAcc via the actions'
`apply_source` / `apply_target_cal`, scored on T_query. Identity is a legal candidate. It does NOT require the
G1 random-specificity gate (it is a utility hindsight upper bound, not a deployable rule). The unconstrained
full-cond greedy remains a ceiling (not a gate).

## P0.3 — Gate-1 applicability + projector dedup
The task-rowspace anchor has rank ≤ C−1, so contested rank is ≤ 3 (4-class) / ≤ 1 (binary). `gate1_applicable
= (projected_contested_rank ≥ 3)`. On a non-applicable dataset (BNCI2015 binary) Gate 1 is `null` (a single
candidate direction is not an observability failure); that dataset carries only actionability + harm-safety +
specificity. Overall GO needs: Gate 1 on ≥1 rankable dataset; Gate 2/3/5 on ≥1 rankable dataset; Gate 4 = no
clear harm on the other. Informed actions are DEDUPED by `projector_hash` (source-greedy prefixes can equal
enumerated subsets), with `aliases` recorded, so no projector is double-weighted in the Spearman.

## P0.4 — Gate 5 multi-capacity (staged, not blocking Gates 1-4)
Gate 5 already uses full posterior-KL + training-only null. Before the GPU Gate-5 run it must reuse the
validated posterior-KL ruler, report linear/MLP-small/MLP-large with a pre-frozen primary capacity +
familywise-max rule, exact-selected-rank random, and NOT read only the linear column. This is staged AFTER
Gates 1-4 and does not block them.

## renames / interpretation
`contested_rank/free_rank` → `projected_contested_rank/projected_free_rank` (projections of B_cond onto
row(W_c)/ker(W_c); NOT an additive split of the full rank). Interpretation uses `head_overlap_energy` (fraction
of B_cond energy inside row(W_c)), not "3+10 of which most is free".

## Pre-authorized full Gates 1-4 (63 EEGNet fold-seed cells) after F2.1c tests + smoke
9×3 (BNCI2014) + 12×3 (BNCI2015). Report: completeness 63/63; identity-fallback rate; selected-rank
distribution; per-subject sign; Δ_TX cluster CI; paired random/source-greedy/whitening/centering CIs;
constrained-hindsight CI; Gate-3 paired inequality; BNCI2014 Gate-1 observability; BNCI2015 Gate-1
applicability flag; pooled sensitivity; NO favorable-subject filtering. If BOTH datasets fail Gate 2 -> close
the target-X selector line; do NOT run GPU Gate 5.
