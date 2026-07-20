# C33 - Local Trajectory Boundary / Checkpoint Neighborhood Audit (frozen C19 `664007686afb520f`)

> C32R showed selected OACI is near joint-good candidates but hits at random-like top-1. C33 audits the local boundary: run structure, selected-vs-nearest-joint pairs, adjacent gradients, score plateaus, and local information rungs. Diagnostic-only; no selector, no training, no selected-checkpoint artifact.

- **cases: `B3_source_score_active_misranking, B4_local_gauge_jump_unseen_by_source, B7_target_unlabeled_pooled_only_confirmed, B8_label_margin_instability`**

## Gate 1 - local boundary geometry

- mean transition rate: **+0.224**; median selected-boundary distance: **+1.500**.
- selected +/-1 neighborhood contains joint-good in **+0.574** of units; mean +/-1 joint-good rate **+0.420**.
- mean joint-good run length: **+4.432**. This does **not** clear the B1 dense-boundary gate globally; the local boundary signal is moderate, not the headline.

## Gate 2 - selected vs nearest joint-good

- selected hit **+0.471**; median order/epoch delta to nearest joint-good **+1.000 / +5.000**.
- miss-conditioned source-flat fraction **+0.049**; source-wrong fraction **+0.309**; pair gauge-jump-unseen fraction **+0.358**.

## Gate 3 - adjacent local gradients

- transition pairs: **780** / 3642 (fraction +0.214).
- source gradient sign agreement on transitions **+0.483**; rank agreement **+0.579**; transition gauge-jump fraction **+0.800** (B4 is read as common target-margin jumps with weak source alignment, not a clean pairwise unseen-gauge claim).

## Gate 4 - source-score plateau

- mean/median plateau size at eps=0.02: **+3.370 / +3.000**.
- if selected is bad, plateau contains joint-good in **+0.256** of units.

## Gate 5 - local information ladder

- source pm1 enrichment **+1.000**.
- target-unlabeled pm1/pm2 top-1 gain vs source **+0.025 / -0.049**.
- target-grouped pm1 gain vs source **+0.000** (rank-invariant inside same-target local neighborhoods; non-deployable diagnostic, not a local ceiling).

## Margin sensitivity

- robust cases: **B3_source_score_active_misranking, B4_local_gauge_jump_unseen_by_source, B7_target_unlabeled_pooled_only_confirmed**; changed cases: **B8_label_margin_instability**.
- primary vs robust pm1 joint rate **+0.420 / +0.272**.

## Bottom line

> C33 localizes the C32R miss to active local misranking plus target-margin jumps with weak source alignment, rather than to global dense-boundary jitter or source-score indifference. Selected OACI is usually close to a joint-good candidate, but among actual misses the source score is usually not flat; it often prefers the non-joint selected point. Target-unlabeled R3 remains pooled/gauge-help rather than a local top-k rescue. Target-grouped is rank-invariant in these same-target neighborhoods, and target-label quantities remain diagnostic only, not methods.
