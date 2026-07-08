# C50 - Conditioned-Island Morphology / Fragmentation Audit (frozen C19 `664007686afb520f`)

## Decision

`C50-C_mixed_fragmentation_plus_underuse`

## Locked Witness

- condition_scope: `within_target`
- source_space: `all_source_objectives`
- neighborhood: `eps_q20` radius `3.253`
- min_n: `1`
- inherited C49 hit / coverage / enrichment: **1.000 / 1.000 / 2.360**

## Main Result

- target min hit / coverage: **1.000 / 1.000**.
- trajectory min hit / coverage: **0.000 / 1.000**.
- target / trajectory actionability fail fraction: **0.000 / 0.432**.
- max mean existing-score underuse gap: **0.431**.

## Why Coverage Did Not Become Actionability

C50 keeps the C49 broad witness fixed and audits morphology only. The witness remains diagnostic: coverage is computed inside target-conditioned source space, while actionability fails where groups are fragmented and/or available source scores do not recover the diagnostic islands.

## Red-Team Checks

- self_neighbor_excluded: PASS - Locked epsilon neighborhoods inherit C48 distance matrices with diagonal set to infinity.
- query_row_excluded_from_own_neighborhood: PASS - Per-query morphology was audited against masks where the query row is excluded.
- target_labels_quarantined: PASS - Target endpoint labels appear only as diagnostic labels and report fields carry quarantine flags.
- group_conditioned_baselines_reported: PASS - Same-group random, locked local-Bayes, and source-score baselines are emitted by group type.
- reason_coded_failure_ledger_emitted: PASS - Failed actionability groups carry predeclared reason codes.
- no_selector_or_checkpoint_recommendation: PASS - Tables omit selector-facing identifiers and checkpoint recommendations.
- no_deployable_claim: PASS - Report language is diagnostic-only and non-deployable.

## Claim Ledger

Can say:
- C49's broad conditioned witness is real as a diagnostic ceiling.
- C50 attributes non-actionability to measured fragmentation and/or existing-score underuse.
- Conditioning can reveal target-good islands in source-objective space under diagnostic labels.

Cannot say:
- Do not treat the witness as a checkpoint selection rule.
- Do not frame C49/C50 as restoring OACI.
- Do not claim source-only control has been recovered.
- Do not present target-conditioned local Bayes estimates as a deployment rule.
- Do not turn existing-score underuse into a new trainable-objective claim.
