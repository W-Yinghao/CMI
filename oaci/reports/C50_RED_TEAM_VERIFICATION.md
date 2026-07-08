# C50 - Red-Team Verification

C50 red-team checks were run after artifact generation and before commit.

- self_neighbor_excluded: pass - Locked epsilon neighborhoods inherit C48 distance matrices with diagonal set to infinity.
- query_row_excluded_from_own_neighborhood: pass - Per-query morphology was audited against masks where the query row is excluded.
- target_labels_quarantined: pass - Target endpoint labels appear only as diagnostic labels and report fields carry quarantine flags.
- group_conditioned_baselines_reported: pass - Same-group random, locked local-Bayes, and source-score baselines are emitted by group type.
- reason_coded_failure_ledger_emitted: pass - Failed actionability groups carry predeclared reason codes.
- no_selector_or_checkpoint_recommendation: pass - Tables omit selector-facing identifiers and checkpoint recommendations.
- no_deployable_claim: pass - Report language is diagnostic-only and non-deployable.

Verdict: C50 is a diagnostic morphology audit over a locked C49 witness; it does not emit a selector.
