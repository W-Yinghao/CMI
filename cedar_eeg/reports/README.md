# CEDAR Reports Index

CEDAR source-only frozen-latent route is closed negative as of commit
`107192f`. Do not start P1/P2 from this branch. CEDAR artifacts are
diagnostic-only. Target diagnostics were quarantined and never used for
selection.

## Final State

```text
CEDAR_01F_FEATURE_SUPPLY: PASS
CEDAR_01_REAL_SHADOW_AUDIT: COMPLETE_NEGATIVE
CEDAR_SOURCE_ONLY_LATENT_SURGERY_TO_P1: CLOSED_NEGATIVE
P1_CHANNEL_PRUNING: DENIED
P2_TTA_PRECONDITIONER: DENIED
DEPLOYABLE_MASK_ARTIFACT: FORBIDDEN
GENERALIZATION_OR_SAFETY_CLAIM: FORBIDDEN
CEDAR_RETAINED_ROLE: DIAGNOSTIC_ONLY / MEASUREMENT_TO_CONTROL_NEGATIVE_EVIDENCE
```

## Closeout

```text
CEDAR_01N_NEGATIVE_CLOSEOUT.md
CEDAR_01N_FAILURE_TAXONOMY.md
CEDAR_01N_PM_DECISION.md
```

These files freeze the official negative conclusion: BNCI2014_001 real EEG
source-only frozen-latent shadow audit produced `0/54` accepted candidates.

## Real EEG Audit

```text
CEDAR_01_REAL_FROZEN_LATENT_READOUT.md
CEDAR_01_REAL_FROZEN_LATENT_PROTOCOL.md
CEDAR_01_ACCEPTANCE_CRITERIA.md
```

The real audit used the fixed candidate universe `drop_top_1`, `drop_top_2`,
`drop_top_4`, grouped conditional leakage probes, permutation nulls, matched
random-subspace controls, and target-quarantined diagnostics. It emitted no
deployable mask.

## Feature Supply

```text
CEDAR_01F_FEATURE_SUPPLY_PROTOCOL.md
CEDAR_01F_FEATURE_INVENTORY_READOUT.md
CEDAR_01F_ROUTE_C_FEATURE_DUMP_PLAN.md
CEDAR_01F_ROUTE_C_SUBMISSION_READOUT.md
CEDAR_01F_ROUTE_C_COMPLETION_READOUT.md
```

Feature supply passed with 18/18 compliant BNCI2014_001 artifacts. The feature
schema, handoff manifest, and validation code remain reusable diagnostic assets.

## Scaffold

```text
CEDAR_00_P0_SCAFFOLD_REPORT.md
```

CEDAR_00 remains a synthetic scaffold / red-team contract, not a real EEG
scientific result and not approval for P1/P2.
