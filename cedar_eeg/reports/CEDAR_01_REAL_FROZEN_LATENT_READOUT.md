CEDAR_01 - Real-EEG Frozen-Latent Shadow Audit Readout

This is a source-only frozen-latent shadow audit.
No deployable mask was emitted.
Target metrics are quarantined diagnostic-only.
P1/P2/generalization claims remain blocked.

Status: FAIL_OR_DIAGNOSTIC_ONLY. CEDAR_01 executed on the approved
BNCI2014_001 feature-supply set, but no source-side candidate reached ACCEPT.
This closes the current structured path unless PM explicitly requests a
diagnostic-only follow-up.

Scope

```text
phase: CEDAR_01_REAL_FROZEN_LATENT_SHADOW_AUDIT
dataset: BNCI2014_001
seed: 0
backbones: EEGNetMini, EEGConformerMini
fold universe: 9 LOSO target folds per backbone
feature artifacts read: 18/18
candidate universe: drop_top_1, drop_top_2, drop_top_4
selection regime: source_only_shadow
target_label_role: diagnostic_only
deployable: false
mask_materialized: false
```

Forbidden actions not performed:

```text
P1 channel pruning
P2 TTA preconditioner
fine-tuning
structured pruning
checkpoint write
deployable_mask.json
selected_mask.npz
target bAcc in utility/ranking/tie-break
target improvement claim
safety gate claim
```

Canonical feature-supply handoff

PM required a canonical handoff manifest before CEDAR_01 because `sacct` was
unavailable and shared Route C runtime JSON files were overwritten by array 1.
Generated:

```text
results/cedar/feature_supply/cedar01f_bnci2014_001_seed0/CEDAR_01F_HANDOFF_MANIFEST.json
```

Handoff summary:

```text
source_commit: 724c620
planned_items: 18
completed_items: 18
handoff_manifest_file_sha256: 03d97199352892bf39afeed3ce826aa185735a766351b82aa2083587621d289c
canonical_payload_hash: e5f76423c673db9f7192c327812942385ed81bfe3949d4a7d30c5e86b70e9642
shared_plan_overwritten: true
handoff_manifest_is_canonical: true
deployable: false
```

Cluster provenance caveat fixed in handoff:

```json
{
  "sacct_available": false,
  "scontrol_record_complete": false,
  "compensating_evidence": [
    "array_stdout_hash",
    "array_stderr_hash",
    "per_artifact_manifest_hash",
    "feature_inventory_hash",
    "schema_validation_hash"
  ],
  "pm_disposition": "accepted_for_shadow_audit_not_for_deployment"
}
```

Per-array Route C plan hashes:

```text
890263_0  EEGNetMini        687902ede491d16c22eda3e1e9dc481b7bc0f679b01e401de32eb6fdc16511ae
890263_1  EEGConformerMini  55a20d92d8173d2b302e5ef82554eb70d7d59ddc6d08b84b0e9cb17db2bbb8be
```

CEDAR_01 runner hard-failed if any `.npz` sha256 differed from this handoff
manifest. No mismatch occurred.

Commands

Handoff:

```bash
PYTHONPATH=. python -m cedar_eeg.data.feature_handoff \
  --root results/cedar/feature_supply/cedar01f_bnci2014_001_seed0 \
  --source-commit 724c620 \
  --out results/cedar/feature_supply/cedar01f_bnci2014_001_seed0/CEDAR_01F_HANDOFF_MANIFEST.json
```

CEDAR_01:

```bash
PYTHONPATH=. python -m cedar_eeg.runners.run_01_real_shadow_audit \
  --feature-root results/cedar/feature_supply/cedar01f_bnci2014_001_seed0 \
  --handoff-manifest results/cedar/feature_supply/cedar01f_bnci2014_001_seed0/CEDAR_01F_HANDOFF_MANIFEST.json \
  --out-dir results/cedar/p0_real_shadow/cedar01_bnci2014_001_seed0
```

Required output artifacts

```text
results/cedar/p0_real_shadow/cedar01_bnci2014_001_seed0/run_manifest.json
results/cedar/p0_real_shadow/cedar01_bnci2014_001_seed0/candidate_table.csv
results/cedar/p0_real_shadow/cedar01_bnci2014_001_seed0/candidate_table.hash
results/cedar/p0_real_shadow/cedar01_bnci2014_001_seed0/red_team.json
results/cedar/p0_real_shadow/cedar01_bnci2014_001_seed0/source_decision_summary.json
results/cedar/p0_real_shadow/cedar01_bnci2014_001_seed0/target_diagnostics_DIAGNOSTIC_ONLY.json
results/cedar/p0_real_shadow/cedar01_bnci2014_001_seed0/feature_supply_handoff_hash.txt
```

Runtime hashes:

```text
run_manifest_hash:       04f9572e3a1c37eda0209c8425a99e165ed1c4a653268a7a75193772c0b46fdd
run_manifest_file_sha:   6f6d559a7ece532fa1e31266eb02a8fc20e9c721984e4bd424a424d06a8a9c22
candidate_table_hash:    d87e0277747cd07567e8f5310cff2c3bd2eaa6a32cf76f674eda2ede9aa65137
red_team_file_sha:       1434c187a1c61ba9c8dfbefa3f35820151908fc7d02ed064ec37dfa1817a0647
source_summary_file_sha: 2714d941bb81f4e893316098b36e9ddc6e7a6ac7f8bcb408fb9745fc8f43f08a
target_diag_file_sha:    14abdf14738133de924e33f571ca2bd7cfe7597ce867b9334c993e24a8f45897
```

Red-team result

```text
red_team passed: true
red_team failures: 0
red_team warnings: 18
red_team_zero_warnings: false
```

All 18 warnings were:

```text
no accepted candidate; report-only or abstention outcome
```

This is a CEDAR_01 fail condition under the PM rubric because PASS requires
red-team 0 warnings and a source-side shadow-selected candidate.

Source-side decision result

Candidate table:

```text
rows: 54 candidates + header
decisions:
  REPORT_ONLY: 54
  ACCEPT: 0
  ABSTAIN: 0
```

Reason counts:

```text
leakage_drop_frac below 0.300:          54
random_control_drop_frac exceeds gate:  18
source_bacc_drop exceeds 1 point:        3
```

Backbone summary:

```text
EEGConformerMini:
  folds: 9
  accepted_folds: 0
  grouped_probe_valid_folds: 9
  permutation_null_low_folds: 9
  all_core_pass: false

EEGNetMini:
  folds: 9
  accepted_folds: 0
  grouped_probe_valid_folds: 9
  permutation_null_low_folds: 9
  all_core_pass: false
```

Best observed source-side leakage reductions did not reach the frozen ACCEPT
threshold:

```text
EEGConformerMini best:
  candidate: drop_top_4_of_32
  leakage_drop_frac: 0.198
  leakage_drop_abs: 0.0426
  random_drop_abs: 0.0151
  source_bacc_drop: 0.0000
  decision: REPORT_ONLY
  reasons: leakage_drop_frac < 0.300; random_control_drop_frac exceeds gate

EEGNetMini best:
  candidate: drop_top_4_of_16
  leakage_drop_frac: 0.117
  leakage_drop_abs: 0.0783
  random_drop_abs: 0.0661
  source_bacc_drop: 0.0106
  decision: REPORT_ONLY
  reasons: leakage_drop_frac < 0.300; source_bacc_drop > 0.010; random_control_drop_frac exceeds gate
```

Grouped probe validity

Grouped split remained valid:

```text
18/18 artifact probes had train/eval group overlap = 0
18/18 baseline permutation nulls were low
```

Target diagnostics

Target diagnostics were computed only after source selection completed and were
written only to:

```text
target_diagnostics_DIAGNOSTIC_ONLY.json
```

They were not present in source-side rank keys, source utility, candidate
tie-breaks, or accept logic. The target diagnostic ranges were:

```text
EEGConformerMini:
  target baseline bAcc range: 0.2847 to 0.6545
  candidate target bAcc drop range: -0.0052 to 0.0052

EEGNetMini:
  target baseline bAcc range: 0.2726 to 0.6024
  candidate target bAcc drop range: -0.0122 to 0.0365
```

No target improvement or target-generalization claim is made.

R3 status

R3 bridge is not available in this feature-only CEDAR_01 shadow run. This is
recorded in `source_decision_summary.json` as:

```text
r3_status: not_available_for_feature_only_shadow
r3_caveat: P1 or method claims require an explicit R3 bridge before continuation.
```

Because the audit already failed source-side selection, this R3 caveat does not
change the gate outcome.

Validation

```text
PYTHONPATH=. pytest -q cedar_eeg/tests/test_feature_schema.py cedar_eeg/tests/test_p0_contracts.py
25 passed

python -m compileall -q cedar_eeg
passed
```

Gate outcome

```text
CEDAR_01F_FEATURE_SUPPLY: PASS
CEDAR_01_REAL_SHADOW_AUDIT: FAIL_OR_DIAGNOSTIC_ONLY
P1_CHANNEL_PRUNING: BLOCKED
P2_TTA_PRECONDITIONER: BLOCKED
GENERALIZATION_OR_SAFETY_CLAIM: FORBIDDEN
DEPLOYABLE_MASK_ARTIFACT: FORBIDDEN
```

Next action

Do not request P1. Return this CEDAR_01 readout to PM as a real negative /
diagnostic-only result.
