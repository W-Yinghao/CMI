# C86R2 Claude Code Handoff

## 1. Purpose

This is the operational handoff for a fresh coding agent taking over OACI after
C86R2. It is intentionally narrower and more current than the long historical
`oaci/OACI_CODEX_HANDOFF.md` ledger.

Do not infer execution authority from this document. It records repository
state and the next review boundary; it is not a PM instruction, PI
authorization, execution lock, or authorization record.

## 2. Repository Entry State

```text
repository:
  /home/infres/yinwang/CMI_AAAI_oaci

branch:
  oaci

authoritative C86R2 evidence HEAD before this docs-only handoff:
  13d7b60e250fe0ffdfe27e3e28ba07485b7ae122

required invariant before new work:
  branch == oaci
  HEAD == origin/oaci
  worktree clean
  13d7b60e250fe0ffdfe27e3e28ba07485b7ae122 is an ancestor of HEAD
```

Run first:

```bash
cd /home/infres/yinwang/CMI_AAAI_oaci
git status --short --branch
git rev-parse HEAD
git rev-parse origin/oaci
git merge-base --is-ancestor \
  13d7b60e250fe0ffdfe27e3e28ba07485b7ae122 HEAD
```

Do not reset, rewrite, or remove historical files. The repository intentionally
contains superseded locks, consumed authorizations, failed attempts, and
historical blocker reports.

## 3. Current Gate

C86R2 completed at:

```text
C86_ADULT_UNTOUCHED_MULTI_COHORT_ELIGIBILITY_RESOLVED_READY_FOR_C86LP_PROTOCOL_REVIEW
```

This means only that the adult-only public-metadata input blocker is resolved
and the project is ready for PM review of a possible C86LP protocol milestone.

It does not authorize:

```text
C86LP
C86L
C86D
C86C/F
C86H
new EEG or label access
candidate prediction access
active acquisition
training or forward passes
GPU work
C87
manuscript drafting or modification
```

All prior C84/C85 direct authorizations were scope-specific and consumed or
historical. They cannot be reused for any C86 stage.

## 4. Authoritative C86R2 Identities

```text
C86R2 protocol commit:
  0da4ec39f26ac4bc0d89035e9ad951f452217f05

C86R2 protocol SHA-256:
  2e88e2fef7500b12ca8b3c5b19e6aab06df5a7f388781855b73793a1fe75df92

initial implementation commit:
  bb91b82b3ac083d34c3ceb56687dea9c93fb4f76

exact catalog-count reconciliation commit:
  ccc1807479a2f3b3273991f6b3ca0ab5514184f1

evidence/report commit:
  b6c331d79ec9fae38dbc6c9379c19ef9ecac72a4

regression-report commit:
  13d7b60e250fe0ffdfe27e3e28ba07485b7ae122

C86 effective-program manifest V3 SHA-256:
  c6b7e490e0f78f74f820428cee138782caff1dc0033422723593a7d8e3c5f77e

C86R2 overall Markdown SHA-256:
  d8c7b974838ec1205417829b98f9058aa750742207036a7d2124c37e937de788

C86R2 overall JSON SHA-256:
  a635791e84f6c9cb4032b9b98c3426d4aeafd7ada4ed89a9111f18b97f4d01ae
```

The V3 manifest binds all 15 C86R2 table hashes and the exact resolver byte
SHA. Replay it rather than copying values from prose.

## 5. Read Order

Read these files before proposing or editing anything:

```text
1. oaci/reports/C86R2_CLAUDE_CODE_HANDOFF.md
2. oaci/reports/C86R2_ADULT_PARTICIPANT_AND_COHORT_RESOLUTION_PROTOCOL.json
3. oaci/reports/C86_ACTIVE_TESTING_EFFECTIVE_PROGRAM_MANIFEST_V3.json
4. oaci/reports/C86R2_OVERALL_REPORT.md
5. oaci/reports/C86R2_PROTOCOL_READINESS.md
6. oaci/reports/C86R2_FINAL_REPORT_RED_TEAM.md
7. oaci/reports/C86R2_REGRESSION_VERIFICATION.md
8. oaci/reports/C86R_ELIGIBILITY_BASELINE_AND_DEVELOPMENT_VIEW_REPAIR_PROTOCOL.json
9. oaci/reports/C86_ACTIVE_TESTING_EFFECTIVE_PROGRAM_MANIFEST_V2.json
10. oaci/reports/C86_ACTIVE_TESTING_PROGRAM_PROTOCOL.json
```

Then inspect implementation and tests:

```text
oaci/theory/c86r2_adult_cohort_resolution.py
oaci/tests/test_c86r2_adult_cohort_resolution.py
oaci/tests/test_c86r2_catalog_and_field_resolution.py
oaci/theory/c86r_program_repair.py
oaci/tests/test_c86r_program_repair.py
oaci/tests/test_c86r_field_feasibility.py
```

## 6. Scientific State That Must Not Change

```text
C84 primary gate:
  C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous

C84 label frontier:
  C84-L4

C85 theorem statuses:
  T1 / T3 / T4 / T7 = PROVED
  T2 / T6 = COUNTEREXAMPLE
  T5 = OPEN

C85E status:
  POST_C84S_EXPLORATORY bridge complete
```

C86R2 is public-metadata-only readiness work. It does not change any C84 gate,
C85 theorem status, or C85E exploratory result.

## 7. Final Adult Confirmation Population

The locked participant rule is age at recording `>=18`. A deterministic
interface must include all canonical subjects with valid public adult evidence,
exclude every minor and unknown-age subject, map exactly to loader/BIDS IDs,
and retain at least 12 subjects.

Every passing interface is included. There is no cap at two.

### 7.1 Primary adult interfaces

```text
Brandl2020_CANONICAL_ADULT_V1:
  adult subjects: 16
  canonical IDs: 1 through 16
  minimum documented trials/subject: 504
  subject-list SHA-256:
    cd5cb9fb5ac3c4f4007e8b41d117da21622439cd05c1728f3e82f90e4f869dad
  license: CC-BY-NC-ND-4.0

OpenNeuro_ds007221_HYBRID_ADULT_V1:
  adult subjects: 37
  canonical IDs: sub-37 through sub-73
  age range: 20-27 at recording
  native events: left_hand / right_hand
  documented trials/subject: 600-640
  native channels: 69; all common 11 channels present
  subject-list SHA-256:
    26520d5b53cfdb76915b552f4e3ac8d918deb584a430a6d03c4ac804fefa203c
  OpenNeuro: ds007221 v1.0.1
  NEMAR mirror: on007221 v1.0.0
  license: CC0

total primary adult confirmation population:
  2 native cohorts
  53 target subjects
```

### 7.2 Preserved non-primary roles

```text
Yang2025_2C:
  AGE_MIXED_STRESS_TRACK_ONLY
  no valid subject-level adult map

Kumar2024:
  AGE_UNCERTAIN_STRESS_TRACK_ONLY
  mean/SD and year-like age column do not prove adult eligibility

Dreyer2023:
  DEVELOPMENT_ONLY_HISTORICAL_ACCESS
  untouched target access is not certifiably absent
```

Do not promote these cohorts or relax the adult target population inside a
downstream implementation repair.

### 7.3 `ds007221` native task decisions

```text
task-hybrid:
  PASS

task-graz:
  FAIL_MULTICLASS_NO_CLASS_FILTERING

task-ssmvepmi:
  FAIL_MULTICLASS_NON_NATIVE_EXACT_TASK

task-hybridonline:
  FAIL_MINIMUM_TARGET_SUBJECT_COUNT (11)
```

Do not class-filter Graz or SSMVEP-MI to manufacture another interface.

## 8. Catalog Audit State

The safe metadata universe was frozen after the chronology protocol and before
screening:

```text
installed MOABB imagery catalog:
  53 rows
  SHA-256 5d7b3abca6f56c83ce90a90d3e0c252783a24f04e237d8d09b11f3862f0ce7e4

EEGDash public catalog:
  824 rows
  71 safe task/name candidates
  SHA-256 ff12eabbe4832e7303a2289acce0384a69dc478231e5949fad68d4772784760f

NEMAR public catalog:
  760 rows over four frozen pages
  78 safe task/name candidates
  combined SHA-256 5b6f4c3cca54a6a466a0549f9e21a48220e6e515881abd0306376c53d0afbfdf

public catalog candidates before mirror deduplication:
  149

native candidate datasets after mirror deduplication:
  79

interface rows after expanding the four ds007221 native tasks:
  82
```

The raw temporary catalog responses were not committed and must not be treated
as durable input. Durable evidence is the committed snapshot registry, source
hashes, candidate disposition table, deduplication ledger, and V3 manifest.

An initial generator attempt expected 89 native candidates and failed before
publication. Exact replay established 79. That failed attempt is preserved in
the red-team, overall, and regression reports; no cohort decision changed.

## 9. Common Field Planning Contract

```text
interface ID:
  C86_C84SOURCE_TARGET_11CH_160HZ_0_3S_V3

channels:
  FC5 FC1 FC2 FC6 C3 Cz C4 CP5 CP1 CP2 CP6

resampling / epoch / bandpass:
  160 Hz / [0,3) seconds / 4-38 Hz

source training/audit:
  fixed legacy C84 source panels under one newly trained common-channel zoo

untouched cohort subjects:
  targets only

target in own training:
  0

same candidate zoo across both primary cohorts:
  required
```

Planning arithmetic:

```text
target subjects: 53
target contexts: 424
candidate-context slices: 34,344
candidate units: 648
training phases: 24
unit-cohort artifacts: 1,296
registered max-T draws per cohort: 65,536
```

These are metadata estimates, not resource authorization or evidence that the
field has been engineered. License replay, storage, model training, canary, and
failure handling remain future work.

## 10. C86 Program Values Already Fixed

Downstream work must consume V3, not only the stale parent or V2 manifest.

```text
candidate action structure:
  1 ERM / 40 OACI / 40 SRC = 81 actions

total-query grid:
  4 / 8 / 16 / 32 / FULL

primary passive comparator:
  P0 uniform without replacement

secondary class-aware comparator:
  P1 historical class-stratified Q0

primary active baselines:
  A1 LURE adaptation
  A2H faithful Hara general-K sum-over-pairs score

development-only max-pair heuristic:
  A2M, explicitly not Hara

secondary baseline registry:
  MODEL SELECTOR / CODA

unbiasedness claim:
  registered linear moments only

nonlinear plugin objects:
  balanced accuracy / ECE / midranks / composite utility / selected action

nonlinear plugin unbiasedness claim:
  none
```

The operative method registry remains
`oaci/reports/c86r_tables/active_method_registry_v2.csv`. The operative
synthetic contract remains
`oaci/reports/C86P_SYNTHETIC_CALIBRATION_OPERATIONALIZATION_PROTOCOL_V2.json`.

## 11. Possible Next Milestone

The next possible milestone is C86LP, but only after a new PM instruction
explicitly starts it.

C86LP is expected to be protocol and lock readiness for a later C86L
development-only contribution-field stage. It must preserve:

```text
C86L acquisition pool:
  immutable C84 construction trial IDs and matching frozen probabilities

held development outcome:
  accepted C85U held-evaluation candidate-utility field

direct C84 evaluation-label access in C86L/C86D:
  forbidden

construction/evaluation overlap:
  zero

unsupported target-budget cells:
  INPUT_UNAVAILABLE

replacement, budget substitution, target deletion:
  forbidden
```

C86LP itself should not open real labels, predictions, EEG, Q0 shards, or
candidate arrays. It should create an additive protocol before any future
protected implementation or lock and should stop before C86L execution.

Do not create a C86L execution lock merely because this handoff names C86LP.
Wait for the user's exact next-stage instructions and follow their chronology.

## 12. Validation Baseline

Accepted at clean evidence HEAD
`b6c331d79ec9fae38dbc6c9379c19ef9ecac72a4`:

```text
focused C86P/C86R/C86R2:
  51 passed

C65 cumulative:
  1,103 passed, 1 skipped, 12 deselected

C23 cumulative:
  1,514 passed, 1 skipped, 12 deselected

full OACI:
  2,489 passed, 1 skipped, 12 deselected

accepted stderr:
  0 bytes for every suite
```

The one skip is the finalized historical C78F check. The twelve cumulative
deselections are historical absence assertions superseded by later accepted
scope-specific stages. No C86P, C86R, or C86R2 test was deselected.

Focused tests:

```bash
/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/bin/python \
  -m pytest -q \
  oaci/tests/test_c86_active_program_protocol.py \
  oaci/tests/test_c86_dataset_eligibility.py \
  oaci/tests/test_c86_active_shadow_contract.py \
  oaci/tests/test_c86r_program_repair.py \
  oaci/tests/test_c86r_field_feasibility.py \
  oaci/tests/test_c86r2_adult_cohort_resolution.py \
  oaci/tests/test_c86r2_catalog_and_field_resolution.py
```

The cumulative suite generator is
`python -m oaci.multidataset.c84r_regression_suite`. The existing
`oaci/slurm_c86p_regression.sh` records the accepted historical deselection
set and clean-HEAD checks.

## 13. Fast Integrity Replay

```bash
cd /home/infres/yinwang/CMI_AAAI_oaci/oaci/reports
sha256sum -c C86R2_ADULT_PARTICIPANT_AND_COHORT_RESOLUTION_PROTOCOL.sha256
sha256sum -c C86_ACTIVE_TESTING_EFFECTIVE_PROGRAM_MANIFEST_V3.sha256
sha256sum -c C86R2_OVERALL_REPORT.sha256

cd /home/infres/yinwang/CMI_AAAI_oaci
python -m oaci.theory.c86r2_adult_cohort_resolution \
  validate-effective-manifest
```

Do not rerun `write-readiness` from an uncommitted or reconstructed catalog
projection. The committed artifacts are authoritative.

## 14. Residual Risks For The Next Agent

1. `ds007221` is a newly resolved metadata interface, not an engineered field.
   No project EEG byte or label has been opened.
2. The public dataset is large. Resource estimates are planning values and must
   be canary-validated before any full field generation.
3. Brandl has restrictive license terms. Future internal use and derived
   artifact handling require explicit terms replay.
4. OpenNeuro and NEMAR are mirrors of one native cohort. Never count them as
   separate confirmation cohorts.
5. Public catalogs can change. V3 binds the frozen 2026-07-18 snapshots; a
   future refresh must be additive and must not silently rewrite V3.
6. The common 11-channel interface is metadata-validated only. It does not
   establish preprocessing, training, or numerical equivalence.
7. Passing the adult input gate does not authorize acquisition-policy tuning,
   new model-zoo design, or confirmatory claims.

## 15. Working Style Expected

```text
read the code and bound artifacts before editing;
preserve historical failures and superseded objects;
commit additive protocols before protected inspection or implementation;
use exact hashes and row-count replay;
fail closed on provenance or interface uncertainty;
never reuse an old authorization;
do not claim a gate before all tests and publication identities pass;
keep HEAD == origin/oaci and the worktree clean at accepted execution points.
```

## 16. Minimal Startup Prompt For Claude Code

```text
Continue OACI on branch oaci from the latest origin/oaci.

First read:
  oaci/reports/C86R2_CLAUDE_CODE_HANDOFF.md
  oaci/reports/C86_ACTIVE_TESTING_EFFECTIVE_PROGRAM_MANIFEST_V3.json
  oaci/reports/C86R2_OVERALL_REPORT.md
  oaci/reports/C86R2_FINAL_REPORT_RED_TEAM.md

Verify that 13d7b60e250fe0ffdfe27e3e28ba07485b7ae122 is an ancestor of
HEAD, HEAD equals origin/oaci, and the worktree is clean.

Current accepted gate:
  C86_ADULT_UNTOUCHED_MULTI_COHORT_ELIGIBILITY_RESOLVED_READY_FOR_C86LP_PROTOCOL_REVIEW

Do not infer authorization from the handoff. Do not start C86LP or create a
C86L lock unless the newest user/PM instruction explicitly starts that scope.
Do not access EEG, labels, candidate outputs, active results, training, GPU,
C87, or manuscript code.

When a new scoped instruction arrives, preserve V3, consume the effective V3
program rather than the stale V2 cohort list, follow additive chronology, and
stop at the exact gate specified by that instruction.
```
