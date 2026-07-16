# OACI / EEG-DG Project Memory Through C84SR2

## Current Gate

```text
C84S_STAGE_B_FIELD_DESCRIPTOR_COMPATIBILITY_REPAIRED_AND_V4_LOCK_READY_FOR_FRESH_PI_AUTHORIZATION
```

C84SR2 is complete readiness evidence, not a C84 scientific result. C84S V4
has no authorization record and has not accessed real evaluation labels,
computed real selector scores or produced a scientific taxonomy.

## Current Operative Identities

```text
C84F complete-field manifest:
  cfffcac1a55148941b809b69bed2c9a8957a94729ed7f2c2c29ed8d48c0134d8

C84SR2 repair protocol commit:
  5fa4366f57087f07cf6e290a84f37abbb1ef97c8

C84SR2 repair protocol SHA-256:
  6d7853cd60a85c9f3516cb21fda1c75909f0963e96ad2ac0292647bdc93f1aef

lock-bound implementation commit:
  a737e2b91578aa7e85aab948ca4b0e64929c3073

C84S V4 lock commit:
  8528bd142bbf6c4cca047bdcc558133eebf5e757

C84S V4 lock SHA-256:
  582e5074b4b17d62ff1e5fbfd992f037dd3082b7763b22d707630aa19db81c3d

accepted regression commit:
  b08538d3f399c77bb188246d23472cb5fd39ded5
```

## Scientific State

The latest valid C84 object is still the label-free C84F field:

```text
datasets:                  Lee2019_MI / Cho2017 / PhysionetMI
candidate units:           1,944
levels:                    972 / 972
target subjects:           118
target trial rows:         9,621
target contexts:           944
candidate-context slices:  76,464
target artifacts:          1,944
context/digest sidecars:   1,944
```

It contains no target labels, selector scores or scientific results. The latest
completed scientific comparison remains C82-D on BNCI2014_001; C84 has not yet
established external validity.

## C84S V3 Failed Attempt

Direct authorization was consumed by job `897843`. Stage A provisioned the
locked construction/evaluation split, then Stage B failed during metadata
enumeration before any selector score.

```text
construction rows:             4,773
evaluation rows:               4,848
overlap:                       0
construction access:           1
evaluation access:             0
selector contexts:             0
scientific rows:               0
training / forward / GPU:      0 / 0 / 0
same-label oracle:             0
evaluation descriptor sealed: true
```

The failed root and authorization consumption are immutable. The V3
authorization cannot migrate.

## C84SR2 Repair

The complete field descriptor is authoritative for `level_intervention_id`.
The sidecar audit is exact:

```text
native sidecars:                       1,701
historical omitted-field sidecars:       243
total:                                 1,944
```

All 243 omissions are reused C84C panel-A/seed-5/level-0 units and map to
`C84_LEVEL0_FULL_SOURCE_PANEL_V1`. Present sidecar values must match the field
descriptor and locked level definition. Any other omission or mismatch fails.

The repair changes no candidate identity, method, score, threshold, budget,
split, inference procedure or taxonomy.

## Stage-A Reuse Boundary

V4 hashes and replays the immutable V3 Stage-A objects. It does not invoke the
label provisioner or reload target labels. Stage B receives only the
construction handoff; Stage C receives the evaluation seal only after exact
selection freeze.

```text
Stage-A complete:
  29e77b600184f3f96b65858a40767f562e664e81eb848e2546e5486886ed35bd
construction handoff:
  29b0848b8bbc40d5d346722d7cc479c69995406444ee8b9688f663f9c4256223
evaluation seal:
  54e06dff60d80255631dc4faa20c8c7db651f2af8fc5415671dd9ab6681b5502
```

## Readiness Evidence

```text
real descriptor contexts:      944 / 944
candidates per context:          81 / 81
external objects rehashed:    7,776 / 7,776
external bytes rehashed:      48,072,941,176

synthetic contexts:              944
synthetic Q0 chains:           2,048
synthetic Q0 records:      9,110,448
synthetic final rows:          18,608
red team:                       50 / 50 PASS
```

Synthetic A/B/C/D/E and L1/L2/L3/L4 branches all passed. No precomputed final
rows were injected and no real field array or label was read by synthetic
calibration.

Accepted regressions:

```text
focused:   242 passed
C65:       843 passed, 1 skipped, 3 deselected
C23:     1,254 passed, 1 skipped, 3 deselected
full:    2,178 passed, 1 skipped, 3 deselected
```

All accepted stderr files are empty. `squeue` was used; `sacct` was not.

## Authorization Boundary

The only next execution authorization phrase is:

```text
授权 C84S
```

It must be recorded against the V4 lock SHA above. C84SR2 does not authorize
C85, training, forward, GPU, new methods, retuning, same-label oracle access or
manuscript changes.
