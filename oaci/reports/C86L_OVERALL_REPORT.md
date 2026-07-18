# C86L — Overall Report (executed under 授权 C86L)

**Gate reached**

```text
C86L_C84_CONSTRUCTION_POOL_TRIAL_CONTRIBUTION_FIELD_FROZEN_C86D_PROTOCOL_REVIEW_REQUIRED
```

The real C86L development-only query field was built and frozen under a direct
`授权 C86L`. All fail-closed invariants passed; the field is atomically finalized.
C86D / C86H / C87 / manuscript remain NOT authorized.

## 1. Authorization and scope

Executed strictly as **C86L only**: freeze the development-only query field from
the immutable C84 construction pool, without contaminating the held C85U outcome.
It ran **no** active policy (P0/A1/A2H), produced **no** active-policy scientific
result, and touched **no** untouched Brandl/ds007221 data. `C84-D`, `C84-L4`, and
theorem statuses T1/T3/T4/T7 PROVED, T2/T6 COUNTEREXAMPLE, T5 OPEN are unchanged.

## 2. Execution

```text
authorization : 授权 C86L (direct, standalone trigger)
runner        : oaci/active_testing/c86l_build.py  (execute() wired)
SLURM job     : 901965  (partition cpu-high)
build time    : 10.5 s   (reads only `probabilities`, not the n×1040 `z`)
finalize      : atomic — built into <root>.staging, then os.replace (no partial publish)
commits       : b24a4ee5 (build) / 347688a2 (result identity)
```

## 3. Verified real inputs (opened under authorization)

```text
predictions  : oaci-c84-full-field-target-replay-v2/lock_f0c369ee…/complete_target_unlabeled_v2
               1,944 npz candidate units; per-trial `probabilities[·,2]`, keyed by target_trial_id
construction : oaci-c84s-analysis-v3/stage_a_labels/target_construction_label_view/labels.csv
               4,773 rows, canonical_class_label ∈ {0,1} (balanced 2393/2380)
               sha256 fdf36052d36ad9546cda06cbc567f68cdcced7ad08fd1311ab949471218b3134
split rule   : sort SHA256(C86_TARGET_SPLIT_V1|dataset|subject|trial); floor(n/2)→acquisition
join key     : target_trial_id
held C85U    : oaci-c85u-candidate-utility-v2 — identity-bound only, NOT opened
```

## 4. Topology (verified)

```text
context   = (dataset, subject, panel, seed, level)          = 944  (118 subjects × 8)
candidate = (regime, trajectory_order)                       = 81   (1 ERM + 40 OACI + 40 SRC)
canonical candidate order = ERM:0, OACI:1..40, SRC:1..40
Semantics B: one physical construction label → its 8 (panel×seed×level) contexts
contexts by dataset: Cho2017 160, Lee2019_MI 176, PhysionetMI 608
n_trials per context: min 18, max 100 (Σ = 38,184)
```

## 5. Results — fail-closed invariants (all passed)

```text
contexts                        944
candidates / context             81
construction trials           4,773
context-trials               38,184   = 4,773 × 8
contributions             3,092,904   = 38,184 × 81
binary probability scalars 6,185,808   = 3,092,904 × 2
construction ⟂ evaluation overlap  0
```

## 6. Output (three separate physical roots)

```text
root: /projects/EEG-foundation-model/yinghao/oaci-c86l-development-field-v1  (62 MB)
  acquisition_unlabeled_pool/   944 context .npz  — probabilities[n,81,2] only, NO labels (client-visible)
  query_contribution_store/     944 context .npz  — label-derived nll/correct/confidence/conf_bin/
                                                    signed_calibration + true_label (server-private)
  acquisition_label_oracle/labels.csv             — 4,773 sealed construction labels
  C86L_RESULT_MANIFEST.json   sha256 fea856165aa36a9dfd846618e2a3fc929f2f62f3fc65940d3b08b83b8cb50ce3
  C86L_CONTEXT_INDEX.json     — 944 contexts
```

## 7. Verification

```text
isolation      : client pool carries no label field; contribution/oracle are label-derived/sealed
oracle SHA     : output acquisition_label_oracle/labels.csv == source construction view (fdf36052…) ✓
correctness    : recomputed nll = −log p(true class) and correct = 1[argmax==label] MATCH the store ✓
arithmetic     : 944 / 81 / 4,773 / 38,184 / 3,092,904 / 6,185,808 all asserted at build time ✓
disjointness   : construction ⟂ evaluation trial-id overlap = 0 (checked vs evaluation label view) ✓
```

## 8. Boundary and next step

C86L is complete and frozen; the durable identity is committed at
`oaci/reports/C86L_RESULT_IDENTITY.md`. The next stage is **C86D** (running P0 vs
A1/A2H on this field under the pre-registered probe criteria) — **NOT authorized**.
C86H, C87, and manuscript work are **NOT authorized**; C86H does not auto-start
C87. Any C86D execution requires its own separate direct authorization.
