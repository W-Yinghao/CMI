# C84F Multi-Dataset Dual-Level Field

This file is the concise execution summary. The authoritative standalone
lifecycle report is `C84F_OVERALL_REPORT.md` with SHA-256
`f80089fa03a64da5b2137e005d86eec2b282b4ab5ea33206f2f2a96ac321fe0c`;
its machine-readable companion is `C84F_OVERALL_REPORT.json` with SHA-256
`edb6ffb73e2f65ce56102f75abbe6ee447ca9dbf1cdddb7631f0ecbfa0b30f47`.

## Final Gate

`C84_MULTI_DATASET_DUAL_LEVEL_FIXED_ZOO_FIELD_EXECUTED_AND_MANIFESTED_ANALYSIS_NOT_STARTED`

Authorized target-only job `897048` completed on the V100 partition at node11.
The application ledger, rather than an unavailable scheduler accounting query,
is the terminal-state authority. It records an atomic complete event and
complete-manifest SHA-256
`cfffcac1a55148941b809b69bed2c9a8957a94729ed7f2c2c29ed8d48c0134d8`.

## Frozen Inputs

- Model field: 1,944 units and 7,776 checkpoint/optimizer/sidecar/source-audit
  files, manifest `d8931b81a3d68f4b1e098ac6e3ede3cd44cdb6c70cdef9f18a76e0a8c62ecdb2`.
- Dual-canary replay: 2,430 files across 486 accepted units.
- Target inputs: 118 subjects and 9,621 label-free rows.
- Raw manifest: `9539747e903dfe67295ee04a97441b85c0bb2179c9ef1bd2177788865e0ba5fd`.
- Trial registry: `52526aaf7d9bd941bac693a0947971dc35b9083c1c783619f97055926aceabb8`.
- All six NPZ and five context indices from failed job `896550` were replayed
  as historical evidence and rejected from reuse.

## Complete Field

```text
target artifacts:             1,944 / 1,944
context/digest sidecars:      1,944 / 1,944
target contexts:                944 / 944
candidate-context slices:    76,464 / 76,464
canary witnesses:               486 / 486
target subjects:                118 / 118
target registry rows:         9,621 / 9,621
```

The target NPZ payload is 48,018,748,054 bytes. Context sidecars are
24,209,257 bytes, and the complete external root is 48,058,685,017 bytes over
3,896 files. A post-execution stream replay verified all 1,944 NPZ hashes and
all 1,944 sidecar hashes, exact directory membership, all 21 array digests,
76,464 accumulated contexts and 486 canary statuses.

## Numerical Gates

```text
same-GPU/PyTorch direct classifier max: 0
saved Wz_plus_b versus logits max:      0
saved softmax max:                      1.1920928955078125e-07
repeat logits max:                      0
repeat z max:                           0
operative tolerance:                    1e-06
```

Cross-backend diagnostic maxima were `4.57763671875e-05` for CPU PyTorch
float32, `4.1961669921875e-05` for NumPy float32 and
`1.1513513614502813e-05` for NumPy float64. These are finite diagnostics only;
they did not affect retention, retry, model state, thresholds or field identity.

## Runtime

Application wall time was 7,322.999 seconds, approximately 2:02:03:

```text
guard/import/frozen barrier:     12.452 s
target X and registry replay: 1,240.043 s
instrumentation:              5,806.071 s
aggregate/atomic publication:   264.433 s
```

The 5-hour allocation was not approached, so no sharding was required. The
locked V100 was retained instead of A100/H100/L40S because the execution lock
bound that architecture and the prospective runtime estimate was below five
hours; changing architecture would have required a new lock and strict canary
replay review. The stderr file contains 20 identical, disclosed loader warnings
about zero-buffer continuous-data edge effects and no traceback or
runtime-failure marker.

Two operator-only preflight summary invocations failed after the protected
runtime guard had already passed because the summaries assumed incorrect return
shapes. Neither invocation consumed authorization, imported the protected data
path or accessed EEG. The empty root created by the first root-availability
check was removed with `rmdir`; the final no-data preflight then passed. These
were reporting wrappers, not target-stage application attempts, and their logs
remain outside Git.

## Verification

Final report red-team passed 68/68 and overall-report reconciliation passed
30/30. Post-execution regressions passed 30 focused, 758 C65, 1,169 C23 and
2,093 full-OACI tests. The cumulative suites retain one explained
finalized-C78F skip and three established C79 authorization-state deselections.
Every regression stderr file is empty.

## Protected Boundary

Model retraining, training phases, target-y operations, target-label fields,
construction/evaluation labels, oracle access, selector scores, scientific
statistics and target scientific metrics were all zero. C84S remains
unauthorized and has no execution lock. This milestone creates a frozen field;
it does not establish target accuracy, selector performance, Q1/Q2, a label
frontier, level effects, cross-dataset recurrence or external validity.
