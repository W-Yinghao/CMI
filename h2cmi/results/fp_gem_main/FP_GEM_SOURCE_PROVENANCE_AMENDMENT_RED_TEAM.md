# P12A Source-Provenance Amendment Red-Team Review

Status: **PASS AFTER REMEDIATION**. Review performed before the replacement smoke and before any target-performance observation.

## Adversarial Findings

1. **Resolved, blocking:** the original exact-state SHA gate could not establish checkpoint identity because P9 persisted hashes but no checkpoint weights. The clean V100 retrain matched the committed P9 training trace to printed precision through epoch 10, then showed a `0.0002` final-loss drift and a different byte SHA. Treating that SHA as recoverable would have made P12 impossible without improving validity.
2. **Resolved, blocking:** direct reuse of P9 result rows would not be an exact-checkpoint head-to-head after an exact-configuration retrain. The amended protocol reruns the four frozen official controls and both GEM methods from one persisted checkpoint per unit. No method or hyperparameter was added.
3. **Resolved, medium:** the first amendment draft compared only unit keys. The final gate also binds every P9 reference hash, repaired split hash/count, and hardware-family assignment row by row.
4. **Resolved, medium:** the first amendment draft trusted copied P9 settings in JSON. The runner now enforces code-level constants for source training and official adaptation, and every cached checkpoint sidecar binds the P9 runner/config and P12 config hashes.
5. **Pass:** target adaptation receives dummy-zero labels only. The sole live `ep.y[eval_idx]` read occurs after SPDIM geodesic, SPDIM bias, Joint-GEM, and FP-GEM fits; smoke returns before that read and records no accuracy or bAcc.
6. **Pass:** datasets, seeds, repaired split, FP-GEM/Joint-GEM definitions, bootstrap policy, and interpretation grid are unchanged. P7/P9/P10/P11 canonical artifacts are untouched.

## Independent Checks

- direct tests: `7/7 PASS`
- CPU dry-run: `PASS`, 189 units
- methods: source-only TSMNet, RCT, SPDIM geodesic, SPDIM bias, Joint-GEM, FP-GEM
- expected rows: 378 new-method + 756 same-checkpoint controls = 1,134
- direct P9 rows reused: `0`
- P9 result SHA-256: `95b8f69556a140dc020415753c9694cf9ebdeed1abb0766dd24f523c491289c3`
- repaired manifest semantic SHA-256: `231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e`
- runner SHA-256: `720b91b1b43cdf6a983be1cb8413430a06b98d6f4923166fa14614041ec46abd`
- analyzer SHA-256: `fe5be8b3336ece1814ebbee4a93c94f3cd524823baed37f2e5bd2716e30419b7`
- config SHA-256: `d44fd98aa5913eb45908b7fd398b04e5a268dd4aaa75f15bcc96819f424bf165`
- Slurm accounting script calls: `0`

## Residual Boundary

P12 can support an exact-P9-configuration, within-unit same-checkpoint comparison. It cannot claim that the original P9 checkpoint weights were loaded, and it must not substitute committed P9 performance rows for the rerun controls when state hashes differ.
