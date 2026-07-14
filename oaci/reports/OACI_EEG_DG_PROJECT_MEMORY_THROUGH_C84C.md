# OACI EEG-DG Project Memory Through C84C

C84C authorized replacement job `895441` executed the exact three-dataset
engineering canary bound by C84 canary protocol V4 and execution lock V3. It
completed on `node43` in `01:46:19` with exit `0:0` and no Slurm restart.

The complete gate passed:

```text
datasets:                         Lee2019_MI / Cho2017 / PhysionetMI
scope:                            panel A / seed 5 / level 0
training phases:                  9 / 9
candidate units:                243 / 243
checkpoint/state/sidecar replay:243 / 243
strict-source audit artifacts:  243 / 243
target-unlabeled artifacts:     243 / 243
```

Every real view used the exact 20-channel montage in locked order at 160 Hz and
480 samples under the half-open `[0,3)` epoch. No interpolation, channel
synthesis or dataset-specific mask occurred. Loaded subject sets exactly match
the locked 12-source-train, 4-source-audit and 1-target canary identities per
dataset and are disjoint.

The maximum persisted float32 linear replay error was
`6.67572021484375e-6`, below the locked `1e-5` tolerance. Softmax, repeat-logit
and repeat-z maximum errors were zero under `1e-6`. All three deterministic
prefix fingerprints passed.

Target-y access, target-label fields, target scientific metrics, target-outcome
retention/retry, construction/evaluation view access and same-label-oracle
access remained zero. No target accuracy, selector score, regret, Q1/Q2 or
label-budget result exists. C84C is an engineering canary, not external-validity
evidence.

Historical failed job `895366` remains preserved. Job `895441` used a fresh
authorization and external root; it reused neither authorization nor artifacts.
The complete manifest SHA-256 is
`530471ef370d5fa13a88e7e53cf1add558b8444b66675496187aa192b0606f2b`.

Regression verification passed at result commit `2f541e5`: focused 112, C65
598, C23 1,009 and full 1,933 tests. The cumulative suites have one explained
C78F skip and three historical C79 authorization-fixture deselections; all
stderr files are empty. Final report red-team passed 56/56.

Final gate:
`C84C_COMPLETE_ENGINEERING_REPLAY_PASSED_C84F_REVIEW_REQUIRED`.

C84F and C84S remain unauthorized. C84F requires a separate prospective field
execution lock, PM review and fresh direct PI authorization.
