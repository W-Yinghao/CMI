# C84C Engineering Canary Result

Authorized replacement job `895441` completed with exit `0:0` in `01:46:19` on `node43`. The complete external manifest SHA-256 is `530471ef370d5fa13a88e7e53cf1add558b8444b66675496187aa192b0606f2b`.

The engineering gate passed: 243/243 candidate units, 9/9 training phases, 243/243 checkpoint/state/sidecar replays, 243/243 strict-source audit artifacts, and 243/243 target-unlabeled artifacts. All three datasets returned the exact 20-channel montage in locked order at 160 Hz with 480 half-open `[0,3)` samples; no channel interpolation or synthesis occurred.

The maximum persisted linear replay error was `6.67572021484375e-06` under the locked `1e-5` tolerance. Softmax, repeated-logit, and repeated-z maximum errors were all `0` under the strict `1e-6` tolerance.

Target-y access, target-label fields, construction/evaluation view access, target scientific metrics, target-outcome retention, and target-outcome retries were all zero. No selector score, target accuracy, regret, Q1/Q2 result, or label-budget frontier was computed. The same-label oracle remained closed.

The nonempty Slurm stderr is fully explained by 102 Physionet HTTPS download warnings, 17 Cho continuous-stack edge-effect notices, and progress output; it contains no traceback or runtime-failure marker. Failed job `895366` remains preserved, and none of its authorization or artifacts was reused.

Gate: `C84C_COMPLETE_ENGINEERING_REPLAY_PASSED_C84F_REVIEW_REQUIRED`. C84F and C84S remain unauthorized.
