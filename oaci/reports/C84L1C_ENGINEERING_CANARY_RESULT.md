# C84L1C Engineering Canary Result

Authorized replacement job `896066` completed the application in `4355.285` seconds on `node42`. Scheduler state was tracked with `squeue`; `sacct` was not used. The complete manifest SHA-256 is `3cf1366ccf40efc82a6bb2ffef56045e83c0f0e9670429973f23252371ad1c18`.

The engineering gate passed: 243/243 units, 9/9 phases, 243/243 checkpoint/optimizer/sidecar byte replays, 243/243 strict-source artifacts, and 243/243 target-unlabeled artifacts. Each dataset had exactly one fixed left-hand source-support cell removed, 23/24 retained support cells, paired model initialization, and exact level-0 plan replay.

The maximum in-memory float32 linear replay error was `1.0967254638671875e-05` under `2e-5`; the maximum persisted replay error was `3.337860107421875e-06`. Softmax, repeat-logit, and repeat-z maximum errors were zero under `1e-6`.

Target-y access, target-label fields, construction/evaluation/oracle access, target scientific metrics, target-outcome retention, and target-outcome retry were all zero. No target accuracy, selector score, regret, Q1/Q2, label-budget, or level comparison was computed. The 17 stderr lines are disclosed Cho stacking notices with no failure marker.

Failed job `895928` remains preserved; neither its authorization nor artifacts were reused. C84F and C84S remain unauthorized.

Gate: `C84L1C_COMPLETE_ENGINEERING_REPLAY_PASSED_C84FL2_REVIEW_REQUIRED`.
