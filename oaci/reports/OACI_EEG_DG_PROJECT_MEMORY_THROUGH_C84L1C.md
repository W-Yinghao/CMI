# OACI EEG-DG Project Memory Through C84L1C

C84L1C replacement job `896066` completed the fixed-panel level-1 engineering
canary on Lee2019_MI, Cho2017 and PhysionetMI. Scope was panel A / seed 5 /
level 1, 243 candidate units and nine training phases.

The fixed deleted cells were Lee subject 31, Cho subject 17 and Physionet
subject 103, always `left_hand`. Each dataset retained exactly 23/24 support
cells; minimum retained support was 50, 100 and 21 respectively. Paired model
initialization and accepted C84C level-0 plan replay passed.

All 243 checkpoints, optimizer states, sidecars, strict-source artifacts and
target-unlabeled artifacts passed persisted byte/hash and numerical replay. The
maximum in-memory linear error was `1.0967254638671875e-5` and the maximum
persisted linear error was `3.337860107421875e-6`, both below `2e-5`. Softmax,
repeat-logit and repeat-z errors were zero below `1e-6`.

Target-y, label fields, training target rows/labels, construction/evaluation/
oracle access, scientific metrics and outcome-driven retention/retry were all
zero. C84L1C computed no target accuracy, selector score, regret, Q1/Q2,
label-budget result or level comparison.

Complete manifest SHA-256:
`3cf1366ccf40efc82a6bb2ffef56045e83c0f0e9670429973f23252371ad1c18`.
Result JSON SHA-256:
`5bcccf351704c427d148ca1f44de26ef7e0b137d8de56aa0cf9ca3f6723abaf5`.

Scheduler evidence is `squeue` plus the application attempt ledger; `sacct`
was not used. Job 896066 used one V100, eight CPUs and 64 GiB and completed in
4355.285 seconds. Historical failed job 895928 remains preserved and neither
its authorization nor artifacts were reused.

Regression: focused 183; C65 669; C23 1,080; full 2,004 passed. Cumulative
suites have one explained C78F skip and three established C79 deselections; all
stderr is empty. Final red-team: 68/68 PASS.

After PM review, C84C plus C84L1C provide 486 reusable model/state/source-audit
units and 18 phases. The six target canary contexts / 486 slices remain subset
witnesses only. C84FL2 must still lock 1,458 remaining units / 54 phases and all
944 target contexts / 76,464 slices. C84F and C84S remain unlocked and
unauthorized.

Gate:
`C84L1C_COMPLETE_ENGINEERING_REPLAY_PASSED_C84FL2_REVIEW_REQUIRED`.
