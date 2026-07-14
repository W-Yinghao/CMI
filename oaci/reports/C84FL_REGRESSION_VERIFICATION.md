# C84FL Regression Verification

All suites ran on the committed blocker-audit base
`e141d2a7531d15ac6a420bdb3ee9163395e57407` in the exact C84 environment:

```text
/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/bin/python
partition: cpu-high
allocation per job: 48 CPU, 96 GiB, GPU 0
```

| Suite | Job | Result | stderr |
|---|---:|---|---:|
| focused C84FL/C84 | 895694 | 126 passed | 0 bytes |
| C65-C84FL | 895695 | 612 passed, 1 skipped, 3 deselected | 0 bytes |
| C23-C84FL | 895696 | 1,023 passed, 1 skipped, 3 deselected | 0 bytes |
| full OACI | 895697 | 1,947 passed, 1 skipped, 3 deselected | 0 bytes |

The skip is
`test_c78f_full_seed3_field.py:174`, because the C78F red-team is already
finalized. The three deselections are the established C79P authorization-state
tests that require a future external authorization fixture. No C84FL or C84F
test was skipped or deselected.

The leading-numeric suite parser included both new C84FL contract files. All
jobs exited `0:0`; stdout and empty-stderr hashes are recorded in
`c84fl_tables/regression_verification.csv`.

These passing regressions verify the fail-closed blocker state. They do not
resolve the missing level-1 intervention and do not authorize C84F.

## Post-report integrity verification

After adding the complete Markdown/JSON report and SHA-256 sidecar, the two
focused C84FL contract files were rerun locally in the same exact environment:

```text
15 passed
```

The added check replays both report hashes, parses the JSON, verifies the
operative failure gate and 972-unit level-1 impact, and confirms that no C84F
execution lock was created. This documentation-only verification accessed no
real EEG, label view, training path, forward path, or GPU.
