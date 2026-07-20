# C84L1R1 Regression Verification

All jobs ran CPU-only in `c84c-eeg2025-v3-exact` at commit
`d7ba0c69b193736ab6667e0272d9287a171683e3`, with 48 CPUs, 96 GiB and no GPU.

| Suite | Job | Result | Runtime | Stderr |
|---|---:|---|---:|---:|
| focused | 896000 | 175 passed | 4.42 s | 0 bytes |
| C65 | 896001 | 661 passed, 1 skip, 3 deselected | 37.04 s | 0 bytes |
| C23 | 896002 | 1,072 passed, 1 skip, 3 deselected | 124.85 s | 0 bytes |
| full | 896004 | 1,996 passed, 1 skip, 3 deselected | 265.43 s | 0 bytes |

The skip is
`oaci/tests/test_c78f_full_seed3_field.py:174`, because C78F is finalized. The
three deselections are the established historical C79 authorization-state
tests and conceal no C84L1R1 path. Every stderr file is empty. `squeue` shows
none of jobs `895928`, `896000`, `896001`, `896002` or `896004` active.
