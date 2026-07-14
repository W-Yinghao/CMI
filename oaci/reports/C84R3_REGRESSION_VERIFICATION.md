# C84R3 Regression Verification

All jobs ran CPU-only in `c84c-eeg2025-v3-exact` at commit `91e39690f4de39b13465d6505fa292793f75482e`.

| Suite | Job | Result |
|---|---:|---|
| focused | 895371 | 102 passed |
| C65 | 895372 | 588 passed, 1 skip, 3 deselected |
| C23 | 895373 | 999 passed, 1 skip, 3 deselected |
| full | 895374 | 1,923 passed, 1 skip, 3 deselected |

The skip is finalized C78F. The three deselections are historical C79 authorization-state tests. All stderr files are empty.
