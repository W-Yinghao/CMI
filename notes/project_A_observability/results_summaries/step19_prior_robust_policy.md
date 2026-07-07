# Step 19 — prior-robust adaptation policy

Scope: prior-robust adaptation policy over declared L1 prior sets (C15); not SOTA. worst-case decision rule over DECLARED L1 prior-uncertainty sets (C15); class deltas are oracle/evaluation-only; not a deployable selector; does NOT identify the actual target prior. No SOTA.

- runs: **54** · robust-prior safe adaptation exists (any): **False** · robust adapt never uniform-harmful: **True**
- best prior-robust policy: **none**
- contract required **C15** · actual target prior identified **False**

| ρ | τ | adapt_cov | robust_harm_block | abstain | harm@adapt(uniform) |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.05 | 0.0 | 0.2037 | 0.7963 | None |
| 0.05 | 0.1 | 0.0 | 0.0741 | 0.9259 | None |
| 0.05 | 0.2 | 0.0 | 0.0 | 1.0 | None |
| 0.1 | 0.05 | 0.0 | 0.1667 | 0.8333 | None |
| 0.1 | 0.1 | 0.0 | 0.0556 | 0.9444 | None |
| 0.1 | 0.2 | 0.0 | 0.0 | 1.0 | None |
| 0.2 | 0.05 | 0.0 | 0.1481 | 0.8519 | None |
| 0.2 | 0.1 | 0.0 | 0.0556 | 0.9444 | None |
| 0.2 | 0.2 | 0.0 | 0.0 | 1.0 | None |
| 0.5 | 0.05 | 0.0 | 0.037 | 0.963 | None |
| 0.5 | 0.1 | 0.0 | 0.0 | 1.0 | None |
| 0.5 | 0.2 | 0.0 | 0.0 | 1.0 | None |

> worst-case decision rule over DECLARED L1 prior-uncertainty sets (C15); class deltas are oracle/evaluation-only; not a deployable selector; does NOT identify the actual target prior. No SOTA.
