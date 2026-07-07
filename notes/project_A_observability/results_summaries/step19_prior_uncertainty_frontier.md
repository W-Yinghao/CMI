# Step 19 — prior-uncertainty robustness frontier

Scope: prior-uncertainty robustness frontier (declared set, contract C15); not SOTA. robust gain bounds are over a DECLARED prior-uncertainty set (contract C15); class deltas are oracle/evaluation-only; the actual target prior is NOT identified (that needs TU-1). No SOTA.

- runs: **54** · median L1 flip-radius from uniform **0.165151** (q25 **0.073094** / q75 **0.304194**) · unflippable over simplex **2**
- flip within L1 ≤0.10 **0.2778** · ≤0.20 **0.6296** · ≤0.50 **0.8704**
- prior-uncertainty contract required **C15** · actual target prior identified **False**

| ρ | robust_harm | ambiguous | robust_benefit |
|---:|---:|---:|---:|
| 0.0 | 0.8519 | 0.0 | 0.1481 |
| 0.05 | 0.7593 | 0.1667 | 0.0741 |
| 0.1 | 0.6852 | 0.2778 | 0.037 |
| 0.2 | 0.3704 | 0.6296 | 0.0 |
| 0.3 | 0.2963 | 0.7037 | 0.0 |
| 0.5 | 0.1296 | 0.8704 | 0.0 |
| 1.0 | 0.037 | 0.963 | 0.0 |
| 2.0 | 0.037 | 0.963 | 0.0 |

> robust gain bounds are over a DECLARED prior-uncertainty set (contract C15); class deltas are oracle/evaluation-only; the actual target prior is NOT identified (that needs TU-1). No SOTA.
