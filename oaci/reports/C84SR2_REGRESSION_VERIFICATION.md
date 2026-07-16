# C84SR2 Regression Verification

Final accepted commit: `b08538d3f399c77bb188246d23472cb5fd39ded5`.

| Suite | Job | Result | Pytest time | stderr |
|---|---:|---|---:|---:|
| focused | 897866 | 242 passed | 6.11 s | 0 bytes |
| C65 | 897867 | 843 passed, 1 skipped, 3 deselected | 104.69 s | 0 bytes |
| C23 | 897865 | 1,254 passed, 1 skipped, 3 deselected | 99.52 s | 0 bytes |
| full OACI | 897868 | 2,178 passed, 1 skipped, 3 deselected | 482.60 s | 0 bytes |

The sole skip is the finalized C78F execution test. The three deselections are
the established C79 authorization-state tests. The corrected leading-numeric
suite parser remained active. All accepted jobs were CPU-only on `cpu-high` and
were monitored with `squeue`; `sacct` was not used.

The first regression attempt is preserved in
`c84sr2_tables/regression_attempt_ledger.csv`. Focused, C65 and C23 exposed the
same six historical tests whose exact lock-name sets stopped at V3. The initial
full job was canceled after that deterministic shared failure was established.
Those assertions were updated to recognize the additive V4 lock without
changing runtime code, scientific semantics or the V4 lock. All four suites
then passed from a clean pushed commit.
