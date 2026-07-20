# C78S Final Report Red Team

Result: **PASS** (`20/20`, zero blockers).

This report-level audit ran after the scientific result red team and all four
regression suites. It independently checked the final prose against the compact
tables and result JSON. In particular:

- Active primary taxonomy is exactly `H3 + H4 + H5`.
- H1 is explicitly reported as a counter-result because split-label construction
  passed the registered material-actionability gate.
- H2 and H6 remain inactive; their descriptive effects are not promoted.
- H4/H5 remain registered-candidate nonqualification statements, not universal
  impossibility claims.
- C79 remains protocol-only and unauthorized; seed 4 and the oracle remain untouched.
- All manifest table hashes and the final report/regression hashes replay exactly.

The first checker invocation used a literal contiguous-string assertion across a
Markdown line wrap and stopped. The corrected checker normalized the line break.
The audit also rejected an unsupported initial description of the broad-suite
skip. A dedicated `pytest -rs` replay (`893168`) established that the skip is the
intentional finalized-C78F guard, not a C78S route branch. Both corrections were
report/provenance-only; they changed no data, statistic, taxonomy, or scientific
result.
