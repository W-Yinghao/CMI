# C38 Red-Team Verification

C38 was red-team reviewed before commit for point-vs-UCL orientation, atom-decomposition overclaim, support artifact
claims, target-gauge overreach, and selector/proxy leakage.

## Findings

- The UCL decomposition is internally exact: for every row, `delta_ucl = delta_point + delta_width`.
- Exact UCL and point leakage both prefer selected in 114/114 rows. The UCL margin is point-dominant in 111/114 rows;
  uncertainty-only driving is not supported.
- Selection-UCL to source-audit leakage inversion is 51/114 (0.447), matching C37/C36 reconciliation.
- Source-rational target-wrong classification is 114/114 under recovered exact UCL and C37 source-Pareto conflict.
- Target gauge conflict is 105/114 (0.921). This is a diagnostic local target-gauge comparison; target-gauge factors
  remain non-source-only and are not a method.
- Atom-level class/domain/support leakage contribution tables are not persisted. C38 therefore establishes L10 and
  deliberately does not establish L3 or L4.
- Support/estimability edge-case artifact is not supported: all 38 selected/better pair keys are invariant across
  S0/S2/S3 regime copies, and each regime has 38/38 selected-UCL preferences.
- Reports/tables are compact; the largest C38 table is about 52 KB and the compact JSON is under 9 KB.
- Committed reports/tables do not emit candidate checkpoint hashes.

## Verification

- `python -m pytest oaci/tests/test_c38_leakage_objective_geometry.py -q` passed: 6 tests.
- `python -m pytest oaci/tests/test_c37_selector_trace_recovery.py oaci/tests/test_c38_leakage_objective_geometry.py -q` passed: 12 tests.
- `python -m pytest oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c3[0-8]_*.py -q` passed: 146 tests.
- `python -m compileall -q oaci/leakage_objective_geometry oaci/tests/test_c38_leakage_objective_geometry.py` passed.

## Boundary

C38 remains read-only and diagnostic-only. It does not train, re-infer, tune scores, change OACI/C19/C24/C37, run
feature selection, add BNCI2014_004, run seeds [3,4], create a selector, emit selected-checkpoint artifacts, or claim
deployable joint-good localization.
