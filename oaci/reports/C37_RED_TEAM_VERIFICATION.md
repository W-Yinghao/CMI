# C37 Red-Team Verification

C37 was red-team reviewed before commit for exact-trace orientation, selector/proxy overclaim, artifact hygiene,
target-label leakage, and Slurm split reproducibility.

## Findings

- P0 selected-UCL identity passed exactly: 3/3 selected candidates reproduced persisted point and bootstrap UCL with
  absolute diff 0.0 under the frozen 1e-9 tolerances.
- Better-candidate exact UCL recovery completed for 38/38 unique C35 preference-robust alternatives.
- Pairwise exact UCL orientation is internally consistent: 114/114 recovered pairs have
  `better_ucl - selected_ucl > 1e-9`, so the frozen selector UCL prefers the artifact-selected candidate, while the
  imported C35 utility-cone target endpoint label prefers the alternative.
- Taxonomy was not over-included for trace incompleteness: T7 is false because P0 passed and all better UCLs recovered;
  T8 is only a diagnostic exact-trace misdirection claim, not a selector or deployment claim.
- Source-train feature availability is worker-proven for 38/38 better candidates. Selected feature availability is
  only rechecked for the 3 P0 identity candidates; C37 does not claim full selected-candidate feature enumeration.
- No target labels were loaded for UCL replay. Target endpoint quantities enter only through imported C35 diagnostic
  pair labels.
- Reports/tables are compact; largest C37 table is about 51 KB. No large JSON payload or selected-checkpoint method
  artifact is emitted.
- Committed reports/tables do not emit candidate checkpoint hashes. Code reads frozen hashes internally only to locate
  Phase-A source-train feature entries for exact replay.

## Verification

- `bash -n oaci/slurm_c37_trace_recovery.sh` passed.
- `python -m compileall -q oaci/selector_trace_recovery oaci/tests/test_c37_selector_trace_recovery.py` passed under
  `/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python`.
- `python -m pytest oaci/tests/test_c37_selector_trace_recovery.py -q` passed: 6 tests.
- `python -m pytest oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c3[0-7]_*.py -q` passed: 140 tests.

## Boundary

C37 remains read-only and diagnostic-only. It does not train, re-infer, tune scores, create a selector, emit a selected
checkpoint artifact, add BNCI2014_004, run seeds [3,4], or claim OACI rescue.
