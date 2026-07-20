# C34 Red-Team Verification

C34 was red-team reviewed before commit for threshold artifacts, scalarization artifacts, taxonomy over-inclusion,
and guardrail violations. The review found no hard blocker and no guardrail breach.

## Review outcome

- No checkpoint-hash leakage in C34 reports/tables.
- Seeds are restricted to the committed OACI set (`0,1,2`); no seeds `[3,4]`.
- Frozen C19 config hash remains `664007686afb520f`.
- C34 does not train, re-infer, tune scores, create a selector, or emit selected-checkpoint artifacts.
- Final taxonomy remains `M2 + M7 + M8`; `M1` and `M6` are not established.

## Red-team caveats incorporated

- `real_endpoint_regret` is defined under the fixed C34 scalar/norm summaries, not as pure Pareto dominance.
- The main report now gives the endpoint-vector caveat explicitly: among 153 nearest continuous-better pairs, 72 are
  raw Pareto-nonworse, 81 move at least one raw endpoint backward, and 33 have negative joint-min delta.
- `threshold_only = 0` is the strict tiny-difference artifact only. It does not erase broader binary-label tradeoffs:
  among 81 binary misses, 0 are tiny-threshold, 15 are endpoint tradeoffs, and 3 are scalar-or-norm worse.
- `M2` is read as a substantial selected-pair minority, not a global claim that the source objective points mostly
  backward.
- `M8` is read as scalar-better local alternatives often trading endpoints, not independent evidence that selected
  OACI is Pareto-dominated.
- `M6` is not established only under the C33-style source-insensitivity gate. Gauge jumps are common, but the
  gauge-unseen-by-source fraction is low.

## Verification after red-team edits

- `python -m py_compile oaci/continuous_regret/*.py` passed.
- `python -m pytest oaci/tests/test_c34_continuous_regret.py -q` passed: 8 tests.
- `python -m pytest oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c3[0-4]_*.py -q` passed: 123 tests.
- `git diff --check` passed.
