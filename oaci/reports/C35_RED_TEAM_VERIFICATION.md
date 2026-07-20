# C35 Red-Team Verification

C35 was red-team reviewed before commit for endpoint orientation, C34S artifact gates, Pareto/scalarization
artifacts, utility-cone taxonomy, source/R3 direction claims, and guardrails. The review found no blocking issue.

## Review outcome

- Endpoint orientation is correct: C34 stores NLL/ECE as improvements, so C35 does not flip them again.
- C34S gates resolve the compact manifest, verify bytes/rows/SHA-256, reconstruct key C34 numbers, and reject legacy
  monolithic JSON dependency.
- `U1` is justified under the frozen raw utility grid: 114/153 nearest continuous-better pairs are robust at the
  80% weight-simplex gate.
- `U2` remains necessary because 81/153 nearest continuous-better pairs are Pareto-incomparable endpoint tradeoffs.
- `U3` is justified by 72/153 strict Pareto-better pairs.
- `U5` is only a substantial-minority source-misranking claim in preference-robust cases, not a mostly-backward
  source-score claim.
- `U7` is an R3 local preference-robust non-rescue claim, not a general target-unlabeled failure claim.
- `U8` is not established: robust/dependent/narrow ranges across frozen scalings are below the 0.20 gate.

## Caveats incorporated

- The main report now states that `preference_robust` means at least 80% of a discrete nonnegative raw utility grid
  with step 0.05; it is not a claim over every monotone utility.
- The report distinguishes strict Pareto dominance from utility-cone robustness and keeps the Pareto-incomparable
  tradeoff mass explicit.
- The source-direction report compares source/R3 rates to a frozen 0.5 local random baseline and avoids selector
  language.
- C34S gates are described as artifact-integrity checks; no-training/no-selector rows are code-audit assertions for
  this read-only path.

## Verification

- `python -m py_compile oaci/utility_cone/*.py` passed.
- `python -m pytest oaci/tests/test_c35_utility_cone.py -q` passed.
- `python -m pytest oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c3[0-5]_*.py -q` passed.
- `git diff --check` passed.
- C35 report guard and checkpoint-hash grep passed.
