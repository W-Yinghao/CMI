# C86P Regression Verification

## Execution Context

```text
repository commit:
  5bf5d08e0bf5373ce401776354d1996fd3fe2058

HEAD == origin/oaci:
  true

Python:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/bin/python

CPU / GPU:
  32 / 0
```

All accepted runs used the clean implementation commit before final report
files were added.

## Accepted Runs

| Suite | Result | Pytest time | Wall time | stdout SHA-256 | stderr bytes |
|---|---|---:|---:|---|---:|
| focused | 395 passed, 2 deselected | 10.62 s | 11 s | `7d81e2bb3886058452bb38ec7529a1c0dcb61bcf45f082052672f2599bb26634` | 0 |
| C65 | 1,103 passed, 1 skipped, 12 deselected | 91.69 s | 94 s | `bedc5a3307af35a1430ce0e6aa9ce3c407d8815f1730d0e49f00bf9d03471b23` | 0 |
| C23 | 1,514 passed, 1 skipped, 12 deselected | 120.30 s | 122 s | `127fc75ae8e7f64c5f683f209ea9295c31b81ac7019904fbc4f793496fab255f` | 0 |
| full OACI | 2,460 passed, 1 skipped, 12 deselected | 323.45 s | 326 s | `3328b9f11dd233a4ad5da6e809119dd95247b9cc45c3b5eb65cce0bf545af3c2` | 0 |

Every accepted stderr has SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
The single cumulative skip is the finalized historical C78F check at
`oaci/tests/test_c78f_full_seed3_field.py:174`.

## Deselection Accounting

The cumulative suites deselected exactly twelve historical readiness-only
assertions:

```text
3 C79 unauthorized-adapter assertions;
1 C85P no-C85 real-data/active-lock assertion;
1 C85R no-C85T-result assertion;
1 C85EP no-C85E implementation/lock/authorization assertion;
1 C85URP immutable downstream-absence assertion;
1 C85URP no-C85U/C85E authorization assertion;
1 C85UR1 no-C85U/C85E authorization assertion;
1 C85TR2 no-C85T authorization/result assertion;
1 C85VP no-C85V authorization/result assertion;
1 C85EP2 no-C85E authorization/execution assertion.
```

Each was true at its historical milestone and later superseded by an accepted
scope-specific authorization. No C86P test, current C85E result test, scientific
contract test, or runtime-boundary test was deselected. Focused collected only
two of these historical assertions, hence its two deselections.

## Preserved Pre-Acceptance Attempts

The first focused run passed 20 tests but emitted one invalid-regex warning; the
regex was corrected. After the small-budget Jeffreys operationalization, one
focused assertion used an incorrect hand-calculated expected value and produced
`1 failed, 19 passed`; the implementation value was independently rederived and
the test expectation corrected. The final focused C86-only set is 22/22 passed.

The metadata audit also preserved two pre-readiness corrections: Yang2025's 2C
variant was added after source inspection, and Dreyer2023 was removed from
untouched confirmation after historical access evidence was found. Both
corrections preceded the accepted implementation commit and used no outcome.
An initial four-suite all-pass run at `aa0d1f24` was retained externally, then
superseded and repeated after the synthetic generator contract was fully
operationalized.

## Static And Semantic Replay

```text
5 / 5 protocol and additive sidecar hashes replayed;
53 / 53 imagery catalog rows;
all catalog binary rows loader-source audited;
3 / 3 final untouched interfaces selected with no performance input;
24 / 24 required registries;
11 / 11 synthetic scenarios with zero registered draws;
zero MNE/MOABB-runtime/Torch/project-data imports;
zero C86 execution locks, authorizations, or results;
zero new EEG, label, training, GPU, or active-acquisition access.
```

## Verdict

```text
FOCUSED_C65_C23_FULL_ACCEPTED
ACCEPTED_STDERR_EMPTY
C86_ACTIVE_TESTING_DEVELOPMENT_CONFIRMATION_AND_UNTOUCHED_POPULATION_PROTOCOL_LOCKED_READY_FOR_PI_REVIEW
```
