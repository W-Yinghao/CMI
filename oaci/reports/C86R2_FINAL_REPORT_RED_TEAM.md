# C86R2 Final Report Red Team

## Verdict Matrix

| Check | Verdict | Evidence |
|---|---|---|
| Protocol precedes new metadata inspection | PASS | protocol commit `0da4ec39`; timing audit and zero counters |
| Historical C86R blocker preserved | PASS | supersession ledger records one correctly proven adult cohort and no rewrite |
| Age definition | PASS | age at recording `>=18`; no present-day age |
| Deterministic subset rule | PASS | all verified adults included; minors and unknown ages excluded; minimum 12 |
| Yang participant mapping | PASS_FAIL_CLOSED | constant public age conflicts with source range and anonymization disclosure |
| Kumar participant mapping | PASS_FAIL_CLOSED | mean/SD and constant year-like field cannot prove adult age |
| Catalog universe frozen | PASS | MOABB 53, EEGDash 824, NEMAR 760; hashes and page identities recorded |
| Native cohort deduplication | PASS | 149 public candidate rows reduce to 79 native datasets; mirrors count once |
| All candidates dispositioned | PASS | 82 interface rows after native `ds007221` task expansion; no blanks |
| No outcome-based eligibility field | PASS | eligibility schema has zero outcome fields |
| `ds007221` task isolation | PASS | hybrid passes; Graz/SSMVEP-MI fail task rule; hybrid-online fails target count |
| Adult subject map | PASS | exactly `sub-37` through `sub-73`, 37 adults, frozen digest |
| All passing cohorts included | PASS | Brandl and `ds007221` hybrid both primary; no post-screen cap |
| Minimum two cohorts | PASS | 2 cohorts and 53 target clusters |
| Common field interface | PASS_METADATA_ONLY | five source/target rows share the fixed 11-channel interface |
| Resource arithmetic | PASS_METADATA_ONLY | 424 contexts, 34,344 slices, 1,296 unit-cohort artifacts |
| License boundary | PASS_WITH_FUTURE_GUARD | Brandl restrictions and `ds007221` CC0 recorded; terms replay still required |
| Stale V2 downstream use | PASS_BLOCKED | C86LP guard rejects V2 without V3 |
| Protected boundary | PASS | zero EEG, labels, predictions, active runs, training, synthetic results, GPU |
| Authorization boundary | PASS | no C86 execution lock or authorization record |

## Adversarial Findings

1. The first readiness generation expected 89 deduplicated public candidates.
   Exact replay found 79. The generator failed before publication; the count
   contract and tests were corrected without changing any cohort decision.
2. Treating Yang's constant public value as age would create a false adult
   subset because the primary source includes age 17 and the supplement says
   age was removed. C86R2 correctly excludes all unknown-age IDs.
3. Treating Kumar's value 2020 as age or inferring adulthood from a mean and
   standard deviation would violate the rule. Kumar remains fail-closed.
4. Filtering feet/rest from `ds007221` Graz would violate the native binary
   rule. C86R2 instead uses the separately acquired native hybrid interface.
5. Treating OpenNeuro and NEMAR mirrors as separate cohorts would falsely
   inflate cohort count. The deduplication ledger binds one native cohort.
6. Limiting the result to a preferred pair after screening would permit
   cherry-picking. The truth table requires every passing interface to be
   included.
7. Metadata feasibility is not field execution readiness. The new 11-channel
   zoo, protected labels, engineering canary, license replay, and resources all
   remain future locked work.

## Final Disposition

The adult-only population now satisfies the prospective multi-cohort input
rule without relaxing age eligibility and without opening real data.

```text
C86_ADULT_UNTOUCHED_MULTI_COHORT_ELIGIBILITY_RESOLVED_READY_FOR_C86LP_PROTOCOL_REVIEW
```

This is readiness for protocol review only, not authorization for any C86
real-data stage.
