# C81 Overall Report GitHub Audit Addendum

## Status

This is an additive audit addendum to `C81_OVERALL_REPORT.md`. It does not
rewrite, delete, supersede, or silently amend the historical C81 report or the
C81-E result identity.

The accepted C81 gate remains:

```text
C81-E_protocol_input_implementation_or_provenance_blocker
```

C81 produced no valid frozen baseline comparison. C81-A/B/C/D remain
unavailable, and C80E remains the latest valid scientific result.

## Audit Scope

The audit replays committed Git objects, source code, tests, compact ledgers,
and report schemas only. It does not load or inspect the frozen C81 selection
payload, open any evaluation-label view, reconstruct the failed process memory,
or compute a new scientific statistic.

## A. Initial Malformed Implementation Commit Identity

The repaired lock committed at `6633fd7` recorded this synthetic implementation
commit identity:

```text
d17ffa65a3e7cffeb5cac65292e7840787f6845d
```

That object is not reachable. The correct implementation commit is:

```text
d17ffa62a63b929d36d03f74e4ce79794cd9601b
```

The error was corrected additively before reauthorization by commit:

```text
29f4555b65273bf2329c0154704233cc746ce8f0
```

The correction changed only the malformed commit identity in the replacement
lock. The synthetic blob SHA-256 remained
`bd9e69691917b9e57f2ceaa50c52addb8dd373b4c36f97ce4335cceb672bd6e3`;
the scientific registry and implementation did not change, and evaluation
label access remained zero.

## B. Final CSV Insertion-Order Blocker

Authorized job `894958` constructed 672 method/control context rows in memory
after the frozen selection replay and after opening the registered evaluation
views. The row field sets were identical, but the insertion order differed.

Selector and ceiling rows used:

```text
standardized_regret
selected_utility
top1
top5
top10
coverage_top1
coverage_top5
coverage_top10
```

The analytic random-control row used:

```text
standardized_regret
selected_utility
top1
coverage_top1
top5
coverage_top5
top10
coverage_top10
```

`write_csv` compared `list(row)` with the first row's ordered keys. It therefore
raised `C81 table schema drift` before opening the first result table. This is
an implementation/schema defect, not evidence that a scientific field was
missing or that methods disagreed.

## C. Cross-Seed Same-Method Taxonomy Mismatch

The C81 protocol's scientific meaning requires one fixed method to support a
cross-seed claim. The implementation instead computed an A/B/C category within
each seed and compared only the two category labels:

```text
seed_category(q1_pass, q2_pass)
classify_taxonomy(seed3_category, seed4_category, LOTO_preserved)
```

Consequently, seed 3 could obtain category A from ATC while seed 4 obtained
category A from SND, and the implementation could still classify the result as
C81-A. The same defect applies to B: different improving methods could produce
matching seed-level B labels.

No C81 nonblocker taxonomy was executed because the CSV blocker occurred first.
The mismatch therefore does not change the historical C81-E gate, but it would
have made a successful C81-A/B classification scientifically invalid.

## D. Registered-Analysis Implementation Coverage Gaps

The base protocol registered Q1-Q5 and secondary measurement outputs. The
committed freeze path emitted only:

```text
method_context_results.csv
primary_comparison.csv
leave_one_target_out_sensitivity.csv
C81_FROZEN_FIELD_BASELINE_COMPARISON.json
```

The executable result-freeze path did not fully emit or bind:

```text
Q3 regret versus top-1/top-5/top-10 objective-dependence summary
Q5 fixed I0 -> IS -> IU/ISU -> ILc information-class summary
U16 Accuracy-on-the-Line diagnostic
within-context Spearman and Kendall
pairwise ordering accuracy
utility-estimation MAE
incremental R2 / calibration where semantically applicable
selected-regime distribution
coverage and catastrophic-target result tables
a coherent executable LORO sensitivity
```

The original LORO language was not sufficiently specified for a mixed
81-candidate context. Treating the missing table name as an instruction to
invent a post-outcome regime analysis would add a scientific degree of freedom.

## E. Missing End-to-End Result-Freeze Test

The C81 suite tested score primitives, source-schema repair, heterogeneous
selection descriptor shapes, synthetic family-wise behavior, and simple
taxonomy branches. It did not invoke one public synthetic recovery entrypoint
through the complete sequence:

```text
selection replay
mixed selector/random/ceiling row construction
canonical table validation and writing
Q1/Q2 and max-T
same-method cross-seed logic
LOTO
Q3/Q5 and secondary output coverage
artifact manifest
atomic final result freeze
all A/B/C/D/E branches
```

In particular, there was no end-to-end fixture proving that identical field
sets with different dictionary insertion orders survive the real writer, or
that a partial write cannot expose a final result directory.

## F. Exact Evidence Boundary

The committed evidence establishes:

```text
contexts evaluated in process memory:       32
method/control rows computed in memory:    672
method-context rows frozen:                  0
primary-comparison rows frozen:              0
LOTO rows frozen:                            0
Q1/Q2/max-T executed:                        0
nonblocker taxonomy executed:                0
evaluation-label rows read:              4,746
same-label oracle accesses:                  0
target4 primary rows:                        0
```

Git proves the absence of frozen C81 scientific tables and preserves the failed
job ledger and stderr hash. The statements that the 672 values were not printed,
inspected, or reconstructed are executor and red-team assertions. Git cannot
independently prove what a human did or did not observe outside committed
artifacts. This limitation must accompany future provenance language.

## G. Retry-Policy Attribution Correction

The final C81 report says the "locked C81 policy" forbids patching and rerunning
after evaluation outcomes were read. That wording is stronger than the literal
base-protocol text. The base protocol explicitly provides blocker-first
taxonomy and fail-closed authorization boundaries, but it does not contain a
verbatim clause saying every post-evaluation implementation failure permanently
forbids a same-identity rerun.

The no-rerun decision remains the accepted governance outcome, but its precise
provenance is:

```text
PM governance after the post-evaluation failure
+ consumed scope-specific authorization
+ blocker-first taxonomy
+ additive-history discipline
```

It must not be quoted as an exact original C81 protocol clause.

## Final Disposition

All additional findings reinforce, rather than weaken, the accepted blocker
classification:

```text
C81-E_protocol_input_implementation_or_provenance_blocker
```

The insertion-order failure alone prevented a result freeze. The taxonomy,
coverage, and end-to-end-test findings show that a simple one-line writer patch
would not have been sufficient to recover a valid C81 scientific comparison.

C81 remains closed historically with no valid baseline result. Any recovery
must use a new post-C81-outcome-access protocol, new implementation identity,
new execution lock, and new direct PI authorization. This addendum itself does
not authorize C82E or any real-data computation.
