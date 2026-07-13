# C81 Frozen-Field Literature Baseline Comparative Audit

## Final Gate

```text
C81-E_protocol_input_implementation_or_provenance_blocker
```

C81E does not produce a valid zero-label-versus-source or zero-label-versus-
one-label comparison. Selection completed and froze correctly, but the
authorized held-evaluation process failed before the first result table was
written. No C81-A/B/C/D interpretation is available.

## Execution Summary

| Stage | Status | Evidence |
|---|---|---|
| Direct C81R2 authorization | Bound and consumed | commit `102e466` |
| Frozen selection replay | PASS | manifest `4677ed3a...`, payload `1ed893ac...` |
| Selection recomputation | 0 | forbidden by lock `f82ffa4` |
| Held-evaluation job | FAILED | Slurm `894958`, exit `1:0`, runtime `00:00:03` |
| Method-context rows computed in memory | 672 | not persisted or inspected |
| Method-context rows frozen | 0 | first table write rejected |
| Q1/Q2/max-T/LOTO | NOT RUN | failure preceded inference |
| Same-label oracle / target4 | 0 / 0 | protected routes unchanged |
| Training / forward / GPU | 0 / 0 / 0 | existing-field CPU analysis only |

The process opened 16 seed-specific evaluation views containing 4,746 label
rows and the corresponding construction views containing 4,470 rows. This
occurred only after the committed selection freeze.

## Blocking Defect

All selector and oracle rows produce the same eight scientific fields as the
analytic random control. Their insertion order differs:

```text
selector/oracle:
  regret, utility, top1, top5, top10, coverage1, coverage5, coverage10

random control:
  regret, utility, top1, coverage1, top5, coverage5, top10, coverage10
```

The strict CSV writer compares ordered dictionary keys rather than field sets,
so it raised `C81 table schema drift` before opening the output file. This is a
report-schema implementation defect, not evidence of a missing endpoint or a
scientific disagreement.

Because evaluation outcomes had already been read, the locked C81 policy
forbids patching and rerunning under the same protocol identity. The failed
attempt is retained, the authorization is consumed, and `run-real` again fails
closed.

## Method Accounting

All 34 registered entries are accounted for in
`c81e_tables/method_failure_and_availability.csv`:

```text
19 feasible selector methods: selection frozen; evaluation rows not persisted
B0/B5 controls:               computed in memory; not persisted
7 frozen Q0 comparators:      C80 artifacts remain valid; C81 comparison blocked
5 input-unavailable methods:  excluded prospectively
U16 diagnostic:               no frozen C81 result
```

The 12 paper-ready result tables that would contain regret, top-k,
noninferiority, max-T, cross-seed, LOTO, or catastrophic-target results are
explicitly marked `NOT_EMITTED_BLOCKED_NO_FROZEN_SCIENTIFIC_ROWS`. They are not
reconstructed from logs or process memory.

## Scientific Disposition

```text
Q1 zero-label versus strict source:      BLOCKED
Q2 zero-label versus Q0 B=1:             BLOCKED
Q3 regret versus top-k/top-1:            BLOCKED
Q4 cross-seed and leave-one-target:      BLOCKED
Q5 information-class transition:         BLOCKED
```

C80E remains the latest scientific result. Its full-panel Q0 frontier and its
small-target sensitivity are unchanged. C81E neither supports nor rejects any
registered zero-label baseline and cannot establish an information-class
transition.

## Claim Boundary

C81E is an existing-field, outcome-informed audit attempt. It provides no
independent confirmation, external validation, population generality,
deployability, universal one-label sufficiency, or impossibility result. It
does not authorize C82, a new repair campaign, active acquisition, seed 5,
BNCI2014_004, same-label oracle work, new methods, or manuscript experiments.
