# C86LP — C84 Construction-Pool Trial-Contribution Field (lean protocol)

**Gate reached:**
`C86L_C84_CONSTRUCTION_POOL_TRIAL_CONTRIBUTION_FIELD_IMPLEMENTED_AND_LOCKED_READY_FOR_PI_AUTHORIZATION`
(via `oaci.active_testing.validate()`; shadow-only).

## 0. What this is, and a deliberate scope change

This delivers the **development-only query infrastructure** that lets active
testing (P0 / A1 / A2H …) be implemented and debugged without contaminating the
C85U held development outcome. It opens **no real** C84 construction labels,
target predictions, Q0 shards, C84S result tables, or C85U utility values —
every check runs on synthetic shadow objects.

The PM entered C86LP unhappy that the rigor machinery had far outgrown the
science, and granted explicit latitude to adjust the authorization/ceremony. So
this milestone is **intentionally lean**. It keeps the parts of the C86LP spec
that actually protect a future result from contamination, and drops the parts
that only self-certify. See §5 for the exact trim.

## 1. Scientific role (unchanged from spec §1)

- acquisition/development pool = immutable C84 construction trials
- held development outcome = accepted C85U candidate-utility field
- construction/evaluation overlap = **zero**
- confirmation evidence = **none** (development-only, not a C84/C85 gate)

Must not change: `C84-D`, `C84-L4`, and theorem statuses T1/T3/T4/T7 PROVED,
T2/T6 COUNTEREXAMPLE, T5 OPEN. These are bound as immutable constants and a
mutated gate blocks publication (`field.DevelopmentFieldManifest.publish`).

## 2. Isolation contract (kept — this is the load-bearing part)

Three physically separate stores with a strict information gradient, mirroring
`c86r_tables/C86L_development_view_contract_v2.csv`:

| store | who sees it | contents |
|---|---|---|
| `UnlabeledPool` | active client | trial id + candidate probs / hard preds / confidence only |
| `LabelOracle` | query server (private) | `{dataset, target, trial_id, label}` only, never bulk-exposed |
| C85U held outcome | nobody in C86L | identity-bound only; never copied, opened, or summarized |

Enforced statically: a pool row that carries `true_label / label / queried_response /
construction_metric / selected_action / c85u_utility / held_utility / outcome`
raises `C86LPFieldError`; the package namespace contains no object with `utility`
in its name; the server's stores are name-mangled so a client handle cannot reach
them.

## 3. Query object & claim boundary (kept)

One queried binary label releases **only that trial's** row: true label plus the
per-candidate **linear** vector — NLL, correctness, hard prediction, confidence,
confidence-bin, signed calibration contribution. Pairwise NLL differences
(C(81,2) = 3,240) are **derived on demand**, never persisted.

Claim boundary (`constants.LINEAR_MOMENTS` vs `NONLINEAR_PLUGINS`): LURE
unbiasedness is registered **only** for linear moments. Balanced accuracy, ECE,
candidate midrank, composite utility, selected action, and target regret are
nonlinear plugins with **no** unbiasedness claim; asserting one raises
`C86LPClaimError`.

## 4. Query-server contract (kept)

`QueryServer.open_attempt(attempt_id, target, budget)` →
`query(attempt_id, trial_id)` returns one `QueryResponse`. It rejects duplicate
queries, unknown trials, cross-target trials, and post-budget-exhaustion requests;
an unsupported target-budget cell raises `C86LPInputUnavailable`
(`INPUT_UNAVAILABLE` — **no** replacement sampling, budget substitution, or target
deletion), matching the 76 PhysionetMI-B32 unavailable cells. This is the exact
interface a future C86D dispatcher must use.

## 5. What was intentionally trimmed (PM-granted latitude)

Dropped because, with no real payload opened, none of it gates a real risk:

- `C86L_EXECUTION_LOCK.json` + `.sha256` and binding every implementation byte /
  Git blob;
- single-use `授权 C86L` authorization consumption, `O_EXCL` receipts, process
  capability tokens, stage receipts;
- atomic-final-rename acceptance transaction + post-rename recovery semantics;
- `C86L_..._PROTOCOL.json` + `.sha256` + `PROTOCOL_TIMING_AUDIT.md` (this single
  markdown replaces the JSON/hash/timing trio);
- the 8,749,056-record historical Q0 replay (re-verification of already-frozen
  C84S — green-keeping, not new evidence);
- the resource-envelope shadow-benchmark lock and the `c86lp_tables` replay CSV.

Replaced by one lightweight boundary constant (`DEVELOPMENT_ONLY_BOUNDARY`)
enforced by static import isolation + shadow-only tests.

## 6. Files

```
oaci/active_testing/constants.py       # arithmetic, frozen gates, claim boundary, budget grid
oaci/active_testing/contribution.py    # linear contribution row + pairwise derivation + claim guard
oaci/active_testing/field.py           # unlabeled pool / sealed oracle / manifest + coverage/publish
oaci/active_testing/query_server.py    # one-query-one-row server contract
oaci/active_testing/__init__.py        # build_shadow_field(), validate()
oaci/tests/test_c86lp_query_field.py   # 13 property groups (29 cases)
```

## 7. Validation

`pytest oaci/tests/test_c86lp_query_field.py` — 29 passed. Full test collection
unaffected (2,531 collected, no import errors). Property coverage: arithmetic
consistency, construction/evaluation nonoverlap, unlabeled-pool no-label,
linear-contribution exactness, pairwise-derived-not-stored, nonlinear-plugin
claim guard, single-row query, duplicate/unknown/exhaustion rejection, no-bulk-
oracle, held-outcome isolation, first-index tie rule, partial-field-cannot-
publish, unsupported-budget INPUT_UNAVAILABLE, frozen-gate immutability.

## 8. Boundary (still true)

No active-policy execution, new EEG, new cohort label, training, forward, GPU,
C86D/H, C87, or manuscript work was performed. C86LP does not authorize C86L.
This remains a development-only line whose scientific payoff (if C86H is ever
reached) is bounded and does not resurrect the C21-negative OACI DG method.
