# C86LP — Overall Report (for PM review, reconciled)
## A query instrument + pre-registered probe criteria for the target-information boundary

**Formal status**

```text
C86LP_SCIENTIFIC_REFRAME_ACCEPTED
C86LP_SHADOW_INSTRUMENT_AND_PROBE_CRITERIA_RECONCILIATION_REQUIRED  -> addressed by this report
object gate: C86LP_SHADOW_QUERY_INSTRUMENT_IMPLEMENTED_PROBE_CRITERIA_REVIEW_REQUIRED
C86L_NOT_AUTHORIZED / C86D_NOT_AUTHORIZED / C86H_NOT_AUTHORIZED
```

**Framing.** We are not preparing a submission. C86LP is an *instrument* for
measuring how the measurement–control gap changes as real target-label
information is admitted. This report supersedes the earlier lean-protocol text
(see §0 on why) and folds in all seven PM corrections.

---

## 0. Reconciliation of the previous artifact

The PM correctly caught that the report committed at `02a8c19f` was still the old
116-line lean-protocol text, **not** the scientific rewrite described in that
commit message: an earlier `git add` aborted on a stale pathspec and committed the
renamed-but-old file. That is fixed here, and the commit is now verified against
the working tree before pushing. The prior gate string
(`..._IMPLEMENTED_AND_LOCKED_READY_FOR_PI_AUTHORIZATION`) was also semantically
false once the lock/authorization apparatus was dropped; it is replaced (§2).

## 1. Our current best statement of the problem (tightened)

The recurring finding is a **measurement–control / identifiability gap**, but it
must be stated as a *restricted-policy* result, not an information-void:

> **Within the registered and tested source-only and target-unlabeled policy
> classes, no rule stably and same-method identifies and controls the
> target-optimal action.**

This is **not** the claim that the experiment contains no decision-relevant
information. C84 shows COTT reaches low *mean* regret in all three cohorts, yet no
method delivers same-method, target-tail-robust control across cohorts (empty
cross-dataset intersection). So the evidence supports *restricted-policy
identification/control failure*, and this is exactly why C85 separates a Blackwell
information value from a registered-policy value. Supporting negatives: C10/C14
(source-side control falsified), C24/C28/C29 (the offset carrier is
source-unobservable), C30 (within-target rank real, non-transporting), C25 R4
(target-unlabeled recovery collapses to noise), C80/C84 (no transport).

## 2. Corrected gate

The object is a shadow instrument whose probe criteria need PM review, so its gate
is `C86LP_SHADOW_QUERY_INSTRUMENT_IMPLEMENTED_PROBE_CRITERIA_REVIEW_REQUIRED`
(constant `GATE_INSTRUMENT`). A real C86L that opens construction labels and
target predictions would create its own short, genuinely load-bearing execution
protocol and lock at that time — not now.

## 3. Four scientific tightenings (PM §four)

1. **Not "information absent from the regime"** → restricted-policy
   identification/control failure (§1).
2. **Name the oracle.** The failing oracle is the *source-side / source-audit*
   oracle (cannot stably localize target-good checkpoints). The *same-label target
   endpoint* oracle **succeeds (~0.944 localization) but is circular** — it is
   essentially a restatement of the endpoint threshold and is unavailable at
   selection time. Correct phrasing: *source-side oracle fails; same-label
   endpoint oracle succeeds but is circular and unavailable at selection time.*
3. **Do not reuse the falsified "target gauge" entity.** C73 showed the residual
   decomposition is order-sensitive with weak split stability, negative held-out
   incremental R², negative LOTO increment, permutation p = 1. So the object is an
   **unexplained candidate-specific residual**, not a validated gauge. "Gauge" may
   remain an early model scaffold / synthetic latent only. The probe question is
   therefore: *after adding queried labels, does the candidate-specific target-
   utility uncertainty not yet explained by source/unlabeled information decrease,
   and does that decrease convert into held-evaluation decision control?*
4. **"Few-label axis never measured" is inaccurate.** Split-label and passive
   label budgets (8 / 64 / full) and the fixed passive Q0 frontier (C80/C84) were
   measured — with within-target reliability but incomplete top-1 / coverage /
   actionability. The genuinely untested axis, and the precise novelty, is:
   **under a fair total-query budget, is adaptive acquisition more efficient than
   passive uniform, and does that advantage improve mean AND target-tail risk
   simultaneously across untouched cohorts?**

## 4. Isolation is logical/API, not physical (PM §six)

The three stores (`pool`, `_labels`, `_contrib`) live in one Python object; the
server reaches them by name mangling. That is a **logical / API isolation mock**
(`constants.ISOLATION_LEVEL = "logical_api_mock"`), not physical separation. A
real C86L would use separate processes and filesystem roots. The report and code
no longer say "physically separate."

## 5. Query topology resolved to Semantics B (PM §five)

The earlier field keyed contributions by `trial_id` alone, which would overwrite a
physical trial's per-context vectors and double-bill one physical label across its
repeated contexts. Resolved to **Semantics B**: a physical trial appears in its
several contexts (real field: 2 panels × 2 seeds × 2 levels = 8), each with its
own 81-candidate probabilities and contribution row; **one physical-label query
reveals one label and derives one contribution row per context**, and **the budget
counts physical labels per target**. So a label is billed once and label-complexity
is meaningful. (`_contrib[trial] = {context: row}`; `QueryResponse.contributions`
is a per-context dict; the shadow generator now reuses physical trial IDs across
all 8 contexts.) Active-policy state and action are **target-level** (labels are a
per-target physical cost); per-context deployment problems share that budget.

## 6. Pre-registered probe criteria + bounded shadow pilot (PM §seven, option B)

Frozen **before** any real query. Decision rules (`pilot.classify`, thresholds in
`constants`):

```text
BOUNDARY_OPERATIONALLY_CROSSED     same active policy, same total budget, BOTH cohorts:
                                   improves mean regret AND tail (CVaR) AND near-optimal prob
BOUNDARY_WEAKENED_NOT_ROBUST       mean / near-opt improve, tail or cohort gate fails
POLICY_LIMITED                     an oracle acquisition cheaply beats P0, registered policy does not
ACQUISITION_VIEW_NONTRANSPORTABLE  even FULL construction info leaves weak/heterogeneous actionability
NO_REGISTERED_ACTIVE_GAIN          no registered active policy beats passive P0  (NOT an impossibility claim)
```

Endpoints are **regret / tail-CVaR / ε-near-optimal probability**, never top-1
alone (S5 below shows why). The pilot builds five **known-truth** shadow scenarios
and checks the instrument classifies each correctly — a discriminative-validity
battery for the active regime (the positive control the 0-label line never had),
robust across 20 seeds:

```text
S1 small-label identifiable   -> BOUNDARY_OPERATIONALLY_CROSSED
S2 budget-limited (need FULL)  -> NO_REGISTERED_ACTIVE_GAIN   (exchangeable trials: no acquisition helps)
S3 policy-limited              -> POLICY_LIMITED              (oracle finds it cheaply, registered can't)
S4 mean-only / tail failure    -> BOUNDARY_WEAKENED_NOT_ROBUST
S5 dense near-ties             -> BOUNDARY_OPERATIONALLY_CROSSED with top-1 ~0.15 but near-opt ~1.0
```

No non-passing scenario is called an information-theoretic "hard wall."

**Termination rule (frozen with the criteria).** After PM accepts these criteria:
one real C86L development-field production, then one C86D method freeze, then one
untouched C86H confirmation; C87 does not auto-start after C86H. Each step needs
its own scope-specific lock and fresh authorization.

## 7. Boundary

Shadow-only. No active-policy production run, real EEG, new cohort label,
training, forward pass, GPU, C86D/H, C87, or manuscript work. C86LP does **not**
authorize C86L. `C84-D`, `C84-L4`, and theorem statuses T1/T3/T4/T7 PROVED, T2/T6
COUNTEREXAMPLE, T5 OPEN are bound immutable; any mutation blocks publication.

## 8. Validation

`pytest oaci/tests/test_c86lp_query_field.py oaci/tests/test_c86lp_pilot.py` — 45
passed. Full collection unaffected (2,547). Query-field properties: Semantics-B
one-label-informs-8-contexts, budget counts physical labels, nonoverlap,
unlabeled-pool no-label, linear-contribution exactness, pairwise-derived-not-
stored, nonlinear-plugin claim guard, duplicate/unknown/cross-target/exhaustion
rejection, no-bulk-oracle, held-outcome isolation, logical-not-physical isolation,
corrected gate, unsupported-budget INPUT_UNAVAILABLE, frozen-gate immutability.
Pilot: taxonomy completeness, classifier covers all five labels, and the five
known-truth scenarios classify as expected across seeds.

## 9. Files

```text
oaci/active_testing/{constants,contribution,field,query_server,pilot,__init__}.py
oaci/tests/test_c86lp_query_field.py   oaci/tests/test_c86lp_pilot.py
```

## 10. For PM — the operational, falsifiable question this now supports

> How much does a small amount of queried target-label information reduce
> optimal-action uncertainty, and does that reduction convert into decision
> control across targets, across cohorts, and especially in the target tail?

The instrument and pre-registered criteria are ready for a brief PM review. On
acceptance, the next step is a real C86L (separately authorized). No further stage
is started pending that review.
