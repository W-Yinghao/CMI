# C86H — Terminal Untouched-Confirmation Contract (short, single)

**Status / authorization**

```text
C86D_CORRECTED_LAST_DEVELOPMENT_RESULT_ACCEPTED ; C86D_DEVELOPMENT_BRANCH_CLOSED
C86H_SCIENTIFIC_MASTER_CONTRACT_ACCEPTED (population/methods/chronology/stop rule)
C86H_GATED_IMPLEMENTATION_PREPARATION_GO   (protocol/implementation prep only)
C86H_EXECUTION_NOT_AUTHORIZED / C87_NOT_AUTHORIZED / MANUSCRIPT_NOT_AUTHORIZED
```

> **Addendum (this file, sections 10–13)** resolves the single PM blocker: the confirmatory
> **two-level output taxonomy** (§10), the **registered confirmatory thresholds** — NOT the
> C86D development `TAU=0.02` (§11), the content-addressed **executable bindings** (§12), and
> the **pre-execution implementation review** (§13). No new C-number; no readiness machinery.

This is the single pre-registered contract for the project's terminal experiment. It
locks everything BEFORE any new EEG or label access. It is **not** `授权 C86H`; real
EEG download / training / field generation / label access / active acquisition /
execution require a separate direct `授权 C86H`. No multi-layer readiness/repair
machinery is created.

## 1. Scientific question (locked)

> In two prospectively-selected untouched adult cohorts, does the registered
> full-construction decision limit still fail to transport to held-evaluation
> near-optimal action; and do the registered A1/A2H acquisition rules improve mean,
> target-tail, and near-optimal actionability over P0 at the same total-query budget?

Any outcome (crossed / weakened / nontransportable / no-material-gain / cohort-
heterogeneous) closes the main question.

## 2. Population and field (frozen before access)

```text
cohorts : Brandl2020_CANONICAL_ADULT_V1 (16) + OpenNeuro_ds007221_HYBRID_ADULT_V1 (37)
targets : 53 adult targets ; target in own training = 0
zoo     : the SAME candidate zoo / candidate-action identity across both cohorts
interface: the frozen 11-channel common field
untouched: Brandl + ds007221 were frozen by metadata rule BEFORE the C86D outcome
```

## 3. Physical chronology (label-blind until selection freeze)

```text
train + freeze candidate zoo
 -> generate target-unlabeled predictions on the 53 targets
 -> label-blind acquisition/evaluation split (no labels used to split)
 -> run + freeze P0/A1/A2H selections (Stage-H1, no held labels in-process)
 -> ONLY THEN open held-evaluation labels (Stage-H2)
```

Stage-H1 holds no held path (path-blind worker, as in corrected C86D); Stage-H2 holds
no query-server capability and verifies every freeze before opening held labels.

## 4. Methods (no change)

```text
exact current production dispatchers: P0, A1 (mixture expected-NLL LURE), A2H (Hara sum-over-pairs)
no new methods ; no hyperparameter adjustment ; no method deletion
```

## 5. Numerical integration (locked confirmation program)

```text
2,048 chains  (NOT the 8 development chains) ; target-bound seeds ; paired numerical integration
```

No post-hoc chain reduction. Any change requires an explicit prospective contract
revision BEFORE any confirmation data access — never after seeing results.

## 6. Endpoints, gate, and unsupported budgets

```text
primary risk : target-first held STANDARDIZED regret
tail         : exact target-tail CVaR (fractional boundary)
near-opt     : indicator-first target near-optimal probability (raw-gap epsilon geometry)
comparison   : per-cohort active vs P0 ; FULL construction-view ceiling (mean AND tail AND near-opt)
taxonomy     : TWO-LEVEL output — formal C86-A..E + L1..L4 gate (§10) with registered thresholds (§11);
               secondary interpretive descriptor is reported alongside but never replaces the gate
no pooled cross-dataset p-value
unsupported budgets : INPUT_UNAVAILABLE (never substituted with FULL)
```

## 7. Claim boundaries carried from C86D (must be preserved)

```text
* ACQUISITION_VIEW_NONTRANSPORTABLE means: full construction labels + the registered
  composite decision rule do not yield held-evaluation near-optimal action — a
  RESTRICTED policy-class result, NOT "target labels carry no decision-relevant info".
* A1/A2H are PREDICTION-DRIVEN nonuniform acquisition (scores precomputed from
  unlabeled candidate probabilities, not label-adaptive/posterior-updating). Report
  "the registered prediction-driven samplers did not materially outperform P0", NOT
  "adaptive acquisition generally fails".
* C86H is one untouched confirmation. C86D was convergent evidence in the SAME C84
  development universe, not an independent replication; "Nth measurement->control gap"
  is internal chronology, not a paper evidence count.
* Fix and reuse the FROZEN C86D construction-composite production dispatcher; no
  post-hoc NLL floor / probability precision / plugin-semantics changes. (C86D FULL is
  the frozen implementation, ~1e-3 from historical C85E Q0 FULL, not a byte-exact Q0 replay.)
```

## 8. Stop rule (terminal)

```text
one field generation -> one untouched confirmation -> one final PM/scientific audit -> STOP
after C86H: no auto-C87 ; no new active methods ; no new development cohorts ;
            no result-driven threshold changes.
then only: TPAMI contribution assessment / manuscript go-no-go / project archive.
```

## 9. Execution boundary

```text
C86H protocol / implementation preparation : GO
C86H real EEG download, training, field generation, label access, active
  acquisition, or scientific execution      : NOT AUTHORIZED
```

The only valid execution trigger is a separate direct `授权 C86H`, issued after PM
review of this contract (and of the implemented, gated entrypoint). C86H does not
auto-start C87.

---

# Addendum — confirmatory decision layer and executable bindings

The scientific core above is accepted. This addendum removes the one ambiguity that
would otherwise change the final scientific gate: the earlier "unchanged five-way
taxonomy" phrasing conflated two distinct objects. C86H must output **both**, and must
not use them interchangeably.

## 10. Two-level output taxonomy (the correction)

**Level 1 — primary FORMAL scientific gate** (this is the decision).
Registered in `C86_ACTIVE_TESTING_PROGRAM_PROTOCOL.json`, evaluated in precedence
`E -> A -> B -> C -> D`:

```text
C86-A : one SAME active method + finite budget passes mean AND target-tail
        qualification in EVERY untouched cohort, plus registered stability checks.
C86-B : A false, but one same active method + finite budget passes MEAN qualification
        in every cohort while tail qualification is NOT universal.
C86-C : no active method + finite budget passes mean qualification in ANY cohort.
C86-D : all non-blocking patterns not A/B/C (method / budget / level / panel / seed /
        target heterogeneity).
C86-E : protocol / input / candidate-field / label-view / execution / provenance blocker.
```

Label-complexity frontier (same registered object; frontier domain `[4,8,16,32]`,
qualification = active-vs-P0 mean qualification, closure = smallest finite budget for
one method that qualifies at that budget and every larger finite budget):

```text
C86-L1 : all-cohort frontiers exist for the SAME method, max budget <= 8,  ordinal distance <= 1.
C86-L2 : all-cohort frontiers exist for the SAME method, max budget in {16,32}, ordinal distance <= 1.
C86-L3 : all frontiers exist but method / ordinal / registered stability is heterogeneous.
C86-L4 : one or more cohort frontiers absent.
```

**Level 2 — secondary INTERPRETIVE descriptor** (mechanism reading; reported alongside,
never a substitute for the Level-1 gate):

```text
BOUNDARY_OPERATIONALLY_CROSSED / BOUNDARY_WEAKENED_NOT_ROBUST /
POLICY_LIMITED / ACQUISITION_VIEW_NONTRANSPORTABLE / NO_REGISTERED_ACTIVE_GAIN
```

`POLICY_LIMITED` may be emitted **only** if this contract now defines and locks an exact
oracle-acquisition diagnostic. **This contract defines no such diagnostic**, so the
descriptor is fixed to:

```text
POLICY_LIMITED : NOT_IDENTIFIABLE_IN_C86H
```

and is never inferred from ordinary P0/A1/A2H outcomes.

## 11. Confirmatory thresholds (registered — NOT the C86D development TAU=0.02)

C86H formal inference uses the pre-registered `inference_and_multiplicity_contract`
(from `oaci/theory/c86_active_program.py`), never the C86D development criterion
`TAU=0.02`:

```text
principal cluster           : target_subject      (scientific N = target subjects)
materiality margin (mean)   : 0.05
familywise alpha            : 0.05
max-T draws                 : 65,536              (within-cohort)
family                      : realized active methods x 4 finite budgets, within dataset
favorable target fraction   : >= 0.75
worst-target effect floor   : >= -0.10
positive panel x seed x level cells : >= 6 / 8
tail qualification          : CVaR_0.90 effect >= 0.05  AND  CVaR effect nonnegative at all alpha {0.50,0.75,0.90}
LOTO preservation           : >= 0.75
pooled cross-dataset p-value: FORBIDDEN
```

The C86D standardized-regret gap and `TAU=0.02` may appear only as the Level-2
interpretive descriptor, and never replace this formal inference.

*Reconciliation note (flagged for PM):* the registered family text reads
"4 active methods x 4 finite budgets". The C86D closure froze the production registry
to `P0/A1/A2H` (2 active) **before** C86H, and the contract forbids method deletion,
so the realized within-cohort max-T family is `{A1,A2H} x {4,8,16,32}` = 8 hypotheses.
This follows the pre-C86H method freeze, not a post-hoc loosening; confirm before execution.

## 12. Executable bindings (content-addressed)

### 12.1 Authoritative identity

```text
effective-program manifest V3 sha256 : c6b7e490e0f78f74f820428cee138782caff1dc0033422723593a7d8e3c5f77e
final adult cohort registry          : final_adult_untouched_cohort_registry_v3.csv
                                       sha256 82f329a1125a8ffe106c22ad490589aef84c239077105c22e6301d2e39593737
common 11-channel interface          : common_field_interface_v3.csv
                                       sha256 2e22863fbc447054d196376a48340e09192d99310c01639b67b51879019c99b4
                                       id     C86_C84SOURCE_TARGET_11CH_160HZ_0_3S_V3
frozen C86D production dispatcher     : commit c694315e ; content sha256:
   policies.py    58fb1fc1a5482cc7320f7db13d113156fc9d5fcb7f83cd69f2871c7b4eb1dbc5
   core.py        a0d06a6648f3d8740a81ab9a3194e4004441b6671cfae853ee9660c0ae1fe649
   c85u_config.py 0793bc6d07694452f3f7bbcfe884e2919d6aef3c7cebe84efdc274f83d92522d
method registry                      : P0, A1, A2H   (no add, no delete)
```

Real results must never choose the implementation after download; only these blobs run.

### 12.2 Candidate field identity

```text
panels        : A / B
training seeds : 5 / 6
levels        : 0 / 1
candidates/ctx : 81 = 1 ERM + 40 OACI + 40 SRC
unique models  : 2 x 2 x 2 x 81 = 648
target contexts: 53 x 8 = 424           (8 = 2 panels x 2 seeds x 2 levels)
same zoo across BOTH cohorts : required
```

### 12.3 Label-blind split (`canonical_trial_split`, locked)

```text
salt   : C86_TARGET_SPLIT_V1
order  : by ( SHA256(salt|dataset|subject|trial_id) , trial_id )
pool   : first floor(n/2)   ; held evaluation : remainder
require: n >= 80 trials ; pool >= 40 ; evaluation >= 40
post-access class support : >= 8 labels / class / view
on any target- or cohort-level support failure:
  NO resplit ; NO drop of unfavorable subject ; NO cohort substitution -> C86-E blocker (risk R10)
```

### 12.4 Budgets and endpoints (exact)

```text
budgets            : [4, 8, 16, 32, FULL]
FULL               : all acquisition-pool labels ; held-evaluation labels remain sealed
near-opt epsilon   : [0.005, 0.01, 0.02, 0.05] ; primary boundary epsilon = 0.05
target CVaR alpha  : [0.50, 0.75, 0.90]
formal tail qual.  : CVaR_0.90 primary  AND  nonnegative at every registered alpha
unsupported finite budget : INPUT_UNAVAILABLE (never substituted with FULL)
```

### 12.5 ds007221 task / run inclusion

```text
native cohort   : OpenNeuro_ds007221 ; interface OpenNeuro_ds007221_HYBRID_ADULT_V1
subjects        : sub-37 ... sub-73  (37)
task            : hybrid only
events          : left_hand / right_hand only
session/run rule: exact and exhaustive ; NO outcome-based acquisition filtering
interface       : 11 ch (FC5 FC1 FC2 FC6 C3 Cz C4 CP5 CP1 CP2 CP6) / 160 Hz / [0,3) s / 4-38 Hz
any pre-registered acquisition that cannot be parsed -> C86-E field blocker (no substitution)
```

Brandl2020 binds identically: `Brandl2020_CANONICAL_ADULT_V1`, subjects 1..16 (16),
same interface, same events.

## 13. Pre-execution implementation review (before any real data access)

The gated C86H entrypoint may be implemented now. Before any real EEG/label access, a
final short review must confirm exactly five things — no new milestone:

```text
1. Exact code/input bindings : §12 hashes present and verified (V3 manifest, cohort
   registry, 11-ch interface, frozen C86D dispatcher blobs, P0/A1/A2H registry).
2. Three-stage isolation     : label-free field generation -> H1 acquisition selection
   (path-blind worker) -> H2 held evaluation (no query-server capability).
3. Two-level output          : formal C86-A..E / L1..L4 kept separate from the Level-2
   interpretive descriptor (§10).
4. Confirmation inference     : 65,536-draw within-cohort max-T, target-subject
   clustering, LOTO, and the §11 registered materiality thresholds are implemented.
5. Resource feasibility       : 2,048 chains truly implemented and NOT reduced; an
   outcome-free resource benchmark first; if infeasible, STOP before any EEG/label
   access — never reduce the chain count after seeing results.
```

Completion of this review + a separate direct `授权 C86H` are the only valid triggers for
real C86H execution.
