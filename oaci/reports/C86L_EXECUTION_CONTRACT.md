# C86L — Execution Contract (short)
## Prepared under the PM GO; NOT an execution authorization

**Status**

```text
C86LP_SHADOW_PROBE_CRITERIA_ACCEPTED
GO_TO_ONE_REAL_C86L_PRODUCTION_STAGE   (preparation + path)
C86L_EXECUTION_NOT_AUTHORIZED
C86D_NOT_AUTHORIZED / C86H_NOT_AUTHORIZED / C87_NOT_AUTHORIZED / MANUSCRIPT_NOT_AUTHORIZED
```

This is the short contract the PM asked for before any real C86L touches C84
construction labels or target predictions. It is deliberately lean — it binds only
what carries real risk. The **only** valid execution trigger is a separate direct
`授权 C86L`; C86LP does not authorize execution, and this document does not either.
Code: `oaci/active_testing/c86l_production.py` (guarded, inert — `execute` refuses
without the trigger). No real data is opened by preparation or by
`validate_readiness` (metadata only).

## What the contract binds (the seven required items)

1. **Real input identities.** The frozen upstream SHA-256 set is bound
   (`constants.FROZEN_INPUT_SHA`: C86 effective-program V3, C84F complete-field
   manifest, C84F target trial registry, C84S V5 analysis lock, C84S selection
   freeze, C85U acceptance manifest). Payload SHA verification happens only inside
   an authorized `execute`, never during preparation.
2. **Construction ⟂ evaluation.** Construction and evaluation trial-ID registries
   are distinct and must be declared disjoint; zero overlap is proven on the real
   IDs at authorized execution.
3. **Semantics-B topology.** One physical construction trial spans its 8 contexts
   (2 panels × 2 seeds × 2 levels); each context has its own 81-candidate
   contribution row; arithmetic is bound and checked
   (`4,773 × 8 = 38,184`; `× 81 = 3,092,904`).
4. **Separate process/filesystem roots.** Acquisition-unlabeled, label-oracle, and
   contribution-store roots must be three **distinct** physical roots, none equal
   to the held C85U outcome identity. (C86LP's in-process name mangling is a
   logical mock; C86L upgrades it to real separation.)
5. **One query = one physical label.** The Semantics-B `QueryServer` contract:
   a query names one physical trial, reveals one label + one contribution row per
   context, and the budget counts physical labels per target (no double-billing).
6. **Full contribution-field arithmetic.** 118 targets, 944 contexts, 81
   candidates/context, 4,773 construction + 4,848 held-eval = 9,621 trials; 590
   target-budget cells (514 available, 76 PhysionetMI-B32 `INPUT_UNAVAILABLE`, no
   substitution).
7. **Failure-stop + result manifest.** On any binding/identity/arithmetic failure,
   stop closed with no partial publication; a successful authorized run emits a
   `C86LResultManifest` (gate, trial/context/row counts, endpoint definition,
   isolation level, failure-stop flag).

## Endpoint naming (PM forward requirement)

The pre-registered near-optimal endpoint is precisely
`target_near_opt_prob = P(target 8-context mean regret ≤ ε)`
(`pilot.NEAR_OPT_DEFINITION`). It is **not** the per-context
`P(selected context-action ∈ A_ε)`. If that per-context quantity is ever reported
in C86D/C86H it must be a separately pre-defined endpoint, never conflated. Tail is
the upper-`TAIL_FRACTION` (0.25) CVaR over the cohort's target distribution.

## Boundary and sequence

C86LP does not authorize C86L. After PM review of this contract, the only valid
execution statement is `授权 C86L`. A successful authorized C86L then requires
separate PM review before C86D; C86D before C86H; C87 does not auto-start after
C86H. Each step needs its own scope-specific authorization. No manuscript work is
authorized.
