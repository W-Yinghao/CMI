# C85TL Overall Report

## Disposition

```text
C85T_PROOF_AND_SYNTHETIC_EXECUTION_IMPLEMENTED_AND_LOCKED_READY_FOR_PI_AUTHORIZATION
```

C85TL closes the operational gap between the semantically valid C85R V2
contract and one deterministic future C85T execution. It fixes the RNG engine,
draw order, exact-versus-Monte-Carlo precedence, S9 policy/estimands, rational
LP output, proof artifacts, theorem transition rules, coordinator, runtime
replay, failure ledger, and atomic result publication.

It does **not** execute C85T. No registered scenario output, proof artifact,
status transition, authorization record, real-data access, active acquisition,
C85E object, or manuscript change exists.

## Authoritative Identities

| Object | Commit / SHA-256 |
|---|---|
| C85R accepted starting HEAD | `48022a6ca9683efbe918fb951c8885e107fd8ee4` |
| C85TL protocol commit | `7e8ffdffcbd8aef5a59e6bfa9a2fe0c5aa20a28f` |
| C85TL protocol SHA-256 | `6543d6ebbfccb8158f8f48a4fe6409c6243a708bbb0358d350932dd249e6b7c2` |
| final implementation commit | `dad9d39cccf02771d4e643c0649fd66ab660a1c0` |
| operative lock commit | `9d414ebb889b2cfc3fefa19fa98d7ea5ca9fd691` |
| operative lock SHA-256 | `4a289a46040b10855c6f23def53c328bdce0a8b1c71b7e90523887b6c1db7991` |
| runtime registry SHA-256 | `998370ffd3dc7572339b5d3ab1e876519ea4c0bf592842044fe20922d2d30631` |

The initially committed lock `68101202` was never authorized. Review found
only that its creation timestamp was later than its Git commit. Before any
authorization, the timestamp was corrected, the sidecar was recomputed, and
all regressions were rerun under the operative `9d414ebb` lock. Both regression
log generations remain externally preserved.

## Foundation Replay

```text
C85P protocol:
  af4c2cb35a6b6555d6c9ded3105eb7ad4f061ba237d3e8cc3ed6f5a18aede006

C85R repair protocol:
  e37bb444fdd174ba4ca1f95e91d9193378f11dd0ef2aeac3e03cbf6249a34b68

C85R V2 generator:
  e055c2a785374a3067ce90746a5941b39847b88a4f33e4ff8da5ca8adfde355a
```

C85TL does not alter those objects. The historical V1 contradiction and C85R
repair remain visible. All C85P/C85R registries and reports are bound into the
runtime identity.

## Environment Resolution

The operative environment is:

```text
/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact
```

Observed identities are Python 3.13.7, NumPy runtime 2.4.4, and
`importlib.metadata` first match 2.3.3. Both NumPy 2.3.3 and 2.4.4 dist-info
trees exist. The lock binds both metadata identities and the imported NumPy/
Generator/PCG64DXSM files rather than silently selecting one metadata account.
Eleven environment files are replayed by exact path, byte count, and SHA-256.

## Deterministic RNG

The only registered generator is:

```text
numpy.random.Generator(
  numpy.random.PCG64DXSM(seed)
)
```

with:

```text
seed = low64(SHA256(C85_SYNTHETIC_V1|scenario_id|replicate_id))
byte order = little endian
replicates = 0..4095
action order = canonical integer order
array dtype = float64 unless exact rational
```

Every replicate uses a fresh stream. Canonical serial reduction removes
parallel reduction ambiguity. Readiness tests can access only three
`SHADOW_*` fixtures. Attempts to seed `S0` through `S10` without a consumed
future authorization fail before a random draw.

## Exact And Monte Carlo Roles

Exact finite computation is authoritative for S0-S5, S8, and S10 and for the
geometry portions of S6/S7 and analytic variance in S9. Exactly 4,096
replicates are used only for S6, S7, and S9. Simulation cannot overwrite an
exact object or create a proof status.

The S6/S7 implementation draws an iid action-error vector with the registered
scale, performs first-index argmax, and reports selection counts, top-1,
outside-near-set probability, regret, MC SEs, and descriptive intervals. The
T7 expression remains a proof target.

## S9 Operationalization

The missing S9 semantics are fully executable:

```text
draw:
  51 L Rademacher values, then 46 H values

passive:
  L[:51], H[:13]

Neyman:
  L[:18], H[:46]
```

Both designs use one generator per replicate, creating a prospective paired
common-random-number comparison with unchanged marginal laws. Every action is
estimated using the registered stratum masses. The first canonical minimizer
is selected. Outputs include correct-best, true-best-in-top2, population
regret, `D_hat`, selected counts, MC uncertainty, paired differences, and raw
digests. Analytic `D_hat` variance is authoritative. No universal active-
testing conclusion is allowed.

## S5 And S8

The candidate S5 CVaR region `(13/20,1)` is locked for future proof
verification, with both endpoints excluded. C85TL does not mark it proved.

The S8 exact solver enumerates rational LP vertices and validates simplex,
nonnegativity, extreme-point regret slacks, pure-action value, and
randomization gain. Its canonical tie rule is lexicographic. Only an unrelated
shadow LP was solved in readiness; the registered S8 output remains absent.

## Proof Status Contract

The future C85T result root must contain seven proof/counterexample artifacts,
each with statement, assumptions, argument, boundary cases, independent red
team, and final status. The exact theorem statement is SHA-bound.

The transition validator enforces:

```text
simulation cannot prove;
citation alone cannot prove;
status must be theorem-allowed;
every non-OPEN status requires independent PASS;
failed attempts remain visible;
T5 may remain OPEN.
```

Current status is unchanged:

| Theorem | C85TL status |
|---|---|
| T1 | OPEN |
| T2 | OPEN |
| T3 | OPEN |
| T4 | OPEN |
| T5 | OPEN |
| T6 | OPEN |
| T7 | OPEN |

No canonical proof file or independent proof verdict was produced.

## Runtime Guard And Lock

The single public command is:

```text
python -m oaci.theory.c85t_execute run-locked \
  --execution-lock <path> \
  --output-root <fresh-root>
```

Before a registered action, it verifies the lock sidecar, V2 generator,
operationalization protocol, runtime registry, 106 bound repository objects,
their Git blobs, exact environment, branch, clean HEAD equal to origin, lock
ancestry, fresh root, and fresh direct authorization. The actual lock commit is
resolved from Git and must be bound by the future authorization record.

The lock status is:

```text
LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED
```

No `C85T_PI_AUTHORIZATION_RECORD.json` exists.

## Atomic Result And Failure Semantics

After future authorization consumption, the coordinator creates an attempt
root and immutable consumption receipt. Exact computation, MC, proof
construction/audit, and status application run through one bound process. A
manifest hashes every result artifact before one staging-to-final rename.

Shadow failure injection covered failure before the result JSON, before the
manifest, and before final rename. In all cases the final root remained absent
and the failed staging root remained available. Automatic retry is false.

## Readiness Evidence

New C85TL tests:

```text
shadow execution:
  16

execution lock / chronology:
  11

total:
  27
```

Before formal regression, all 92 C85P/C85R/C85TL-focused theory tests passed.
Shadow tests exercised raw RNG replay, an unrelated rational LP, an unrelated
finite CVaR object, 4,096-replicate near-optimal selection, 4,096-replicate
paired full-information selection, proof transition denial, T5 OPEN retention,
and atomic failure behavior.

None used a registered scenario stream.

## Accepted Regression

All final accepted runs use lock commit
`9d414ebb889b2cfc3fefa19fa98d7ea5ca9fd691`:

| Suite | Result | Runtime | stderr |
|---|---|---:|---:|
| focused | 348 passed | 6.92 s | 0 bytes |
| C65 | 959 passed, 1 skipped, 3 deselected | 68.12 s | 0 bytes |
| C23 | 1,370 passed, 1 skipped, 3 deselected | 95.62 s | 0 bytes |
| full | 2,294 passed, 1 skipped, 3 deselected | 296.85 s | 0 bytes |

The increase from C85R is exactly 27 in each suite. The only skip is the
historical finalized C78F node. The three standing C79P unauthorized-adapter
nodes are explicitly deselected by the committed wrapper. `squeue` reported no
active C84/C85/OACI job; `sacct` was not used.

## Artifact Inventory

```text
operationalization protocol/timing/sidecar: 3
implementation modules:                    6
contract/readiness CSV tables:             15
new test files/nodes:                       2 / 27
runtime-bound repository objects:           106
execution locks:                            1
authorization records:                      0
canonical proof files:                      0
registered scientific result files:        0
```

The implementation/tables/tests total 127,525 bytes. No raw data, label root,
checkpoint, candidate array, weight, optimizer state, cache, or file larger
than 50 MiB is added to Git.

## Red Team

The final red team passed 75/75 checks covering chronology, historical
preservation, dual NumPy metadata, RNG streams, exact/MC separation, S9
estimands, S5/S8 boundaries, proof statuses, lock replay, authorization,
atomicity, isolation, regression, and Git hygiene.

## Protected Boundary

All protected counters are zero:

```text
registered S0-S10 execution;
registered MC replicate;
canonical proof artifact;
proof audit and theorem transition;
C85T authorization;
real project data/label access;
selector or empirical inference;
training, forward, GPU;
active acquisition;
C85E;
new data/model zoo;
manuscript work.
```

## Future Authorization

The shortest future statement is:

```text
授权 C85T
```

This document is not authorization. A future successful C85T must use this
exact lock and stop at:

```text
C85T_DECISION_THEORY_PROOF_AUDIT_AND_SYNTHETIC_VALIDATION_COMPLETE_C85E_PROTOCOL_REVIEW_REQUIRED
```

No C85T outcome automatically authorizes C85E, active acquisition, real data,
new data/model zoos, or manuscript changes.

## Detailed Reports

The complete supporting records are:

```text
oaci/reports/C85TL_PROTOCOL_READINESS.md
oaci/reports/C85TL_FINAL_REPORT_RED_TEAM.md
oaci/reports/C85TL_REGRESSION_VERIFICATION.md
oaci/reports/C85TL_OVERALL_REPORT.json
oaci/reports/c85tl_tables/
```

