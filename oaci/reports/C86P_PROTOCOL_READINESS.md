# C86P Prospective Active Testing Program Protocol Readiness

## Final Gate

```text
C86_ACTIVE_TESTING_DEVELOPMENT_CONFIRMATION_AND_UNTOUCHED_POPULATION_PROTOCOL_LOCKED_READY_FOR_PI_REVIEW
```

C86P is protocol/readiness work only. It created no real-data execution lock,
authorization record, active-acquisition result, candidate field, or scientific
gate.

## Chronology And Identities

| Object | Commit | SHA-256 |
|---|---|---|
| C86 main protocol | `430b7b59` | `d4feac535f8c1144a55d77cd7f322ae961d5c7d5a899dfd15c371484d88fbb7a` |
| low-budget estimator operationalization | `a1a1e736` | `0cdb05c113a1c681584dec907a002af809aa019c933e042f15065a4f30c1f1dd` |
| Yang2025 2C variant correction | `d89ec8d1` | `5948b76a2d08c45c88e157aace1cc421a8c551b1c763a265376ad25921103c0d` |
| historical-access correction | `7e19c99f` | `fd7e214c6c6675b2a9b071b2bf278e2c4393495b2f4cbe03f82528c3098e7064` |
| synthetic calibration operationalization | `0217485f` | `a80e8cca75eaa4d22b374794c06a9304ef9bb21605ec75f5d6aa53509f86b54b` |
| final readiness implementation/tables/tests | `5bf5d08e` | Git commit |

The parent C85E lock, result, manifest, completion, lifecycle, and compact
reports replayed exactly. The immutable state remains C84-D / C84-L4, with
T1/T3/T4/T7 PROVED, T2/T6 COUNTEREXAMPLE, and T5 OPEN.

## Untouched Population

The installed 53-row MOABB imagery catalog was audited without opening EEG.
Every catalog binary row received a loader-source task audit. Selection used
only the prospectively fixed interface/availability rule and never published
benchmark performance.

| Confirmation interface | Subjects | Minimum trials/subject | EEG channels | Sampling | Event interval | License |
|---|---:|---:|---:|---:|---:|---|
| `Brandl2020` | 16 | 504 | 63 | 1000 Hz | 4.5 s | CC-BY-NC-ND-4.0 loader metadata |
| `Kumar2024` | 18 | 400 | 22 | 512 Hz | 5.0 s | CC-BY-4.0 |
| `Yang2025_2C` | 51 | 600 | 59 | 1000 Hz | 4.0 s | CC-BY-4.0 |

The catalog combines Yang2025's 2C/3C variants; loader-source audit identified
the canonical 51-subject `paradigm_type=2C` interface. Conversely,
`Dreyer2023` is development-only because an archived project note proves a
local preprocessed store and loader had already been verified, so absence of
prior target access cannot be certified. `BNCI2014_004` remains a stress track
because it has nine subjects. Every passing untouched interface is included.

Primary metadata sources are the [Brandl paper](https://doi.org/10.3389/fnins.2020.566147),
[Kumar repository](https://zenodo.org/records/10694880), and
[Yang data descriptor](https://doi.org/10.1038/s41597-025-04826-y). Loader
files, sizes, and SHA-256 values are frozen in the eligibility registry. Future
engineering must replay current repository terms and loader bytes before any
download.

## Program Contract

The primary action set remains 81 candidates: one ERM, 40 OACI, and 40 SRC.
One queried target label reveals the complete 81-action loss vector. Physical
acquisition-pool, label-server, held-evaluation, and same-label-oracle views are
separate; queried trials cannot enter held evaluation.

The primary total-query grid is `[4, 8, 16, 32, FULL]`. P0 uniform sampling
without replacement is the fair comparator. Historical class-stratified P1 is
secondary and explicitly class-aware. FULL is an all-label consistency and
ceiling reference, not an active-versus-passive superiority test.

The frozen registry contains A1 importance-weighted active testing, A2
pairwise-loss variance design, A3 disagreement acquisition, and A4 a
prospectively specified plausible-best heuristic. Their observable inputs,
scores, probabilities, warm starts, estimators, failure states, tie rules,
complexities, and fidelity limits are explicit. None is claimed to be a
byte-exact reproduction of a source algorithm. Primary sources include
[Kossen et al.](https://proceedings.mlr.press/v139/kossen21a.html),
[Farquhar et al.](https://openreview.net/forum?id=JiYq3eqTKY),
[Hara et al.](https://doi.org/10.1007/s10994-024-06603-1), and
[Karimi et al.](https://proceedings.mlr.press/v130/reza-karimi21a.html).

Sequential paths use 2,048 paired chains, a 0.05 uniform probability mixture,
and LURE for linear finite-pool moments. The historical bAcc/NLL/ECE composite
is a nonlinear plug-in. A prospectively fixed Jeffreys `0.5` binary
pseudocount keeps total-query paths defined when a small prefix lacks a class;
no unbiasedness claim is made for that smoothed ratio, ECE absolute values,
midranks, or selected action.

Primary endpoints separate mean standardized regret, worst target, empirical
CVaR at `[0.50, 0.75, 0.90]`, and raw-utility epsilon-near-optimal probability
at `[0.005, 0.01, 0.02, 0.05]`. The scientific unit is target subject. Chains,
queries, trials, candidates, panels, seeds, and levels do not increase N.

Within-dataset max-T uses 65,536 draws over the fixed active-method/budget
family, with no pooled three-dataset p-value. Mean, tail, panel/seed/level, and
LOTO qualifications and the C86-A--E / C86-L1--L4 taxonomy are fixed before
confirmation access.

The synthetic calibration contract fixes all 11 scenario laws, PCG64DXSM seed
namespace, 12 target groups, eight contexts per target, 128 acquisition and 128
held trials, 81 candidates, 2,048 chains, failure injections, required outputs,
and validation targets. It requires the future production dispatcher and has
not opened a registered random stream or published a result.

## Stage Boundary

```text
C86L  C84 trial-level full-loss-vector artifact; development only
C86D  active-policy development and retrospective/synthetic calibration
C86C/F untouched engineering canary and 81-candidate field
C86H  untouched multi-cohort confirmation
```

Each future stage requires a scope-specific protocol, lock, and fresh direct
authorization. C86P performed zero EEG downloads/opens, zero label reads, zero
active runs, zero training/forward/GPU work, and zero manuscript actions.

## Validation

Twenty-four required registries were generated deterministically. Twenty-two
C86P focused contract tests pass. Focused, C65, C23, and full OACI regressions
all pass with empty stderr; details are in `C86P_REGRESSION_VERIFICATION.md`.

## Readiness Decision

The eligibility, budget, policy, estimator, physical-view, robust-risk,
inference, development/confirmation, and claim contracts are closed for PM
review. This gate does not authorize C86L, C86D, C86C/F, or C86H.
