# C86R Eligibility, Baseline, Development-View, and Field Reconciliation

## Chronology

The additive repair protocol was committed and pushed at
`1ec508e64655edd71f278a0696c4c8d757700e29` before the public metadata and
literature audit. Its SHA-256 is
`d3550350e2e4f9ff300d03a03f9353fcef09d324b69af21608f5a6f6b45741d3`.

Before that commit, C86R had opened zero new EEG files, read zero new labels,
run zero registered active policies, performed zero training/forward calls,
used zero GPU work, and produced zero registered synthetic results.

## Effective Program

`C86_ACTIVE_TESTING_EFFECTIVE_PROGRAM_MANIFEST_V2.json` resolves the parent
protocol and every additive correction in one precedence-ordered object. Its
SHA-256 is
`c19ca4090b64dec3cc98971e44cbf09f7a1367e4a754535529d972b691c7ca65`.
The stale parent protocol alone is explicitly insufficient for downstream
binding.

Authoritative retained values include:

```text
query budgets: 4 / 8 / 16 / 32 / FULL total queries
candidate structure: 1 ERM / 40 OACI / 40 SRC
primary methods: P0 / A1 / A2H
secondary methods: P1 / MODEL_SELECTOR / CODA
development methods: A2M / A3D / A4 / ASE-XWED / Online AMS
preferred metadata interface: 11 channels / 160 Hz / [0,3) s / 4-38 Hz
stage order: C86LP -> C86L -> C86DP -> C86D -> C86C/F -> C86H
```

No C86 execution lock or authorization was created.

## Adult Eligibility

The locked rule was applied without imputing a minimum age from a mean:

| Cohort | Public evidence | C86R role |
|---|---|---|
| Brandl2020 | explicit age range 22-30 | primary adult untouched cohort |
| Kumar2024 | mean 23.22, SD 3.59; public participant age field invalid | age-uncertain stress track only |
| Yang2025_2C | public minimum 17, maximum 30 | age-mixed stress track only |

The primary adult-only set therefore has one cohort, below the locked minimum
of two. The criterion was not relaxed and no performance outcome was used.

## Baseline Fidelity

The current literature registry was refreshed through 2026-07-18 using primary
sources. `A2H` implements the faithful general-K Hara query score as the sum of
all unordered pairwise expected absolute loss differences. The former max-pair
rule is now `A2M_project_max_pair_heuristic`, development-only, and is never
described as Hara.

ASE/XWED, Online Active Model Selection, MODEL SELECTOR, CODA, and emerging
prediction-powered active testing all have explicit dispositions and interface
limitations. None may be silently omitted from future method freezing.

## Development View

C86L is prospectively restricted to immutable C84 construction trial IDs and
matching frozen predictions. Its held development outcome is the accepted C85U
evaluation-derived utility field. Direct C84 evaluation-label access is
forbidden in C86L/C86D.

The metadata-only availability table has 590 rows: 118 targets by five total
query budgets. Lee and Cho support all registered budgets. Physionet supports
4, 8, 16 and cell-specific FULL; budget 32 is `INPUT_UNAVAILABLE` for every
target. No replacement, substitution, or target deletion is allowed.

## Information Object

A queried binary label reveals per-candidate NLL, correctness, class, signed
confidence-bin terms, and pairwise NLL differences. LURE unbiasedness is
claimed only for registered linear moments under the locked sampling and
positivity assumptions.

Balanced accuracy, ECE, candidate midranks, composite utility, and selected
action are nonlinear plugin objects. C86R makes no unbiasedness claim for them.

## Field Feasibility

Public loader metadata supports a common prospective 11-channel interface
across the three C84 source cohorts and Brandl/Kumar/Yang targets. It requires a
new shared candidate zoo; existing C84 20-channel model bytes are not reusable.

Metadata arithmetic for the intended three-cohort field is:

```text
unique candidate units: 648
training phases: 24
target contexts: 680
candidate-context slices: 55,080
unit-cohort artifacts: 1,944
max-T draws: 65,536
target clusters: 16 / 18 / 51
```

This is a planning envelope, not execution authority. Licensing and derived
artifact terms require future institutional replay, especially for Brandl's
CC-BY-NC-ND-4.0 terms.

## Disposition

The baseline, development-view, information-object, effective-manifest, and
metadata field-interface repairs pass. The adult-only multi-cohort eligibility
condition does not.

```text
C86_UNTOUCHED_COHORT_AGE_ACCESS_OR_INTERFACE_ELIGIBILITY_RECONCILIATION_REQUIRED
```

C86L, C86D, C86C/F, C86H, new data access, active acquisition, C87, and
manuscript work remain unauthorized.
