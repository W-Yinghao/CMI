# Project A — Claims and Contributions

## Thesis

EEG adaptation should be reported as

```
observed information  +  declared contracts   ->   identifiable estimands
```

not as

```
target metric observed in evaluation   ->   target adaptation claim.
```

The observation regime (what is measurable at deployment) and the declared contracts (what shared
structure is assumed) — not the evaluation harness — determine which quantities can be claimed. A
metric can be **reportable** (computed with oracle target labels for evaluation) without being
**identifiable** (pinned down by what is actually observed under the deployment regime).

## Contributions

**C1 — OACI formalism.** Observability-Aware Contracted Identifiability: a world class `M(C)`,
observation operators `O_R` for regimes R0 (source-only) / R1 (target-unlabeled) / R2
(minimal-paired), compatibility sets `K_{R,C}`, the identifiability definition **OA-0**
(a functional is identifiable iff constant on every compatibility set), information monotonicity
**MONO-1** (`R0 ⊑ R1 ⊑ R2`), and contract-strength monotonicity (assuming more can manufacture
point-identifiability the data did not provide — so checkable vs assumed contracts are tracked).
Source: `06_oaci_identifiability.md`, `01_information_regimes.md`.

**C2 — TOS-1 source-only ceiling.** Under any target-free-coordinate contract set, every
non-trivial target functional — target risk, target prior, target concept, and adaptation gain
*including its sign* — is non-identifiable under R0, unless the target law is fixed by an external
axiom (which is then recorded as a declared target-law axiom, never as source-only evidence).
Source: `03_tos_source_only_ceiling.md`, certificates CE-R0-1/2/3.

**C3 — Target-unlabeled boundary.** In R1 the target marginal / support becomes observable; the
target prior is identifiable **only** under contracts C1∧C2∧C3 (**TU-1**), and the target concept
`p_T(Y|X)` remains non-identifiable (**TU-2** / CE-R1-1). Source: `04_prior_decoupled_theory.md`,
`06`, CE-R1-1/2.

**C4 — Prior-decoupled CMI.** Separate label-prior shift from conditional transport with the
chain-rule identity **ID-1** and, after prior decoupling, the additive relation **PD-1**
(`Ĩ(Z;D|Y) = Ĩ(Y;D|Z) + Ĩ(Z;D)`, all terms ≥ 0) — avoiding the posterior-KL-upper-bound,
zero-Bayes-error, and concept-shift overclaims that the legacy notes had asserted and that
`h2cmi/THEORY.md` (P0-2..P0-5) retracts. Source: `04_prior_decoupled_theory.md`.

**C5 — Executable audit layer.** A machine-checked mapping Claim → Verdict → ObservabilityReport
that enforces allowed vs rejected claims, marks R0/R1 target metrics oracle/evaluation-only with
`identifiable_estimand=null`, admits a target prior only under TU-1, keeps leakage diagnostic, and
fails loudly on a forbidden-claim list. Source: `h2cmi/observability/` + tests.

**C6 — Audited real-EEG evidence.** End-to-end audited reports on MOABB motor-imagery
`BNCI2014_001` (4-class) and `BNCI2014_004` (binary), with a chance-normalized multi-dataset digest
that refuses to pool raw balanced accuracy across different class counts. Source: Steps 8–10
tracked digests.

## What we do NOT claim

- **No SOTA claim.** The grids are audited claim-boundary + stability validations, not tuned
  performance comparisons.
- **No source-only target-gain identification.** TOS-1 forbids it.
- **No R1 target-concept detection** from unlabeled target data (TU-2 / CE-R1-1).
- **No leakage → accuracy guarantee.** Conditional leakage stays a diagnostic.
- **No mixed-class raw balanced-accuracy pooling** — cross-dataset comparison uses chance-normalized
  excess only.
- **No causal EEG mechanism** is identified from these experiments.
