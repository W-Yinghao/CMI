# FSR_28 — PC2-E4 Protocol Update (design only; NO GPU authorization)

**Project FSR — PC2 preflight update.** Successor to FSR_23, whose repair primitive was Phase-4D **D1**
(counterfactual head, now dead: `repair_claim_level = none`). Phase 4F earned
`pc2_gpu_gate = eligible_for_protocol_update` (E4 first-moment mean alignment,
`strong_within_controlled_first_moment_scope`), which authorizes **this ledger/protocol update for PM review
only** — `pc2_gpu_run_authorized = false`, **PC2 GPU stays PAUSED**. This document commits **no GPU**.

## The load-bearing scientific question PC2-E4 must answer
Phase 4F showed E4 repairs an injected **constant-offset (first-moment)** shortcut — but **73% of that
"recovery" is a mechanical identity** (a first-moment aligner inverts a first-moment offset by algebra) and it
**fails leave-one-dataset-out**. So the open question is precisely:

> Does the E4 first-moment neutralizer (or a moment-generalized successor) repair a **learned** subject reliance
> — which is generally **not** a clean constant offset — or does its efficacy stay confined to the
> construction-matched first-moment case?

**Guarded expectation (pre-registered honesty):** a learned reliance induced by prevalence stress lives in
higher-order / covariance / nonlinear structure, so **E4-first-moment alone may NOT repair it.** A PC2-E4
negative would be *informative*: it would sharpen the scope to "repair certified for first-moment injected
shortcuts only," consistent with the project's measurement→control gap. This preflight is designed so that
outcome is a clean, reportable result — not a failure to avoid.

## Induction of the learned reliance (unchanged from FSR_23 Q1–Q3)
GPU ERM refit with skewed source prevalence: assign each source subject `d` a spurious class `c_d`; skew
`P(y|subject=d)` toward `c_d` at stress `ρ ∈ {0.0, 0.5, 0.8}` **while holding the source class marginal `P(y)`
fixed** (coupling in the joint, not imbalance); a **shuffled-stress** control (spurious structure destroyed)
rules out class-imbalance; **dose-response** (subject-decodability + target harm monotone in ρ) confirms the
reliance was *learned*. Target kept as-is (balanced accuracy). All source-only; target labels score only. This
is the expensive part and is the GPU that stays unauthorized here.

## Repair primitives tested on the learned shortcut (the FSR_28 change vs FSR_23)
All target-X-only, source-fit, netted against a clean-target arm, judged with the **Phase-4F-corrected gate**:
- **E4 — first-moment mean alignment** (`z − λ(mean(z_T) − μ_src)`): the Phase-4F primitive. Primary test of
  whether first-moment repair generalizes to a learned reliance.
- **E4b — second-moment / whitening extension** (CORAL-style: align target feature covariance to source, in
  addition to the mean): pre-registered successor for the likely case that the learned reliance is **not**
  first-moment. Deployable, target-X-only.
- **E3 random-direction control**, **ERASE negative control** (structural, per 4F), **E0 exact** (oracle bound;
  note on a *learned* reliance there is no known injected token, so E0 is not available — replaced by an
  oracle-subject-subspace bound estimated with source labels only).
- **Clean-target netting** and the **non-identity subset** report are mandatory (a learned reliance will
  generally NOT produce mechanical-identity rows, so the 4F tautology confound should be absent — which itself
  is a useful diagnostic).

## Gate (inherits every Phase-4F correction)
- **Structural veto set** {E4-family, E3} vs {ERASE} negative-control-by-construction + one-sided falsification.
- **Clustered bootstrap** over `(dataset, target_subject, ρ)` clusters for all gate CIs.
- **Leave-one-DATASET-out is the BINDING robustness gate** (the 4F lesson — LOSO-seed was near-inert). A claim
  must survive dropping the carrying dataset. More datasets than 2 are strongly preferred (see below).
- **Non-identity netted** is the headline (exclude any mechanical-identity rows).
- **PRIMARY = absolute netted gain (bAcc)**; ratio is secondary/marginal-denominator-flagged.
- Grades `none / partial / strong` with the identical thresholds; `strong` requires leave-one-dataset-out.

## GPU go/no-go (pre-registered; PM-gated)
```text
PC2-E4 GPU run authorized ONLY IF the PM approves this preflight AND:
  1. the induction preflight (CPU/small-scale synthetic) shows a rho dose-response in subject-decodability
     (else the stress does not induce a learned reliance -> re-design, no GPU);
  2. >= 3 datasets are available for the leave-one-dataset-out binding gate (2 is too few, per 4F);
  3. GPU budget + per-fold checkpointing plan approved (FSR_23 estimate: ~252 ERM refits, ~25-60 GPU-h).
Even with a pass, PC2 remains a CONTROLLED learned-reliance positive control -> certifies learned first/second-
moment repair, NOT natural shortcut repair. Natural 4B/4D = none is untouched.
```

## What this preflight does NOT do
- Does **not** authorize any GPU run (`pc2_gpu_run_authorized = false`).
- Does **not** re-score Phase 4E (`none`) or upgrade Phase 4F beyond `strong_within_controlled_first_moment_scope`.
- Does **not** claim E4 repairs learned/natural shortcuts — that is the *question* PC2-E4 would test, with a
  guarded (possibly negative) expectation.

## Recommendation to PM
Phase 4F is an honest, scoped positive (first deployable repair to clear a corrected pre-registered bar, but
construction-matched + dataset-carried). The scientifically highest-value next step is **PC2-E4** — it directly
tests whether first-moment (or second-moment) neutralization survives contact with a **learned** reliance, and
either outcome is publishable (positive → repair generalizes a step beyond construction; negative → repair is
first-moment-specific, sharpening the measurement→control gap at the intervention layer). But PC2-E4 needs
(a) ≥3 datasets to make leave-one-dataset-out meaningful, and (b) explicit GPU authorization. **Awaiting PM
decision; no GPU is committed by this document.**
