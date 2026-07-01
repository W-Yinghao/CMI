# ACAR_FROZEN_v5.md — V5 protocol: Substrate-Robust Constrained-Utility Router **(DRAFT — UNTAGGED — NON-BINDING)**

```
STATE              : DRAFT / UNTAGGED / NON-BINDING
                     NO CODE RUNS · NO DEV SELECTION · NO SUBSTRATE TRAINING · NO COMPATIBILITY REPLAY ·
                     NO EXTERNAL READ · NO LOCKBOX CONSUMED
binding when       : committed AND tagged `acar-v5-protocol` after explicit sign-off (NOT before)
lineage            : v2 MEASUREMENT_ONLY (9b2f0c1/1528a94) · v3 DEV_STOP (817b04f/9f4e83f) ·
                     v4 SUBSTRATE_COMPATIBILITY_FAIL (b99fa4f/5237378/c605e24; see notes/ACAR_V4_CLOSEOUT.md)
prior on v4        : the v4 DEV #001 candidate is a NEGATIVE PRIOR ONLY (did not survive substrate regeneration); it is NOT a
                     starting point and its score family / thresholds are NOT to be reused as defaults
authoring sub-docs : notes/ACAR_V5_CANDIDATE_SPACE.md · notes/ACAR_V5_ENDPOINTS.md · notes/ACAR_V5_SPLITS.md
step 2b (pre-tag)  : PINNED — policy space = EXACTLY P1–P5 (no P6–P8 without a dated pre-run amendment); ≤24 total configs;
                     λ/threshold grid = FIT-only quantiles {q50,q60,q70,q80,q85,q90}, ≤6 pts/family; LTT Holm α=0.05 over H1–H3
                     ONLY (H4 is the ε=0.02 effect-size gate, not Holm); G5 red_upper + best-fixed-P3 comparator defined; G6 =
                     modules S1–S3 each required, S1 pass = ≥2/3 seeds, NO reselection; P4 seed-agreement = frozen 3-substrate
                     ensemble or excluded; S2 = FIT-only unlabeled standardization, held-out labels evaluation-only
```

> **⛔ HARD NO-EXECUTION CLAUSE.** No V5 code scaffold, DEV run, substrate training, candidate selection, compatibility replay,
> or external/held-out read is authorized by this draft alone. Execution requires a later explicit sign-off and the
> `acar-v5-protocol` tag. This file pins the SCIENCE; it does not authorize compute.

## 0. Objective
**ACAR-V5 = a Substrate-Robust Constrained-Utility Router.** Learn a *label-free* adaptation router on an
**external-compatible frozen substrate** such that, per disease, it simultaneously satisfies (i) deployed **utility** beating the
v2-replay comparator by a pre-registered margin, (ii) a **coverage** floor, (iii) an **all-batch harm** cap, (iv) a **conditional
adapted-harm** cap, and (v) **stability across pre-registered substrate / seed / cohort stress tests**. The contribution is a
**protocol + objective correction**, not a new feature: V5 puts *substrate robustness* and *conditional adapted-harm control* at
the center — the two failure modes that killed v4 (`notes/ACAR_V4_CLOSEOUT.md`).

## 1. Core principle (the v4 lesson, inverted)
**Freeze the deployable representation FIRST, then select the router.** v4's root error was selecting a router on per-fold LOSO
embeddings and only checking transfer to an all-DEV (external-compatible) substrate afterward — the candidate was a local
regularity of the OOF substrate, not a substrate-invariant law. V5 uses the **external-compatible substrate from day one** for DEV
embedding, selection, robustness, and external execution. Any embedding without a pinned substrate hash (Stage 0) is inadmissible
for selection.

## 2. Protocol stages (each is a SEPARATELY gated decision; this draft authorizes none)
- **Stage 0 — V4 closeout + artifact hygiene.** v4 is CLOSED (`notes/ACAR_V4_CLOSEOUT.md`). New isolated package `acar/v5/`
  (no reuse of v4 selection results except as a negative prior). Every V5 artifact MUST carry the full provenance hash set
  (Stage-0 registry; see `ACAR_V5_SPLITS.md` §Artifact hygiene): encoder weights · source-state/normalization state ·
  preprocessing config · channel montage · sampling rate · windowing config · cohort inclusion list · random seed · git commit ·
  environment hash · feature-dump hash. **No hash ⇒ the embedding is inadmissible for candidate selection.**
- **Stage 1 — external-compatible DEV substrate** (`ACAR_V5_SPLITS.md`). Per-disease all-source DEV encoder + source-state on the
  pre-registered DEV cohorts, under the FROZEN pipeline; DEV evaluation embeddings come from the SAME kind of substrate the
  external arm will use (NOT old LOSO dumps).
- **Stage 2 — pre-registered DEV selection** (`ACAR_V5_CANDIDATE_SPACE.md`). The pre-registered policy space is EXACTLY the five
  families P1–P5 with ≤24 total configs (NO open sweep). Selection target = constrained utility (Stage-3 gates), NOT a single
  macro-red.
- **Stage 3 — endpoints / gates** (`ACAR_V5_ENDPOINTS.md`). Primary gates G1–G6, statistical certification, stop rules.
- **Stage 4 — substrate-robustness compatibility (BUILT-IN gate, not post-hoc).** A candidate must pass G1–G5 on ALL pre-registered
  stress tests S1–S3 (`ACAR_V5_SPLITS.md` §Robustness) = G6. Fail ⇒ STOP; do NOT proceed to external.
- **Stage 5 — external Arm B (ONCE).** Only after Stage 4 passes. Single-site held-out per disease (pinned in `ACAR_V5_SPLITS.md`).
  Report as "single-site held-out confirmation", NOT a cross-site generalization claim.

## 3. Stop rules & honest outcomes (no rescue, no tuning)
- **No candidate passes G1–G6 on DEV+robustness ⇒ `DEV_STOP / NO_LOCKBOX_CONSUMED`** (as v3/v4). No external read.
- **Safety/coverage certified but utility not ⇒ NOT a pass.** A policy that "adapts rarely but safely" or "adapts a little but
  badly" FAILS — this is the explicit v4-derived rule (G4 + utility, see `ACAR_V5_ENDPOINTS.md`).
- **No post-hoc tuning.** After DEV selection the candidate is FIXED; no search over candidate / score family / policy / loss /
  λ-grid / comparator / thresholds after seeing robustness or external results. Continuation = a NEW dated protocol (v6), never an
  in-place edit of this file or its results.
- **Negative-result branch is acceptable and pre-committed.** If even a conservative abstaining router cannot hold positive
  utility under robustness, the ACAR scientific conclusion turns NEGATIVE: *label-free adaptation-risk features are measurable but
  insufficient for stable deployment-utility control in these EEG disease-transfer settings.* That is a publishable result, not a
  failure to be tuned away.

## 4. Boundaries inherited (unchanged)
Deployment routing uses NO labels (labels enter only risk/evaluation code — the v2 leakage discipline). v2 (`MEASUREMENT_ONLY`)
and v3 (`DEV_STOP`) and v4 (`SUBSTRATE_COMPATIBILITY_FAIL`) results are immutable. TUAB sealed. The held-out lockbox stays SEALED
until Stage 5 under an explicit external authorization. Subject is the statistical unit throughout.
