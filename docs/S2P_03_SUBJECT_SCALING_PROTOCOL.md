# S2P_03 — Subject-Scaling Pretraining Protocol (Phase 9; pre-registration)

**Project S2P — Phase 9.** Pre-registration of the controlled subject-count pretraining design. Encodes the
FSR-hardened metric/statistics lessons (dimension-invariant pairwise-L1; clustered CIs; variance-matched L5; L1 on
held-out subjects; capacity/hours isolation). CBraMod primary, CodeBrain-Stage2 secondary. Downstream audit reuses
the FSR L1/L4/L5/L6 ladder. To be design-red-teamed before any P1/P2 run.

## Design — subject count × hours condition (the crux) — REVISED after 9B-0 inventory
The **19-common canonical corpus is subject-RICH (6,535 subj) but hours-POOR (3,483 h, median 0.35 h/subj)**, so the
original `{32,128,512}@H0{250,500}` was both **too conservative at the top** (corpus supports N≫512) **and
infeasible at the low end** (H0=250 h × N=32 ⇒ 7.8 h/subj, only 5 subjects qualify). The binding constraint is
**per-subject hours**, not subject count. Revised **P1 grid** (PM):
- **fixed-hours `H0 = 100 h`** with an explicit **`min_per_subject = 0.05 h` (3 min)** exposure floor →
  `N_subjects ∈ {32, 128, 512, 1024, 2000}` (per-subject 3.1 / 0.78 / 0.20 / 0.098 / 0.05 h) — a real high-subject
  range with minimal protected exposure. `seed ∈ {0,1,2}`.
- **growing-hours** (per-subject cap fixed, total grows) → the CodeBrain-style data-volume axis (sample-size).
- **Optional exploratory endpoint** (NOT primary): `N = all-eligible` at H0=100 h ⇒ ultra-sparse (<3 min/subj) —
  label **"ultra-sparse subject-diversity condition"**; not comparable to the exposure-protected cells.
- **Diversity claim requires the FIXED-hours arm** (fixed total, more subjects). growing-only ⇒ *sample-size*, not
  diversity (FSR-8C lesson).

## Firewall (pretraining + downstream)
Checkpoint selected by **pretrain-val loss on held-out pretraining subjects** only (S2P_02). Downstream: PCA/head/
subspaces on **source-train**; task gate on **source-val**; L1 on **held-out** source subjects; **target labels
final-scoring only**. No target label in pretraining-subset choice, checkpoint selection, PCA, head, rank, or probe.

## Downstream evaluation
- **Primary:** SHU-MI (decodable MI — the FSR-8B setting where L4/L5/L6 are interpretable), PhysioNetMI
  (large-subject, weak-task, disclosed). **Secondary/sanity:** BNCI2014_001 / 2015_001 (few subjects, alignment
  only). Per-dataset only (no cross-dataset magnitude claims).
- Per pretrained checkpoint: target bAcc, macro-F1; **L1 = mean pairwise subject separability (dimension-invariant,
  2-way, run/session-held-out)** on downstream embeddings; L4 task-head↔subject-subspace alignment; **L5
  subject-subspace erase vs variance-MATCHED null** (per removed-variance); L6 target consequence. Per-cell **task
  gate** (source-val ≥ 0.58) → L4/L5/L6 interpretable, else `WEAK_TASK_NOT_INTERPRETED`.

## Statistics (FSR-hardened)
Slopes on `log(N_subjects)` with **CIs clustered by target/eval subject** (not cell-level — the FSR-8C correction),
+ **hours as a covariate**; mixed-effects (random effects over eval subject + pretrain seed). N=all/max = single
composition (no seed pseudo-replication). Pre-declared **minimum detectable effect** reported; a null ≠ "no effect"
unless MDE < the effect of interest.

## Interpretation grid (pre-registered)
```
growing improves, fixed flat                    -> hours/sample-size effect (NOT diversity)
fixed improves with N                           -> subject-DIVERSITY improves transfer
transfer improves, pairwise-L1 falls with N     -> diversity suppresses subject-identifiable structure
transfer improves, L1 stays high, L5 stays null -> diversity changes the ROLE of subject info (strongest FSR result)
transfer improves, L1 high, L5-erase HURTS      -> subject info is task-useful/physiological, not harmful
no transfer scaling, no L1/L5 change            -> pilot budget insufficient (NOT "no effect"; report MDE)
```

## Gates
- **P0 smoke:** pipeline runs end-to-end (pretrain → checkpoint → feature dump → downstream audit); no science.
- **P1 pilot PASS (→ expand P2, PM review first):** a **CI-clustered, MDE-cleared** effect in the fixed-hours arm on
  ≥1 downstream (target bAcc slope on log N excludes 0 **and** ≥ MDE), **or** a pairwise-L1 / L5 change with N under
  fixed-hours; **stable across pretrain seeds** (seed SD < effect).
- **P1 NULL:** fixed-hours arm behaves like the smallest N (no transfer / L1 / L5 change) and MDE > effect-of-interest
  → **stop** or reconsider budget (do not claim absence). **P1 RISK:** seed-dependent / source-val-up-but-target-down
  → unstable, do not interpret.

## Claim hygiene (permanent)
- **Allowed:** "under controlled pretraining subject-count subsets at fixed hours, we estimate whether subject
  diversity changes downstream subject separability / task coupling / transfer."
- **Forbidden:** "subject diversity removes subject leakage"; "foundation models become subject-invariant"; "large
  TUEG pretraining solves cross-subject generalization"; any growing-hours result read as a diversity effect;
  SOTA/full-FT/leaderboard.

Phase 9A (this) → PM review → 9B-0 smoke → 9B-1 pilot (design-red-teamed) → PM review → P2. PC2 paused; Paper 1
unaffected; Paper 2 frozen.
