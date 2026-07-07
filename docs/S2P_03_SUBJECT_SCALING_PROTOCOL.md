# S2P_03 — Subject-Scaling Pretraining Protocol (Phase 9; pre-registration)

**Project S2P — Phase 9.** Pre-registration of the controlled subject-count pretraining design. Encodes the
FSR-hardened metric/statistics lessons (dimension-invariant pairwise-L1; clustered CIs; variance-matched L5; L1 on
held-out subjects; capacity/hours isolation). CBraMod primary, CodeBrain-Stage2 secondary. Downstream audit reuses
the FSR L1/L4/L5/L6 ladder. To be design-red-teamed before any P1/P2 run.

## Design — subject count × hours condition (the crux)
Corpus = TUEG (4704743c). For each cell: pretrain a model **from scratch** (or Stage-2-from-frozen-tokenizer) on a
**subject subset**, then audit downstream.
- `N_subjects ∈ {32, 128, 512, all_or_max}` (all feasible, S2P_01).
- **fixed-hours** (H0 total held constant; per-subject cap = H0/N shrinks) → isolates **subject diversity** from
  total data. **growing-hours** (per-subject cap fixed; total grows) → the CodeBrain-style data-volume axis.
- `H0 ∈ {250, 500}` h (pilot 250). `seed ∈ {0,1,2}` (subset + init).
- **Diversity claim requires the FIXED-hours arm** (fixed total, more subjects). growing-only ⇒ *sample-size*, not
  diversity (the explicit FSR-8C lesson).

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
