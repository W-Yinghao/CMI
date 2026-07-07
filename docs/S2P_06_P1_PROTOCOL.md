# S2P_06 — P1 CBraMod Fixed-Hour Subject-Diversity Pilot Protocol (pre-registration)

**Project S2P — P1.** Pre-registration of the CBraMod controlled subject-diversity pretraining pilot. PM-approved in
principle; **launch only after design-red-team (S2P_07) + `p1_launch_go_nogo.json` pass**. Runs CBraMod's **native**
pretraining (9A.5 audit: `pretrain_main → Trainer_valid → CBraMod → generate_mask + MSE`) via a **thin TUEG
subject-subset loader adapter** — no objective/mask/architecture rewrite. CodeBrain is NOT in P1 (P2 + infra smoke
9B-0C only). PC2 paused; Paper 1 unaffected; Paper 2 frozen.

## What P1 is (and is NOT)
> **Fixed-hour subject-diversity scaling under sparse per-subject exposure.** At **fixed** pretraining hours
> (H0=100 h), move from few-subject/deep-exposure to many-subject/sparse-exposure and measure whether the learned
> representation and downstream transfer change.

- **NOT** "more subjects alone helps" (H0 is fixed; per-subject exposure *shrinks* as N grows — an explicit design
  tradeoff, reported, not a confound to hide).
- **NOT** sample-size scaling (H0 fixed controls total hours; it does **not** control per-subject exposure).
- **NOT** full-TUEG-scale foundation pretraining or a reproduction of published CBraMod (0.5–45 Hz processed band,
  19-common subset, from-scratch, small budget).

## Grid (PM-approved)
- **Model:** CBraMod (native), from scratch. **Corpus:** 19-common canonical TUEG subset (6,535 subj / 3,483 h).
- **H0 = 100 h** fixed. **`N_subjects ∈ {32, 128, 512, 1024, 2000}`**. **`min_exposure = 0.05 h/subj` (3 min).**
  **`seeds ∈ {0,1,2}`** (subset + init). 15 pretraining runs.
- **Explicit exposure tradeoff (must be reported per cell):**
  | N | per-subject exposure (H0/N) |
  |---|---|
  | 32 | ~3.125 h | 128 | ~0.78 h | 512 | ~0.195 h | 1024 | ~0.098 h | 2000 | 0.05 h |

## Native training (no rewrite)
`Trainer_valid` + `CBraMod`; masked-patch **reconstruction** MSE on `x[mask==1]`, mask ratio **0.5** per-(B,C,patch),
zeros mask-token, fp32, seed. **Thin adapter only:** emit bare fp32 `(B,19,30,200)`; **neutralize the hardcoded
129-ch `EEGNormalizer`** (we pre-z-score per window; do NOT also `/100`); hand our loaders to `Trainer_valid`.

## Subject-contribution balance (CRITICAL — 9A.5 sampler finding)
CBraMod's native DataLoader is **uniform-over-sequences** → high-hours subjects dominate. P1 enforces the per-subject
budget in the **loader** (each subject capped at H0/N hours). **Every run must output** `actual_hours_per_subject`,
`actual_segments_per_subject`, max/min/mean per-subject segments, and **subject-contribution Gini**. **If high-hours
subjects still dominate (Gini above a pre-declared threshold), the cell is NOT interpretable** (STOP-balance).

## Validation + checkpoint (firewall)
Pretrain-**val** is **subject-disjoint** from pretrain-train (not recording-disjoint only). **Primary checkpoint =
best pretrain-val loss; secondary = last.** **Downstream/target performance NEVER selects the checkpoint.** No target
label in subset choice, checkpoint selection, PCA, head, rank, or probe.

## Downstream evaluation (representation scaling)
Per checkpoint × dataset (SHU-MI primary decodable; PhysioNetMI large/weak; BNCI sanity), frozen encoder → F1 spatial
→ source-only PCA/head. Metrics (FSR-hardened): target bAcc / macro-F1; **L1 = mean PAIRWISE subject separability**
(dimension-invariant, 2-way, run/session-held-out, on HELD-OUT subjects); L4 alignment; **L5 subject-subspace vs
variance-MATCHED null**; L6 target consequence. Per-cell task gate (source-val ≥0.58) → L4/L5/L6 interpretable.
Slopes on `log(N_subjects)` with **CIs clustered by eval subject** + **per-subject-exposure as a covariate** (so a
diversity effect is separated from the exposure shrink). Target labels **final scoring only**.

## Interpretation grid (pre-registered)
```
target transfer improves with N AND L1 falls        -> subject-diverse pretraining reduces separability + improves transfer
transfer improves, L1 high, L5 null/decreases       -> diversity changes the ROLE of subject info (FSR-positive, strongest)
no transfer, no L1/L5 change                         -> "under this 100h fixed-budget pilot, no subject-diversity signal" (NOT "diversity doesn't matter")
seed-dependent / source-val up but target down       -> unstable; P2 not justified
```
**Forbidden:** "subject diversity does not matter"; "foundation encoders become subject-invariant"; "more subjects
alone helps"; reproduction of published CBraMod; SOTA/full-FT. **Every claim carries the fixed-hour/sparse-exposure
qualifier.**

## Outputs (`results/s2p_p1_cbramod/`)
`p1_run_manifest.csv`, `p1_subject_subset_manifest.csv`, `p1_recording_subset_manifest.csv`,
`p1_actual_exposure_by_subject.csv`, `p1_pretrain_logs.csv`, `p1_checkpoint_manifest.csv`,
`p1_downstream_task_performance.csv`, `p1_pairwise_subject_separability.csv`, `p1_l4_task_alignment.csv`,
`p1_l5_replay.csv`, `p1_l6_target_consequence.csv`, `p1_fixed_hours_scaling_summary.csv`,
`p1_target_label_firewall.json`, `p1_verdict.json` (`subject_diversity_signal`, `representation_level_effect_seen`,
`recommend_p2`, `target_labels_used_for_selection:false`).

## STOP rules
```text
1  any objective/mask/architecture reimplementation (must be native Trainer_valid).
2  target labels in subset / checkpoint / PCA / head / rank / probe selection.
3  subject-contribution Gini above threshold (high-hours subjects dominate) -> cell not interpretable.
4  pretrain-val not subject-disjoint from pretrain-train.
5  a cell infeasible under min_exposure=0.05h (should not occur; verify in go_nogo).
6  seed-dependent pretraining (seeds disagree beyond tolerance) -> unstable, do not interpret.
7  CodeBrain enters the P1 science grid (P2 only).
```

## P2 conditionality
P2 (larger grid / growing-hours / CodeBrain native) **only if** P1 shows a clear target-transfer gain, an L1
reduction, an L5 functional-reliance change, or a strong mixed signal. Null/unstable → no P2, no CodeBrain full grid,
no 33-channel corpus. Return for PM review before P2.
