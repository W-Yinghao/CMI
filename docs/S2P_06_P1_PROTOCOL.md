# S2P_06 — P1 Matched-Exposure Subject-Scaling Pilot (pre-registration v3)

**Project S2P — P1.** Pre-registration of the CBraMod controlled pretraining pilot. **v3 supersedes v2.** v1 (single
fixed-H0 line) was not identifiable; v2 (two-H0 crossed decomposition) restored algebraic identifiability but the v2
red-team (S2P_07) showed the crossed **exposure** axis is (a) internally inconsistent for a frozen encoder
(a cell cannot be in a diversity pool and an exposure common-pool at once), (b) near-singular/low-power (VIF 6.36),
and (c) population-confounded when pooled. **PM decision (2026-07-08): DROP the exposure coefficient and the crossed
regression; keep the three within-pool matched-exposure NESTED pairs; descriptive slopes + leave-one-pair
sign-stability (no extra off-diagonal cells).** P1 is now a **narrow, clean matched-exposure subject-scaling pilot**,
not a full subject-diversity/exposure decomposition. Runs CBraMod **native** pretraining (`pretrain_main →
Trainer_valid → CBraMod → generate_mask + MSE`, 9A.5 audit) via the rewritten TUEG loader — no objective/mask/
architecture rewrite. CodeBrain is NOT in P1 (P2 + non-blocking infra smoke only). PC2 paused; Paper 1 unaffected;
Paper 2 frozen. **Launch only after all conditions in `p1_launch_go_nogo.json` pass; report the checklist and WAIT
for explicit PM go — do not auto-launch.**

## Primary question (narrowed)
> **Matched-exposure subject-scaling.** Within a **single eligibility pool** and at a **fixed per-subject exposure**,
> does adding more pretraining subjects (few-deep → many-deep, *same depth per subject*) change the learned
> representation and downstream transfer?

- **NOT** a pure subject-diversity *causal* effect at fixed total budget (BL-6): the three pairs sit on different
  populations, so they are reported **separately**, never pooled into one causal diversity slope.
- **NOT** a per-subject-exposure effect (BL-5): the exposure coefficient / crossed regression is **dropped** — the
  exposure axis is triple-confounded (population × window-redundancy × depth) and uninterpretable in this corpus.
- **NOT** sample-size scaling within-subject (each subject contributes the *same* exact window budget in both cells
  of a pair), full-TUEG pretraining, or a reproduction of published CBraMod.

## Grid — three within-pool matched-exposure NESTED pairs (PM-approved; 18 runs)
Each pair fixes per-subject exposure e and a common eligibility pool; the low-N training subjects are a **nested
subset** of the high-N training subjects (small ⊂ large), so the contrast is *"same per-subject depth, more subjects
added."* From scratch, seeds {0,1,2}. **N=32 and the exposure-contrast cell (200h/N512) are NOT in P1.**

| pair | low cell | high cell | nominal e (h/subj) | quantized e (cap_windows) | pool (floored-win, MJ-8) |
|---|---|---|---|---|---|
| **A** | 100h / N=128 | 200h / N=256 | 0.781 | 0.783 (94 win) | 713 |
| **B** | 100h / N=512 | 200h / N=1024 | 0.195 | 0.192 (23 win) | 6486 |
| **C** | 100h / N=1024 | 200h / N=2048 | 0.098 | 0.100 (12 win) | 6516 |

- **Window quantization (documented):** the per-subject budget is `cap_windows·30 s`; nominal vs quantized e differs
  by −0.27% / +1.87% / −2.40% for A/B/C — a **constant shared by both cells of a pair** (same `cap_windows`) that
  **cancels exactly in the within-pair contrast**. The quantized value is the real matched budget; the "h" label is
  nominal (`p1_primary_matched_exposure_pairs.csv`).
- **Compute:** 6 cells × seeds{0,1,2} = **18 CBraMod pretraining runs.** Optional high-N diagonal
  (`100h/N2000`, `200h/N4000` @ e=0.05) is **deferred, NOT launched** — only considered if the 18-run pilot shows
  signal (`stage=P1_highN_deferred` in the manifest).

## Primary statistic (frozen — `p1_primary_statistic_spec.json`)
```
Δ_pair = target_bAcc(high-N, high-H0, same exposure) − target_bAcc(low-N, low-H0, same exposure)
PRIMARY = Δ_subject_scale = unweighted_mean(Δ_A, Δ_B, Δ_C)        # equal weight per exposure regime (pilot)
```
- **Metric:** SHU-MI target-bAcc (single primary target). **Aggregation:** unweighted mean (no pair dominates).
- **Stability (required):** leave-one-pair-out sign-stability; ≥2/3 seeds agree in sign per pair; seed-SD ≤ 0.03;
  no single seed > 70% of the pooled effect (MJ-6).
- **Secondary:** `Δ_L1` (pairwise subject separability), `Δ_L4` (alignment), `Δ_L5` (reliance vs variance-null),
  `Δ_L6` (target consequence) — each per pair, reported with the same stability checks.
- **Descriptive slope only:** a `logN` slope may be shown descriptively; **no** inferential `β_N`/`β_e`, **no**
  crossed regression, **no** pooled causal claim.

## Loader (BL-4/BL-8/MJ-8 fix — `s2p/scripts/tueg_subject_loader.py`, VERIFIED)
`build_matched_exposure_pair(exposure_h, n_low, n_high, subset_seed, n_val=64)` implements the design in code:
- **Common eligibility pool** = subjects with **floored available windows ≥ cap_windows** (MJ-8, not summed hours).
- **Fixed per-contrast pretrain-val pool** (n_val=64), drawn with a per-exposure RNG **invariant to subset_seed**,
  **disjoint from every training subject** → comparable best-val-loss checkpoint selection within a pair (MJ-2).
- **Nested** low ⊂ high training subsets (MN-2). **Exact per-subject cap** (each subject = cap_windows windows).
- **subset_seed ≠ init_seed** (MJ-5): the loader seed controls the subject draw only; the training/init seed is
  separate (trainer-side).

**Verified on the real corpus (`p1_loader_balance_verification.csv`, all 18 cells + val × 3 seeds):**
per-subject window max−min = **0**, Gini = **0.0**, total hours within **0.0002%** of the quantized budget, nested
low⊂high **True**, train/val disjoint **True**, all pools feasible (713 / 6486 / 6516 ≥ n_val + n_high).

## Native training (no rewrite)
`Trainer_valid` + `CBraMod`; masked-patch **reconstruction** MSE, mask ratio **0.5** per-(B,C,patch), zeros
mask-token, fp32. **Thin adapter only:** emit fp32 `(B,19,30,200)`; **neutralize the hardcoded 129-ch
`EEGNormalizer`** (loader already per-window z-scores; do NOT also `/100`); hand our loaders to `Trainer_valid`.

## FROZEN gates (BL-2/BL-3 — pre-registered)
**Sampling balance (hard launch gate; VERIFIED above):** per-subject window max−min ≤ 1; Gini ≤ 0.02; total hours
within ±1% of the quantized budget.
**Convergence / positive-control (per cell):** pretrain-val-loss relative decrease ≥ 20%; downstream source-val
bAcc ≥ random-init-frozen + 0.02; no NaN/Inf; checkpoint reload exact. **If gate-pass is N-dependent within a pair →
report as a convergence-mediated effect, NOT a subject-scaling effect** (no silent censoring: report Δ with and
without gated cells).
**MDE:** target-bAcc +0.02; pairwise-L1 −0.03 (abs); L5 |Δ|≥0.01 vs variance-null.
**Seed stability:** seed-SD ≤ 0.03; ≥2/3 sign agreement; ≤70% single-seed share; leave-one-pair-out sign-stable.

## Validation + checkpoint (firewall)
Pretrain-**val** = the fixed per-contrast pool, subject-disjoint from both training cells. **Primary checkpoint =
best pretrain-val loss; secondary = last.** Downstream/target performance **never** selects the checkpoint. No target
label in subset / checkpoint / PCA / head / rank / probe. Target labels **final scoring only**.

## Downstream evaluation
Per checkpoint × dataset (SHU-MI primary; PhysioNetMI large/weak; BNCI sanity), frozen encoder → F1 spatial →
source-only PCA/head. **L1 = mean PAIRWISE subject separability** (dimension-invariant, 2-way, run/session-held-out,
HELD-OUT subjects, fixed probe + fixed PCA rank across cells, 3-way subject-disjoint split, dynamic-range check off
the 0.5 floor and ~0.95 ceiling — and confirm per-window z-score has not floored L1). Per-cell task gate
(source-val ≥0.58) → L4/L5/L6 interpretable. CIs clustered by eval subject. **Multiplicity:** one primary
(Δ_subject_scale on SHU-MI target-bAcc), Holm across the pre-registered secondary family.

## Population disclosure (MJ-1/MJ-7 — `p1_population_balance_diagnostics.csv`, `p1_common_eligibility_pools.csv`)
Each pair reports pool size, median available windows/subject, and **window-redundancy** (contiguous windows/subject
= cap_windows: 94 / 23 / 12 for A/B/C). Redundancy and population differ **across** pairs but are **matched within**
each pair (both cells identical), so the within-pair contrast is clean; the three pairs are reported **separately**
with their pool/population/redundancy labels. Safe wording: *"adding subjects within a fixed eligibility pool at
matched per-subject exposure"* — never *"a pure subject-diversity causal effect."*

## Interpretation grid (pre-registered)
```
Δ_pair transfer ↑ AND Δ_L1 ↓, consistent across pairs + seeds   -> more subjects at matched depth reduce separability + improve transfer
Δ_pair transfer ↑, L1 high, Δ_L5 null/decreases                 -> more subjects change the ROLE of subject info (FSR-positive, strongest)
Δ_pair ≈ 0 across pairs                                          -> "under this matched-exposure pilot, no subject-scaling signal" (NOT "diversity doesn't matter")
sign flips across pairs / seeds / leave-one-pair-out            -> unstable; P2 not justified
gate-pass N-dependent                                           -> convergence-mediated, not subject-scaling
```
**Forbidden:** "subject diversity does not matter"; "foundation encoders become subject-invariant"; a per-subject-
exposure causal claim; a pooled fixed-budget diversity causal claim; growing-hours read as diversity; reproduction of
published CBraMod; SOTA/full-FT. **Every claim carries the matched-exposure / within-pool / single-target qualifier.**

## Outputs (`results/s2p_p1_cbramod/`)
`p1_run_manifest.csv`, `p1_pair_subset_manifest.csv` (pool + nested lineage + subset/init seeds),
`p1_actual_exposure_by_subject.csv`, `p1_sampling_balance_checks.csv`, `p1_pretrain_logs.csv`,
`p1_positive_control.csv`, `p1_checkpoint_manifest.csv`, `p1_downstream_task_performance.csv`,
`p1_pairwise_subject_separability.csv`, `p1_l4_task_alignment.csv`, `p1_l5_replay.csv`, `p1_l6_target_consequence.csv`,
`p1_matched_exposure_summary.csv` (Δ_pair + Δ_subject_scale + leave-one-pair-out + stability),
`p1_target_label_firewall.json`, `p1_verdict.json` (`subject_scaling_signal`, `representation_level_effect_seen`,
`stable_across_pairs_and_seeds`, `recommend_p2`, `target_labels_used_for_selection:false`).

## STOP rules
```text
1  any objective/mask/architecture reimplementation (must be native Trainer_valid).
2  target labels in subset / checkpoint / PCA / head / rank / probe selection.
3  sampling-balance gate fail (window max−min ≤1 / Gini ≤0.02 / ±1% quantized budget) -> that cell not launched.
4  pretrain-val not the fixed per-contrast subject-disjoint pool.
5  nested low ⊄ high, or train∩val ≠ ∅ -> pair not interpretable.
6  any pooled causal diversity claim or a per-subject-exposure coefficient (both dropped by design).
7  max-data cell fails positive-control floor -> under-powered -> STOP (no null claim).
8  seed/pair-unstable (SD>0.03 / <2/3 sign / one seed>70% / leave-one-pair flips) -> do not interpret.
9  CodeBrain enters the P1 science grid (P2 only).
```

## Fallback
The matched-exposure pilot is the narrowed fallback of S2P_07 option (c). If even the within-pool pairs prove
uninterpretable, report a descriptive "few-deep vs many-deep at matched exposure" observation only.

## P2 conditionality
P2 (larger grid / constant-exposure growing-hours diagnostic / CodeBrain native) **only if** P1 shows a clear,
stable Δ_subject_scale transfer gain, an L1 reduction, or an L5 reliance change. Null/unstable → no P2, no CodeBrain
full grid, no 33-channel corpus. Return for PM review before P2.
