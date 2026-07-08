# S2P_06 — P1 Two-Budget Subject-Count × Per-Subject-Exposure Decomposition Pilot (pre-registration v2)

**Project S2P — P1.** Pre-registration of the CBraMod controlled pretraining pilot. **Supersedes v1** (single
fixed-H0 line), which the design red-team (S2P_07) proved **not identifiable** for the subject-diversity claim
(at one budget, per-subject exposure `e = H0/N` is deterministic ⇒ `corr(log N, log e) = −1` ⇒ diversity and
exposure fuse into one slope). **PM decision (2026-07-08): break the collinearity with two fixed-H0 lines
(100 h + 200 h) on a lean crossed grid; keep the subject-diversity question; do NOT reframe to a frontier unless
the two-H0 design is infeasible.** Runs CBraMod **native** pretraining (`pretrain_main → Trainer_valid → CBraMod →
generate_mask + MSE`, 9A.5 audit) via a thin TUEG subject-subset loader adapter — no objective/mask/architecture
rewrite. CodeBrain is NOT in P1 (P2 + infra smoke only). PC2 paused; Paper 1 unaffected; Paper 2 frozen.
**Launch only after the revised design red-team + all launch conditions in `p1_launch_go_nogo.json` pass; report
the checklist and WAIT for explicit PM go — do not auto-launch.**

## What P1 is (and is NOT)
> **Two-budget decomposition.** At two fixed pretraining budgets (H0 ∈ {100, 200} h) we vary subject count N so
> that some cell pairs hold **per-subject exposure** constant while N changes (a **diversity** contrast) and others
> hold **N** constant while exposure changes (an **exposure/depth** contrast). This makes both slopes estimable.

- **Primary question:** *At controlled pretraining budgets, can we separate the effect of increasing subject count
  from the effect of reducing per-subject exposure?*
- **NOT** "more subjects alone helps" (budgets are fixed; exposure is an explicit, reported axis).
- **NOT** sample-size scaling, full-TUEG-scale pretraining, or a reproduction of published CBraMod (0.5–45 Hz
  processed band, 19-common subset, from scratch, small budget).

## Grid (PM-approved two-H0 crossed; N=32 removed from primary — MJ-1 population confound)
- **Model:** CBraMod (native), from scratch. **Corpus:** 19-common canonical TUEG subset (6,535 subj / 3,440 usable-h).
- **Primary cells (7):** `H0=100h: N∈{128,512,1024}` · `H0=200h: N∈{256,512,1024,2048}`. **Seeds {0,1,2}** ⇒ 21 runs.
- **Seed-stability extension (extremes):** `100h/N128` and `200h/N2048` × seeds {3,4} ⇒ +4 runs (subset-seed ×
  init-seed factorial at the extremes, MJ-5). **Total primary compute = 25 CBraMod pretraining runs (≈2× v1, PM-accepted).**
- **Optional high-N diagonal (NOT a launch blocker):** `100h/N2000`, `200h/N4000` (both e=0.05 h) × {0,1,2} ⇒ +6,
  **only if** feasibility + subject-balance + no whole-recording overshoot all pass at those N.
- **`min_exposure = 0.05 h/subj`.** **N=32:** removed from primary (draws ~80 extreme long-recording clinical
  subjects, MJ-1); available only as an exploratory deep-exposure tail, never in the primary slope.

### Exposure crosswalk (verified feasible on real 19-common corpus — `p1_feasibility_by_cell_v2.csv`)
| H0 | N | exposure e=H0/N (h) | eligible subjects (≥e usable-h) |
|---|---|---|---|
| 100 | 128 | 0.781 | 713 | 100 | 512 | 0.195 | 6485 | 100 | 1024 | 0.098 | 6516 |
| 200 | 256 | 0.781 | 713 | 200 | 512 | 0.391 | 2231 | 200 | 1024 | 0.195 | 6485 | 200 | 2048 | 0.098 | 6516 |

- **Matched-exposure DIVERSITY contrasts (primary; each drawn from ONE shared pool ⇒ population-matched):**
  `e=0.781`: 100h/N128 vs 200h/N256 (pool 713) · `e=0.195`: 100h/N512 vs 200h/N1024 (pool 6485) ·
  `e=0.098`: 100h/N1024 vs 200h/N2048 (pool 6516).
- **Matched-N EXPOSURE contrasts (secondary; both cells drawn from the HIGHER-exposure common pool):**
  `N=512`: 100h(e=0.195) vs 200h(e=0.391), common pool ≥0.391 h = 2231 · `N=1024`: 100h(e=0.098) vs 200h(e=0.195),
  common pool ≥0.195 h = 6485.

### Identifiability (verified — `p1_identifiability_matrix.csv`)
Design matrix `[1, log N, log e]` over the 7 primary cells is **rank 3/3**; `corr(log N, log e) = −0.918`
(single-line was exactly −1). Both slopes estimable: `outcome ~ log N + log e` is full rank. **Condition #6 PASS.**

## STRUCTURAL confound to disclose (MJ-1, quantified — `p1_population_balance_diagnostics.csv`)
In TUEG's 19-common subset, **only clinical (epilepsy-monitoring) patients have long recordings**, so exposure
**range** and clinical **population** are structurally entangled:

| exposure pool | n subj | median usable-h | median n_rec | % single-recording |
|---|---|---|---|---|
| ≥0.781 h | 713 | 1.10 | 4 | 11.9% |
| ≥0.391 h | 2231 | 0.67 | 2 | 32.4% |
| ≥0.195 h | 6485 | 0.34 | 1 | 66.7% |
| ≥0.098 h | 6516 | 0.34 | 1 | 66.8% |

**Consequence:** the three **diversity** contrasts are each internally population-matched (same pool) and are the
**primary, most-defensible estimate**. The **exposure** axis (and the `log e` coefficient of the crossed model)
inherently drags population and **cannot be cleanly separated from clinical population in this corpus** — reported
**with population as a named, quantified confound**, not as a clean causal exposure effect. The deep-exposure
diversity contrast (e=0.781, pool 713) is valid **within** its clinical population but is **not pooled naively**
with the general-population contrasts into a single diversity slope; per-pool diversity effects are reported, and
the pooled slope is reported **only** alongside the population diagnostics.

## Common eligibility pools + nested sampling (MJ-1 fix — `p1_common_eligibility_pools.csv`)
Every cell draws from the eligibility pool of **its own exposure** (`usable-h ≥ e`). For each matched contrast the
paired cells share ONE pool and are drawn **nested** (smaller-N subset ⊂ larger-N subset, same pool) so the
contrast is not polluted by a subject-population difference (MN-2 nested). Matched-N exposure contrasts draw both
cells from the **higher-exposure** common pool. Because a cell can appear in two contrasts under different pools,
each trained cell is drawn once from **its own exposure pool**; contrast-specific common-pool restriction is applied
in the *analysis* (restrict-and-report), not by retraining.

## Native training (no rewrite)
`Trainer_valid` + `CBraMod`; masked-patch **reconstruction** MSE on `x[mask==1]`, mask ratio **0.5** per-(B,C,patch),
zeros mask-token, fp32, seed. **Thin adapter only:** emit bare fp32 `(B,19,30,200)`; **neutralize the hardcoded
129-ch `EEGNormalizer`** (we pre-z-score per window; do NOT also `/100`); hand our loaders to `Trainer_valid`.

## FROZEN thresholds (BL-3 — pre-registered, no post-hoc changes)
**Sampling balance (per cell; hard launch gate — loader window-budget fix required first):**
```
actual_total_hours within ±1% of H0
actual_hours_per_subject within ±1 window (30 s) of planned cap
max−min per-subject window count ≤ 1
subject-contribution Gini ≤ 0.02
```
**Convergence / positive-control (BL-2; per cell):**
```
pretrain-val-loss relative decrease ≥ 20%   (else cell = under-converged)
downstream source-val bAcc ≥ (random-init frozen source-val) + 0.02   (positive-control floor)
no NaN/Inf; checkpoint reload byte/param exact
```
**Minimum detectable / meaningful effects (for a P2-expansion decision):**
```
target-bAcc effect        ≥ +0.02
pairwise-L1 effect         ≤ −0.03 (absolute)
L5 effect                  |Δ| ≥ 0.01 bAcc vs variance-matched null
```
**Seed stability (primary contrast):**
```
seed-SD of target-bAcc ≤ 0.03
≥ 2/3 seeds agree in sign on the primary contrast
no single seed contributes > 70% of the pooled effect
```

## BL-2 selection-bias handling (no silent censoring)
Treat **gate-pass as an analyzed outcome**: report every slope **with and without** under-converged/floor-failing
cells. If gate-pass is **N-dependent**, do **not** claim a subject-diversity slope — report a
**convergence-mediated** result. If the max-data cell fails the positive-control floor (does not beat random-init
frozen), declare **under-powered → STOP** (no "null" / "diversity doesn't matter" claim).

## Validation + checkpoint (firewall)
Pretrain-**val** is **subject-disjoint** from pretrain-train. **One FIXED common external pretrain-val set**
(constant subjects + exposure, disjoint from every training pool; MJ-2) — its loss is comparable across cells and
is reported as a target-free outcome. **Primary checkpoint = best fixed-common-pretrain-val loss; secondary = last.**
**Downstream/target performance NEVER selects the checkpoint.** No target label in subset choice, checkpoint
selection, PCA, head, rank, or probe.

## Downstream evaluation (representation scaling)
Per checkpoint × dataset (SHU-MI primary decodable; PhysioNetMI large/weak; BNCI sanity), frozen encoder → F1
spatial → source-only PCA/head. Metrics (FSR-hardened): **primary = SHU-MI target-bAcc**; secondary macro-F1;
**L1 = mean PAIRWISE subject separability** (dimension-invariant, 2-way, run/session-held-out, HELD-OUT subjects,
fixed probe + fixed PCA rank across cells, 3-way subject-disjoint split, dynamic-range check off the 0.5 floor and
~0.95 ceiling); L4 alignment; **L5 subject-subspace vs variance-MATCHED null**; L6 target consequence. Per-cell task
gate (source-val ≥0.58) → L4/L5/L6 interpretable. **Analysis model:** `outcome ~ log N + log e` (crossed, full rank)
with CIs **clustered by eval subject**; **primary diversity estimate = the three matched-exposure within-pool
contrasts**; exposure axis reported with population diagnostics. **Multiplicity: one primary (SHU-MI target-bAcc
diversity contrast), Holm across the pre-registered family** (MJ-3). Target labels **final scoring only**.

## Interpretation grid (pre-registered)
```
diversity contrast: transfer ↑ AND L1 ↓ (within-pool, matched exposure)   -> subject-diverse pretraining reduces separability + improves transfer
diversity contrast: transfer ↑, L1 high, L5 null/decreases                 -> diversity changes the ROLE of subject info (FSR-positive, strongest)
no diversity effect at fixed exposure                                      -> "under this two-budget pilot, no subject-diversity signal" (NOT "diversity doesn't matter")
effect only via exposure axis                                              -> depth/exposure (population-confounded) effect, NOT a diversity claim
gate-pass N-dependent / seed-unstable                                      -> convergence-mediated / unstable; P2 not justified
```
**Forbidden:** "subject diversity does not matter"; "foundation encoders become subject-invariant"; "more subjects
alone helps"; growing-hours read as diversity; reproduction of published CBraMod; SOTA/full-FT. **Every claim
carries the two-budget / sparse-exposure / population-confound qualifier.**

## Outputs (`results/s2p_p1_cbramod/`)
`p1_run_manifest.csv`, `p1_subject_subset_manifest.csv` (with pool + nested lineage),
`p1_recording_subset_manifest.csv`, `p1_actual_exposure_by_subject.csv`, `p1_sampling_balance_checks.csv`
(the 4 frozen gates per cell), `p1_pretrain_logs.csv`, `p1_positive_control.csv` (random-init frozen floor),
`p1_checkpoint_manifest.csv`, `p1_downstream_task_performance.csv`, `p1_pairwise_subject_separability.csv`,
`p1_l4_task_alignment.csv`, `p1_l5_replay.csv`, `p1_l6_target_consequence.csv`,
`p1_decomposition_summary.csv` (diversity vs exposure slopes ± population diagnostics),
`p1_target_label_firewall.json`, `p1_verdict.json` (`diversity_signal`, `exposure_signal`,
`population_confound_disclosed`, `representation_level_effect_seen`, `recommend_p2`,
`target_labels_used_for_selection:false`).

## STOP rules
```text
1  any objective/mask/architecture reimplementation (must be native Trainer_valid).
2  target labels in subset / checkpoint / PCA / head / rank / probe selection.
3  sampling-balance gate fail (total-hours ±1% / per-subject-window / Gini ≤0.02) -> that cell not launched.
4  pretrain-val not subject-disjoint; or not the fixed common external val set.
5  design matrix not full rank (must stay rank 3) -> not identifiable, do not launch.
6  a matched-exposure diversity contrast pool population-imbalanced vs its pair -> not interpretable.
7  max-data cell fails positive-control floor -> under-powered -> STOP (no null claim).
8  seed-unstable (SD > 0.03 / < 2/3 sign agreement / one seed > 70%) -> unstable, do not interpret.
9  CodeBrain enters the P1 science grid (P2 only).
```

## Fallback
If the two-H0 crossed grid becomes infeasible (it is currently **feasible** — all 7 cells + both contrasts verified),
fall back to the single-line **fixed-budget subject-vs-depth frontier** framing (S2P_07 option c): no diversity
claim, fused slope only.

## P2 conditionality
P2 (larger grid / constant-exposure growing-hours diagnostic / CodeBrain native) **only if** P1 shows a clear
diversity target-transfer gain, an L1 reduction, an L5 functional-reliance change, or a strong mixed signal.
Null/unstable → no P2, no CodeBrain full grid, no 33-channel corpus. Return for PM review before P2.
