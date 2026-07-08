# S2P_06 — P1 Fixed-Budget Subject-vs-Depth Frontier (pre-registration v4)

**Project S2P — P1.** Pre-registration of the CBraMod controlled pretraining pilot. **v4 supersedes v1–v3.** The
design history is an identifiability lesson: v1 (single fixed-H0 line) confounded subjects with exposure; v2 (two-H0
crossed) confounded the exposure axis with population + was near-singular; v3 (matched-exposure pairs) confounded
subjects with **total data** (BL-9). The v3 re-red-team surfaced the root cause — **the identifiability triangle
`T = N · e`**: when you vary subject count N, you can hold at most one of {per-subject exposure e, total data T}
fixed, so a **pure subject-diversity effect (N↑, both e and T fixed) is mathematically unidentifiable** at
pretraining scale. **PM decision (2026-07-08): accept BL-9 as structural; pivot to the fixed-budget subject-vs-depth
FRONTIER.** P1 now asks a clean, identifiable, deployment-relevant question and makes **no** diversity claim. Runs
CBraMod **native** pretraining (`pretrain_main → Trainer_valid → CBraMod → generate_mask + MSE`, 9A.5 audit) via the
rewritten loader — no objective/mask/architecture rewrite. CodeBrain is NOT in P1 (P2 + non-blocking infra smoke
only). PC2 paused; Paper 1 unaffected; Paper 2 frozen. **Launch only after all `p1_launch_go_nogo.json` conditions
pass; report the checklist and WAIT for explicit PM go — do not auto-launch.**

## Primary question
> **Fixed-budget allocation.** Given a **fixed** pretraining data budget (T = 200 h), should EEG foundation
> pretraining allocate it to **more subjects with shallower per-subject exposure**, or **fewer subjects with deeper
> exposure**? How does this allocation affect downstream transfer, subject separability, and functional subject
> reliance?

- This is an **allocation frontier**, **NOT** a pure subject-diversity effect (unidentifiable — the triangle).
- Per-subject exposure is **not controlled** (it is the depth axis, e = T/N); total data **is** controlled (fixed T).
- S2P's contribution: move from published *data-volume / model-size* scaling (CodeBrain, CBraMod) to the
  **subject-coverage-vs-subject-depth allocation** axis at a fixed budget.

## Grid (PM-approved; 15 runs)
- **Model:** CBraMod (native), from scratch. **Corpus:** 19-common canonical TUEG subset (6,535 subj / 3,440 usable-h).
- **T = 200 h fixed.** **`N ∈ {128, 256, 512, 1024, 2048}`.** **Seeds {0,1,2}.** 5 × 3 = **15 runs.**
- **Per-subject exposure** e = T/N: 1.56 / 0.78 / 0.39 / 0.195 / 0.098 h. **Total windows held EXACTLY at
  24000 (=200 h)** via a remainder distribution (each subject gets `base` or `base+1` windows, max−min = 1).

| N | e (h/subj) | eligible pool (floored-win) | endpoint |
|---|---|---|---|
| 128 | 1.56 | **201** | **deep — long-recording clinical (FLAGGED)** |
| 256 | 0.78 | 704 | intermediate |
| 512 | 0.39 | 2195 | general-ish |
| 1024 | 0.195 | 6357 | general |
| 2048 | 0.098 | 6388 | general (shallow) |

## N=128 handling (PM — keep but flag; NOT a population control)
N=128 draws 128 of only **201** subjects with ≥1.567 h — the extreme long-recording clinical (epilepsy-monitoring)
endpoint (0.8% single-recording vs 69% at N=2048). Because 128/201, the seed-to-seed **subset overlap is ~64%**
(MN-9), so its seed-SD measures init noise + correlated subsets, **not** subset stability — do not read N=128 seed-SD
as sampling stability. It is the **deep-subject endpoint** and is kept; the slope is reported **full frontier**
N∈{128…2048} and **robust (excl N=128)** N∈{256…2048}. **The robust cut is NOT a population control** (BL-10:
population drift persists 256→2048); it merely drops the single most extreme endpoint and is the **weakest-powered**
subset (4 points). Neither cut is population-adjusted.

## Primary estimand (frozen — `p1_primary_statistic_spec.json`)
```
PRIMARY = BUNDLED net fixed-budget allocation slope of SHU-MI target_bAcc vs log(N) at T=200 h
        = a slope that BUNDLES coverage ⊕ depth ⊕ clinical→general POPULATION ⊕ within-subject REDUNDANCY.
          It is NOT population-adjusted at ANY N, and NOT a coverage-vs-depth causal effect.
```
**BL-10 (whole-frontier population/redundancy confound — the 3rd face of the triangle).** Fixing T killed the v3
total-data confound but exposed a **structural depth↔clinical-population** confound: in TUEG-19common only clinical
patients have deep recordings (~201 subjects at the deepest threshold), so per-subject depth is inseparable from a
clinical→general population gradient. Per `p1_population_diagnostics.csv`, %single-recording rises **monotonically
across the WHOLE frontier** (256→512→1024→2048 = 12.5→32→66→69), and redundancy_ratio falls 0.649→0.286 — both
collinear with log N. So **"robust slope (excl N=128)" removes only the most extreme population endpoint, NOT the
confound** (drift persists 256→2048). The slope is population-adjusted **at no N**; allocation-vs-population
attribution is **declined**. The budget is fixed in **raw** windows/hours, not effective-independent windows (high-N
cells carry more independent information at equal 24000 windows — part of the bundled effect). **Report per-cell
`%single-recording` + `redundancy_ratio` as covariates next to every slope.** Wording is **bundled net allocation
slope** — never "coverage-vs-depth", "population-adjusted", or "subject-diversity" effect.

**Secondary frontier slopes** (each vs log N, same bundling caveat): L1 pairwise subject separability, L4 alignment,
L5 reliance vs variance-null, L6 target consequence, pretrain-val loss. **Analyses:** full-frontier slope; robust
slope (excl N=128, the **weakest-powered** cut, NOT a clean primary); **quadratic curvature + peak-location (MJ-12)**;
per-N means (reported regardless of slope); seed-specific slopes; **leave-one-N-out sign-stability**; pretrain-val-loss
frontier. CIs clustered by eval subject. **A straddling CI is a pre-registered null** ("no detectable allocation
effect at 200 h"). **Multiplicity:** one primary (bundled allocation slope on SHU-MI target-bAcc), Holm across the
secondary family.

## Loader (BL-9 fix — `s2p/scripts/tueg_subject_loader.build_frontier_cell`, VERIFIED)
Fixed total budget, variable N. Common eligibility on **floored available windows** (MJ-8); **exact 24000-window
budget** via remainder distribution; **fixed GLOBAL pretrain-val pool** (n_val=128 @ 24 windows, seed- AND
N-independent, drawn from subjects shallower than the deepest endpoint so it can never enter any training cell);
`subset_seed ≠ init_seed` (trainer-side). **Verified on the real corpus across all 15 cells × 3 seeds
(`p1_frontier_loader_balance_verification.csv`):** total = **exactly 200.0 h** (0.0% off), per-subject window
max−min = **1**, Gini ≤ **0.017**, train/val **disjoint**, **val identical across all 15 cells**.

## Native training (no rewrite)
`Trainer_valid` + `CBraMod`; masked-patch **reconstruction** MSE, mask 0.5 per-(B,C,patch), zeros token, fp32.
**Thin adapter only:** emit fp32 `(B,19,30,200)`; **neutralize the hardcoded 129-ch `EEGNormalizer`** (loader
per-PATCH (200-sample) z-scores; do NOT also `/100`); hand our loaders to `Trainer_valid`.

## FROZEN gates (BL-2/BL-3)
**Sampling balance (hard launch gate; VERIFIED):** total hours within ±1% of 200 h; per-subject window max−min ≤ 1;
Gini ≤ 0.02; no train/val overlap.
**Convergence / positive-control (per cell):** pretrain-val-loss relative decrease ≥ 20%; source-val bAcc ≥
random-init-frozen + 0.02; no NaN/Inf; checkpoint reload exact. **If low-N and high-N differ in convergence → report
a convergence-mediated frontier (no silent censoring: slopes with & without weak cells).**
**Slope robustness:** ≥3 of 5 N-points available; leave-one-N-out sign-stable. **Seed stability:** seed-SD of
target-bAcc ≤ 0.03; no single seed > 70% of the pooled effect.
**MDE:** target-bAcc slope detectable at +0.02 per log2(N); L1 −0.03; L5 |Δ|≥0.01 vs variance-null.

## Validation + checkpoint (firewall)
Pretrain-**val** = the FIXED GLOBAL pool (identical across all N and seeds; subject-disjoint from every training
cell). **Primary checkpoint = best pretrain-val loss; secondary = last.** Downstream/target performance **never**
selects the checkpoint. No target label in subset / checkpoint / PCA / head / rank / probe. Target labels **final
scoring only**.

## Downstream evaluation
Per checkpoint × dataset (SHU-MI primary; PhysioNetMI large/weak; BNCI sanity), frozen encoder → F1 spatial →
source-only PCA/head. **L1 = mean PAIRWISE subject separability** (dimension-invariant, 2-way, run/session-held-out,
HELD-OUT subjects, fixed probe + fixed PCA rank across cells, 3-way subject-disjoint split, dynamic-range check off
the 0.5 floor / ~0.95 ceiling — run per N; confirm the per-patch z-score has not floored L1). Per-cell task gate
(source-val ≥0.58) → L4/L5/L6 interpretable. **Downstream normalization parity (MN-10):** the loader z-scores
**per 1-s PATCH (200 samples) per channel** — the SHU-MI probe loader must apply the **identical per-patch
granularity** (NOT per-30 s-window) + the same `EEGNormalizer` neutralization used in pretraining (else the frozen
encoder is probed OOD).

## Population disclosure (`p1_population_diagnostics.csv`) — the confound spans the WHOLE frontier (BL-10)
Every cell reports eligible-pool size, sampled available-windows median/p90, budget windows/subject, redundancy
ratio, and % single-recording. Population drifts **monotonically at every step** (%single-recording 0.8→12.5→32→66→69
across N=128…2048), collinear with log N — **not** just at N=128. This is intrinsic to a fixed-budget frontier in
TUEG (only clinical patients have deep recordings), is **disclosed and carried as a per-cell covariate next to every
slope**, and is **not** neutralized by excl-N=128. Safe wording: *"allocating a fixed budget across more vs fewer
subjects, bundling a clinical→general population gradient"* — never *"population-adjusted"*, *"coverage-vs-depth"*, or
*"subject-diversity"* effect.

**Validation-population mismatch (MJ-14).** The fixed global val pool is **shallow/general** subjects, so best-val-loss
checkpoint selection rewards general-population reconstruction, and train↔val population alignment **improves with N**
(worst at N=128 clinical-train/general-val, best at N=2048). Report per-cell train/val population alignment and
**last-checkpoint** robustness alongside best-val; val loss is comparable in **data** but not in **population**.

## Interpretation grid (pre-registered) — read SHAPE (peak), not just slope sign (MJ-12)
```
target-bAcc ↑ with N (fixed budget), L1 ↓ along frontier        -> more coverage (shallower) improves transfer + lowers separability (BUNDLED w/ population)
target-bAcc ↑ with N, L1 high, L5 null/decreases                -> more coverage changes the ROLE of subject info without erasing it (strongest; still bundled)
target-bAcc ↓ with N                                            -> shallow per-subject exposure harms representation at this budget
interior PEAK (curvature<0, peak at intermediate N)             -> optimal fixed-budget allocation is ~N* subjects (the deployment answer; a single slope would misread this as flat)
target-bAcc flat AND no curvature                              -> "at 200h, allocation along this frontier gives no detectable transfer change" (NOT "diversity doesn't matter")
sign flips across seeds / leave-one-N-out / driven by N=128     -> unstable / population-driven; P2 not justified
gate-pass N-dependent                                           -> convergence-mediated frontier, not an allocation effect
```
Fit a **quadratic in log N** and report **per-N means + peak-location CI + monotonicity test** regardless of the
linear slope, so an interior optimum is not misclassified as "flat".
**Forbidden:** "more subjects independently improve transfer"; "subject diversity causes better generalization";
"per-subject exposure is controlled"; "pure diversity effect estimated"; growing-hours read as diversity;
reproduction of published CBraMod; SOTA/full-FT. **Every claim carries the fixed-budget / allocation / population
qualifier.**

## Outputs (`results/s2p_p1_cbramod/`)
`p1_run_manifest.csv`, `p1_frontier_subset_manifest.csv` (pool + subset/init seeds + per-subject windows),
`p1_actual_budget_by_cell.csv`, `p1_sampling_balance_checks.csv`, `p1_pretrain_logs.csv`, `p1_positive_control.csv`,
`p1_checkpoint_manifest.csv`, `p1_downstream_task_performance.csv`, `p1_pairwise_subject_separability.csv`,
`p1_l4_task_alignment.csv`, `p1_l5_replay.csv`, `p1_l6_target_consequence.csv`,
`p1_frontier_summary.csv` (full + robust allocation slopes + leave-one-N-out + secondary frontiers + stability),
`p1_target_label_firewall.json`, `p1_verdict.json` (`allocation_slope`, `robust_slope_excl_N128`,
`representation_level_effect_seen`, `stable_across_seeds_and_leave_one_N`, `population_confound_disclosed`,
`recommend_p2`, `target_labels_used_for_selection:false`).

## STOP rules
```text
1  any objective/mask/architecture reimplementation (must be native Trainer_valid).
2  target labels in subset / checkpoint / PCA / head / rank / probe selection.
3  sampling-balance gate fail (±1% of 200h / window max−min ≤1 / Gini ≤0.02 / train∩val=∅) -> that cell not launched.
4  pretrain-val not the FIXED GLOBAL subject-disjoint pool.
5  any pure-subject-diversity or controlled-exposure claim (both forbidden — the triangle).
6  max-data cell fails positive-control floor -> under-powered -> STOP (no null claim).
7  slope driven solely by N=128 / seed-unstable / leave-one-N-out flips -> do not interpret as allocation effect.
8  CodeBrain enters the P1 science grid (P2 only).
```

## P2 conditionality
P2 (second budget T to test frontier-shape stability / CodeBrain native / growing-budget) **only if** P1 shows a
clear, stable allocation slope (transfer, L1, or L5). Null/unstable → no P2, no CodeBrain full grid, no 33-ch corpus.
Return for PM review before P2.
