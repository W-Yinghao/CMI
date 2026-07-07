# FSR_48 — CodeBrain + CBraMod 8C Subject-Scaling Protocol (Phase 8C; pre-registration, v2 red-teamed)

**Project FSR — Phase 8C.** Pre-registration of the **source-subject-count scaling** audit of frozen EEG foundation
encoders. **v2 incorporates a design red-team (agent ae53b01d) that found 4 BLOCKERs + 5 MAJORs in v1**; all fixed
below. Frozen encoders, no fine-tuning, target labels **final scoring only**. Execute **only after** 8B closure
(done: invariance PASS + temporal collapsed + BNCI sanity) and this v2 is frozen.

## Primary scientific question
> Does increasing **source-subject diversity** change target performance by **reducing subject decodability**, or by
> **changing the functional role** of subject information (encoding without harmful reliance)? — separated from a
> pure **sample-size** effect.

## Dataset — PhysioNetMI (EEGMMIDB), PINNED (fixes BLOCKER-4, BLOCKER-3)
- **Task/classes (pinned):** imagined **left-fist vs right-fist**, runs **4, 8, 12** (T1=left, T2=right). **K=2.**
  Epoch **0–4 s post-cue** → 4 patches @200 Hz. (A second pinned pair, imagined fists-vs-feet runs 6/10/14, is a
  **pre-declared secondary** only.)
- **Excluded subjects (pinned):** S088, S089, S092, S100 (inconsistent sampling/run structure). Analyzable set is
  **105 subjects** → "N=all" = **104 source** (target held out). The exact analyzable subject id list is frozen in
  `physionetmi_manifest.csv`.
- **Single session (BLOCKER-3):** EEGMMIDB is one visit → the 8B **session**-held-out L1 probe is inapplicable. L1
  uses a **run-held-out** split (train events from runs 4,8; test run 12) as the closest quasi-session control;
  **disclosed as weaker than cross-session** (residual same-day drift). All L1 claims bounded accordingly.
- **Resample/channels (pinned + hashed):** 160→200 Hz (5/4, anti-aliased method pinned), **64-channel native
  ordered montage** (EEGMMIDB dotted names, pinned list). Hashes recorded in the manifest (STOP-1/9).

## 8C-0 gate — manifest + frozen plan + 64ch ENCODER-SANITY (fixes MAJOR-B)
Before any scaling: (1) `physionetmi_manifest.csv` (analyzable ids, per-subject trial counts, class balance, channel
map+hash, resample hash) + `source_subset_plan.csv` (every (N,seed,condition) subset + trial budget, frozen). (2)
**64ch encoder-sanity gate** — do NOT inherit 8B's 19–32ch QC: re-verify on PhysioNetMI at 64ch that (a) determinism
+ batch-invariance hold (SGConv FFT magnitude may differ at 64ch), (b) embeddings finite/non-degenerate, (c) ≥1
encoder is **above-chance** on the pinned task with a source-only head. Fail → STOP, do not run the grid.

## Feature / head / PCA — FIXED a priori across N (fixes MAJOR-C)
- **F1 spatial** (per-channel: mean over patches, keep channels → 64×200). **PCA output dim `d` FIXED a priori**
  (one `d`, chosen once on the N=all source reference, e.g. d=128) so **all N share one feature space**; where n<d at
  small N the space is under-determined → **disclosed**, not floated. **Head = linear probe with FIXED
  regularization** (no per-(N,seed) selection). Any unavoidable selection uses **trial-level nested CV** (constant in
  kind across N), never subject-level val (whose composition changes with N). Firewall: no target label in
  subset/PCA/head/rank/early-stop; z-score per-trial within-window.

## Design — N grid × conditions × seeds × targets (fixes MAJOR-A, MAJOR-D)
- `N_source ∈ {2, 4, 8, 16, all}`; **fixed panel of held-out target subjects** (pinned, e.g. 15 subjects) evaluated
  identically at **every** grid cell (no target-difficulty confound).
- **growing-trials** (per-subject cap fixed; total grows) and **fixed-trials** (total fixed; per-subject shrinks).
  **Fixed-trials grid is FEASIBILITY-TRUNCATED:** with ~45 trials/subject, the fixed total is bounded so per-subject-n
  stays ≥ a floor (also ≥ the run-held-out L1 minimum); **fixed-trials cannot reach N=all → disclosed**, the grid
  stops where per-subject-n < floor. A **matched-subsample arm** (growing-trials down-sampled to fixed-trials
  per-subject-n at matched total) isolates the pure per-subject-n effect. **mean per-subject-n is a covariate** in
  every model.
- **Seeds:** 10 subset seeds with a **coverage-balanced** sampler at small N (each source subject appears comparably).
  **N=all is a single composition** (all-but-target) → credited **1 df, not 10** (no seed pseudo-replication).
  Overlap across seeds at N=8,16 disclosed (not independent draws).

## Metrics — dimension-invariant / rank-honest (fixes BLOCKER-1, BLOCKER-2)
- **Performance:** target balanced accuracy, worst-subject bAcc, macro-F1 (chance printed).
- **L1 subject decodability — DIMENSION-INVARIANT (BLOCKER-1).** The v1 `(acc−chance)/(1−chance)` N-way excess is
  **artifactual** (mechanically decreases with N for a constant representation) → **REPLACED**. Primary L1 =
  **mean pairwise subject separability**: average over all source-subject **pairs** of the **2-way** run-held-out
  discriminability (bAcc/AUC), which is always a 2-class problem regardless of N (firewall-clean, source ids only).
  Secondary = **intensive geometry** (between/within-subject variance ratio, mean pairwise Mahalanobis, silhouette).
  Raw N-way probe accuracy is **descriptive-only**, never the scaling metric. Equal per-subject probe sampling. Both
  marginal and class-conditional variants use the pairwise fix.
- **L4** task-head↔subject-subspace alignment.
- **L5/L6 — RANK-HONEST (BLOCKER-2).** Subject subspace rank ceiling = **K·(N_source−1)** (K=2 → N=2:2, N=4:6,
  N=8:14…). **Primary rank = energy/removed-variance-MATCHED** (nominal fixed-k is NOT comparable when the ceiling
  varies); set `k = min(8, K·(N−1))` and **report the ceiling per level**. **All L5/L6 deltas normalized by
  removed-variance fraction**, vs an equal-**removed-variance** null. The subspace **scaling** comparison is
  restricted to N where k is feasible (**N≥8 for K=2**); N=2,4 are descriptive-only (rank-starved; null degenerate,
  disclosed). Primary L5 claim = subject intervention **exceeds the variance-matched null**, per removed-variance.

## Analysis + statistics (fixes MAJOR-E)
- **Confirmatory family (small, pre-declared):** (i) CodeBrain **mean-pairwise-L1** growing-slope on log(N); (ii) the
  **growing-vs-fixed slope difference** for L5-vs-variance-null (per removed-variance). **Holm-corrected**; α=0.05.
  Everything else **exploratory**.
- **Quantitative interpretation grid** (no qualitative sign-reading): "improves/decreases" = slope CI excludes 0
  **and** |slope| ≥ a pre-set magnitude per doubling of N; "persists" = CI includes 0 **and** |slope| < that
  magnitude. Report **per-level estimates + CIs** and a pre-specified **N=2-vs-N=all contrast**, not only a 5-point
  regression (leveraged by the far, seed-collapsed anchor).
- **Mixed-effects** (`subject_scaling_mixed_effects.json`): slope on log(N) with **crossed random effects (target
  subject + subset seed)** and **mean per-subject-n as a fixed covariate**; N=all weighted as one composition.
- Interpretation grid (pre-registered, per removed-variance / dimension-invariant metrics):
  ```
  only growing improves                       -> sample-size effect
  fixed also improves (feasible N)            -> subject-diversity effect
  perf improves but pairwise-L1 persists      -> subject info persists but may not be harmful
  pairwise-L1 decreases with N                -> diversity reduces subject-distinguishability (NOT the v1 artifact)
  L5/L6(per-removed-var) decrease, L1 persists -> functional role changes (strongest FSR result)
  all slopes ~0                                -> no scaling of subject role detected in THIS N range / paradigm /
                                                  at this power (NEVER "subject-invariant")
  ```

## Staging
`8C-0` manifest + frozen plan + 64ch encoder-sanity gate → must pass. `8C-1` CodeBrain + CBraMod scaling (no
baselines). **Return for PM review before 8C-2** (specialist baselines), which run only if 8C-1 shows signal.

## Outputs (`results/fsr_codebrain_cbramod_8c/`)
```
physionetmi_manifest.csv          source_subset_plan.csv          encoder_sanity_64ch.csv
subject_scaling_performance.csv   subject_scaling_l1_pairwise.csv (mean-pairwise + geometry; raw N-way descriptive)
subject_scaling_l4_alignment.csv  subject_scaling_l5_replay.csv (per removed-variance, rank ceiling per level)
subject_scaling_l6_consequence.csv fixed_vs_growing_trials.csv (+ matched-subsample arm, per-subject-n)
subject_scaling_mixed_effects.json target_label_firewall.json
codebrain_cbramod_8c_verdict.json
```
`codebrain_cbramod_8c_verdict.json`: as v1 + `{"analyzable_subjects":105,"K":2,"l1_metric":"mean_pairwise_separability",
"rank_rule":"energy_matched_min8_KxNminus1","fixed_trials_max_feasible_N":null,"encoder_sanity_64ch_pass":null,
"confirmatory_family_holm":null,"per_subject_n_covariate":true}`.

## STOP rules (PM v1 + red-team additions)
```text
1  PhysioNetMI channel mapping / resample not reproducible (hash).            [v1-1/9]
2  64ch encoder-sanity gate fails (determinism/non-degeneracy/at-chance).     [MAJOR-B]
3  fixed-trials cannot satisfy the per-subject-n floor even at small N.       [MAJOR-A]
4  target labels enter subset / PCA / head / rank / early-stop.               [v1-4]
5  head/PCA cannot be held FIXED across N (comparability broken).             [MAJOR-C]
6  both models at/near chance on the pinned task for all N -> no baselines; return. [v1-6]
7  pairwise-L1 or geometry cannot be computed comparably across N.            [BLOCKER-1]
8  subject-subspace rank ceiling K(N-1) makes the feasible-N scaling set empty.[BLOCKER-2]
```

## Claims / framing
C22 (scaling changes functional role) = **PENDING_8C**, claimable only from this run under the v2 metrics. A null is
"no scaling detected in this range/paradigm/power," **never** "subject-invariant." Forbidden: SOTA/leaderboard;
reading the v1 L1 artifact as a finding; mislabeling a growing-only effect as diversity. PC2 paused; Paper 1
unaffected; **Paper 2 frozen** — a clean 8C signal is a candidate for a follow-up, decided later.
