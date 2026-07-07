# FSR_48 — CodeBrain + CBraMod 8C Subject-Scaling Protocol (Phase 8C; pre-registration)

**Project FSR — Phase 8C.** Pre-registration of the **source-subject-count scaling** audit of frozen EEG
foundation encoders. PM-approved for protocol + design-red-team **now**; execution **only after** the 8B closure
(BNCI2014_001 sanity + temporal-token side-check + F1 batch-invariance append) with **no stop rule**. Frozen
encoders, no fine-tuning, target labels **final scoring only**. To be design-red-teamed before the run.

## Primary scientific question
> Does increasing **source-subject diversity** change target performance by **reducing subject decodability**, or by
> **changing the functional role** of subject information (encoding without harmful reliance)?

**NOT** "do foundation models beat baselines." Specialist baselines are deferred (8C-2), only if 8C-1 shows signal.

## Dataset
**PhysioNetMI (EEGMMIDB), 109 subjects** — the only ≥30-subject MI set on disk, suitable for a source-subject-count
curve. `physionetmi_manifest.csv` records per-subject trial counts, class balance, channel map (native order,
resample→200 Hz), segment length; reproducible channel mapping is a gate (STOP-1/9). MI runs/classes fixed a priori
(e.g. left/right fist), documented; task labels never used in any fit/selection.

## Models / feature
- **CodeBrain (primary foundation substrate)** — EEGSSM, **F1 spatial** feature (per-channel: mean over patches,
  keep channels → C×200, PCA-95%/cap-128 on source-train). GPU. Claim-invariant across batch size (8B).
- **CBraMod (clean deterministic control)** — F1 spatial. CPU-runnable; single-stage, no tokenizer confound.
- Frozen; `proj_out=Identity`; input 200 Hz, 1 s patches. Discrete tokens **not** used (temporal collapsed; C23).

## Design — N_source grid × two trial conditions (the crux)
`N_source ∈ {2, 4, 8, 16, all}`, **`subset_seeds = 10`** (source-subject subsets drawn by seed; target subject
always disjoint, LOSO-style over held-out targets). **Both conditions mandatory:**
- **A. growing-trials** — each source subject contributes a fixed per-subject trial cap; more subjects ⇒ more total
  trials (diversity **and** sample size grow together).
- **B. fixed-trials** — total training trials held constant across `N_source`; more subjects ⇒ fewer trials each
  (diversity grows, sample size fixed).
Without A-vs-B one cannot separate **subject-diversity** from **sample-size** effects (`fixed_vs_growing_trials.csv`).
`source_subset_plan.csv` pins every (N_source, seed, condition) subset + trial budget **before** the run.

## Head / selection / firewall
Frozen encoder → F1 → **PCA on source-train only** → **task head selected on source-val only** → target scored.
**No target labels** for: source-subset selection, PCA, head selection, probe selection, rank selection, early
stopping. z-score per-trial within-window. `target_label_firewall.json` logs every target-label read (final
scoring only). At `N_source=2`, source-val may be too small to select a head reliably → STOP-5 (report, don't force).

## Metrics — normalized so N_source levels are comparable
- **Performance:** target balanced accuracy, worst-subject bAcc, macro-F1 (chance printed).
- **L1 subject leakage (chance = 1/N_source varies!):** report `raw_subject_probe_acc`, `chance = 1/N_source`, and
  **`normalized_subject_excess = (acc − chance)/(1 − chance)`** (marginal AND class-conditional). Raw accuracy is
  **not** comparable across N_source; only the normalized excess is. Within-subject session-held-out probe (8B).
- **L4 task-head↔subject-subspace alignment.**
- **L5 subject-subspace reliance:** held-out-source bAcc drop after erase, **vs a variance-matched null**. The
  **primary L5 claim is "subject intervention EXCEEDS the variance null,"** not the raw subject erase delta.
- **L6 target consequence:** target bAcc before/after erase (final scoring only), with the conservative-null caveat.

## Rank rule (pre-registered; subspace rank must not confound N_source)
**Primary rank `k = 8`** (fixed across all N_source). **Secondary:** `k = 16` and a **source-val energy-matched**
rank. If embedding dim / sample size does not support k=16 at small N_source, keep **k=8 primary** and disclose.
All L5/L6 erases paired with an equal-rank **variance-matched** null.

## Analysis — four pre-registered panels
- **Panel A — performance scaling:** `target_bAcc ~ log(N_source)`, separately for growing / fixed × CodeBrain /
  CBraMod.
- **Panel B — subject-decodability scaling:** `normalized_subject_excess ~ log(N_source)` (does subject identity
  fall, rise, or persist?).
- **Panel C — functional-reliance scaling:** L4 alignment, L5 (subject vs variance null), L6 vs `log(N_source)`.
- **Panel D — diversity vs sample-size decomposition:** compare **fixed-trials slope** vs **growing-trials slope**.
  Interpretation grid (pre-registered):
  ```
  only growing improves            -> sample-size effect
  fixed also improves              -> subject-diversity effect
  perf improves but L1 persists    -> subject info persists but may not be harmful
  L1 excess decreases with N_source-> diversity suppresses subject-identifiable structure
  L5/L6 decrease while L1 persists  -> functional role changes (the strongest FSR result)
  ```
Mixed-effects (`subject_scaling_mixed_effects.json`): slope of each metric on log(N_source), random effects over
target subject + subset seed, per condition/model.

## Staging
`8C-0` PhysioNetMI manifest + `source_subset_plan` (integrity, channel map, class balance, fixed-trials feasibility)
→ gate. `8C-1` CodeBrain + CBraMod scaling (no baselines). **Return for PM review before 8C-2** (specialist
baselines EEGNet / FBCSP-LGG / CSP-LDA), which run **only if** 8C-1 shows a scaling signal.

## Outputs (`results/fsr_codebrain_cbramod_8c/`)
```
physionetmi_manifest.csv          source_subset_plan.csv
subject_scaling_performance.csv   subject_scaling_l1_leakage.csv (raw + chance + normalized_excess)
subject_scaling_l4_alignment.csv  subject_scaling_l5_replay.csv (subject vs variance null)
subject_scaling_l6_consequence.csv fixed_vs_growing_trials.csv
subject_scaling_mixed_effects.json target_label_firewall.json
codebrain_cbramod_8c_verdict.json
```
`codebrain_cbramod_8c_verdict.json`:
```json
{"dataset":"PhysioNetMI","models":["CodeBrain","CBraMod"],"n_source_grid":[2,4,8,16,"all"],"subset_seeds":10,
 "fixed_trials_condition_valid":null,"target_labels_used_for_selection":false,
 "codebrain_scaling_signal":"none|sample_size|subject_diversity|mixed","cbramod_scaling_signal":"none|sample_size|subject_diversity|mixed",
 "l1_leakage_trend":null,"l5_reliance_trend":null,"proceed_to_baselines":null}
```

## STOP rules (PM)
```text
1  PhysioNetMI channel mapping not reproducible.
2  fixed-trials condition cannot be balanced across N_source.
3  either encoder's embeddings unstable.
4  target labels enter subset / head / probe / rank selection.
5  N_source=2 source-val too small to select heads reliably.
6  both models near chance for all N_source -> do NOT add baselines; return for review.
7  L1 subject probe not comparable across N_source (normalization fails).
8  L5 subject-subspace rank confounds N_source despite the fixed-k rule.
```

## Framing / claims
Provisional ledger: **C22 (scaling changes functional role) = PENDING_8C** — claimable only from this run. Allowed
result forms (pre-registered): sample-size-only / subject-diversity / persists-but-not-harmful / diversity-suppresses
/ role-changes. **Forbidden:** "foundation models solve cross-subject generalization"; any SOTA/leaderboard framing;
reading a null as invariance. PC2 paused; Paper 1 unaffected; **Paper 2 frozen** — Phase 8 is a separate axis; a
clean 8C signal is a candidate for a follow-up ("Scaling Subjects, Not Erasing Subjects"), decided later.
