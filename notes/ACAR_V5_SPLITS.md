# ACAR V5 — Substrate, Splits, Robustness & Held-out **(DRAFT — UNTAGGED — NON-BINDING; pinned at sign-off)**

Companion to `ACAR_FROZEN_v5.md`, `ACAR_V5_CANDIDATE_SPACE.md`, `ACAR_V5_ENDPOINTS.md`. **No runs / no substrate training / no
external read authorized by this draft** (hard no-execution clause).

## 1. Substrate (FROZEN FIRST — the v5 ordering)
- **Pipeline (pinned, == the v4 FROZEN_PIPELINE):** 19-ch 10–20 montage · 128 Hz resample · 0.5–45 Hz band-pass · 4 s windows
  (512 samples) · per-trial z-score · EEGNet backbone · 16-dim embedding · 2 classes. External sites are mapped onto THIS pipeline
  by the held-out reader (channel/Fs harmonization), so the substrate geometry is identical DEV↔external.
- **"All-source" = all listed source COHORTS, not all subjects (PINNED — Step 2d).** The V5 substrate is a per-disease encoder
  over the DEV source cohorts below (NOT per-fold LOSO on the OLD dumps) — but for **Stage-2 DEV selection it is FOLD-CONTAINED**
  (Interpretation B): for each disease and each outer fold, the DEV-selection substrate (encoder + source-state + FIT
  standardization + threshold quantiles) is fit **ONLY on that fold's FIT (TRAIN+VAL) subjects**. **CAL is calibration/LTT only;
  EVAL is G1–G5 evaluation only. No CAL/EVAL label or record may enter encoder training, source-state fitting, early stopping,
  standardization, or threshold construction** (subject-disjoint end to end — closes the "encoder saw EVAL subjects" leak).
- **Final external-execution substrate (Stage-5).** A single per-disease all-source-cohort encoder/source-state may be trained
  ONLY AFTER the candidate identity is fixed (post Stage-4) and before Stage-5 external authorization; it is used for external
  execution only and **may NOT be used for candidate selection or reselection**.
- **Substrate-frozen-before-selection (HARD ordering):** each fold's Stage-2 substrate is frozen + hashed before its embeddings are
  read; Stage-2 selection may only read embeddings carrying the Stage-0 registry hashes. Enforced by `test_substrate_hash_required`.

## 2. Artifact hygiene — Stage-0 registry (NO hash ⇒ inadmissible)
Every V5 embedding/artifact must record, and the registry must verify:
```
encoder_state_dict_sha256 · encoder_checkpoint_file_sha256
source_state_artifact_sha256 · source_state_file_sha256
preprocessing_config_sha256 (pipeline above) · channel_montage · sampling_rate · windowing_config
cohort_inclusion_list (+ excluded subjects) · random_seed · git_commit (40-hex) · env_lock_sha256 · feat_dump_sha256
```
(This is the direct fix for v4's encoder/source-state archival blocker — V5 cannot reach selection without these.)

## 3. DEV cohorts (source; pinned)
```
PD  source DEV : ds002778, ds003490, ds004584          (EXACT_ELIGIBLE to be re-counted at Stage-1; v4 had PD 230)
SCZ source DEV : ds003944, ds003947, ds004000, ds004367 (v4 had SCZ 225, ds004000/sub-042 excluded)
```
(Re-counts + exclusions are re-derived at Stage-1 and pinned in the substrate registry, not copied from v4.)

## 4. Held-out / external arm (pinned; single-site per disease; Stage-5 ONLY)
```
SCZ external : Zenodo 14808296   (admissible: ~38+39 subj, 64ch/1000Hz, resting, CC-BY, raw) — PRIMARY
PD  external : OpenNeuro ds007526 (admissible: 144 = 116 PD + 28 HC, 65ch/250Hz, resting, CC0)  — PRIMARY
ASZED / Zenodo 14178398 (SCZ 2nd site): PROVISIONAL / NOT ADMITTED — eligible only by a separately dated amendment AFTER a
                         content-blind data-integrity review (2 Nigerian units/devices, 16ch, 200/256Hz; integrity preprint
                         unverified). NOT part of the primary V5 external arm.
ds007020 (PD): EXCLUDED (no usable HC-vs-PD label; overlap/label issues).
```
Each disease's external arm is **single-site**; V5 results are reported as "single-site held-out confirmation", never a cross-site
generalization claim. The external arm runs ONCE, after Stage-4 passes, under an explicit external authorization (the lockbox
stays SEALED until then). `test_no_external_before_tag` enforces no external read before the `acar-v5-protocol` tag + sign-off.

## 5. DEV split (split-as-ONE-algorithm; subject-disjoint; disease-stratified) — RATIOS/SEEDS PINNED (Step 2c)
A single deterministic, permutation-independent canonical-SubjectKey hashing (as in v2/v3). The router/predictor sees FIT, the
calibration (λ/threshold + LTT bounds) sees CAL, and G1–G5 are measured on outer EVAL. Subject is the cluster everywhere;
FIT/CAL/EVAL are subject-disjoint. Enforced by `test_subject_disjoint`. **Pinned scheme:**
```
outer K = 5 folds, assigned by stable canonical SubjectKey hash; each subject is EVAL exactly once
within each outer fold, over the non-EVAL subjects:   FIT = 70% · CAL = 30%
within FIT:                                           TRAIN = 80% · VAL = 20%   (encoder/router early stopping where applicable)
all splits: subject-disjoint · disease-stratified · deterministic by hash (NOT RNG permutation)
canonical base split salt = "ACAR_V5_SPLIT_V1"        (no other salt may be substituted after tag)
canonical Stage-2 DEV-selection encoder seed = 20260711   (seeds 20260712/20260713 are S1 robustness ONLY — never selection)
```
(Consistent with v2's subject-level K=5, non-EVAL FIT 70% / CAL 30%.)

**Disease handling (PINNED — Step 2d).** PD and SCZ are **trained/instantiated separately** (2 substrates, 2 source-states, 2 FIT
standardizations, per-disease quantile values). Candidate **identity is selected JOINTLY across diseases**: exactly ONE
`candidate_id` is chosen by the Stage-2 min-disease constrained objective (`ACAR_V5_ENDPOINTS.md` §Selection); the SAME
`candidate_id`, rule form, and tie-break are used for BOTH diseases; only the numeric FIT-only quantile thresholds are recomputed
per disease. (i.e. "trained/instantiated separately, selected jointly" — PD and SCZ may NOT choose different candidate_ids.)

## 6. Substrate-robustness stress tests S1–S3 (= G6; BUILT-IN, run BEFORE external)
G6 = the three modules S1–S3; a candidate must pass **EVERY module** with G1–G5. The **selected FIXED candidate** (identity,
family, operating-point rule, tie-break) is used UNCHANGED across all modules — **NO reselection across seeds/cohorts/modules**
(enforced by `test_fixed_candidate_no_reselection`). Stress-test results may NOT be used to construct or alter a policy (e.g. the
P4 agreement rule; see `ACAR_V5_CANDIDATE_SPACE.md` §1).
- **S1 — seed robustness.** Same disease + cohort set, train **3 all-DEV encoders with the PINNED seed set (Step 2c):
  `{20260711, 20260712, 20260713}`** (no other seed may be substituted after tag). **S1 module pass = the selected fixed candidate
  passes G1–G5 on ≥ 2 of 3** of these seed substrates (3/3 is reported as strong robustness but is not required). Catches policies
  that depend on one substrate's random geometry (the v4 mode).
- **S2 — leave-one-source-cohort pseudo-external.** For each DEV source cohort, train the encoder on the OTHER source cohorts and
  evaluate the held-out cohort as a pseudo-external site. The policy must hold across cohort compositions, not just one.
  **FIT-only rule (PINNED — Step 2b):** the selected policy family + operating-point rule are FIXED; the source-side FIT-only
  standardization / unlabeled score-quantile thresholds may be recomputed **by the frozen algorithm on the training-side cohorts
  only**; the held-out source cohort's LABELS are used ONLY for the final gate evaluation. **No threshold, family, score, or action
  may be chosen using held-out source labels** (substrate-local unlabeled normalization is allowed; label-driven retuning is not).
- **S3 — representation-family robustness (STRICT hard module; frozen spectral-z baseline, PINNED — Step 2d).** A single
  pre-registered baseline substrate — NO "e.g."/free choice:
  ```
  spectral-z baseline substrate:
    same 19-ch / 128 Hz / 0.5–45 Hz / 4 s-window (512) pipeline;
    per-window log-bandpower over fixed bands  delta 0.5–4 · theta 4–8 · alpha 8–13 · beta 13–30 · low-gamma 30–45;
    subject-balanced FIT-only standardization;
    PCA → 16 dims fit on FIT only;
    source classifier = logistic regression fit on TRAIN, tuned/early-stopped on VAL only;
    same z-space action set: identity · matched_coral · spdim · t3a.
  ```
  **S3 pass criterion:** the FIXED selected `candidate_id` is executable UNCHANGED under this substrate and **G1–G5 pass** under
  the same subject-level evaluation + Holm rules. Utility magnitude need not equal EEGNet but must still **clear the G2 margin
  (≥ 0.02)**. Any signed-score direction reversal that flips the adapt/abstain decision relative to the candidate rule is recorded;
  if it causes a G1–G5 failure, S3 fails. (Directly targets the v4 `d_margin` flip.)

Fail any S1–S3 module ⇒ the candidate is ineligible (STOP; no external). This is the gate v4 lacked as a precondition.

## 7. Proposed acar/v5/ layout (NOT scaffolded yet — for Stage-0 only)
```
acar/v5/
  protocol/  (this draft + the 3 companions, promoted to ACAR_FROZEN_v5 on tag)
  substrate/ train_all_source.py · dump_embeddings.py · registry.py · verify_artifacts.py
  features/  paired_features.py · standardize.py · action_records.py
  policies/  policy_space.py · benefit_harm_veto.py · action_agreement.py · best_fixed_abstain.py
  evaluation/ subject_metrics.py · ltt_constraints.py · bootstrap_ci.py · v2_replay.py · robustness.py
  runs/      run_v5_dev_select.py · run_v5_compat.py · run_v5_external.py
  tests/     test_no_label_in_route · test_subject_disjoint · test_no_external_before_tag ·
             test_low_coverage_degeneracy · test_substrate_hash_required · test_fixed_candidate_no_reselection
```
`test_low_coverage_degeneracy` is the v4-derived regression test: a policy with very low coverage but high adapted-harm must be
REJECTED by G4 / utility even if it passes G3 (`L_harm_all`). Scaffolding `acar/v5/` is a SEPARATE gated step (not authorized here).
