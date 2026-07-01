# ACAR V5 — Substrate, Splits, Robustness & Held-out **(DRAFT — UNTAGGED — NON-BINDING; pinned at sign-off)**

Companion to `ACAR_FROZEN_v5.md`, `ACAR_V5_CANDIDATE_SPACE.md`, `ACAR_V5_ENDPOINTS.md`. **No runs / no substrate training / no
external read authorized by this draft** (hard no-execution clause).

## 1. Substrate (FROZEN FIRST — the v5 ordering)
- **Pipeline (pinned, == the v4 FROZEN_PIPELINE):** 19-ch 10–20 montage · 128 Hz resample · 0.5–45 Hz band-pass · 4 s windows
  (512 samples) · per-trial z-score · EEGNet backbone · 16-dim embedding · 2 classes. External sites are mapped onto THIS pipeline
  by the held-out reader (channel/Fs harmonization), so the substrate geometry is identical DEV↔external.
- **Per-disease all-source DEV encoder + source-state**, trained on the DEV cohorts below (NOT per-fold LOSO). This is the
  external-compatible substrate; DEV selection embeddings come from it (Stage-1), never from old LOSO dumps.
- **Substrate-frozen-before-selection (HARD ordering):** Stage-1 freezes + hashes the substrate; Stage-2 selection may only read
  embeddings that carry the Stage-0 registry hashes. Enforced by `test_substrate_hash_required`.

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

## 5. DEV split (split-as-ONE-algorithm; subject-disjoint; disease-stratified)
As in v3: a single deterministic, permutation-independent canonical-SubjectKey hashing. Outer K folds give EVAL; non-EVAL splits
into FIT / CAL; FIT splits into TRAIN / VAL (early stop). The router/predictor sees FIT, the calibration (λ/threshold + LTT bounds)
sees CAL, and G1–G5 are measured on outer EVAL. Subject is the cluster everywhere; FIT/CAL/EVAL are subject-disjoint; diseases are
trained/selected separately (2 substrates, 2 routers). Seeds/ratios pinned at sign-off. Enforced by `test_subject_disjoint`.

## 6. Substrate-robustness stress tests S1–S3 (= G6; BUILT-IN, run BEFORE external)
G6 = the three modules S1–S3; a candidate must pass **EVERY module** with G1–G5. The **selected FIXED candidate** (identity,
family, operating-point rule, tie-break) is used UNCHANGED across all modules — **NO reselection across seeds/cohorts/modules**
(enforced by `test_fixed_candidate_no_reselection`). Stress-test results may NOT be used to construct or alter a policy (e.g. the
P4 agreement rule; see `ACAR_V5_CANDIDATE_SPACE.md` §1).
- **S1 — seed robustness.** Same disease + cohort set, train **3 all-DEV encoders with different seeds**. **S1 module pass = the
  selected fixed candidate passes G1–G5 on ≥ 2 of 3** pre-registered seed substrates (3/3 is reported as strong robustness but is
  not required). Catches policies that depend on one substrate's random geometry (the v4 mode).
- **S2 — leave-one-source-cohort pseudo-external.** For each DEV source cohort, train the encoder on the OTHER source cohorts and
  evaluate the held-out cohort as a pseudo-external site. The policy must hold across cohort compositions, not just one.
  **FIT-only rule (PINNED — Step 2b):** the selected policy family + operating-point rule are FIXED; the source-side FIT-only
  standardization / unlabeled score-quantile thresholds may be recomputed **by the frozen algorithm on the training-side cohorts
  only**; the held-out source cohort's LABELS are used ONLY for the final gate evaluation. **No threshold, family, score, or action
  may be chosen using held-out source labels** (substrate-local unlabeled normalization is allowed; label-driven retuning is not).
- **S3 — representation-family robustness.** Beyond the EEGNet substrate, evaluate a light baseline substrate (e.g.
  log-cov / Riemannian tangent or frozen spectral features). Utility need not match, but the **action-gate direction must not
  invert** (a signed score that flips sign across representation families is disqualified — directly targets the v4 `d_margin`
  flip).

Fail any S1–S3 ⇒ the candidate is ineligible (STOP; no external). This is the gate v4 lacked as a precondition.

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
