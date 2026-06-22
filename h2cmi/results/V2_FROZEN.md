# V2_FROZEN — real-EEG external validation pre-registration (frozen BEFORE any model runs)

V2_PRIMARY = A + B. TASK = binary left/right ONLY (BNCI2014_001 restricted to left/right).
Offline MOABB from /projects/EEG-foundation-model/datalake/raw (no download). Methods frozen
(H2CMI_METHOD_FREEZE): NO target-only eligibility gate, NO SPD/rotation deployment, NO method/
threshold retuning. Target labels = evaluation only (never in source training, operator fitting,
metadata extraction, or method selection).

## A — out-of-support abstention audit (operator-support safety, NOT "real null-safety")
Cross-dataset acquisition mismatch is a REAL shift that lies OUTSIDE the frozen diagonal operator
family. A tests whether metadata holds identity when geometry is UNSUPPORTED.
A_PRIMARY_DATASETS = {BNCI2014_001-LR, Cho2017, Lee2019_MI}; all 6 directed source->target pairs,
on ONE pre-computed frozen DENSE common-channel grid (global normalized-name intersection of all 3;
ordered list + SHA-256 to manifest BEFORE reading results; NO pair-specific outcome-driven channel
selection).
A_SEVERE_SECONDARY = BNCI2014_001-LR -> BNCI2014_004 on C3/Cz/C4 only; DESCRIPTIVE, not in A
aggregate/verdict (3ch info-loss would dominate mechanism).

## B — supported-regime utility test (does adapting help when family IS supported)
Same acquisition configuration, cross-session drift -> DIAG_COMPATIBLE. Subject-specific cross-
session (source = subject's earlier session labels; target = same subject's later session). NOT LOSO.
B_DATASETS:
  BNCI2014_001-LR: session 1 -> 2
  Lee2019_MI:      session 1 -> 2   (offline-training runs only; online test runs lack usable labels)
  BNCI2014_004:    1->2, 3->4, 3->5   (NOT 2->3: that crosses screening<->feedback regime)
Cho2017 NOT in B (single session).

## METHODS (primary policy table)
identity | always_pooled | always_canonical_CC (gen_oneshot_diag) | metadata_only | Euclidean_Alignment
EA = always-align comparator (NOT routed by metadata). Strict EA: source ref cov from source TRAIN
trials only; target ref cov from UNLABELED ADAPT split only; eval trials only have the frozen EA
transform applied (never in cov estimation).
current_joint = diagnostic-only 6th comparator IF cost negligible; NOT in V2 verdict; no separate tuning.

## METADATA mapping (strict original-acquisition; two tightenings)
Geometry (from ORIGINAL acquisition metadata, not post-preprocessing appearance):
  DIAG_COMPATIBLE = same dataset/system + same channel set&order + montage + reference + device family
                    + same task&feedback regime, different day/session.
  UNSUPPORTED = channel-set/layout | montage | reference | device-family | bipolar-vs-monopolar |
                task/feedback-change mismatch.
  UNKNOWN = required metadata missing/contradictory.
  -> Cho<->Lee, BNCI001<->Cho, BNCI001<->Lee = UNSUPPORTED; BNCI001 s1->2, Lee s1->2,
     BNCI004 {1->2,3->4,3->5} = DIAG_COMPATIBLE; BNCI004 2->3 = NOT RUN (not force-classified).
  Sampling/filter/epoch differences = preprocessing harmonization (RESOLVED if harmonized, else
  UNSUPPORTED/UNKNOWN) -- NEVER "adaptation evidence". Common-channel intersection + resample do NOT
  convert reference/device mismatch from UNSUPPORTED to DIAG_COMPATIBLE.
Prevalence (cue-schedule, NOT cohort/disease):
  SAME = documented same class set + balanced/equal class-sampling. DIFFERENT = documented
  enrichment/unequal sampling. UNKNOWN = design unavailable. MI L/R here = SAME.
  NEVER infer prevalence from dataset/cohort identity, target predictions, occupancy, or target labels.
HONEST SCOPE: V2 MI validates identity<->pooled; it does NOT adequately validate pooled<->class-
conditional (no real prevalence risk). B is overwhelmingly DIAG x SAME -> pooled.

## ADAPT/EVAL split (mutually exclusive per subject/session; saved BEFORE runs)
explicit runs: first-half runs -> adapt, second-half runs -> eval.
single long run: first 50% contiguous trials -> adapt, last 50% contiguous -> eval. NO random split.
split manifest (dataset/subject/session/run/trial-indices/role/hash) saved before any model runs.

## SOURCE training
A: per directed pair, source = ALL source-dataset subjects; ONE frozen source model per source dataset
   (reused for its 2 targets); target = each target subject, own adapt/eval split.
B: subject-specific; source model from subject's earlier session labels; target = same subject later session.

## VERDICTS — separate A and B, NO single headline mean
A (out-of-support audit), unit = target subject nested in ordered source->target dataset pair (pair top
   level): report unsupported-route adaptation count = 0/N, exact prediction-equivalence with identity
   (yes/no), binomial upper CI on N; always_pooled/always_CC/EA harm; target-subject DbAcc; worst-
   quartile DbAcc. metadata_only=identity is a RULE result, not "learned false-adapt=0".
B (supported utility), unit = subject (dataset stratified): mean paired DbAcc(metadata_only - identity)
   > 0; target-subject harm rate <= 0.20; worst-quartile DbAcc; dataset-stratified effects; transform
   magnitude. metadata_only coverage ~1 (family supported); no terminated target-only gate re-added.

## ADDITIONAL ARM (non-blocking, separate record) — real-signal prevalence stress audit
Reuse A/B source checkpoints. Fixed eval set; vary UNLABELED adaptation-pool class ratio {1:1, 3:1,
1:3} (target labels used ONLY by the offline benchmark builder to construct the pool, never given to
the adapter). Compare current_joint / always_pooled / always_canonical_CC / identity; measure whether
transform magnitude tracks adaptation-pool prevalence. NOT in metadata-routing verdict (MI cue
imbalance is not a deployable prevalence descriptor).
