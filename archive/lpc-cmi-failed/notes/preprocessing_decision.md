# Tri-CMI Preprocessing Decision Document (Unified Re-run)

Lead-methodologist synthesis of the five per-topic literature findings, reconciled against the **actual local stores and loaders** verified on this machine. The governing constraints are: (1) **2a's MI period is only 4 s** so no 6 s window is a real MI window; (2) **covariance/Riemannian methods need monopolar** channels (bipolar TCP is rank-deficient and SPD-unsafe); (3) **the leakage/diagnostic story stays on RAW signal**; (4) **normalization choice interacts with the I(Z;D|Y) leakage metric** and can silently do the regularizer's job.

## 0. Local ground truth (verified this session)

| Store | Path | Specs verified |
|---|---|---|
| MI LMDB (250Hz/6s, monopolar) | `/projects/EEG-foundation-model/eeg_pipeline_preprocessed_datasets_lmdb_250Hz_6s/` | 25 MI datasets. `BNCI2014-001` = **22 monopolar motor ch** (FZ,FC3...POZ); `BNCI2014-004` = **3 ch [C3,CZ,C4]**; `Schirrmeister2017` = **128 ch**. Stored as 6 s windows. Includes Cho2017, Dreyer2023(+A/B/C), Lee2019-MI, PhysionetMotorImagery, Stieger2021, Weibo2014, BNCI2015-001, etc. |
| Datalake `5e77943a` (200Hz) | `/projects/EEG-foundation-model/datalake/processed/5e77943a/<name>/` | `sfreq=200.0` confirmed for Lee2019_MI, Cho2017, PhysionetMI, Stieger2021, **TUAB, TUEV**. Bandpass 0.3-75 + 60 Hz notch (per `diagnosis_data.py` comment). Monopolar 10-20 (`-REF`). Per-recording `[T,C]` npy + `metadata.parquet` + `events.parquet`. |
| CBraMod-format clinical (bipolar) | `/projects/EEG-foundation-model/tuab_processed_0215/{train,val,test}`, `tuev_processed_0218/` | BIOT/CBraMod convention: 16-ch bipolar TCP, 200 Hz, 0.3-75 + 60 Hz notch, TUAB 10 s / TUEV 5 s event-locked, `/100` scaling. ~37k TUAB windows. |
| Raw resting clinical | `/projects/EEG-foundation-model/datalake/raw/ADFTD/ds004504-1.0.8`, `.../raw/mumtaz/edf` | ADFTD 19ch@500Hz `.set`; MUMTAZ `.edf` 19ch@256Hz linked-ear. |
| Emotion raw | `/projects/EEG-foundation-model/SEED`, `SEED_IV/eeg_raw_data`, `DEAP/data_preprocessed_python` | SEED/SEED-IV 62ch@200Hz `.mat`; DEAP 32ch@128Hz preprocessed `.dat`. |

**Current local loaders default to 128 Hz + per-trial z-score** (`moabb_data.py` tmin0.5/tmax3.5=3 s, resample=128; `emotion_data.py`/`diagnosis_data.py`/`processed_data.py` win=4 s, resample=128). The unified re-run **changes the spine to 250 Hz for MI and keeps 200 Hz for clinical/emotion**, re-windows to be 2a-safe, and adds the dual-band covariance path.

---

## 1. Canonical preprocessing per task (what the literature settled on)

**Motor imagery (MI).** Converged de-facto recipe (MOABB / braindecode / TSMNet / Euclidean-Alignment school): resample **250 Hz**, monopolar, short task-locked window **≤4 s** (because 2a's MI is `[2,6]` s), per-subject **Euclidean Alignment** for calibration-free transfer. Two band conventions split by method family: **8-30 Hz mu+beta** for CSP/covariance/Riemannian (sharpens SPD structure; MOABB MI default is 8-32, TSMNet 4-36), vs **4-38/4-40 Hz broadband** for deep encoders (braindecode 2a/HGD). HGD uniquely needs up to 125 Hz (high-gamma is signal).

**Emotion (SEED/SEED-IV/DEAP).** Settled: SEED = 200 Hz / 0-75 Hz / 62ch monopolar / 3-class; SEED-IV = 200 Hz / **1-75 Hz** / 62ch / 4-class; DEAP preprocessed = 128 Hz / 4-45 Hz / common-average / EOG-removed / 60 s trial + 3 s baseline, valence/arousal binarized at midpoint 5. Two input lineages: **DE band-power 62×5** (delta1-4/theta4-8/alpha8-14/beta14-31/gamma31-50, LDS-smoothed) is the classical DG feature; **raw 1 s @200 Hz patches** is the foundation-model lineage matching our encoders/covariance stack. De-facto eval = single-session LOSO with **per-session normalization fit on source only**.

**Clinical (TUAB/TUEV/ADFTD/MUMTAZ).** Two irreconcilable lineages on exactly our monopolar-vs-bipolar axis: **(L1) Gemein/braindecode monopolar** (TUAB 21-ch 10-20 average-ref, 100 Hz, clip ±800 µV, 6 s crops, full-rank 21×21 covariance — the covariance-friendly convention); **(L2) BIOT/CBraMod bipolar TCP** (16-ch, 200 Hz, 0.3-75 + 60 Hz notch, TUAB 10 s / TUEV 5 s, /100 — the foundation-model convention). ADFTD dataset-canonical = 0.5-45 Hz + A1-A2 mastoid + ASR + ICA/ICLabel, 19ch@500Hz. MUMTAZ has **no canonical recipe** — must be pinned (19ch, linked-ear native, 0.5-45, 200 Hz).

---

## 2. The conflicts and how each is resolved

**(A) Monopolar vs bipolar (decisive).** A 16-ch bipolar TCP montage is a fixed linear map `W` of the referential montage, so `cov_bipolar = W cov_mono Wᵀ`; chained pairs sharing electrodes (FP1-F7, F7-T3 …) make the 16×16 covariance ill-conditioned / near-rank-deficient, breaking SPD positive-definiteness and inflating LogCov/tangent features. **Resolution: keep MONOPOLAR everywhere for the LogCov/SPDNet/Riemannian arm and for our method's primary tables; use bipolar TCP ONLY for an explicit CBraMod/BIOT foundation-model baseline, reported separately.** The MI LMDB (monopolar) and datalake `5e77943a` (monopolar `-REF`) are already correct; the `tuab_processed_0215` bipolar dirs are baseline-only.

**(B) Window length (2a is the binding constraint).** No 6 s window fits 2a's 4 s MI period; Cho2017/PhysionetMI are only ~3 s. **Resolution: task-keyed window family, never one global number.** MI ≤ 3 s common (or per-dataset 0.5-3.5 s post-cue); the LMDB "6 s" field is a **storage container** that must be **cropped to the post-cue 3 s window** for 2a/Cho/Physionet at epoch-extraction time. Emotion 4 s; clinical 5 s (FM) / 6 s (Gemein covariance).

**(C) Resample rate (250 vs 200).** MI-DG/braindecode/TSMNet = 250 Hz (preserves gamma, matches LMDB); FM/clinical = 200 Hz (matches datalake + CBraMod/BIOT/LaBraM). **Resolution: 250 Hz spine for MI (LMDB native), 200 Hz for clinical/emotion (datalake native).** Do not mix 128/250 in one table. PhysionetMI is natively 160 Hz but its datalake copy is already 200 Hz, so no upsampling decision is needed if we consume the datalake version.

**(D) Filter band (broadband vs task-band).** "Broadband for deep nets" vs "task-band for covariance" genuinely pull opposite directions (and "How EEG preprocessing shapes decoding" / Zanola 2025 show deep nets want minimal, lightly-filtered pipelines). **Resolution: dual-band off one broadband store.** Encoders consume **4-40 Hz** (MI) / 0.3-75 (clinical/emotion); the covariance feature extractor **internally narrows to 8-30 Hz** (binary) / 4-38 Hz (4-class). One physical store, two bands — no double storage.

**(E) Mains notch (50 vs 60).** LaBraM 50 Hz (EU) vs CBraMod/BIOT/our datalake 60 Hz (US). Our `5e77943a` is 60 Hz, matching CBraMod/BIOT. SEED/SEED-IV are Chinese-mains (50 Hz); ADFTD is EU (50 Hz). Apply notch **per recording site** and cite the specific FM when reporting its baseline.

**(F) Normalization vs leakage (most important for this paper).** Per-trial z-score removes DC/scale but **not subject identity** (which lives in relative channel amplitudes, spectral shape, spatial covariance). **Per-subject/per-recording normalization drives a subject-ID probe to near chance** (Fdez 2021: ~31-33%), i.e. it performs much of the leakage reduction LPC-CMI is meant to perform — confounding I(Z;D|Y). **Resolution: (i) keep the leakage probe on RAW / minimally-normalized referential signal so baseline leakage is high and the regularizer's effect is visible; (ii) for the accuracy benchmark use leakage-neutral per-trial z-score (current default) or braindecode causal exponential-moving standardization; (iii) NEVER per-subject/per-recording normalize for headline DG tables — only as an ablation, showing the regularizer reduces residual leakage on top of it. All stats fit within-trial or source-only (the loaders' `trial_zscore` already is).**

---

## 3. What DG / benchmark / foundation papers actually do

- **TSMNet/SPDDSMBN** (NeurIPS 2022) — closest covariance-DG baseline: MOABB+MNE, 250/256 Hz, **4-36 Hz Butterworth**, ≤3 s monopolar windows, per-dataset (2a = 0.5-3.5 s, 22 ch). The template for our LogCov/SPDNet arm.
- **Euclidean Alignment** (He & Wu, TBME 2020) + **systematic-EA-with-DL** (2401.10746) + **Revisiting-EA** (J-NE 2025): per-subject `R̄ = mean trial covariance`, `X̃ = R̄^{-1/2} X`, applied **between temporal and spatial filtering**, label-free → the canonical calibration-free centering. Improves cross-subject DL ~4.3%.
- **EEG-DG** (2311.05415) — our nearest method comparator: 8-35 Hz, min-max [0,1], 4 s windows; but it runs 2a and 2b **separately** (no montage mixing) so it is a *method* baseline, **not** a cross-dataset preprocessing template.
- **MOABB benchmark** (2404.15319): paradigm-default bands (MI 8-32), native per-dataset rates/windows, CrossSubject splits, Wilcoxon/permutation + Stouffer's Z — adopt as our eval/stats standard.
- **CBraMod/BIOT/LaBraM/EEGPT/BENDR**: 200 Hz (EEGPT/BENDR 256), 0.1-0.3 highpass, 1 s patches; **bipolar TCP for TUH clinical**, monopolar for MI/emotion. Our `5e77943a` and `tuab_processed_0215` exactly match the CBraMod/BIOT clinical recipe.
- **DG survey** (2604.27033): preprocessing variation alone can rival method gains → **freeze one pipeline and report it transparently** (this document).

---

## 4. Bottom line

There is no single (window, rate, band, montage) canonical for both MI and clinical EEG. Commit to a **task-keyed unified pipeline** with a **250 Hz MI spine (monopolar LMDB)** and a **200 Hz clinical/emotion spine (monopolar datalake)**; MI window ≤3 s (2a-bound, re-cropped from the 6 s LMDB container); **dual-band** (broadband to encoders, 8-30/4-38 narrowed inside covariance features); per-subject Euclidean Alignment for calibration-free transfer; **monopolar throughout** with bipolar TCP reserved for an explicit CBraMod/BIOT baseline; and the **leakage probe on RAW signal with per-trial-only normalization** so the metric measures the encoder, not the preprocessing.


---
## PER-DATASET SPEC

Format per dataset: {filter | resample | window | reference | channels | normalization} -> LOCAL source. Citations in brackets.

=== MOTOR IMAGERY (250 Hz spine, monopolar, dual-band) ===

BNCI2014_001 (2a, 4-class): {encoder 4-40 Hz / cov 4-38 Hz | 250 Hz | 0.5-3.5 s post-cue = 3 s (NEVER 6 s; cue@2s, MI to 6s) | monopolar native, no re-ref (or single CAR) | 22 motor ch | per-trial z-score} -> LOCAL: eeg_pipeline_preprocessed_datasets_lmdb_250Hz_6s/BNCI2014-001 (22 monopolar ch verified), CROP the 6s container to the post-cue 3 s. [TSMNet 2206.01323; braindecode 2a; EEG-DG 2311.05415]

BNCI2014_004 (2b, binary, 3ch): {encoder/cov 8-30 Hz | 250 Hz | 0.5-3.5 s = 3 s | monopolar | 3 ch [C3,CZ,C4] verified | per-trial z-score} -> LOCAL: lmdb_250Hz_6s/BNCI2014-004 (3 ch verified), crop to 3 s. Flag: 3x3 covariance is weak for SPDNet. [EEG-DG; TSMNet]

Lee2019_MI (54 subj, binary): {8-30 Hz | 250 Hz (downsample from native 1000) | 1.0-3.5 s = 2.5 s | monopolar | 62 ch (or 21-ch CANON for cross-dataset) | per-trial z-score} -> PRIMARY: lmdb_250Hz_6s/Lee2019-MI (250 Hz monopolar). ALT 200 Hz FM-comparable: datalake 5e77943a/Lee2019_MI (sfreq=200 verified). [TSMNet; 2401.10746]

Cho2017 (~49 subj, binary): {8-30 Hz | 250 Hz | 0-3 s (MI only ~3 s, hard ceiling) | monopolar | 64 ch (or 21-ch CANON) | per-trial z-score} -> PRIMARY: lmdb_250Hz_6s/Cho2017. ALT: datalake 5e77943a/Cho2017 (200 Hz). [MOABB Cho2017]

PhysionetMI (109 subj): {encoder 4-40 / cov 8-30 Hz | 200 Hz (native 160; datalake already 200) | 0-3 s (~3 s tasks) | monopolar | 64 ch | per-trial z-score} -> PRIMARY: datalake 5e77943a/PhysionetMI (sfreq=200 verified, event-locked) via processed_data.py. ALT 250 Hz: lmdb_250Hz_6s/PhysionetMotorImagery. [MOABB PhysionetMI]

Stieger2021 (~51-62 subj): {8-30 Hz | 200/250 Hz | fixed 3 s from segment onset (cursor trials variable) | monopolar | ~24 motor ch (reanalysis) or full 64 | per-trial z-score} -> PRIMARY: datalake 5e77943a/Stieger2021 (sfreq=200, event-locked) via processed_data.py. ALT: lmdb_250Hz_6s/Stieger2021. [Sci Data 2021 PMC9436944]

Schirrmeister2017 (HGD, 4-class): {encoder 4-125 Hz (high-gamma is signal; do NOT 38 Hz LP) / cov 4-38 Hz | 250 Hz | 0.5-4.0 s = 3.5 s | monopolar | 44 motor ch (subset of 128) | exp-moving-standardize OR per-trial z-score} -> LOCAL: lmdb_250Hz_6s/Schirrmeister2017 (128 ch verified; select 44 motor). [Schirrmeister 1703.05051]

Dreyer2023: {8-30 Hz | 250 Hz | 0.5-3.5 s | monopolar | dataset ch | per-trial z-score} -> LOCAL: lmdb_250Hz_6s/Dreyer2023 (+A/B/C variants present). [MOABB]

Cross-dataset (Protocol C): {8-30 Hz | 250 Hz (or 200 for FM) | min common window ~2.5-3 s | monopolar | 21-ch CANON intersection (cross_dataset.py CANON, verified shared by 2a/Lee2019/Cho2017) | per-trial z-score; +per-subject Euclidean Alignment} -> via cmi/data/cross_dataset.py. For deep-encoder-only arm, IMAC-style 10-20-coordinate interpolation to 64-ch is permissible but EXCLUDED from covariance features. [IMAC 2508.03437; EA 1808.05464]

=== EMOTION (200 Hz spine, monopolar; raw primary + DE secondary) ===

SEED (3-class): {raw: 0.5-45 Hz (harmonize) or native 0-75 + 50 Hz notch | 200 Hz | 1 s (CBraMod-match) or 4 s windows | monopolar 62 ch | per-trial z-score (per-session only as ablation) | DE track: 62x5 bands delta1-4/theta4-8/alpha8-14/beta14-31/gamma31-50, 1 s, LDS} -> LOCAL: /projects/EEG-foundation-model/SEED via emotion_data.py (200 Hz, 62 ch). Single-session LOSO. [SEED TAMD 2015; CBraMod 2412.07236]

SEED_IV (4-class): {raw: native 1-75 (or harmonize 0.5-45) + 50 Hz notch | 200 Hz | 4 s non-overlap | monopolar 62 ch | per-trial z-score | DE track: same 5 bands, 4 s} -> LOCAL: SEED_IV/eeg_raw_data via emotion_data.py. [EmotionMeter TCyb 2019]

DEAP (valence/arousal binary): {use supplied preprocessed: 4-45 Hz, common-average, EOG-removed | 128 Hz native (keep) | drop 3 s baseline then 4 s windows | common-avg 32 ch | per-trial z-score | labels binarized at midpoint 5 (NOT 3/6 discard for headline) | DE track: 4 bands only (no delta)} -> LOCAL: DEAP/data_preprocessed_python via emotion_data.py (drops 3 s baseline, >5 threshold - already correct). [DEAP TAFFC 2012]

=== CLINICAL / SCPS (LOSO only; monopolar primary, bipolar baseline-only) ===

ADFTD (3-class A/F/C; or binary A/C): {dataset-canonical/cov branch: 0.5-45 Hz Butterworth + A1-A2 mastoid + ASR(SD17) + ICA/ICLabel, 19 ch full-rank | FM/encoder branch: 0.3-75 + notch, average-ref, NO ICA | resample 200 (FM) or 100-250 (cov) | 4-5 s windows | monopolar 19 ch | per-trial z-score; RAW path for leakage} -> LOCAL: datalake/raw/ADFTD/ds004504-1.0.8 via diagnosis_data.py (currently 0.5-45, 128 Hz, 4 s - bump to 200 Hz, document ICA-free deviation from canonical). [Miltiadous Data 2023; Gemein 2002.05115]

MUMTAZ (binary MDD/HC): {0.5-45 Hz + 50 Hz notch | 200 Hz (from native 256) | 4-5 s non-overlap | linked-ear native (cov) or average (FM) | monopolar 19 ch | per-trial z-score | state EC-vs-EO session explicitly} -> LOCAL: datalake/raw/mumtaz/edf via diagnosis_data._load_mumtaz (uses *EC.edf, 19-ch MUMTAZ_CH, currently 128 Hz - bump to 200). NO canonical recipe -> pin these. [Mumtaz 2016/2017]

TUAB (binary normal/abnormal): {COVARIANCE/method PRIMARY: 0.3-75 + 60 notch, 200 Hz, 4 s windows, average-ref, 19 MONOPOLAR 10-20 ch, crop 60-240 s, per-trial z-score} -> LOCAL: datalake/processed/5e77943a/TUAB (200 Hz monopolar, TUAB_CH 19 verified) via diagnosis_data._load_tuab. {FM BASELINE ONLY: 16-ch bipolar TCP, 10 s, /100} -> LOCAL: tuab_processed_0215/{train,val,test} (CBraMod format). DO NOT feed bipolar to covariance branch. [Gemein 2002.05115; BIOT 2305.10351; CBraMod 2412.07236]

TUEV (6-class events): {COVARIANCE: 0.3-75 + 60 notch, 200 Hz, 5 s event-locked (event +-2 s), 19 monopolar, per-trial z-score} -> LOCAL: datalake/processed/5e77943a/TUEV (200 Hz monopolar, event_id verified) via processed_data.py. {FM BASELINE: 16-ch bipolar TCP, 5 s} -> LOCAL: tuev_processed_0218 (CBraMod format). Event-locked windowing must be preserved. [BIOT; CBraMod]

=== RAW LEAKAGE PATH (all datasets) ===
{resample-only (no clip, no ICA, no heavy bandpass, no per-subject norm) | referential/monopolar montage | per-trial z-score at most} -> use the monopolar LMDB (MI) and 5e77943a monopolar (clinical/emotion) read with minimal processing; report I(Z;D|Y) here so baseline leakage stays high. [Fdez 2021 fnins.2021.626277]

---
## UNIFIED RECOMMENDATION

RECOMMENDED UNIFIED PIPELINE (with mandated per-dataset deviations).

SPINE: A task-keyed pipeline, NOT one global tuple. Two spines:
- MI spine = 250 Hz, monopolar, from eeg_pipeline_preprocessed_datasets_lmdb_250Hz_6s (already 250 Hz monopolar — the right covariance substrate; the "6 s" is only a storage container).
- Clinical+Emotion spine = 200 Hz, monopolar, from datalake processed/5e77943a (already 0.3-75 + 60 Hz notch; matches CBraMod/BIOT for FM comparability) and the 200 Hz emotion raw dirs.
Do NOT mix 128/250/200 in one results table; report MI tables at 250 Hz, clinical/emotion at 200 Hz. (This changes the current loaders' resample=128 default.)

BAND (dual, off one broadband store — no double storage):
- Deep encoders (EEGNet/ShallowConvNet/Deep4Net/EEGConformer) consume broadband: MI 4-40 Hz (HGD up to 125 Hz for high-gamma), clinical/emotion 0.3-75 Hz.
- Covariance/Riemannian (LogCov/SPDNet) internally narrow to 8-30 Hz (binary MI) / 4-38 Hz (4-class) — TSMNet-style — for well-conditioned SPD matrices.

WINDOW: task-keyed family.
- MI ≤3 s common, capped by Cho2017/PhysionetMI (~3 s) and below 2a's 4 s. MANDATED DEVIATION: 2a uses the post-cue 0.5-3.5 s window CROPPED from the 6 s LMDB container (the 6 s field is NOT a valid 2a MI window); HGD uses 0.5-4.0 s (3.5 s); Cho/Physionet 0-3 s.
- Emotion 4 s (1 s optional for CBraMod-match). Clinical 5 s (FM) / 6 s (Gemein covariance variant). TUEV must stay event-locked (5 s, event +-2 s).

REFERENCE/MONTAGE: MONOPOLAR throughout for our method's primary tables and the entire covariance/Riemannian arm (LMDB and 5e77943a are already monopolar). Bipolar 16-ch TCP (tuab_processed_0215 / tuev_processed_0218) is used ONLY for an explicit CBraMod/BIOT foundation-model baseline and reported as a separate, baseline-specific pipeline — because bipolar collapses SPD covariance rank.

ALIGNMENT (calibration-free): per-subject/per-session Euclidean Alignment (R̄ = mean trial covariance; X̃ = R̄^{-1/2} X), applied after temporal filtering and before encoder/covariance. Label-free; never Label Alignment at test time. SPDDSMBN is the in-model analogue for SPDNet.

NORMALIZATION x LEAKAGE (explicit, load-bearing for the paper):
- ACCURACY benchmark: per-trial per-channel z-score over time (current loader default — leakage-neutral, source-only) or braindecode causal exponential-moving standardization for the deep branch. Statistics fit within-trial / source-only; NEVER on the target subject.
- LEAKAGE/diagnostic story stays on RAW, minimally-normalized, referential signal. WHY THE INTERACTION MATTERS: per-subject/per-recording normalization itself drives a subject-ID probe toward chance (Fdez 2021, ~31-33%), i.e. it performs much of the domain-info removal that LPC-CMI is designed to perform. If we per-subject-normalize before the I(Z;D|Y) probe, baseline leakage starts low for ALL methods (including ERM), the method's gap shrinks, AND the measured leakage becomes an artifact of the normalization choice rather than of the encoder. Therefore: (1) run the leakage probe on raw/un-normalized signal so baseline leakage is high and the regularizer's effect is visible; (2) use only leakage-neutral per-trial z-score for the main accuracy tables; (3) report per-subject normalization solely as an ablation, demonstrating the regularizer still reduces residual leakage on top of it.

EVAL/STATS (MOABB standard): CrossSubject LOSO + leave-one-dataset-out; balanced accuracy / kappa / macro-F1 + ECE/NLL; Wilcoxon signed-rank (N>20) or permutation t-tests (small N), Stouffer's Z to aggregate datasets.

NET DEVIATIONS FROM CURRENT CODE: (a) bump MI resample 128->250 (consume LMDB directly) and clinical/emotion 128->200; (b) crop 2a/Cho/Physionet to ≤3 s rather than relying on the 6 s field; (c) add the 8-30 Hz internal narrowing inside the covariance extractor while feeding 4-40 Hz to encoders; (d) add per-subject Euclidean Alignment; (e) keep a separate raw, un-normalized leakage path.

---
## OPEN QUESTIONS (need user sign-off)

1. MI covariance arm: monopolar is mandatory, but for cross-dataset pooling do we (i) intersect to the 21-ch CANON set (clean SPD, loses channels) or (ii) keep per-dataset full montages and only intersect when 2b's 3 ch must be pooled? Recommend (i) for the LogCov/SPDNet headline; need your sign-off.
2. MI resample: confirm 250 Hz spine (consume the LMDB directly) vs 200 Hz (reuse datalake 5e77943a, FM-comparable). 250 preserves HGD high-gamma and matches braindecode/TSMNet; 200 unifies with clinical. Recommend 250 for MI tables, 200 only for FM-baseline comparisons.
3. Emotion input: raw 1 s/4 s patches (matches our encoders + CBraMod) as PRIMARY, with DE 62x5 as the literature-comparable SECONDARY? Confirm whether the DE track is in scope for AAAI or deferred.
4. DEAP label binarization: lock the canonical midpoint-5 split for the headline (3/6-discard only as ablation)? And valence vs arousal vs 4-quadrant as the reported target?
5. ADFTD: use dataset-canonical (0.5-45, A1-A2, ASR+ICA, 19 ch) for the covariance branch, or stay on the ICA-free FM recipe already in diagnosis_data.py? ICA removes leakage-relevant structure, so for the RAW leakage story we must NOT use the ICA'd version — confirm we keep an ICA-free path for the leakage probe.
6. Clinical TUAB/TUEV: confirm the monopolar 19-ch (5e77943a) variant is the PRIMARY for our method and the bipolar tuab_processed_0215/tuev_processed_0218 is strictly a CBraMod/BIOT baseline (not merged into covariance tables).
7. MUMTAZ has no canonical recipe and we must pin it: confirm eyes-closed (EC) session only, linked-ear reference, 0.5-45 Hz, 200 Hz, 5 s windows — and that EC/EO are never mixed (mixing injects label-correlated domain structure).

---
## KEY PAPERS

- arXiv:2206.01323 — SPD domain-specific batch normalization to crack interpretable unsupervised domain adaptation in EEG (TSMNet/SPDDSMBN): Closest covariance/Riemannian DG baseline; exact MOABB+MNE template (250/256 Hz, 4-36 Hz Butterworth, <=3 s monopolar per-dataset windows, 2a 0.5-3.5 s) to copy for our LogCov/SPDNet + SPD-batch-norm arm. Local repo already at repos/TSMNet.
- arXiv:1808.05464 (IEEE TBME 2020) — Transfer Learning for BCIs: A Euclidean Space Data Alignment Approach (Euclidean Alignment): Defines the calibration-free per-subject alignment (R-bar = mean trial covariance; X-tilde = R-bar^{-1/2}X) we adopt before encoders/covariances; 8-30 Hz / 0.5-3.5 s concrete numbers.
- arXiv:2401.10746 — A Systematic Evaluation of Euclidean Alignment with Deep Learning for EEG Decoding: Shows per-subject EA improves cross-subject deep nets ~4.3% and harmonizes HGD (128ch/512Hz -> 22ch/250Hz, 8-32 Hz) for cross-dataset use; template for our encoder arm.
- arXiv:1703.05051 — Deep learning with convolutional neural networks for EEG decoding and visualization (Deep4Net/ShallowConvNet; High-Gamma Dataset): Canonical Deep4Net/ShallowConvNet + HGD numbers: 250 Hz, 4-38 Hz (2a) / up to 125 Hz (HGD high-gamma), 0.5-4 s window, 44 motor ch, exponential running standardization (decay 0.999).
- arXiv:2412.07236 (ICLR 2025) — CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding: Defines the exact clinical recipe of our local stores (200 Hz, 0.3-75 + 60 Hz notch, 16-ch bipolar TCP for TUH, /100 scaling, 1 s patches); the foundation-model baseline our tuab_processed_0215/tuev_processed_0218 dirs follow.
- arXiv:2305.10351 (NeurIPS 2023) — BIOT: Biosignal Transformer for Cross-data Learning in the Wild: Source of the 16-ch bipolar TCP montage list, TUAB 10 s / TUEV 5 s, 95th-percentile normalization; the bipolar convention that conflicts with monopolar covariance and is baseline-only for us.
- arXiv:2002.05115 (NeuroImage 2020) — Machine-learning-based diagnostics of EEG pathology (Gemein): Canonical MONOPOLAR TUAB recipe and the Riemannian story: 21-ch 10-20 average-ref, 100 Hz, clip +-800 uV, 6 s crops, full-rank 21x21 covariance (231 tangent dims). Reference for our clinical covariance branch.
- Fdez et al. 2021, doi:10.3389/fnins.2021.626277 — Stratified Normalization for Cross-Subject EEG Emotion Recognition: Direct evidence that per-subject/session normalization drops a subject-ID probe to ~31-33% (near chance) while keeping emotion accuracy — the key confound for our I(Z;D|Y) leakage metric; grounds the raw-leakage-path + per-trial-only-norm decision.
- arXiv:2311.05415 — EEG-DG: A Multi-Source Domain Generalization Framework for MI EEG Classification: Nearest method comparator (marginal+conditional alignment, close to our CMI); LOSO/LMSO on 2a (81.79%) / 2b (87.12%) treating subject as domain — identical to our setup. Note: does NOT mix montages, so a method baseline not a cross-dataset preprocessing template.
- arXiv:2404.15319 — The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark: De-facto eval/stats standard we adopt: paradigm-default bands (MI 8-32), CrossSubject splits, Wilcoxon/permutation/Stouffer aggregation.
- Miltiadous et al. 2023, MDPI Data 8(6):95 — A Dataset of Scalp EEG Recordings of Alzheimer's, FTD and Healthy Subjects (ADFTD, ds004504): Dataset-canonical ADFTD pipeline (19-ch, eyes-closed, 500 Hz, 0.5-45 Hz Butterworth, A1-A2 mastoid, ASR+ICA/ICLabel) vs our local ICA-free FM recipe; needed to pin the ADFTD covariance vs leakage paths.