## 5. ECG interface, candidate zoo & patient-level split

*This section freezes the physical input interface, the architecture-CONTROLLED
primary candidate zoo (with content-addressed candidate IDs mirroring the EEG c86 scheme),
and the patient-level acquisition/held split. Nothing here reads any model outcome.
Choices that cannot be pinned from public metadata alone are deferred to explicit
NO-OUTCOME rules (a resource + signal-integrity audit or a metadata audit), stated as
rules below, never as outcome-conditioned values.*

Data source for **all** cohorts is the George B. Moody PhysioNet / CinC Challenge 2021
public training release **v1.0.3** (CC BY 4.0, pre-verified downloadable without
credentials), supplemented for PTB-XL by its native PhysioNet release. Only the **public
training partition** of Challenge-2021 is used; the withheld validation/test partitions of
CPSC and Georgia are never touched.

### 5.1 Physical interface: 12-lead, 10-second, fixed lead order

All six cohorts are **12-lead** clinical ECG in WFDB format. The frozen lead order is the
standard clinical order (backbone `in_chans = 12`):

```text
INTERFACE.LEAD_ORDER (frozen)
  index : lead
    0   : I
    1   : II
    2   : III
    3   : aVR
    4   : aVL
    5   : aVF
    6   : V1
    7   : V2
    8   : V3
    9   : V4
   10   : V5
   11   : V6
Reordering rule: match each cohort's WFDB header lead names to this order;
any record missing or misnaming any of the 12 leads is reason-coded C87-E
(fail-closed, excluded) — NEVER silently zero-filled or re-derived.
```

Native record characteristics (public Challenge-2021 v1.0.3 training partition; verified
from the challenge documentation and per-dataset papers — see §5.9 facts):

```text
SOURCE domains D0 (used to train the zoo; OACI/SRC domain axis):
  PTB-XL      21,837 records / 18,885 patients  500 Hz native (+100 Hz vendor downsample)  10 s        12-lead
  CPSC2018     6,877 records                     500 Hz native                              6–144 s     12-lead
  CPSC-Extra   3,453 records                     500 Hz native                              6–144 s     12-lead
  (CPSC2018 + CPSC-Extra = 10,330 public training records)

TARGET cohorts (UNTOUCHED until selection freeze; never used to train the zoo):
  Georgia (Emory)         10,344 records         500 Hz native                              5–10 s      12-lead
  Chapman-Shaoxing        10,247 records         500 Hz native                              10 s        12-lead
  Ningbo                  34,905 records         500 Hz native                              10 s        12-lead
  (Chapman-Shaoxing + Ningbo = 45,152; same Zheng-group SNOMED pipeline — independence
   caveat is a cohort-inclusion concern, tracked in the cohort section, not here.)
```

**Every cohort in scope is natively 500 Hz.** The only sub-500 Hz waveform available is
PTB-XL's vendor-supplied 100 Hz downsample; no cohort in scope is natively below 500 Hz, so
the unified interface is reachable from a single common 500 Hz native rate by downsample-only
(no cohort is ever upsampled/interpolated). The physical window is **the first 10.0 s** of
each record (see §5.3).

### 5.2 Unified sampling rate — NO-OUTCOME rule (resource + signal-integrity audit)

The unified rate `r*` is **not pinned to a value here**; it is chosen from
`R = {100, 250, 500} Hz` by a pre-registered audit that reads only waveform spectra,
morphology, and resource meters — never any model performance.

```text
RULE C87-RATE (resolve r* before any candidate is trained; writes into INTERFACE spec hash)
  Admissibility, per candidate r in {100, 250, 500}:
    (R1 no-synthesis)     500 mod r == 0  AND  r <= 500
                          -> obtain r from each cohort's 500 Hz native by ONE fixed
                             zero-phase FIR anti-alias low-pass (cutoff 0.9*(r/2)) + integer
                             decimation (factor 500/r). No cohort is upsampled/interpolated.
                             PTB-XL is decimated from its 500 Hz file; its vendor 100 Hz file
                             is NOT used (avoids a mixed-anti-alias-filter confound).
                             {100,250,500} all satisfy R1 (factors 5,2,1).
    (R2 signal-integrity) On a LABEL-FREE waveform sample, the audit measures per r:
                             (a) per-lead spectral-energy fraction retained below r-Nyquist
                                 vs the 500 Hz reference;
                             (b) QRS morphology fidelity = correlation of R-peak-aligned
                                 median beats (decimated vs 500 Hz);
                             (c) diagnostic-band energy preservation up to a pre-registered
                                 f_diag (reference diagnostic ECG band per AHA ~0.05-150 Hz).
                             r admissible iff (a),(b),(c) each >= pre-registered thresholds.
    (R3 resource)         Full 648-model zoo training + 3-cohort field inference + frozen
                             field storage at r must fit the compute/storage envelope; the
                             audit records CPU/GPU-hours and bytes at each r.
  Decision:
    Among r in R passing R1 AND R2 AND R3, pick the SMALLEST r (lowest cost); if the smallest
    fails R2's diagnostic-band threshold, pick the smallest admissible r that clears R2; ties
    -> larger retained bandwidth within the envelope. Output r* is content-hashed into the
    interface spec BEFORE training. The audit NEVER reads a model result.
```

### 5.3 Fixed windowing, normalization & QC (structurally leakage-free)

```text
INTERFACE.PREPROCESS (frozen, applied identically to source and target)
  1. Decode WFDB integer samples to millivolts via per-record adc_gain + baseline from the
     header; fail-closed (C87-E) if gain/units are missing or non-mV.
  2. Anti-alias + integer-decimate from 500 Hz native to r* (RULE C87-RATE), one fixed filter.
  3. Window = first 10.0 s (= 10*r* samples) from t0:
        native duration  > 10 s  -> crop to first 10*r* samples
        native duration  = 10 s  -> as-is
        native duration  < 10 s  -> right zero-pad to 10*r* samples (affects Georgia 5-10 s)
  4. Per-record, per-lead z-score: for each (record, lead) subtract the window mean and divide
     by window std (ddof=0), std floor eps = 1e-6 mV; a lead with std < eps -> all-zeros.
     (Fully LOCAL to one record x lead: no fitted parameters, no cross-record / cross-view /
      cross-cohort statistics -> structurally leakage-free across the acquisition/held split.)
  5. Fixed hard clip to [-20, +20] (post-z-score, dimensionless) to bound artifact spikes.
  Output tensor: shape [12, 10*r*], float32, lead axis in INTERFACE.LEAD_ORDER.

QC (NO-OUTCOME; uses ONLY header/waveform statistics, never labels or model outputs):
  exclude + reason-code (C87-E) records that fail header decode, are missing any of the 12
  leads, are flat across more than K leads, or have native duration below the audit floor.
  K and the duration floor are finalized in the signal-integrity audit (§5.2), not by outcome.
```

We deliberately choose a **per-record-local** normalization so that **zero** fitted
parameters exist; there is therefore no source statistic that could leak into targets and no
acquisition statistic that could leak into the held view. Any alternative that fitted a
global/dataset statistic would be required to fit on **source-train only** — we avoid that
surface entirely.

### 5.4 Primary candidate zoo — architecture-CONTROLLED, content-addressed

One **shared 1D ECG backbone** (a fixed 1D residual CNN family, `in_chans = 12`, multi-label
head over the registered scored SNOMED class set; concrete depth/width/kernel spec pinned by a
NO-OUTCOME reference-implementation rule and frozen by spec hash **before** any weight is
trained, never tuned on any cohort). **All 648 candidates share this one architecture** — only
the training objective, panel, seed, support level, and checkpoint epoch differ. Architecture
is thus held constant so any Measurement→Control effect is attributable to modality, not to
architecture family.

```text
ZOO.TOPOLOGY (frozen)
  Contexts (source-side only; 8 total) = panel x seed x support_level:
    panel         in {A, B}      # 2 hash-locked source-patient train/audit partitions (§5.6)
    seed          in {s0, s1}    # 2 fixed training seeds
    support_level in {0, 1}      # 0 = full source support; 1 = one registered (source-cohort,
                                 #   SNOMED-class) cell deleted from source-train (OACI
                                 #   missing-cell intervention); target/held views byte-invariant
    => 2 x 2 x 2 = 8 contexts

  Candidates per context (81):
    ERM  : 1   candidate   canonical index 0        # Stage-1 ERM checkpoint (shared)
    OACI : 40  candidates  canonical index 1..40    # Stage-2 OACI checkpoints, epochs ascending
    SRC  : 40  candidates  canonical index 41..80   # Stage-2 SRC checkpoints, epochs ascending
    Checkpoint cadence (OACI, SRC): epochs range(4, 200, 5) = {4,9,14,...,199} = 40 checkpoints
    => 81 candidates/context

  Total zoo = 8 contexts x 81 = 648 content-addressed models.
  The candidate set A over which a*C_e, a*H_e, and the finite-budget selection â_{pi,B,e} are
  defined (formal-quantities section) is the FULL 648-model zoo per target cohort e; the 8-way
  context factorization is training PROVENANCE carried in each candidate ID, not a restriction
  on A. (This differs from EEG c86, where selection was within a target-subject context; here
  the deployment unit is the target COHORT and candidates pool across the 8 source contexts.)
```

**Content-addressed candidate IDs** (mirroring the EEG c86 scheme, in a **disjoint
namespace** so ECG IDs can never collide with the EEG c84/c86 20-ch / 11-ch namespaces):

```text
CANDIDATE_ID = c87_candidate_id(...) = SHA-256 over canonical-JSON:
  {
    "namespace"             : "C87_ECG_12LEAD_V1",   # disjoint from all EEG namespaces
    "interface_id"          : <ecg interface spec hash: leads|r*|preprocess|window>,
    "training_manifest_hash": <zoo training-manifest hash: optimizer, epochs, cadence, ...>,
    "panel"                 : "A" | "B",
    "seed"                  : s0 | s1,
    "support_level"         : 0 | 1,
    "regime"                : "ERM" | "OACI" | "SRC",
    "canonical_epoch"       : canonical_epoch(order),   # ERM sentinel; else the range(4,200,5) epoch
    "weight_state_hash"     : <state_hash>              # see below
  }
  weight_state_hash / state_hash = SHA-256 over model_state = params + ALL buffers
    (incl BN running_mean/var/num_batches_tracked), binding keylen|key|dtype|ndim|shape|bytes;
    DEVICE-INDEPENDENT (CPU/GPU byte-identical). Weights stored content-addressed as
    <weight_state_hash>.pt.
  Canonical within-context order: ERM(0), OACI(1..40 by ascending epoch), SRC(41..80 by
    ascending epoch) -> canonical_candidate_ids[0..80].
VALIDATOR (fail-closed, mirrors c86 acceptance):
  recompute every candidate ID from its weight file + provenance tuple; require
  candidate_ids_by_context, in genealogy order (ERM -> OACI1..40 -> SRC1..40), to equal
  canonical_candidate_ids for the full 81. Any within-context permutation, epoch-order swap,
  or content mismatch -> C87-E (rejected). Exactly 1 ERM, 40 unique OACI, 40 unique SRC per context.
```

### 5.5 OACI / SRC / ERM objectives — held identical to the EEG engine

The training engine is reused **byte-identical** from the frozen EEG OACI engine
(`train_stage1` + `train_stage2`, `checkpoint.py`, `primal_dual.py`, the OACI/SRC objective
definitions, optimizer = Adam with weight_decay 0, `deterministic_algorithms = True`,
checkpoint cadence `range(4,200,5)`). **Only** (i) the input adapter (the 12-lead ECG interface
of §5.1–5.3) and (ii) the shared 1D backbone (`in_chans = 12`, multi-label head) change.

```text
ENGINE (unchanged wrappers; modality touches only the input adapter + backbone + task-risk term)
  ERM  (Stage-1) : minimize source-train task risk -> 1 ERM checkpoint/context; yields
                   R_ERM_hat and the risk-feasibility budget tau = R_ERM_hat + eps.
  OACI (Stage-2) : "Overlap-Aware Conditional Invariance" — adversarial minimization of the
                   extractable conditional-domain leakage L_Q^ov on ESTIMATOR-ELIGIBLE
                   (support >= m) (domain,class) cells ONLY, s.t. R_src <= R_ERM_hat + eps;
                   primal-dual, lambda = dual multiplier of the RISK constraint (not a leakage
                   weight); per-class conditional domain critic over the source-domain support
                   set; ineligible cells excluded, never smoothed.
  SRC  (Stage-2) : "Source-Robust" — NON-adversarial minimization of a source-side
                   worst-domain (log-sum-exp over source domains D0) risk endpoint under the
                   SAME risk-feasibility constraint; selected by source-guard.
  Source domains D0 = {PTB-XL, CPSC2018, CPSC-Extra} (3 domains). Source-audit (from the panel
  split, §5.6) supplies the source-guard; the target is NEVER used by the engine.
```

**Forced, outcome-independent modality adaptation to surface (not silently frozen here):** the
EEG task was single-label 2-class (softmax cross-entropy); ECG is **multi-label SNOMED**, so the
inner task-risk term becomes per-class BCE and the "class" axis of the OACI (domain,class) cell
decomposition must be redefined over SNOMED codes. The OACI/SRC **wrappers** (risk-feasibility
constraint, adversary structure, worst-domain LSE, cadence, primal-dual, λ semantics) are
unchanged; the (domain,class)-cell redefinition and the exact scored-class set are owned by the
labels/estimand section and must be pinned there — see §5.9 discrepancies.

### 5.6 Patient-level split — acquisition view vs held-evaluation view

Statistical unit = **PATIENT**. Every WFDB record of a patient is assigned to exactly one
view; **no patient appears in two views** (record → patient → view is a function, so overlap
is 0 by construction). One query costs **one record's** SNOMED label.

```text
SPLIT (applied independently to each TARGET cohort: Georgia, Chapman-Shaoxing, Ningbo)
  Two views per cohort:
    ACQUISITION view (construction / L^C): finite target labels MAY be queried here;
                                           1 query = 1 record's SNOMED label.
    HELD-EVALUATION view (deployment / L^H): SEALED; opened ONLY AFTER the selection freeze;
                                             never queried during acquisition.
  Deterministic, outcome-free assignment:
    key(patient) = SHA-256("C87_TARGET_SPLIT_V1" | cohort_id | patient_id)
    sort patients by key ascending; lower floor(n_pat / 2) -> ACQUISITION,
    remainder -> HELD-EVALUATION.  (mirrors c86 C86_TARGET_SPLIT_V1 floor(n/2) rule)
  Patient-ID resolution (METADATA AUDIT, NO-OUTCOME; resolved before any split materializes):
    - cohort exposes patient IDs (e.g. PTB-XL patient_id) -> group all records by patient.
    - cohort does NOT expose patient IDs -> each WFDB record is its own patient-unit
      (record == patient), DISCLOSED per cohort. (Georgia / Chapman / Ningbo patient-ID
      availability is confirmed in the metadata audit; no default is assumed silently.)
  Min-support gate (NO-OUTCOME): each view must have >= N_min patients and >= m_class records
    for each registered scored SNOMED class; a cohort/class failing -> reason-coded C87-E
    (NO re-split, NO threshold change). N_min, m_class fixed by the metadata audit.

SOURCE panels (train/audit within the source pool; define the zoo contexts of §5.4):
  Panel P in {A, B} partitions SOURCE patients into source-train / source-audit by
    key(patient) = SHA-256("C87_SOURCE_PANEL_" | P | "_V1" | source_cohort | patient_id)
    at a fixed pre-registered audit fraction. A and B use DIFFERENT salts -> two distinct
    partitions (a source-partition-sensitivity axis). Source-audit feeds the SRC/OACI
    source-guard; it is never the target.
```

### 5.7 Architecture-DIVERSE zoo = SECONDARY stress test (deferred)

The primary zoo is architecture-CONTROLLED (one backbone) **on purpose**: it isolates the
modality effect. An **architecture-DIVERSE** zoo (multiple backbone families) is a **SECONDARY,
later stress test only** and is **not** launched with the primary campaign — launching it
together would confound "modality vs architecture family" and make any null/positive
uninterpretable. It is registered here as deferred, to be considered only after the primary
result is frozen, under a separately authorized stage.

### 5.8 Leakage & NO-OUTCOME closure (this section)

- Sampling rate, window, normalization, lead order, zoo cardinality/structure, candidate IDs,
  and the split rule are all fixed **a priori** or by an audit that reads only
  resources/waveforms/metadata — **no model result** enters any of them.
- Normalization is per-record-local ⇒ zero fitted parameters ⇒ no source→target and no
  acquisition→held leakage surface.
- The held-evaluation view is sealed until the selection freeze; candidate IDs are content
  hashes of weights (verifiable by rehashing); the validator is fail-closed.

### 5.9 Verified facts, discrepancies, and open NO-OUTCOME decisions

(Facts, discrepancies with the PI spec, and deferred NO-OUTCOME rules are enumerated in the
structured fields accompanying this section.)