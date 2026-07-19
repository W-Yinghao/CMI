# C87P — Pre-Registration Freeze
## Cross-Modal Measurement–Control Separation under Extreme Action Selection and Conditional Nontransportability (12-lead ECG)

---

### SCOPE BANNER (read first)

```text
C87P = protocol / metadata / literature FREEZE ONLY.
  - This document is DESIGN ONLY. C87 real execution (hereafter "C87E") is NOT authorized by C87P.
  - No data are downloaded, no model is trained, and NO real target-cohort MODEL OUTCOME is read
    anywhere in this freeze. Every value below is public dataset METADATA, published method
    description, cost realism, or a signal-integrity / resource / metadata AUDIT RULE.
  - Nothing in the frozen design (task, interface, sampling rate, budget ladder, thresholds, cohort
    inclusion, zoo structure, gate) depends on any model result. Any not-yet-pinnable choice is
    resolved by an explicit NO-OUTCOME RULE (stated as a rule, never a value chosen from a result).
  - There is NO automatic C88. Authorization of C87E, and any successor, is a separate decision made
    AFTER this freeze and AFTER the synthetic pipeline-validation gate (§6) emits CONTROL_PASS.
```

---

## FREEZE v2 — adversarial red-team pass & resolutions (READ SECOND)

This document was drafted (§1–§6) and then adversarially red-teamed on three independent lenses —
**outcome-leakage / prospective-integrity**, **post-hoc researcher degrees-of-freedom**, and
**method / statistical fidelity**. All three returned **NEEDS_REVISION**. Every surviving BLOCKER and
MAJOR is resolved below and the affected specs (marked `[v2]`) are rewritten in place. Nothing here is
tuned to any ECG model result; every fix either REMOVES a leakage/DoF path or makes the design MORE
conservative.

```text
RESOLVED — BLOCKERS
B1 [leakage]  Degeneracy guard was OUTCOME-DEPENDENT and could EASE the gate (a "vacuous" cohort with
              small held-loss span D_e could be silently dropped, shrinking the all-cohorts conjunction;
              worst case drops Georgia, the sole cross-lineage cohort). FIX (SPEC 4.A/4.F [v2]): vacuity
              may only DOWNGRADE, never a silent drop — a vacuous cohort counts as NON-PASS for the IUT
              and caps the program verdict at WITH-CAVEAT/INCONCLUSIVE. Reconciles D-20 (metadata-only
              inclusion) with D-22 (analysis-time vacuity diagnostic).
B2 [posthoc]  Primary selection-loss LABEL AXIS was left unpinned (single §2 task vs multi-label S).
              FIX (P-21/SPEC 4.A [v2], D-24 deleted): PIN option (a) — the single §2 binary task — as the
              primary selection loss; multi-label S NLL demoted to a pre-declared SECONDARY robustness
              variant. [PM-RATIFY #1]
B3 [posthoc]  Georgia (sole cross-lineage cohort) was EXCLUDABLE via SPEC 4.E branch G-c(ii) with no
              verdict cap → "DEMONSTRATED" could rest on two same-lineage (Zheng) cohorts. FIX (SPEC 4.E/
              4.F [v2]): Georgia is PINNED record==patient / PROCEED-with-conservatism-caveat and is NOT
              G-c(ii)-excludable; the G-c "material grouping" threshold is numerically pinned; HARD verdict
              rule — an unqualified DEMONSTRATED REQUIRES Georgia INCLUDED and passing; any Georgia
              exclusion caps the verdict at WITH-CAVEAT / within-lineage-only; min included cohorts = 3.
B4 [fidelity] Synthetic controls drove everything from a per-record LOSS tensor, but the two label-adaptive
              PRIMARY methods (MODEL SELECTOR needs hard predictions/consensus pseudo-labels; CODA needs
              predicted categories/confusion matrices) CANNOT be exercised by losses alone → CONTROL_PASS
              would certify only P0 + the estimator family, never the label-adaptive path C87 exists to add.
              FIX (§6.1/§3.4 [v2]): the synthetic generator emits, per (candidate, record), a full PREDICTION
              object + a KNOWN true multi-hot label; the loss tensor is DERIVED from (prediction, label) so
              the SAME predictions drive LURE/VMA/ASE (losses), MODEL SELECTOR (hard preds) and CODA
              (confusions); POS plants a prediction/confusion structure a label-adaptive method beats P0 on,
              NEG-B plants none.

RESOLVED — MAJORS
M1 [fidelity] Winner's-curse: a*H_e=argmin / a#_e=argmax over 648 in-sample held losses made T_e and D_e
              extremized order statistics, BIASING T_e upward — i.e. toward the pre-registered NON-TRANSPORT
              (C86D-reproducing) conclusion; BCa does not correct max/min optimism. FIX (SPEC 4.A [v2]):
              CROSS-FIT the held ceiling — select a*H_e/a#_e on one held patient sub-sample, evaluate L^H on
              a DISJOINT held sub-sample, average over folds; plus a new DENSE near-tie control world (§6.2b)
              that stresses the argmin-instability regime. [PM-RATIFY #3]
M2 [fidelity] Non-robust standardizer: D_e = best-to-WORST span is dominated by the near-chance epoch-4
              checkpoint, so requiring active to recover τ_G=0.10 of a best-to-near-chance span may be
              UNREACHABLE by construction (pre-ordaining "NO ADVANTAGE"), and it disagreed with ε_e=0.25·s_e.
              FIX (SPEC 4.A/4.D [v2]): PRIMARY standardizer = robust s_e (SD of held candidate losses),
              ONE consistent scale reference with ε_e; τ_G re-expressed as 0.25 in s_e units; the D_e span
              is demoted to the degeneracy diagnostic + a secondary scale. [PM-RATIFY #2]
M3 [all]      M=5 acquisition/held splits (SPEC 4.D) contradicted the single deterministic P-20 split and
              was an unpinned split-family DoF (patients would straddle views). FIX: resolve to M=1 (single
              frozen P-20 split); variance = K=10 policy seeds × patient-cluster bootstrap; the held-view
              CROSS-FIT (M1) lives INSIDE the held view, not as re-partitioning.
M4 [posthoc]  Cohort/class gate-inclusion thresholds (N_min, m_class, flat-lead K, duration floor) were
              unpinned. FIX (D-03/D-18/§5.6 [v2]): pinned — N_min=50 patients/view; m_class=20 records/
              scored-class/view; flat-lead exclude if >2 of 12 leads flat; native-duration floor=5.0 s.
M5 [fidelity] Main-gate bootstrap (§4.5) did not state G's paired resample. FIX (SPEC 4.F [v2]): every
              policy's loss is evaluated on a COMMON (paired / common-random-number) patient resample within
              each bootstrap replicate.

RESOLVED — MINORS (applied)
  τ_G/τ_T provenance note (pre-declared effect sizes, NOT tuned to any ECG result); r*/QC signal-integrity
  audit pinned to SOURCE-cohort/reference waveforms with a native-500 Hz invariance note (removes the
  target-waveform dependence); "UNTOUCHED" qualified to "untouched w.r.t. any model OUTCOME"; 1D backbone
  named (resnet1d_wang, PTB-XL benchmark, Strodthoff et al. 2020) + spec-hash; Holm family locked at
  CONTROL_PASS; VMA cited as Hara, Matsuura, Honda et al. 2024 (Hara first author); XWED = Expected Weighted
  Disagreement (ASE's acquisition fn, Kossen 2022 — verified, caveat removed); MODEL SELECTOR cited as
  arXiv 2410.13609 pending AISTATS confirmation; Active Testing pinned to the FIXED-surrogate (non-retraining)
  variant so LURE stays a genuinely prediction-driven contrast.
```

### PM RATIFICATION POINTS (estimand/threshold decisions — reserved for the PI)

Three resolutions touch the **estimand/endpoint/threshold** definitions the PI reserved. The red-team-
recommended resolution is APPLIED so the freeze is complete and self-consistent, but each is flagged for
explicit PI ratification (or override) before C87E:

```text
PM-RATIFY #1  Primary selection loss = single §2 binary task NLL (multi-label S = secondary). [B2]
PM-RATIFY #2  Primary standardizer = robust s_e (SD), τ_G = 0.25 in s_e units (was D_e span, τ_G=0.10). [M2]
PM-RATIFY #3  Held ceiling a*H_e / a#_e / T_e / D_e estimated by held-view CROSS-FIT (anti-winner's-curse). [M1]
```

All other resolutions are leakage-removal / DoF-pinning / correctness fixes within the freeze author's
scope and do not require ratification.

---

**Scientific object (fixed; not redefined here).** Measurement–Control Separation: association / reliability ≠ prediction ≠ transport ≠ actionability. The concrete extreme-action control problem is: *per target COHORT, select the single deployment-best model from a large candidate zoo under a small target-label budget.* C87 asks whether the C86D EEG finding — even FULL acquisition-view labels did not identify the deployment-optimal model (FULL nontransportability), and non-adaptive prediction-driven active acquisition gave ≈0 gain over uniform — **reproduces on a different modality (12-lead ECG) once a genuinely label-ADAPTIVE method is added to the registry.**

---

## Registered degrees-of-freedom lock

Every design degree of freedom is either **PINNED** (frozen value/rule with a concrete value) or **DEFERRED** (a NO-OUTCOME resolution rule keyed to a metadata / signal-integrity / resource / synthetic-toy audit, never to a model result). Owner section in brackets.

```text
=== PINNED CHOICES ===========================================================================
ID    ITEM                                   FROZEN VALUE / RULE                                     [OWNER]
----  -------------------------------------  ------------------------------------------------------  --------
P-01  Data release                           PhysioNet/CinC Challenge-2021 v1.0.3 training split,     [§1]
                                             CC BY 4.0, WFDB, SNOMED-CT labels; public, no credential
P-02  Source-training cohorts                PTB-XL 21,837 (Challenge count) + CPSC2018 6,877 +       [§1,§5]
                                             CPSC-Extra 3,453 = 32,167 records (may be trained on)
P-03  Untouched target cohorts (E=3)         Georgia 10,344; Chapman-Shaoxing 10,247; Ningbo 34,905   [§1,§4]
                                             = 55,496 records; frozen BEFORE any model result
P-04  Excluded databases                     St Petersburg INCART (257 Hz), PTB (1000 Hz)             [§1]
P-05  Independence structure of 3 targets    TWO curation/labeling lineages, NOT three (Emory/USA;    [§1.6]
                                             shared Zheng/Chapman Zhejiang lineage = Chapman+Ningbo)
P-06  Negative class (all candidate tasks)   NSR = SNOMED 426783006                                   [§2]
P-07  Candidate task slots                   T1 AF {164889003}; T2 BBB default RBBB-union             [§2]
                                             {59118001,713427006}, fallback LBBB {164909002};
                                             T3 RATE default SB {426177001}, fallback STach {427084000}
P-08  Forbidden unions                       AFL 164890007 NEVER unioned into AF; IRBBB 713426002     [§2]
                                             excluded; no per-cohort widening of P
P-09  Task-selection constants + order       τ_pat=100, τ_src=50, δ=0.10; order =                     [§2]
                                             consistency/presence gate → maximin support S →
                                             ambiguity/dedup tie-break → smallest SNOMED code
P-10  Multi-label ambiguity rule             drop records carrying BOTH a P-code and NSR; pos/neg     [§2]
                                             example sets disjoint by construction
P-11  Method registry PRIMARY                P0 uniform (baseline) + Active Testing/LURE (Kossen 2021, [§3]
                                             prediction-driven) + MODEL SELECTOR (Okanovic 2025,
                                             label-adaptive) MANDATORY; CODA (Kay 2025) PRIMARY-
                                             conditional on §3.4 audit
P-12  Method registry SECONDARY              VMA (Matsuura&Hara 2024); ASE/XWED (Kossen 2022);         [§3]
                                             CODA if it fails the primary-conditional audit
P-13  LURE estimator                         R_LURE=(1/M)Σ v_m·L_m,                                   [§3,§6]
                                             v_m=1+((N−M)/(N−m))·(1/((N−m+1)·q(i_m))−1); E[v_m]=1
P-14  Interface lead order (in_chans=12)     I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6; missing/         [§5]
                                             misnamed lead → C87-E fail-closed
P-15  Preprocessing chain                    mV decode → anti-alias+integer-decimate to r* → first-   [§5]
                                             10 s crop / right-zero-pad → per-record per-lead z-score
                                             (std floor 1e-6 mV) → clip [−20,+20]; per-record-LOCAL
                                             (zero fitted params → no leakage surface)
P-16  Zoo topology                           per context {1 ERM + 40 OACI + 40 SRC}=81; contexts =    [§5]
                                             2 panels × 2 seeds × 2 support-levels = 8; total = 648;
                                             checkpoint cadence range(4,200,5) = 40 checkpoints
P-17  Candidate-ID scheme                    SHA-256, namespace "C87_ECG_12LEAD_V1" (disjoint from    [§5]
                                             EEG); state_hash over params + ALL buffers; fail-closed
                                             validator; canonical order ERM(0)/OACI(1..40)/SRC(41..80)
P-18  Selection pool A_e                     FULL 648-model zoo per target cohort (pooled across the  [§5; RECONCILED]
                                             8 source contexts); gate algebra invariant to per-context
                                             (81) vs pooled (648) choice
P-19  Training engine                        OACI/SRC/ERM byte-identical to frozen EEG OACI engine;   [§5]
                                             ONLY input adapter + 1D backbone change; inner task term
                                             → per-class BCE (multi-label); λ = dual multiplier of the
                                             source-RISK constraint (not a leakage weight)
P-20  Patient-level split                    key=SHA-256("C87_TARGET_SPLIT_V1"|cohort|patient_id),    [§5]
                                             sort asc, lower floor(n/2) → ACQUISITION, rest → HELD;
                                             held-eval SEALED until selection freeze; 1 query = 1 rec
P-21  Primary loss [v2 / PM-RATIFY #1]        patient-level mean binary NLL on the SINGLE §2-selected   [§4]
                                             binary task (label axis PINNED; D-24 deleted); secondary
                                             (reported) = multi-label-S NLL + Hamming@0.5 (robustness)
P-22  Per-cohort estimands [v2]              T_e=L^H(a*C_e)−L^H(a*H_e) (=R at B=0); R_{e,π,B}=        [§4]
                                             L^H(â)−L^H(a*H_e); G_{e,π,B}=R_{P0,B}−R_{π,B}; held optima
                                             a*H_e/a#_e/T_e/D_e via HELD-VIEW CROSS-FIT (anti-winner's-
                                             curse, PM-RATIFY #3); span D_e=L^H(a#)−L^H(a*H_e) (diagnostic);
                                             PRIMARY spread s_e=SD_a L^H (robust standardizer, PM-RATIFY #2)
P-23  Endpoints [v2]                         E1 R~=R/s_e; E2 L^H(â); E3 CVaR_0.10; E4 P(ε-near-opt),   [§4]
                                             ε_e=0.25·s_e; E5a G~=G/s_e / E5b ρ; E6 T_e/T~_e=T_e/s_e
                                             (all standardized by robust s_e, one consistent scale)
P-24  Budget ladder (single authoritative)   B ∈ {0,8,16,32,64,128,256,512} held record-labels/cohort [§4; RECONCILED]
                                             (§3, §6 REFERENCE this ladder verbatim)
P-25  Thresholds + inference [v2]             τ_G=0.25 (in s_e units, PM-RATIFY #2); τ_T=0.05·s_e;     [§4]
                                             ε_e=0.25·s_e; τ_near=0.80; CVaR level 0.10 tail-no-harm;
                                             α=0.05 one-sided; BCa patient-CLUSTER bootstrap 10,000 (CRN-
                                             paired across policies); K=10 policy seeds × M=1 split (single
                                             deterministic P-20 split; held cross-fit is INSIDE the held view)
P-26  Formal gate                            Intersection-Union Test (same method + same budget clears [§4; RECONCILED
                                             LCB_95(G~)≥τ_G in EVERY included cohort; NO pooling) +     registry]
                                             Holm-Bonferroni FWER over registry×ladder; verdicts
                                             {DEMONSTRATED / WITH CAVEAT / NO ADVANTAGE}
P-27  4-way failure decomposition            Cells 1–4 mapped to E6/E5a/E1 triggers (transport ×      [§4]
                                             active-gain), Cell 4 = conditional nontransportability
P-28  Synthetic control gate                 POS + NEG(A,B) + CALIB + mutation tests (CALIB-M1,       [§6]
                                             CALIB-M2, NEG-M3) all must pass → signed CONTROL_PASS
                                             token is a HARD precondition for any C87E real-field read
P-29  Control design constants               SEP_POS=0.05, TAU_NEG=0.05, phi=0.15, K≥1000, α=0.05,    [§6]
                                             coverage band [93.5%,96.5%]

=== DEFERRED CHOICES (NO-OUTCOME resolution rules) ===========================================
ID    ITEM                                   NO-OUTCOME RESOLUTION RULE (resource / integrity /       [OWNER]
                                             metadata / synthetic-toy audit; never a model result)
----  -------------------------------------  ------------------------------------------------------  --------
D-01  Unified sampling rate r*               RULE C87-RATE: r* ∈ {100,250,500} Hz by downsample-only  [§1 R,§5]
                                             integer decimation (500 mod r==0) + signal-integrity
                                             audit (spectral-energy retention, QRS morphology, diag-
                                             band preservation) + resource envelope; smallest
                                             admissible r; content-hashed before training
D-02  Fixed window (Rule W)                  10 s window = 10·r* samples (5,000 at 500 Hz), crop /    [§1,§5]
                                             right-zero-pad uniformly; length from duration-metadata audit
D-03  QC exclusion params (Rule Q) [v2]      PINNED: exclude a record if >2 of 12 leads flat OR native  [§1,§5]
                                             duration <5.0 s; reason-code C87-E; header/waveform stats
                                             only, no perf-based drop (M4)
D-04  Patient-unit per cohort                SPEC 4.E audit: G-a documented linkage → patient; G-b    [§1 G,§2,§4,§5]
      (Rule G / OD-2 / G-a,b,c)              documented one-record-per-patient → record=patient;
                                             G-c no linkage → metadata near-dup audit, default record-
                                             proxy-with-caveat else report-but-exclude. (Chapman &
                                             Ningbo = G-b per source doc; Georgia = record=patient
                                             unless an ID field is found.)
D-05  Scored-label set S (Rule L)            intersection of Challenge-2021 scored SNOMED classes      [§1,§2,§4]
                                             present in ALL included cohorts (dx_mapping_scored.csv),
                                             gated by per-class positive SUPPORT counts only; |S| via
                                             equivalence-merge (26/30 nominal)
D-06  Binding task (OD-1)                     §2.7 algorithm executed on counts recomputed from        [§2]
                                             downloaded v1.0.3 .hea headers; output = single binding
                                             task, committed as a signed manifest BEFORE any scoring
D-07  BBB / RATE within-slot (OD-3/OD-4)      T2 = RBBB-union unless only LBBB passes gate; T3 = SB    [§2]
                                             unless maximin support selects STach; decided by counts
D-08  Source-feasibility per panel (OD-5)     per-panel class-presence (≥τ_src) evaluated vs §5 panel  [§2→§5]
                                             compositions; RESOLVED: every §5 panel contains all 3
                                             source domains incl PTB-XL, so T3 is NOT dropped for panel
                                             reasons (residual: support-level-1 deletion must not
                                             remove the selected positive class from its sole source)
D-09  Empty-admissible (OD-6)                 if no candidate passes the gate → DEV_STOP; no threshold  [§2]
                                             relaxation, no distinct-concept union
D-10  Georgia length (OD-7)                   5–10 s vs 10 s → handled by the interface window rule    [§2→§5]
                                             (right-zero-pad); does not enter task selection
D-11  Primary label-adaptive membership       {MODEL SELECTOR, CODA} PRIMARY iff they pass the §3.4    [§3]
                                             synthetic toy-oracle reference-fidelity audit (A–E) AND
                                             admit a registered multi-label + patient-cluster mapping;
                                             else SECONDARY. Synthetic + public-code only
D-12  Decomposable pointwise loss             for LURE/VMA/ASE unbiasedness, register a decomposable    [§3,§6]
                                             per-record acquisition loss; if deployment metric non-
                                             decomposable, register the additive surrogate separately
                                             from the held-view metric used for T/R/G (estimator theory)
D-13  Reference-implementation pinning        content-address + pin jlko/active-testing, okanovic/     [§3]
                                             model-selector, justinkay/coda BEFORE the audit
D-14  Multi-candidate query rules             VMA O(K)/O(K²) multi-model rule + Active-Testing shared- [§3]
                                             proposal-plus-per-candidate-LURE selection pinned in audit
D-15  Patient-cluster sampling adaptation      one query = one RECORD but CIs PATIENT-clustered; verify [§3,§4,§6]
                                             each method's without-replacement weighting cluster-valid
                                             on the toy, else register the cluster-level modification
D-16  1D backbone architecture [v2]           NAMED: resnet1d_wang (PTB-XL benchmark, Strodthoff et al. [§5]
                                             2020; helme/ecg_ptbxl_benchmarking), in_chans=12, multi-
                                             label head; exact depth/width/kernel spec-hash frozen BEFORE
                                             training; never tuned on any cohort
D-17  Support-level-1 deletion cell            one (source-cohort, SNOMED-class) mid-prevalence cell    [§5]
                                             (present in ≥2 source cohorts, deleted from exactly one);
                                             fixed by metadata rule; target/held byte-invariant
D-18  Min-support gate (N_min, m_class) [v2]   PINNED: N_min=50 patients/view; m_class=20 records per   [§5]
                                             scored class/view; failing cohort/class → C87-E, no re-split,
                                             no threshold change (M4)
D-19  Source panel fraction + salts            audit fraction and salts C87_SOURCE_PANEL_A/B_V1 fixed  [§5]
                                             pre-registered before training (partition-sensitivity axis)
D-20  Cohort inclusion in formal gate          INCLUDED = untouched cohorts passing the SPEC 4.E       [§4]
                                             metadata/integrity audit; decided with zero model info
D-21  Standardizer / ceiling numeric values    D_e, s_e, T_e formulas frozen (SPEC 4.A); numeric       [§4]
                                             values computed at analysis from losses that do not exist yet
D-22  Degenerate-span guard [v2]               cohort with s_e (and D_e) inside patient-cluster        [§4]
                                             bootstrap noise = "no selection signal" → counts as
                                             NON-PASS for the IUT and caps verdict at WITH-CAVEAT; may
                                             only DOWNGRADE, never a silent drop (B1). Reconciles D-20
D-23  Control structural params                A_syn, E=3, B_grid, N_p^(e), cluster-size dist, per-    [§6]
                                             record additive loss, policy set = taken verbatim from the
                                             frozen C87E config / public metadata; never from a result
D-24  Selection-loss label axis [v2 RESOLVED]  RESOLVED (B2/PM-RATIFY #1): PINNED to (a) — binary NLL   [§2/§4]
                                             on the single §2-selected task. Multi-label-S NLL is a
                                             pre-declared SECONDARY robustness variant. No longer a
                                             residual; decided by design, not by any outcome
```

---

## Editor's cross-section reconciliation record

Reconciliations applied so the six sections are internally consistent (details in the "Residual conflicts" appendix and structured fields):

- **[RC-1] Method-registry classification (§4 SPEC 4.F ↔ §3).** §4's gate text listed "A1 expected-loss / A2H pairwise-disagreement (C86D lineage)" as the prediction-driven method and grouped Active-Testing/LURE with the label-adaptive methods. §3 is the authoritative registry: Active Testing/LURE is **prediction-driven**, and the **genuinely label-adaptive** methods are MODEL SELECTOR and CODA. The A1/A2H names are EEG-line methods not carried into the ECG registry. SPEC 4.F below is rewritten to reference §3's registry (P0 + LURE + MODEL SELECTOR mandatory; CODA conditional). §4's gate machinery is method-agnostic, so no gate logic changes.
- **[RC-2] Budget ladder — pinned, single source of truth.** §4 PINS B ∈ {0,8,16,32,64,128,256,512} from label-cost realism (outcome-independent) and cross-checks it against §1 cohort sizes (512 ≤ ~5% of Georgia/Chapman, ≤1.5% of Ningbo). §3's "budget grid deferred to a metadata audit" is subsumed by this pin (the metadata cross-check is satisfied). §6's phrase "the C87E interface-section budget grid" is corrected to **the §4 endpoints/estimand budget ladder**, reused verbatim.
- **[RC-3] Selection pool A_e.** §5 (zoo owner) pins A_e = the **full 648-model zoo** (pooled across 8 source contexts). §3/§6's "81" is the **per-context building block**. §4 confirms endpoint/gate algebra is invariant to per-context vs pooled. Reconciled to 648 pooled.
- **[RC-4] Ningbo patient-unit.** §1/§2 say Ningbo is one-ECG-per-patient (rule G-b, record=patient); §4/§5 flagged Ningbo as a multiple-records-per-patient case (rule G-c). **Web-verified:** the PhysioNet `ecg-arrhythmia` v1.0.0 source documents both Chapman-Shaoxing and Ningbo as one 10-s ECG per patient. Reconciled to **G-b for both** Chapman-Shaoxing and Ningbo; the residual caveat (the Challenge distribution ships no patient-ID field, so this rests on source documentation) is carried by the SPEC 4.E audit, and operationally both readings cluster at record level.
- **[RC-5] PTB-XL counts.** Challenge-2021 v1.0.3 distribution lists PTB-XL = 21,837 records; native PTB-XL v1.0.3 = 21,799 records / 18,869 patients (18,885 is the canonical Wagner-2020 patient count). Reconciled: anything reading **Challenge files** uses 21,837; anything reading **native PTB-XL** for patient linkage uses 21,799/18,869. PTB-XL is source-side and enters no target endpoint.
- **[RC-6] Scored-class count.** §4 cites |S| ≤ 26; §3/§6 cite "30 scored." Reconciled: the Challenge-2021 scored set is 30 nominal diagnoses with equivalent-class merges yielding ~26 distinct scored classes; the exact |S| is fixed by Rule L / `dx_mapping_scored.csv` (D-05). Only material if the selection loss uses the multi-label S (see R-1).
- **[RC-7] T3 source-feasibility.** §2's conditional ("T3 dropped if any source panel is CPSC-only") does **not** fire: §5 source domains D0 = {PTB-XL, CPSC2018, CPSC-Extra} are all present in every panel, so PTB-XL (the SB/STach-bearing source) is always available. Residual flag routed to the metadata audit (D-08).
- **[R-1] RESOLVED in freeze v2 (B2 / PM-RATIFY #1).** The selection-loss label axis is now PINNED to **(a) the single §2 binary task** (§5 still trains a multi-label head, but SELECTION and every gate use the single-task binary NLL; multi-label-S NLL is a pre-declared secondary robustness variant). D-24 is deleted as a deferral. This is one of the three PM-ratification points; the recommended resolution is applied so the freeze is complete.

---

## 1. Data & Cohort Metadata Audit

**Data release (frozen).** George B. Moody PhysioNet/Computing in Cardiology Challenge 2021 — *"Will Two Do? Varying Dimensions in Electrocardiography"* — public training release **v1.0.3**. All redistributed 12-lead ECG databases carry **CC BY 4.0** and are **publicly downloadable without credentialing** (access verified). Labels are **SNOMED-CT** codes via the Challenge `ConditionNames_SNOMED-CT` mapping. Signals are **WFDB** (MATLAB v4 `.mat` + `.hea`; header carries Age and Sex only).

### 1.1 Registered cohort table (public metadata only)

```text
FROZEN COHORT TABLE — PhysioNet/CinC Challenge 2021 v1.0.3 (training release)
Counts = challenge training-split records. Labels = SNOMED-CT (all). License = CC BY 4.0 (all). Format = WFDB.

ROLE     COHORT (challenge id)   #RECORDS  PATIENT IDENTITY        FS(Hz)   LEADS  DURATION      SOURCE INSTITUTION / COUNTRY
------------------------------------------------------------------------------------------------------------------------------------
SOURCE   PTB-XL                    21,837  patient_id present;     500 (also  12    10 s         Physikalisch-Technische Bundesanstalt
         (ptb_xl)                          21,837 rec / 18,885 pt  1000 raw)                      (PTB) / Germany  (Schiller AG devices)
SOURCE   CPSC (cpsc_2018)           6,877  per-record only         500        12    6-60 s       CPSC2018 / ICBEB, ~11 hospitals / China
SOURCE   CPSC-Extra                 3,453  per-record only         500        12    6-60 s       CPSC2018 extra set / China
         (cpsc_2018_extra)
------------------------------------------------------------------------------------------------------------------------------------
TARGET   Georgia (georgia)         10,344  per-record only         500        12    5-10 s       Emory University, Atlanta, GA /
         [UNTOUCHED]                       (no patient_id field)                                  Southeastern USA
TARGET   Chapman-Shaoxing          10,247  1 ECG per patient       500        12    10 s         Chapman University + Shaoxing People's
         (chapman_shaoxing)                (record = patient)                                     Hospital, Zhejiang / China
         [UNTOUCHED]
TARGET   Ningbo (ningbo)           34,905  1 ECG per patient       500        12    10 s         Chapman University + Ningbo First
         [UNTOUCHED]                       (record = patient)                                     Hospital, Zhejiang / China
------------------------------------------------------------------------------------------------------------------------------------
NOT USED St Petersburg INCART          74  per-record             257        12    30 min       INCART / Russia
NOT USED PTB                          516  patient_id present      500-1000   12    10-120 s     PTB / Germany
====================================================================================================================================
Design-used total (6 cohorts): SOURCE 32,167 rec ; TARGET 55,496 rec.
Full challenge training total (8 cohorts): 88,253 records.
```

**[EDITOR — RC-4]** The "1 ECG per patient" entry for Ningbo is confirmed by the `ecg-arrhythmia` v1.0.0 source descriptor (Ningbo raw = 40,258 signals from 40,258 patients; Challenge subset retains 34,905). This supersedes the §4/§5 "Ningbo = multiple records/patient" flag. **[EDITOR — RC-5]** PTB-XL 21,837 is the Challenge-file count; native PTB-XL v1.0.3 = 21,799/18,869.

INCART (257 Hz, 30-min) and PTB (1000 Hz, up-to-120 s) are **excluded** by the source/target split; they are listed for completeness and because their properties would break the sampling-rate/duration homogeneity that the six design cohorts share (all six native **500 Hz, 12-lead**).

### 1.2 Public accessibility & license (confirmed)
- All eight databases are redistributed under **CC BY 4.0**; PTB-XL and the combined Chapman/Ningbo `ecg-arrhythmia` are CC BY 4.0 on PhysioNet.
- Only the **training split** is used; the withheld validation/test partitions are never required.

### 1.3 Patient-identity characterization (unit = PATIENT)
- **Chapman-Shaoxing** and **Ningbo**: each subject contributed **exactly one 10-s ECG** ⇒ **record-level = patient-level**; any record partition is a valid patient split. ✔ (source-documented one-per-patient; rule G-b).
- **Georgia (Emory)**: WFDB headers carry **Age and Sex only — no patient identifier**. Identity is per-record/unavailable; organizers state cross-partition overlap is "vanishingly small" but no field can enforce it → NO-OUTCOME Rule G.
- **PTB-XL** (source): genuine `patient_id` (21,837 rec / 18,885 pt canonical).
- **CPSC / CPSC-Extra** (source): per-record only.

### 1.4 Discrepancy flags vs PI figures
All six PI figures reproduce the canonical Challenge-2021 training-split counts **exactly** (CPSC 6,877; CPSC-Extra 3,453; PTB-XL 21,837; Georgia 10,344; Chapman-Shaoxing 10,247; Ningbo 34,905). Two provenance caveats:
1. **Challenge subset ≠ raw source size**: Georgia challenge 10,344 vs raw Emory ~20,672 (train+val+test); Chapman-Shaoxing 10,247 vs raw Zheng-2020 10,646 patients; Ningbo 34,905 vs raw `ecg-arrhythmia` Ningbo 40,258. PTB-XL 21,837 ≈ raw; CPSC 6,877 = raw CPSC2018 training.
2. **The two "targets" reconstitute one database**: Chapman-Shaoxing 10,247 + Ningbo 34,905 = **45,152** = the single combined **Zheng `ecg-arrhythmia` v1.0.0** database. The challenge split ONE combined database into its two contributing-hospital cohorts (web-verified).

### 1.5 Registered untouched-target freeze list

```text
REGISTERED UNTOUCHED TARGET COHORTS (frozen before ANY model result is read)
  T1  Georgia            (challenge id: georgia)            10,344 records   [Emory / USA lineage]
  T2  Chapman-Shaoxing   (challenge id: chapman_shaoxing)   10,247 records   ]  shared Zheng /
  T3  Ningbo             (challenge id: ningbo)             34,905 records   ]  Chapman-University lineage
SOURCE-TRAINING COHORTS (may be inspected/trained on):
  S1  PTB-XL   21,837   |   S2  CPSC   6,877   |   S3  CPSC-Extra   3,453
EXCLUDED (not in this program): INCART (257 Hz), PTB.
Independence structure of the 3 targets: 2 curation/labeling lineages, NOT 3 (see caveat).
```
**[EDITOR — v2, minor] Precise sense of "UNTOUCHED":** untouched **with respect to any model OUTCOME**. The
targets' SNOMED label-COUNT metadata (§2 task selection) and LABEL-FREE waveform signal-integrity (§5.2, and
even that pinned to source-only) are inspected before the freeze; only the held-view MODEL outcomes are
sealed until the selection freeze. No target model result is read anywhere in C87P.

### 1.6 Independence caveat — do NOT claim "three fully independent cohorts"
The three untouched targets span **two, not three**, independent data-collection/labeling lineages:
- **Lineage A (independent):** Georgia / Emory University (Southeastern USA).
- **Lineage B (shared):** Chapman-Shaoxing and Ningbo are two contributing-hospital cohorts of ONE curation project (Jianwei Zheng et al., Chapman University, with Shaoxing People's and Ningbo First Hospitals; `ecg-arrhythmia`, 45,152 = 10,247 + 34,905). They share (i) the same SNOMED-CT labeling pipeline; (ii) the same physician-adjudication protocol; (iii) identical acquisition standardization (10 s, 500 Hz, 12-lead, one ECG/patient); (iv) same region (Zhejiang Province, China). They differ only in contributing hospital and collection period.

**Consequence for the formal gate.** The gate (same method + same budget must replicate in ALL untouched cohorts; patient-level clustered inference; NO pooled p-value) stands, but replication across Chapman-Shaoxing and Ningbo tests **cross-HOSPITAL transport WITHIN one lineage**; only **Georgia vs {Chapman-Shaoxing, Ningbo}** is genuinely cross-lineage. The report must phrase this as *"three untouched target cohorts across two independent curation/labeling lineages (Emory/USA; and a shared Zheng/Chapman-University Zhejiang lineage covering Chapman-Shaoxing and Ningbo)"* and never assert three fully independent cohorts. **[EDITOR]** This caveat feeds §4 Cell 4 (conditional nontransportability): a Georgia-vs-Zheng-pair split is the expected cross-lineage signature.

### 1.7 Open choices deferred to NO-OUTCOME rules (this section)
- **Rule W (windowing):** one fixed window = 10 s at 500 Hz = 5,000 samples (crop/right-zero-pad uniformly); length from duration-metadata audit. (At r*, the window is 10·r* samples — see §5.)
- **Rule R (resampling):** register 500 Hz common rate (native for all six); resample only records whose header declares another rate. Header-driven.
- **Rule G (Georgia patient unit):** no patient-id field ⇒ register "one record = one patient-unit"; header-integrity audit switches only if an ID field is found.
- **Rule L (scored label set):** fix the scored SNOMED-CT set by per-class positive-support metadata audit (adequate support in source and each target); counts only.
- **Rule Q (record QC):** drop only records failing WFDB signal/header integrity (corrupt, wrong lead count, flatline); no performance-based exclusion.

---

## 2. Label semantics & task-selection protocol (metadata-only)

### 2.0 Scope and invariant
This section freezes **how the single binary task is chosen** using *only* label metadata (SNOMED-CT definitions, per-class label counts, cohort/patient documentation). No model output is read. The candidate set, code-sets, admissibility gate, ranking metric, tie-break chain, and all numeric constants are frozen here; only the *counts* are (re)read in C87E. **Invariant (red-team target):** the frozen task never depends on which task yields a positive result. Selection order = `semantic consistency across cohorts → support → ambiguity/dedup → mechanical tie-break`.

**[EDITOR — R-1]** Whether the *selection loss* is defined on this single binary task or on the multi-label scored set S (§4) is an unreconciled residual conflict; see D-24 / R-1. This section owns the single-binary-task definition; §4 owns the loss functional form.

### 2.1 Data provenance and label system
- Corpus: Challenge 2021 **v1.0.3** training set (CC BY 4.0). SNOMED codes carried in `.hea` comment lines (`#Age,#Sex,#Dx,#Rx,#Hx,#Sx`); each `Dx` is a **set** (multi-label).
- Source: PTB-XL 21,837; CPSC2018 6,877; CPSC-Extra 3,453. **Untouched targets:** Georgia 10,344; Chapman-Shaoxing 10,247; Ningbo 34,905; all 500 Hz.
- Linkage: Challenge headers expose **no patient-ID field**; `ecg-arrhythmia` v1.0.0 is documented one-ECG-per-subject (45,152 = 10,247 + 34,905) ⇒ **record = patient** for all three targets unless the C87E audit surfaces an ID field (rule 2.6).

### 2.2 Pre-registered candidate task set (three slots, frozen code-sets)
Negative class is always **NSR, SNOMED 426783006**.

```text
FROZEN CANDIDATE SET  (positive class  vs  426783006 = sinus rhythm)
  T1  AF        P = {164889003}                         (atrial fibrillation)
  T2  BBB       P = {59118001, 713427006}  [default]    (RBBB, complete/unspecified — Challenge-equivalent union)
                within-slot fallback P = {164909002}    (LBBB), selected only per rule 2.7
                EXCLUDED from P: 713426002 (incomplete RBBB) — distinct concept
  T3  RATE      P = {426177001} [default]                (sinus bradycardia)
                within-slot fallback P = {427084000}    (sinus tachycardia), selected only per rule 2.7
  NEVER unioned into any positive set:
     164890007 (atrial flutter)  — distinct sibling concept, not equivalent to AF
```

```text
Documented per-cohort support (evaluation-2021 dx_mapping_scored.csv; record counts; for the three
targets these equal patient counts and are recomputed at patient level from downloaded headers in C87E):

SNOMED     class     | CPSC  CPSCx | PTB-XL |  GEORGIA  CHAPMAN  NINGBO  | Total
426783006  NSR (neg) |  918     4  | 18092  |   1752     1826     6299   | 28971
164889003  AF        | 1221   153  |  1514  |    570     1780     [ 0 ]  |  5255   <- Ningbo = 0
164890007  AFL       |    0    54  |    73  |    186      445     7615   |  8374   (NOT unioned into AF)
 59118001  RBBB      | 1857     1  |     0  |    542      454      195   |  3051
713427006  CRBBB     |    0   113  |   542  |     28        0     1096   |  1779
713426002  IRBBB     |    0    86  |  1118  |    407        0      246   |  1857   (excluded)
164909002  LBBB      |  236    38  |   536  |    231      205     [35 ]  |  1281   <- Ningbo = 35
426177001  SB        |    0    45  |   637  |   1677     3889    12670   | 18918
427084000  STach     |    0   303  |   826  |   1261     1568     5687   |  9657
427393009  SA        |    0    11  |   772  |    455     [ 0 ]    2550   |  3790   (Chapman = 0)
Target-cohort union for T2 default RBBB (59118001 ∪ 713427006):
                        GEORGIA 570   CHAPMAN 454   NINGBO 1291
```

### 2.3 Semantic-consistency requirement (admissibility gate)
A candidate is **admissible** only if its positive code-set `P` denotes the **same clinical concept in every cohort** and both classes exceed the floor in every target cohort *and* source pool.
1. **Fixed code-set, no per-cohort widening.**
2. **Allowed unions** = is-a/subtype of one concept or Challenge-declared equivalent (RBBB∪CRBBB allowed; Challenge scored `713427006`≡`59118001`; institutions were deliberately not relabeled).
3. **Forbidden unions** = merging distinct siblings to manufacture support (AFL `164890007` is **never** merged into AF; Ningbo's 0 AF is not rescued).
4. **Zeros are disqualifying**, structural or sampling alike (`<τ_pat` in any target cohort → inadmissible).

Presence test: with frozen `P`, `N={426783006}`, per target cohort e ∈ {Georgia, Chapman-Shaoxing, Ningbo}: `n⁺_e ≥ τ_pat` and `n⁻_e ≥ τ_pat` (patient-level, post-dedup, post-ambiguity); and per registered source panel both classes `≥ τ_src`.

### 2.4 Metadata selection inputs
Per candidate/cohort: `n⁺_e`, `n⁻_e` (patient-level, post-dedup/ambiguity); `amb_e` (patients dropped for both classes); source/per-panel class presence. **No performance quantity enters.**

### 2.5 Multi-label ambiguity handling
```text
pos(r) := Dx(r) ∩ P ≠ ∅ ;  neg(r) := Dx(r) ∩ N ≠ ∅
  AMBIGUOUS  (drop, count into amb_e):  pos(r) ∧ neg(r)
  POSITIVE example:                     pos(r) ∧ ¬neg(r)
  NEGATIVE example:                     neg(r) ∧ ¬pos(r)
  NEITHER (ignored, not in the task):   ¬pos(r) ∧ ¬neg(r)
```
Pos/neg example sets disjoint by construction. (Reported-only robustness variant, NOT used for selection: stricter negative `Dx == {426783006}`.)

### 2.6 Patient-level de-duplication
```text
Grouping key K(r):
  if the downloaded v1.0.3 metadata exposes a patient/subject-ID field (C87E audit, rule OD-2):  K(r) = that ID
  else:                                                                                          K(r) = record ID
Patient-level label:
  POSITIVE if ≥1 in-task record is positive and none ambiguous/negative; NEGATIVE symmetrically;
  intra-patient class disagreement → DROPPED.
  (For the three targets, documented one-ECG-per-patient ⇒ no-op, K = record ID.)
```
Source side: PTB-XL groups by external `ptbxl_database.csv` patient_id (handled in §5).

### 2.7 The frozen selection algorithm
```text
INPUT: patient-level {n⁺_e, n⁻_e, amb_e} per candidate over e∈{GEO,CHA,NIN}, plus source-panel presence.
CONSTANTS (frozen, NO-OUTCOME):  τ_pat = 100 ; τ_src = 50 ; δ = 0.10
STEP A — resolve within-slot variants (metadata only): T2 ∈ {RBBB-union, LBBB}; T3 ∈ {SB, STach}.
   keep per slot the variant(s) that PASS §2.3; if >1 passes, keep the larger maximin support S.
STEP B — semantic-consistency / presence gate (§2.3): drop candidates failing in ANY target cohort or
   ANY source panel. ADMISSIBLE = survivors. If ADMISSIBLE = ∅ → DEV_STOP (do NOT relax).
STEP C — support ranking:  S(cand) = min_{e∈{GEO,CHA,NIN}} min(n⁺_e , n⁻_e) ;  winner = argmax_cand S.
STEP D — ambiguity/dedup tie-break (only if top-two S within δ):  A(cand)=max_e amb_e/(n⁺+n⁻+amb); smaller better.
STEP E — mechanical tie-break: smallest SNOMED-CT code min(P).
OUTPUT: exactly one frozen binary task (P, N=426783006) with example sets per target cohort.
```

### 2.8 Exact computation executed in C87E (pre-registered; metadata only)
```text
for each cohort in {source panels} ∪ {GEO, CHA, NIN}:
    parse every .hea header → (record_id, Dx_set)          # signals never read
    K ← patient-ID field if audit finds one, else record_id
for each candidate task (P, N):
    build pos/neg/ambiguous (§2.5); collapse to patients (§2.6); compute n⁺_e, n⁻_e, amb_e; source presence
run STEP A..E (§2.7) → frozen task
emit a signed manifest {selected P,N, per-cohort n⁺/n⁻/amb, S, A, gate pass/fail per candidate}
committed BEFORE any candidate model is scored on any target cohort.
```

### 2.9 Expected (non-binding) outcome under currently-documented metadata
- **T1 (AF)** — expected **inadmissible** (AF Ningbo = 0 < τ_pat; AFL rescue refused).
- **T2 (BBB)** — resolves to **RBBB-union** (LBBB Ningbo 35 fails); clears all three (GEO 570 / CHA 454 / NIN 1291) → admissible, S = 454.
- **T3 (RATE)** — resolves to **SB** (GEO 1677 / CHA 3889 / NIN 12670) → admissible, S = 1677. **[EDITOR — RC-7]** SB has 0 support in CPSC2018 and 45 in CPSC-Extra, so T3 is trainable only in PTB-XL-inclusive panels; since every §5 source panel contains all three source domains (incl PTB-XL), T3 is **not** dropped for panel reasons.
- Illustrative winner = **T3 SB-vs-NSR** (S=1677 > 454); AF-vs-NSR is out either way, decided by label metadata, never a result. Binding pick = C87E recomputation.

### 2.10 Cross-cohort semantic-consistency summary
The one negative class (NSR) and each frozen positive concept are verified present with the same code-set in all three targets (2.2), with two intended disqualifications (AF's Ningbo structural zero; SA's Chapman zero). 500 Hz consistent; Georgia's 5–10 s vs 10 s → interface (§5, right-zero-pad).

---

## 3. Acquisition/selection method registry (reference-fidelity)

**Scope.** Register candidate methods, classify each (*baseline / prediction-driven active / genuinely label-adaptive*), map each to the estimand (choose the single deployment-best of the **648-model zoo per target cohort** under a target-label budget; unit = patient; one query = one record's SNOMED-CT label), and specify the reference-fidelity audit gating the primary family. **[EDITOR — RC-3]** The selector ranges over the pooled 648 zoo; "81" below is the per-context building block. Classifications are from public descriptions; primary/secondary is resolved by a **synthetic toy-oracle audit + public-code numerical agreement + metadata**, never by any real ECG target-model outcome.

### 3.0 Terminology
- **Prediction-driven (active):** the query rule uses only candidate predictions on unlabeled target records; it does not maintain an updated preference over which candidate is best. (C86D's tested class.)
- **Genuinely label-adaptive:** maintains an explicit belief over which candidate is best and **updates it from queried labels**, choosing the next query to disambiguate that belief.
- **Estimator vs selector:** estimator (LURE/ASE/VMA) produces per-candidate risk estimates then `argmin`; selector (MODEL SELECTOR/CODA) maintains a posterior over the best-model index.

### 3.1 Method-by-method registry
**(a) P0 — uniform without replacement (mandatory reference floor).** Draw held records uniformly without replacement, query one label per draw, select lowest empirical held loss after B (deterministic tie-break). Denominator of `G_{e,π,B}`. **Class: baseline.**

**(b) Active Testing / LURE — Kossen, Farquhar, Gal, Rainforth, ICML 2021 (PMLR v139; arXiv 2103.05331).** LURE (Farquhar, Gal & Rainforth, ICLR 2021, arXiv 2101.11665):
```text
R_LURE = (1/M) Σ_{m=1..M} v_m · L_m ,
v_m = 1 + ((N − M)/(N − m)) · ( 1/((N − m + 1)·q(i_m)) − 1 ).
```
Acquisition proposal `q` ∝ surrogate/predicted expected loss (**prediction-driven**); the estimator consumes true labels to stay unbiased under any `q`. **Class: prediction-driven active + unbiased estimator; NOT genuinely label-adaptive.** **[EDITOR — v2, minor]** The ECG registry pins the FIXED-surrogate (non-retraining) Active Testing variant so `q` does not become label-adaptive-in-proposal — keeping LURE a clean prediction-driven contrast to MODEL SELECTOR (pinned in the §3.4 audit). Flags: (i) basic Active Testing evaluates ONE model; selection across the zoo needs a registered multi-candidate proposal (shared proposal + per-candidate LURE, then argmin) — an explicit adaptation, flagged; (ii) LURE assumes a decomposable pointwise loss — non-decomposable metrics require the registered decomposable acquisition loss (D-12). Code: `github.com/jlko/active-testing`.

**(c) ASE / XWED — Kossen, Farquhar, Gal, Rainforth, NeurIPS 2022 (arXiv 2202.06881).** Surrogate imputes loss of every unlabeled point; XWED (eXpected Weighted Disagreement) prefers high-epistemic-uncertainty points contributing to the risk estimate. **Class: prediction-driven active, label-informed surrogate estimator; NOT a best-model posterior.** Heavy (trains an ECG surrogate); **secondary tier on cost, not performance.** **[EDITOR — v2]** XWED = **eXpected Weighted Disagreement**, the named ACQUISITION FUNCTION of ASE (Kossen et al., NeurIPS 2022) — verified; ASE is the surrogate estimator. (Earlier "unverified" caveat withdrawn.)

**(d) VMA — Hara, Matsuura, Honda et al., "Active model selection: A variance minimization approach," *Machine Learning* (Springer) 2024 (DOI 10.1007/s10994-024-06603-1; Satoshi Hara FIRST author).** Estimates the sign of pairwise test-loss differences via LURE, derives the variance-minimizing query distribution, labels adaptively. **Class: prediction-driven active with adaptive query distribution** (label-refined query dist., but estimator ranking, not a Bayesian posterior). Flags: pairwise/multiple-model framing → scaling to the zoo needs O(K)/O(K²) or the multi-model variant (registered, flagged); exact multi-model rule moderately thin → pinned via audit-B.

**(e) MODEL SELECTOR — Okanovic, Kirsch, Kasper, Hoefler, Krause, Gürel, AISTATS 2025 (PMLR v258; arXiv 2410.13609).** Maintains a **posterior over which candidate is best**; each candidate = one scalar error `ε` (bootstrapped from consensus pseudo-labels); queries the record maximizing expected information gain about best-model identity; posterior updated by Bayes from each true label (multiplicative); uses hard predictions. **Class: genuinely label-adaptive (preference-posterior). Primary label-adaptive candidate.** Flags: single-`ε` is a strong simplification for multi-label SNOMED → registered multi-label reduction (D-12); hard predictions map cleanly to the zoo.

**(f) CODA — Kay, Van Horn, Maji, Sheldon, Beery, ICCV 2025 Highlight (arXiv 2507.23771; `github.com/justinkay/coda`).** Dawid–Skene Bayesian over (classifiers × categories × points); consensus-seeded Dirichlet confusion priors; posterior `P_Best`; queries max EIG = `H(P_Best) − Σ_c π̂(c|x)·H(P_Best^c)`; Dirichlet updates (η default 0.01) from true labels. **Class: genuinely label-adaptive (preference-posterior). Second label-adaptive candidate.** Flags: single-label confusion parameterization → multi-label ECG adaptation is nontrivial and gates its primary status; pinned against public code. Its own benchmarks already include Active Testing, VMA, and MODEL SELECTOR — the registry mirrors that comparison set.

**Lineage (context only, not run):** Sawade et al. NeurIPS 2012; Karimi et al. AISTATS 2021.

### 3.2 Classification summary (registry table)
```text
Method            Venue                     Object        Acquisition uses      Updates best-model    Class
                                                          true labels to        posterior from
                                                          re-target queries?    true labels?
----------------- ------------------------- ------------- --------------------- --------------------- -----------------------------
P0 uniform        —                         empirical     no                    no                    BASELINE (reference floor)
Active Testing/   ICML 2021 (Kossen)        per-cand.     no (proposal from     no                    PREDICTION-DRIVEN active
  LURE                                      risk est.       predicted loss)                             + unbiased estimator
ASE / XWED        NeurIPS 2022 (Kossen)     per-cand.     partial (surrogate    no                    PREDICTION-DRIVEN active
                                            risk est.       retrains on labels)                         (label-informed surrogate)
VMA (Matsuura/    Springer ML 2024          loss-diff     yes (adaptive query   no (estimator rank,   PREDICTION-DRIVEN active
  Hara)                                     sign est.       dist. refines)        not posterior)        + adaptive query dist.
MODEL SELECTOR    AISTATS 2025 (Okanovic)   best-model    yes                   yes                   GENUINELY LABEL-ADAPTIVE
                                            posterior
CODA              ICCV 2025 (Kay)           best-model    yes                   yes                   GENUINELY LABEL-ADAPTIVE
                                            posterior
```

### 3.3 Primary family, secondary tier, freeze gate
```text
PRIMARY (mandatory):
  1. P0 uniform-without-replacement            [baseline / denominator]         — MANDATORY
  2. Active Testing / LURE  (Kossen 2021)      [prediction-driven active]       — MANDATORY
  3. MODEL SELECTOR         (Okanovic 2025)    [genuinely label-adaptive #1]    — MANDATORY
PRIMARY (conditional):
  4. CODA                   (Kay 2025)         [genuinely label-adaptive #2]    — PRIMARY iff it PASSES
                                                                                  the §3.4 reference-fidelity
                                                                                  audit AND a multi-label +
                                                                                  patient-cluster mapping is
                                                                                  registered; else SECONDARY
SECONDARY (reported; NOT under the all-cohort formal gate):
  - Hara / Matsuura VMA        (Springer ML 2024)  [adaptive query dist.; pairwise→multi-model scaling flagged]
  - ASE / XWED                 (Kossen 2022)       [surrogate estimator; heavy; complexity-demoted]
  - CODA                       (Kay 2025)          [only if it fails the primary-conditional audit]
```
This satisfies **P0 + ≥1 prediction-driven active (LURE) + ≥1 genuinely label-adaptive (MODEL SELECTOR)**; CODA is a mechanistically-independent second label-adaptive method (Dawid–Skene posterior vs single-`ε`) that strengthens falsification if it clears the audit.

### 3.4 Reference-fidelity audit (toy oracle) — the freeze gate
```text
TOY-ORACLE CONSTRUCTION (synthetic; no real data, no real model):
  - N synthetic records grouped into synthetic PATIENTS (clustered), KNOWN multi-hot labels.
  - K synthetic "models" with KNOWN per-record losses; UNIQUE best model, KNOWN margin; include a
    decomposable pointwise loss + a non-decomposable metric variant to exercise the loss-mapping rule.
  - Ground-truth best index and full-pool risks known by construction.
AUDIT CHECKS (pass/fail at pre-pinned tolerance):
  A. Unbiasedness/consistency — LURE/VMA/ASE recover true full-pool risk within tau_bias; active-var ≤ uniform-var.
  B. Reference-code agreement — LURE weights v_m; MODEL SELECTOR multiplicative updates; CODA Dirichlet+EIG;
     VMA query dist. match public code (jlko/active-testing; okanovic model-selector; justinkay/coda) within tau_num.
  C. Adaptive-method correctness — MODEL SELECTOR / CODA P_Best → KNOWN best as budget→full; EIG ≥ 0, beats P0.
  D. Clustered-sampling validity — one query = one RECORD but CIs PATIENT-clustered; verify each method's
     without-replacement weighting stays valid under patient-cluster sampling, else register the exact modification.
  E. Multi-label mapping check — the registered multi-label→pointwise reduction preserves each method's
     assumptions (decomposable loss for LURE/VMA/ASE; confusion structure for MODEL SELECTOR/CODA).
GATE RULE:
  FREEZE the primary family iff P0 + Active Testing/LURE + MODEL SELECTOR each PASS A–E
  (≥ {baseline + 1 prediction-driven + 1 genuinely label-adaptive}). A method failing any check is DEMOTED
  or EXCLUDED (CODA's primary/secondary status decided exactly here). No pass/fail reads any real ECG result.
```

### 3.5 NO-OUTCOME rules (this section)
1. **Primary label-adaptive membership** ({MODEL SELECTOR, CODA}) — PRIMARY iff pass §3.4 (A–E) + registered multi-label + patient-cluster mapping; else SECONDARY. Synthetic-toy + public-code only.
2. **Budget grid `B`.** **[EDITOR — RC-2]** Superseded by the §4 authoritative ladder B ∈ {0,8,16,32,64,128,256,512}; this registry consumes that ladder verbatim. (Its cost-realism/metadata basis satisfies the original "pre-declared fractions of median cohort patient count" intent.)
3. **Decomposable pointwise loss** for LURE/VMA/ASE; register a decomposable acquisition surrogate separate from the held-view T/R/G metric if the deployment metric is non-decomposable.
4. **Reference-implementation pinning** — content-address + pin the three public commits before the audit.
5. **VMA multi-model rule + Active-Testing multi-candidate proposal** — pinned against reference code / re-derivation in the audit before freeze.

### 3.6 Honesty flags
- VMA multi-model query rule moderately under-specified → audit-B. MODEL SELECTOR single-`ε` for multi-label → reduction (D-12). CODA single-label confusion → multi-label adaptation gates primary status. Active Testing default is single-model *evaluation* → multi-candidate *selection* is our registered adaptation. ASE requires an ECG surrogate → secondary on cost.

---

## 4. Estimand, endpoints & statistical gates

All quantities are **frozen as formulas/rules**. No value is read from any target-cohort model outcome. Standardizers, optima, and ceilings are *estimators*; the freeze pins the estimator, never the number.

### 4.1 Views, loss, per-cohort estimands
Two disjoint **patient** partitions of each target cohort `e`: **View C** (acquisition/construction; fully-labeled reference, no query cost) and **View H** (held/deployment; sealed; labels revealed only through the `B` paid queries). Unit = patient; every record of a patient in exactly one view; cost per record, split/inference per patient.

```text
SPEC 4.A — LOSS AND PER-COHORT ESTIMANDS (frozen)
------------------------------------------------------------
Selection task       : the SINGLE binary task frozen by §2 (P vs NSR). [v2/B2/PM-RATIFY #1] The primary
                       selection loss label axis is PINNED to this single task; the multi-label scored set
                       S (§Data Rule L; |S| via equivalence-merge, ≤26/30) defines a SECONDARY robustness
                       variant only. D-24 is resolved, not deferred.
Candidate set   A_e  : FULL 648-model zoo per target cohort e (§Zoo; per-context 81 is provenance,
                       not a restriction). a*_ ranges over A_e.

Per-record loss  l(a,r):
  PRIMARY  = binary NLL on the single §2 task:
             l(a,r) = -[ y(r) log p_a(r) + (1-y(r)) log(1-p_a(r)) ] ,  y ∈ {0,1} = §2 pos/neg label.
  SECONDARY (reported robustness): multi-label-S mean binary NLL; multi-label Hamming@0.5.
  Selection & every gate use PRIMARY. No metric-shopping.

Patient loss:  lbar(a,j) = (1/|R_j|) Σ_{r in R_j} l(a,r)
View loss:     L^V_e(a)  = (1/|P^V_e|) Σ_{j in P^V_e} lbar(a,j)     (V ∈ {C,H})
Acq optimum:   a*C_e = argmin_a L^C_e(a)   (View C fully labeled; out-of-sample vs View H ⇒ unbiased)

HELD-VIEW CROSS-FIT of the held optima/ceiling [v2/M1/PM-RATIFY #3] — removes winner's-curse:
  Partition held patients P^H_e into F=5 folds (deterministic, salt "C87_HELD_XFIT_V1"|cohort|patient_id).
  For fold f:  select on the OTHER 4 folds  ĥ_f = argmin_a L^{H\f}_e(a) ,  ŵ_f = argmax_a L^{H\f}_e(a);
               evaluate on the HELD-OUT fold f:  L^H_e(ĥ_f)|_f , L^H_e(ŵ_f)|_f.
  Cross-fit estimates:  L^H_e(a*H_e) := (1/F) Σ_f L^H_e(ĥ_f)|_f  (unbiased best-held loss);
                        L^H_e(a#_e)  := (1/F) Σ_f L^H_e(ŵ_f)|_f  (unbiased worst-held loss).
  (a*H_e / a#_e identities are reported per fold; the load-bearing quantity is the cross-fit LOSS.)
Span:          D_e   = L^H_e(a#_e) - L^H_e(a*H_e)   (>0; degeneracy diagnostic + secondary scale)
Spread (PRIMARY standardizer, robust) [v2/M2/PM-RATIFY #2]:  s_e = SD_a L^H_e(a) over candidate held losses
Policy pick:   â_{π,B,e}; CONVENTION at B=0 deploy the acquisition-view pick a*C_e.

Transport gap: T_e        = L^H_e(a*C_e) - L^H_e(a*H_e)          ( = R at B=0 ; L^H via cross-fit )
Held regret:   R_{e,π,B}  = L^H_e(â_{π,B,e}) - L^H_e(a*H_e)      ( >= 0 )
Active gain:   G_{e,π,B}  = R_{e,P0,B} - R_{e,π,B}
Standardized (ONE consistent robust scale s_e):  R~=R/s_e ; T~=T_e/s_e ; G~=G/s_e   (D_e-scaled versions
             reported as a secondary robustness view; ε_e=0.25·s_e and τ_G=0.25 share the s_e scale).

DEGENERACY GUARD [v2/B1]: if s_e (and D_e) is not bounded away from 0 relative to the patient-cluster
bootstrap SE of L^H (all candidates within held-loss noise), cohort e is VACUOUS = "no selection signal".
A vacuous cohort counts as NON-PASS for the all-cohorts IUT and CAPS the program verdict at WITH-CAVEAT /
INCONCLUSIVE — it may only DOWNGRADE, never be silently dropped (reconciles D-20 metadata-inclusion with
this analysis-time diagnostic; a vacuous Georgia can never yield an unqualified positive).
```
The active question: does querying **held-view** labels close regret from the ceiling `T_e` (B=0) toward 0 faster for a label-adaptive method than uniform, robustly in every untouched cohort? `T_e` is the zero-budget anchor and reproduces the C86D object (large `T_e` = FULL nontransportability) while adding the finite-budget-adaptive axis.

### 4.2 Endpoint set
```text
SPEC 4.B — ENDPOINT REGISTRY (frozen; PRIMARY = E1 & E5a)
E1  Cohort-level standardized selection regret  R~_{e,π,B} = R_{e,π,B}/s_e  (curve over budget)   PRIMARY
E2  Patient-level held loss of the pick          L^H_e(â_{π,B,e})
E3  Patient-tail CVaR of the pick                CVaR_{0.10}[ lbar(â,j) : j in P^H_e ]  (worst-decile)
E4  Prob. of ε-near-optimal selection            P_near = Pr_{seeds}[ L^H(â)-L^H(a*H_e) <= ε_e ],
                                                 ε_e = 0.25 * s_e     (M=1 split; over K policy seeds)
E5  Active-minus-passive label efficiency
   (a) G~_{e,π,B} = G_{e,π,B}/s_e                                                                   PRIMARY
   (b) Budget-equivalence factor  ρ_{e,π,B} = min{ B'/B : R_{e,P0,B'} <= R_{e,π,B} }
E6  FULL acquisition-view ceiling                T_e and T~_e = T_e/s_e  ( = R~ at B=0 )  [C86D reproduction]
    (all standardized by the robust s_e [v2/M2]; D_e-scaled versions reported secondarily)
```

### 4.3 Budget ladder (pre-registered from label-cost realism)
```text
SPEC 4.C — BUDGET LADDER (frozen; single authoritative ladder; §3 and §6 reference this verbatim)
------------------------------------------------------------
B in { 0, 8, 16, 32, 64, 128, 256, 512 }   held-view record-labels per cohort.
  B=0 : ceiling anchor (deploy a*C_e; R = T_e).   8–32 : opportunistic / one reading session.
  64–128 : modest funded pilot.                    256–512 : funded annotation batch.
Geometric doubling. A board-certified reader labels one 12-lead ECG's scored set in ~1–3 min, so
512 labels ≈ 1–2 reader-days. Every budget << cohort size:
  512 <= ~5.0% of Chapman-Shaoxing (10,247) and Georgia (10,344) ; 512 <= ~1.5% of Ningbo (34,905).
No budget or ladder endpoint is set from any model outcome.
```

### 4.4 Pre-registered thresholds and inference config
```text
SPEC 4.D — THRESHOLDS & INFERENCE (frozen)
------------------------------------------------------------
tau_G     = 0.25   PRIMARY active-control threshold in s_e units [v2/M2/PM-RATIFY #2]; gate =
                   LCB_95(G~)>=tau_G, i.e. active recovers >=1/4 SD of held-loss spread over uniform.
eps_e     = 0.25 * s_e                    near-optimal margin (E4) — SAME s_e scale as tau_G.
tau_near  = 0.80   secondary: LCB_95(P_near) >= 0.80 at the gate budget.
alpha_CVaR= 0.10   tail level (E3); tail-no-harm secondary: LCB_95( CVaR^{P0} - CVaR^{pi} ) >= 0.
tau_T     = 0.05*s_e (i.e. T~ threshold 0.05 in s_e units) [v2]: TRANSPORTS iff UCB_95(T~_e)<=0.05 ;
                   NON-TRANSPORT iff LCB_95(T~_e)>0.05 ; AMBIGUOUS otherwise.
PROVENANCE [v2, minor]: tau_G, tau_T, tau_near are PRE-DECLARED minimum scientifically-meaningful
                   standardized effect sizes (label-cost / effect-size reasoning), explicitly NOT
                   derived from any ECG target-model result (no ECG result exists at freeze time).
alpha     = 0.05 one-sided (gains); two-sided where a CI is reported.
Bootstrap : B_boot = 10,000 patient-CLUSTER resamples; BCa intervals; CRN-PAIRED across policies for G.
Runs      : K = 10 policy seeds x M = 1 acquisition/held split (single deterministic P-20 split) [v2/M3];
            endpoint = mean over K; the held-view CROSS-FIT (SPEC 4.A) lives INSIDE the sealed held view,
            it is NOT a re-partition of acquisition-vs-held.
```

### 4.5 Patient-clustered inference (per cohort; never pooled)
Inference is run **separately per cohort**; **no pooled cross-dataset p-value**; the only cross-cohort combination is the all-must-pass conjunction (§4.6). Each CI is a **BCa cluster bootstrap over held-view patients** (cluster = patient): each resample draws patients with replacement, recomputes the **cross-fit** `L^H` (and `a*H_e, a#_e, D_e, s_e`), re-evaluates the K=10 policy-seed runs (M=1 split), recomputes the endpoint; **all policies share the SAME resampled patients per replicate (CRN-paired) so G is a paired difference**; `LCB_95`/`UCB_95` come from this per-cohort distribution.

```text
SPEC 4.E — PATIENT-GROUPING NO-OUTCOME AUDIT (frozen rule)
------------------------------------------------------------
Scope: resource / signal-integrity + metadata audit. Reads ONLY .hea fields, native source metadata,
waveform first-moment summaries. Reads NO queried label and NO model output. Per cohort, BEFORE any result.

G-a  Documented patient-ID linkage recoverable: adopt it; all a patient's records in one view; cluster = patient.
G-b  Source documented one-record-per-patient: record == patient; record-level clustering IS patient-level.
     [EDITOR — RC-4: both Chapman-Shaoxing (10,247) AND Ningbo (34,905) are documented one-ECG-per-patient
      per ecg-arrhythmia v1.0.0 (web-verified) ⇒ G-b for both. The Challenge distribution ships no patient-ID
      field, so G-b rests on the source descriptor read by this audit.]
G-c  Linkage NEITHER in the challenge distribution NOR recoverable, AND source not documented-singleton:
     run a metadata-only near-duplicate audit (cluster by exact/near header demographics + low-dim waveform
     summaries; NO labels, NO model outputs).
       (i)  Near-duplicate fraction (records sharing an exact (age,sex,leadstats-hash) key with >=1 other
            record) < 1.0% : treat records as patient-proxy clusters and PROCEED with a documented
            CONSERVATISM CAVEAT.  [DEFAULT]
       (ii) Near-duplicate fraction >= 1.0% : cohort is REPORTED-BUT-EXCLUDED from the formal gate, subject
            to the verdict-cap rule (SPEC 4.F): any target exclusion caps the verdict at WITH-CAVEAT.
GEORGIA PIN [v2/B3]: Georgia has no patient-ID field and is NOT a documented singleton, but the organizers
     document cross-partition patient overlap as "vanishingly small". Georgia is therefore PINNED to
     record==patient / PROCEED-with-conservatism-caveat and is NOT eligible for G-c(ii) exclusion — because
     Georgia is the SOLE cross-lineage cohort (§1.6) and an outcome-agnostic exclusion of it would still
     collapse the cross-lineage test. (Chapman-Shaoxing & Ningbo = G-b per source doc.)
This rule fixes the clustering UNIT by audit, never by a value or outcome.
```

### 4.6 The formal gate
```text
SPEC 4.F — FORMAL GATE (frozen)  [EDITOR — RC-1: registry aligned to §3]
------------------------------------------------------------
Registry (per §Method Registry): P0 uniform (reference) + Active Testing/LURE (Kossen 2021,
prediction-driven) + MODEL SELECTOR (Okanovic 2025, genuinely label-adaptive) MANDATORY; CODA (Kay 2025)
primary-conditional on the §3.4 audit. Secondary (not under this gate): VMA (Matsuura & Hara 2024),
ASE/XWED (Kossen 2022). [Prior C86D-lineage names A1/A2H are NOT part of the ECG registry.]

Included set [v2]: the untouched cohorts passing the metadata-only SPEC 4.E audit. Georgia is PINNED
  included (B3). MIN included cohorts = 3 (all of {Georgia, Chapman-Shaoxing, Ningbo}); fewer → the gate is
  UN-evaluable and the verdict is capped at INCONCLUSIVE.
Per (pi in registry\{P0}, B in ladder\{0}):
  Per cohort e (included set): one-sided patient-cluster bootstrap test, CRN-PAIRED [v2/M5] — within each of
     the 10,000 replicates ALL policies (P0 and pi) are evaluated on the SAME resampled held patients, so G
     is a paired difference:
     H0: G~_{e,pi,B} <= tau_G   vs   H1: G~_{e,pi,B} > tau_G   -> p_{e,pi,B}
     (PASS_e iff LCB_95(G~_{e,pi,B}) >= tau_G).
  CONJUNCTION (Intersection-Union Test = same method + same budget clears tau_G in EVERY included cohort):
     (pi,B) passes iff PASS_e for ALL included cohorts. p^conj_{pi,B} = max_e p_{e,pi,B}. IUT valid at level
     alpha with NO correction, conservative — exactly the all-cohorts requirement (no pooling). A VACUOUS
     cohort (SPEC 4.A degeneracy guard) counts as NON-PASS here [v2/B1].
  FAMILY over the grid: Holm-Bonferroni across {p^conj_{pi,B}} to control FWER at alpha = 0.05. The Holm
     family membership + cardinality are LOCKED at the moment the §6 CONTROL_PASS token is signed and are
     immutable once any real target p-value is computed [v2, minor].

PROGRAM VERDICT [v2 — verdict caps]:
  "ACTIVE CONTROL DEMONSTRATED"  iff >= 1 (pi,B) survives Holm AND, for that (pi,B), in EVERY included cohort
        E4 LCB_95(P_near) >= tau_near AND E3 LCB_95(CVaR^{P0}-CVaR^{pi}) >= 0 (tail-no-harm) — AND Georgia
        (sole cross-lineage cohort) is INCLUDED, non-vacuous, and among the passing cohorts.
  "ACTIVE CONTROL WITH CAVEAT"   iff primary Holm passes but (a secondary fails) OR (Georgia is excluded/
        vacuous, so the result rests on the within-Zheng-lineage pair) OR (any cohort is vacuous).
  "NO ACTIVE ADVANTAGE"          iff no (pi,B) survives Holm.
  HARD CAP: an unqualified DEMONSTRATED can NEVER rest on a Georgia exclusion/vacuity — a vacuous or excluded
        Georgia may only DOWNGRADE the verdict, never upgrade it.

Transport gate (diagnostic; feeds §4.7): per cohort classify FULL via E6/T~_e using tau_T.
Interpretation caveat (§1.6): replication across Chapman-Shaoxing & Ningbo = cross-HOSPITAL within one
Zheng lineage; only Georgia vs {Chapman-Shaoxing, Ningbo} is cross-lineage.
```

### 4.7 Four-way failure decomposition
```text
TABLE 4.G — 4-WAY DECOMPOSITION (frozen mapping to endpoints & verdicts)
------------------------------------------------------------------------------
Cell 1  FULL TRANSPORTS + ACTIVE WINS
  Trigger : UCB(T~_e)<=tau_T all cohorts  AND  Holm-surviving (pi,B) passes.
  Reading : acquisition view informative; label-adaptive querying closes residual finite-budget regret.
  Endpoints: E6 low; E1 low even at small B; E5a LCB(G~)>=tau_G all cohorts; E4 high; E3 tail-no-harm.
Cell 2  FULL TRANSPORTS + ACTIVE FAILS
  Trigger : UCB(T~_e)<=tau_T all cohorts  AND  no (pi,B) survives Holm.
  Reading : transport holds; passive uniform near-saturates; extreme-action control achievable PASSIVELY.
Cell 3  FULL DOES NOT TRANSPORT
  Trigger : LCB(T~_e)>tau_T in >=1 cohort (C86D reproduces on ECG).
  Reading : 3a active-adaptive HELD querying still closes regret (Holm passes) -> active RESCUES nontransport
            (novel positive beyond C86D); 3b it does not -> Measurement->Control gap PERSISTS across modality
            AND under adaptivity (strong negative reproduction).
Cell 4  HETEROGENEOUS ACROSS COHORTS  (Conditional Nontransportability)
  Trigger : transport class OR sign(G~) differs across cohorts, OR the conjunction fails only via ONE cohort.
  Reading : cohort-conditional actionability; the all-cohorts gate cannot pass. Report per-cohort E1/E5/E6,
            name the discordant cohort(s). NO pooled p-value; heterogeneity is descriptive. (Georgia-vs-Zheng-
            pair split is the expected cross-lineage signature per §1.6.)
```

### 4.8 Open NO-OUTCOME decisions carried by this section
1. Patient-grouping unit per cohort → SPEC 4.E (G-a/G-b/G-c). 2. Cohort inclusion → INCLUDED = untouched cohorts passing SPEC 4.E, metadata-only. 3. Standardizers `D_e, s_e`, ceiling `T_e` → formulas frozen; values at analysis. 4. Scored-label set `S` → Rule L. 5. Selection pool `A_e` → 648 pooled (§5), algebra-invariant. 6. Degenerate-span guard.

---

## 5. ECG interface, candidate zoo & patient-level split

Data source for all cohorts = Challenge-2021 public training release **v1.0.3** (CC BY 4.0), supplemented for PTB-XL by its native PhysioNet release; only the public training partition is used. Nothing here reads a model outcome.

### 5.1 Physical interface: 12-lead, 10-second, fixed lead order
```text
INTERFACE.LEAD_ORDER (frozen)
  0:I  1:II  2:III  3:aVR  4:aVL  5:aVF  6:V1  7:V2  8:V3  9:V4  10:V5  11:V6
Reordering rule: match each cohort's WFDB header lead names to this order; any record missing or
misnaming any of the 12 leads is reason-coded C87-E (fail-closed, excluded) — NEVER silently
zero-filled or re-derived.
```
```text
SOURCE domains D0 (train the zoo; OACI/SRC domain axis):
  PTB-XL      21,837 rec / 18,885 pt (Challenge count)  500 Hz native (+100 Hz vendor downsample)  10 s     12-lead
  CPSC2018     6,877 rec                                 500 Hz native                              6–144 s  12-lead
  CPSC-Extra   3,453 rec                                 500 Hz native                              6–144 s  12-lead
  (CPSC2018 + CPSC-Extra = 10,330 public training records)
TARGET cohorts (UNTOUCHED; never train the zoo):
  Georgia (Emory)   10,344 rec   500 Hz native   5–10 s   12-lead
  Chapman-Shaoxing  10,247 rec   500 Hz native   10 s     12-lead
  Ningbo            34,905 rec   500 Hz native   10 s     12-lead
  (Chapman-Shaoxing + Ningbo = 45,152; same Zheng-group SNOMED pipeline — independence caveat = §1.6.)
```
**Every cohort in scope is natively 500 Hz.** The only sub-500 Hz waveform is PTB-XL's vendor 100 Hz downsample; the unified interface is reachable from a single common 500 Hz native rate by **downsample-only** (no cohort is upsampled). Physical window = first 10.0 s (§5.3). **[EDITOR — RC-5]** Native PTB-XL patient-linkage uses 21,799/18,869; Challenge-file reads use 21,837.

### 5.2 Unified sampling rate — NO-OUTCOME rule (RULE C87-RATE)
```text
RULE C87-RATE (resolve r* in {100,250,500} Hz before any candidate is trained; writes the interface spec hash)
  (R1 no-synthesis)     500 mod r == 0 AND r <= 500 -> obtain r from each cohort's 500 Hz native by ONE fixed
                        zero-phase FIR anti-alias low-pass (cutoff 0.9*(r/2)) + integer decimation (500/r).
                        No cohort is upsampled/interpolated. PTB-XL is decimated from its 500 Hz file; its
                        vendor 100 Hz file is NOT used (avoids a mixed-anti-alias confound). {100,250,500} all pass R1.
  (R2 signal-integrity) [v2, minor] On a LABEL-FREE SOURCE-cohort waveform sample ONLY (not target
                        waveforms — keeps the frozen interface independent of target signal properties),
                        measure per r: (a) per-lead spectral-energy fraction below r-Nyquist vs 500 Hz
                        reference; (b) QRS morphology fidelity (R-peak-aligned median-beat correlation);
                        (c) diagnostic-band energy preservation up to f_diag (AHA ~0.05-150 Hz). r admissible
                        iff (a),(b),(c) each >= pre-registered thresholds. Since ALL six cohorts are native
                        500 Hz with identical downsample-only decimation, r* is invariant to whether source
                        or target waveforms are measured; source-only is used to avoid any target dependence.
  (R3 resource)         Full 648-model zoo training + 3-cohort field inference + frozen field storage at r must
                        fit the compute/storage envelope; audit records CPU/GPU-hours and bytes.
  Decision:             among r passing R1 AND R2 AND R3, pick the SMALLEST r; if it fails R2's diagnostic-band
                        threshold, pick the smallest admissible r that clears R2; ties -> larger bandwidth.
                        r* is content-hashed into the interface spec BEFORE training. The audit NEVER reads a model result.
```

### 5.3 Fixed windowing, normalization & QC (structurally leakage-free)
```text
INTERFACE.PREPROCESS (frozen; identical to source and target)
  1. Decode WFDB integer samples to millivolts via per-record adc_gain + baseline; fail-closed (C87-E) if
     gain/units missing or non-mV.
  2. Anti-alias + integer-decimate from 500 Hz native to r* (RULE C87-RATE), one fixed filter.
  3. Window = first 10.0 s (= 10*r* samples) from t0: >10 s -> crop; =10 s -> as-is; <10 s -> right zero-pad
     (affects Georgia 5-10 s).
  4. Per-record, per-lead z-score: subtract window mean, divide by window std (ddof=0), std floor 1e-6 mV;
     lead with std < eps -> all-zeros. Fully LOCAL to one record x lead: no fitted params, no cross-record/
     view/cohort statistics -> structurally leakage-free across the acquisition/held split.
  5. Fixed hard clip to [-20,+20] (post-z-score) to bound artifact spikes.
  Output tensor: [12, 10*r*], float32, lead axis in INTERFACE.LEAD_ORDER.
QC (NO-OUTCOME; header/waveform statistics only, never labels/outputs):
  exclude + reason-code (C87-E) records failing header decode, missing any of the 12 leads, flat across > K
  leads, or below the native-duration floor. K and the floor are finalized in the signal-integrity audit (§5.2).
```
Per-record-local normalization ⇒ **zero fitted parameters** ⇒ no source→target and no acquisition→held leakage surface.

### 5.4 Primary candidate zoo — architecture-CONTROLLED, content-addressed
One **shared 1D ECG backbone** (fixed 1D residual CNN family, `in_chans=12`, multi-label head over the registered scored SNOMED set; concrete depth/width/kernel pinned by a NO-OUTCOME reference-impl rule, spec-hash frozen BEFORE any weight is trained, never tuned). **All 648 candidates share this one architecture** — only objective, panel, seed, support level, checkpoint epoch differ, so any Measurement→Control effect is attributable to modality, not architecture family.
```text
ZOO.TOPOLOGY (frozen)
  Contexts (source-side only; 8) = panel x seed x support_level:
    panel in {A,B} (2 hash-locked source-patient train/audit partitions, §5.6); seed in {s0,s1};
    support_level in {0,1} (0 = full source support; 1 = one registered (source-cohort, SNOMED-class) cell
    deleted from source-train (OACI missing-cell intervention); target/held views byte-invariant).
    => 2 x 2 x 2 = 8 contexts.
  Candidates per context (81): ERM 1 (index 0); OACI 40 (index 1..40, epochs ascending);
    SRC 40 (index 41..80, epochs ascending). Cadence (OACI,SRC) = range(4,200,5) = {4,9,...,199} = 40 checkpoints.
  Total zoo = 8 x 81 = 648 content-addressed models.
  The candidate set A over which a*C_e, a*H_e, and â_{pi,B,e} are defined is the FULL 648-model zoo per target
  cohort e; the 8-way factorization is training PROVENANCE in each candidate ID, not a restriction on A.
  (Differs from EEG c86 where selection was within a target-subject context; here the deployment unit is the
  target COHORT and candidates pool across the 8 source contexts.)
```
```text
CANDIDATE_ID = c87_candidate_id(...) = SHA-256 over canonical-JSON:
  { "namespace":"C87_ECG_12LEAD_V1",   # disjoint from all EEG namespaces
    "interface_id":<leads|r*|preprocess|window hash>, "training_manifest_hash":<optimizer,epochs,cadence,...>,
    "panel":"A"|"B", "seed":s0|s1, "support_level":0|1, "regime":"ERM"|"OACI"|"SRC",
    "canonical_epoch":canonical_epoch(order), "weight_state_hash":<state_hash> }
  state_hash = SHA-256 over model_state = params + ALL buffers (incl BN running_mean/var/num_batches_tracked),
    binding keylen|key|dtype|ndim|shape|bytes; DEVICE-INDEPENDENT. Weights stored as <weight_state_hash>.pt.
  Canonical order: ERM(0), OACI(1..40 asc epoch), SRC(41..80 asc epoch) -> canonical_candidate_ids[0..80].
VALIDATOR (fail-closed): recompute every ID; require candidate_ids_by_context in genealogy order to equal
  canonical_candidate_ids for all 81. Any permutation, epoch swap, or content mismatch -> C87-E. Exactly 1 ERM,
  40 unique OACI, 40 unique SRC per context.
```

### 5.5 OACI / SRC / ERM objectives — held identical to the EEG engine
```text
ENGINE (unchanged wrappers; modality touches only input adapter + backbone + task-risk term)
  ERM  (Stage-1): minimize source-train task risk -> 1 ERM checkpoint/context; yields R_ERM_hat and the
                  risk-feasibility budget tau = R_ERM_hat + eps.
  OACI (Stage-2): "Overlap-Aware Conditional Invariance" — adversarial min of extractable conditional-domain
                  leakage L_Q^ov on ESTIMATOR-ELIGIBLE (support >= m) (domain,class) cells ONLY, s.t.
                  R_src <= R_ERM_hat + eps; primal-dual; lambda = dual multiplier of the RISK constraint
                  (not a leakage weight); per-class conditional domain critic; ineligible cells excluded, never smoothed.
  SRC  (Stage-2): "Source-Robust" — NON-adversarial min of a source-side worst-domain (log-sum-exp over D0) risk
                  under the SAME risk-feasibility constraint; selected by source-guard.
  Source domains D0 = {PTB-XL, CPSC2018, CPSC-Extra} (3). Source-audit (panel split §5.6) supplies the
  source-guard; the target is NEVER used by the engine.
```
**Forced, outcome-independent modality adaptation (surfaced, not silently frozen):** EEG was single-label 2-class (softmax CE); ECG is **multi-label SNOMED**, so the inner task-risk term becomes **per-class BCE** and the OACI (domain,class) cell "class" axis is redefined over SNOMED codes. The OACI/SRC **wrappers** (risk-feasibility constraint, adversary structure, worst-domain LSE, cadence, primal-dual, λ semantics) are unchanged; the (domain,class)-cell redefinition and the exact scored-class set are owned by §2/§4 (Rule L; see R-1).

### 5.6 Patient-level split — acquisition view vs held-evaluation view
```text
SPLIT (applied independently to each TARGET cohort)
  ACQUISITION view (construction / L^C): finite target labels MAY be queried; 1 query = 1 record's SNOMED label.
  HELD-EVALUATION view (deployment / L^H): SEALED; opened ONLY AFTER the selection freeze; never queried during acquisition.
  Deterministic, outcome-free assignment:
    key(patient) = SHA-256("C87_TARGET_SPLIT_V1" | cohort_id | patient_id)
    sort patients ascending; lower floor(n_pat/2) -> ACQUISITION, remainder -> HELD.  (mirrors c86 floor(n/2))
  Patient-ID resolution (METADATA AUDIT, NO-OUTCOME; before any split materializes):
    - cohort exposes patient IDs -> group all records by patient.
    - cohort does NOT expose patient IDs -> record == patient-unit, DISCLOSED per cohort.
      [EDITOR — RC-4: Chapman-Shaoxing & Ningbo are source-documented one-ECG-per-patient (G-b); Georgia has no
       ID field -> record == patient unless the audit finds one. No default is assumed silently.]
  Min-support gate (NO-OUTCOME) [v2/M4]: each view >= N_min=50 patients and >= m_class=20 records per registered
    scored SNOMED class; a cohort/class failing -> C87-E (NO re-split, NO threshold change). Values PINNED here.
SOURCE panels (define the zoo contexts of §5.4):
  Panel P in {A,B} partitions SOURCE patients into source-train / source-audit by
    key(patient) = SHA-256("C87_SOURCE_PANEL_" | P | "_V1" | source_cohort | patient_id) at a fixed audit fraction.
    A and B use DIFFERENT salts -> two distinct partitions (source-partition-sensitivity axis). Source-audit feeds
    the SRC/OACI source-guard; it is never the target.
```

### 5.7 Architecture-DIVERSE zoo = SECONDARY stress test (deferred)
The primary zoo is architecture-CONTROLLED on purpose (isolates the modality effect). An architecture-DIVERSE zoo (multiple backbone families) is a SECONDARY, later stress test, **not launched with the primary campaign** (else "modality vs architecture family" confound), considered only after the primary result is frozen, under a separately authorized stage.

### 5.8 Leakage & NO-OUTCOME closure (this section)
Sampling rate, window, normalization, lead order, zoo cardinality/structure, candidate IDs, split rule are all fixed a priori or by audits reading only resources/waveforms/metadata. Normalization is per-record-local ⇒ no leakage surface. Held-evaluation view sealed until selection freeze; candidate IDs are content hashes; validator fail-closed.

---

## 6. Synthetic positive/negative controls & pipeline-validation gate

**Purpose.** Before C87E reads any real target outcome, the *exact same* selection/estimation/inference code is exercised on synthetic worlds with known ground truth. Three controls: **POS** (must *detect* success), **NEG** (must *not manufacture* success), **CALIB** (estimators unbiased + patient-cluster bootstrap at nominal coverage). All three must pass; a failure is an **engineering blocker, never a scientific result about ECG.** No control parameter depends on any real model outcome.

### 6.0 What is validated / why it is leak-free
The controls validate the machinery producing `a*C_e, a*H_e, T_e, R_{e,π,B}, G_{e,π,B}` and their patient-clustered CIs on data where answers are known a priori. They run on a **synthetic loss tensor** from frozen constants; they never touch PhysioNet fields, never train a model, never read a real target outcome. A real NULL is credible only if the pipeline provably can detect a planted positive (POS/CALIB) and provably does not invent one (NEG).

### 6.1 Shared synthetic generator (one simulator, three parameterizations) [v2/B4]
The generator emits, per (candidate a, record r), a full **PREDICTION object** together with a KNOWN true
label, and DERIVES the loss from (prediction, true label). This is mandatory: the loss tensor alone cannot
exercise the two genuinely label-adaptive PRIMARY methods — MODEL SELECTOR consumes **hard predictions** to
build consensus pseudo-labels and its best-model posterior, and CODA consumes **predicted categories** to
build Dawid–Skene confusion matrices. Driving losses AND predictions from one generative object lets the
SAME synthetic world exercise LURE/VMA/ASE (losses), MODEL SELECTOR (hard preds) and CODA (confusions), so
CONTROL_PASS actually certifies the label-adaptive path.
```text
GENERATIVE MODEL (frozen skeleton; per-world settings in 6.2/6.3/6.4)
  Patients p clustered; records r; true single-§2-task label  y(r) ~ Bernoulli(pi_pos)  [KNOWN].
  Per candidate a, view v, record r:  a LATENT competence  c_{v,a,r} = q_{v,a} + s_p + b_{p,a}
                                                                       + D_{p,r}*g_{v,a} + eta_{v,a,p,r}
    q_{v,a}   : candidate base competence (view-dependent; sets which candidate is truly best on each view)
    s_p ~ N(0,rho*sigma^2) ; b_{p,a} ~ N(0,kappa*sigma^2) ; eta ~ N(0,(1-rho-kappa)*sigma^2)  [cluster corr]
    D_{p,r} ~ Bernoulli(phi) : per-record "informative" indicator; competence gap g carried ONLY by D=1
  PREDICTION:  predicted prob  p_{v,a,r} = sigmoid( alpha_pred*(2*y(r)-1)*c_{v,a,r} + xi_{v,a,r} )  in (0,1)
               hard pred  yhat_{v,a,r} = 1[p_{v,a,r} > 0.5]  (feeds MODEL SELECTOR / CODA confusion)
  DERIVED LOSS: loss_{v,a,p,r} = binary NLL( p_{v,a,r}, y(r) )   (feeds LURE / VMA / ASE)
  Ground truth (KNOWN, never revealed to the pipeline):
    L^v_e(a)=E_r[loss] ; a*C_e/a*H_e via held CROSS-FIT of these ; T_e = L^H(a*C_e)-L^H(a*H_e).
  The pipeline sees, for a QUERIED record r only (one label = one record): the true y(r) AND every
    candidate's (p_{H,a,r}, yhat_{H,a,r}) on that record; plus the cost-free acquisition-view proxy a*C_e.
    It NEVER sees un-queried held labels/losses. Competence structure {q,g} is planted per world (6.2/6.3).
```
Structural params matching C87E (via 6.5, not any outcome): candidate cardinality `A`, cohorts `E`, budget grid, patients-per-cohort, records-per-patient. Effect/correlation constants (`phi, rho, kappa`, separations, transport magnitudes) are frozen a priori. `phi` creates genuine label-efficiency structure (only D=1 records distinguish candidates); equal `mu_{H,a}` removes it (null); `rho, kappa` make clustering matter.

### 6.2 POS — recover positive transport-consistency AND positive active gain
```text
POS WORLD (planted via candidate competence {q_{v,a}, g_{v,a}}: FULL transports AND a label-adaptive
policy genuinely beats P0 — the plant is in the PREDICTION/confusion structure, not just the loss)
  Transport:  Spearman(q_C,q_H) >= 0.9 (same argmin, top-k>=3 order) => a*C_e = a*H_e, T_e = 0 in ALL E cohorts.
  Separation: L^H_e(a*H_e) is SEP_POS=0.05 below the runner-up.
  Label-adaptive plant: the true-best candidate's hard predictions yhat agree with y on D=1 records while
    near-optimal rivals disagree there (a confusion structure MODEL SELECTOR's posterior / CODA's Dawid–Skene
    can exploit); phi=0.15, rho>0, kappa>0 => an info/EIG-seeking adaptive policy identifies a*H_e faster than P0.
  SEP_POS, g and phi set so POS active gain clears the gate with >=80% MC power at the REAL held n and frozen B_grid.
POS pass (all):
  1. T-consistency recovered: patient-clustered 95% CI for T̂_e (cross-fit) covers 0 AND â_C_e == â_H_e in every E cohort.
  2. Active gain recovered — EXERCISING THE LABEL-ADAPTIVE PATH: some B in B_grid and some GENUINELY
     LABEL-ADAPTIVE pi (MODEL SELECTOR and, if primary, CODA — driven by the planted predictions/confusions,
     NOT a loss-only method) with LCB[G_{e,pi,B}] > 0 in ALL E cohorts (same pi, same B, no pooling = real gate).
  3. Power floor: over K>=1000 POS redraws, criterion 2 met in >=80%; report the gate MDE as a design output.
```
```text
POS-DENSE WORLD (6.2b) [v2/M1 — stresses the cross-fit ceiling in the argmin-instability regime]
  Same transport plant, but a DENSE set of near-tied candidates around the best (no single well-separated
  best; competence gaps << patient-cluster SE) so naive argmin/argmax over 648 is unstable.
POS-DENSE pass:
  4. The CROSS-FIT held ceiling (SPEC 4.A) is UNBIASED — |mean_K(T̂_e^xfit) - 0| <= max(2*SE_MC, 0.003) — while
     the NAIVE in-sample argmin/argmax ceiling shows the expected upward T bias (documents winner's-curse removal).
  5. Patient-cluster BCa CIs for the cross-fit T_e, R, G attain coverage in [93.5%,96.5%] under argmin instability.
```

### 6.3 NEG — must not manufacture transport or active gain
```text
NEG-A WORLD (planted: FULL does NOT transport; held-view signal EXISTS)
  Nontransport: Spearman(mu_C,mu_H)<=0, a*C_e among the WORST third on held => T_e >= TAU_NEG=0.05 in ALL E cohorts.
  Held signal:  a*H_e well-separated (SEP_POS) so active methods CAN reduce held regret.
NEG-B WORLD (planted: NO information — no candidate truly better on held)
  Null: mu_{H,a} = mu_0 for ALL a; a*H_e by sampling noise only; rho>0, kappa>0 retained.
NEG pass (all):
  1. No false transport (NEG-A): patient-clustered 95% LCB[T̂_e] > 0 and â_C_e != â_H_e in every cohort.
  2. No spurious gain (NEG-B): for EVERY pi and EVERY B, NOT (LCB[G_{e,pi,B}] > 0 in all E cohorts). Over K>=1000
     redraws, family-wise rate of "same-pi, same-B, all-E-cohort G>0" <= alpha = 0.05.
  3. Sign discipline: |mean_K Ĝ| <= 2*SE_MC for label-adaptive policies in NEG-B.
```

### 6.4 CALIB — estimator unbiasedness & patient-cluster bootstrap coverage
```text
LURE (Levelled Unbiased Risk Estimator) — VERIFIED reference (Farquhar et al. 2021; Kossen et al. 2021)
  R_LURE = (1/M) * sum_{m=1..M} v_m * loss(i_m)     ;   v_m = 1 + ((N-M)/(N-m))*(1/((N-m+1)*q(i_m)) - 1)
  E[v_m] = 1 for all m,M,N,q => UNBIASED. Uniform proposal => every v_m = 1 => R_LURE == naive mean. M=N => v_m=1.
  For active model selection (Hara et al. 2024) the same weights give an unbiased loss-DIFFERENCE estimate; ideal q minimizes its variance.
```
```text
CALIB THRESHOLDS
  Exact identity: uniform-proposal LURE weights all == 1 (bit-exact) and R_LURE == naive mean to ||.||_inf <= 1e-9.
  Unbiasedness:   for EACH estimand and estimator, under BOTH proposals, |mean_K(estimate)-truth| <= max(2*SE_MC, 0.003).
  Coverage:       patient-cluster (resample PATIENTS w/ replacement, carry all their records) 95% bootstrap CIs
                  attain empirical coverage in [93.5%,96.5%] for T_e, R_{e,P0,B}, G_{e,pi,B} over K>=1000 cohorts.
                  Paired/common-random-number bootstrap across policies for G.
```
```text
Mutation tests (each MUST trigger its failure signature, else the control is non-diagnostic -> blocker):
  CALIB-M1  unweighted estimator under the ADAPTIVE proposal => MUST show |bias| > 2*SE_MC.
  CALIB-M2  record-level (NON-clustered) bootstrap under rho>0 => MUST under-cover (< 90% at nominal 95%).
  NEG-M3    oracle-leak variant (selector peeks at the FULL held loss field) => MUST produce spurious all-E G>0 in NEG-B.
```

### 6.5 NO-OUTCOME resolution of not-yet-pinnable choices
```text
  A_syn        := |candidate set the C87E selector ranges over| = FULL 648 zoo (per context: 1 ERM + 40 OACI + 40 SRC
                  = 81; contexts = 2 panels x 2 seeds x 2 support levels)  [frozen config].
  E_syn        := number of untouched target cohorts = 3 (Georgia, Chapman-Shaoxing, Ningbo)  [metadata].
  B_grid       := the §4 endpoints-section budget ladder {0,8,16,32,64,128,256,512}, reused verbatim  [frozen config].
                  [EDITOR — RC-2: corrected from "interface-section" to the §4 estimand ladder.]
  N_p^{(e)}    := held-view patient counts of the corresponding target cohort  [public metadata / headers].
  cluster-size := empirical records-per-patient distribution [metadata audit]; if a cohort is ~1 record/patient,
                  INJECT multi-record patients + rho>0 anyway in CALIB so the clustered bootstrap is stressed.
  per-record loss ell := additive per-record loss pinned by §2/§4 (see D-24/R-1); if the deployment metric is
                  non-additive (e.g., Challenge-2021 weighted score), controls validate the estimators for its
                  additive surrogate; the non-additivity is routed to the metric section.
  policy set   := the frozen §3 registry (P0 + Active Testing/LURE + MODEL SELECTOR; CODA-conditional); same set.
  design constants (frozen a priori): SEP_POS=0.05, TAU_NEG=0.05, phi=0.15, rho, kappa, K>=1000, alpha=0.05,
                  coverage band [93.5%,96.5%].
```

### 6.6 The gating rule (registered)
```text
GATING RULE — C87E may read real target fields IFF all controls pass
  PASS(controls) := POS(6.2 crit 1-3) AND NEG(6.3 crit 1-3) AND CALIB(6.4 all) AND all mutation tests
                    (CALIB-M1, CALIB-M2, NEG-M3) trigger their failure signatures.
  1. Controls run under the FROZEN commit, on the SAME modules C87E uses (selector, estimators, patient-cluster
     bootstrap, budget grid, A, E). A separate toy reimplementation is FORBIDDEN.
  2. Determinism/provenance: fixed seeds; bit-identical self-replay (sha256 of control outputs equal on re-run);
     record code_sig + synthetic-gen param hash + config; emit a sha256 manifest.
  3. On PASS: emit a signed CONTROL_PASS token. The C87E real-field stage checks this token as a HARD precondition
     and REFUSES to run without it.
  4. On FAIL or AMBIGUOUS (any statistic between pass/fail bands): ENGINEERING BLOCKER. Fix the pipeline and re-run.
     NEVER relax a threshold to pass. NEVER report a control outcome as an ECG finding. Ambiguous == FAIL.
  5. Power/MC discipline: K >= 1000 everywhere (a 100->1000 change has flipped a conclusion before).
```

### 6.7 Scope boundaries
The controls certify the pipeline can *detect* planted transport/gain, *refuses* to invent them, and produces unbiased, nominally-covered estimates. They do **not** predict the real ECG outcome, tune any real threshold, or substitute for the untouched-cohort gate. Passing controls is a **necessary engineering precondition**, not evidence about the Measurement→Control gap. The synthetic effect sizes are deliberately generous; the reported MDE (6.2-3) is the pipeline's sensitivity floor, not a claim about real magnitude.

---

## Appendix — Residual conflicts (flagged; not silently overridden)

- **R-1 (RESOLVED, freeze v2 / B2 / PM-RATIFY #1): selection-loss label axis** = PINNED to the single §2 binary task (multi-label-S NLL = secondary robustness variant; §5 still trains a multi-label head). D-24 deferral deleted. If the PM overrides to (b) multi-label S, §3 audit check E and the multi-label→pointwise reductions (D-12) become load-bearing and the scored-class count (26 vs 30, D-05) must be pinned — hence the ratification flag.
- **R-2 (residual sub-flag): support-level-1 deletion cell vs the selected task.** §5's OACI missing-cell intervention deletes one (source-cohort, SNOMED-class) cell in support-level-1 contexts (D-17). The metadata rule must ensure this deletion does not remove the finally-selected task's positive class from its *sole* supporting source cohort (e.g., deleting SB from PTB-XL when PTB-XL is the only SB-bearing source), which would silently break trainability of T3 in those 4 contexts. Routed to the metadata audit; flag preserved.
- **Minor bookkeeping (reconciled but recorded):** PTB-XL patient count 18,885 (canonical, §1/§2/§5) vs 18,869 (native v1.0.3, §4); scored-class count 26 (§4) vs 30 (§3/§6); "XWED" estimator label unverified (ASE verified). All source-side or metadata-audit-resolvable; none affects a target endpoint.

*(Web verification used to resolve R-4/RC-4 and the 45,152 identity: PhysioNet `ecg-arrhythmia` v1.0.0 database page and the Zheng et al. Scientific Data descriptor — both Chapman-Shaoxing and Ningbo documented as one 10-s 12-lead ECG per patient; combined 45,152 = 10,247 + 34,905.)*