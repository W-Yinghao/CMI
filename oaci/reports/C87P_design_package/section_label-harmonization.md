## 2. Label semantics & task-selection protocol (metadata-only)

### 2.0 Scope and invariant

This section freezes **how the single binary task used in C87E is chosen**, using *only* label metadata (SNOMED-CT code definitions, per-class label counts, cohort/patient documentation). No model output, loss, accuracy, or held-view target label content is read at any point of this rule. The candidate set, the code-sets, the admissibility gate, the ranking metric, the tie-break chain, and all numeric constants are frozen here. Only the *numeric counts* are (re)read in C87E from the downloaded v1.0.3 header files; the decision procedure that turns those counts into a task is fixed now.

Invariant (red-team target): **the frozen task never depends on which task yields a positive result.** The selection order is exactly, and only, `semantic consistency across cohorts → support → ambiguity/dedup → mechanical tie-break`.

### 2.1 Data provenance and label system

- Corpus: PhysioNet/CinC Challenge 2021 **v1.0.3** public training set (CC BY 4.0). Labels are SNOMED-CT codes carried in the WFDB `.hea` header comment lines (`#Age`, `#Sex`, `#Dx`, `#Rx`, `#Hx`, `#Sx`). Each record's `Dx` field is a **set** of SNOMED codes (multi-label). [challenge-2021 v1.0.3]
- Source-training cohorts (models trained here): PTB-XL (21,837 records), CPSC2018 (6,877), CPSC-Extra (3,453). [challenge-2021 v1.0.3]
- **Untouched target cohorts** (deployment/held view; frozen before any model result is read): **Georgia (10,344 records), Chapman-Shaoxing (10,247), Ningbo (34,905)**; all 500 Hz; Chapman/Ningbo 10 s, Georgia 5-10 s. [challenge-2021 v1.0.3]
- Patient linkage: the Challenge headers expose **no patient-identifier field** (only Age/Sex/Dx/Rx/Hx/Sx). The Chapman-Shaoxing + Ningbo source database (PhysioNet `ecg-arrhythmia` v1.0.0) is documented as **one 12-lead ECG per subject, cross-sectional** (45,152 subjects = 10,247 + 34,905). Georgia records likewise carry no within-cohort patient ID in the Challenge format. Consequence for the three target cohorts: **record = patient** unless the C87E metadata audit surfaces an explicit ID field (rule 2.6). [ecg-arrhythmia v1.0.0]

### 2.2 Pre-registered candidate task set (three slots, frozen code-sets)

Exactly three candidate tasks. The **negative class is always Sinus Rhythm (NSR), SNOMED `426783006`**, shared by all three. Positive code-sets are fixed here; within-slot variants (BBB, rate) are resolved by the metadata rule of 2.7, not by any result.

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

**Documented per-cohort support** (evaluation-2021 `dx_mapping_scored.csv`; **record** counts — for the three target cohorts these equal patient counts, and are recomputed at patient level from downloaded headers in C87E):

```text
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

### 2.3 Semantic-consistency requirement (the admissibility gate)

A candidate is **admissible** only if its positive code-set `P` denotes the **same clinical concept in every cohort** and both classes are present above the floor in every target cohort *and* in the source pool.

1. **Fixed code-set, no per-cohort widening.** `P` is identical across all cohorts. It is forbidden to add cohort-specific codes to rescue support.
2. **Allowed unions** = codes that are is-a / subtype of one concept, or declared equivalent by the Challenge. RBBB∪CRBBB is allowed because CRBBB is complete RBBB and the Challenge scored `713427006`≡`59118001` as equivalent (and deliberately did **not** relabel data across institutions, so cross-cohort coding really does differ). [challenge-2020 paper]
3. **Forbidden unions** = merging distinct sibling concepts to manufacture support. Atrial **flutter** (`164890007`) is **never** merged into AF (`164889003`). This is decisive: Ningbo carries 0 AF but 7,615 AFL; the rule refuses to redefine "AF" as "AF-or-flutter," so the missing AF support in Ningbo is **not** rescued.
4. **Zeros are disqualifying, structural or sampling alike.** A per-class count of 0 (or `< τ_pat`) in any target cohort — whether the concept is absent from that cohort's coding scheme (a "structural zero," e.g. CPSC2018 never labels SB/STach/SA) or merely rare — makes the candidate **inadmissible**; we never distinguish the two by rescuing with a sibling code.

Presence test — a candidate passes the gate iff, using the frozen `P` and `N={426783006}`:
- for **each** target cohort e ∈ {Georgia, Chapman-Shaoxing, Ningbo}: `n⁺_e ≥ τ_pat` **and** `n⁻_e ≥ τ_pat` (patient-level, post-dedup, post-ambiguity, §2.5-2.6); **and**
- for **each** registered source panel (compositions defined in the candidate-zoo section): both classes `≥ τ_src` so the zoo is trainable there.

### 2.4 Metadata selection inputs

The only inputs to selection are, per candidate and per cohort:
- `n⁺_e`, `n⁻_e` = patient-level positive / negative support after de-duplication (§2.6) and ambiguity exclusion (§2.5);
- `amb_e` = number of patients dropped for carrying both `P` and `N`;
- source-pool / per-panel class presence for the training-feasibility check.

No performance quantity (loss, accuracy, calibration, transport gap, regret) enters. Support magnitudes are used **only** to gate and rank presence, never as a proxy for which task "works."

### 2.5 Multi-label ambiguity handling (records carrying both classes)

For positive set `P`, negative set `N = {426783006}`, and a record's code set `Dx`:

```text
pos(r) := Dx(r) ∩ P ≠ ∅
neg(r) := Dx(r) ∩ N ≠ ∅
  AMBIGUOUS  (drop, count into amb_e):  pos(r) ∧ neg(r)
  POSITIVE example:                     pos(r) ∧ ¬neg(r)
  NEGATIVE example:                     neg(r) ∧ ¬pos(r)
  NEITHER (ignored, not in the task):   ¬pos(r) ∧ ¬neg(r)
```

Positive and negative example sets are therefore disjoint by construction. Records carrying both the positive concept and sinus rhythm are excluded, not arbitrated. (Reported-only robustness variant, **not** used for selection: a stricter negative requiring NSR-exclusive labels `Dx == {426783006}`.)

### 2.6 Patient-level de-duplication

Statistical unit = **patient**; all of a patient's records fall in one acquisition/held view (per the deployment-object spec).

```text
Grouping key K(r):
  if the downloaded v1.0.3 metadata for the cohort exposes a patient/subject-ID field
       (C87E audit, rule OD-2):  K(r) = that ID
  else:                          K(r) = record ID           (record = patient)
Patient-level label:
  a patient is POSITIVE if ≥1 of its in-task records is positive and none is ambiguous/negative;
  NEGATIVE symmetrically; a patient with intra-patient class disagreement is DROPPED.
  (For the three target cohorts, documented one-ECG-per-patient ⇒ this is a no-op and K = record ID.)
```

For the **source** side, PTB-XL has multiple records per patient (21,837 records / 18,885 patients); source-side patient grouping uses the external `ptbxl_database.csv` `patient_id` and is handled in the candidate-zoo/split section — cross-referenced here, out of scope for target-task selection.

### 2.7 The frozen selection algorithm

```text
INPUT: patient-level counts {n⁺_e, n⁻_e, amb_e} for each candidate (with its within-slot
       variants) over target cohorts e∈{GEO,CHA,NIN}, plus source-panel class presence.
CONSTANTS (frozen, NO-OUTCOME; not tuned to any result):
   τ_pat = 100   patients per class per target cohort (estimability floor)
   τ_src = 50    patients per class per registered source panel (trainability floor)
   δ     = 0.10  relative-support margin that triggers the ambiguity tie-break

STEP A — resolve within-slot variants (metadata only):
   for slot T2 (BBB) consider {RBBB-union, LBBB}; for slot T3 (RATE) consider {SB, STach}.
   keep, per slot, the variant(s) that PASS the §2.3 gate; if >1 passes, keep the one with the
   larger maximin support S (below). This fixes one code-set per slot.

STEP B — semantic-consistency / presence gate (§2.3):
   drop every candidate failing the gate in ANY target cohort or ANY source panel.
   ADMISSIBLE = survivors.  If ADMISSIBLE = ∅ → DEV_STOP (no metadata-valid task; do NOT relax).

STEP C — support ranking (maximin over the weakest target cohort):
   S(cand) = min_{e∈{GEO,CHA,NIN}} min(n⁺_e , n⁻_e)          # higher is better
   winner = argmax_cand S(cand)

STEP D — ambiguity/dedup tie-break (only if top-two S within δ):
   A(cand) = max_e  amb_e / (n⁺_e + n⁻_e + amb_e)             # lower is better
   prefer smaller A.

STEP E — mechanical tie-break (guarantees uniqueness, outcome-free):
   prefer the candidate whose positive set P has the smallest SNOMED-CT code min(P).

OUTPUT: exactly one frozen binary task (P, N=426783006) with its example sets per target cohort.
```

The order is exactly the mandated chain: consistency (B) → support (C) → ambiguity/dedup (D) → deterministic tie-break (E).

### 2.8 Exact computation executed in C87E (pre-registered; metadata only)

```text
for each cohort in {source panels} ∪ {GEO, CHA, NIN}:
    parse every .hea header  → (record_id, Dx_set)            # signals never read
    K ← patient-ID field if audit finds one, else record_id   # rule OD-2
for each candidate task (P, N):
    build pos/neg/ambiguous per §2.5 ; collapse to patients per §2.6
    compute n⁺_e, n⁻_e, amb_e for each target cohort ; class presence per source panel
run STEP A..E of §2.7  →  frozen task
emit a signed manifest: {selected P, N, per-cohort n⁺/n⁻/amb, S, A, gate pass/fail per candidate}
This manifest is produced and committed BEFORE any candidate model is scored on any target cohort.
```

The manifest is the auditable object proving the choice used only counts.

### 2.9 Expected (non-binding) outcome under currently-documented metadata

Shown only to demonstrate the rule is well-posed; the **binding** values are the C87E recomputation from downloaded headers.
- **T1 (AF)** — expected **inadmissible**: AF Ningbo support = 0 < τ_pat; rule refuses AFL rescue.
- **T2 (BBB)** — within-slot resolves to **RBBB-union** (LBBB Ningbo = 35 < τ_pat fails); RBBB-union clears the floor in all three (GEO 570 / CHA 454 / NIN 1291) → **admissible**, S = min(454, 570, 1291, and NSR≥1752) = 454.
- **T3 (RATE)** — within-slot resolves to **SB** (largest support); SB clears all three (GEO 1677 / CHA 3889 / NIN 12670) → **admissible**, S = 1677. **Caveat:** SB (and STach) have 0 support in CPSC2018 and only 45 (303) in CPSC-Extra, so T3's admissibility is **conditional on every registered source panel containing PTB-XL** (or another SB-labeled source); if any source panel is CPSC-only, T3 is dropped at the gate. **Flagged to the candidate-zoo section.**
- Under documented metadata the winner would be **T3 SB-vs-NSR** (S=1677 > 454) if all source panels are PTB-XL-inclusive; otherwise **T2 RBBB-union-vs-NSR**. Either way **AF-vs-NSR is out** — decided by cross-cohort label metadata, never by a result.

### 2.10 Cross-cohort semantic-consistency summary

The one negative class (NSR `426783006`) and each frozen positive concept are verified present with the *same* code-set in all three untouched cohorts (2.2 table), with two intended disqualifications that the consistency rule is designed to catch: AF's Ningbo structural zero and SA's Chapman zero. Sampling rate (500 Hz) is consistent across the three target cohorts; Georgia's 5-10 s length vs 10 s elsewhere is an interface matter handed to the acquisition-interface section.