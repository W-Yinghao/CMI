## 1. Data & Cohort Metadata Audit

**Data release (frozen):** George B. Moody PhysioNet/Computing in Cardiology Challenge 2021 — *"Will Two Do? Varying Dimensions in Electrocardiography"* — public training release **v1.0.3**. All redistributed 12-lead ECG databases carry the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license and are **publicly downloadable without credentialing** (access already verified). Labels for every database are provided as **SNOMED-CT** codes via the Challenge `ConditionNames_SNOMED-CT` mapping. All signals are distributed in **WFDB** format (MATLAB v4 `.mat` signal + `.hea` header; header carries Age and Sex only — see patient-identity column).

### 1.1 Registered cohort table (public metadata only — no model outcome used)

The table registers the **Challenge-2021 training-split record counts** (the counts the PI listed), which are the operative sizes for this program. Where the raw source database differs, that is flagged in §1.4.

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

Two Challenge-2021 databases (**INCART**, **PTB**) are **explicitly excluded** by the design's source/target split. They are listed for completeness and because INCART's **257 Hz / 30-min** and PTB's **1000 Hz / up-to-120 s** properties would break the sampling-rate and duration homogeneity that the six design cohorts otherwise share (all six are native **500 Hz, 12-lead**).

### 1.2 Public accessibility & license (confirmed)

- All eight databases are redistributed under **CC BY 4.0** in the Challenge 2021 release; the aggregate resource and each constituent (PTB-XL; the combined Chapman/Ningbo "ecg-arrhythmia" database) are CC BY 4.0 on PhysioNet.
- The **training split is fully public**; only the challenge validation/test partitions were withheld. This design uses **only publicly released training records**, so no withheld partition is required.

### 1.3 Patient-identity characterization (critical — statistical unit is PATIENT)

The design's unit is the **patient**, with a patient-level acquisition/held split (no patient's records straddle both views). Identity availability differs by cohort and is the governing constraint:

- **Chapman-Shaoxing** and **Ningbo**: each subject contributed **exactly one 10-s ECG** (the source database is "one ECG per patient"). Therefore **record-level = patient-level**: any record-level partition of these two cohorts is automatically a valid patient-level split, because no patient can appear twice. ✔ No linkage needed.
- **Georgia (Emory)**: the distributed WFDB headers carry **Age and Sex only — no patient identifier field**. Patient identity is therefore **per-record / unavailable**. The Challenge organizers state cross-partition patient overlap is "vanishingly small," but no field exists to enforce or verify a within-cohort patient-level split. This is a real gap → resolved by a NO-OUTCOME rule (§1.5, rule G).
- **PTB-XL** (source): genuine `patient_id` (**21,837 records / 18,885 patients**); if ever promoted to a target it must be split by `patient_id`.
- **CPSC / CPSC-Extra** (source): per-record only, no patient-id field.

### 1.4 Discrepancy flags vs PI-stated figures

**All six PI figures reproduce the canonical Challenge-2021 training-split counts exactly** (CPSC 6,877; CPSC-Extra 3,453; PTB-XL 21,837; Georgia 10,344; Chapman-Shaoxing 10,247; Ningbo 34,905). No figure is contradicted. Two provenance caveats must nonetheless be recorded so the frozen report does not later mis-cite source-paper sizes:

1. **Challenge subset ≠ raw source-database size** for the target cohorts. The PI figures are challenge *training-split* records, not full source-database sizes:
   - Georgia: challenge training = 10,344, but the full Emory/Georgia source (train+val+test) is ~20,672 records.
   - Chapman-Shaoxing: challenge = 10,247, but the **raw Zheng et al. (2020) Scientific Data** source reports **10,646 patients** (399 more; challenge subset dropped records, presumably QC).
   - PTB-XL 21,837 happens to equal the raw PTB-XL size; CPSC 6,877 equals the raw CPSC2018 training set.
2. **The two "targets" reconstitute a single database.** Challenge Chapman-Shaoxing (10,247) **+** Ningbo (34,905) **= 45,152**, which is **exactly** the size of the single combined **Zheng et al. "A large scale 12-lead ECG database for arrhythmia study"** (PhysioNet `ecg-arrhythmia` v1.0.0, 45,152 records). The challenge did not import two independently published datasets — it **split one combined Zheng database back into its two contributing-hospital cohorts.** This is the core of the independence caveat (§1.6).

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

### 1.6 Independence caveat — do NOT claim "three fully independent cohorts"

The three untouched targets span **two — not three — independent data-collection / labeling lineages**:

- **Lineage A (independent):** **Georgia / Emory University** (Southeastern USA). Distinct institution, country, acquisition program, and annotation.
- **Lineage B (shared):** **Chapman-Shaoxing** and **Ningbo** are **NOT independent data-collection projects.** They are **two contributing-hospital cohorts of a single curation project** run by the same group (Jianwei Zheng et al., **Chapman University**, with Shaoxing People's Hospital and Ningbo First Hospital), released as one combined PhysioNet database (`ecg-arrhythmia`, 45,152 = 10,247 + 34,905). They share: (i) the **same SNOMED-CT labeling pipeline** and `ConditionNames_SNOMED-CT` mapping; (ii) the **same physician-adjudication annotation protocol** (primary physician label → secondary physician validation → senior-physician adjudication on disagreement); (iii) **identical acquisition standardization** (10-s, 500 Hz, 12-lead, one ECG per patient); (iv) **same geographic region** (both hospitals in **Zhejiang Province, China**). They differ only in **contributing hospital** and collection period.

**Consequence for the formal gate.** The registered gate ("same method + same budget must replicate in ALL untouched cohorts; patient-level clustered inference; NO pooled cross-dataset p-value") remains as stated, but its interpretation is bounded: replication across **Chapman-Shaoxing and Ningbo tests cross-HOSPITAL transport WITHIN one curation lineage**, not cross-project generalization. Only **Georgia vs {Chapman-Shaoxing, Ningbo}** constitutes a genuinely cross-lineage replication. The final report must phrase this as **"three untouched target cohorts across two independent curation/labeling lineages (Emory/USA; and a shared Zheng/Chapman-University Zhejiang-Province lineage covering Chapman-Shaoxing and Ningbo)"** and must never assert three fully independent cohorts.

### 1.7 Open choices deferred to NO-OUTCOME rules

The following are not yet pinnable from metadata alone; each is registered as an outcome-independent rule (resource / signal-integrity / metadata audit), never a value chosen from a model result:

- **Rule W (fixed-length windowing):** duration is heterogeneous (CPSC 6–60 s; Georgia 5–10 s; others 10 s). Pin a single fixed window (crop / zero-pad to **10 s at 500 Hz = 5,000 samples**) applied uniformly to all cohorts; the window length is fixed from the **duration-metadata audit**, before any model is fit.
- **Rule R (resampling):** all six design cohorts are native **500 Hz**; register 500 Hz as the common rate; any record whose header declares a different rate is resampled to 500 Hz. Header-metadata driven only.
- **Rule G (Georgia patient unit):** because **no patient-id field is distributed for Georgia**, register **"one record = one patient-unit"** for Georgia; a pre-analysis **header-integrity audit** checks for any patient-id field, and only if one is found does the split switch to it. No model outcome enters this choice.
- **Rule L (scored label set):** the SNOMED-CT label set to be scored is fixed by a **per-class support metadata audit** — restrict to SNOMED codes with adequate positive support present in the source panel and in each target cohort — using **support counts only**, never any classifier performance.
- **Rule Q (record QC):** drop only records failing **WFDB signal-integrity / header-integrity** checks (corrupt file, wrong lead count, flatline), by a pre-registered signal-integrity rule; no performance-based exclusion.