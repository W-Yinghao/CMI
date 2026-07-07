# Experiment Table

Every experiment is either an **exact certificate** (a proof object) or an **audited real-EEG grid**
(an illustration of the audit discipline, not a proof and not a SOTA claim). Values are copied from
tracked summaries; raw run dirs are gitignored under `results/`.

| Artifact | Dataset / regime | Runs | Metric payload | Identifiability status | Claim boundary | Tracked file |
|---|---|---|---|---|---|---|
| **Tier 0 — exact certificates** | synthetic exact worlds (R0/R1/R2) | 10 certificates | binary G/B worlds; rank/anchor/support flags | **proves non-identifiability** (OA-0 certificate pattern) | proof layer; asserts + exits 0 | `counterexamples/run_counterexamples.py` |
| **Tier 1 — simulator illustrations** | EEG-shaped simulator | illustrative | mirrors CE-R1-1 on realistic arrays | illustration, **not** a proof | not evidence for a theorem | `07_counterexample_catalog.md §6` |
| **Step 8 — BNCI2014_001 mini-grid** | BNCI2014_001, 4-class, LOSO (R0/R1 eval) | 9 ok | raw strict bAcc 0.3383, offline gain −0.0926, online 0.3206 | target metrics oracle/eval-only, `identifiable=null` | audited interface + boundary; not SOTA | `results_summaries/step8_bnci2014_001_minigrid_summary.{json,md}` |
| **Step 9 — BNCI2014_001 expanded** | BNCI2014_001, 4-class, 9×9×3=27 | 27 ok | strict bAcc **0.3946**, offline gain −0.0502, online 0.3753, excess-norm 0.1928, harm-rate **0.8148** | oracle/eval-only, `identifiable=null`, prior `rejected_conclusion_false` | stability + boundary; not SOTA | `results_summaries/step9_bnci2014_001_expanded_summary.{json,md}` |
| **Step 10 — BNCI2014_004 (binary)** | BNCI2014_004, 2-class, all×all×3=27 | 27 ok | raw strict bAcc **0.6282**, excess-norm **0.2563**, offline gain-norm −0.0639, harm-rate **0.8519** | oracle/eval-only, `identifiable=null` | cross-dataset audit; not SOTA | `results_summaries/step10_bnci2014_004_summary.{json,md}` |
| **Step 10 — BNCI2015_001 (legal skip)** | BNCI2015_001; invalid for LeftRightImagery paradigm | 0 ok / 36 skip | none | **not_applicable_all_skipped** (no ok runs → boundary flags null) | legal skip, `missing_cells=[]`; asserts nothing | `results_summaries/step10_bnci2015_001_summary.{json,md}` |
| **Step 10 — combined MOABB digest** | BNCI2014_001 + BNCI2014_004 (+ BNCI2015_001 skipped) | 54 ok | overall strict excess-norm **0.2246**, offline gain-norm −0.0654, harm-rate **0.8333** | raw bAcc pooling **refused** across K; normalized only | mixed-K audited digest; not SOTA | `results_summaries/step10_moabb_multidataset_summary.{json,md}` |

## Reading notes

- **Raw bAcc is not comparable across K.** BNCI2014_004's raw 0.6282 (binary, chance 0.5) is not
  "better" than BNCI2014_001's 0.3946 (4-class, chance 0.25); chance-normalized excess (0.2563 vs
  0.1928) is the comparable number, and the combiner refuses to pool raw bAcc.
- **TTA fragility.** Offline-TTA harm-rate (`P(gain<0)`) is ≈ 0.81–0.85 per dataset and **0.8333**
  pooled over 54 ok runs — reported as fragility evidence under the audit, never as a theorem or a
  performance claim.
- **All-skip is not a failure.** BNCI2015_001 is a legal skip (the dataset is not a left/right-hand task
  so the binary `LeftRightImagery` paradigm rejects it); its boundary flags are `null`
  (not_applicable), and it is excluded from the combined boundary roll-up while still counted as valid.
