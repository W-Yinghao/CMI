# Table Plan

Planned manuscript tables. Each is assembled from an existing package file / tracked artifact.

## Table 1 — Theorem ledger
- OA-0 / MONO-1 / TOS-1 / TU-1 / TU-2 / MP-1 / PD-1 / ID-1 with regime, contracts, identifiable vs
  non-identifiable object, and proof/certificate file.
- Source: `03_theorem_table.md`.

## Table 2 — Contract taxonomy
- C1–C12 with role, R0/R1/R2 checkability, failure certificate, and the overclaim each blocks.
- Source: `04_contract_table.md`, `02_contract_taxonomy.md`.

## Table 3 — Counterexample catalog
- CE-R0-1/2/3, CE-R1-1/2, CE-C1-1, CE-MP-1, CE-C11-1, CE-MONO-1: held-equal observation, disagreeing
  functional, and the script line that asserts it.
- Source: `07_counterexample_catalog.md`, `counterexamples/run_counterexamples.py`.

## Table 4 — Real-EEG audited digests
- Step 8 / 9 (BNCI2014_001), Step 10 (BNCI2014_004 binary, BNCI2015_001 legal-skip), combined: runs,
  raw bAcc (within-dataset), chance-normalized excess, offline-TTA gain-norm, harm-rate, boundary
  flags, tracked file.
- Source: `05_experiment_table.md`, `06_results_digest.md`, tracked summaries.

## Table 5 — Forbidden claims and audit rules
- Each forbidden claim (source-only target-gain certification, unlabeled-target concept detection,
  leakage→accuracy, mixed-K raw pooling, posterior-KL upper bound, zero-Bayes escape, GLS source-only
  prior) mapped to the audit rule / certificate that blocks it.
- Source: `07_limitations_and_claim_boundary.md`, `h2cmi/observability/registry.py`.
