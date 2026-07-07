# Limitations and Claim Boundary

## Limitations

1. **Modality scope.** Current real evidence is MOABB **motor-imagery** only (BNCI2014_001,
   BNCI2014_004). No clinical / sleep / P300 evidence yet.
2. **BNCI2015_001 contributes nothing.** It is unavailable under the current MOABB/paradigm loader
   configuration (not a left/right-hand task, so `LeftRightImagery` rejects it) — a legal skip with no
   ok runs. It is reported but adds no target-metric evidence.
3. **Not SOTA-tuned.** Runs use fast / modest training (`--fast`, 50 epochs); they are audited
   claim-boundary and stability validations, not tuned performance comparisons.
4. **TTA baselines are not exhaustive.** The offline/online TTA methods are the current H2-CMI
   implementations (class-conditional density TTA + safety gate), not a survey of TTA baselines.
5. **Oracle labels are evaluation-only.** Target labels are used solely to *compute* target metrics for
   auditing; they are never treated as observations available to adaptation.
6. **TU-1 priors are reported, not identified.** A target-prior estimate is emitted as evidence but is
   `rejected_conclusion_false` unless the contracts C1∧C2∧C3 are explicitly declared.
7. **No causal mechanism.** No causal EEG mechanism is identified from these experiments.
8. **Proof vs illustration.** The exact counterexamples prove non-identifiability; the real-EEG grids
   *illustrate* the audit discipline and TTA fragility — they do not prove a theorem.

## Claim boundary

**Allowed.**
- Source-side metrics and diagnostics under R0 (source `π_d(y)`, PD-1/ID-1 source-side relations).
- Oracle/evaluation-only target metrics with `identifiable_estimand=null` (reportable, not identified).
- Target prior `π_T` **only** under TU-1 (C1∧C2∧C3 declared).
- Conditional leakage `I(Z;D|Y)` as a **diagnostic**.
- Cross-dataset comparison via **chance-normalized excess** only.

**Forbidden (audit rejects / fails loud).**
- Source-only target-gain or target-safety **certification** (TOS-1 / CE-R0-2).
- **Unlabeled-target concept detection** (TU-2 / CE-R1-1).
- **Leakage → accuracy** guarantee.
- **Mixed-class raw balanced-accuracy pooling** across datasets with different K.
- Posterior-KL-as-upper-bound, zero-Bayes-error escape, and "GLS gives the target prior source-only"
  (retracted; P0-2/P0-3/P0-5).
- **No SOTA claim.**

## One-line summary

Observed information + declared contracts determine what can be claimed; target metrics measured with
oracle labels are reportable but not identifiable under source-only or target-unlabeled deployment, and
the audit enforces that boundary mechanically.
