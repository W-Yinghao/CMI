# Project B Step-3C Real EEG Benchmark Expansion Protocol

> A **bounded real benchmark expansion** (NOT a full MOABB benchmark) on top of the Step-3A bridge.
> Experiment driver + report only: modifies no `h2cmi/**` or `cmi/**`. Branch `project-b-refusal-router`.

## 1. Goal
Scale the Step-3A real-EEG bridge from a 2-target smoke to a bounded benchmark: more held-out target
subjects, both **subject-level** and **session-level** routing, both support-calibration modes, with a
full OACI reason-code audit — all source-only and label-safe. Success is measured by protocol
correctness and honest characterization, not accuracy.

## 2. Datasets
Primary: **BNCI2014_004** (2b, binary L/R MI, 3 bipolar channels) — the only pass/fail dataset.
Optional probe: BNCI2014_001 (2a, 4-class) — availability/probe only, never pass/fail; failures are
recorded, not fatal, under `--allow_dataset_failures`.

## 3. LOSO protocol
Leave-one-subject-out over the first `max_subjects`. The first `max_targets` subjects are held out one
at a time; the rest are source. The base H²-CMI model is trained **once** per target and reused across
eval units and support modes.

## 4. Eval units: subject vs session
`subject` gives one coarse target domain per held-out subject; `session` gives one domain per
`subject|session` pair. Both are attempted per target with the same trained model and thresholds. If a
dataset has one session, session-level routing collapses to subject-level — a valid outcome, not a
failure.

## 5. Support calibration modes (source-only)
- `in_source_subject_q95` = q95 over base-model source-subject target-prior density NLL.
- `nested_source_subject_excess_q95` = `base_q95 + max(0, q95(nested source-subject excess))`, where
  `excess = held-out-source-subject NLL − q95(nested-model in-training source-subject NLL)`
  (scale-normalised within each of ≤`max_nested_folds` nested folds, added to the base scale). The
  pseudo-subjects used are recorded. Thresholds never use target labels or target NLLs.

## 6. Router decision semantics
Unchanged Step-2D policy: action-specific blockers; `OFFLINE_TTA` blocked when ACAR-harm is
degenerate/unavailable; support-valid IDENTITY allowed; low-ESS / support-mismatch refused;
safe-beneficial-then-identity selection.

## 7. Label-safety rules
Target labels `y` are used **only after** each `RouterDecision`, inside the harness, for post-hoc
metrics. No threshold, DAG, diagnostic, or action reads target labels. The driver fails loudly if any
`OFFLINE_TTA` is selected while source ACAR-harm is degenerate/unavailable.

## 8. Metrics and reason-code audit
Per (dataset, target, eval_unit, support_mode): strict bAcc, raw offline TTA Δ, router coverage /
action rates / accepted bAcc / missed_benefit / avoided_harm, ACAR-harm state, support threshold. A
reason-code audit separates top-level decision reasons from identity-action reasons and TTA-action
blockers; a TTA blocker is never counted as "identity unsafe" unless identity itself emits it.

## 9. Known limitations
Bounded: ≤6 subjects, ≤4 targets, 8 epochs, ≤2 nested folds — **not a full benchmark**. ACAR-harm is
expected to be unavailable when source subjects are few (n_pseudo < min calibration). Session counts and
per-session trial counts are dataset-dependent; low-trial sessions may trigger too-few-target / low-ESS
refusals, which is correct behavior.
