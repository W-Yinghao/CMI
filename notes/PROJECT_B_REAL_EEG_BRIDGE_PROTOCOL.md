# Project B Real EEG Bridge Protocol

> Step-3A. A label-safe, source-only LOSO bridge from the AAAI `cmi/` MOABB loader to the Step-2
> Project B router. This is a **bridge smoke**, not a MOABB benchmark. It modifies no `cmi/` loader
> and no H2-CMI trainer / TTA / router / harness. Branch `project-b-refusal-router`.

## 1. Purpose
Prove the real-EEG substrate runs end-to-end: MOABB load → H2-CMI `DomainDAG`/`DomainLabels` →
`train_h2` on source subjects → source-only support calibration → `evaluate_router_offline_tta` on a
held-out target subject → OACI reason-coded decisions. The point is **interface + protocol
correctness** and surfacing the real failure mode, before drafting the paper (Step-3B).

## 2. Dataset entrypoint
`h2cmi/data/real_eeg_bridge.load_moabb_real_eeg(dataset, ...)` calls `cmi.data.moabb_data.load`
(unmodified), which returns `(X[float32, n_trials, n_chans, n_times], y[int64], meta[DataFrame],
classes)`. First dataset: **`BNCI2014_004`** (2b, binary left/right MI, 8–30 Hz, 3 bipolar channels)
— the lowest-risk first bridge. `n_chans`/`n_times`/`fs` are read from the loaded array (not assumed
from the simulator's 16×128).

## 3. Domain semantics
Single-dataset, **two-factor** DAG (no fabricated `site`):
```
subject (random_effect, budget 0.05)  ->  session (random_effect, budget 0.10)
```
`session` global level = unique `subject|session` pair. `train_h2(..., align_factor="subject")`.
Built by `make_subject_session_dag(meta)` / `make_source_domain_labels(meta_source)`.

## 4. LOSO split and label-safety
`split_loso_by_subject(meta, target_subject)` → disjoint `(source_idx, target_idx)`; the held-out
**subject** is the target. Target `y` is used **only** post-hoc, inside `evaluate_router_offline_tta`,
after each `RouterDecision`. No threshold, diagnostic, or action ever reads target `y`.
`eval_unit ∈ {subject, session, run}` (first smoke: subject).

## 5. Support calibration modes (source-only)
- `in_source_subject_q95` — baseline = q95 over base-model **source-subject** `density_nll_target_prior`.
- `nested_source_subject_excess_q95` — the real analogue of the Step-2F fix:
  `base_source_q95 + max(0, q95(nested source-subject excess))`, where
  `excess = held-out-source-subject NLL − q95(nested-model in-training source-subject NLL)`
  (scale-normalised within each nested fold, then added to the base scale). Bounded to
  `--max_nested_folds` (default 2) nested retrains for the smoke — proof of computation, not a full
  K-fold. All source-only; never target-label tuned.

## 6. Router evaluation outputs
`fold_summary.csv` (per target × mode), `per_domain_decisions.csv`, `support_calibration_details.csv`,
`real_bridge_summary.json`, per-fold `<dataset>_target<subj>_<mode>_router_report.json`. Reports carry
OACI reason histograms, per-domain decisions, support thresholds, and ACAR-harm state. If data is
unavailable and `--allow_missing_data` is set, `availability_error.json` is written and the run exits 0.

## 7. What this bridge can claim
- The real-EEG substrate (MOABB → H2-CMI DAG → train → router harness) runs end-to-end under LOSO.
- Support thresholds and pseudo-harm gains are computed **source-only**; target labels are post-hoc.
- The Step-2 router posture transfers unchanged: `OFFLINE_TTA` is blocked under degenerate ACAR-harm.

## 8. What this bridge cannot claim yet
- No real-EEG **performance** claim (this is a bounded smoke: ≤4 subjects, ≤2 targets, 8 epochs, ≤2
  nested folds).
- Not a MOABB benchmark; no cross-dataset / cross-session deployment protocol yet.
- Whether nested source-**subject** excess calibration helps on real EEG the way source-**site** excess
  did on synthetic is an empirical question answered by the run, not assumed.

## 9. Step-3B paper integration plan
Step-3B drafts the method/protocol/results/limitations section from the Step-2 frozen synthetic tables
(`notes/PROJECT_B_STEP2_SYNTHETIC_REPORT.md`, `claim_boundary.json`) **plus** the Step-3A real-bridge
status (real end-to-end run or documented data-availability limitation). Step-3C then expands the real
benchmark (more subjects/targets, BNCI2014_001 / Lee2019_MI, session-level deployment route).
