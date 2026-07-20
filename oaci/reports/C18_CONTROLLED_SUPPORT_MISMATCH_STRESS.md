# C18 — Controlled Support-Mismatch x Identifiability Stress Test

> Identifiability stress test (NOT a new OACI/SRC experiment, NOT retraining, NOT a selector). GPU re-inference of the C17 candidate checkpoints (source-forward only) makes H1/H2 genuinely mask-recomputed. Target labels are diagnostic-only; no selector is produced.

- **CASE: `collapsed_by_accuracy_endpoint_nonestimability`** — the weak signal SURVIVES cell-present stress (rare/nonestimable); it collapses only under cell DELETION, and there because the worst-domain accuracy ENDPOINT becomes non-estimable (a domain loses a class -> reference bAcc NaN), not because the model signal vanished. Support deletion destroys accuracy-observable availability before it forces leakage abstention.
- next science: the limiter is estimator-level accuracy-endpoint availability under cell deletion, not signal loss; a pre-registered competence probe should use deletion-robust (calibration/leakage) observables + report endpoint estimability. Still diagnostic, NOT deployable.
- boundary-rotation classes (from C16): [1, 3]  ·  folds analysed: 27

## Gates (G1-G6 acceptance)

- **G1 replay identity**: True  (C18_P0_SMOKE.json verdict=PASS identity=8/8 s0=42/42)
- **G2 C17 identity probe** (all-column AUC≈0.602): True  (got 0.6023104389834069, oracle ρ 0.119523893328065, 0 strong scalars: True)
- **G3 static columns excluded from mask claims**: True
- **G4 target labels joined after source generation**: True
- **G5 finite-filter (None/NaN/±inf)**: True  ·  **G6 no selector artifact**: True
- all gates pass: **True**

## S0 split (two baselines, kept distinct)

- C17 all-column identity probe: **+0.602** (reproduces C17 0.6023)
- C18 recomputable-column S0 (genuine mask-recompute baseline; static risk/objective scalars excluded): **+0.593**, beats_perm True

## H2 — identifiability under support stress (recomputable-column probe, reason-coded)

| regime | cell_action | n_feat | loto_auc | perm_p | beats | collapse_reason |
|---|---|---:|---:|---:|:--:|---|
| S0_full_support | none | 8 | +0.593 | +0.008 | True | none |
| S1_label_marginal_skew | reweight_noop | 8 | +0.593 | +0.008 | True | implemented_noop |
| S2_rare_cells | cell_present_downweight | 8 | +0.603 | +0.017 | True | none |
| S3_nonestimable_cells | cell_present_downweight | 8 | +0.562 | +0.008 | True | none |
| S4_missing_cells | cell_deletion | 6 | +0.460 | +0.934 | False | endpoint_metric_nonestimability |
| S5_block_class_by_domain | cell_present_downweight | 8 | +0.577 | +0.099 | False | signal_loss |
| S6_boundary_aligned_mask | cell_deletion | 6 | +0.514 | +0.074 | False | endpoint_metric_nonestimability |
| S7_random_matched_mask | cell_deletion | 6 | +0.448 | +0.008 | False | endpoint_metric_nonestimability |

> cell-present preserved fraction = +0.667; cell-deletion endpoint-nonestimability fraction = +1.000. S1 is an `implemented_noop` negative-control (row-based recompute; reference bAcc + fixed-prior leakage are marginal-invariant) and is excluded from the main severity conclusion.

## H3 — calibration vs accuracy visibility

- accuracy-endpoint availability drops (bAcc→NaN) under cell DELETION; calibration visibility persists. mean accuracy-vis (deleting) +0.076 vs calibration-vis +0.142.

## H4 — class-boundary source-visibility: boundary-aligned (S6) vs random-matched (S7)

- S0 corr +0.547 (reproduces C17 +0.547)  ·  S6 corr +0.511  ·  S7 corr +0.462  ·  boundary-aligned destroys mirror vs random: **False** (mirror is support-ROBUST here)

## H5 — support-aware leakage estimability / abstention

- source-estimable fraction stays +1.000 across regimes; abstains under degradation: **False**. At the tested mild deletion severity, leakage-cell estimability is intact — so accuracy-ENDPOINT non-estimability (worst-domain bAcc) precedes any leakage abstention.

## C18-D (secondary) — observability-dropout proxy

- secondary_observability_proxy_not_source_distribution_recompute · is_primary=False (drops would-be-non-estimable columns; NOT source-distribution recompute; for comparison only)

## Interpretation

> weak signal survives cell-present stress; collapses under cell DELETION via accuracy-endpoint non-estimability

## Appendix — pre-claim validation and superseded first run

1. A pre-claim validation pass (gate-first, before interpretation) traced an initial `n_features` drop to a leakage-recomputation bug, so the first run was NOT interpreted.
2. Bug: the masked leakage support graph's cell_mass did not match the actual masked rows (skew/rare), and a blanket `except Exception: return None` converted the engineering error into missing features — which would have manufactured a false `collapsed_to_case_II_calibration_only` verdict.
3. Fix: derive the leakage support graph from the masked rows (row-consistent cell_mass, fixed reference prior); fail loud except for genuine `LeakageNonEstimableError`. This report is the corrected run; the superseded first-run taxonomy is discarded (engineering appendix only, never a result).