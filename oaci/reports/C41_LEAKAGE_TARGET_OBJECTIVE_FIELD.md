# C41 - Global Leakage-Target Utility Objective Field Audit (frozen C19 `664007686afb520f`)

> Read-only diagnostic audit over committed candidate-level artifacts. No training, no GPU, no selector repair, no atom-level claims, and no proxy selector UCL.

- **cases: `O2_global_leakage_target_decoupling, O5_source_audit_leakage_no_better, O6_source_rank_better_than_leakage`**
- candidate rows / trajectories: **3804 / 162**.

## Leakage-Target Field

- selection leakage mean AUC vs target utility: **0.494**.
- source-audit leakage mean AUC vs target utility: **0.498**.
- C30 aggregate source-rank AUC: **0.659**.

## Low-Leakage Enrichment

- top-3 low-leakage joint-good enrichment: **0.918**.
- top-3 low-leakage Pareto-good enrichment: **1.198**.
- top-3 low-leakage preference-robust-local-alternative enrichment: **2.412**.
- O4 status: **not active**; joint-good is below baseline, but sparse robust-local-alternative ratios prevent a clean all-label no-enrichment call.
- Enrichment is trajectory-conditioned and compared to within-trajectory random baselines.

## Local-Global Consistency

- local conflict representative fraction: **0.789**.
- local tail-only fraction: **0.000**.
- mean selected low-leakage rank percentile: **0.014**.
- O8 status: **not active**; the representative fraction is near the pre-registered 0.800 gate but does not pass it.

## Boundaries

- C30 source-rank and target-gauge fields are not candidate-level absolute fields in current artifacts; they are reported as aggregate/local diagnostics only.
- Target endpoints and target gauge remain diagnostic-only and non-source-only where applicable.

## Bottom Line

> C41 establishes O2 + O5 + O6: global selection leakage is mostly decoupled from target utility, source-audit leakage does not materially improve the target-utility alignment, and the aggregate C30 source-rank axis is stronger than leakage. O4 and O8 remain below the pre-registered gates.
