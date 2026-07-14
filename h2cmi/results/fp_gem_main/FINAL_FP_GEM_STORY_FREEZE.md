# Final FP-GEM Story Freeze

Status: `FROZEN`.

Main method: **Fixed-Prior Geometry EM (FP-GEM)**.

## Theory Claim

- Joint prior fitting creates a prior-to-geometry feedback path.
- FP-GEM removes the target-prior M-step while retaining iterative geometry fitting.
- Fixing the prior removes feedback, not all soft-assignment or prior-misspecification bias.

## Method Claim

- FP-GEM is a simple theory-derived decoupled geometry estimator.
- It is not claimed to be universally prevalence-invariant.
- It is not claimed to be a state-of-the-art universal EEG adapter.

## Empirical Claim

- Controlled mechanism experiments show positive fixed-prior-minus-joint effects under the tested covariance-family mechanisms.
- Sleep-EDF shows a positive fixed-prior-minus-joint geometry contrast.
- The P12 same-backbone comparison shows FP-GEM improves over source-only.
- FP-GEM does not outperform RCT, SPDIM geodesic, or SPDIM bias in the P12 primary aggregate.
- The P13 primary FP-GEM-minus-Joint-GEM sensitivity claim is unsupported.
- P13 shows only that FP-GEM is less sensitive than RCT on the frozen sensitivity endpoint, while its endpoint performance is lower.

## Claim Gate

- `fp_gem_is_main_method = true`
- `theory_prior_feedback_claim_supported = true`
- `controlled_fixed_prior_vs_joint_supported = true`
- `sleep_fixed_prior_vs_joint_supported = true`
- `fp_gem_improves_over_source_only_supported = true`
- `fp_gem_improves_over_rct_supported = false`
- `fp_gem_improves_over_spdim_geodesic_supported = false`
- `fp_gem_improves_over_spdim_bias_supported = false`
- `fp_gem_improves_over_joint_gem_natural_transfer_supported = false`
- `fp_gem_lower_sensitivity_than_joint_supported = false`
- `fp_gem_lower_sensitivity_than_rct_supported = true`
- `universal_prevalence_robustness_supported = false`
- `sota_claim_supported = false`
- `equivalence_claim_supported = false`
- `noninferiority_claim_supported = false`
- `additional_experiment_required = false`

Frozen inputs: P12 result SHA-256 `f3e4ca699b81e4fa2cab404109aa2dfe7aa1fbe58f25e2779d3d11651e40d48d`; P13 result SHA-256 `cf9e403eb8be1c0548a95f9007eb7089ee3f93d8bee2401af22587903bffdb2f`.
