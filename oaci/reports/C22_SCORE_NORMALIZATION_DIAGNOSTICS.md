# C22 — Score-normalization diagnostics (MECHANISM only, NON-deployable)

> post-hoc MECHANISM diagnostics only; target/regime-wise normalization needs the target/regime identity at score time and is NON-deployable. Recovery => the failure is score offset/calibration (rank-like signal); no recovery => regime-specific relationship shift.


## in_regime
- pooled (none): +0.543
- by normalization: {'none': 0.543, 'target_center': 0.634, 'target_zscore': 0.643, 'target_rank': 0.648, 'regime_center': 0.543, 'target_regime_center': 0.637, 'quantile': 0.648}
- target-normalization recovers pooled: True

## cross_regime
- pooled (none): +0.518
- by normalization: {'none': 0.518, 'target_center': 0.613, 'target_zscore': 0.62, 'target_rank': 0.628, 'regime_center': 0.519, 'target_regime_center': 0.616, 'quantile': 0.628}
- target-normalization recovers pooled: True