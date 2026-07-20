# C23 — Target-identity-leakage audit (HARD GATE, reported FIRST)

> Runs BEFORE any positive calibration claim. In LOSO the source composition ≈ target identity, so a gauge that merely re-encodes target id is NOT target-free calibration (G3).

If source features predict target id far above chance (1/9), the per-target gauge is identity-laden; a positive calibration then only counts if the offset relationship GENERALIZES leave-one-target-out (offset_model.loto), else it is G3.

- 9-way target-id accuracy from raw source features: **+0.541**
- chance (1/9): +0.111; identity-laden ceiling: 0.35
- source features identity-separable: **True**
- n_candidates 3804, n_targets 9

If identity-separable AND the offset does NOT generalize leave-one-target-out, any apparent calibration is target-identity leakage (G3), not a target-free gauge.
