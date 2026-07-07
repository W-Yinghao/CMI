# Step 12 — Offline-TTA Harm Hypotheses

A hypothesis ledger for *why* offline TTA harms the oracle target metric in the audited grids. Each
row names the regime under which its diagnostic is observable, the diagnostic actually extracted (or
marked missing) in `harm_attribution.py`, the oracle outcome it is tested against, and the contract /
theorem whose failure it corresponds to.

| ID | Hypothesis | Observable under | Diagnostic (extracted) | Oracle outcome | Contract / theorem failure |
|---|---|---|---|---|---|
| H1 | target prior estimate instability | R1 | `target_prior_entropy_hat`, `target_prior_shift_l1_hat`, `tta_prior_shift_mean` | offline-TTA gain | CE-R1-2 (rank-deficient mixture) |
| H2 | pseudo-label overconfidence | R1 | `tta_pred_disagreement_mean` (confidence/entropy = **missing**) | offline-TTA harm | high-confidence-wrong |
| H3 | support mismatch | R1 | `tta_condition_number_mean` (direct support proxy = **missing**) | offline-TTA harm | C1 support-overlap failure |
| H4 | source leakage not predictive of target harm | R0 | `source_leakage_subject_I_hat`, `source_leakage_session_I_hat`, `source_mean_pseudo_gain` | offline-TTA harm | TOS-1 ceiling (source→target) |
| H5 | target marginal shift too large for transport | R1 | `tta_transform_norm_mean`, `tta_delta_density_nll_mean` | offline-TTA harm | C8/MP-1 transport, TU-2 boundary |
| H6 | minimal labels rescue harm estimability | R2 (k labels) | `minimal_paired.py` k-label slice | offline-TTA harm sign | phase transition (labeled slice) |

## Notes

- **Missing diagnostics are recorded, not faked.** `tta_confidence_mean`, `tta_entropy_mean`,
  `target_support_proxy`, and `target_marginal_shift_proxy` are not present in the current raw
  outputs; `harm_attribution.py` lists them in `missing_diagnostics` (reason-coded). H2/H3 are
  therefore tested with the closest available proxy (`pred_disagreement`, `condition_number`) and
  flagged as partial.
- **H4 is the crux.** If source-only diagnostics (R0) do not predict target harm, that is TOS-1 in
  action — the source-only ceiling — and motivates why R1 (and ultimately minimal R2) information is
  needed. `harm_predictor.py` measures exactly the R0→R1 lift.
- No hypothesis, if confirmed retrospectively, upgrades the oracle gain to an identifiable quantity.
