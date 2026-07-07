# Step 16 — Sequential Policy Frontier Results

Numbers live in `results_summaries/step16_sequential_harm_control.{json,md}`,
`step16_policy_frontier.{json,md}`, and the Step-16 dashboard.

## Questions

- **Does sequential labeling improve over fixed-k policies?** Sequential policies acquire labels batch
  by batch and stop when the paired-gain CI is decisive, so they can spend fewer labels on easy cells.
- **What label budget is needed for safe adaptation?** The predeclared best sequential policy requires
  harm ≤ 0.05 and coverage ≥ 0.05; its `mean_labels_used` is the calibration burden.
- **Is there a harm-control frontier?** `policy_frontier.py` reports whether any deployable policy meets
  harm thresholds 0.05 / 0.10 / 0.20 / 0.50 at min coverage 0.01 / 0.05 / 0.10.
- **What remains unknown?**

## Reading (three possible conclusions, decided by the tracked numbers)

1. **Sequential policies still fail harm ≤ 0.05** → minimal labels do not support safe adaptation
   selection; identity/default remains safest; safe adaptation needs (near-)full labels.
2. **They meet harm ≤ 0.05 only near full budget** → safe adaptation requires a large calibration
   burden (full-label calibration policy, `calibration_burden = full`).
3. **They meet harm ≤ 0.05 with a moderate budget** → R2 minimal labels can control harm under an iid
   sampling contract (but still NOT R1 target-gain identification).

The frontier at relaxed thresholds (0.10 / 0.20) shows whether the 0.05 constraint was simply too
strict, or whether the failure is structural (benefit too rare/small — see `09_benefit_anatomy_results.md`).

## Claim boundary

Sequential policies are R2 labeled-slice procedures under an iid sampling / coverage contract. They are
NOT R1 target-gain identification and NOT source-only adaptation claims. The oracle full-target policy
is an evaluation-only upper bound, excluded from the deployable frontier. No SOTA claim.
