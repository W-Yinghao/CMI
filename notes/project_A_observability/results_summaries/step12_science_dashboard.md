# Step 12 — Science Dashboard

Scope: scientific exploration (harm attribution + minimal-information phase transition); not SOTA.

## Key metrics

- real runs: **54** · real harm-rate: **0.8333**
- harm-predictor balanced-acc — R0: **0.2556** · R1: **0.4222** · R1−R0 delta: **0.1666** (majority baseline **0.5**)
- minimal-paired: k0 **not_identified_R1** · phase transition **True** · best k **256**
- claim boundary ok: **True** · oracle gain evaluation-only: **True**

## What we learned

1. Offline TTA harms most audited cells (real harm-rate 0.8333).
2. R0/R1 diagnostics do NOT retrospectively predict TTA harm above the 0.5 majority baseline (R0 bAcc 0.2556, R1 bAcc 0.4222; underpowered, minority n=9) — consistent with the TOS-1 source-only ceiling; NULL result, not identifiability.
3. R1 target-unlabeled diagnostics do NOT make target gain identifiable (TOS-1/TU-2 stand).
4. Minimal paired information: harm-sign estimability is a phase transition in k (observed=True, per-shift k {'prior_shift_only': None, 'concept_shift': None, 'support_failure': 256, 'montage_transport_shift': None}); small true gains need more labels, tiny gains stay unresolved — a labeled slice under an iid sampling contract.
5. Exact counterexamples remain the proof layer; the real-EEG grids illustrate, they do not prove.

## What remains unknown

1. Whether these patterns hold on clinical / non-motor-imagery EEG.
2. Whether stronger TTA baselines reduce the harm rate.
3. Whether label-free target support/marginal diagnostics can be made reliable.
4. Whether minimal-paired anchors can be collected cheaply in realistic BCI workflows.

> Oracle target gain is an evaluation label throughout; R0/R1 harm prediction is retrospective, not target-gain identifiability; k>0 slices are labeled slices under an iid sampling contract. No SOTA claim.
