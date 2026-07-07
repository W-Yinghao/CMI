# Step 17 — Estimand Consistency

## Motivation

Step 16 reported two different beneficial rates:
- bAcc-gain beneficial rate: **0.0926**
- accuracy-gain beneficial rate: **0.1481**

This was the *design-time hypothesis* that accuracy-gain and bAcc-gain diverge as estimands. Whether
that divergence is real on this grid — or an artifact of an inconsistent benefit threshold — is
**tested, not assumed**; see [12_estimand_consistency_results.md](12_estimand_consistency_results.md)
for the empirical resolution (spoiler: on the class-balanced BNCI grid the two coincide per run and the
gap is a threshold artifact; the machinery is validated on imbalanced synthetic data where they truly
diverge).

Earlier audited EEG reports primarily used **balanced accuracy**. However, the minimal-label policy
decisions (Steps 15–16) used paired per-trial correctness deltas
`delta_i = 1[adapt_pred_i == y_i] - 1[identity_pred_i == y_i]`, which estimate **ordinary accuracy**
gain, not balanced-accuracy gain. A policy can be safe for accuracy gain but unsafe for bAcc gain (or
vice versa), especially under class imbalance or class-specific adaptation effects.

## Goal

Evaluate harm-control policies under two explicit estimands — **ordinary accuracy gain** and
**balanced-accuracy gain** — kept strictly separate, and report cross-estimand disagreement.

## Regime boundary

Both require k>0 target labels for policy decisions. They are R2 labeled-slice policies under
sampling / design contracts. They do NOT identify target gain under R1.

## Sampling / design contracts

- **IID label slice** — estimates ordinary accuracy naturally; the bAcc estimate is **unstable or
  undefined** when a class is absent from a small iid slice (`bacc_slice_status = missing_class`).
- **Class-balanced calibration slice (contract C13)** — a controlled calibration design that elicits a
  fixed number of labels per class. Stronger than iid; must be **declared** as a contract. Plausible in
  cued BCI protocols, not automatic. If `k < n_classes` the slice is `under_class_budget` → abstain.

## Hard rule

Accuracy-gain control and bAcc-gain control are **different target functionals**. A policy licensed for
one is NEVER reported as controlling the other (`accuracy_policy_controls_bacc = false`).

## Non-goals

- no new datasets · no retraining · no SOTA claim · no manuscript writing.
