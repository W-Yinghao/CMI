# Step 13 — Real Minimal-Label Curves

**Goal.** Use real held-out target predictions and labels to ask: how many iid-sampled target labels
are needed to recover the sign of the full-target offline-TTA gain?

**Instrumentation.** Each Step-13 run stores a `per_trial_oracle_predictions` block (target trial
index, `y_true`, identity/adapt argmax predictions and confidences, domain) in `raw_results.json`.
This block is **oracle/evaluation-only**: it is read ONLY by `real_minimal_labels.py` for the R2
labeled-slice analysis and never by the R0/R1 harm-attribution feature groups.

**Regimes.**
- **k = 0** — R1 target-unlabeled: the gain sign is non-identifiable (chance; `identified_status =
  not_identified_R1`).
- **k > 0** — R2 labeled slice: draw k iid target trials, estimate the paired per-trial accuracy gain
  and its finite-sample CI, and call the sign (abstain if the CI straddles 0). Compared to the oracle
  full-target accuracy-gain sign.

**Output** (`step13_real_minimal_label_curves.{json,md}`): per k, harm-sign accuracy, decisive rate,
abstention rate, mean CI width; plus `best_k_for_0_8_accuracy` / `best_k_for_0_9_accuracy`.

**Boundary.** The full target gain remains an oracle evaluation quantity. The k-label curve estimates
the **labeled-slice** gain and evaluates how quickly it approaches the oracle full-target sign **under
an iid sampling contract**. Without a sampling / coverage contract, a k-label slice does NOT identify
full target risk — every k > 0 record carries that caveat.
