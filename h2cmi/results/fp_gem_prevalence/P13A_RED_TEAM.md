# P13A Independent Red-Team Review

Review scope: frozen P13 prevalence manifest, runner, analyzer, launchers, and tests. This review was performed before writing the P13A freeze report and before any P13 GPU submission.

## Challenges And Disposition

1. **Checkpoint persistence may be illusory.** Ordinary `load_state_dict` does not reconstruct the official SPD batch-normalization state because train/test running buffers alias storage at model construction. The P13 loader explicitly separates those buffers before loading. A CPU tensor-state check reproduces the persisted state hash, but this is not enough to approve the fleet. The precommitted V100 target-1/seed-0 gate must reproduce the P12 source-only prediction and logits hashes and all six q=0.5 method hashes. Status: `OPEN_UNTIL_GPU_GATE`.
2. **Target labels are required to build the intervention.** Labels are confined to `prepare_fp_gem_prevalence.py` and the builder-facing CSV. The runtime JSON contains ordered trial IDs and class counts but no per-trial labels. `adapt_methods` has no label or q argument, and evaluation labels are first read after every q-specific fit. Status: `CONTROLLED_INTERVENTION_ONLY`.
3. **q=0.5 could be silently recomputed as a new result.** P12 retained prediction/logit hashes but not vectors needed for disagreement. P13 therefore performs a hash-gated deterministic replay solely to recover vectors, then uses the committed P12 metrics as the accepted q=0.5 rows. Any prediction, logits, or GEM geometry mismatch blocks the unit. Status: `MITIGATED_BY_EXACT_HASH_GATE`.
4. **Class ordering could be an unintended intervention.** Endpoint batches use a frozen class-order then modulo repeat/crop rule. Every adaptation routine receives one full 50-trial batch; no mini-batch sequence or shuffle is used. The rule is identical for all methods and q is hidden. Interpretation remains restricted to this deterministic intervention. Status: `DISCLOSED_FIXED_PROTOCOL`.
5. **Fresh source training could enter through fallback logic.** The P13 runner imports no source-training entry point and contains no `trainer.fit` call. Missing or mismatched checkpoints fail closed. Status: `MITIGATED`.
6. **GPU type could break exact replay.** Each unit is assigned to its P12 hardware group (`V100` or `A100`), and the runner rejects a mismatched device. Status: `MITIGATED_PENDING_GPU_GATE`.
7. **Result-dependent endpoint changes remain possible if analysis is written later.** The complete analyzer, primary comparison, three external comparisons, secondary endpoints, 10,000-replicate paired bootstrap, and support rule are frozen in this commit before compute. Status: `MITIGATED`.
8. **A favorable endpoint could be overgeneralized.** P13 is Lee2019_MI only and tests controlled fixed-reservoir prevalence sensitivity, not natural-transfer superiority, equivalence, or noninferiority. Status: `UNDERCLAIM_BOUNDARY_REQUIRED`.

## Verdict

`CONDITIONALLY_READY_FOR_SINGLE_GATE_ONLY`

The full fleet is not approved by this red-team review until the target-1/seed-0 checkpoint/q=0.5 GPU gate passes exactly. No performance metric may be used to alter the frozen protocol.
