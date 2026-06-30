# results/cigl/phase3a_backbone_sanity/ — Phase 3A-S backbone sanity outputs

**EXPLORATORY diagnostic outputs** from `scripts/run_cigl_phase3a_backbone_sanity.py` (CIGL Phase 3A-S;
see `docs/CIGL_20_PHASE3A_S_BACKBONE_SANITY.md`). Known-good MI decoders vs the GraphCMINet reference
under the **same strict source-only** BNCI2014_001 fold-0 protocol, ERM only (no CMI regularization).
Generated `*.json` are **gitignored**; the tracked record is the `docs/CIGL_2x` write-up.

## Files

- `<fold>_<candidate>_seed<k>.json` — per candidate × seed: `train` / `source_probe` / `target_eval`
  (`balanced_acc`, `macro_f1`; `target_eval.evaluation_only=true`), `train_minus_source_gap`,
  `is_graph_backbone`, and — **only** for the graph reference (`graphcmi_current_ref`) — a light
  `leakage` audit (`graph`/`node`/`edge` `kl_mean`, `permutation_p`). Non-graph CNNs carry **no**
  leakage fields. Each file embeds source-only `meta` flags.
- `<fold>_backbone_sanity_summary.json` — per-candidate means (`train_bacc`, `source_probe_bacc`,
  `target_eval_bacc`, `train_minus_source_gap`, optional `leakage_kl`), plus:
  - `selected_successful_models` — candidates with `source_probe_bacc ≥ success_bacc_floor` (0.45),
    chosen from **source_probe only**;
  - `known_good_decoders_succeed`, `graphcmi_succeeds`;
  - firewall flags `success_selection_uses_target_eval=false`,
    `used_target_labels_for_selection=false`, `used_target_labels_for_training=false`,
    `used_target_covariates=false`, `target_eval_is_evaluation_only=true`.

## Reading the result

- known-good decoder ≥ 0.45 while GraphCMINet ~0.33 → **A**: protocol usable, GraphCMINet is the blocker.
- all near 0.33 → **B**: protocol / preprocessing / data diagnosis.
- (C/D in `docs/CIGL_20`). The runner prints only an exploratory read; the **reviewer** decides the gate.
