# results/cigl/phase3a_graph_backbone_redesign/ — Phase 3A-G outputs

**EXPLORATORY diagnostic outputs** from `scripts/run_cigl_phase3a_graph_backbone_redesign.py` (CIGL
Phase 3A-G; see `docs/CIGL_22_PHASE3A_G_GRAPH_BACKBONE_REDESIGN.md`). Task-capable graph-compatible
backbones under the **same strict source-only** BNCI2014_001 fold-0 protocol, **ERM only (no CMI
regularization)**. Generated `*.json` are **gitignored**; the tracked record is the `docs/CIGL_2x`
write-up.

## Files

- `<fold>_<candidate>_seed<k>.json` — per candidate × seed: `train` / `source_probe` / `target_eval`
  (`balanced_acc`, `macro_f1`; `target_eval.evaluation_only=true`), `train_minus_source_gap`,
  `meta_arch` (`graph_compatible`, `edge_logits_dynamic`, `node_identity_preserved`), `forward_graph`
  (graph_z/node_z shapes + finite + std + `valid`/`nondegenerate`), `graph_usage`
  (`zero_graph_bacc`, `permute_nodes_bacc`, drops, `graph_path_used`), and either a light `leakage`
  audit (dynamic-edge candidates) or `leakage=null` + `leakage_skipped_reason` (static adapter).
- `<fold>_graph_backbone_redesign_summary.json` — per-candidate means + the booleans
  `forward_graph_valid`, `forward_graph_nondegenerate`, `graph_path_used`, `passes`; plus
  `selected_successful_graph_backbones` (source-only), `any_graph_backbone_succeeds`,
  `dynamic_edge_backbone_succeeds`, `only_static_adapter_succeeds`, and firewall meta flags
  (`used_target_labels_for_{training,selection}=false`, `target_eval_is_evaluation_only=true`,
  `graph_backbone_selection_uses_target_eval=false`, `cmi_regularization_used=false`).

## Reading the result

- a **dynamic-edge** backbone passes → **A** (repaired-backbone Gate-2 next).
- only the static DGCNN adapter passes → **C** (graph/node CIGL path, not edge-CMI).
- none pass → **B** (pause method path; keep diagnostic framework).
- passes the task but fails the graph-usage check → **D** (invalid for CIGL; a bypass).

## Provenance / reproduce

- Each summary embeds `meta.commit_hash` / `meta.config_hash`; per-seed JSON embed the held-out subject
  and enc-train / probe-pool sizes, so every number is traceable to a commit.
- Real run: `sbatch scripts/sbatch_cigl_phase3a_graph_backbone_redesign_bnci001.sh`. CPU dry-run:
  `python scripts/run_cigl_phase3a_graph_backbone_redesign.py --dry_run_synthetic --device cpu --seeds 0 1 --epochs 3 --leak_n_perm 5`.
- Exploratory diagnostic (one dataset, one fold) — not a benchmark/SOTA table; feeds the Gate-3A-G
  decision only.
