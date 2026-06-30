# results/cigl/phase3a_dgcnn_gn_second_dataset_confirmation/ — Phase 3A-K outputs

**EXPLORATORY second-dataset confirmation outputs** from
`scripts/run_cigl_phase3a_dgcnn_gn_second_dataset_confirmation.py` (CIGL Phase 3A-K; see
`docs/CIGL_30_PHASE3A_K_SECOND_DATASET_CONFIRMATION.md`). Replication of the FIXED candidate
`graph_node_010` (λ_g=λ_node=0.010, **no edge**) vs `erm_fixed` on **BNCI2015_001** (binary), **strict
source-only**, **graph/node only**. Generated `*.json` are **gitignored**; the tracked record is
`docs/CIGL_31`.

## Files

- `<dataset>_fold<f>_<config>_seed<k>.json` — per fold × config × seed: `train`/`source_probe`/
  `target_eval` (`balanced_acc`,`macro_f1`; `target_eval.evaluation_only=true`), `graph_usage`,
  `leakage`={`graph`,`node`} (`kl_mean`,`permutation_mean`,`permutation_p`,`clears_null`),
  `edge_audit_skipped=true`+reason, and source-only `meta` (incl. `n_classes`, `chance_bacc`,
  `source_mean_floor`, `source_seed_floor`, `fixed_candidate`, `second_dataset_confirmation`).
- `<dataset>_dgcnn_gn_2nd_dataset_summary.json` — `meta` (binary floors + `n_classes`/`chance_bacc`),
  `configs` (fixed λ map), `per_fold` (per-fold `erm_fixed`/`graph_node_010` aggregates + `flags`:
  `erm_adequate`, `erm_leakage_exists`, `reg_reduces`, `source_retained`, `target_guardrail`, paired
  `graph_reduction`/`node_reduction`, `source_drop_vs_erm`, `target_drop_vs_erm`, `fold_pass`), and the
  decision group **`second_dataset_confirmation`** (all folds; no dev fold). `all_folds_descriptive`
  equals it (no dev fold on a second dataset).

## Reading the result

Binary dataset → chance bAcc 0.50; floors are mean **0.60** / seed **0.55**. Criteria over all
BNCI2015_001 folds: `crit1_erm_adequate` (≥8/12), `crit2_erm_leakage` (≥8/12), `crit3_reg_reduces`
(≥7/12), `crit4_source_retained` (≥7/12), `confirmed`, `decision`:

- **A** confirmed → method framing may begin.
- **B** partial → bounded single-dataset method signal.
- **C** not confirmed → BNCI2014_001-only finding.
- **D** ERM unstable → dataset/backbone diagnosis.

`reduction = (ERM_KL − graph_node_010_KL)/ERM_KL` paired by fold+seed; `clears_null = kl_mean >
permutation_mean AND permutation_p ≤ gate_alpha (0.05)`, `n_perm=50`. Configs are **fixed** (no
selection); `target_eval` is an evaluation-only guardrail. **No edge term/audit.** If the loaded dataset
is not binary (`n_classes != 2`), the runner stops for reviewer re-authorization. Exploratory — not a
benchmark/SOTA table.

## Remote integrity (authoritative byte check)

If a `raw.githubusercontent.com` preview looks "compressed to a few lines", that is a viewer/CDN artifact
— verify the git object bytes: `git show origin/<branch>:<file> | wc -l` (or the GitHub **blob page**),
not the branch-ref raw preview.
