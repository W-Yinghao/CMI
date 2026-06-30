# results/cigl/phase3a_dgcnn_gn_multifold_confirmation/ — Phase 3A-J outputs

**EXPLORATORY confirmation outputs** from `scripts/run_cigl_phase3a_dgcnn_gn_multifold_confirmation.py`
(CIGL Phase 3A-J; see `docs/CIGL_28_PHASE3A_J_MULTIFOLD_CONFIRMATION.md`). Replication test of the FIXED
candidate `graph_node_010` (λ_g=λ_node=0.010, **no edge**) vs `erm_fixed` across BNCI2014_001 LOSO folds,
**strict source-only**, **graph/node only**. Generated `*.json` are **gitignored**; the tracked record is
`docs/CIGL_29`.

## Files

- `<dataset>_fold<f>_<config>_seed<k>.json` — per fold × config × seed: `train`/`source_probe`/
  `target_eval` (`balanced_acc`,`macro_f1`; `target_eval.evaluation_only=true`), `graph_usage`,
  `leakage`={`graph`,`node`} (`kl_mean`,`permutation_mean`,`permutation_p`,`clears_null`),
  `edge_audit_skipped=true`+reason, and source-only `meta` (incl. `fold`, `is_dev_fold`,
  `fixed_candidate`).
- `<dataset>_dgcnn_gn_multifold_summary.json` — `configs` (fixed λ map), `per_fold` (per-fold `erm_fixed`
  / `graph_node_010` aggregates + `flags`: `erm_adequate`, `erm_leakage_exists`, `reg_reduces`,
  `source_retained`, `target_guardrail`, paired `graph_reduction`/`node_reduction`, `*_reduce30_seeds`,
  `source_drop_vs_erm`, `target_drop_vs_erm`, `fold_pass`), and three decision groups:
  `fold0_dev` (separate), **`folds1_8_confirmation` (PRIMARY)**, `all_folds_descriptive`.

## Reading the result

The **primary decision is `folds1_8_confirmation`** (fold-0 is the development fold that selected
`graph_node_010` and is excluded). Each group reports `crit1_erm_adequate` (≥6/8), `crit2_erm_leakage`
(≥6/8), `crit3_reg_reduces` (≥5/8), `crit4_source_retained` (≥5/8), `confirmed`, and `decision`:

- **A** confirmed (crit1–4) → method framing / second-dataset confirmation.
- **B** partial → bounded finding / refine.
- **C** not confirmed → pilot-only, no method claim.
- **D** ERM unstable across folds → DGCNN stability diagnosis.

`reduction = (ERM_KL − graph_node_010_KL)/ERM_KL`, paired by fold and seed; `clears_null = kl_mean >
permutation_mean AND permutation_p ≤ gate_alpha (0.05)`, `n_perm=50`. Configs are **fixed** (no
selection); `target_eval` is an evaluation-only guardrail. **No edge term/audit.** Exploratory
confirmation — not a benchmark/SOTA table.

## Remote integrity (authoritative byte check)

The committed files are normal multi-line source (LF line endings; `.gitattributes` enforces `eol=lf`).
If a `raw.githubusercontent.com` preview ever appears "compressed to a few physical lines", that is a
fetch/CDN artifact on the viewer side — verify the authoritative git object bytes instead:

```bash
git fetch origin
B=project/cigl-phase3a-dgcnn-gn-multifold-confirmation
for f in scripts/run_cigl_phase3a_dgcnn_gn_multifold_confirmation.py \
         scripts/sbatch_cigl_phase3a_dgcnn_gn_multifold_confirmation_bnci001.sh \
         tests/test_phase3a_dgcnn_gn_multifold_confirmation.py; do
  echo "$f $(git show origin/$B:$f | wc -l)"
done
# expect ~249 / 36 / 114 physical lines (NOT 7 / 3 / 3)
```

The GitHub **blob page** (`github.com/.../blob/<commit>/<file>`) renders numbered lines and is a reliable
cross-check; the branch-ref raw preview is not.
