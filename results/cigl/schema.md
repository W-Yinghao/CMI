# CIGL result schema

Canonical record format for every CIGL / GraphCMI experiment. One JSON object per
`(dataset, protocol, backbone, method, seed, target_subject)` cell. Maps (node/edge
leakage) are stored as sidecar `.npy`/`.npz` files and referenced by path.

This schema is the contract a reviewer uses to reconstruct *exactly which data each
method saw* (Gate 4, `docs/CIGL_05_ACCEPTANCE_CRITERIA.md`). A result that omits a
mandatory field is not analyzable and does not count as evidence.

---

## Mandatory scalar fields

| field | type | meaning |
|---|---|---|
| `setting` | str | **`strict_source_only_DG`** for CIGL. (`transductive_TTA` only for separate CITA rows; never mixed in one table.) |
| `dataset` | str | e.g. `SEED`, `SEED_IV`, `BNCI2014_001`, `Lee2019_MI`. |
| `protocol` | str | `loso` (Protocol A) / `cross_session` (Protocol B) / `cross_dataset` (C) / `synthetic` (D). |
| `feature_protocol` | str | `raw` (default) or `de` — raw and DE rows are **never** compared as if identical. |
| `backbone` | str | `GraphCMI` / `DGCNN` / `RGNN` / `EEGNet` / ... |
| `method` | str | `graphcmi` (+ the lambda triple below), `erm`, `lpc_prior`, `cdann`, ... |
| `lambda_g` | float | graph-level weight λ_g on `I(Z_g;D\|Y)`. |
| `lambda_node` | float | node-level weight λ_node on `(1/C)·Σ_v I(Z_v;D\|Y)`. |
| `lambda_edge` | float | edge-level weight λ_edge on `I(A;D\|Y)`. |
| `seed` | int | RNG seed. |
| `target_subject` | str | held-out target (LOSO) or held-out session (cross-session). |
| `source_subjects` | list[str] | training domains. |
| `balanced_acc` | float | per-target balanced accuracy (primary task metric). |
| `macro_f1` | float | macro F1. |
| `worst_subject_bacc` | float | worst-target balanced accuracy across the run. |
| `graph_leakage_kl` | float | held-out `Î(Z_g;D\|Y)` (posterior-KL proxy). |
| `node_leakage_mean` | float | mean held-out `Î(Z_v;D\|Y)` over channels. |
| `edge_leakage_kl` | float | held-out `Î(A;D\|Y)`. |
| `label_separability` | float | linear-probe task separability of the representation. |
| `ece` | float | expected calibration error. |
| `nll` | float | negative log-likelihood. |
| `commit_hash` | str | `git rev-parse HEAD` at run time. |
| `config_hash` | str | hash of the resolved run config (args + method string). |

## Optional / diagnostic scalar fields

| field | type | meaning |
|---|---|---|
| `loss_ce` | float | mean task CE over the last epoch (from trainer diagnostics). |
| `reg_graph` / `reg_node` / `reg_edge` | float | per-component held-in leakage breakdown. |
| `stepA_graph_dom_acc` / `stepA_node_dom_acc` / `stepA_edge_dom_acc` | float | Step-A domain-classification accuracy of each posterior head (critic-capacity diagnostic). |
| `edge_subject_acc` | float | accuracy of predicting subject/domain from `edge_logits` conditional on `Y`. |
| `accuracy_leakage_pareto_rank` | int | rank on the accuracy–leakage (or worst-subject–leakage) Pareto front. |
| `permutation_null_q` | float | within-label domain-permutation null quantile for the leakage object (Gate 2). |

## Map / array sidecar fields (paths, relative to the result file)

| field | type | meaning |
|---|---|---|
| `node_leakage_map_path` | str | `*.npy`, length-`C` per-channel residual leakage map. |
| `edge_leakage_matrix_path` | str | `*.npy`, `C×C` per-edge leakage (diagnostic-only in v1; **no** per-edge training heads). |

---

## Notes / red lines (carried from the acceptance criteria)

- `graph_leakage_kl` / `node_leakage_mean` / `edge_leakage_kl` are **posterior-KL plug-in
  proxies**, not unbiased CMI estimates. Never label them "CMI" without the proxy caveat.
- Every accuracy number must ship with leakage; every leakage number with task preservation.
- Strict-DG rows use **no** target labels and **no** target covariates in training, model
  selection, normalization, or alignment.
- Main claims require ≥5 seeds (Gate 5); single-seed rows are exploratory only.
