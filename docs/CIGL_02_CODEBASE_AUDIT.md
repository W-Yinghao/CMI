# CIGL Codebase Audit

This audit defines what already exists and what must be changed before CIGL can be treated as a new project rather than a loose branch.

---

## 1. Existing assets

### 1.1 Repository top-level identity

The current root `README.md` still presents the project as Tri-CMI / LPC-CMI for calibration-free EEG domain generalization. CIGL must therefore add a new active-project layer without deleting the historical Tri-CMI work.

Required action:

- Add a clear “Active project: CIGL / GraphCMI” section near the top of `README.md`.
- Keep Tri-CMI as foundation/historical context.
- Mark CITA as existing transductive branch, not the new main project.

### 1.2 Existing GraphCMI design document

`notes/gnn_design.md` already specifies the intended design:

- raw temporal encoder;
- per-sample adjacency;
- graph, node, and edge CMI;
- plain PyTorch, no PyG;
- GraphCMINet exposure of `forward` and `forward_graph`.

This is strong starting material, but it is still in `notes/` and not organized as a project specification.

Required action:

- Keep `notes/gnn_design.md` as historical design.
- Use `docs/CIGL_01_METHOD_SPEC.md` as the canonical current spec.

### 1.3 Existing model code

`cmi/models/gnn.py` already includes:

- `TemporalNodeEncoder`
- `Adjacency`
- `DGCNNBackbone`
- `RGNNBackbone`
- `GraphCMINet`
- `forward(x) -> (logits, graph_Z)`
- `forward_graph(x) -> (logits, graph_Z, node_Z, edge_logits)`

This means Phase 1 should not start by reimplementing the backbone. It should start with tests, interface hardening, logging, and diagnostics.

### 1.4 Existing graph regularizers

`cmi/methods/graph_regularizers.py` already includes:

- `NodePosterior`
- `EdgePosterior`
- KL-to-prior helper
- `leakage_map` for node posteriors

Required action:

- Add explicit tests for shapes, finite loss, gradient flow, and prior shape handling.
- Add documentation comments clarifying that this is a plug-in leakage proxy.
- Ensure `priors` indexing is correct and robust.

### 1.5 Existing trainer branch

`cmi/train/trainer.py` already has:

- `method == "graphcmi"`
- graph/node/edge posterior creation
- Step A fitting for all three heads
- Step B loss with graph, node, and edge terms

Required action:

- Rename/report `gamma` as `lambda_node` in CLI and logs.
- Ensure `lam_edge` is logged everywhere.
- Add loss breakdown logging: `reg_graph`, `reg_node`, `reg_edge`.
- Confirm `method="graphcmi"` fails loudly when backbone lacks `forward_graph`.
- Confirm `graphcmi:...` parsing exists at runner level; add if missing.

---

## 2. Missing project infrastructure

### 2.1 Tests

Create at minimum:

```text
tests/test_graphcmi_backbone.py
tests/test_graph_regularizers.py
tests/test_graphcmi_trainer_smoke.py
```

Required assertions:

- `forward` returns exactly a 2-tuple.
- `forward_graph` returns exactly a 4-tuple.
- Shape checks for multiple channel counts: 19, 22, 32, 62.
- `edge_logits` is symmetric or explicitly documented if pre-symmetry.
- All outputs are finite.
- One backward pass works.
- `graphcmi` with `lambda_g=lambda_n=lambda_e=0` behaves like ERM up to stochastic tolerance.

### 2.2 Graph leakage diagnostics

Add a dedicated file:

```text
cmi/eval/graph_leakage.py
```

Functions:

```python
graph_feature_cache(backbone, X, device, bs=512)
fit_node_leakage_probe(node_Z, y, d, n_cls, n_dom, priors, ...)
fit_edge_leakage_probe(edge_logits, y, d, n_cls, n_dom, priors, ...)
node_leakage_map(...)
edge_leakage_score(...)
permutation_null(...)
```

Diagnostics must work with frozen source splits. They should not require target labels or target covariates in strict DG main evaluation.

### 2.3 Result schema

Add a structured result schema under:

```text
results/cigl/schema.md
```

Mandatory fields:

```text
dataset
protocol
backbone
method
lambda_g
lambda_node
lambda_edge
seed
target_subject
source_subjects
balanced_acc
macro_f1
worst_subject_bacc
graph_leakage_kl
node_leakage_mean
edge_leakage_kl
label_separability
ece
nll
commit_hash
config_hash
```

For maps:

```text
node_leakage_map_path
edge_leakage_matrix_path
```

### 2.4 Experiment matrix

Create:

```text
docs/CIGL_03_EXPERIMENT_MATRIX.md
```

This should be generated or manually maintained with status labels:

```text
TODO / RUNNING / DONE / FAILED / DO_NOT_USE
```

Each row must include dataset × protocol × method × seed × result path.

---

## 3. Code cleanup priorities

### Priority 1: make existing GraphCMI testable

Do not add new methods until the current `GraphCMINet` and graph regularizers are tested.

### Priority 2: separate strict DG from TTA

CITA and label-shift transductive code must remain available, but CIGL scripts and results must clearly mark:

```text
setting = strict_source_only_DG
```

CITA belongs to:

```text
setting = transductive_TTA
```

Never mix them in a main table.

### Priority 3: logging

The trainer must log all three CIGL components:

```text
loss_ce
reg_graph
reg_node
reg_edge
lambda_g
lambda_node
lambda_edge
stepA_graph_dom_acc
stepA_node_dom_acc
stepA_edge_dom_acc
```

If these are not logged, failed experiments cannot be reviewed.

### Priority 4: graph-map artifacts

For each completed fold, save:

```text
node_leakage_map.npy
edge_leakage_summary.json
edge_leakage_matrix.npy  # optional first version
```

If maps are not stable, the paper claim must be downgraded.

---

## 4. First pull request scope

The first PR should not run full experiments. It should only do:

1. Add project docs.
2. Add README active-project banner.
3. Add/repair tests.
4. Add smoke script.
5. Add result schema.
6. Add loss breakdown logging.

First PR acceptance:

- `pytest tests/test_graphcmi_backbone.py tests/test_graph_regularizers.py` passes.
- `python scripts/smoke_graphcmi.py --device cpu` passes.
- No unrelated CITA/DualPC behavior changes.
