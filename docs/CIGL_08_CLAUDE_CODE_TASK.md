# Claude Code Task: Start CIGL Project

This is the prompt to give Claude Code after the CIGL docs are placed in the repository.

---

## Working mode

You are working as implementation engineer under a strict project manager/reviewer protocol. Do not broaden the project. Do not convert this into CITA, DualPC, H²-CMI, SSL-CMI, or a generic GNN project.

The active project is:

> CIGL / GraphCMI — Conditional Information Graph Learning for Calibration-Free EEG Generalization.

CITA is an existing transductive/TTA branch and must remain separate.

---

## First task scope

Implement only **Phase 0 + Phase 1** from the CIGL docs.

Do not run full EEG experiments. Do not modify unrelated methods. Do not refactor the whole repository.

---

## Required reads before coding

Read these files first:

```text
README.md
cmi/models/gnn.py
cmi/methods/graph_regularizers.py
cmi/train/trainer.py
cmi/models/backbones.py
cmi/run_loso.py
```

Optional historical references (may be ABSENT in the current tree — do NOT recreate them just to
satisfy a reference; the canonical spec is `docs/CIGL_01_METHOD_SPEC.md`):

```text
PROJECT_SUMMARY.md      # historical Tri-CMI summary; not present on the CIGL branches
notes/gnn_design.md     # original GraphCMI design notes; superseded by docs/CIGL_01_METHOD_SPEC.md
```

Then read the new docs:

```text
docs/CIGL_00_PROJECT_CHARTER.md
docs/CIGL_01_METHOD_SPEC.md
docs/CIGL_02_CODEBASE_AUDIT.md
docs/CIGL_03_IMPLEMENTATION_PLAN.md
docs/CIGL_05_ACCEPTANCE_CRITERIA.md
```

---

## Concrete deliverables

### Deliverable 1 — README project banner

Add a concise banner near the top of `README.md`:

```md
## Active project: CIGL / GraphCMI
CIGL studies label-conditional domain leakage in EEG graph representations at graph, node, and edge levels. The original Tri-CMI/LPC-CMI work remains the foundation. CITA is a separate transductive/test-time alignment branch and is not the main strict-DG project.
```

Do not delete historical README content.

### Deliverable 2 — tests

Create:

```text
tests/test_graphcmi_backbone.py
tests/test_graph_regularizers.py
```

`test_graphcmi_backbone.py` must verify:

- GraphCMINet builds for channel counts 19, 22, 32, 62.
- `forward(x)` returns `(logits, graph_Z)`.
- `forward_graph(x)` returns `(logits, graph_Z, node_Z, edge_logits)`.
- Shapes are correct.
- All tensors are finite.
- One scalar loss backward pass works.
- `DGCNNBackbone` and `RGNNBackbone` still return `(logits, Z)`.

`test_graph_regularizers.py` must verify:

- `NodePosterior.step_a_loss` is finite.
- `NodePosterior.reg` is finite and backpropagates to `node_Z`.
- `NodePosterior.leakage_map` returns length `C`.
- `EdgePosterior.step_a_loss` is finite.
- `EdgePosterior.reg` is finite and backpropagates to `edge_logits`.
- Prior tensor handling works for empirical prior shape.

Use small synthetic tensors. CPU only.

### Deliverable 3 — smoke script

Create:

```text
scripts/smoke_graphcmi.py
```

The smoke script should:

1. build `GraphCMINet`;
2. generate random synthetic EEG tensor `X [N,C,T]`, labels `y`, domains `d`;
3. run one tiny `train_model(... method="graphcmi" ...)` call on CPU;
4. print loss diagnostics and output shape checks;
5. exit nonzero on NaN or shape failure.

Keep it small enough for a laptop CPU.

### Deliverable 4 — result schema

Create:

```text
results/cigl/README.md
results/cigl/schema.md
```

Include the mandatory fields listed in `docs/CIGL_02_CODEBASE_AUDIT.md`.

### Deliverable 5 — trainer hardening

Make the smallest safe changes needed so that:

- `method="graphcmi"` raises a clear error if the backbone lacks `forward_graph`.
- GraphCMI branch logs separate values for graph, node, and edge regularizers.
- User-facing diagnostics use names `lambda_g`, `lambda_node`, `lambda_edge` even if existing internal arguments are `lam`, `gamma`, `lam_edge`.
- Existing non-graph methods remain byte-for-byte or behaviorally unchanged.

### Deliverable 6 — optional config parsing repair

If `run_loso.py` does not already parse `graphcmi:<lambda_g>:<lambda_node>:<lambda_edge>`, add it. If adding parser logic is risky, leave a TODO and do not break current CLI.

---

## Do not do

- Do not run full SEED / SEED-IV / DEAP experiments.
- Do not implement per-edge R2 training heads.
- Do not add PyG.
- Do not touch CITA transductive code except to label it clearly in docs.
- Do not change old Tri-CMI behavior.
- Do not claim results.

---

## Acceptance command

At the end, the following should run:

```bash
pytest -q tests/test_graphcmi_backbone.py tests/test_graph_regularizers.py
python scripts/smoke_graphcmi.py --device cpu
```

---

## Final response format

Report:

1. Files changed.
2. Tests run and pass/fail output.
3. Any blockers.
4. Any behavior changes to existing methods.
5. Next recommended PR.
