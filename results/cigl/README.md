# `results/cigl/` — CIGL / GraphCMI experiment outputs

This directory holds **only** CIGL (Conditional Information Graph Learning) results — the
strict source-only EEG domain-generalization project that studies label-conditional domain
leakage in EEG graph representations at graph, node, and edge levels.

It is kept separate from the historical Tri-CMI / LPC-CMI results (top-level `results/`,
`archive/lpc-cmi-failed/`) and from any CITA transductive/TTA results. **Do not** drop
transductive (`setting = transductive_TTA`) records here, and do not combine CIGL and CITA
rows in one table — see `docs/CIGL_05_ACCEPTANCE_CRITERIA.md` (absolute red lines).

## Layout

```
results/cigl/
  README.md            # this file
  schema.md            # canonical per-record field contract (READ THIS FIRST)
  <dataset>_<backbone>_<protocol>.json   # one runner invocation: a method sweep over shared splits
  <dataset>_<backbone>_<protocol>.preds.npz   # per-fold probability sidecar (recompute ECE/NLL w/o GPU)
  maps/
    <run>__node_leakage_map.npy          # length-C per-channel leakage map (diagnostic)
    <run>__edge_leakage_matrix.npy       # C×C per-edge leakage (diagnostic-only in v1)
```

The JSON layout mirrors the existing harness (`cmi/run_loso.py` writes
`{config, classes, summary}` with a per-config `summary[label]`); every CIGL record must
additionally carry the fields in [`schema.md`](schema.md), especially `setting`,
`feature_protocol`, the `lambda_g`/`lambda_node`/`lambda_edge` triple, `commit_hash`, and
`config_hash`.

## How CIGL runs are launched

The method config grammar is `graphcmi:<lambda_g>:<lambda_node>:<lambda_edge>` (parsed in
`cmi/run_loso.py`; the trainer maps these to its internal `lam`/`gamma`/`lam_edge`). The
mandatory ablation ladder (`docs/CIGL_01_METHOD_SPEC.md` §5):

```
graphcmi:0:0:0          # GraphCMI-ERM
graphcmi:0.3:0:0        # global (graph) only
graphcmi:0.3:0.3:0      # graph + node
graphcmi:0.3:0:0.1      # graph + edge
graphcmi:0.3:0.3:0.1    # full CIGL
```

> Phase 0/1 status: infrastructure and tests only. **No real EEG results live here yet** —
> the first results land after the Phase-2 probe-only leakage hypothesis passes Gate 2.
> Until then this directory documents the schema the runs will conform to.
