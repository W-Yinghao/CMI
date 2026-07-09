# SPDIM W1 Repaired-Split Seed-0 Protocol

Label: W1 repaired-split seed-0 official SPDIM expansion, not full three-seed baseline.

## Scope

- datasets: `BNCI2014_001`, `Cho2017`, `Lee2019_MI`.
- split: frozen P7 `class_stratified_half` repaired W1 manifest.
- manifest_hash: `231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e`.
- source seed: `0` only.
- methods: `source_only_tsmnet`, `rct`, `spdim_geodesic`, `spdim_bias`.
- no SPDIM seeds 1/2.
- no full three-seed SPDIM baseline.
- no official pretrained weights.
- no vendored third-party SPDIM code.

## Runtime Label Policy

Target labels are used only by the frozen split manifest and final evaluation metrics. Target adaptation loaders are built with dummy labels, and method selection is not based on target performance.

## Expected Rows

| dataset | targets | methods | expected rows |
|---|---:|---:|---:|
| BNCI2014_001 | 9 | 4 | 36 |
| Cho2017 | 52 | 4 | 208 |
| Lee2019_MI | 54 | 4 | 216 |
| total | 115 | 4 | 460 |

## Monitoring And Validation

- use `squeue` only; do not use `sacct`.
- final job state must be absent from `squeue`.
- stderr must be empty or contain only declared harmless warnings.
- stdout must exist and record clean launch provenance.
- CSV parse, row count, dataset row counts, JSON parse, checksums, no single-class eval, adapt/eval disjointness, P7 manifest hash, no target-label leakage, no target-performance method selection, no official pretrained weights, no vendoring, prediction/logits hash completeness, and `git show --check` must pass.

## Red Team Review

- Old W1 and old SPDIM P6 remain legacy diagnostic only.
- This protocol does not approve SPDIM seeds 1/2 or full SPDIM.
- Launch is blocked unless the P8A dry-run gate passes and the post-P8A worktree is clean.
