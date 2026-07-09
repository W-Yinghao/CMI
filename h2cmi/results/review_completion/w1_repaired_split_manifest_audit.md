# W1 Repaired Split Manifest Audit

- split_family: `class_stratified_half`
- manifest_hash: `231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e`
- manifest_rows: `345`
- target_units: `115`
- expected_h2cmi_rows: `3450`
- source_seeds: `[0, 1, 2]`
- labels_used_only_for_split_construction: `True`
- target_labels_hidden_from_adaptation: `True`

## Dataset Summary

| dataset | targets | manifest rows | n_adapt values | n_eval values | eval counts |
|---|---:|---:|---|---|---|
| BNCI2014_001 | 9 | 27 | `[72]` | `[72]` | `[(36, 36)]` |
| Cho2017 | 52 | 156 | `[100, 120]` | `[100, 120]` | `[(50, 50), (60, 60)]` |
| Lee2019_MI | 54 | 162 | `[50]` | `[50]` | `[(25, 25)]` |

## Label Policy

`class_stratified_half` uses target labels only before model execution to freeze a benchmark split with both classes present in adaptation and evaluation. The manifest passed to runtime adaptation contains trial IDs and class counts; adaptation code receives target X/embeddings only, while evaluation labels are used only for final metrics.

## Red Team Review

- The manifest is immutable through `manifest_hash` and per-row `split_hash` values.
- Every row has disjoint adapt/eval IDs and both classes in both sides.
- This audit does not launch GPU work.
