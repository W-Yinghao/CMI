# P8 SPDIM Repaired Seed-0 Shard Evidence

This directory contains the four result-carrying shard CSVs, their machine-readable summaries, and their Slurm stdout/stderr evidence used by the final postprocessor.

- Shard CSV and summary bytes are copied unchanged from the repository-external clean-run cache.
- Text logs have trailing horizontal whitespace stripped so repository whitespace validation passes; no non-whitespace content was changed.
- The final summary records a SHA-256 for every committed input.
- Job `891435` is excluded and is not represented in these accepted shard files.

| shard | rows | result sha256 | summary sha256 | stdout sha256 | stderr sha256 |
|---|---:|---|---|---|---|
| shard0 | 116 | `6837ac032d66623daf0fdafccc58530db3ac5798c5111a1d9b90e4bd7774b87c` | `b1dfb1918dece3d67be2faa7245bd82a7a51dfd3d8eaa33f716d3e68819e8213` | `61468c01ee2e02e13eb23b35931955d533c0d49973d0b03111acddd96d3a028f` | `73ab8fa6eeacd7956bc5025e0c7f8b56257e61a2f62e3c24c73ad99d3cd8b28e` |
| shard1 | 116 | `2e3c81ed3e98c5fd4d8949eb9d325635c75b48e66a12ea7f7aff7813a303e1da` | `45c294963425a3d92180012db162c4d4e101dbd2b2b60b6a9f607b75e62139df` | `8963c3396c7f3e7346d9e4b25ccd660feb37a736b544d292f2b1eb481d1761f1` | `376ac19e817966624e151b038f2439ed5971820e01e5430b00a09bc8f942be27` |
| shard2 | 116 | `6249945538afae247be4b00f0df0e86e867272852d108dd8625f4741f37d21ea` | `5a08525e879400483c8aa54763a67e63f47b6e73facb96bd5c376328dd67245c` | `74b9f05289ef68edb472a15a40cd58acb3afc30e0471c16788a226ffb13a9cbb` | `73ab8fa6eeacd7956bc5025e0c7f8b56257e61a2f62e3c24c73ad99d3cd8b28e` |
| shard3 | 112 | `297fc0619187edf96396685906becdd5a4323db5ad6a31b845bd828dbf9cdbd1` | `f52695cdc118a6f3ad13a5d08f2dac818673521bb125e4af17b8e34156081fdf` | `a8d820ad4d0574d4fa6117e34cc061bf2b23e2020a5db2f3950f1c95d0f9df9d` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

## Red Team Review

- These are raw execution outputs, not reconstructed predictions.
- Final acceptance depends on the postprocessor's combined coverage, provenance, and leakage gates.
