# Bounded BNCI2014-001 SPDIM Expansion Protocol

Status: gated by `spdim_probe_integrity_audit.json`, with
`approve_bnci001_expansion=true`.

This is a bounded expansion of the accepted official-SPDIM feasibility probe.
It is not the full W1 SPDIM sweep.

## Scope

- Dataset: `BNCI2014_001` only.
- Split: H2CMI W1-style LOSO.
- Source seed: `0` only.
- Target subjects: all BNCI2014-001 subjects discovered from H2CMI metadata.
- Methods: `source_only_tsmnet`, `rct`, `spdim_geodesic`, `spdim_bias`.
- No Cho2017 run.
- No Lee2019-MI run.
- No additional seeds.
- No official pretrained weights.
- No target-label-based subject filtering.

## Split Rule

For each target subject, all non-target subjects form the labeled source pool.
The target subject's first session is split by H2CMI `contiguous_split(...)`
into:

- target adaptation split: first contiguous half of the target first session;
- target evaluation split: second contiguous half of the target first session.

The target adaptation dataset is constructed with dummy labels. Target labels
are used only after each method has completed adaptation/refit, inside final
evaluation metrics and artifact hashing.

## External Code Policy

- Official external repository: `https://github.com/fightlesliefigt/SPDIM`
- Required external SHA: `1b0de0ccd4c48a4ff28f087b866a0b671b029c39`
- External checkout path:
  `/home/infres/yinwang/.cache/h2cmi_external/SPDIM_1b0de0ccd4c48a4ff28f087b866a0b671b029c39`
- Import mechanism: `PYTHONPATH` only.
- Vendoring policy: no `spdnets/` or SPDIM source copy under `h2cmi/`.
- Pretrained policy: do not use official BNCI2015_001 pretrained weights.

## Output Contract

Results are written to
`h2cmi/results/review_completion/spdim_bnci001_results.csv` using the
`bnci001` result schema. Required row columns are:

- `dataset`
- `target_subject`
- `source_seed`
- `method`
- `n_eval`
- `class_counts_eval`
- `acc`
- `bacc`
- `prediction_hash`
- `logits_hash`
- `status`
- `failure_reason`

The run also records protocol/provenance fields, source/adaptation/evaluation
index hashes, timing fields, source model hash, and macro-F1.

Expected successful row count is `9 * 4 = 36`.

## Slurm Monitoring And Validation

Do not use `sacct` on this server.

Accepted completion requires:

- Slurm job absent from `squeue`;
- stderr exists and is empty or contains only known harmless warnings;
- stdout exists;
- expected artifacts exist;
- CSV parses successfully;
- expected row count is present;
- all expected target/method rows are present;
- prediction/logit hashes are non-empty for every successful row;
- artifact checksum is recorded;
- no target-label leakage audit failure.

## Submit Command

```bash
sbatch h2cmi/results/review_completion/slurm/spdim_bnci001.slurm
```
