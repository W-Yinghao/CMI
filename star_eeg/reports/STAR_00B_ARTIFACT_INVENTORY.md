# STAR_00B Artifact Inventory

## External runtime payloads (not in Git)

- Two H200 SHA-named immutable start checkpoints under `/home/infres/yinwang/CMI_AAAI_star_runtime/results/star_h200_starts_immutable/`, mode `0444`.
- Three step-10 diagnostic smoke checkpoints under `/home/infres/yinwang/CMI_AAAI_star_runtime/results/star00b_realpath_smoke/`.

No checkpoint payload is committed.

## Committed preflight artifacts

`results/star/star00b_preflight/` contains:

- `h200_immutable_manifest.json`, `h200_immutable_go_nogo.json`
- `faced_source_train_inventory.json`
- `anchor_manifest.json`, `shuffled_label_manifest.json`
- `anchor_exposure_table.csv`, `anchor_batch_stream_hashes.json`
- `ssl_batch_stream_hashes.json`, `compute_match_contract.json`
- `source_loader_firewall.json`, `realpath_runner_contract.json`
- `realpath_smoke_job.json`, `realpath_smoke_summary.json`, `realpath_smoke_telemetry.json`
- `runtime_memory_estimate.json`
- `blind_evaluation_chain.json`, `star01_training_tasks.csv`
- `star00b_dependency_manifest.json`, `star00b_launch_manifest.json`
- `star00b_preflight_summary.json`, `star00b_red_team.json`
- the small Slurm stdout/stderr records for preliminary jobs 892998/892999 and
  authoritative final-source job 893001

Full source manifests contain sample identifiers, labels, subjects, shapes, and integrity hashes, but no raw EEG array or feature dump. Telemetry contains only bounded smoke integrity values and hashes; it is not a scientific result table.
