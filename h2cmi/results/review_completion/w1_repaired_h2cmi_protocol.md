# W1 Repaired H2CMI Protocol

- status: P7A DRY-RUN GATE ONLY
- launch base commit: `f001a9260c71af251daeb7d092861bf319a9d829`
- split_family: `class_stratified_half`
- manifest_hash: `231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e`
- datasets: `['BNCI2014_001', 'Cho2017', 'Lee2019_MI']`
- source seeds: `[0, 1, 2]`
- expected rows: `3450`
- old W1 raw source: `/home/infres/yinwang/CMI_AAAI_qxu/results/h2cmi/p0_w1_all.jsonl`
- old W1 raw sha256: `c1ef4a6f4bb52a0561e14b5c26d1b40290f5ac44c4a7a86876f97639f2112633`

## Runtime Scope

- H2CMI W1 only.
- No SPDIM, Cho/Lee rerun outside H2CMI, extra methods, TeX edits, geometry stress, or orthogonal-score work.
- Use `h2cmi.run_w1_repaired_p0` with the frozen manifest only.
- Use source seeds 0, 1, and 2 from the original W1 source-training policy.
- Output branches are the original W1 P0 branches plus `__decomposition__`; no new method is added.

## Split Policy

- Target labels are used only to construct and freeze `class_stratified_half` before any model run.
- Runtime adaptation receives target adaptation trials without labels.
- Target evaluation labels are used only for final metrics.
- No target-label-based model selection, method selection, early stopping, or subsampling is allowed after manifest freeze.

## Clean Run Policy

- Launch only after this P7A commit is pushed and the worktree is clean.
- Record launch commit, manifest hash, runner checksum, config checksum, command line, and Slurm job IDs.
- Use `squeue` only for monitoring.
- Do not use Slurm accounting commands.

## Validation Gates

- final job absent from `squeue`
- stderr empty or only declared harmless warnings
- stdout exists
- result CSV parses
- summary JSON parses
- expected rows = 3450
- no single-class eval rows
- all adapt/eval trial IDs disjoint
- manifest hash matches this P7A hash
- prediction hashes complete
- clean provenance JSON consistency

## Red Team Review

- This protocol repairs the main H2CMI W1 evidence first, as requested.
- It does not approve SPDIM expansion.
- Dry-run approval is necessary but not a result claim.
