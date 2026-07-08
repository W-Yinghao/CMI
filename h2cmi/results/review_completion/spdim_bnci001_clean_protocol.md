# P5.2 Clean-Provenance BNCI2014-001 SPDIM Rerun Protocol

## Scope

- Dataset: `BNCI2014_001` only.
- Source seed: `0` only.
- Target subjects: all 9 BNCI2014-001 subjects, evaluated LOSO.
- Methods: `source_only_tsmnet`, `rct`, `spdim_geodesic`, `spdim_bias`.
- Runner commit at launch: `a8b93682c152a428f9689f9941efaff486606336`.
- External SPDIM checkout: `fightlesliefigt/SPDIM` at `1b0de0ccd4c48a4ff28f087b866a0b671b029c39`.
- No official pretrained weights.
- No third-party code vendoring.
- No Cho2017, Lee2019-MI, extra seeds, full W1 sweep, geometry stress, orthogonal-score implementation, or TeX edits.

## Split

The split is `W1_LOSO_first_session_contiguous_split`.

For each target subject:

- Source train subjects are all other BNCI2014-001 subjects.
- Target adaptation uses the first contiguous half of the target first session.
- Final evaluation uses the held-out contiguous half of the target first session.
- Target labels are not available to adaptation, refit, subsampling, filtering, or model selection. They are read only by the final evaluation metric code.

Each target subject has `n_eval=72` with `class_counts_eval=[36,36]`; therefore ordinary accuracy and balanced accuracy are equal for every row.

## Clean Launch Guard

Confirmatory SPDIM launches require an empty:

```bash
git status --porcelain=v1 --untracked-files=all
```

The clean run was launched from a detached clean worktree at:

```text
/home/infres/yinwang/CMI_AAAI_spdim_clean_a8b9368
```

The runner refused dirty launches by default. `runner_dirty_allowed=false` for the clean run.

## Slurm Monitoring

Slurm monitoring uses `squeue` only. `sacct` is not used on this server.

Completion rule:

```text
job absent from squeue + artifact parse/count/checksum validation passed
```

Submitted job:

```bash
sbatch h2cmi/results/review_completion/slurm/spdim_bnci001_clean.slurm
```

Slurm job id: `889192`.

## Required Artifacts

- `h2cmi/results/review_completion/spdim_bnci001_clean_results.csv`
- `h2cmi/results/review_completion/spdim_bnci001_clean_audit.md`
- `h2cmi/results/review_completion/spdim_bnci001_clean_summary.json`
- `h2cmi/results/review_completion/spdim_bnci001_clean_compare_to_exploratory.csv`

Expected result row count is 36 rows: 9 target subjects times 4 methods.
