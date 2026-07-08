# P6 Official SPDIM W1 Seed-0 Same-Split Expansion Protocol

## Scope

- Label: W1 seed-0 same-split official SPDIM expansion, not full three-seed baseline.
- Datasets: `BNCI2014_001`, `Cho2017`, `Lee2019_MI` (`Lee2019-MI` in PM wording).
- Source seed: `0` only.
- Target subjects: all W1 target subjects for each dataset.
- Methods: `source_only_tsmnet`, `rct`, `spdim_geodesic`, `spdim_bias`.
- No seeds `1` or `2`.
- No full three-seed SPDIM baseline.
- No Cho2017/Lee2019-only side run outside this P6 scope.
- No TeX edits, orthogonal-score implementation, or geometry stress.

## Split

The split is the existing H2CMI W1-style LOSO split:

- For each target subject, source training uses every other subject in the same dataset.
- For the target subject, the earliest session is selected.
- Target adaptation/evaluation are disjoint contiguous blocks from that target session using `contiguous_split`.
- If multiple runs are available, first-half runs are adaptation and second-half runs are evaluation.
- If only one run is available, the first contiguous half of trials is adaptation and the second contiguous half is evaluation.

Target labels are not available to adaptation, refit, subsampling, filtering, or model selection. They are read only by final evaluation and dry-run split auditing.

## Dry-Run Gate Outcome

The CPU-only P6A gate passed and approved GPU launch.

| dataset | target subjects | tensor shape | expected rows | eval class-count range |
|---|---:|---|---:|---|
| `BNCI2014_001` | 9 | `[2592, 22, 500]` | 36 | `[36,36]` for every target |
| `Cho2017` | 52 | `[10520, 64, 500]` | 208 | class-0 `0`, class-1 `100-120` |
| `Lee2019_MI` | 54 | `[10800, 62, 500]` | 216 | class-0 `21-30`, class-1 `20-29` |

Total expected rows: `460`.

The Cho2017 class-count pattern is an explicit consequence of the exact contiguous W1 split and is not treated as a shape/model/no-label blocker. It must be kept visible in interpretation because these rows are not balanced like BNCI2014-001.

Machine-readable split evidence is in:

```text
h2cmi/results/review_completion/spdim_w1_seed0_dryrun_audit.json
```

It contains exact target subject IDs, source subject IDs per target, adaptation/evaluation trial indices, split SHA-256 values, tensor shapes, channel lists, and evaluation class counts.

## Provenance And External Code

- Official SPDIM repository: `https://github.com/fightlesliefigt/SPDIM`
- External SPDIM commit: `1b0de0ccd4c48a4ff28f087b866a0b671b029c39`
- External path: `/home/infres/yinwang/.cache/h2cmi_external/SPDIM_1b0de0ccd4c48a4ff28f087b866a0b671b029c39`
- No third-party SPDIM code is vendored into this repository.
- No official pretrained weights are used.
- Official SPDIM/TSMNet model instantiation passed for every dataset shape.
- One CPU forward pass without target labels passed for every dataset.

## GPU Run Plan

Launch only after the P6A commit is pushed and the worktree is clean.

Command:

```bash
sbatch h2cmi/results/review_completion/slurm/spdim_w1_seed0.slurm
```

The Slurm script runs one clean-guarded Python process:

```bash
python -m h2cmi.run_spdim_w1_seed0 --mode run --datasets BNCI2014_001,Cho2017,Lee2019_MI --seed 0
```

The runner refuses launch if `git status --porcelain=v1 --untracked-files=all` is nonempty, or if `--allow-dirty` is provided.

Estimated GPU time: `18.0` hours. Slurm walltime request: `48:00:00`.

## Required Outputs

- `h2cmi/results/review_completion/spdim_w1_seed0_results.csv`
- `h2cmi/results/review_completion/spdim_w1_seed0_audit.md`
- `h2cmi/results/review_completion/spdim_w1_seed0_summary.json`
- `h2cmi/results/review_completion/spdim_w1_seed0_result_digest.md`
- `h2cmi/results/review_completion/spdim_w1_seed0_failure_trace.txt` if blocked
- updated `h2cmi/results/review_completion/COMMAND_LOG.md`

## Monitoring And Validation

Use `squeue` only. Do not use `sacct`.

Completion rule:

```text
job absent from squeue + stderr/stdout status + artifact parse/count/checksum validation passed
```

Required validation:

- final job absent from `squeue`;
- stderr empty or only declared harmless warnings;
- stdout exists;
- CSV parses;
- expected row count is `460`;
- checksum recorded;
- JSON parses;
- no `sacct` script calls;
- clean provenance JSON consistency;
- no target-label leakage;
- no official pretrained weights;
- no vendored third-party code;
- `git show --check`.
