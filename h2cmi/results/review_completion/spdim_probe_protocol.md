# P4 Frozen Protocol: Bounded Official SPDIM Feasibility Probe

Status: frozen before GPU compute.

## Question

Can the official external SPDIM implementation be executed on the frozen H2CMI
BNCI2014_001 motor-imagery split without target-label use before final
evaluation?

This is a feasibility probe only. It is not a full SPDIM benchmark and cannot be
used as a manuscript head-to-head table.

## External Code

- official repo: `https://github.com/fightlesliefigt/SPDIM`
- required commit SHA: `1b0de0ccd4c48a4ff28f087b866a0b671b029c39`
- external checkout path: `/home/infres/yinwang/.cache/h2cmi_external/SPDIM_1b0de0ccd4c48a4ff28f087b866a0b671b029c39`
- no-code-copy policy: official SPDIM code is not vendored into this repo; it is
  imported only through `PYTHONPATH`.
- pretrained weights: not used. The provided official BNCI2015_001/13-channel
  weights are incompatible with this H2CMI 22-channel split.

## Environment

- Python: `/home/infres/yinwang/anaconda3/envs/icml/bin/python`
- H2CMI worktree: `/home/infres/yinwang/CMI_AAAI_qxu`
- branch: `exp/h2cmi-wave0-mechanism`
- base commit for this protocol: `5fbc0300c1db82d22950b911bd0f6d209088e16e`
- device: one Slurm GPU job, `V100`, with 8 CPUs and 48 GB memory.
- time budget: 6 hours walltime. Exceeding this budget is a P4 blocker, not a
  reason to expand the run.

## Data And Splits

- dataset: `BNCI2014_001`
- preprocessing: H2CMI frozen `load_dataset` pipeline, native 22 EEG channels,
  500 time samples, binary left/right labels.
- protocol: W1-style LOSO. For each target subject, source data are all
  non-target subjects. Target adaptation/evaluation uses the target subject's
  first session and the H2CMI `contiguous_split`.
- source seed: `0`
- target-subject rule: first and last sorted target subject IDs, selected before
  running and using subject metadata only.
- selected target subjects: `1`, `9`

Frozen split IDs are SHA-256 hashes of comma-joined H2CMI epoch indices:

| target | source subjects | source n | target session | adapt n | eval n | source idx sha256 | adapt idx sha256 | eval idx sha256 |
|---:|---|---:|---:|---:|---:|---|---|---|
| 1 | 2 3 4 5 6 7 8 9 | 2304 | 0 | 72 | 72 | `e465e339076ddda8a860f1cf45b08ae0f29c0edb6d28ce5a64253851137c54f0` | `a795e5bc03ed5f1f953e4539af97da663030b8db4faa31de506ad5cdd98478a0` | `bd9a2c8fbf053af0e055841916ee360363cf260d55f96945c7db0b7a4c89e968` |
| 9 | 1 2 3 4 5 6 7 8 | 2304 | 0 | 72 | 72 | `f74bb71ff9113b5d5601f4f9b6ab0846db25b819402a53cdb2738628b3a9b928` | `aed590dfda5daa6583dd5d3b464b7647b1b347b119357bd1ccd65cfb987dc2a7` | `c6b7acaf45ac2d38863c0e83fbae97f79d679f0cd069d98c5d809ef6693e0b21` |

## Target Label Policy

Target labels are not used for adaptation, target subsampling, model selection,
early stopping, or subject selection. The target adaptation `Dataset` supplies
dummy zero labels because the official trainer API expects a `(features, y)`
batch, but SPDIM information-maximization routines use predictions only.
Target labels enter only inside the final metric computation on the evaluation
indices after each refit/adaptation mode has completed.

## Model And Channel Handling

- official model: `spdnets.models.TSMNet`
- `nclasses=2`, `nchannels=22`, `nsamples=500`
- temporal filters: `4`
- spatial filters: `40`
- SPD subspace dimensions: `20`
- SPD batchnorm dispersion: `SCALAR`
- official SPD layers run on CPU (`spd_device=cpu`, `spd_dtype=torch.double`),
  matching the official demo pattern; CNN/classifier use the Slurm GPU.
- domain IDs are H2CMI subject IDs. Each target subject is present as a target
  domain at model construction, then refit/adapted from unlabeled target inputs.

## Commands

CPU dry-run gate:

```bash
PYTHONPATH=/home/infres/yinwang/CMI_AAAI_qxu:/home/infres/yinwang/.cache/h2cmi_external/SPDIM_1b0de0ccd4c48a4ff28f087b866a0b671b029c39 \
MPLCONFIGDIR=/tmp/matplotlib-spdim-probe-dryrun \
/home/infres/yinwang/anaconda3/envs/icml/bin/python -m h2cmi.run_spdim_probe \
  --external-spdim-path /home/infres/yinwang/.cache/h2cmi_external/SPDIM_1b0de0ccd4c48a4ff28f087b866a0b671b029c39 \
  --subjects 1 \
  --device cpu \
  --dry-run \
  --allow-dirty \
  --audit /tmp/h2cmi_spdim_probe_dryrun_audit.md \
  --failure-trace /tmp/h2cmi_spdim_probe_dryrun_failure.txt
```

GPU feasibility probe:

```bash
sbatch h2cmi/results/review_completion/slurm/spdim_probe.slurm
```

Equivalent direct command inside the Slurm allocation:

```bash
/home/infres/yinwang/anaconda3/envs/icml/bin/python -m h2cmi.run_spdim_probe \
  --external-spdim-path /home/infres/yinwang/.cache/h2cmi_external/SPDIM_1b0de0ccd4c48a4ff28f087b866a0b671b029c39 \
  --dataset BNCI2014_001 \
  --subjects 1,9 \
  --seed 0 \
  --epochs 20 \
  --adapt-epochs 30 \
  --device cuda \
  --out h2cmi/results/review_completion/spdim_probe_results.csv \
  --audit h2cmi/results/review_completion/spdim_probe_audit.md \
  --failure-trace h2cmi/results/review_completion/spdim_probe_failure_trace.txt
```

## Outputs

- `h2cmi/results/review_completion/spdim_probe_results.csv`
- `h2cmi/results/review_completion/spdim_probe_audit.md`
- `h2cmi/results/review_completion/spdim_probe_failure_trace.txt` if blocked
- Slurm logs under `h2cmi/results/review_completion/slurm/logs/`

## Success Criteria

P4 succeeds only if all of the following hold:

- official `TSMNet` source models train from H2CMI BNCI2014_001 source data for
  both target subjects;
- target adaptation loaders run without target labels;
- `source_only`, `rct_refit`, `spdim_geodesic`, and `spdim_bias` rows are
  produced for each target subject;
- each produced row reports balanced accuracy and ordinary accuracy on the H2CMI
  evaluation split;
- command, external SHA, split IDs, and artifacts are reproducible from the
  audit file.

## Blocker Criteria

Stop and report P4 as blocked if any of the following occurs:

- channel-shape incompatibility cannot be fixed without changing official model
  semantics;
- official code requires target labels for adaptation/subsampling;
- no license-compatible no-vendor execution path is available;
- source training exceeds the predeclared 6-hour one-GPU budget;
- the environment conflict cannot be resolved without changing the global H2CMI
  environment.

## Interpretation Grid

- All success criteria pass: official SPDIM is feasible on the bounded H2CMI W1
  split; this authorizes only a feasibility statement, not a full benchmark
  claim.
- Import/model construction passes but training/adaptation fails under a blocker
  criterion: report as official-SPDIM feasibility blocked, with trace.
- Rows complete but performance is mixed or poor: still only feasibility; do not
  compare against manuscript tables without explicit approval for a same-split
  full benchmark.
- Any accidental target-label use before final metrics: invalidate the probe and
  report the violation.
