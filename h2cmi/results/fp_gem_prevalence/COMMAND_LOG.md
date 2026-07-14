# P13 Command Log

## P13A CPU Freeze

```bash
/home/infres/yinwang/anaconda3/envs/icml/bin/python -m h2cmi.prepare_fp_gem_prevalence
/home/infres/yinwang/anaconda3/envs/icml/bin/python -m h2cmi.run_fp_gem_prevalence --mode dry-run
/home/infres/yinwang/anaconda3/envs/icml/bin/python -m py_compile h2cmi/prepare_fp_gem_prevalence.py h2cmi/run_fp_gem_prevalence.py h2cmi/analyze_fp_gem_prevalence.py
```

The `icml` environment contains PyTorch but not pytest; base contains pytest but not PyTorch. The 13 pure-assert P12/P13 tests were therefore imported and called directly under the `icml` interpreter. All 13 passed.

No GPU job had been submitted at the time of the P13A freeze. Future monitoring is `squeue` only; completion requires queue absence plus stdout/stderr and artifact parse/count/checksum validation.

## Checkpoint-Reuse Gate

```bash
sbatch --parsable --partition=V100 \
  --export=ALL,FP_GEM_P13_REPO=/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/repo_7b48813 \
  scripts/fp_gem_prevalence_checkpoint_gate.slurm
# 894726

squeue -h -j 894726 -o '%i|%T|%P|%M|%R'
```

Job 894726 launched from clean commit `7b48813e1430a700a8246604e9def58772e09a4f` on `node42`, Tesla V100-PCIE-32GB. Final `squeue` output was empty. The gate JSON parsed with status `pass`, stdout passed provenance validation, stderr was empty, and all checkpoint/density/q=0.5 method/geometry hashes matched P12. No performance metric or evaluation label was read.

## P13B Fleet

The accepted full run launched from clean commit `afa21f2e9ae3b448bab271f30a399eb6cad765b0`:

```bash
sbatch --parsable --partition=V100 --array=0-5 \
  --export=ALL,FP_GEM_P13_REPO=/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/repo_afa21f2,FP_GEM_P13_GROUP=V100,FP_GEM_P13_GROUP_COUNT=134,FP_GEM_P13_GROUP_STRIDE=6 \
  scripts/fp_gem_prevalence_array.slurm
# 894784

sbatch --parsable --partition=A100 --array=0-1 \
  --export=ALL,FP_GEM_P13_REPO=/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/repo_afa21f2,FP_GEM_P13_GROUP=A100,FP_GEM_P13_GROUP_COUNT=28,FP_GEM_P13_GROUP_STRIDE=2 \
  scripts/fp_gem_prevalence_array.slurm
# 894790
```

Exact frozen retries used jobs `894841`, `894863`, `894879`, `894897`, `894903`, `894918`, and `894921`. Pending zero-result jobs `894872` and `894902` were canceled. The complete commands, node constraints, roles, and checksums are preserved in `fp_gem_prevalence_submission_record.json` and `fp_gem_prevalence_execution_audit.md`.

Monitoring used:

```bash
squeue -h -j 894726,894784,894790,894841,894863,894872,894879,894897,894902,894903,894918,894921 -o '%i|%T|%M|%N'
```

Final output was empty after `162/162` raw units existed. Completion was then gated on stdout, stderr, JSON parse, expected row counts, key coverage, and checksums.

## Aggregation And Red Team

```bash
/home/infres/yinwang/anaconda3/envs/icml/bin/python -m h2cmi.analyze_fp_gem_prevalence \
  --raw-root /home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13 \
  --out-dir h2cmi/results/fp_gem_prevalence \
  --job-record /home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/submission_record.json

/home/infres/yinwang/anaconda3/envs/icml/bin/python -m h2cmi.redteam_fp_gem_prevalence \
  --raw-root /home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13 \
  --out-dir h2cmi/results/fp_gem_prevalence \
  --report-json h2cmi/results/fp_gem_prevalence/P13_FINAL_RED_TEAM.json
```

The independent red team reloaded all Lee2019_MI evaluation labels, recomputed all 2,916 metrics and all frozen subject-bootstrap endpoints, and passed with zero mismatches.
