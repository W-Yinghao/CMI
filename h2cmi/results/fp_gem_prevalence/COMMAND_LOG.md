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
