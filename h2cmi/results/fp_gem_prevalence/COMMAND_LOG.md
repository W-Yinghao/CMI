# P13 Command Log

## P13A CPU Freeze

```bash
/home/infres/yinwang/anaconda3/envs/icml/bin/python -m h2cmi.prepare_fp_gem_prevalence
/home/infres/yinwang/anaconda3/envs/icml/bin/python -m h2cmi.run_fp_gem_prevalence --mode dry-run
/home/infres/yinwang/anaconda3/envs/icml/bin/python -m py_compile h2cmi/prepare_fp_gem_prevalence.py h2cmi/run_fp_gem_prevalence.py h2cmi/analyze_fp_gem_prevalence.py
```

The `icml` environment contains PyTorch but not pytest; base contains pytest but not PyTorch. The 13 pure-assert P12/P13 tests were therefore imported and called directly under the `icml` interpreter. All 13 passed.

No GPU job had been submitted at the time of the P13A freeze. Future monitoring is `squeue` only; completion requires queue absence plus stdout/stderr and artifact parse/count/checksum validation.
