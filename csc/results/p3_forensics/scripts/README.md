# P3 forensics reproducibility scripts (diagnostic-only)

```
Scope:
  P3.0b/c + P3.0d forensic archive
  development diagnostic only
  no method change | no certifier change | no h1/statistic/feature/montage change
  no B4 canary | no confirmatory tag | no deployment claim
```

Archival copies of the exact dev workers/mergers/sbatch that produced the P3 forensic artifacts.
diagnostic_only=true; not_deployable=true; no_method_change; no_B4. Paths inside are hardcoded to the dev
scratch (/home/infres/yinwang/realeeg_feas); these are for provenance/reference, not to run from the repo.

- realeeg_failure_forensics.py / forensics_tables.py  : P3.0   raw-Z overlap forensic (+ 4 tables)
- realeeg_internal_forensic.py                         : P3.0b/c certifier-internal worker (per shard = 100 cohorts)
- realeeg_forensic_merge.py                            : P3.0b/c fail-closed merge (1800)
- forensic_tables_internal.py                          : P3.0b/c 4 judgment tables
- realeeg_oracle_forensic.py                           : P3.0d ORACLE-generator fixed-margin worker (in-process
                                                         method certify + oracle null; strict fidelity; invalid-cohort N/A)
- realeeg_oracle_merge.py                              : P3.0d fail-closed merge (1200, all fidelity_ok)
- oracle_tables.py                                     : P3.0d Tables A-D
- p3_shard.sbatch / p3_watcher.sh                      : P3.0b/c SLURM shard + throttled watcher
- p3_oracle_shard.sbatch / p3_oracle_watcher.sh        : P3.0d  SLURM shard + throttled watcher

Oracle null = Y* ~ _pooled_clf(coh_Z, coh_y) (true session-independent bank generator), fixed-margin, same
statistic path as the byte-frozen certifier; only logp0 swapped. Uses the true DGP -> diagnostic, NOT deployable.
