# B4 Stage 1 canary scripts (diagnostic-only)

```
Scope: B4 Stage 1 canary | development diagnostic only | NULL-GENERATION only
  observed T / studentized / folds / features / margins / invalid accounting UNCHANGED
  no overlap gate | no richer h1 | no feature/montage change | no new statistic
  no oracle generator in any candidate arm | not deployable | no confirmatory tag
```
Arms: method (existing fixed-margin full-audit-h0 null) ; B4a bagged h0 (K=50 subject-cluster bootstraps,
rotated per replicate) ; B4b nested h0 (K=B=200, fresh bag per replicate) ; variance-inflation baseline (diagnostic).
- realeeg_b4_stage1.py : worker (observed computed ONCE + shared; only the null h0 generator differs per arm)
- realeeg_b4_merge.py  : fail-closed merge (206 unique, 0 worker/fidelity fail)
- realeeg_b4_tables.py : canary pass-criteria tables
- b4_shard.sbatch / b4_watcher.sh : SLURM shard + throttled watcher
