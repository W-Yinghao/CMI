# B4c Stage-1b canary scripts (diagnostic-only)

```
Scope: B4c Stage-1b canary | development diagnostic only | RICHER SHARED NUISANCE TRUNK only
  interaction channel c x PC1:3 UNCHANGED | estimand CE(h0)-CE(h1) unchanged | SM16 features unchanged
  no overlap gate | no richer interaction | no new features/montage | no new statistic
  no oracle generator in fitting or null | not deployable | no confirmatory tag | no method promotion
```
Candidate B4c-Q3: shared trunk = [Z1:16, c, u1^2,u2^2,u3^2, u1u2,u1u3,u2u3] (u=top-3 PC of weighted-std Z) in
BOTH h0 and h1; h1 adds the UNCHANGED c x PC1:3. C=0.25 (predeclared), B=200. Because the shared family changes,
observed T MAY change -> logged (archived_B3_observed_T vs b4c_observed_T) + observed_T_repro_ok (determinism).
- realeeg_b4c_stage1b.py : worker  - realeeg_b4c_merge.py : fail-closed merge  - realeeg_b4c_tables.py : screens
- b4c_shard.sbatch / b4c_watcher.sh : SLURM shard + throttled watcher
