# B5.1 — LEARNED SSL encoder canary (diagnostic-only)

```
Scope: B5.1 SSL feature canary | development diagnostic only | NOT deployable | NO tag | NOT confirmatory
  B3 certifier + injection BYTE-UNCHANGED | SSL trained LABEL-FREE | labels regenerated on Z_ssl | base 40e6
```

**Question:** the B5.0 random encoder was a near-isometry no-op; does a LEARNED representation change the certifier's
fitted-null FAIL / give concept-vs-covariate separation? **Answer (red-team-corrected):** even a learned label-free SSL
EEGNet encoder (masked-reconstruction, MSE 26.82→17.73) leaves the fitted-null under-dispersion FAIL **fully intact**
(NULL_cov false-confirm 0.225 ≈ 9× budget, fixed-margin p floor-pinned 92.5%, T_z 6–9, no-concept>concept inversion),
and does NOT clear the positive-control gate (POS_concept 0.30 vs NULL_cov 0.225, Fisher p=0.18, **~24% power** — the
n≥60 bar confirms a FAIL, it does not power the separation test). So this is **inconclusive/underpowered** on
separation, not a strong "learned features fail"; "feature richness is not the lever" is scoped to random + this one
SSL encoder.

## Contents
- `b5_1_feature_manifest.json`, `b5_1_cache_checks.json` — frozen SSL cache (npz sha `04d07143…`, weight `800d7b72…`,
  seed 20260707, masked-recon 60ep; raw 16MB npz not committed — see `LEE2019_B5_1.npz.POINTER.txt`).
- `canary/` — 6 per-condition JSONL (n=80) + prov/sha; `b5_1_canary_tables.json` (with red_team_caveat);
  `b5_1_redteam_checks.json` (posfirst PASS / overclaim MINOR_ISSUE).
- `scripts/` — `b5_1_build_cache.py`, `realeeg_b5_canary.py` (env-parametrized cache), `b5_1_canary_merge.py`
  (positive-control-first), `realeeg_internal_forensic.py`, sbatch.

Cache red-team CLEAN (label-free SSL PASS + integrity PASS). See notes/b5_1_ssl_encoder.md. Related:
b5_0_random_encoder.md, router_r1_scaleup.md, p3_oracle_diagnostic.md.
