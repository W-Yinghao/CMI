# B5.0 — frozen random deep-feature canary (diagnostic-only, NO-OP control)

```
Scope: B5.0 feature-robustness canary | development diagnostic only | NOT deployable | NO tag | NOT confirmatory
  B3 certifier + injection engine BYTE-UNCHANGED (only the feature input is swapped to a frozen deep embedding)
  encoder never sees any label | labels REGENERATED on Z_deep (not transplanted) | fresh dev seed base 30e6
```

**Question:** does the SM16 fitted-h0 under-dispersion FAIL (NULL_cov false-confirm) persist under a DEEP feature
family? **Answer (red-team-corrected):** a frozen random-init EEGNet embedding is a near-isometry that reproduces
SM16 on every axis — a **NO-OP control**. The FAIL reproduces unchanged (NULL_cov 10/50=0.20; fixed-margin p
floor-pinned 90%; T_z~6) AND the injected concept yields NO decision-level power (POS_concept 10/50=0.20 is
statistically IDENTICAL to NULL_cov; Fisher OR=1.0, p=1.0). With no separated positive control the canary is
**INCONCLUSIVE** for safety — it cannot adjudicate whether deep features resolve the FAIL. `any_pos_signal` in the
tables is MISLEADING (means only >=1 confirm, NOT power). The informative continuation is **B5.1 (learned/SSL
features)**: positive-control-FIRST, then safety, n>=60, subject-cluster bound.

## Contents
- `b5_0_feature_manifest.json`, `b5_0_cache_checks.json` — frozen cache provenance (npz sha256 `3670f174…`, weight
  sha256 `a9f2dc94…`, seed 20260706; preprocessing IDENTICAL to SM16). Raw 16MB npz not committed — see
  `LEE2019_B5_0.npz.POINTER.txt` (rebuild via `scripts/b5_0_build_cache.py`).
- `canary/` — 12 per-condition JSONL shards (6 conds x 2 cohort-halves, n=50/cond) + `.prov.json`/`.sha256`;
  `b5_canary_tables.json` (with `red_team_caveat`); `b5_redteam_checks.json` (fidelity PASS / persist FAIL / overclaim
  MINOR_ISSUE).
- `scripts/` — `realeeg_b5_canary.py` (thin CACHE swap over the byte-frozen `realeeg_internal_forensic.run_cohort`),
  `b5_canary_merge.py`, `b5_0_build_cache.py`, `realeeg_internal_forensic.py`, sbatch + lifecycle.

Cache red-team CLEAN (leakage PASS + integrity PASS: hash/shape/repro/preprocessing-parity/trial-inventory-vs-SM16).
See notes/b5_0_random_encoder.md. Related: p3_oracle_diagnostic.md, b4_stage1_canary.md, router_stage1_validation.md.
