# C5 — Full-Bootstrap Staged One-Fold Timing & Inference-Path Validation

> **Full-bootstrap staged one-fold timing and inference-path validation.**
> **This is not final confirmatory efficacy evidence.**

The full confirmatory bootstrap (selection 200 / audit 2000 / paired 2000) run through the C4b staged
executor: Phase A (V100) trains + GPU-prefetches every feasible candidate, releases the V100; Phase B
(CPU) replays select → audit → finalize at full bootstrap with no GPU. A single fold + single seed is
descriptive only.

## Reuse check (per C5 Option A → Option B)

The C4b Phase-A store from job `876816` (reduced bootstrap) was checked for reuse before rerunning:

```
reused_existing_phase_a: false
existing_store_optimization_identity_matched: true
optimization_identity_hash: e17e15999fcbd8b0   (reduced == full)
rerun_reason: fold_scope / manifest / split_manifest / bootstrap-plan identities are part of the
              full-bootstrap manifest-bound fold identity (the audit plan lives in fold_scope, the
              selection plan in level_plans); per "do not weaken the checks", Phase A was rerun for
              coherent full-bootstrap provenance.
phase_a_job_id: 877374
```

**Empirical confirmation that the optimization identity is bootstrap-independent:** the reduced (876980)
and full (877446) staged runs produced **bit-identical training** — same ERM checkpoint hashes
(`1f9fb65266` L0 / `094598ab4c` L1), `R_ERM_hat` and `tau`, and the same feasible-candidate counts (38 / 60).
Only the leakage UCBs tightened at full bootstrap. So the reduced store *was* reusable in the training
sense; Phase A was rerun only so the official C5 artifact carries the full-bootstrap manifest identity.

## Job & code

| item | value |
|------|-------|
| commit | `7f5dacc` |
| git tree hash | `4ed52db09b4c4b54f36e413d5bab55e7c43b30d1` |
| Phase A (GPU record) | job `877374`, V100, exit 0 |
| Phase B (CPU replay) | job `877446`, nodecpu08, exit 0 |
| **V100-held wall-clock (Phase A)** | **≈ 1 h 37 m** (17:16 → 18:53) |
| **CPU full-bootstrap wall-clock (Phase B)** | **≈ 5 h 44 m** (18:53 → 00:37, **no GPU**) |
| **total wall-clock** | **≈ 7 h 21 m** |
| CPU workers / threads | 16 / 1 (`process_bootstrap_replicate`) |

## Store

| item | value |
|------|-------|
| reused existing Phase A | false (rerun 877374) |
| optimization_identity_hash | `e17e15999fcbd8b0` |
| staging size | 3.10 GB |
| feasible unique checkpoints | level 0 = **38**, level 1 = **60** |
| stored feature arrays | 196 (98 candidates × {source_train, source_audit}) |
| stored prediction arrays | 294 (98 × {source_guard, source_audit, target_audit}) |

## Budget

`stage1_epochs=200`, `stage2_epochs=200`; **selection_bootstrap = 200**, **audit_bootstrap = 2000**,
**paired_bootstrap = 2000**; `bootstrap_mode = full`; **not_confirmatory_evidence = true**.

## Artifact

| item | value |
|------|-------|
| `artifact_scientific_hash` | `3a590a9dabfb86acd52b2b3a2e4aad5ba2da6c71aa13263e714e04cbd8ca314c` |
| `artifact_pure_science_hash` | `d3e5918cdecc0681…` |
| `fold_result_hash` | `6303aa5de9c8fd2a…` |
| deep verification | **OK**, summary matches memory |
| indexed files | **548** |
| verified checkpoints / plans | **202 / 14** |
| `target_fit_ids` | **∅** |

## Science endpoints (descriptive — full bootstrap, 1 fold, seed 0)

Support: domains × 4 classes `[left_hand, right_hand, feet, tongue]`. Level 0 = no deletion; level 1 =
deleted cell `BNCI2014_001|subject-004 / feet` (count 0 / mass 0 / ≈144 rows removed). `R = selected risk`,
`ep = epoch` (−1 = ERM), `selLQ/selUCL` = selection `extractable_LQ_ov`/`bootstrap_ucl`, `audLQ/audUCL` =
audit ditto.

**Level 0** — `R_ERM_hat = 0.8458`, `tau = 0.8758`

| method | R | ep | selLQ | selUCL | audLQ | audUCL | srcA bAcc | tgt bAcc | tgt NLL |
|--------|--:|---:|------:|-------:|------:|-------:|----------:|---------:|--------:|
| ERM | 0.8458 | −1 | 1.5485 | 1.6837 | 0.6159 | 0.7418 | 0.375 | 0.510 | 1.279 |
| OACI | 0.8458 | 164 | 0.9500 | 1.2607 | 0.6014 | 0.8280 | 0.357 | 0.441 | 1.244 |
| global_lpc | 0.8573 | 149 | 1.0484 | 1.3360 | 0.5650 | 0.7897 | 0.370 | 0.460 | 1.216 |
| uniform | 0.8573 | 149 | 1.0484 | 1.3360 | 0.5650 | 0.7897 | 0.370 | 0.460 | 1.216 |

**Level 1** — `R_ERM_hat = 0.8129`, `tau = 0.8429`

| method | R | ep | selLQ | selUCL | audLQ | audUCL | srcA bAcc | tgt bAcc | tgt NLL |
|--------|--:|---:|------:|-------:|------:|-------:|----------:|---------:|--------:|
| ERM | 0.8129 | −1 | 1.5321 | 1.6562 | 0.6098 | 0.7325 | 0.390 | 0.498 | 1.221 |
| OACI | 0.7950 | 154 | 1.0409 | 1.3210 | 0.6302 | 0.8158 | 0.373 | 0.502 | 1.157 |
| global_lpc | 0.8225 | 99 | 1.1872 | 1.4511 | 0.6023 | 0.8586 | 0.364 | 0.460 | 1.229 |
| uniform | 0.8152 | 174 | 1.0836 | 1.3785 | 0.6253 | 0.9062 | 0.359 | 0.457 | 1.211 |

**k1-style** (`OACI − ERM` audit `bootstrap_ucl`): level 0 = **+0.0862**, level 1 = **+0.0833**.
**k2-style** (worst-domain = pooled for the single target domain): `tgt bAcc` / `tgt NLL` columns above.

> Descriptive only — single fold, single seed. Not efficacy evidence, not a confirmatory CI; no
> permutation test / decision rule. At full bootstrap OACI's audit-leakage UCL is slightly above ERM at
> both levels and its target bAcc is not above ERM here — inconclusive on one fold.

## Timing comparison

| run | V100-held | leakage location | total |
|-----|----------:|------------------|------:|
| **C3** monolithic full bootstrap (876133) | **6 h 26 m** | on the V100 (GPU idle ~88 %) | 6 h 26 m |
| **C4b** staged reduced bootstrap (876816 + 876980) | ~1 h 40 m | CPU (Phase B, no GPU) | ~4 h 23 m |
| **C5** staged full bootstrap (877374 + 877446) | **≈ 1 h 37 m** | **CPU (Phase B ≈ 5 h 44 m, no GPU)** | ≈ 7 h 21 m |

The staged full-bootstrap one-fold holds the **V100 for ≈ 1 h 37 m** instead of **6 h 26 m** — the ~5 h 44 m
leakage (≈ the monolithic's 5 h 38 m) runs on CPU. Phase A is bootstrap-independent, so the V100 hold is the
same as the reduced run. The staged total wall-clock is longer (Phase A over-extracts all feasible
candidates, and Phase B re-does no GPU work), but the **V100 occupancy — the scarce resource — drops ~4×**.

## Status & next

- ✅ The full-bootstrap staged path runs end-to-end to a deep-verified artifact (`target_fit ∅`), with the
  V100 freed during the full-bootstrap leakage. Bit-exactness vs monolithic is proven by the CPU
  staged-executor tests; the reduced↔full training identity is confirmed empirically.
- ⏭ **C6: BNCI2014_001 LOSO** — all 9 held-out targets, seed 0 only, staged executor, full bootstrap. Then
  k1/k2 aggregation + decision → BNCI2014_004 → multi-dataset sweep.
