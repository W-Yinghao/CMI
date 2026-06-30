# C3 — One-Fold Full-Bootstrap Timing & Inference-Path Validation

> **Full-budget one-fold timing and full-bootstrap inference-path validation.**
> **This is not final confirmatory efficacy evidence.**

The same held-out target (BNCI2014_001, subject-001, seed 0, deletion levels 0+1, four methods) run at the
**full training budget AND the confirmatory full bootstrap** (selection 200 / audit 2000 / paired 2000)
with the C2 **process-parallel leakage bootstrap** (16 workers). Purpose: measure real wall-clock and
confirm the full-bootstrap inference path produces a deep-verified artifact. One fold + one seed is
**descriptive only** — no permutation test, no decision rule, no efficacy claim.

---

## Job & code

| item | value |
|------|-------|
| SLURM job ID | `876133` |
| node / GPU | `node42` / `Tesla V100-PCIE-32GB` |
| exit code | `0` (`demo=0`, `verify=0`, `OVERALL: PASS`) |
| commit SHA | `db4d15d` |
| git tree hash | `16c48f6ffb3ed47bdbb32e02b9a67326d50ddb6f` |
| wall-clock (running, post-queue) | **6 h 26 m** (21:09:30 → 03:35:34) |
| `parallel_n_jobs` | `16` (`leakage_parallel_mode = process_bootstrap_replicate`) |
| worker thread settings | `worker_threads = 1` (each worker pins OMP/MKL/OpenBLAS/BLIS/NUMEXPR/VECLIB = 1 + `threadpool_limits(1)`) |

### Utilization (sampled on the node every 60 s, 386 samples)

| phase | duration | GPU util | CPU (my job) | worker procs |
|-------|---------:|---------:|-------------:|-------------:|
| **training** (1 python proc) | ≈ **48 min** | peak 75 %, mean **58 %** | low | 1 |
| **leakage scoring** (16 workers) | ≈ **5 h 38 m** | mean **8 %** (idle) | peak **1633 %**, mean **1196 %** (≈ 12 cores) | peak **18** (16 + demo + pool) |

The leakage phase dominates (≈ 88 % of wall-clock) and is CPU-bound with the GPU idle — but the 16 workers
are genuinely busy (≈ 12–16 cores), i.e. the parallel bootstrap is operational. GPU memory ≈ 15.8 GB held
throughout. (sampler: `c3_util_876133.tsv`.)

## Budget (full / confirmatory)

| item | value |
|------|-------|
| training budget | full — `stage1_epochs=200`, `stage2_epochs=200`, `stage2_steps_per_epoch=20` |
| `selection_bootstrap` | **200** |
| `audit_bootstrap` | **2000** |
| `paired_bootstrap` | **2000** |
| probe `capacities` × `folds` | `[0, 64]` × `5` |
| `bootstrap_mode` | `full_budget` (`not_confirmatory_ci = false`) |

## Artifact

| item | value |
|------|-------|
| manifest hash | `121414fb297fcb3a…` |
| `artifact_scientific_hash` | `2f0dd88669f9d578a6d2760e27c33ca71c53fbb0f8026a5e56a3ce0d02a1aad6` |
| `artifact_pure_science_hash` | `6e7c3817f9fda08bd1265ab1e78fdaa143a5af981ab1e42d4410667c4494389a` |
| `fold_result_hash` | `0be21a591d4b…` |
| `data_evidence_hash` / `split_manifest_hash` | `c5e26ec4dd32…` / `bnci_split:86114…` |
| deep verification | **OK** |
| indexed / total files | **548 / 550** |
| verified checkpoints / plans | **202 / 14** |
| `target_fit_ids` | **∅** |

## Science endpoints (descriptive — full bootstrap, 1 fold, seed 0)

Support: domains (rows = source subjects 4–9, then audit) × 4 classes `[left_hand, right_hand, feet, tongue]`,
`reference_prior = [0.25]×4`.

- **Level 0** (no deletion): all cells 144; `R_ERM_hat = 0.8909`, `tau = 0.9209`.
- **Level 1** (deleted cell `BNCI2014_001|subject-004 / feet`): cell `(row 0, class 2)` count **0**, mass
  **0.0** (≈ 144 rows removed); rest 144; `R_ERM_hat = 0.8326`, `tau = 0.8626`.

Per method — selected checkpoint, risk `R` / gap `R−τ` / epoch / λ; selection `extractable_LQ_ov` / `bootstrap_ucl`;
audit `extractable_LQ_ov` / `bootstrap_ucl`; source-audit & target bAcc/NLL/ECE:

**Level 0**

| method | ckpt | R | gap | ep | λ | sel LQ | sel UCL | aud LQ | aud UCL | srcA bAcc/NLL/ECE | tgt bAcc/NLL/ECE |
|--------|------|--:|----:|---:|--:|-------:|--------:|-------:|--------:|--------|--------|
| ERM | `3ca0327e8f` | 0.8909 | −0.0300 | −1 | — | 1.5017 | 1.6652 | 0.5793 | 0.8174 | 0.361/1.479/0.211 | 0.429/1.363/0.145 |
| OACI | `661596a55c` | 0.8972 | −0.0238 | 159 | 2.319 | 0.8191 | 1.2499 | 0.5558 | 0.8689 | 0.343/1.516/0.195 | 0.401/1.384/0.148 |
| global_lpc | `a8b1e7b9dd` | 0.9183 | −0.0027 | 129 | 1.168 | 1.0094 | 1.3419 | 0.5970 | 0.8401 | 0.365/1.439/0.126 | 0.401/1.286/0.074 |
| uniform | `a8b1e7b9dd` | 0.9183 | −0.0027 | 129 | 1.168 | 1.0094 | 1.3419 | 0.5970 | 0.8401 | 0.365/1.439/0.126 | 0.401/1.286/0.074 |

**Level 1**

| method | ckpt | R | gap | ep | λ | sel LQ | sel UCL | aud LQ | aud UCL | srcA bAcc/NLL/ECE | tgt bAcc/NLL/ECE |
|--------|------|--:|----:|---:|--:|-------:|--------:|-------:|--------:|--------|--------|
| ERM | `70d9dc54ca` | 0.8326 | −0.0300 | −1 | — | 1.4815 | 1.6120 | 0.6187 | 0.8061 | 0.392/1.394/0.172 | 0.477/1.206/0.118 |
| OACI | `e398f76f09` | 0.8471 | −0.0155 | 34 | 2.023 | 1.1870 | 1.4358 | 0.6114 | 0.8472 | 0.394/1.533/0.179 | 0.488/1.297/0.119 |
| global_lpc | `10e62f23d6` | 0.7485 | −0.1141 | 9 | 6.270 | 1.1015 | 1.3589 | 0.6061 | 0.7677 | 0.406/1.505/0.182 | 0.498/1.165/0.096 |
| uniform | `3a1282ce56` | 0.8560 | −0.0066 | 189 | 1.849 | 1.0340 | 1.3641 | 0.5498 | 0.8703 | 0.395/1.384/0.074 | 0.538/1.137/0.058 |

**k1-style** (`OACI − ERM` audit `bootstrap_ucl`): L0 = **+0.0515**, L1 = **+0.0412**.
**k2-style** (worst-domain, single seed): tgt bAcc/NLL per method = the `tgt` columns above (pooled == worst
for a single target domain).

> Descriptive only — single fold, single seed, **not** efficacy evidence, **not** a confirmatory CI.

---

## Comparison vs the reduced-bootstrap validation (C2 / job 875714)

| | reduced (875714) | full (876133) |
|--|--|--|
| selection / audit / paired bootstrap | 64 / 256 / 256 | 200 / 2000 / 2000 |
| wall-clock | 3 h 17 m | **6 h 26 m** |
| k1 (OACI−ERM audit UCL) L0 / L1 | −0.0263 / +0.1853 | **+0.0515 / +0.0412** |
| selected checkpoints | — | **changed for every method** (e.g. L0 OACI ep 184→159; L1 OACI 134→34, global_lpc 69→9, uniform 74→189) |

**Speedup vs the prior single-thread behaviour.** The leakage phase ran ≈ 5 h 38 m on ≈ 12–16 busy cores;
single-threaded that work is ≈ **68–90 h**, so the parallel bootstrap is a **~11–14×** speedup. More to the
point: the earlier single-thread runs (`875401`, `875531`) **hung for hours and never finished a seed** —
the full-bootstrap one-fold was effectively **intractable** before C2; it now completes in 6.4 h.

**Did selected checkpoints / endpoint signs change? Yes — but the comparison is confounded.** `R_ERM_hat`
and `tau` also differ between the two runs (L0 R_ERM 0.872 → 0.891). Those are bootstrap-INDEPENDENT
training quantities, so the **trained models themselves differ** between the reduced and full runs. The
cause: the bootstrap budget is part of the manifest, so the two runs have different manifest hashes
(`822f1e89…` vs `121414fb…`), and the manifest hash seeds the run (sampling/training RNG). Therefore the
reduced↔full differences (selected checkpoints, the k1 sign flip at L0) conflate **bootstrap precision**
with a **different training trajectory** — this is **not** a clean bootstrap-precision ablation.

> **Follow-up to consider** (not acted on here): if a clean reduced-vs-full ablation is ever wanted, the
> training seed would need to be decoupled from the bootstrap-inclusive manifest hash. For the actual
> confirmatory this does not matter — every fold uses the full-bootstrap manifest consistently — but it
> means the C2 reduced run is **not** the same trained model as this C3 full run.

## Status & next

- ✅ The full-bootstrap inference path runs end-to-end to a **deep-verified** artifact (`target_fit = ∅`)
  in **6 h 26 m**; the process-parallel bootstrap is operational at the real statistical budget.
- ⚠️ Leakage scoring is ≈ 88 % of wall-clock and CPU-bound (GPU idle). For a full LOSO sweep (9 folds ×
  ~6 h ≈ 2–3 days of GPU-held CPU work) this is the cost driver; a future option is candidate-level
  parallelism / more cores so the V100 is not held idle during leakage.
- ⏭ **Next milestone:** the **BNCI2014_001 LOSO driver** (all 9 held-out targets), still before
  BNCI2014_004 and before the k1/k2 permutation/decision machinery.
