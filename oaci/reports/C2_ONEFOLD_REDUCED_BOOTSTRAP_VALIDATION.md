# C2 — One-Fold Reduced-Bootstrap Pipeline Validation

> **Full-budget-training one-fold pipeline validation with reduced-bootstrap uncertainty.**
> **This is not final confirmatory statistical evidence.**

A single held-out target (subject-001) of BNCI2014_001 was run end-to-end at the **full training budget**
with a **reduced leakage/eval bootstrap** and the new **process-parallel leakage bootstrap** (C2). Purpose:
prove the confirmatory pipeline runs to a verified artifact in tractable wall-clock and exercise the
parallel leakage path on real data. A single fold + single seed + reduced bootstrap is **descriptive
only** — no permutation test, no decision rule, no efficacy claim.

---

## Code

| item | value |
|------|-------|
| C2 commit SHA | `13d676c` |
| git tree hash | `16c48f6ffb3ed47bdbb32e02b9a67326d50ddb6f` |
| manifest hash (materialized pilot) | `822f1e89acd7427ee9b8bab0c14d05340519a8f583cade2f4c64d4558d467775` |
| provenance-bound `artifact_scientific_hash` | `d3f4c7b34a50f82dac6381fa0c9bcb589f91b8541b964135e66768178e01f715` |
| commit-independent `artifact_pure_science_hash` | `24a7b723dfb818638bb04c48467124320ba237cbcb0c8222e6bccd2332e181bb` |
| `fold_result_hash` | `943d453e9a83e91d…` |

The materialized manifest, artifacts and logs live outside the repo
(`/projects/EEG-foundation-model/yinghao/oaci-confirmatory-onefold/875714/`); the scientific tree was clean
at `13d676c`.

## Job

| item | value |
|------|-------|
| SLURM job ID | `875714` |
| node / GPU | `node42` / `Tesla V100-PCIE-16GB` |
| exit code | `0` (`demo=0`, `verify=0`, `OVERALL: PASS`) |
| wall-clock (running, post-queue) | ≈ **3 h 17 m** (16:17:01 → 19:33:41) |
| bootstrap `parallel_n_jobs` | `16` (`leakage_parallel_mode = process_bootstrap_replicate`) |
| worker thread settings | `worker_threads = 1` (each worker pins OMP/MKL/OpenBLAS/BLIS/NUMEXPR/VECLIB = 1 + `threadpool_limits(1)`) |
| seq↔parallel equivalence | `verified_by_ci` (13 `test_leakage_parallel` tests, incl. end-to-end fake two-level) |

> **Utilization not sampled live for this run** (it ran during an unattended wait, and the cluster's
> `sacct` accounting returned no usage rows). The earlier diagnostic on the *sequential* run (`875531`)
> showed GPU 0 % / one core at 98 % during leakage scoring; C2 distributes that across the 16 workers.
> Peak/mean GPU + CPU utilization **will be captured** on the full-bootstrap one-fold timing run.

## Budget

| item | value |
|------|-------|
| full training budget | **true** — `stage1_epochs=200`, `stage2_epochs=200`, `stage2_steps_per_epoch=20` |
| `selection_bootstrap` | `64` (reduced) |
| `audit_bootstrap` | `256` (reduced) |
| `paired_bootstrap` | `256` (reduced) |
| `bootstrap_mode` | `pipeline_validation_reduced` |
| `not_confirmatory_evidence` | **true** |

`confirmatory_v2.yaml` was **not** edited; the reduction is a materialization-time `bootstrap_override`
recorded in the manifest hash (`protocol_id … -pilot-validredbootstrap-…`).

## Selection workload

| item | value |
|------|-------|
| unique checkpoints stored/verified | **202** (`.pt`), all deep-verified |
| trajectory length per Stage-2 method | 40 (`stage2_epochs 200 ÷ checkpoint_every 5`) — upper bound on candidates scored/method-level |
| bootstrap replicates (selection / audit) | 64 / 256 |
| cross-fit folds × capacities | 5 × 2 (`capacities = [0, 16]`) |
| LogReg fits per scoring (selection) | `(1 + 64) × 5 × 2 = 650` |
| LogReg fits per scoring (audit) | `(1 + 256) × 5 × 2 = 2570` |
| **estimated** total sklearn LogReg fits | **~10⁵** (≈ feasible-candidates × 650 over the 3 Stage-2 methods × 2 levels, + audit) — exact per-method `n_feasible` is not instrumented in this artifact |
| seq-vs-parallel equivalence tests passed | 13 / 13 (CI `875713`) |

This is exactly the cost that was single-threaded before C2 (the V100 sat idle); C2 fans it across the 16
allocated CPUs **bit-identically**.

## Artifact

| item | value |
|------|-------|
| `artifact_scientific_hash` | `d3f4c7b34a50f82d…` |
| `artifact_pure_science_hash` | `24a7b723dfb81863…` |
| `fold_result_hash` | `943d453e9a83e91d…` |
| deep verification | **OK** |
| indexed / total files | **548 / 550** |
| verified checkpoints / plans | **202 / 14** |
| `target_fit_ids` | **∅** (no target id entered any fit) |

## Science endpoints (descriptive — reduced bootstrap, 1 fold, seed 0)

Support tables are domains (rows = the 6 source subjects 4–9, then audit) × 4 classes
(`left_hand, right_hand, feet, tongue`); `reference_prior = [0.25, 0.25, 0.25, 0.25]`.

- **Level 0** (no deletion): eligibility counts all `144`; `R_ERM_hat = 0.8716`, `tau = 0.9016`.
- **Level 1** (deleted cell `BNCI2014_001|subject-004 / feet`): one cell `0` (count 0, mass 0, rows 0 —
  the deletion took effect), the rest `144`; `R_ERM_hat = 0.9545`, `tau = 0.9845`.

Per-method (selected risk · gap `R−τ` · epoch · λ · selection-UCL · audit-leakage-UCL · source-audit
bAcc · target pooled bAcc / NLL / ECE):

**Level 0**

| method | R_src | gap | epoch | λ | sel UCL | audit UCL | srcA bAcc | tgt bAcc | tgt NLL | tgt ECE |
|--------|------:|----:|------:|--:|--------:|----------:|----------:|---------:|--------:|--------:|
| ERM | 0.8716 | −0.0300 | −1 | — | 1.6880 | 0.7623 | 0.3837 | 0.4844 | 1.1625 | 0.1096 |
| OACI | 0.8916 | −0.0100 | 184 | 2.0142 | 1.4057 | 0.7360 | 0.3438 | 0.4670 | 1.2469 | 0.0875 |
| global_lpc | 0.8351 | −0.0665 | 129 | 0.0000 | 1.2083 | 0.8695 | 0.4036 | 0.4635 | 1.3013 | 0.1160 |
| uniform | 0.8351 | −0.0665 | 129 | 0.0000 | 1.2083 | 0.8695 | 0.4036 | 0.4635 | 1.3013 | 0.1160 |

**Level 1**

| method | R_src | gap | epoch | λ | sel UCL | audit UCL | srcA bAcc | tgt bAcc | tgt NLL | tgt ECE |
|--------|------:|----:|------:|--:|--------:|----------:|----------:|---------:|--------:|--------:|
| ERM | 0.9545 | −0.0300 | −1 | — | 1.6180 | 0.7553 | 0.3733 | 0.4514 | 1.4019 | 0.1769 |
| OACI | 0.9816 | −0.0029 | 134 | 1.2601 | 1.1553 | 0.9406 | 0.3507 | 0.4149 | 1.2679 | 0.1108 |
| global_lpc | 0.9563 | −0.0282 | 69 | 0.7008 | 1.3885 | 0.8198 | 0.3759 | 0.4427 | 1.2114 | 0.0808 |
| uniform | 0.9730 | −0.0115 | 74 | 0.8059 | 1.3690 | 0.8921 | 0.3655 | 0.4236 | 1.2247 | 0.0467 |

**k1-style** (`OACI − ERM` audit-leakage UCL, descriptive): level 0 = **−0.0263**, level 1 = **+0.1853**.
**k2-style** (worst-domain endpoints, per method, single seed): see the per-method tables above
(`tgt bAcc` / `tgt NLL` are pooled; worst-domain values are in the artifact's metrics).

> These numbers are a single fold, a single model seed and a **reduced** bootstrap. They are descriptive
> pipeline output, **not** efficacy evidence and **not** a confirmatory CI. No permutation test or decision
> rule was run.

---

## Status & next

- ✅ The confirmatory pipeline runs end-to-end to a **deep-verified** artifact (`target_fit = ∅`) at the
  full training budget, with the **process-parallel leakage bootstrap** operational on real data.
- ⏭ **Next (single step):** one **full-bootstrap one-fold timing run** — same target-001, split, seed 0,
  deletion levels and four methods, the confirmatory bootstrap budget **restored**, the parallel path kept
  — to measure real wall-clock, capture GPU/CPU utilization, and verify the full-bootstrap inference path.
  **Not** the LOSO sweep, **not** BNCI2014_004.
- Milestone order after that: full-bootstrap one-fold → LOSO (BNCI2014_001 only) → k1/k2 aggregation +
  decision machinery → BNCI2014_004 → multi-dataset confirmatory sweep.
