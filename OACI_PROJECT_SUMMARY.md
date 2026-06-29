# OACI Project — Complete Summary

> **OACI** = *Overlap-Aware Risk-Feasible Conditional Invariance under domain–class support mismatch.*
> A strict domain-generalization (DG) EEG method, developed as an **isolated package** `oaci/` inside the
> `CMI_AAAI` repo, on its own git worktree/branch `oaci`.
>
> This document is a standalone reference: read it and you understand the whole project — what it is, why
> it exists, how the code is organized, what has been built (milestone by milestone), how the
> scientific-integrity machinery works, how to run everything, and the current status.

---

## 1. What OACI is (the research idea)

OACI is the successor to a **failed** earlier direction (LPC-CMI, archived under `archive/lpc-cmi-failed/`).
The honest post-mortem of LPC found its three claimed pillars all collapsed (leakage reduction was via
representation collapse; the calibration "win" was a temperature side-effect; the accuracy was matched by a
plain CORAL baseline). OACI is designed from the start to **avoid LPC's three failure modes**.

**The problem.** In cross-subject / cross-site EEG, the source and target differ in BOTH the marginal over
domains AND in *which (domain, class) cells are even observed* — i.e. **support mismatch**. Naively enforcing
conditional invariance `I(Z;D|Y)=0` over cells that don't exist in the target is what let LPC "cheat".

**The OACI estimand / method (the four ingredients):**
1. **Support graph + observed-cell mutual information `I_ov`.** Invariance is measured/penalized **only on the
   (domain,class) cells that are actually observed** — never on empty cells. The `SupportGraph` encodes the
   eligibility of each cell and the reference prior `p(y)`.
2. **Risk-feasible primal–dual training.** Minimize the leakage penalty **subject to** a source-risk
   constraint (the classifier must still be good), via a dual variable λ — rather than blindly trading
   accuracy for invariance.
3. **Honest UCB on leakage.** The leakage estimate `I(Z;D|Y)` is reported with a cross-fit,
   permutation-calibrated **upper confidence bound**, so "low leakage" claims are statistically honest, not
   point-estimate optimism.
4. **Refuse-when-unidentifiable.** When there is no safe subspace / the cells don't support the estimand, the
   method abstains instead of producing a false certificate.

**Why this matters for the paper.** The contribution is framed as a *measurement→control* discipline: most of
the value is in being able to **honestly measure** conditional leakage on observed support and **prove** when
a method is or isn't actually controlling it — with a fully reproducible, hash-audited artifact trail.

### The four methods compared (every run trains all four)
| key | what it is | role |
|-----|------------|------|
| `ERM` | plain empirical-risk Stage-1 model | baseline / Stage-2 BN reference |
| `OACI` | the proposed risk-feasible, support-aware, λ-penalized method | the candidate |
| `global_lpc` | global LPC-style invariance (the failed-style comparator) | negative control |
| `uniform` | uniform/неinformative reweighting | floor control |

A core acceptance property: **method order must not change any scientific result** — running the four in any
permutation must produce bit-identical hashes (this is what the B2b GPU smoke proves on real data).

---

## 2. Repository layout & worktree discipline

- **Primary repo working dir:** `/home/infres/yinwang/CMI_AAAI` (branch `csc` — a *different* direction;
  must stay untouched, including its stash `csc-tests-wip-2026-06-21`).
- **OACI worktree:** `/home/infres/yinwang/CMI_AAAI_oaci` (branch `oaci`). **All OACI work happens here.**
- **Hard constraints (always in effect):**
  - Work strictly in the `oaci` worktree; **never import `cmi/` or `h2cmi/` at runtime** (there are tests
    asserting no such import sneaks in).
  - Do **not** modify or move `csc` or its stash.
  - **Offline only**: data loaders never download; the raw **datalake is read-only**.
  - **Use SLURM for compute** (CPU tests on the CPU partition; GPU smoke on V100) — no walltime `-t` per
    cluster policy.
  - Artifacts/logs for GPU runs live **outside** the repo
    (`/projects/EEG-foundation-model/yinghao/oaci-gpu-{logs,artifacts}`).

### Package structure (`oaci/`)
```
oaci/
  data/        eeg/bnci.py (MOABB BNCI2014-001 loader), eeg/offline.py (forbid_network, datalake root),
               sampler/mass — class-stratified sampling, sample-mass weighting
  models/      shallow.py (ShallowConvNet), mlp.py, bn_audit.py (BatchNorm buffer audit)
  train/       engine.py (the primal-dual training loop), risk.py (source risk),
               adversary.py, primal_dual.py, selector.py, evaluate.py (chunked guard),
               bn.py (all_eval ctx), checkpoint.py (state hashing), rng.py (forked_rng)
  leakage/     estimate.py, critic.py (DomainProbe = sklearn LogReg), crossfit.py, cache.py, ucb.py
  eval/        metrics: nll / ece / balanced-accuracy / paired bootstrap
  protocol/    manifest_v2.py (typed manifest + validation), *.yaml (smoke / confirmatory / fake manifests)
  runner/      the orchestration: fold.py, stage1.py, bnci*.py, fake_artifact.py, provenance.py,
               scientific_hash.py, config.py, bnci_gpu_*.py (the B2b GPU smoke stack)
  runtime/     cuda.py (determinism config + evidence), rng_state.py (RNG snapshot/assert)
  artifacts/   canonical_json.py, writer.py (atomic write + git evidence), verify.py (deep verifier)
  tests/       ~40 standalone+pytest test modules (each runnable as `python -m oaci.tests.<name>`)
  slurm_ci.sh           CPU CI (all test modules + demos in parallel, folds every rc)
  slurm_bnci_preflight.sh   dedicated real-data CPU preflight job
  slurm_gpu_smoke.sh    V100 two-order real-data GPU smoke
```

---

## 3. The scientific-integrity architecture (the heart of the project)

Everything is built so that a result is **reproducible and tamper-evident**. The key concepts:

- **Canonical JSON + hashing.** `artifacts/canonical_json.py` gives a deterministic byte encoding;
  `canonical_json_hash` is the workhorse. Note there are *two* canonical encoders with different rules — the
  artifact `canonical_json` and the runner `keys.canonical_json_hash` — and they must be used consistently
  (a past bug: the writer gate used the wrong one for `execution_config_hash`; fixed to `_keys_hash`).
- **Scientific hashes** (in `runner/scientific_hash.py` and friends): `scientific_value_hash`,
  `leakage_result_hash` (normalizes int-keyed reference-entropy dicts), `state_hash`/`model_state_hash`
  (checkpoints), `execution_config_hash`, plan/support/run-key hashes, prediction-content & metrics hashes.
- **Artifact tree + closed loop.** A run is persisted by `write_artifact_tree_atomic`, then immediately
  `verify_artifact_tree(deep=True)`, then `read_completed_artifact`, then
  `compare_artifact_summary_to_memory`. So every run is write → deep-verify → read-back → compare.
- **GitEvidence.** Collected live via `git` subprocess; the writer **rejects a dirty scientific tree** and an
  in-repo destination. Status is scoped to the `oaci/` pathspec. The deep verifier recomputes the
  `context_hash` from `context/*.json`. (This is why temporary files must live outside `oaci/`.)
- **Determinism.** `forked_rng(seed, device)` requires an *indexed* CUDA device and seeds only that one
  device (never `manual_seed_all`); `snapshot_rng_state` / `assert_rng_unchanged` cover
  python/numpy/torch-CPU/one-CUDA streams.
- **Save-everything.** Predictions are saved (`.preds`), so metrics can be recomputed without retraining.

### The two-level runner pipeline
`run_level_complete` = **train → select → audit → predict → COMPLETE**, assembled into a `FoldRunResult`:
- **Stage 1** (`run_stage1_once` → `train_stage1`): the ERM model + guard evaluation + `tau`/`R_ERM_hat`.
- **Stage 2** (per method, `train_risk_feasible`): adversary + primal–dual λ updates; **BatchNorm is frozen
  at the ERM running stats** (only affine/encoder params train) — this is audited (see §5).
- **Selection** (`selector`): pick the checkpoint by source risk under the constraint.
- **Audit**: leakage estimate with cross-fit + permutation-calibrated UCB on observed support.
- **Predict / metrics**: three roles — `source_guard`, `source_audit`, `target_audit` — with bAcc/NLL/ECE.

`run_bnci_two_level_in_memory(bnci_fold, model_seed, method_order, device)` runs all of the above for any
4-method permutation; `run_bnci_artifact_once` wraps it in the write→verify→read→compare closed loop.

---

## 4. Data & model

- **Dataset:** BNCI2014-001 (a.k.a. BCI Competition IV-2a) motor imagery, loaded **offline** via MOABB from
  the datalake (`load_moabb_confirmatory`, `forbid_network()`, `moabb_offline_root`). 4 classes
  (left/right hand, feet, tongue), 22 EEG channels.
- **Fold contract (the six-subject smoke fold):** `X` shape **(3456, 22, 385)**; role index counts
  **source_train=1728 / source_audit=1152 / target_audit=576**; total class-count table 864 = 6×144 per the
  preflight. `BNCIRealFold` / `build_bnci_real_fold` adapt MOABB → `FoldData`/`FoldScope`.
- **Model:** **ShallowConvNet** (`models/shallow.py`): input `[B,22,385]` → logits `[B,4]`, features
  `z [B,800]` (dropout-free log-power). `validate_shallow_geometry(22,385)` →
  `{post_temporal:361, pooled:20, feat_dim:800}`. A dummy forward in `__init__` infers the feature dim while
  preserving BN.
- **Leakage critic:** `DomainProbe` = sklearn `LogisticRegression` with **frozen** solver=`lbfgs`,
  tol=`1e-4`, fit_intercept=`True` (bound into `critic_config_hash`). Runs on detached **CPU/numpy**
  features (so it is *not* a GPU-device concern).
- **Preprocessing:** MOABB 250 Hz; raw-signal path for leakage; zscore normalization (EEG is volts-scale
  ~1e-6, so post-norm std tolerances are deliberately relaxed).

---

## 5. Milestones (in build order)

The project advanced through many tightly-specified commits. The recent, load-bearing ones:

### A2b — in-memory four-method runner + artifact persistence + fake closed loop
The full runner over a synthetic "fake" fold, with the complete write→deep-verify→read→compare loop and the
scientific-hash stack. `run_fake_two_level` is the order-invariant CPU fixture used everywhere downstream.

### B1a — real-data blockers + offline loader + typed manifest
Fixed five real-data blockers; added `load_moabb_confirmatory` (offline-first, `resolved_preprocess_hash`),
the typed `manifest_v2` blocks (MI/synthetic preprocessing, deleted-cell, probe block) with strict range
validation, and `manifest_payload_hash`.

### B1b — BNCI → FoldData adapter + six-subject CPU preflight
`BNCIRealFold`, `build_bnci_fold_from_bundle`, `target_seen_by_fit`; a dedicated SLURM preflight
(`slurm_bnci_preflight.sh`) that loads the real data once and checks exact role counts/support tables, audit
estimability, and **0 network attempts**. (Fix: MNE logs "Adding metadata" to *stdout*, polluting canonical
JSON — wrapped the CLI in `redirect_stdout(sys.stderr)`.)

### B2a — GPU blockers + frozen probe + minimized smoke budget
`runtime/cuda.py` (`configure_cuda_determinism` → `CudaRuntimeEvidence`), `runtime/rng_state.py`; froze the
probe solver/tol/fit_intercept into the config + manifest + hash; reduced the smoke budget
(epochs/steps/bootstraps) so the GPU smoke is fast. CI: **569 tests**.

### B2b-i — GPU runtime acceptance + BN audit + cross-run comparator  *(commit `bf27ff9`)*
The acceptance machinery (all CPU-tested; full GPU run is B2b-ii). **Four runtime residuals** hardened:
1. `configure_cuda_determinism` sets inter-op threads **strictly** (no swallowed failure) and verifies
   `get_num_threads()==1` and `get_num_interop_threads()==1`.
2. **Native thread-pool audit**: `NativeThreadPoolRecord` + `collect_native_threadpools` (via
   `threadpoolctl`) require **every** BLAS/OpenMP pool `num_threads==1`; the OMP/MKL/OPENBLAS/BLIS/NUMEXPR/
   VECLIB/MKL_DYNAMIC env is recorded (`thread_env_record`); a `driver_version=="unknown"` **fails** the
   formal smoke; `runtime_evidence_hash` is now the **canonical-JSON hash** (not a hand-written repr).
3. `assert_cuda_runtime_unchanged(initial)` re-checks the 9 live determinism flags after each run.
4. **Scientific-vs-transport separation**: the cross-run comparator covers only the *logical* scientific
   hashes and **excludes** `.pt`/index file SHAs, directory paths and the runtime report.

New modules:
- `models/bn_audit.py`: `bn_buffer_state`/`bn_buffer_hash` select & hash only the BN buffers
  (`running_mean`/`running_var`/`num_batches_tracked`); `audit_level_bn_buffers` checks every Stage-2
  trajectory + selected checkpoint against the shared ERM BN buffers.
- `runner/bnci_gpu_compare.py`: `flatten_scientific` → a flat `{canonical-path: value-hash}` map of a run
  (fold / per-level run-key, support, plans, ERM / per-method activity, full trajectory, selection, audit,
  the 3 prediction roles & metrics, signatures, target fits); `compare_scientific_results` →
  `BNCIOrderComparison` with per-group equality and the **exact first-mismatch path**
  (e.g. `levels/1/methods/OACI/selection/model_hash`); `comparison_all_equal`.
- `runner/bnci_gpu_smoke.py`: `run_bnci_gpu_smoke` = configure determinism **before** CUDA → load real fold
  **once** → run canonical order → assert RNG/runtime unchanged → `gc`+`empty_cache` → run reversed order →
  assert again → bit-exact compare → BN audit → report hash.

Tests: +3 cuda-runtime, +8 gpu-compare (CPU via two order-permuted fake runs). CI: **580 tests**.

### B2b-ii — GPU smoke CLI + V100 SLURM + GPU-targeted tests  *(commit `931def4`)*
- `runner/bnci_gpu_demo.py`: CLI that runs the smoke and prints **one canonical-JSON report** to stdout
  (runtime evidence, RNG before/after, both artifact hashes + verify counts, the per-group order comparison
  with first-mismatch path, BN-all-equal, per-level/per-method scientific summary). It carries the explicit
  notice that **method-to-method numerical differences are NOT efficacy evidence** (this is an *execution
  validation*, not a results run). MNE/MOABB/training chatter is redirected to stderr.
- `oaci/slurm_gpu_smoke.sh`: V100 (`--gres=gpu:1`), exports the determinism env **before any Python**
  (so `CUBLAS_WORKSPACE_CONFIG=:4096:8` exists before the CUDA context), never overrides the SLURM GPU
  allocation (Python asserts exactly one visible GPU), keeps artifacts/logs **outside** the repo, requires a
  clean `oaci/` tree + the datalake, runs the GPU tests + demo + **independent deep-verify of both
  artifacts**, then checks the order comparison is all-equal — folding every rc under `set -euo pipefail`.
- GPU-gated tests `test_bnci_gpu_runtime` / `test_bnci_gpu_runner` (skip cleanly on CPU; run on the V100) +
  CPU `test_bnci_gpu_script`. CI: **587 tests** (ALL-PASS, demo/verify rc=0 on a clean tree).

### B2b device fix — real CUDA bugs the CPU CI cannot see  *(commit `4e5b796`)*
The **first** V100 run (job 864437) crashed with
`RuntimeError: Expected all tensors to be on the same device, cuda:0 and cpu`. Root cause: **float64
sufficient-statistic accumulators were created on the default CPU device, then combined with cuda
model-output tensors**. Fixed three sites — all create the accumulator on the compute device:
- `train/evaluate.py` `evaluate_guard`: `ce_num/ce_den/pc_num/pc_den/err_num`.
- `train/risk.py` `per_class_ce_sums`: `sums/mass` (`device=per.device`).
- `models/shallow.py` `_infer_feat_dim`: the dummy-forward probe tensor (`device=next(params).device`).

Because `device == cpu` for CPU runs, **every CPU result and scientific hash is byte-identical** (the
affected suites pass unchanged); within a GPU run both method orders share the device, so the bit-exact
canonical-vs-reversed comparison still holds. An Explore audit of the rest of the GPU path
(engine/adversary/objective/features/predict/leakage/eval/checkpoint) found **no further device mixing** —
feature/prediction paths already `.to(device)` then `.cpu()` before numpy; leakage/eval are pure
numpy/sklearn.

> **Key lesson:** a CPU-only test suite *cannot* catch cuda-vs-cpu tensor mixing. The GPU smoke is the only
> check for this class of bug — so it is a required gate, not a nicety.

---

## 6. How to run things

All from the worktree root `/home/infres/yinwang/CMI_AAAI_oaci`, env `icml`
(`/home/infres/yinwang/anaconda3/envs/icml/bin/python`).

- **Any single test module (CPU, no SLURM needed for a quick check):**
  `OMP_NUM_THREADS=4 python -m oaci.tests.<module>`  (e.g. `test_bnci_gpu_compare`).
- **Full CPU CI (SLURM, CPU partition):** `sbatch oaci/slurm_ci.sh` → `logs/oaci-ci-<jobid>.out`; look for
  `=== OVERALL: ALL-PASS (exit 0) ===` and the per-`PASS N` counts. **Commit first** — the fake demo step
  needs a clean `oaci/` tree.
- **Real-data preflight (SLURM, CPU):** `sbatch oaci/slurm_bnci_preflight.sh`.
- **V100 two-order GPU smoke (SLURM):**
  ```
  mkdir -p /projects/EEG-foundation-model/yinghao/oaci-gpu-logs
  OACI_DATALAKE_ROOT=/projects/EEG-foundation-model/datalake/raw \
    sbatch --output=/projects/EEG-foundation-model/yinghao/oaci-gpu-logs/%x-%j.out oaci/slurm_gpu_smoke.sh
  ```
  Success = `=== OVERALL: PASS (exit 0) ===` and a `gpu-smoke-report.json` with `order_comparison.all_equal:
  true`, `bn_all_equal_to_erm: true`, `rng_unchanged: true`, both artifacts `deep_verification_ok: true`.

> **Cluster caveat:** the SLURM account is shared and frequently saturated by large unrelated sweeps
> (`p0w2pri_*` A40/V100, `g1/g2` CPU). Submissions can hit `QOSMaxSubmitJobPerUserLimit` (global submit cap)
> or `QOSMaxGRESPerUser` (GPU pending) — wait for the queue to drain; never cancel the unrelated jobs.

---

## 7. Current status (as of 2026-06-29) — B2b COMPLETE ✅

- **Pushed commits on `oaci`** (GitHub remote `W-Yinghao/CMI`): `bf27ff9` (B2b-i) → `931def4` (B2b-ii) →
  `4e5b796` (device fix) → `f36e9c2` (robust nondeterminism probe).
- **CPU CI:** **587 tests ALL-PASS**, demo/verify rc=0 (clean tree, commit `4e5b796`).
- **GPU smoke: GREEN.** Job **866832** (commit `f36e9c2`, V100 node42): `OVERALL: PASS (exit 0)` —
  rc runtime=0 / runner=0 / demo=0 / validator=0.
  - **Order-invariance proven bit-exactly on REAL BNCI:** canonical AND reversed method orders →
    identical `artifact_scientific_hash = 1a7c92f8438a`; `comparison all_equal=True`.
  - **BN audit:** every Stage-2 checkpoint BN buffer == ERM (`bn_equal=True`).
  - **RNG + runtime flags unchanged** across both runs; **both artifacts independently deep-verified**
    (158 indexed / 160 files, 7 checkpoints, 14 plans each).
  - History: the first run (864437) crashed on the device bug → fixed (§5). The second run (866829) passed
    runner+demo+validator bit-exactly but failed one over-specific runtime self-check (`torch.bincount` has
    a deterministic CUDA path on this build) → probe made robust → 866832 fully green.
- `csc` and its stash remain untouched; the `oaci/` tree is clean.

> **Provenance-hash caveat — RESOLVED** (`f95f288`, CPU CI **595** ALL-PASS, no GPU). The
> `artifact_scientific_hash` binds the git commit/tree (so it differs across commits even when only a test
> file changed: `03460f…`@4e5b796 vs `1a7c92…`@f36e9c2) — correct for a "this exact code state produced this
> artifact" identity. Alongside it there is now a commit-**independent** `artifact_pure_science_hash` that
> binds the science only (manifest + execution config + model spec + fold result), so identical science
> hashes the same across commits. Both are written to the COMMITTED marker, recomputed by the verifier,
> carried in `ArtifactSummary`, and reported by both demos (with the git commit/tree/provenance). The B2b
> order-invariance contract (same commit, two orders → identical hash) was already bit-exact and is
> unchanged. See §3.

### Confirmatory layer (milestone C — in progress)
`confirmatory_v2.yaml` is a **protocol** schema, not a runnable manifest. The `oaci/confirmatory/` package
bridges them:
- `schema.py` — parse/validate the confirmatory protocol.
- `materialize.py` — turn one (dataset, held-out target) into a runnable **full-budget `pilot` manifest_v2**
  (a new additive `pilot` block carries an explicit, un-pinned subject split: target / source_audit /
  source_train); deterministic, no hand editing.
- `onefold.py` — load the real fold once, run the requested model seeds through the full
  write→verify→read→compare loop.
- `report.py` — per-seed/level/method endpoints + descriptive k1/k2 view.
- `demo.py` + `slurm_confirmatory_onefold.sh` — the V100 CLI/job.

**First step (current):** BNCI2014_001, target=subject-001 (all 9 subjects: audit=2,3, train=4–9), levels
0+1, full budget, **seeds 0,1,2** — labeled *pipeline validation, NOT confirmatory efficacy*. No LOSO, no
BNCI2014_004 yet. CPU adapter/report tests pass; the GPU run is `slurm_confirmatory_onefold.sh`.

### Open / next
- Confirm the one-fold V100 run is green (all seeds deep-verify, target_fit empty), then decide on the
  **full LOSO sweep + BNCI2014_004 + the real k1/k2 permutation/decision machinery** = the actual
  confirmatory results. The one-fold is **machinery validation**, not efficacy.

---

## 8. Pointers
- Authoritative live notes: the memory file `oaci-direction.md` (project memory index `MEMORY.md`).
- Failed predecessor & its post-mortem: `archive/lpc-cmi-failed/`, `notes/EVIDENCE_LEDGER.md`.
- This summary lives **outside** the `oaci/` scientific pathspec on purpose (so it never dirties the audited
  tree); move/commit it into `oaci/notes/` only when no GPU/CI job is mid-flight.
