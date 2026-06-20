# H²-CMI experiment infrastructure — shift grid

Infrastructure ONLY (no gate / SSL / disentangle / alignment / online-transform / real EEG).
Answers, on the EEG mechanism simulator, **when class-conditional TTA helps vs hurts**, and
**whether CMI changes that** — before any confirmatory run.

## The factorial (one source-model pair per `seed × target_site`)

```
Source-A (CMI off):  M0 = A + identity   M2 = A + offline diagonal TTA
Source-B (CMI on):   M1 = B + identity   M3 = B + offline diagonal TTA
```

A and B start from the **same init** and see the **same minibatch order** (deterministic
DataLoader generator + re-seed before the loop); `CMIConfig.enabled=False` skips critics /
dual / penalty entirely. M0/M2 literally share Source-A; M1/M3 share Source-B. Verified by
`test_factorial_checkpoint_sharing`.

## Paired, mechanism-orthogonal simulator

`data/paired_simulator.py`: each mechanism (labels / phase / mixing / noise / prior / target
knobs) draws from an independent `SeedSequence` stream, the shift is applied **only** to the
held-out `target_site`, and the **source data is element-wise identical across scenarios**
(so the two source models are trained once and reused for every scenario). Verified by
`test_shift_pairing`. First-batch scenarios: `no_shift, cov, prior, concept, cov_prior,
cov_concept` (no montage/noise/label-corruption yet).

## Oracle diagnostics (`tta/oracles.py`)

* `oracle_prior` — true target prior fixed, transform still unsupervised → isolates **prior
  estimation**.
* `oracle_labels` — true labels as one-hot responsibilities → isolates **EM
  responsibilities**.
* `oracle_supervised_transform` — cross-fitted supervised transform, held-out evidence
  ceiling → isolates the **transform family / density geometry**.

No `oracle_simulator_transform`: the known mixing is in sensor space (then z-scored and
passed through a non-linear encoder), so no exact latent diagonal inverts it.

Interpretation logic for a negative `ΔbAcc_TTA`:
* oracle_transform helps, oracle_labels helps, unsupervised fails → **responsibilities/prior**;
* oracle_labels still fails → **density geometry or diagonal family**;
* oracle_transform helps but evidence↑ while accuracy↓ → **density ⟂ decision boundary**;
* frequent adaptation under no_shift → **rollback/evidence threshold** unreliable;
* CMI lowers harm-rate without changing mean gain → **"CMI improves adaptation safety"**.

## Run

Tiny integration first (schema / pairing / resume / oracles):
```
conda run -n icml python -m h2cmi.run_shift_grid \
  --scenarios no_shift,cov,prior,concept --seeds 0 --target-sites all \
  --sites 3 --subjects 2 --sessions 2 --trials 12 --epochs 2 --fast \
  --out results/h2cmi/shift_grid_smoke.jsonl
```
Then the 3-seed screening (GPU):
```
conda run -n icml python -m h2cmi.run_shift_grid \
  --scenarios no_shift,cov,prior,concept,cov_prior,cov_concept --seeds 0,1,2 \
  --target-sites all --sites 5 --subjects 3 --sessions 2 --trials 40 \
  --epochs 20 --leak-perm 10 --device cuda --out results/h2cmi/shift_grid_screen.jsonl
```

Output is JSONL (atomic append, resumable: completed `(data_seed, target_site, scenario,
method, cmi)` units are skipped). Rows carry `commit_sha, config_hash, *_seed, target_site,
target_size, scenario, method, factorial_cell, strict_bacc, adapted_bacc, delta_bacc,
macro_f1/nll/brier/ece, crossfit_evidence_gain, nll_before/after, adapted, rollback_reason,
transform_norm/bias_norm/logdet/condition_number, estimated_prior/true_prior/prior_l1_error,
site/subject/session_leakage, final_lambdas, critic_ce, source_data_hash,
source_checkpoint_hash`.

3 seeds × 5 sites → 15 paired units/scenario. Analyse with a `seed × target_site` paired
bootstrap; the headline interaction is
`[ Δ_TTA(CMI on) − Δ_TTA(CMI off) ]`. The goal is **diagnosis, not a positive number**:
locate where negative gain comes from before any 10-seed confirmatory run.
```
```
