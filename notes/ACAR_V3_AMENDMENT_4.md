# ACAR v3 — Amendment 4 (training/artifact/conformal/data-contract completion; DESIGN-ONLY, NON-BINDING)

**Date:** 2026-06-22 · **Status:** `NON-BINDING / NO DEV NUMERICAL RUN / NO LOCKBOX ENDPOINT ACCESSED / NOT TAGGED`
Closes the 8 tag-blockers from the `ca8bed2`/`03fbf8b` review. Skeleton S5/S13 updated in place. v2 endpoint
`1528a94` / tag `acar-v2-protocol` @ `9b2f0c1` unchanged. All synthetic-only.

## Resolutions (1:1 with the review)

1. **C2 missing-action σ_min now fails.** `training._sigma_min` raises if any action lacks a FIT σ̂ prediction;
   each floor must be finite and `>0`. `FittedCandidateArtifact` validates C2 `sigma_min` == NON_IDENTITY with all
   floors finite `>0` (C1/C3 must be empty). Per-fold floor = `Q₀.₀₅` over the **full fold FIT = TRAIN∪VAL**; the
   **final** artifact uses the **OOF** floor (passed to `refit_candidate_fixed_epochs`).
2. **Real immutability + collision-free hash + arch cross-check.** `FittedCandidateArtifact` no longer holds a live
   `nn.Module`; it stores **canonical parameter bytes** `state_items=(name,dtype,shape,float64-LE bytes)` and rebuilds
   the net via `build_net()`. `verify_integrity()` recomputes the hash (floors hashed as **raw float64 bytes — no
   rounding**, so a 1e-13 change differs) and re-checks `net.candidate == artifact.candidate` and
   `heads == NON_IDENTITY`; tampering a field after construction (stale stored hash) → integrity failure. Disease is
   bound via `assert_disease`.
3. **Protocol/code conflicts resolved.** No action embedding (per-action `ModuleDict` heads); **target SD floor =
   1e-3**, **input-feature SD floor = 1e-6**; **normalizers fit on TRAIN only** (VAL uses TRAIN stats); per-fold σ_min
   over full FIT after best-state restoration; final σ_min from OOF. (Skeleton S5 corrected.)
4. **TRAIN/VAL subject isolation enforced.** `TrainExample` is a validated dataclass
   `(SubjectKey, deployment_batch_digest, action, window_action_set, delta_r)`; `fit_candidate_earlystop` raises on
   TRAIN∩VAL SubjectKey overlap, on any eligible batch whose actions ≠ NON_IDENTITY, on duplicate
   `(batch_digest, action)`, and on non-finite targets/loss/grad/params.
5. **Final all-DEV fixed-epoch refit implemented.** Split into `fit_candidate_earlystop` (per fold) and
   `refit_candidate_fixed_epochs(all_dev, n_epochs, sigma_min_oof, …)`. `final_epochs(best_epochs) =
   round_half_up(median_k(best_epoch_k + 1))` (best_epoch is **0-based ⇒ +1**). Deterministic failure (raise) on
   non-finite loss/grad/param or no improving epoch — never `best_state=None → load_state_dict`.
6. **Conformal calibration-unit shape locked.** `conformal_q` requires a **1-D** per-subject vector (a nested 2-D
   object raises, not miscounted as m subjects); finite scores; empty CAL → `(+∞, k=1)`; `route` requires the full
   non-identity set, returns canonical-order `U`, accepts `q` finite or `+∞` (rejects NaN / `-∞`), `δ` finite `≥0`;
   harmful rates must lie in `[0,1]`.
7. **Data contract aligned to the batching protocol.** `DeploymentBatch` gains `disease`; enforces
   `fallback ⇔ n_windows < MIN_BATCH`, `1 ≤ n_windows ≤ B`, **64-hex `source_state_ref`**, validated keys (non-empty
   ids, `window_index` int `≥0`); digest now covers schema/disease/shape/dtype/source/fallback + per-window bytes.
   `LabeledRiskRecord` requires **canonical action order**, full-hex digest, finite ΔR. `build_deployment_batches`
   checks **window-index uniqueness across the whole recording** before chunking (no chunk-boundary escape).
8. **Disease cross-use rejection is real (artifact side).** `FittedCandidateArtifact.assert_disease(disease)` raises
   on mismatch; the real loader/wiring (next step) will bind `DeploymentBatch.disease == artifact.disease` and add the
   end-to-end "PD artifact on SCZ batch → raise" guard.

## Guards (synthetic; toy only)
All v3 suites green: set-feature, **data-layer** (subject disambiguation, no-label/immutable, fallback/n/hex/key
validation, dup-index, digest), **training** (loss exactness, β-NLL stop-grad, subject-balanced, TRAIN/VAL overlap,
deterministic earlystop hash, σ_min completeness + 1e-13 hash + tamper→integrity-fail + arch mismatch, non-finite
loss fail-closed, fixed-epoch refit + final_epochs + OOF floor, normalizers), **predictor/conformal** (CandidatePred
validation, perm+action-order invariance, mask-consumption, C2 scale/q⁺ + C1/C3 no-clamp + C3 no-cross, rank/+∞/empty/
alpha/**nested-shape**, subject-joint fail-closed, route fail-closed incl. −∞/neg-δ/mixed-disease, CAL/EVAL isolation,
serialize + **assert_disease**, tie-aware harmful + rate-range). v2 guards still pass.

## Remaining before `acar-v3-dev-design-v1` (no DEV/lockbox)
Real v3 loader (DeploymentBatch from `z_te/subject_id_te/recording_id_te/window_index_te`; separate label loader for
`y_te` → `LabeledRiskRecord`; deployment path never returns/caches `y_te`; full SHA-256 on dumps/source/subject lists)
→ `develop.py` (S5 split-as-one-algorithm orchestration: fit_candidate_earlystop per fold; S2/S4 gates; C0/v2 replay;
final refit) + structural guards (incl. PD-artifact-on-SCZ-batch → raise; deployment-loader-never-touches-labels) →
env lock → full green re-run → tag → first DEV gate (`DEV_STOP / NO_LOCKBOX_CONSUMED` if no candidate passes).
