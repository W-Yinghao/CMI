# WAVE 1 / W1.geometry — geometry-only latent-diagonal falsification (FROZEN PRE-REGISTRATION)

**Phase:** Wave 1 (single module, per owner: geometry-only falsification only; Lee cross-session / P300
deferred). Branch `exp/h2cmi-wave0-mechanism` (or a new `exp/h2cmi-wave1-geometry`). **Freeze timestamp :=
git commit timestamp of this file.** Experiments-only; no manuscript prose.

## Question

Do the diagonal-positive-affine **latent** operators (FRSC / fixed_iterative / joint / Latent-IM-Diag /
pooled) actually correct a **geometry-only** perturbation that is linear in **sensor** space — as well as a
**full-covariance** operator does? A real re-reference / channel-gain / channel-dropout is linear in sensor
space but need not become a diagonal-positive-affine map in the encoder's latent space.

## Falsification hypothesis (pre-committed, directional)

If a full-covariance operator (latent CORAL and/or sensor-space EA) recovers balanced accuracy on a
geometry-only perturbation **significantly better** than the best diagonal-latent operator — while the
NULL (no perturbation) contrast is ≈ 0 — then the **latent-diagonal family is BOUNDED**: the sensor-linear
geometry shift does not reduce to a diagonal-latent map. This is reported as *bounding where the method
applies*, not as a failure. If the diagonal-latent operators recover as well as full-covariance, the
latent-diagonal family is **adequate** for these perturbations (no falsification).

## Data

Reuse the frozen V2P B-pair bundles (`v2_bundles` / `p0_v2pw_bundles`, code_sig `763bf49d`, read-only),
MI, multi-channel raw EEG `X:[n,ch,t]`. Source seeds {0,1,2}. Per target unit: fixed adaptation reservoir
`Xa` + fixed evaluation set `Xe` (same split as V2P). Class prevalence held constant (perturbation is
geometry-only; no label/prevalence shift).

## Perturbations (frozen, channel-space linear `X'_i = P X_i`; applied to BOTH `Xa` and `Xe`)

- **none** — `P = I` (NULL control).
- **reref** — re-reference: subtract channel 0 from all, `X'_c = X_c − X_0` (`P = I − 1·e_0^T`).
- **gain** — per-channel gain `g_c ~ LogNormal(0, 0.3)`, seed = `stable_hash("W1gain", dataset)`
  (fixed per dataset, same for all its subjects/seeds), `P = diag(g)`.
- **dropout** — zero 20% of channels (montage change), seed = `stable_hash("W1drop", dataset)`,
  `P = diag(mask)`.

Perturbation magnitudes/seeds are frozen here; the same `P` is applied to a dataset's every subject/seed.

## Operators (frozen)

- **identity** — no correction (baseline).
- **diagonal-latent** — `fixed_reference_oneshot` (FRSC), `fixed_iterative`, `joint`, `latent_im_diag`,
  `pooled` (from `eval_unit_p0`).
- **full-covariance** — **CORAL-latent** (NEW: whiten target latent covariance, recolor to source latent
  covariance — a full-matrix latent alignment) and **EA-sensor** (`source_recolored_ea`, sensor-space
  covariance recoloring, existing).

All decoded at **identity decision prior = uniform** (balanced-accuracy metric). Deterministic where the
eval path supports it; the aggregate is seed-averaged + cluster-bootstrapped regardless.

## Endpoints (frozen)

Per (unit, seed, perturbation, operator): balanced accuracy on the (perturbed) evaluation set. Seeds
averaged within unit; cluster bootstrap by **(dataset, subject)**, 10k.

**Primary falsification contrast (per geometry perturbation):**
`Δ_fullcov−diag = max(BA_CORAL, BA_EA) − max(BA_FRSC, BA_fixed_iter, BA_joint)`.
Paired per unit. Falsify diagonal-adequacy for that perturbation iff its CI excludes 0 (Δ > 0) **and** the
same contrast on **none** has a CI including 0.

**Secondary:** `BA_CORAL − BA_FRSC` (diagonal vs full-cov *within latent*); `BA_EA − BA_CORAL` (sensor vs
latent full-cov); per-operator BA drop `BA(none) − BA(perturbation)` (how much each perturbation hurts —
identity's drop confirms the perturbation is a real geometry stress the encoder is sensitive to).

## Pre-committed interpretation grid

- Δ_fullcov−diag > 0 (sig) on a geometry perturbation, null ≈ 0 → **latent-diagonal family bounded** for
  that shift; full-covariance correction is needed.
- Δ_fullcov−diag ≈ 0 on all geometry perturbations → **latent-diagonal adequate** for these shifts.
- identity BA drop ≈ 0 for a perturbation → the encoder is (near-)invariant to it → that perturbation is
  not a valid geometry stress; report and exclude it from the falsification claim.
- CORAL−FRSC > 0 but EA−CORAL ≈ 0 → the latent shift is full-cov-but-not-diagonal (latent correction
  suffices). EA−CORAL > 0 → sensor-space correction is needed beyond any latent operator.

## QC sentinels (hard gates, before trusting any number)

1. real (dataset, subject) fan-out coverage matches the V2P expected set; no bench index in any identity.
2. `P = I` (none) reproduces the un-perturbed BA per operator (perturbation harness is correct).
3. perturbation applied identically to `Xa` and `Xe` (same `P`); `P` recorded in the manifest.
4. per-operator BA drop under a non-trivial perturbation is > 0 for identity (else the perturbation is
   inert — flag).
5. CORAL-latent reduces to identity when target latent covariance == source latent covariance (unit test).
6. results addressed by real ids; checksums + manifest emitted.

## Discipline

Two-commit (result-only, then — only if explicitly requested — staged interpretation; default = neutral
results doc only, no inserts). Reuse frozen bundles (strict provenance; ProvenanceError → STOP unit).
GPU via SLURM only. **Probe first (1 unit): check QC-2 (none reproduces clean BA) + QC-4 (a perturbation
hurts identity) + no bench-index; only then launch the babysitter.** Never a refilling babysitter without
a passing probe. Terminal + Wave-0 artifacts are read-only.
