# Route A — GLS-VAE: a structured-latent attempt at fight-free dual-CMI + a concept-shift test

**Code:** `synthetic/gls_vae.py` (reuses the DGP + held-out CMI probes of
`synthetic/dual_cmi_v2.py` verbatim).
**Raw output:** `results/route_A_gls_vae.txt` (compare table),
`results/route_A_gls_vae_rest.txt` (fight sweep + concept test),
`results/route_A_gls_vae_full.txt` (concatenated).
**Run:** `/home/infres/yinwang/anaconda3/envs/icml/bin/python synthetic/gls_vae.py`
(CPU, ~3 h for the full 4-seed matrix; the compare table is the slow part).

---

## TL;DR (honest)

Route A **partly works and partly does not**, and the failure is the interesting part.

- **WORKS — the tension is dissolved (no fight).** Under label shift, pushing the
  naive encoder penalty up *raises* the decoder leakage `I(Y;D|Z)` (the A2 tension,
  reproduced on a learned model: `0.149 → 0.175` as `lam_enc: 0→4`). The GLS-VAE's
  single encoder penalty does **not**: `I(Y;D|Z)` stays flat/low
  (`0.136 → 0.125 → 0.127` as `lam_inv: 0→2→6`) while `I(Z;D|Y)` comes down. The GLS
  reference-prior decode neutralizes the label term *by construction*, so the
  remaining encoder pressure is fight-free. **This is the core Route-A claim and it
  holds.**
- **WORKS — lower leakage at matched accuracy** on covariate shift. `glsvae+inv`
  reaches the lowest `I(Z;D|Y)` **and** the lowest `I(Y;D|Z)` of all four methods at
  the same target accuracy (covariate+label: `I(Z;D|Y) 0.371`, `I(Y;D|Z) 0.106` vs
  erm `0.624 / 0.179`, dual `0.574 / 0.189`).
- **WORKS — the concept-shift test separates.** The per-domain likelihood
  correction `delta_d`'s held-out ELBO gain is `~0.41–0.48` on concept-bearing DGPs
  vs `~0.17–0.22` on pure covariate/label DGPs — a `~2.5×` separation, in the
  predicted direction.
- **DOES NOT WORK — "both CMIs vanish BY CONSTRUCTION" is false** for an amortized
  encoder. `I(Z;D|Y)` never reaches ~0; it floors at `0.37–0.47` and **requires the
  explicit penalty** (the structure alone gives `0.45–0.49`, only ~25–30% below
  ERM). The shared prior `p(z_y|y)` constrains only the *aggregate* per-class law,
  so the encoder still parks domain-`d` samples in a domain-specific *region* of
  class `y`'s cluster. The structural win is that the needed penalty is *fight-free*,
  **not** that it is unnecessary. The original task framing ("makes both CMIs vanish
  by construction") overpromises; the defensible framing is "dissolves the label
  tension so a single encoder penalty drives both down without fighting."
- **COST — concept shift still breaks transfer accuracy.** On concept DGPs the GLS
  invariant decode cannot fix the flipped target (concept-only acc `38–40` vs erm/
  dual `54–61`). This is *correct* — concept shift is unfixable by invariance/GLS,
  only detectable — but it is a real DG accuracy cost, not a win.

**Verdict:** Route A is a legitimate *reframing* (fight-free co-minimization + a
working concept diagnostic), **not** a free-lunch architecture. It does not beat the
naive `dual`/`erm` on accuracy; its edge is lower simultaneous leakage without the
A2 fight, plus the concept-shift test. Worth keeping as the theory-illustrating
synthetic and as the seed for Route A on real EEG, *if* the `I(Z;D|Y)` floor is
acknowledged and the encoder penalty is kept.

---

## The model (`synthetic/gls_vae.py`)

DIVA-style (Ilse 2020) partitioned latent so the GLS correction is structural:

```
 y    ~ pi_d(y)              per-domain FREE label prior (learned logits)
 z_y  ~ p_theta(z_y | y)     SHARED class-conditional Gaussian  (domain-free)
 z_d  ~ p(z_d | d)           per-domain Gaussian  (absorbs the covariate offset)
 x    ~ p(x | z_y, z_d)      small Gaussian decoder (recon keeps the latent informative)
 amortized q_phi(z_y, z_d | x);  DIVA aux heads q(y|z_y), q(d|z_d) identify the split.
 GLS decode (transfer):  p(y|z_y) ∝ pi*(y) p_theta(z_y|y),  pi* = uniform.
 OPTIONAL encoder penalty: lam_inv * E KL( q(d|z_y,y) || uniform )  (BA upper bd on I(z_y;D|Y)).
 CONCEPT correction:  p_theta(z_y|y,d) = p_theta(z_y|y) exp(delta_d(z_y,y))/Z_d.
```

Why the partition is needed (validated below): a *single* latent + reconstruction
drives `I(Z;D|Y)` **up** as `beta_recon` grows (`0.38 → 0.65` for `beta 0.1 → 3`),
because reconstruction forces the per-domain covariate offset into `z`. Splitting
off `z_d` lets reconstruction explain the offset away from `z_y`. The reconstruction
term is also what makes the concept feature *visible* in the latent at all — without
it the encoder discards the (cross-domain-contradictory) concept channel and the
`delta_d` test reads ~0 even under real concept shift.

---

## Result 1 — GLS-VAE vs naive dual (held-out CMIs, same probes), 4 seeds

`I(Z;D|Y)`, `I(Y;D|Z)` measured by the *frozen-encoder held-out probes from
dual_cmi_v2* (apples-to-apples). For GLS-VAE the feature is `z_y` (mean).

| DGP | method | tgtAcc | I(Z;D\|Y) | I(Y;D\|Z) |
|---|---|---:|---:|---:|
| covariate-only | erm | 76.9 | 0.670 | 0.131 |
| covariate-only | dual | 77.0 | 0.633 | 0.141 |
| covariate-only | glsvae | 76.2 | 0.485 | 0.064 |
| covariate-only | **glsvae+inv** | 75.8 | **0.471** | **0.059** |
| covariate+label | erm | 76.7 | 0.624 | 0.179 |
| covariate+label | dual | 78.6 | 0.574 | **0.189** ↑ |
| covariate+label | glsvae | 76.9 | 0.445 | 0.113 |
| covariate+label | **glsvae+inv** | 76.4 | **0.371** | **0.106** |
| all-three | erm | 59.0 | 0.917 | 0.149 |
| all-three | dual | 60.4 | 0.904 | 0.149 |
| all-three | glsvae | 54.6 | 0.826 | 0.162 |
| all-three | **glsvae+inv** | 57.1 | **0.709** | **0.134** |
| concept-only | erm | 53.9 | 0.598 | 0.397 |
| concept-only | dual | 60.7 | 0.602 | 0.405 |
| concept-only | glsvae | 37.7 | 0.633 | 0.438 |
| concept-only | glsvae+inv | 39.8 | 0.627 | 0.437 |

Reading:
- **glsvae+inv has the lowest BOTH leakages on the covariate DGPs at matched
  accuracy.** That is the desired "drive both down" outcome, achieved by the GLS
  structure + one penalty.
- **The naive `dual` shows the tension:** covariate+label `dual` *raises* `I(Y;D|Z)`
  to `0.189` (vs erm `0.179`) — its encoder penalty under label shift pushes the
  label-prior shift into the decoder gap, exactly A2. glsvae+inv instead *lowers*
  `I(Y;D|Z)` to `0.106`.
- **`I(Z;D|Y)` never approaches 0** for any GLS-VAE arm (`0.37–0.71`). The structure
  is *not* sufficient; the floor is set by the amortized encoder, not the generative
  prior. (Cf. the rock-solid leakage-KL estimator on real EEG hits ~0.02; that uses
  a direct posterior-KL penalty, not structure.)
- **Concept-only: GLS-VAE pays in accuracy** (38–40 vs 54–61) because the invariant
  GLS decode mispredicts the sign-flipped target. `I(Y;D|Z)` stays high (~0.44) for
  *every* method including glsvae+inv — the encoder penalty correctly does **not**
  remove concept leakage; only `delta_d` / target adaptation could.

## Result 2 — the "no fight" test under label shift (covariate+label DGP, 3 seeds)

Push the encoder pressure up; watch whether the decoder leakage `I(Y;D|Z)` is forced
up (the A2 tension):

| model | enc pressure | I(Z;D\|Y) | I(Y;D\|Z) | tgtAcc |
|---|---:|---:|---:|---:|
| naive-enc | 0.0 | 0.474 | 0.149 | 75.9 |
| naive-enc | 1.0 | 0.399 | 0.150 | 75.7 |
| naive-enc | 4.0 | 0.406 | **0.175 ↑** | 82.8 |
| glsvae+inv | 0.0 | 0.426 | 0.136 | 75.2 |
| glsvae+inv | 2.0 | **0.353** | 0.125 | 75.0 |
| glsvae+inv | 6.0 | 0.428 | 0.127 | 71.2 |

- **naive-enc:** `I(Z;D|Y)` floors at ~0.40 (it cannot push lower), and at high
  pressure the decoder leakage is forced **up** (`0.149 → 0.175`). The fight.
- **glsvae+inv:** the decoder leakage stays **flat and lower** (`0.136 → 0.125 →
  0.127`) across all pressures — the GLS decode already neutralized the label term,
  so there is nothing to fight. `I(Z;D|Y)` is minimized at `lam_inv≈2` (`0.353`);
  at `lam_inv=6` it over-regularizes and the encoder term creeps back up while the
  decoder term stays low (honest caveat: the sweet spot is moderate, not large).

This is the cleanest single piece of evidence for the Route-A thesis.

## Result 3 — variational concept-shift test (`delta_d` held-out ELBO gain, 3 seeds)

| DGP | ELBO(shared) | ELBO(+delta) | gain | ± | calibrated verdict (thr≈0.30) |
|---|---:|---:|---:|---:|---|
| covariate-only | -6.776 | -6.560 | 0.216 | 0.009 | no concept shift |
| covariate+label | -6.621 | -6.448 | 0.172 | 0.051 | no concept shift |
| all-three | -8.117 | -7.704 | **0.413** | 0.016 | **CONCEPT SHIFT** |
| concept-only | -7.514 | -7.032 | **0.482** | 0.067 | **CONCEPT SHIFT** |

- Concept-bearing DGPs (`0.41, 0.48`) separate from pure covariate/label DGPs
  (`0.17, 0.22`) by `~2.5×`, in the right direction → **a usable concept diagnostic.**
- **Caveat (the script's built-in `thr=0.20` gives a false positive):** covariate-only
  reads `0.216`, just over `0.20`, flagged "CONCEPT SHIFT" in the raw output. The
  separation is real but the absolute threshold must be **calibrated** — `~0.30`
  cleanly separates all four here. The leak is the residual covariate offset that
  reconstruction pushes into `z_y`, which `delta_d` then partly fits; it is small but
  nonzero. A null-calibrated threshold (permute domain labels, take the 95th
  percentile of the null gain) is the principled fix and the recommended next step
  before any real-data use.

---

## Honest assessment vs the baselines

- **Does NOT beat naive `dual`/`erm` on accuracy.** On balanced covariate DGPs it
  ties (75–77); on concept DGPs it is *worse* (the invariant decode cost). Consistent
  with the project's standing finding that invariance ties ERM on mean accuracy.
- **Beats them on simultaneous leakage without the fight** — the only place Route A
  is strictly better, and it is exactly the theory-predicted place.
- **The "by construction" promise is the overclaim.** For amortized inference the
  generative structure does not zero `I(Z;D|Y)`; an explicit penalty is still
  required. What the structure buys is *decoupling* (A4 realized in the model instead
  of in the loss), so that penalty no longer fights the decoder term.

## Recommended framing for the paper (if Route A is used)

State it as **"structured GLS decoupling makes the dual objective well-posed"**, not
"both CMIs vanish for free":
1. partitioned latent + GLS reference-prior decode ⇒ the label-shift term is removed
   *in the model*, so the A2 tension is dissolved (Result 2);
2. one encoder invariance penalty then drives `I(Z;D|Y)` down *without* raising
   `I(Y;D|Z)` (Result 1, glsvae+inv columns);
3. the surviving `delta_d` ELBO gain is a calibrated concept-shift test (Result 3).
This matches the `DUAL_CMI_THEORY.md` spine (covariate = `I(Z;D|Y)`, label = `pi_d(y)`
absorbed structurally, concept = residual measured by `delta_d`).

## Next steps
- **Null-calibrate** the `delta_d` test (domain-permutation null) and re-run; replace
  the hard `0.20` threshold. This is required before the real-EEG concept-shift test.
- Strengthen the `I(Z;D|Y)` floor: try a stronger encoder penalty schedule or a tight
  fixed prior variance; report whether it can reach the ~0.02 regime of the direct
  leakage-KL estimator (likely not — that is the honest limit of amortized structure).
- Port to real cross-site EEG (ADFTD, where `I(Y;D|Z)~0.20` is a *real* concept
  signal per `cmi-empirical-findings`): does `delta_d` fire there and stay quiet on
  MUMTAZ (`I(Y;D|Z)~0.005`)? That is the decisive test of whether Route A's diagnostic
  is clinically real, mirroring the `iib`-best-on-ADFTD result.
```
