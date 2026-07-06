# W1.geometry — geometry-only latent-diagonal falsification (results, facts only)

- **Status:** COMPLETE · **Units:** 72 (dataset,subject) — Lee2019_MI 54, BNCI2014_004 9, BNCI2014_001 9
  (90 (pair,subject)). Source seeds {0,1,2} averaged within unit; (dataset,subject) cluster bootstrap 10k.
- **Design:** frozen `W1_GEOMETRY_FROZEN.md`. Reuse frozen V2P bundles; channel-space perturbations
  `X'=PX` applied to Xa+Xe; balanced accuracy per operator per perturbation at uniform decision prior.
- **Data:** `w1g_per_operator_BA.csv`, `w1g_contrasts.csv`, `wave1_geom.report.json`.

## QC

- Perturbation harness correct (`none` reproduces per-operator BA; CORAL nonzero after argmax fix).
- Perturbations are real geometry stresses (identity BA drop vs `none`, all significant):
  reref +0.037 [+0.023,+0.053], gain +0.021 [+0.013,+0.030], dropout +0.028 [+0.019,+0.037].
- Null (`none`) primary contrast CI includes 0 (−0.0033 [−0.008,+0.001]).
- Real (dataset,subject) addressing.

## Per-operator balanced accuracy

| perturbation | identity | FRSC | fixed_iter | joint | latent_im_diag | pooled | CORAL-latent | EA-sensor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.582 | 0.524 | 0.502 | 0.502 | 0.580 | 0.583 | 0.582 | 0.579 |
| reref | 0.544 | 0.510 | 0.500 | 0.500 | 0.562 | 0.564 | 0.560 | 0.552 |
| gain | 0.561 | 0.513 | 0.500 | 0.500 | 0.580 | 0.576 | 0.577 | 0.579 |
| dropout | 0.554 | 0.507 | 0.500 | 0.500 | 0.575 | 0.576 | 0.574 | 0.577 |

## Primary contrast + secondaries (cluster bootstrap)

| perturbation | max(full-cov) − max(diagonal) | CORAL − FRSC | EA − CORAL |
|---|---|---|---|
| none | −0.0033 [−0.008, +0.001] ns | +0.058 [+0.043,+0.072] | −0.002 [−0.009,+0.004] |
| reref | −0.0063 [−0.012, −0.001] **sig, negative** | +0.051 [+0.036,+0.067] | −0.009 [−0.017,+0.000] |
| gain | −0.0036 [−0.010, +0.002] ns | +0.064 [+0.046,+0.082] | +0.002 [−0.004,+0.008] |
| dropout | −0.0001 [−0.007, +0.006] ns | +0.067 [+0.048,+0.086] | +0.003 [−0.006,+0.011] |

## Verdict (per the frozen interpretation grid)

- **Pre-registered falsification criterion (diagonal-adequacy falsified iff `max(full-cov) − max(diagonal)`
  CI excludes 0 with Δ>0 on a geometry perturbation, null including 0): NOT MET on any geometry
  perturbation.** On no geometry perturbation does a full-covariance operator significantly beat the best
  diagonal-latent operator (>0). On reref the contrast is significantly *negative* (best diagonal slightly
  above full-cov); on gain/dropout it is not significant. → **the latent-diagonal family is ADEQUATE for
  these sensor-geometry perturbations (falsification fails).**
- **Operator-choice within the diagonal family (secondary):** CORAL − FRSC = +0.05 to +0.07 (significant)
  and FRSC/fixed_iterative/joint sit near chance (0.50–0.52) under every condition, while pooled and
  latent_im_diag (also diagonal-latent) match the full-covariance operators. So the *best diagonal member*
  is adequate; FRSC/fixed_iterative/joint specifically are weak for geometry correction.
- **No sensor-space advantage:** EA − CORAL ≈ 0 (not significant) → latent full-covariance correction
  suffices; no sensor-space-only recoverable component beyond it.

## Checksums

`w1g.sha256` covers `W1_GEOMETRY_RESULTS.md`, `W1_GEOMETRY_FROZEN.md`, `w1g_per_operator_BA.csv`,
`w1g_contrasts.csv`, `wave1_geom.report.json`.
