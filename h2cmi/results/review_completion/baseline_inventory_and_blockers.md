# Baseline Inventory And Blockers

| baseline | present | runnable same split | representation | target labels for adaptation | note |
|---|---:|---:|---|---|---|
| EA / source-recolored EA | yes | yes | sensor-space EA/source-recolored, appears as `source_recolored_ea` | no | present in W1/W2/W1.geometry |
| CORAL | yes | yes | latent CORAL/full-cov in W1.geometry; pooled/CORAL-like comparators | no | present; not every natural panel labels it CORAL |
| Latent-IM-Diag | yes | yes | frozen latent diagonal affine IM-style | no | internal comparator only; not official SPDIM |
| Official SPDIM | no | no | SPD manifold features required | unknown | blocked: official code/env absent |
| T3A | not in this h2cmi branch artifacts | no | classifier/prototype TTA | no labels expected | mentioned in ACAR notes outside this branch but no same-split H2CMI artifact |
| Tent/SHOT/CoTTA/EATA | not in h2cmi artifacts | no | gradient/prototype TTA | no labels expected | not implemented in current H2CMI same-split pipeline |
| CMMN/STMA | no | no | signal/SPD alignment | no | not present |
| BTTA-DG | partial external reproduction | not same H2CMI split | official external SincAdaptNet panel | no | see W1B_REPRODUCTION; not cross-ranked against W1-A |
