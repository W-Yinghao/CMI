# S2P_37 - FMScope-FSR Bridge Panel-1 Results

**Status:** PANEL-1 COMPUTE COMPLETE / INDEPENDENT VERIFIER PASS / PM REVIEW REQUIRED.

This is a technical experiment record, not manuscript text. It does not authorize Panel 2, Phase-D1 training,
fine-tuning, an aperiodic audit, or writing.

## Authority

```text
S2P parent node:
  b981a41777aaca6f529a72b93d427cdc51bc4a92

FMScope public authority:
  https://github.com/Jimmy110101013/fmscope
  09885016a00db6c7de0074304c455c50685100c9

Public caches:
  EEGMAT x CBraMod   b4ed9917...03875bc
  SleepDep x CBraMod da8280e0...5a2b091
```

No raw EEG, FM weight, new pretraining, or fine-tuning was used. All results are conditional on the two public
frozen-feature caches.

## B0 exact public-path gate

| Cell | Label BA pre | Label BA post global LEACE | Delta | Subject BA pre | Subject BA post | Nonlinear post | Rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EEGMAT | 0.6944 | 0.7500 | +0.0556 | 0.7047 | 0.0000 | 0.0253 | 35 |
| SleepDep | 0.5694 | 0.6111 | +0.0417 | 0.7321 | 0.0036 | 0.0554 | 35 |

EEGMAT is within 0.0047 of the bundled historical delta (`+0.0602`) and passes the frozen positive-reference
gate. SleepDep executes and is interpretable, but its live delta differs from the bundled historical value
(`+0.0093`) by 0.0324. Its current live value is retained as an implementation/cache-version boundary; the
historical near-zero value is not substituted.

## Unified bridge task endpoints

Mean recording balanced accuracy across `3 seeds x 5 folds`:

| Cell | Endpoint | Unchanged | Global subject LEACE | Source-only subject LEACE | Global delta | Source delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| EEGMAT | Fresh probe | 0.6946 | 0.7464 | 0.6946 | +0.0518 | 0.0000 |
| EEGMAT | Exact head | 0.6946 | 0.6643 | 0.7089 | -0.0304 | +0.0143 |
| SleepDep | Fresh probe | 0.5696 | 0.6143 | 0.5661 | +0.0446 | -0.0036 |
| SleepDep | Exact head | 0.5696 | 0.6286 | 0.5679 | +0.0589 | -0.0018 |

The fresh-probe global-minus-source gaps are `+0.0518` for EEGMAT and `+0.0482` for SleepDep. The public-style
gain therefore does not survive the source-only information contract in either cell.

EEGMAT provides the cleanest endpoint separation: global LEACE helps a newly refit head by 5.18 pp while harming
the frozen original head by 3.04 pp. This licenses a fresh-head conditioning/adaptation interpretation, not a
claim that the original EEGMAT predictor relied harmfully on the erased subject axis.

## Random-removal attribution

Each cell, regime, and endpoint uses 100 paired same-rank draws and 100 paired removed-variance-matched draws.
Empirical p-values are Holm-corrected across all eight bridge comparisons within each null family.

```text
EEGMAT global fresh:
  subject delta                  +0.0518
  same-rank random mean          +0.0147
  raw / Holm p                    0.0198 / 0.1386
  variance-matched random mean   +0.0256
  raw / Holm p                    0.0297 / 0.0891

SleepDep global fresh:
  subject delta                  +0.0446
  same-rank random mean          +0.0102
  raw / Holm p                    0.0198 / 0.1386
  variance-matched random mean   +0.0067
  raw / Holm p                    0.0099 / 0.0792
```

No bridge cell clears both familywise-corrected null families. The global fresh-probe gains are positive and larger
than the average random-removal gains, but they are `POSITIVE_NOT_FAMILYWISE_SPECIFIC`. They cannot be called a
confirmed identity-specific utility benefit.

## Subject-axis transferability

Mean source-to-held-out diagnostics across folds:

| Cell | Source/target subspace overlap | Mean principal angle | Held-out subject scatter removed | Cross-recording subject BA pre | BA post source erasure | Chance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EEGMAT | 0.7856 | 25.7 deg | 0.7890 | 0.5179 | 0.3631 | 0.1393 |
| SleepDep | 0.9052 | 15.5 deg | 0.9047 | 0.5278 | 0.3184 | 0.1393 |

The source-fitted axis does transfer geometrically: it removes most held-out between-subject mean scatter and
reduces cross-recording identity decoding. It does not reduce identity to chance, and this geometric transfer does
not produce fresh-head task benefit. The diagnostic-to-deployment gap is therefore not explained by a complete
failure to find the held-out subject geometry; it is a gap between removing that geometry and improving task
utility.

## Task-geometry modifiers

| Cell | Source direction consistency | Source subject-task overlap | Post-erasure source MLP subject BA |
| --- | ---: | ---: | ---: |
| EEGMAT | 0.1565 | 0.9130 | 0.0269 |
| SleepDep | 0.0259 | 0.9382 | 0.0099 |

EEGMAT has higher cross-subject task-direction consistency than SleepDep, as expected for the positive-marker
cell. Both task directions nevertheless have high projection into the fitted source subject span under this
rank-35/approximately-rank-28 LEACE geometry. The absolute overlap is descriptive and cannot by itself establish
functional reliance.

## Independent verification

The verifier passes 21/21 checks:

```text
public commit and cache hashes
B0 gate replay
2,160 exact fold-assignment rows
90 fresh and 90 exact-head rows
12,000 random-null rows
same-rank and 1e-10 variance-match contracts
independent dense-operator feature canaries
independent task-metric canaries
independent Holm/claim-ledger replay
target-information firewall
no auto-launch scope
```

Dense and algebraically equivalent low-rank LEACE application differ by at most `3.5e-6` on one EEGMAT NLL
canary; BA, Kappa, and features quantized to eight decimals agree. This is recorded as floating-point operation-
order tolerance, not hidden as byte identity.

## Licensed conclusions

```text
READY:
  The public FMScope-style EEGMAT global-LEACE fresh-probe gain reproduces on the pinned cache.

READY:
  That fresh-probe gain disappears under fold-wise source-only eraser fitting.

READY:
  Source-fitted subject geometry transfers substantially to held-out subjects, but removing it does not improve
  fresh-head task utility in either Panel-1 cell.

READY:
  On EEGMAT, fresh-probe improvement and exact-head reliance have opposite signs under global LEACE.

READY:
  No identity-specific utility benefit survives the frozen same-rank, variance-matched, and Holm attribution gate.
```

## Forbidden conclusions

```text
FMScope is invalid.
Subject erasure never helps.
Subject axes do not transfer across subjects.
LEACE benefit is purely random dimensionality reduction.
The original EEGMAT predictor relies harmfully on subject identity.
Source-only erasure is deployably beneficial.
SleepDep is a stable near-zero negative control under every public implementation snapshot.
Panel 1 generalizes to FACED, SEED-V, ISRUC, task-trained encoders, or other FMs.
```

## Disposition

```text
Panel 1:
  COMPLETE / VERIFIED / PM REVIEW READY

Panel 2 on immutable S2P checkpoints:
  HELD

Phase D1 protocol:
  FROZEN / unchanged

Phase D1 training:
  HELD

1/f carrier audit:
  NOT STARTED

Fine-tuning / new pretraining / writing:
  NOT AUTHORIZED
```
