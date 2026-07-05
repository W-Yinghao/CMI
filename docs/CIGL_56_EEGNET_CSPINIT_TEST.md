# CIGL_56 — P10: does source-CSP init help the strongest decoder? (EEGNetMiniCSPInit)

Branch `project/eegnet-cspinit-test` off `d1ebdb7`. PI gate after P9: P8-D repositioned (mechanism-positive,
NOT SOTA — a plain EEGNetMini decodes the 2a subjects better). **Highest-information next question: does the
CSP-init mechanism (proven on the FBCSP-LGG graph model in P8) also improve the strongest compact decoder,
EEGNetMini?**

- If YES → the mechanism upgrades from "helps our graph model" to a **transferable spatial-filter
  initialization principle for EEG decoders**.
- If NO → clean negative: "CSP-init helps the graph/FBCSP-LGG architecture, but compact CNNs already learn
  better spatial filters on their own." Either way, informative.

## P10-A — EEGNetMiniCSPInit (implemented)

`cmi/models/sanity_backbones.py::EEGNetMiniCSPInit(EEGNetMini)`: source-CSP-initializes the **depthwise
SPATIAL conv** (`block1[2]` = `Conv2d(F1, F1*D, (C,1), groups=F1)`, F1*D=16 spatial filters) from
`cmi/models/csp_init.source_csp_filters` (one-vs-rest CSP for 2a, binary for 2015; top-16 by discriminability),
then trains normally (not frozen). `spatial_init='source_csp'` triggers train_model's generic CSP hook.
**FIREWALL:** CSP fit on source-train only — target excluded by LOSO, source-val excluded before the fit;
target labels eval-only. Metadata: `csp_fit_subjects/excluded_target/excluded_source_val/n_filters_used/rank/
shrinkage/classes`. **EEGNetMini itself is UNCHANGED** (frozen P9 baseline; a test asserts forward-identity).

## P10-B — factorial (round 1: CSP-init only, NO aux)

```text
A. EEGNetMini          (random init, frozen P9 baseline)
B. EEGNetMiniCSPInit   (source-CSP init)
```
No auxiliary loss this round — isolate whether CSP-init alone helps EEGNet. Aux only if B shows a stable gain.

## Contract / tests

`EEGNetMiniCSPInit` reachable via `--backbone`; no braindecode import; no `forward_graph` (non-graph CNN);
`forward->(logits,z)`. tests/test_eegnet_cspinit.py (6): CLI+build no-braindecode/no-graph; CSP-init changes
the depthwise spatial conv (2a 1-vs-rest & 2015 binary); firewall (target+source-val excluded); **EEGNetMini
baseline forward-identical/untouched**; trains + firewall-deterministic. + full regression.

## GPU gate — full-LOSO seed0 (no 2-fold)

CPU smoke first (2a+2015 × {EEGNetMini, EEGNetMiniCSPInit}, target 0, max_subjects 7, epochs 2: CLI / no
braindecode / CSP firewall / metadata / 0 NaN / JSON). Then:

```text
Backbones: EEGNetMini erm:0, EEGNetMiniCSPInit erm:0
Datasets: BNCI2014_001 (9) + BNCI2015_001 (12)   Seed: 0   source-only, target eval-only, --source_val_early_stop
21 folds x 2 variants = 42 jobs, max concurrency 8   out results/p10_eegnet_cspinit_s0/
```

**PASS (pre-committed):** PRIMARY 2a CSP-decodable {1,3,8,9}, `EEGNetMiniCSPInit − EEGNetMini ≥ +0.02`.
SECONDARY: 2a full non-negative; 2015 full ≥ −0.01; no source memorization (source_bacc not ~1.0 with
source-val/target drop). If seed0 passes → seeds 1/2 (best variant). If not → NO seeds, NO CSP-filter-count
sweep.

## Frozen
P8-D (no tuning), EEGNetMini baseline, P6/P7/P7b, CMI-as-main, aux (round 1), official-braindecode 189-job GPU,
new-dataset GPU. Baseline was positioning, not a tuning trigger.
