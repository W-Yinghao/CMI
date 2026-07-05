# Baseline sidecar — CSP-init FBCSP-LGG (P8-D) vs deep + classical baselines

189 GPU jobs (EEGNetMini/ShallowConvNetMini/DeepConvNetMini × 21 folds × 3 seeds), full-LOSO, source-only,
target eval-only, 0 NaN. + CSP+LDA classical (2 protocols, CPU). All **pure-torch faithful *minimal*
reimplementations** (`cmi/models/sanity_backbones.py`) — **NOT official braindecode nets** (braindecode absent
in eeg2025; available in envs `acar-v4-regen`/`icml` if official reruns are later required). P8 reference from
committed `09484e1` (`results/p8_cspinit/P8_AGGREGATE.csv`). Primary numbers independently recomputed.

## Table 1 — per-dataset aggregate (3-seed)

| dataset | model | mean | worst | std | 2a_dec | 2a_hard | source | best_srcval |
|---|---|---|---|---|---|---|---|---|
| 2a | **EEGNetMini** | 0.424 | 0.269 | .010 | **0.573** | 0.305 | 0.620 | 0.512 |
| 2a | ShallowConvNetMini | 0.398 | 0.254 | .001 | 0.530 | 0.292 | 0.685 | 0.488 |
| 2a | DeepConvNetMini | 0.398 | 0.240 | .010 | 0.527 | 0.296 | 0.610 | 0.518 |
| 2a | CSP_all_source | 0.373 | 0.231 | 0 | 0.491 | 0.278 | — | — |
| 2a | CSP_srcval_matched | 0.372 | 0.229 | .006 | 0.485 | 0.282 | — | — |
| 2a | **P8-D (csp+aux)** | 0.396 | — | — | **0.515** | — | — | — |
| 2a | P8-B (csp only) | 0.376 | — | — | 0.481 | — | — | — |
| 2a | P8-A (random) | 0.344 | — | — | 0.420 | — | — | — |
| 2015 | EEGNetMini | 0.623 | 0.505 | .004 | — | — | 0.785 | 0.690 |
| 2015 | ShallowConvNetMini | 0.619 | 0.503 | .013 | — | — | 0.803 | 0.709 |
| 2015 | **DeepConvNetMini** | **0.654** | 0.495 | .005 | — | — | 0.767 | 0.722 |
| 2015 | CSP_all_source | 0.614 | 0.488 | 0 | — | — | — | — |
| 2015 | **P8-D (csp+aux)** | 0.629 | — | — | — | — | — | — |

## Table 2 — Δ vs P8-D (2a-decodable is the primary)

| model | 2a_decodable Δ vs P8-D | 2a_full Δ | 2015 Δ |
|---|---|---|---|
| **EEGNetMini** | **+0.057** | +0.029 | −0.005 |
| ShallowConvNetMini | +0.014 | +0.002 | −0.010 |
| DeepConvNetMini | +0.012 | +0.003 | +0.025 |
| CSP_all_source | −0.024 | −0.023 | −0.015 |
| CSP_srcval_matched | −0.031 | −0.023 | −0.018 |
| P8-B (csp only) | −0.034 | −0.020 | +0.003 |
| P8-A (random) | −0.096 | −0.052 | −0.019 |

## Verdict (PI interpretation rules) — **D is NOT SOTA; reposition**

- **All three deep Mini baselines BEAT P8-D on the primary 2a-decodable endpoint** (EEGNetMini +0.057 — all 3
  seeds above D, 0.551/0.587/0.579; Shallow +0.014; Deep +0.012), and match/beat it on 2a-full. On 2015 D is
  mid-pack (beats EEGNet/Shallow, loses to DeepConvNet). Per the pre-committed rule *"a Mini baseline beats D →
  reposition D as a competitive graph/CSP-init method, not SOTA."*
- **The CSP-init MECHANISM survives**: P8-D beats classical **CSP+LDA** on every axis (2a-dec +0.024, 2a-full
  +0.023, 2015 +0.015) and beats its own random-init baseline (A) by +0.096 on 2a-dec. So "source-CSP
  initialization improves the FBCSP-LGG graph model, above classical CSP" holds — but a plain EEGNet decodes
  the same subjects better without any of it.
- **Honest framing:** the contribution is the **CSP-init mechanism** (+ the graph/leakage story), **not a
  raw-accuracy SOTA claim**. A standard EEGNet is the stronger decoder on the subjects that decode.

## Caveats
Minimal reimplementations, not official braindecode (a real EEGNet may differ — but the gap is large and
seed-stable). Single site, 3 seeds, n=4 decodable subjects. **P8-D remains frozen — no in-dataset tuning; the
baseline is positioning, not a tuning trigger.** Next gate is the PI's: keep D as a competitive/mechanism
result, and/or pursue the graph-leakage-audit angle, and/or an EEGNet-vs-CSP-init combination — PI's call.
