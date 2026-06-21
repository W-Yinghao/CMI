# `acar` — Action-Conditional Counterfactual Adaptation-Risk Router

> *Predicting Negative Transfer, Not Distribution Shift, for Safe EEG Test-Time Adaptation.*

Isolated package for **Direction 2**: the leak-proof successor to the closed gate-falsification line
(A0 / A0′ / A0′-R / A0-PILOT). Pre-registration & frozen kill criterion: [`../notes/ACAR_FROZEN.md`](../notes/ACAR_FROZEN.md).

## Thesis

For each candidate test-time-adaptation **action** `a` and unlabeled batch `B`, estimate the **paired incremental
risk**

```
ΔR_a(B) = R_B(f_a) − R_B(f_0)          # f_0 = frozen, no-adaptation reference
```

— *does executing `a` beat doing nothing on this batch?* — from **label-free paired pre→post observables**
`φ_a(B) = [Δentropy, Δmargin, flip_rate, JS(p_0,p_a), Bures, post_sep, n_eff]`, and route conservatively: act only
when a leave-one-source-cohort-out **conformal upper bound** `U_a(B) = ΔR̂_a(B) + q_{1−α} < −δ`.

This is **not** shift detection and **not** absolute-accuracy estimation. The three deltas vs the dead A0 line —
paired-incremental (not absolute) target, action-conditional (not one score for all adapters), conformal control
validated by closed-loop loss (not AUROC) — are spelled out in the pre-registration.

## Why it might work where A0 failed

A0 scored *static* functions of the target ("is this batch weird?") and they were anti-aligned with harm. ACAR
scores what the action *actually did* to the batch (counterfactual pre→post change), which is mechanistically closer
to negative transfer. The go/no-go honestly tests whether that closes the gap; if not, it says `TERMINATE`.

## Layout

| file | role |
|---|---|
| `config.py` | frozen constants (cohorts, dump path, B, α, δ) + `ACARConfig` dataclass |
| `data.py` | load `erm_0` dumps; fit serialized source state; build natural recording-ordered batches |
| `actions.py` | action registry: `identity`, `matched_coral`, `spdim`, `t3a` → `(p_a, z̃_a)` |
| `features.py` | label-free paired `φ_a(B)` (+ A0 background scores as context coordinates) |
| `risk.py` | Phase-2 estimand `ΔR_a(B)` (NLL + 0-1) and harm labels — the ONLY place `y` is read |
| `regressor.py` | per-action `ĝ_a: φ_a ↦ ΔR̂_a` (monotone GBM; DeepSets upgrade deferred) |
| `conformal.py` | LOCO cohort-clustered conformal `U_a` + router decision + closed-loop replay |
| `run_gonogo.py` | leak-proof go/no-go harness: Phase-1 (y-free) + metamorphic guard + Phase-2 + G1/G2 decision |
| `tests/` | leakage-guard (label-permutation invariance) + determinism unit tests |

## Leakage discipline (enforced, not aspirational)

`route(state, z_batch)` never receives labels. Permuting `y_target` must leave `φ_a`, `U_a`, and the routed action
**bit-identical** (asserted per batch; violation aborts the run). Whole-batch aggregation only; no class-conditional
batch deletion; serializable source state verified bit-exactly against the deployed predictor. These are the exact
five repairs that killed the A0′ leak, carried over as hard guards.

## Run

```bash
# unit guards first (synthetic, fast)
python -m acar.tests.test_leakage_guard

# the go/no-go (GPU-free; reads archived erm_0 dumps)
python -m acar.run_gonogo --alpha 0.1 --delta 0.0 --out results/acar_gonogo
```

Decision (`PROCEED` / `MEASUREMENT_ONLY` / `TERMINATE`) is written to
`results/acar_gonogo/<hash>/acar_gonogo_summary.json` per the frozen schema. **A failed go/no-go terminates the
direction** — do not swap features and retry (pre-registered).
