# `acar` — Action-Conditional Counterfactual Adaptation-Risk Router

> *Predicting Negative Transfer, Not Distribution Shift, for Safe EEG Test-Time Adaptation.*

Isolated package for **Direction 2**: the leak-proof successor to the closed gate-falsification line
(A0 / A0′ / A0′-R / A0-PILOT). Pre-registration: [`../notes/ACAR_FROZEN.md`](../notes/ACAR_FROZEN.md) (v1), **amended
and superseded for execution by** [`../notes/ACAR_FROZEN_v2.md`](../notes/ACAR_FROZEN_v2.md) — the v2 amendment is the
binding protocol after an expert review found the v1 *implementation* (not the hypothesis) protocol-invalid.

## Thesis

For each candidate test-time-adaptation **action** `a` and unlabeled batch `B`, estimate the **paired incremental
risk**

```
ΔR_a(B) = R_B(f_a) − R_B(f_0)          # f_0 = frozen, no-adaptation reference
```

— *does executing `a` beat doing nothing on this batch?* — from **label-free paired pre→post observables**
`φ_a(B) = [Δentropy, Δmargin, flip_rate, JS(p_0,p_a), Bures, post_sep, n_eff]`, and route conservatively: act only
when the **subject-clustered conformal upper bound** `U_a(B) = ĝ_a(φ_a) + q_{1−α} < −δ`.

This is **not** shift detection and **not** absolute-accuracy estimation. The deltas vs the dead A0 line:
paired-incremental (not absolute) target, action-conditional (not one score for all adapters), and conformal control
validated by **closed-loop loss** (not AUROC).

## Guarantee scope (v2 — honest)

Disease-stratified (PD/SCZ separate), for an exchangeable **new subject / recording cluster** from the same
calibration population, the joint one-sided `U_a` has ≥ (1−α) **finite-sample marginal coverage** (calibration unit
= subject; 230 PD / 225 SCZ units → valid at α=0.1). It does **not** extend to new cohorts/clinics; leave-one-cohort-
out is reported as **empirical cohort robustness only**. (v1 wrongly used 2–3 cohorts as calibration units, where the
conformal rank exceeds the count — no coverage; that is fixed here.)

## Why it might work where A0 failed

A0 scored *static* functions of the target ("is this batch weird?") and they were anti-aligned with harm. ACAR
scores what the action *actually did* to the batch (counterfactual pre→post change), mechanistically closer to
negative transfer. The go/no-go honestly tests whether that closes the measurement→control gap; if not it says so.

## Layout

| file | role |
|---|---|
| `config.py` | frozen constants (cohorts, dump path, B, α, δ, K folds) + `ACARConfig` (with `config_hash`) |
| `data.py` | load `erm_0` dumps; fit serialized source state; natural recording-ordered batches; **subject** ids; retains `len<8` batches (forced identity) |
| `actions.py` | action registry: `identity`, `matched_coral`, `spdim`, `t3a` → `(p_a, z̃_a)` |
| `features.py` | label-free paired `φ_a(B)` (+ A0 background scores as context coordinates, no asserted direction) |
| `risk.py` | Phase-2 estimand `ΔR_a(B)` (NLL + 0-1) and harm labels — the ONLY module that reads `y` |
| `scoring.py` | shared label-free scoring path `score_actions(state, z, actions)` (single source of φ_a) |
| `deploy.py` | end-to-end deployment API `route_batch(state, routers, z) → (action, U, φ)` |
| `regressor.py` | per-action `ĝ_a: φ_a ↦ ΔR̂_a` (unconstrained — sign of harm unknown a priori; see v2 §A7) |
| `conformal.py` | subject-disjoint FIT/CAL split, joint nonconformity, honest `+∞` quantile, disease-strat `q` |
| `run_gonogo.py` | go/no-go: Phase-1 (y-free) + guards + Phase-2 + subject-CV G1/G2/coverage + LOCO descriptive + manifest |
| `tests/` | 8 hard guards (no-label API, route_batch label-invariance, whole-batch, ΔR label-sensitivity, serialize roundtrip, fallback retention, spdim drift, finiteness) |

## Leakage discipline (enforced, not aspirational)

`route_batch(state, routers, z)` never receives labels — proven structurally (no label arg) and behaviorally
(`φ_a`, `U_a`, action bit-identical across calls and a serialize round-trip; decision invariant to row order).
Whole-batch aggregation only; no class-conditional batch deletion (`len<8` → forced identity, **retained**). `y` is
reachable only in `risk.py` (Phase-2), and `ΔR_a` is asserted label-sensitive there.

## Run

```bash
# the npz dumps are gitignored; point at the main checkout (or symlink them in)
export ACAR_FEAT_DUMP=/home/infres/yinwang/CMI_AAAI/archive/lpc-cmi-failed/results/feat_dump_v4

python -m acar.tests.test_leakage_guard                 # 8 hard guards (fast)
python -m acar.run_gonogo --alpha 0.1 --delta 0.0 --out results/acar_gonogo
```

Decision (`PROCEED` / `MEASUREMENT_ONLY` / `TERMINATE`) + full per-disease metrics → `results/acar_gonogo/<git>_<cfg>/
acar_gonogo_summary.json`, with `run_manifest.json` (dump hashes, double-run hash, config hash). A failed go/no-go
closes the direction — do not swap features and retry (pre-registered). `RUN_QUARANTINED / PROTOCOL_INVALID`
(a broken protocol) is distinct from `TERMINATE` (a falsified hypothesis under a valid protocol).
