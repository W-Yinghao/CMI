# CIGL Phase 2 — Probe-Only Leakage Audit

This phase answers a single scientific question, **before** any regularizer is claimed and with
**no** target information:

> Does a GraphCMINet-ERM encoder, trained source-only, produce graph objects that carry
> **label-conditional domain leakage**?

The three objects and their leakage estimands:

| object | symbol | estimand |
|---|---|---|
| graph embedding `Z_g` | graph | `I(Z_g; D \| Y)` |
| node features `Z_v`   | node  | `(1/C) Σ_v I(Z_v; D \| Y)` |
| learned adjacency `A` | edge  | `I(A; D \| Y)` |

Phase 2 is **diagnostic only**. It does not train CIGL, does not sweep λ, does not touch
benchmarks. It produces the evidence that Gate 2 (`docs/CIGL_05_ACCEPTANCE_CRITERIA.md`) needs to
decide whether CIGL proceeds as a regularization method or pivots to a diagnostic framework.

Implementation: `cmi/eval/graph_leakage.py`. Tests: `tests/test_graph_leakage.py`. Synthetic smoke:
`scripts/smoke_graph_leakage.py` → `results/cigl/smoke_graph_leakage.json`.

---

## 1. What is measured

For an object `O ∈ {Z_g, Z_v, A}` we fit a held-out conditional domain probe `q(D | O, Y)` on
**frozen, detached** features and report the posterior-KL leakage proxy

```
leakage(O) = E_i KL( q(D | O_i, Y_i) ‖ π_{Y_i}(D) )
```

where `π_y(D) = p(D | Y=y)` is the Laplace-smoothed label-conditional domain prior. This is a
**plug-in proxy, not an unbiased CMI estimate** (see `cmi/methods/regularizers.py`): it equals the
true CMI only at probe optimality and otherwise under-estimates it. We additionally report:

- `domain_acc` — probe accuracy at predicting `D` on held-out trials;
- `prior_acc` — accuracy of the label-only prior baseline (`argmax_d π_y(d)`);
- `leakage_advantage = domain_acc − prior_acc` — interpretable "above label-only" signal;
- per-channel `node_leakage_map` (length `C`) and a non-neural per-edge `edge_leakage_map` (`C×C`).

The per-edge map is a **non-neural binned plug-in CMI** for interpretation only. CIGL never trains
per-edge heads; the trainable edge regularizer uses a compact upper-triangular summary.

---

## 2. Strict source-only rule

Everything in Phase 2 is computed on **source data only**:

- No target labels, no target covariates — not in probe fitting, model selection, or normalization.
- Probe fit and probe evaluation use **disjoint trial splits** (held-out by trial, never by row —
  node-rows of the same trial never straddle the split).
- `π_y(D)` is estimated from the **train** split only.
- The audit records `meta.strict_source_only = true`, `meta.used_target_labels = false`,
  `meta.used_target_covariates = false`. Any real-EEG probing in this PR is **exploratory** and
  must be labelled as such; it is not a benchmark result.

---

## 3. Why the permutation null MUST retrain the probe

The leakage estimate

```
E KL( q(D | O, Y) ‖ π_y(D) )
```

is a function of `(O, Y)` (through the probe) and of `Y` (through the prior). **It does not depend
on the observed `D` at evaluation time.** Therefore, shuffling the *validation* domain labels leaves
the KL essentially unchanged and produces a **false null** that looks like "no leakage" even when
the encoder leaks heavily.

The only valid null breaks the association the probe can *learn*:

1. Permute `D` **within each label group** of `Y`, **restricted to the probe TRAINING split**
   (`within_label_permutation_on_indices(y, d, train_idx, seed)`); the held-out (validation) `D` is
   left **unchanged**. Restricting the shuffle to the training indices preserves the **train-split**
   `π_y(D) = p(D|Y)` exactly — only the per-sample `O → D` pairing is destroyed. (A whole-dataset
   within-label permutation would preserve only the *global* `p(D|Y)`; it can move domains across the
   train/val boundary and so corrupt the train-split prior that the probe uses as its KL reference.
   The node null permutes at the **trial** level and is then repeated over channels.)
2. **Refit** the probe on the permuted training data.
3. Evaluate held-out KL (validation `D` is original throughout).
4. Repeat `n_perm` times → null distribution. Report `permutation_mean`, `permutation_std`, and the
   `(+1)`-smoothed one-sided p-value `p = (1 + #{null ≥ observed}) / (1 + n_perm)`.

This mirrors the repository's existing `decoder_leakage_probe` null (`cmi/eval/metrics.py`), which
also retrains under within-class domain permutation.

> **Statistical-strength note.** The synthetic smoke uses a tiny `n_perm` (5) — that is an
> engineering/directional check only (the smallest attainable p is `1/(n_perm+1) ≈ 0.167`). Real-EEG
> probing must use `n_perm ≥ 50` (preferably 99); do not cite the smoke's `p` as statistical evidence.

---

## 4. Audit output JSON schema

`audit_graph_objects(...)` returns `{"graph": {...}, "node": {...}, "edge": {...}}`. The smoke script
adds a `meta` block and writes `results/cigl/smoke_graph_leakage.json`.

```text
graph:
  kl_mean              float   observed held-out E KL(q‖π_y)
  permutation_mean     float   mean held-out KL over retrained within-label-permuted nulls
  permutation_std      float
  permutation_p        float   (1 + #{null >= observed}) / (1 + n_perm)
  n_perm               int
  excess_over_null     float   kl_mean - permutation_mean
  domain_acc           float
  prior_acc            float   label-only prior baseline accuracy
  leakage_advantage    float   domain_acc - prior_acc
  kl_ci                {mean, ci_low, ci_high, std, n}   bootstrap CI of per-sample held-out KL
  n_train, n_val       int

node:  (all of the above) +
  node_leakage_map         list[float]   length C, per-channel held-out KL
  node_leakage_map_path    str           sidecar .npy (written by the smoke/runner)

edge:  (all graph fields) +
  edge_leakage_map         list[list[float]]   C x C non-neural binned CMI, symmetric, zero diagonal
  edge_leakage_map_path    str                 sidecar .npy

meta:
  n_samples, n_channels, n_classes, n_domains   int
  seed, n_perm, epochs                          int
  strict_source_only                            bool (true)
  used_target_labels, used_target_covariates    bool (false)
  setting                                       "strict_source_only_DG"
```

---

## 5. Gate 2 pass/fail criteria

Gate 2 passes when **all** of the following hold:

- **A. Above-null leakage.** On the synthetic smoke, each of graph / node / edge held-out
  `kl_mean` exceeds its own retrained `permutation_mean` by a positive margin (`excess_over_null > margin`).
  On real EEG (if run, exploratory), at least **two of three** objects clear their permutation null,
  and edge leakage in particular is non-zero / non-noise.
- **B. Well-formed, non-degenerate maps.** `node_leakage_map` has length `C`; `edge_leakage_map` is
  `C×C`, symmetric, zero-diagonal; both are finite and not identically zero where leakage is present.
- **C. Strict source-only.** No target labels or covariates anywhere; probe fit/eval splits are
  disjoint by trial; `meta.strict_source_only = true`.
- **D. Real-EEG probing is optional here and explicitly exploratory.** The synthetic smoke is the
  binding gate for this PR; any real-data numbers are marked exploratory and are not benchmark claims.

### Fail action

If edge leakage is absent on real data, **do not** claim edge-CMI as a method. Per
`docs/CIGL_03_IMPLEMENTATION_PLAN.md` Phase 2, pivot to a node/graph diagnostic, narrow the main
claim to the object that does leak, or return to CITA. Do not advance to Phase 3 (regularizer
effect) until Gate 2 passes.

---

## 6. Acceptance commands

```bash
pytest -q tests/test_graphcmi_backbone.py tests/test_graph_regularizers.py tests/test_graph_leakage.py
python scripts/smoke_graphcmi.py --device cpu
python scripts/smoke_graph_leakage.py --device cpu
```

## 7. Deferred (not done in this PR)

- Real-EEG GraphCMI-ERM probing across seeds/datasets (Phase 2 exploratory → Phase 4 evidence).
- Seed-stability analysis of `node_leakage_map` / `edge_leakage_map` vs a random-map baseline.
- Wiring `audit_graph_objects` into `cmi/run_loso.py` to emit maps per fold (the trainer already
  reports `stepA_graph/node/edge_dom_acc` and `reg_graph/node/edge`; see the trainer diagnostics).
