# CIGL_62 (R3) — Reliance: is label-conditional subject leakage *load-bearing* for the task? (non-GPU scaffold)

```
Status:
  Engineering scaffold complete (R3).
  Scientific evidence PENDING real EEG.

Validated by tests / synthetic fixtures (27 tests, CPU):
  label-conditional subspace recovers the planted subject direction; reliance-row schema; head-replay path;
  source-fit probe fallback; 8 firewall guarantees + 1 CMI distinguishability check; node-masking structure +
  determinism; spatial-correlation Spearman/Pearson + bootstrap CI + NaN/constant safety; headless topomap grid.

NOT validated (needs real-EEG full-LOSO .audit.npz):
  whether ERM's task drop under subject-subspace removal actually EXCEEDS CIGL's (the reliance claim);
  whether the effect is stable across folds/seeds/datasets. No such claim is made from synthetic fixtures.
```

Branch `project/cigl-r123-scaffold`. Consumes the R2 `.audit.npz` sidecar; **never retrains the backbone**; no GPU.

## The question (framing matters)
R3 does **not** ask "does masking leakage-looking dimensions reduce accuracy?" — any dimension you delete costs
something. It asks: **does the CMI-measured, label-conditional subject subspace become *functionally
load-bearing* for the task classifier, and does CIGL reduce that reliance relative to ERM?** The test is a
*differential*: fit a subject-predictive subspace on **source only**, remove it from the frozen representation,
and compare the resulting task drop for ERM vs CIGL on the **same** backbone. Primary comparison:
**`ERM task_drop > CIGL(graph+node) task_drop`**.

## Flagship — `cmi/eval/leakage_removal.py`
Fit a k-dim subject subspace on SOURCE `z`, project it out (`P = I − dirsᵀdirs`), re-measure task + subject bAcc.
- **Conditioning** (`CONDITIONINGS`): `label_conditional` (**primary** — subject-within-label offset Δ_{y,d}=
  mean(z|y,d)−μ_y, weighted √count, top-k SVD), `marginal_domain` (**control** — ignores label), `random_subspace`
  (**control** — deterministic random directions). The CMI test (`test_CMI_...`) shows label_conditional and
  marginal_domain recover **different** directions when the subject offset is label-conditional (cancels
  marginally) — exactly the regime CIGL targets.
- **Subject-leakage metric is itself label-conditional** (`_subject_bacc` decodes subject *within each label*,
  averaged) — a marginal decoder would miss the leakage CIGL removes.
- **k** is fixed (`PRIMARY_K=2`) / curve-reported (`DEFAULT_K_CURVE=(1,2,4,8)`), **never** target-selected.
- **Two eval modes**:
  - **head-replay** (sidecar carries `task_head_*`): `logits = z_removed @ Wᵀ + b` → **classifier reliance** (preferred).
  - **source-fit probe fallback** (no head): source-fit LDA task probe on z → **representation reliance** (weaker,
    explicitly labeled via `removal_mode="probe_replay"` / `probe_replay_used=True`). *We do not overclaim: a probe
    result is a statement about the representation, not the deployed classifier.*

## Secondary / descriptive (supportive only, never dispositive)
- **`node_masking.py`** — zero the top-k / bottom-k / random-k leakage nodes (by the per-node leakage map) in the
  frozen node features; a source-fit probe measures the drop. Reports `top_leak_mask_drop`, `bottom_leak_mask_drop`,
  `random_mask_drop_mean`, `random_mask_drop_ci`, `top_exceeds_random`. **Top-leak > random masking is supportive
  evidence, not proof of reliance.**
- **`spatial_correlation.py`** — Spearman (primary) + Pearson (secondary) between two per-node maps, cluster
  bootstrap CI over (fold, seed) groups; NaN-/constant-map safe. **Descriptive.**
- **`topomaps.py`** — headless (Agg, no GPU/display) topomap grid + optional PNG of a per-node scalar map.
  **Visualization.**

> **Topomaps are visualization. Spatial correlation is descriptive. Neither alone proves reliance.**
> **head-replay = classifier reliance; source-fit probe = representation-level reliance.**
> The only reliance *claim* R3 will make on real EEG comes from the flagship differential (ERM vs CIGL task drop).

## `.audit.npz` interface (R2→R3) + R2.5 verified head-replay export
Consumes `graph_z, node_z, y, d, model_logits` (+ `fold/seed/target_subject/method/dataset`) and optional
`node_leakage_map, task_saliency_map`.

**R2.5 (head-replay export wiring).** For the *classifier*-reliance claim, `.audit.npz` now carries the task
classifier's linear head and a **fail-closed** replay check (`cmi/eval/audit_npz.py:pack_task_head_fields`,
`cmi/eval/head_export.py`):
`task_head_weight [n_cls,Zin]`, `task_head_bias [n_cls]`, `task_head_kind`, `task_head_input`, and
`task_head_replay_ok`, `task_head_replay_max_abs_diff`, `task_head_replay_mean_abs_diff`.
`task_head_replay_ok` is **True only when** `kind=="linear"` **and** `max|model_logits − (graph_z @ Wᵀ + b)| ≤ 1e-5`.
The DGCNN adapter classifies with a single `nn.Linear(64→n_cls)` over the post-ELU `graph_z`, so replay is exact
(observed `max_abs_diff ≈ 3e-8` on real forward). Nonlinear/BN/dropout/fusion heads → `replay_ok=False` and **no
fabricated head-replay**. `cmi/eval/leakage_removal.py` uses head-replay **only when `head_replay_ok(data)`**, else
falls back to the source-fit probe (representation reliance, labeled). Also exported for firewall proof:
`source_indices, target_indices, source_val_indices`. Back-compat: absent head fields → validation still passes,
R3 uses the probe path. Wired into the run path at `scripts/run_cigl_phase3a_dgcnn_gn_regularizer_pilot.py`
`_train_eval` (guarded by `--audit_dir`; off by default so frozen confirmation runners are byte-unaffected); the
LOSO target subject is appended **eval-only** with a distinct domain id so R3 can hold it out.

## Firewall suite (`tests/test_leakage_removal.py`) — 8 + 1
F1 corrupting target labels leaves the fitted subspace unchanged · F2 corrupting target logits leaves the reliance
row unchanged · F3 source-only fit excludes the target subject (`firewall_passed`) · F4 target-z corruption is
inert for source metrics (μ_y / subspace / source bAcc use source only) · F5 k is fixed/curve, never
target-selected · F6 `random_subspace` deterministic under a fixed seed · F7 removal preserves shape + finiteness ·
F8 constant / rank-deficient z does not crash · **+CMI** label_conditional vs marginal_domain projectors are
distinguishable when the subject offset is label-conditional.

## R3 output schema (row, CSV/JSON)
`dataset, fold, seed, target_subject, method, representation, removal_mode, conditioning, k,
source_task_bacc_before/after, target_task_bacc_before/after, task_drop, source_subject_bacc_before/after,
subject_leakage_drop, head_replay_available, probe_replay_used, firewall_passed`. Conditioning ∈
{label_conditional (primary), marginal_domain (control), random_subspace (control)}.

## Tests: 37 CPU tests pass — R3 scaffold (leakage_removal 12 · node_masking 3 · spatial_correlation 6 ·
topomaps 6) + R2.5 head-replay export (head_export 8 incl. real DGCNN-adapter replay roundtrip, mismatch/
unsupported fail-closed, back-compat, R3-uses-head-when-ok, R3-probe-fallback, save_fold_audit e2e) + audit_npz 2.
End-to-end run-path check: `_train_eval --audit_dir` writes verified sidecars (`head_replay_ok=True`,
`max_abs_diff≈3e-8`) for ERM + CIGL graph_node. No GPU.

## Still gated (NOT launched — awaits PM)
Real-EEG full-LOSO seed0 `.audit.npz` for ERM / CIGL(graph+node) / DANN / cond-DANN / CDAN on 2a + 2015, then run
the flagship differential. **No GPU, no λ-curve, no 10-seed, no P10 expansion, no method-level scientific claim**
until the PM opens that gate. R2a is frozen; P10 is frozen.
