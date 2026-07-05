# CIGL_66 — Measurement→control gap diagnostic (CPU-only, frozen artifacts)

```
Explanation/diagnostic phase — NOT a method search. CPU only, no retraining, no GPU. Uses only the frozen
real-EEG .audit.npz + CIGL_65 final tables. CIGL_65 tables are immutable (manifest appended, not rewritten).
Question: when CIGL reduced measured label-conditional leakage, what changed in the representation, and why did
the task classifier's reliance on the residual subject subspace NOT fall?
```

## Methodological guard (read first)
- **Cross-model subspace overlap is not well-defined.** ERM and CIGL are separately trained; even at equal
  dimension their coordinate systems differ by an arbitrary rotation. We therefore **never** compare raw ERM-vs-
  CIGL axes. All primary diagnostics are **within-run scalars** (each run's own subject subspace vs its own task
  head); cross-method comparison is done on the **scalars**.
- **Scale caveat:** `subject_energy` (raw ‖offset‖²) is **not scale-invariant** — it tracks activation magnitude,
  which differs wildly across methods (CDAN's adversarial logits blow it up to ~1e11 on 2015). So cross-method
  claims use only the **scale-invariant** scalars: **task_head_alignment** (a ratio), **effective_rank**,
  **top-k energy fraction**. Raw energy is reported but used only *within* a method (ERM-vs-ERM before/after is
  N/A here; we compare CIGL's own energy drop only as a within-CIGL sanity check).
- Fits are **source-only** (d ≠ target); target labels never enter any fit.

## Within-run graph_z scalars (mean over seeds/folds; `gap_diagnostic_summary.yaml`)
`align_k2` = fraction of the linear task head's row-space energy lying in the top-2 label-conditional subject
subspace (0 = orthogonal, 1 = head fully inside the subject subspace; random 2-of-64 ≈ 0.03).

| dataset | method | effective_rank | top2_energy_frac | **align_k2** |
|---|---|---|---|---|
| 2a | erm | 14.04 | 0.641 | 0.0044 |
| 2a | **cigl** | 14.69 | 0.655 | **0.0321** (7× ERM) |
| 2a | cdan | 9.88 | 0.862 | 0.0106 |
| 2015 | erm | 11.24 | 0.669 | 0.0497 |
| 2015 | **cigl** | 8.22 | **0.871** | **0.4396** (9× ERM) |
| 2015 | cdan | 7.03 | 0.910 | 0.0186 |

## Correlation with R3 task_drop (label_conditional k2; ERM+CIGL, n=126; Spearman, bootstrap 95% CI)
`gap_correlations.csv`:

| predictor | ρ (pooled) [95% CI] | sig | direction |
|---|---|---|---|
| **task_head_alignment_k2** | **+0.338 [+0.168, +0.504]** | **yes** | higher alignment → more reliance (removal hurts task) |
| graph_kl (measured proxy) | −0.342 [−0.507, −0.166] | yes | **higher measured leakage → LESS reliance** |
| node_kl | −0.182 [−0.367, −0.002] | (barely) | — |
| graphz_subject_energy | −0.270 [−0.444, −0.091] | yes | (scale-confounded; not primary) |

## Outcome A — the gap is explained by residual-subspace alignment, not by the KL proxy
1. **CIGL genuinely shrinks the graph_z subject subspace** (within-CIGL energy drop 2a 80050→13580, 2015
   64574→5741; and it reduces graph_kl *more* than node_kl — `gap_graph_node_mismatch.csv` — so the graph task
   path is the one being controlled, ruling out a graph/node-mismatch explanation).
2. **But the residual subject subspace becomes MORE task-head-aligned and MORE concentrated** — `align_k2` rises
   **7× (2a) / 9× (2015)** over ERM, `top2_energy_fraction` rises (2015 0.67→0.87), `effective_rank` falls (2015
   11.2→8.2). CIGL's classifier decision directions sit *inside* the residual subject directions far more than
   ERM's do.
3. **Alignment — not the KL proxy — tracks reliance.** Across folds, `align_k2` **positively** predicts R3
   task_drop (ρ=+0.34, CI excludes 0), while `graph_kl` **negatively** predicts it (ρ=−0.34). The measured proxy
   does not positively predict what the classifier leans on; the head/subject-subspace alignment does.
4. **Mechanism of the gap:** CIGL preferentially removes the *task-irrelevant* part of the subject leakage
   (dropping the measured KL), while the *task-entangled* subject directions persist and become a **larger
   fraction** of the residual — so removing the residual subject subspace (R3) costs CIGL's task *at least as
   much* as ERM's. Measured-leakage control ↓ without functional-reliance control ↓.

## Honest caveats
- The effect is **strong and clean on 2015** (target ≈ 0.59, above chance): align_k2 0.05→0.44. On **2a** (target
  ≈ chance) it is directionally the same (0.004→0.032, 7×) but **small in absolute terms** — a near-chance
  classifier has little reliance to measure. So the mechanism is best evidenced on 2015.
- `align_k2` and `graph_kl` correlations with task_drop are **comparable in magnitude** (±0.34); the claim is that
  alignment is in the mechanistically-correct *direction* and the KL proxy is not — not that alignment is a
  dominant predictor.
- **CDAN is a caveated comparator only:** its raw subject_energy is pathologically inflated by adversarial logit
  scale, so only its scale-invariant scalars are interpretable; it is not part of the CMI central story.
- This explains the gap *from saved representations*; it does not claim to be the unique cause.

## Frozen-interpretation update (does not change CIGL_65; adds the mechanism)
CIGL_65 point #8 ("CIGL's measured-leakage reduction does not reduce classifier reliance") now has a mechanism:
**CIGL removes task-orthogonal subject leakage (measured KL ↓) but the residual, task-aligned subject subspace
persists and concentrates — so reliance does not fall.** This is a *within-representation* explanation via
scale-invariant scalars; it does not rest on any cross-model coordinate comparison.

## Artifacts (`results/cigl_r123/final/`)
`gap_spectrum.csv` (378), `gap_alignment.csv` (756), `gap_correlations.csv` (24), `gap_graph_node_mismatch.csv`
(4), `gap_diagnostic_summary.yaml`. Code: `cmi/eval/gap_diagnostic.py`, `scripts/analyze_cigl_gap.py`,
`tests/test_gap_diagnostic.py` (8 pass). Figures deferred (the scalars carry the evidence).
```
```
