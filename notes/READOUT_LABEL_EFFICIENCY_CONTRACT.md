# Target Readout Calibration Ladder — pre-registration (PM-directed new PRIMARY line; manuscript FROZEN)

**Branch** `agent/cmi-trace-readout-label-efficiency` (worktree `/home/infres/yinwang/CMI_AAAI_readout`, base `de170ede`).
Only the owner stops/redirects a line. Reframes the CMI-Trace question from *"which subject subspace to delete?"* to:
**when the frozen representation Z already holds target task information, how do we correct the source-trained readout
into a future-session readout with as FEW target labels as possible?** Prior heads for review: Track A `a72f533b`
(FS-C), Track B `b17d05fe` (IL-C).

## Motivation (from the verified IL-C result)
Selection-only subspace identification failed at every information level; the one robust positive was that a fresh
readout on the FULL calibration session beats the frozen source head (+0.125/+0.135), while a fresh few-shot head is
WORSE than the frozen source head, and the benefit survives deleting the informed subspace (generic). So target
labels are useful for the **readout**, not for subspace surgery — but a from-scratch head throws away the structure
already in the source head and needs many labels. Hypothesis: **target labels help as a CONSTRAINED UPDATE of the
source head, not a from-scratch refit.**

## Datasets (4, run IN PARALLEL after methods+hyperparams are frozen; external drives status)
| role | dataset |
|---|---|
| existing evidence (dev/repro) | BNCI2014_001 |
| existing evidence (dev/repro) | BNCI2015_001 |
| confirmatory 1 | Lee2019/OpenBMI |
| confirmatory 2 | BCI-IV-2b |
Do NOT fully develop on the first two then shop for a favorable third: after the method+hyperparams freeze, all four
run together. Frozen ERM EEGNet features per fold (encoder never sees target labels). Each dataset FIRST emits and
freezes `configs/session_manifests/{DS}_session_split_manifest.csv`: source sessions, target calibration session(s),
target future-query session(s), trial counts, class counts, fallback/exclusion reason. **Sessions are NEVER chosen by
result.**

## Target-label budgets
`k ∈ {1, 2, 4, 8, 16, 32, Full}` labeled trials per class. For every finite k: **50 deterministic class-balanced
draws**; the SAME draw indices are used across ALL readout methods AND all representation controls; aggregate
**draw → seed → target subject** (draws averaged FIRST; draws are NOT independent samples; inference unit = target
subject, subject-cluster bootstrap + exact sign-flip). If a class has fewer than k cal trials in a dataset, that
budget is **reason-coded** (not oversampled beyond the real available count).

## Four readout methods (all on the same frozen Z; the head is the only thing that varies)
- **H0 — Frozen source head.** The source-trained head, no target labels. Reference `U_frozen`.
- **H1 — Fresh target head.** Standardized logistic head fit independently on the k cal labels (the existing
  few-shot/full-cal baseline). `U_fresh(k)`.
- **H2 — Source-anchored MAP head (PRIMARY).** From source head `(W_s, b_s)`:
  `min_{W,b} mean_i CE(W z_i + b, y_i) + α_W‖W − W_s‖²_F + α_b‖b − b_s‖²_2`.
  `(α_W, α_b)` selected ONLY via an outer-source **early→later pseudo-target** protocol (each source subject's
  early session = pseudo-cal, later session = pseudo-query; pick the α minimizing pseudo-query loss averaged over
  outer-source subjects). Fixed source-only grid `α_W ∈ {1e-3, 1e-2, 1e-1, 1, 10, 100}`; **the same α constrains the
  bias (`α_b = α_W`)** to avoid a 2-D hyperparameter search. Answers: *can a few target labels CORRECT the source
  predictor rather than relearn the task?* `U_MAP(k)`.
- **H3 — Bias-and-temperature calibration.** Fix `W_s`; optimize only per-class bias + one positive temperature T
  (logits = T·(W_s z) + b). Answers whether the full-cal gain is mostly class prior / decision threshold / logit
  scale rather than a full direction rotation. `U_bias(k)`.

## Representation controls — matched-random-subspace head refit (MANDATORY)
Each fold runs THREE representations; for EACH, refit the source head on the transformed source features, adapt with
the same cal draw, evaluate on the same query session:
- **Z0 — Native**: `Z`.
- **ZI — Source-fitted informed deletion**: delete the SAME source-fitted `B_cond` subspace used in the info-ladder
  (`whitened_cond_basis`, rank ≤ 8, whole informed subspace, NO target labels to select rank/subset).
- **ZR — Matched random deletion**: ≥50 random projectors matched on rank + source-only transform fitting + source
  head refit + source task retention + action count. A random transform enters the PRIMARY matched set ONLY if its
  source bAcc is within a pre-frozen tolerance of the informed transform's source bAcc (so a MORE task-destructive
  random deletion cannot manufacture a larger "head recovery" illusion).
Head adaptation gain: `G_h(P, k) = U_adapt(P, k) − U_source_head(P)`. Subspace-specific head effect:
`ΔG_h^specific(k) = G_h(P_informed, k) − E_R[ G_h(P_random, k) ]`. If this is ~0 while Z0/ZI/ZR all get similar gains →
`GENERIC_TARGET_READOUT_REFIT` (subspace erasure is not causal).

## Primary endpoints (inference unit = target subject; draw→seed→subject; subject-cluster bootstrap + exact sign-flip)
1. **Label-efficient utility** `ΔU_MAP-frozen(k) = U_MAP(k) − U_frozen`.
2. **Anchoring value** `ΔU_MAP-fresh(k) = U_MAP(k) − U_fresh(k)` (does anchoring to the source head beat from-scratch?).
3. **Low-capacity sufficiency** `ΔU_MAP-bias(k) = U_MAP(k) − U_bias(k)` (is a full head update needed, or bias/temp enough?).
4. **Representation specificity** `ΔG_h^specific(k)`.
5. **Minimal label thresholds**
   `k*_utility = min{ k : LCB95[ΔU_MAP-frozen(k)] > 0 }`, `k*_anchor = min{ k : LCB95[ΔU_MAP-fresh(k)] > 0 }`.
   Holm correction across budgets; NO post-hoc best-k from the curve.

## Result routing (owner decides next; nothing run unilaterally)
- **R-A** MAP wins at few labels (k≤8): MAP > frozen AND MAP > fresh, replicated on ≥1 confirmatory dataset, no clear
  harm elsewhere → a real deployable direction: fix source-prior readout adaptation; add stronger few-shot baselines;
  study whether only bias or a full head-direction update is needed.
- **R-B** only Full calibration is positive → readout adaptation works but needs many labels → next question =
  which supervision/structure lowers calibration sample complexity (NOT back to erasure).
- **R-C** bias/temperature ≈ MAP → bottleneck is prior shift / class bias / logit scale; no full head refit needed.
- **R-D** MAP works but Z0 ≈ ZI ≈ ZR (ΔG_h^specific ~ 0) → `READOUT_SHIFT_IS_GENERIC; SUBSPACE_ERASURE_IS_NOT_CAUSAL`
  (the current most-likely outcome).
- **R-E** informed representation significantly beats matched random: only if `LCB95[ΔG_h^specific] > 0` AND it
  replicates on ≥2 independent datasets do we RE-ACTIVATE the subspace-actionability hypothesis.
- **R-F** the confirmatory datasets do NOT replicate the full-cal head gain → the +0.13 is dataset/session-specific;
  do not develop it into a general readout method.
External-dataset status rules: dev-positive but new-negative = `DEVELOPMENT_ONLY`; ≥1 new replicates + the other no
clear harm = supports the readout hypothesis; new reverses = analyze task/session heterogeneity, no general claim.

## Status of the parked lines (PM)
`SUBSPACE_SELECTION = PARKED_NO_CALIBRATION_IDENTIFIABILITY` (not scientifically stopped); new erasure methods = HOLD;
`QUERY-LABEL_HINDSIGHT = BOUNDARY_DIAGNOSTIC_ONLY` (no transductive method branch); TOS/CMI = retained as measurement
& geometry diagnostics; full-strength cross-session = FS-C, no further same-class variants. HOLD: new target-X score /
new source proxy / new mechanism-consistency loss / new learned projector / TTE — unless the readout-genericity
control or an external dataset produces subspace-specific evidence.

## Deliverables (this task; NO manuscript changes; no amendment)
`notes/READOUT_LABEL_EFFICIENCY_CONTRACT.md` (this), `configs/cmi_trace_readout_label_efficiency.yaml`,
`tos_cmi/eval/readout_calibration.py`, `scripts/run_readout_label_efficiency.py`,
`scripts/aggregate_readout_label_efficiency.py`, `configs/session_manifests/{4 datasets}.csv`, tests, full real-EEG
matrix. Must cover: (1) source-only MAP α selection (pseudo-target); (2) target-query firewall (query X/Y only in the
final utility); (3) same label draws across all methods; (4) native/informed/random transform fairness; (5) random
rank + source-retention matching; (6) frozen source-head reproduction; (7) draw→seed→subject aggregation order;
(8) external-dataset independent status; (9) no manuscript changes.
