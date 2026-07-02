# CIGL_48 — BNCI2014 (4-class MI) bottleneck analysis (P4, non-GPU)

## 0. Question and verdict

F0 showed FBLGGGraph ERM **fails** the BNCI2014 (4-class) cross-subject gate (mean 0.296 vs DGCNN 0.342,
chance 0.25) while **passing** BNCI2015 (binary, 0.627 vs 0.575). This P4 analysis (no GPU; only the F0
JSON/preds + a CPU CSP/LDA baseline) answers: *why does FBLGG fail 4-class BNCI2014 but improve binary
BNCI2015?*

**Diagnosis: primarily a BACKBONE / feature-extraction bottleneck** — on the decodable BNCI2014 fold, a
classical CSP+LDA transfers cross-subject far better than FBLGG (0.483 vs 0.306), and even the simpler
static DGCNN beats FBLGG there (0.403 vs 0.306). FBLGG's node-preserving filterbank stem + local–global
graph + gated fusion does **not** extract the transferable spatial–spectral (CSP-style) discriminative
features that 4-class MI needs. **Secondarily**, one of the two F0 folds (subj 2) is an intrinsically hard
cross-subject target — CSP transfer to it is at chance (0.248) — so the small 2-fold F0 is partly noisy.

## 1. P4-B — data / split sanity (clean; not the cause)

```
BNCI2014_001: 9 subjects x 576 trials, class-balanced 144/144/144/144
              classes = [feet, left_hand, right_hand, tongue]
BNCI2015_001: 12 subjects, class-balanced 2-class [feet, right_hand]
```

No class imbalance, no label-order mismatch, channel counts/names as expected (22 / 13), source_val
subject held out by subject. The failure is not a data or split artifact.

## 2. P4-F — classical CSP+LDA baseline vs the graph backbones (the decisive test)

Cross-subject LOSO on the **exact F0 folds** (source-only), same moabb data as F0. Within-subject 5-fold
CV is the decodability ceiling. (`results/fblgg_f0/F0_CSP_BASELINE.json`.)

### BNCI2014_001 (4-class, chance 0.25)
| fold (target) | within-subj CSP (ceiling) | cross-subj CSP+LDA | DGCNN ERM | **FBLGG ERM** |
|---|---|---|---|---|
| subj 1 | 0.717 | **0.483** | 0.403 | **0.306** |
| subj 2 | 0.521 | 0.248 (chance) | 0.281 | 0.281 |
| mean | 0.619 | **0.365** | 0.342 | **0.296** |

- On the **decodable** fold (subj 1, ceiling 0.72): CSP transfers to **0.483**, DGCNN 0.403, **FBLGG only
  0.306**. FBLGG loses ~18pp to CSP and ~10pp to the simpler DGCNN → **the FBLGG feature pathway is the
  bottleneck**, not the task.
- On subj 2: CSP transfer is chance (0.248) and within-subject is only 0.521 → subj 2 is an intrinsically
  hard cross-subject target. Both FBLGG and DGCNN also sit at ~0.281 here. This fold is near-unlearnable
  cross-subject and drags the 2-fold mean.

### BNCI2015_001 (binary, chance 0.50)
| fold (target) | within-subj CSP (ceiling) | cross-subj CSP+LDA | FBLGG ERM |
|---|---|---|---|
| subj 1 | 0.980 | 0.672 | 0.715 |
| subj 10 | 0.673 | 0.585 | 0.568 |
| mean | 0.827 | 0.629 | 0.627 |

- FBLGG **matches** CSP cross-subject on the binary task (0.627 vs 0.629) and beats DGCNN (0.575). The
  graph design is adequate here. The gap on 2a is 4-class-specific, not a global backbone failure.

## 3. P4-C — class-wise confusion on 2a (pooled over all 6 cells)

```
             pred: feet  left  right tongue   recall
       feet        164   234   190   276      0.190
  left_hand        123   311   197   233      0.360
 right_hand        132   277   247   208      0.286
     tongue        154   234   176   300      0.347
one-vs-rest bAcc:  0.516 0.536 0.534 0.535
```

- Failure is **diffuse**: all four classes are weakly separable (one-vs-rest bAcc ≈ 0.52–0.54, barely
  above 0.5); no single class collapses and all four are predicted. `feet` is worst (recall 0.19, mostly
  mistaken for left_hand / tongue).
- This is the signature of a representation lacking transferable 4-class structure — not a specific
  left/right confusion or a one-class collapse (both ruled out).

## 4. P4-D — branch ablation deltas (main − ablation; larger ⇒ more load-bearing)

| dataset | Δ zero_graph | Δ zero_temporal | Δ permute_nodes | reading |
|---|---|---|---|---|
| BNCI2014_001 | +0.023 | +0.012 | +0.020 | near-zero: NO branch carries transferable signal |
| BNCI2015_001 | +0.085 | +0.003 | +0.114 | graph + node content strongly load-bearing; temporal not |

On 2a neither branch is load-bearing (there is little signal to remove). On 2015 the graph/node pathway
is clearly load-bearing — consistent with FBLGG working there. Gate values were not instrumented in F0;
recommend saving the fusion gate per-trial in any future run (non-GPU note; no rerun now).

## 5. P4-A — early-stopping / source-val behavior

| dataset | best_ep range | best_src_val | target bAcc | final_train | final_val |
|---|---|---|---|---|---|
| BNCI2014_001 | [1, 155] | 0.326 | 0.296 | 0.990 | 0.278 |
| BNCI2015_001 | [10, 34] | 0.638 | 0.627 | 1.000 | 0.581 |

On 2a the model fits source-train ~0.99 but **best-source-val is ≈ chance (0.33) at every best_ep** — there
is no epoch at which it generalizes cross-subject. Early stopping is working mechanically (it restores the
best epoch); there simply is no good epoch. Grouping is correct (`central_strip_v1`, verified in F0).

## 6. Diagnosis

1. **Primary — backbone feature extraction (actionable).** CSP+LDA extracts spatial–spectral discriminative
   features that transfer to subj 1 (0.483) where FBLGG reaches only 0.306; DGCNN (0.403) also beats FBLGG.
   FBLGG's design deliberately **omits spatial collapse** (keeps channels as graph nodes) and delegates
   spatial integration to the local–global graph. On 4-class this graph-based spatial integration is weaker
   than CSP's discriminative spatial filters. The `central_strip_v1` grouping also fragments 22 channels
   into 9 small groups (≤3 nodes each), limiting within-group spatial mixing. Net: the extra FBLGG machinery
   is a step **backward** vs DGCNN on 4-class.
2. **Secondary — hard fold (not architecture).** subj 2 is near-unlearnable cross-subject (CSP 0.248,
   within-subj 0.521). The 2-fold F0 is partly noisy; a fuller-fold 2a evaluation would de-noise the mean
   (a GPU decision — frozen).
3. **Not the cause:** data/splits (balanced, clean), grouping (correct), early stopping (working), CMI
   (never run).

## 7. Recommended architecture patch (NOT implemented; for PI approval)

Direction is a **spatial–spectral front end**, since the CSP gap is the clearest signal:

- **Add EEGNet/FBCSP-style depthwise SPATIAL filtering** (the component FBLGG omitted) — learnable spatial
  filters on band-power, either inside the temporal stem or as a parallel feature branch fused with the
  graph readout. This directly targets the CSP transfer gap on 4-class.
- **Keep the graph branch** (load-bearing on 2015) but do not rely on it alone for 4-class spatial
  discrimination; let a CSP-like branch carry the spatial-filter features.
- **Reconsider grouping granularity for 4-class**: 9 tiny groups may over-fragment; a coarser or overlapping
  local structure could preserve more spatial contrast (evaluate on CPU before any GPU).
- **Instrument the fusion gate** (save per-trial gate values) so a future run shows whether fusion collapses
  to one branch.

Do NOT implement until the PI approves a direction; this section is a recommendation, not a change.

## 8. Frozen

No F1 / graphcmi / graphdualpc / dec_scale=300 / λ,γ sweep / more FB-LGG seeds / extra BNCI2014 folds /
EEGNet-Shallow GPU sidecar / DGCNN reruns / dynamic edge. The only compute used here was CPU analysis +
one CPU CSP/LDA baseline. `dec_scale=300` remains the F1 default candidate, unused until a backbone clears
the 4-class gate.
