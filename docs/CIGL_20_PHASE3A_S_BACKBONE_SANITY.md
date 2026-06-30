# CIGL Phase 3A-S — Known-Good MI Decoder Sanity Check (BNCI2014_001, fold-0)

> **EXPLORATORY diagnostic — NOT a benchmark / SOTA result.** One dataset, one LOSO fold, source-only,
> ERM only (no CMI regularization). The question is narrow and prerequisite, not a method claim.

## Why Phase 3A-S exists

Phase 3A-R showed that **GraphCMINet-ERM cannot learn** BNCI2014_001 4-class MI under strict source-only
training: all six repair candidates stayed at `source_probe` bAcc ≈ 0.33 (chance 0.25), the controls
passed (overfit 0.53, label-shuffle 0.26), and the train−source gap was tiny — i.e. **underfitting**,
not leakage. Before any further CIGL method work we must answer the prerequisite:

> Under the **same** strict source-only fold-0 protocol, can **known-good** MI decoders reach a
> non-degenerate source baseline that GraphCMINet cannot?

This separates two very different worlds: **GraphCMINet-specific underfitting** (a known-good decoder
succeeds) vs. a **protocol / preprocessing / data problem** (everything fails). It does not try to
rescue CIGL.

## Candidate models (small; pure-torch; no new dependencies)

| name | family | graph objects? | leakage audit? |
|---|---|---|---|
| `graphcmi_current_ref` | GraphCMINet (Phase 3A-R reference) | yes (`forward_graph`) | yes (light, n_perm=10) |
| `eegnet` | EEGNet (compact CNN) | no | no |
| `shallow_convnet` | ShallowConvNet (FBCSP-inspired) | no | no |
| `deep_convnet` | DeepConvNet (compact) | no | no |
| `dgcnn` | existing DGCNN graph baseline | no `forward_graph` | no |

The CNNs are **faithful minimal internal reimplementations** (`cmi/models/sanity_backbones.py`, pure
PyTorch — the repo's braindecode-backed wrappers are unavailable in the `eeg2025` run env, and the
reviewer authorized minimal internal wrappers). They are validated to **learn a learnable task**
(EEGNet/ShallowConvNet reach bAcc 1.000 on a separable synthetic in ~10 epochs) so a near-chance result
on real data is a genuine signal, not a broken model. Non-graph CNNs deliberately expose **no**
graph/node/edge objects and emit **no** leakage fields — they must not fabricate graph leakage.

## Strict source-only protocol (identical to Phase 3A-R)

- Same LOSO fold-0: target subject **excluded** from training and from `source_probe`.
- Per-config: train ERM on the source enc-train split; report `train`, `source_probe`, and `target_eval`
  bAcc/macro-F1 + the train−source gap.
- **Target labels are evaluation-only**: used solely for after-the-fact `target_eval`, never for
  training, early stopping, normalization, model selection, or the success decision.
- **Success is judged on `source_probe` only** (`source_probe` bAcc ≥ 0.45). `target_eval` never enters
  `selected_successful_models`. Summary records `success_selection_uses_target_eval=false`,
  `used_target_labels_for_selection=false`, `target_eval_is_evaluation_only=true`.

## Success / failure decision rules (reviewer decides; runner only reports a read)

- **A — known-good decoder reaches `source_probe` bAcc ≥ 0.45 while GraphCMINet stays ~0.33:** protocol
  is **usable**, GraphCMINet is the **bottleneck** → redesign/replace the graph task backbone (e.g.
  around a known-good temporal stem) **before** any CIGL regularizer claim.
- **B — all decoders stay near 0.33:** protocol / preprocessing / data split is **suspect** → diagnose
  MOABB loading, time window, scaling, labels, and fold construction.
- **C — a known-good decoder succeeds and exposes usable intermediate graph/node-like features:** later
  consider a CIGL-v2 built on a **repaired** backbone (not in this phase).
- **D — a known-good decoder succeeds but no graph-compatible model does:** CIGL **method** path stays
  paused; keep CIGL as a **diagnostic / audit** framework (Gate-2 leakage evidence) until a
  graph-compatible task backbone exists.

## Run

```bash
# CPU dry-run (pipeline + firewall validation; no GPU):
python scripts/run_cigl_phase3a_backbone_sanity.py --dry_run_synthetic --device cpu --seeds 0 1 --epochs 3

# Real run (after reviewer approval of the dry-run):
sbatch scripts/sbatch_cigl_phase3a_backbone_sanity_bnci001.sh
#  -> python scripts/run_cigl_phase3a_backbone_sanity.py --dataset BNCI2014_001 --device cuda \
#       --fold 0 --seeds 0 1 2 --epochs 80 --probe_epochs 100 --leak_n_perm 10
```

**Not authorized** (unchanged from Gate-3A-R): no full LOSO, no SEED/DEAP, no large λ-grid, no SOTA
table, no CMI regularizer in this phase, no CITA/DualPC/Tri-CMI, no PyG, no per-edge neural heads.
Outputs land in `results/cigl/phase3a_backbone_sanity/` (generated JSON gitignored; the tracked record
will be `docs/CIGL_21_...` after the real run).
