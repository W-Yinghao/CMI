# CIGL_46 — Graph-DualCMI SOTA Roadmap (main-line pivot)

## 0. Project decision (PI, 2026-07-01)

The main line pivots from an **audit/control framework** to a **decoder**:

- **Main line — CIGL_46 / Graph-DualCMI:** a graph-aware, source-free, cross-subject EEG **decoder** trained
  with a **dual conditional-mutual-information objective** (encoder-side `I(Z;D|Y)` **plus** decoder-side
  residual `I(Y;D|Z)`).
- **Primary metric:** target bAcc / worst-subject / cross-dataset generalization — **not** leakage reduction.
- **Fallback — CIGL_45 (retained, not discarded):** the audit-hardening track (more permutations/seeds/FDR +
  baseline Pareto + reliance). If the SOTA thesis does not pan out, CIGL_45 stands on its own.

## 1. Protocol (unchanged)

- Every experiment is reviewer-GPU-gated.
- Non-GPU scaffolding + tests are pushed for review **before** any run; no GPU without an approved run spec.
- Fixed-λ=0.010 stays as an honesty anchor.
- No CITA/DualPC/Tri-CMI changes; no fabrication; honest-null branches pre-committed.

## 2. Method thesis (the objective)

```text
L(θ) =  CE(Y | Z)                                   # task
      + λ_g · Ĩ(Z_g ; D | Y)                         # encoder graph  CMI  (GLS, reference = marginal)
      + λ_n · (1/C) Σ_v Ĩ(Z_v ; D | Y)               # encoder node   CMI  (GLS)
      + λ_e · Ĩ(A   ; D | Y)                          # encoder edge   CMI  (GLS; only if a per-sample edge object exists)
      + γ   · [ JS( h_full(Y|Z,D), h0(Y|Z,D) ) − τ ]_+   # decoder residual  I(Y;D|Z)  (JS residual, gated)
      + Ω_graph(A)                                   # adjacency structure penalties (future, constrained edge)
```

- **Encoder terms** remove class-conditional subject *style* from the representation.
- **Decoder residual** removes subject-dependent *decision rule* (concept shift, the IRM-style part); the
  intercept-only `h0` absorbs label-prior / calibration shift, so only genuine boundary shift is penalized.
- **GLS reweighting** `w_i = π*(y_i) / π_{d_i}(y_i)` on the CMI side makes encoder- and decoder-CMI decouple
  (`Ĩ(Y;D) = 0`).

**Why dual, not either alone.**

- `I(Z;D|Y)` alone can over-erase task-useful but subject-correlated EEG variation (individual ERD/ERS).
- `I(Y;D|Z)` alone is **gameable**: a high-capacity encoder can *write D into Z*, so extra `D` is
  uninformative given `Z` while the shortcut survives.

**Contribution reframing.** The audit machinery (posterior-KL leakage, decoder-CMI probe, node/edge maps)
becomes **diagnostics / mechanism evidence**, not the headline. The headline is a **stronger cross-subject
decoder**.

## 3. Grounding corrections (verified in `project/cigl`, 2026-07-01)

1. `cdann` in `cmi/methods/dg_penalties.py` is **conditional-DANN** (`concat[Z, one_hot(Y)]`), **not** Long et
   al. CDAN (`Z⊗Ŷ`). CDAN is a to-implement item.
2. The static DGCNN graph adapter was **not** in the runner registry. → **fixed** (`build_backbone('DGCNNGraph')`).
3. `NodePosterior` did not condition on node id. → **fixed** (added `e_v` embedding → `q(D|Z_v,e_v,Y)`).

## 4. Backbone plan: FB-LGG-DualCMI (do not bet on raw dynamic GraphCMINet)

The v0.6 negative result stands (free per-sample `A(x)` overfits → subject fingerprint). The SOTA backbone is
a strong static / constrained-dynamic hybrid:

1. Filterbank temporal stem (multi-band / depthwise temporal conv + log-var pooling over several windows).
2. Local–global electrode graph (local motor/10–20 partitions + shared learnable global `A₀`; static first).
3. Constrained dynamic residual `A(x)=A₀+ΔA(x)` with Frobenius / L1 / symmetry / anatomical-prior penalties —
   opened **only after** the static backbone beats strong task baselines.
4. Graph + CNN gated fusion (SOTA must not be crippled by a no-bypass audit constraint); ablate graph-off /
   CNN-off / random-static-learned adjacency.
5. Expose `forward_graph(x) → (logits, graph_z, node_z, edge_logits_or_none, fused_z)`; decoder-CMI acts on
   the representation the classifier uses (`fused_z`/`graph_z`).

**Task baselines to beat:** EEGNet, ShallowConvNet, Deep4Net, EEG-Conformer, DGCNN/RGNN, TSMNet.

## 5. Scaffolding status (non-GPU; done)

| item | status | where |
|---|---|---|
| `DGCNNGraph` backbone registered | ✅ | `cmi/models/backbones.py` |
| `NodePosterior` node-id `e_v` (+ backward compat) | ✅ | `cmi/methods/graph_regularizers.py` |
| `graphdualpc` method (GLS enc graph/node/edge + JS-residual decoder, 4 weights, fail-closed) | ✅ | `cmi/train/trainer.py` |
| node/edge posteriors `weight`/`reference` + `log_pd_ref` | ✅ | `cmi/methods/graph_regularizers.py` |
| `label_correct` GLS-CE + distinct-`fused_z` fail-closed + GLS critic diagnostics | ✅ | `cmi/train/trainer.py` |
| `run_loso` grammar `graphdualpc:<λg>:<λn>:<λe>:<γdec>` + source/ablation/GLS metrics | ✅ | `cmi/run_loso.py` |
| target-label firewall test | ✅ | `tests/test_target_label_firewall.py` |
| MOABB/braindecode preflight | ✅ | `scripts/preflight_moabb_env.py` |
| tests (scaffold + firewall + graph_leakage regression) | ✅ 13 + 14 pass | `tests/` |

## 6. What we are NOT doing now (per PI)

- No `n_perm=1000`, no 10-seed runs, no full baseline suite, no R1 evidence-hardening megasuite **yet** (that
  is the CIGL_45 fallback).
- No "leakage-control framework" framing for the main paper.

---

## 7. First GPU pilot spec (request approval AFTER preflight passes; do NOT run yet)

### Datasets / folds / seeds

| dataset | folds | note |
|---|---|---|
| BNCI2014_001 | 0, 1 | 4-class |
| BNCI2015_001 | 0, 9 | fold 9 = the prior source-retention boundary case (must be in the pilot) |
| seeds | 0, 1, 2 | — |

### Methods (config strings)

| label | config | role |
|---|---|---|
| ERM-DGCNNGraph | `erm:0` (backbone `DGCNNGraph`) | task baseline |
| graphcmi plain | `graphcmi:0.010:0.010:0.000` | encoder-only I(Z;D\|Y) ablation |
| graphdualpc decoder-only | `graphdualpc:0.000:0.000:0.000:0.100` | decoder residual only |
| graphdualpc dual | `graphdualpc:0.010:0.010:0.000:0.100` | main method |
| CDANN-DGCNNGraph | `cdann:1` | conditional-adversarial baseline |

Pilot 1 is **graph-only** (5 configs above). EEGNet / ShallowConvNet ERM are a **sidecar** task-sanity
baseline, run separately in a braindecode-capable env (`icml`) after their own preflight — **not** part of
pilot 1's approval gate.

Gate metric note: use `worst_target_balanced_acc` and `per_target_balanced_acc_mean` (balanced accuracy),
**not** the raw `worst_target_acc`, from the `run_loso` summary.

### Scale-up decision gate (NOT leakage-based)

| criterion | threshold |
|---|---|
| mean target bAcc vs ERM-DGCNNGraph | **≥ +2 pp** |
| or worst-subject bAcc vs ERM-DGCNNGraph | **≥ +3 pp** |
| source bAcc drop | **≤ 2 pp** |
| decoder residual JS/CE | **decreases** |
| encoder leakage | does not explode |
| graph ablation (`zero_graph`≈chance, `permute_nodes`≪normal) | graph branch contributes |

If no target improvement → change architecture/objective **before** spending big compute.

### Environment (graph-only preflight PASSES as of P2c, 23ae29c/992c6ed)

Env: `eeg2025` (do **not** switch to `icml` for pilot 1). Export the BNCI2015 readable mirror before any run:

```bash
export MNE_DATASETS_BNCI_PATH=/projects/EEG-foundation-model/yinghao/cigl_bnci_readable
export MNE_DATA=/projects/EEG-foundation-model/yinghao/cigl_bnci_readable
```

`python scripts/preflight_moabb_env.py --datasets BNCI2014_001 BNCI2015_001 --backbones DGCNNGraph
--subjects 1 --no-download` → **PASS (exit 0)**: `DGCNNGraph` builds; BNCI2014_001 loads (576/22ch/4cls);
**BNCI2015_001 loads** (400/13ch/2cls) via the mirror (`results/preflight/bnci2015_readable_mirror_README.md`).
braindecode/EEGNet remains a WARN — **sidecar only, not a pilot-1 blocker**.

### Run-spec template (submit separately for approval)

```text
Branch / Commit SHA:
Environment (conda env, torch/moabb/braindecode):
Preflight command + output path:
Test command + output path:
Datasets / folds / seeds:
Methods / config strings / backbones:
Epochs / batch size / lr / warmup / n_inner:
Expected walltime / V100-hours:
Artifacts (per-fold JSON + aggregate):
Decision gate / abort criteria:
```
