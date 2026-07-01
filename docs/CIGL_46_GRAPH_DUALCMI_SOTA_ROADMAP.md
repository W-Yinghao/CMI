# CIGL_46 — Graph-DualCMI SOTA Roadmap (main-line pivot)

> **Project decision (PI, 2026-07-01):** the main line pivots from "CIGL = leakage audit/control framework"
> to **Graph-DualCMI: a graph-aware source-free cross-subject EEG decoder** whose thesis is a **dual
> conditional-mutual-information objective** — encoder-side `I(Z;D|Y)` **plus** decoder-side residual
> `I(Y;D|Z)`. **Primary metric = target bAcc / worst-subject / cross-dataset generalization**, not leakage
> reduction. `docs/CIGL_45` (audit hardening) is **retained as a fallback audit track**, not discarded.
>
> **Protocol unchanged:** every experiment is reviewer-GPU-gated; non-GPU scaffolding + tests are pushed for
> review before any run; no GPU without an approved run spec; fixed-λ=0.010 stays as an honesty anchor; no
> CITA/DualPC/Tri-CMI changes; no fabrication; honest-null branches pre-committed.

---

## 1. Thesis and positioning

- **Method thesis.** A graph EEG decoder trained with a **dual-CMI objective**:
  `L = CE + λ_g R_g(Z_g;D|Y) + λ_n R_n(Z_v;D|Y) + λ_e R_e(A;D|Y) + γ·[JS(h_full(Y|Z,D), h0(Y|Z,D)) − τ]_+ + Ω_graph`.
  Encoder-side terms remove class-conditional subject *style* from the representation; the decoder-side
  **residual** term removes subject-dependent *decision rule* (concept shift) — the part IRM-style methods
  target — while the intercept-only `h0` absorbs label-prior / calibration shift so we penalize only genuine
  boundary shift.
- **Why dual (not either alone).** `I(Z;D|Y)` alone can over-erase task-useful but subject-correlated EEG
  variation (individual ERD/ERS morphology). `I(Y;D|Z)` alone is **gameable**: a high-capacity encoder can
  *write D into Z*, making the extra `D` uninformative given `Z` (small `I(Y;D|Z)`) while keeping the
  shortcut. Combining both — with **GLS reweighting** `w_i = π*(y_i)/π_{d_i}(y_i)` on the CMI side so the
  encoder- and decoder-CMI decouple (`Ĩ(Y;D)=0`) — is the safe, SOTA-oriented objective.
- **Contribution reframing.** The audit machinery (posterior-KL leakage, decoder-CMI probe, node/edge maps)
  becomes **diagnostics / mechanism evidence**, not the headline. The headline is a **stronger cross-subject
  decoder**.
- **Audit fallback (CIGL_45).** If the SOTA thesis does not pan out (no target improvement across
  architecture/objective iterations), we fall back to the CIGL_45 audit-paper track (hardened stats +
  baseline Pareto + reliance), which stands on its own.

---

## 2. Grounding corrections (verified in `project/cigl` code, 2026-07-01) — must be fixed in scaffolding

1. **CDANN in `cmi/methods/dg_penalties.py` is conditional-DANN, not CDAN.** `adv_penalty(conditional=True)`
   concatenates `[Z, one_hot(Y)]` (line ~154). Real CDAN (Long et al.) uses the multilinear `Z⊗Ŷ`
   conditioning. → **CDAN is a to-implement item**, not an existing baseline; keep the current one labeled
   `cdann` (conditional-DANN).
2. **The static DGCNN graph adapter is not in the runner registry.** `cmi/models/backbones.py::build_backbone`
   maps `"DGCNN" → gnn.DGCNNBackbone` (no `forward_graph`). The graph adapter with
   `forward_graph(x)→(logits, graph_z, node_z, edge_logits_or_none)` lives in
   `cmi/models/graph_task_backbones.py` but is unreachable from `run_loso.py`. → **register a new backbone
   name** (`DGCNNGraph`) so `--backbone DGCNNGraph --method graphcmi/graphdualpc` reaches a `forward_graph`
   backbone.
3. **`NodePosterior` does not condition on node id `v`.** `_logits(node_Z, y)` concatenates only
   `[node_Z, one_hot(Y)]` (line ~34), i.e. it implements `q(D|Z_v,Y)`, not the manuscript's `q(D|Z_v,v,Y)`.
   → **add a node embedding** `e_v` (`nn.Embedding(n_chans, node_emb_dim)`) so the head sees
   `[Z_v, e_v, one_hot(Y)]`, OR change the paper formula to `q(D|Z_v,Y)`. We take the code-fix (add `e_v`).

---

## 3. Backbone plan: FB-LGG-DualCMI (do not bet on raw dynamic GraphCMINet)

The v0.6 negative result stands: free per-sample `A(x)` overfits (train≈1.0, source≈chance; likely a
subject-fingerprint channel). So the SOTA backbone is a **strong static / constrained-dynamic hybrid**:

1. **Filterbank temporal stem** — multi-band / depthwise temporal conv + log-variance pooling over several
   temporal windows (not a single global power value).
2. **Local–global electrode graph** — local graph by 10–20 / motor partitions + a shared learnable global
   adjacency `A₀` (static first).
3. **Constrained dynamic residual** `A(x)=A₀+ΔA(x)` with `‖ΔA‖_F²`, `‖ΔA‖_1`, `‖A−Aᵀ‖_F²`,
   `‖A₀−A_phys‖_F²` — **opened only after** the static backbone beats strong task baselines.
4. **Graph + CNN gated fusion** — keep a temporal-CNN branch and a graph branch with gated fusion (SOTA
   should not be crippled by a no-bypass audit constraint); **ablate** graph-off / CNN-off /
   random-static-learned adjacency.
5. **Expose** `forward_graph(x) → (logits, graph_z, node_z, edge_logits_or_none, fused_z)`; decoder-CMI acts
   on the representation the classifier actually uses (`fused_z`/`graph_z`), not a bypass tensor.

**Task baselines to beat (target bAcc / worst-subject):** EEGNet, ShallowConvNet, Deep4Net, EEG-Conformer,
DGCNN/RGNN, TSMNet. (These become the *task* comparison; the DG-penalty methods from CIGL_45/R2 become the
*invariance* comparison.)

---

## 4. Approved first-stage scaffolding (non-GPU only — implement, test, push per step)

1. **Wire `DGCNNForwardGraphAdapter` into the unified runner** as a new backbone name **`DGCNNGraph`** (do
   not reroute the existing `DGCNN`→`DGCNNBackbone`). Runner uses `forward_graph` when the backbone exposes it.
2. **Add a `graphdualpc` method branch** in `cmi/train/trainer.py`, reusing the existing `dualpc`/`dualc`
   decoder heads (`dec_js_residual` with intercept-only `h0`), attached to the graph backbone's
   `graph_z`/`fused_z`, combined with the encoder-side graph/node/edge CMI terms. Additive; do not alter
   existing method semantics.
3. **Fix `NodePosterior` node-id conditioning** (add `nn.Embedding(n_chans, node_emb_dim)`; concat
   `[Z_v, e_v, one_hot(y)]`), keeping the shared-probe / node-map behavior.
4. **Emit dual-CMI + task metrics per fold**: encoder graph/node/edge KL; decoder residual CE; decoder
   residual JS; **target bAcc**; source bAcc; **worst-subject bAcc**; graph-ablation bAcc.
5. **Smoke tests**: graphdualpc forward/backward shapes; static DGCNN (`edge_logits=None`) with `λ_e>0`
   **fails closed**; node-posterior node-id conditioning; decoder residual loss finite & ≥0; target-label
   firewall; CPU tiny run (`max_subjects=3, epochs=2`).

All of the above are **CPU/code only** — reviewable on GitHub with zero compute.

---

## 5. First GPU gate (small pilot — request approval AFTER scaffolding; do NOT run yet)

- **Datasets:** BNCI2014_001 folds {0,1}; BNCI2015_001 folds {0,9} (**fold9 = the prior source-retention
  boundary case — must be in the pilot**).
- **Methods:** ERM-`DGCNNGraph`; encoder-only `graphcmi (λ_g=λ_n=0.010)`; decoder-only residual
  `graphdualpc (λ=0, γ>0)`; dual `graphdualpc (λ_g/λ_n + γ)`; `cdann`-`DGCNNGraph`; EEGNet **or**
  ShallowConvNet ERM (non-graph task sanity).
- **Seeds:** 0,1,2.
- **Decision gate to scale up (NOT leakage-based):** target bAcc **≥ +2 pp** vs ERM-`DGCNNGraph` (or
  worst-subject **≥ +3 pp**); source bAcc drop **≤ 2 pp**; decoder residual JS/CE **decreases**; encoder
  leakage does not explode; graph-ablation shows the graph branch contributes.
- **If no target improvement:** change architecture/objective **before** spending big compute (no 2k-hr
  hardening on a method that doesn't move target).

---

## 6. What we are explicitly NOT doing now (per PI)

- No `n_perm=1000`, no 10-seed runs, no full baseline suite, no R1 evidence-hardening megasuite **yet** —
  those are the CIGL_45 audit track, retained as fallback, not the current bottleneck.
- No "leakage-control framework" framing for the main paper; the main paper is a **cross-subject EEG graph
  decoder** with a dual-CMI objective.

---

## 7. Gate protocol (unchanged)

Non-GPU scaffolding + tests on `project/graph-dualcmi-*` branches → pushed per step → PI reviews the diff →
I submit an explicit GPU run spec (`--partition A100,V100,V100-32GB,A40`, default QOS, no `--qos`/`--time`,
fail-closed on no-CUDA, no silent seed/fold/perm reduction) → PI approves → run → `docs/CIGL_NN` results →
next gate. Honest-null outcomes reported without spin.

**Immediate next action:** implement the §4 scaffolding (non-GPU), push per step, then request the §5 GPU
pilot approval.
