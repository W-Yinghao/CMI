# Tri-CMI GNN backbone + node/edge CMI — design doc

## 0. Where this plugs in (verified against the harness)

The harness contract (read from `cmi/models/backbones.py`, `cmi/train/trainer.py`, `cmi/methods/regularizers.py`, `cmi/run_loso.py`):

- A backbone is an `nn.Module` with `forward(x:[B,C,T]) -> (logits, Z)` and a `.z_dim` int, built by `build_backbone(name, n_chans, n_times, n_classes, device)`. Emotion loaders deliver raw `float32 [B,C,T]` (SEED 62ch@200Hz, DEAP 32ch@128Hz, trial z-scored), **no DE, no channel-position metadata**.
- The 2-step trainer: **Step A** fits `DomainPosteriors` on `backbone(xb)[1].detach()` for `n_inner` steps; **Step B** does `CE + lam_t * post.reg(method, z, y)` with `lam_t` warmed up. `DomainPosteriors.reg("lpc_prior", z, y) = E KL(q_psi(D|z,y) || pi_y(D))` — exactly `I(Z;D|Y)`.
- **The existing interface only carries a single pooled Z.** `post.reg` and `post.posterior_loss` never see per-channel features or an adjacency. So node/edge CMI requires the backbone to additionally expose `node_Z [B,C,d]` and a per-sample adjacency `A [B,C,C]`, plus two new posterior heads that compose additively with the existing global term inside Step A / Step B. This is the *whole* integration surface.
- Env reality: the run env (`icml`) has torch 2.8 + braindecode 0.8 + **PyG 2.6.1 already installed**, but `backbones.py` currently imports only torch + braindecode, so the backbone is importable in leaner envs too. Adding a PyG import to the backbone path is a *new hard dependency*, and a fixed dense ≤62-node channel graph buys nothing from PyG's sparse machinery (see §6).

## 1. SOTA landscape (honest, protocol-flagged)

**Canonical EEG-emotion GNN recipe (all DE → all forbidden as *inputs*):**
- **DGCNN** (Song et al., TAFFC 2018, DOI 10.1109/TAFFC.2018.2817622): nodes = DE 62×5; a single **global learnable dense** `A=nn.Parameter` (ReLU+symmetrize+degree-norm); **ChebNet order K** (torcheeg K=2); flatten readout. Cross-subject LOSO SEED **79.95%**, SEED-IV **52.82%** (within-subject 90.4% — do **not** cite as DG). Plain PyTorch (torcheeg), `(A@X)@W`.
- **RGNN** (Zhong et al., TAFFC 2020, arXiv:1907.07835): **SGC** `Z=S^L X W` (L=2); adjacency **biologically initialized** — local `A_ij=min(1, 5/d_ij²)` (~20% sparse) + 9 negative inter-hemisphere pairs — symmetric, learnable, L1-sparse; sum-pool readout. **NodeDAT** = per-node gradient-reversal domain-adversarial training (the *direct prior art for our node-CMI*: per-node, but **adversarial and not conditioned on Y**). **EmotionDL** = soft-label KL (orthogonal). LOSO SEED **85.30%**, SEED-IV **73.84%** — the strongest small-param honest-LOSO GNN. Official code is PyG-based.

**Genuine LOSO SOTA among GNNs (still DE):** SOGNN (instance-adaptive `A=softmax(GGᵀ)`+top-k, SEED 86.81 / SEED-IV 75.27), ST-SCGNN (85.90 / 76.37), V-IAG (variational+adversarial adjacency). The recurring lesson: **a single global learned A overfits subjects; per-instance A generalizes better — but the per-instance A is itself the subject fingerprint.** FreqDGT (2025, arXiv:2506.22807) is the current DG comparator (LOSO SEED 81.1 / SEED-IV 71.9) — adversarial subject-disentanglement on rPSD (band-power, not raw).

**Raw-signal GNNs (satisfy our policy):**
- **LGGNet** (Ding et al., TNNLS 2023, arXiv:2105.02786): raw `[B,1,C,T]` → multi-scale 1-D temporal convs + `PowerLayer = log(avgpool(x²))` (a learned, end-to-end band-power surrogate computed **from raw** — not precomputed DE) + kernel attentive fusion → per-channel node embeddings; local (region) graph + global trainable adjacency; **pure PyTorch matmul, no PyG**. The cleanest raw "temporal-conv-per-channel → node embedding" template. Only ever validated **within-subject** → its cross-subject behavior is open (our opening). License CBCR non-commercial → re-implement the (trivial) layers.
- GCNs-Net (arXiv:2006.08924): graph from |Pearson| of raw signals; raw-signal precedent. Tang/Stanford DCRNN (ICLR 2022): FFT-log-amplitude nodes + dist-graph + dynamic corr-graph + diffusion conv, cross-patient TUSZ — best raw cross-patient template; its dist-vs-corr split maps onto our prior-vs-learned adjacency.

**Graph-IB / interpretability foundations for the CMI terms:**
- **GIB** (Wu et al., NeurIPS 2020): per-node variational bottleneck and an explicit **structure-MI (AIB) vs feature-MI (XIB)** split — the exact scaffold for our node-vs-edge decomposition.
- **BrainIB** (Zheng et al., TNNLS 2024, arXiv:2205.03612): **edge-assignment** subgraph via `p_ij=σ(X_iWX_jᵀ)`+Gumbel, MINE-free **matrix-Rényi** MI, leave-one-site-out (cross-dataset analogue) — precedent for treating the adjacency as the selected object.
- **GNNExplainer/PGExplainer**: node/edge importance **defined as mutual information**; PGExplainer's shared edge-MLP (`σ` over node-pair embeddings, Gumbel relaxation) is the amortized per-edge head we reuse — flipped from *maximize* MI(edge;Y) to *minimize* `I(edge;D|Y)`.

**The gap (our novelty, stated honestly):** no EEG-GNN (RGNN, DGCNN, SOGNN, V-IAG, FreqDGT) and no general graph method (GIB, BrainIB, IS-GIB, PrunE) imposes a **label-conditional** MI penalty on a **named domain D** at **both per-node features Z_v AND the learned adjacency A** as a non-adversarial plug-in. NodeDAT (node, adversarial, unconditional) and BrainIB/PrunE (edge, but task/spurious not domain-conditional) bracket us cleanly.

## 2. The chosen design and WHY: GraphCMINet

**One plain-PyTorch backbone** = (LGGNet-style raw temporal-conv node encoder) → (RGNN-style biologically-initializable learnable adjacency) → (SGC propagation, L=2) → readout, exposing `(logits, graph_Z)` for the existing harness **plus** `node_Z [B,C,d]` and a per-sample adjacency `A [B,C,C]` for the new terms. Rationale:

1. **Raw node encoder (LGGNet PowerLayer path)** is the only published recipe that turns raw per-channel series into node embeddings end-to-end without DE, satisfying the hard constraint while keeping the leakage story on raw.
2. **SGC over ChebNet-K** (RGNN's choice): linear `S^L X W`, ~1e4 params, closed-form smoothing — least overfit on small LOSO data, and `S^L X` composes cleanly with a CMI penalty (no nonlinearity to fight). ChebNet K=2 kept as a one-flag ablation.
3. **Per-channel nodes (C=62/32), not region pooling, by default.** Region pooling (LGGNet/PGCN) is great for params/interpretability, but our headline diagnostic is *"which channels leak subject identity"* (`Σ_v I(Z_v;D|Y)`), which is most compelling **per electrode** and is the direct, non-adversarial generalization of NodeDAT (per node). Region pooling is offered as a `region_list` ablation for the param-starved clinical montages.
4. **A *per-sample* adjacency** `A(x)` (not DGCNN's single shared static A). Edge-CMI needs something subject-*varying* to act on; the fingerprinting literature shows the **learned/inferred adjacency is the *strongest* subject discriminator** (superior to fixed PLV/correlation), so `A(x)` is exactly the object to neutralize with `I(A;D|Y)`. A shared static A would have nothing for edge-CMI to bite.
5. **Plain PyTorch, no PyG** (§6).

## 3. How it fits cross-subject DG + our CMI

- Domain D = subject (LOSO) / subject×session — already what the runner passes as `dtr`. The backbone is dataset-agnostic (only C changes), so it drops onto SEED/SEED-IV/DEAP, 2a/2b/Lee2019, ADFTD/TUAB unchanged.
- The existing **global** `I(Z;D|Y)` keeps acting on `graph_Z` exactly as today (zero change to `DomainPosteriors.reg`). The new **node** and **edge** terms are additional KL terms fit by the *same* 2-step alternation (Step A fits their posteriors on detached features; Step B adds `λ_node·Σ_v KL + λ_edge·KL` to the loss). So the method is `CE + λ·I(graph_Z;D|Y) + λ_node·Σ_v I(Z_v;D|Y) + λ_edge·I(A;D|Y)`, a strict superset of the current `lpc_prior`.
- This is *why GNN + why CMI here*: EEG subject leakage lives in spatial covariance/connectivity; a GNN with a learned `A(x)` models that object explicitly, so the invariance penalty can be placed **on the connectivity structure itself**, not just a flat embedding — impossible with EEGNet/Deep4Net.

## 4. Protocol hygiene (for the paper)
Report strict LOSO + cross-dataset. Honest GNN baselines to beat: **RGNN LOSO SEED 85.30 / SEED-IV 73.84**, FreqDGT 81.1 / 71.9. Never quote DGCNN 90.4 or LGGNet DEAP numbers as cross-subject. Provide a fair *raw-input* baseline (our temporal stem + plain readout, no graph) so the GNN's contribution is isolated. Position node-CMI as the non-adversarial, label-conditional generalization of NodeDAT; edge-CMI as genuinely new.

## 5. Citations
DGCNN (TAFFC 2018); RGNN (TAFFC 2020, arXiv:1907.07835); SGC (Wu et al. ICML 2019, arXiv:1902.07153); LGGNet (TNNLS 2023, arXiv:2105.02786); SOGNN (PMC8221183); V-IAG (TAFFC 2021); ST-SCGNN (JBHI 2023); FreqDGT (arXiv:2506.22807); GCNs-Net (arXiv:2006.08924); Tang DCRNN (ICLR 2022, arXiv:2104.08336); GIB (NeurIPS 2020, arXiv:2010.12811); BrainIB (TNNLS 2024, arXiv:2205.03612); GNNExplainer (NeurIPS 2019); PGExplainer (NeurIPS 2020); BrainPrint (Pattern Recognition 2020); EEG-graph-inference fingerprinting (EUSIPCO 2023, IEEE 10289864).

## 6. PyG verdict — NOT worth it
The harness env has PyG 2.6.1, but `backbones.py` imports only torch+braindecode and a fixed dense ≤62-node channel graph is just batched `bmm`: SGC is `(D^{-1/2}AD^{-1/2})^L X W`, ChebNet-K is a 3-line recurrence on `[B,C,C]`. PyG's `edge_index`/MessagePassing/torch-scatter exist for large, sparse, variable-topology graphs; here they add a version-brittle hard dep (RGNN even pins PyG 1.2.1) for zero speed/clarity benefit, and a dense `node_Z [B,C,d]` + `A [B,C,C]` is the *cleanest* substrate for indexing the node/edge CMI heads. Keep the backbone plain-torch; reserve PyG only if we later need GAT or sparse clinical montages.


---
## BACKBONE SPEC

CONCRETE SPEC — `GraphCMINet` (plain torch, file `cmi/models/gnn.py`), wired into `build_backbone(name="GraphCMI", ...)`.

INPUT: x [B, C, T] float32 (C=62 SEED, 32 DEAP/2a-ish, 19 TUAB, ... ; dataset-agnostic). All shapes below for SEED (C=62, T=800 @200Hz/4s; works for any C,T).

--- (1) PER-CHANNEL RAW TEMPORAL ENCODER (LGGNet PowerLayer path, no DE) ---
Treat x as [B,1,C,T]. Shared across channels (depthwise over the C axis), three parallel temporal branches:
  for k in kernels=[round(0.5*fs), round(0.25*fs), round(0.125*fs)]:
    Conv2d(1, num_T=16, kernel=(1,k), padding=(0,k//2))      # temporal only, shared over channels
    -> PowerLayer: z = log(AvgPool2d((1,P), stride=(1,P))(conv**2) + 1e-6)   # learned band-power surrogate FROM RAW
  concat 3 branches on feature axis -> BN2d -> Conv2d(3*num_T, F=16, (1,1)) -> ELU
  -> adaptive-avg-pool the time axis to 1  =>  node_feat X0 [B, F=16, C, 1] -> reshape [B, C, F]
  node_Z_raw = X0  (this is the pre-graph node embedding; d_in = F = 16)
Params: ~ (1*16*k for 3 k's) + BN + (48*16 1x1) ≈ 3–4k. Honors raw-only: PowerLayer is differentiable log-avg-pool-of-square, NOT precomputed 62x5 DE.

--- (2) PER-SAMPLE LEARNABLE ADJACENCY A(x) [B,C,C] ---
Two additive parts, then symmetrize+normalize:
  (a) PRIOR (optional, data-agnostic default = off; on when montage available):
      A_prior[C,C] buffer = RGNN init: local min(1, 5/d_ij^2) from electrode 3D coords (+ 9 negative inter-hemisphere pairs for SEED). If no montage metadata (the emotion loaders carry none), fall back to A_prior = 0 and rely on a free symmetric learnable base B = nn.Parameter(C,C) (Xavier), used as B+Bᵀ.  [montage table hardcoded per dataset in gnn.py; SEED 62-ch ESI order known.]
  (b) DATA-DEPENDENT (the fingerprint, SOGNN/PGExplainer-style amortized edge head):
      g = node_Z_raw @ W_e            # W_e: Linear(F, e=16), node-pair embedding
      S = g @ g.transpose(1,2)        # [B,C,C] similarity (per-sample)
  A_raw = A_prior(broadcast) + (B+Bᵀ)(broadcast) + S
  A = ReLU(A_raw); A = 0.5*(A + A.transpose(1,2))          # nonneg + symmetric
  edge_logits = A_raw (pre-ReLU, used for the edge-CMI head & sparsity)   # [B,C,C]
  S_hat = D^{-1/2} (A+I) D^{-1/2}     # sym-normalized w/ self-loops, deg from A.sum(-1)
Params: W_e (F*e≈256) + B (C*C, ~3.8k for 62). Optional L1 sparsity on A (RGNN), weight 1e-4.

--- (3) GRAPH CONV = SGC (RGNN), L=2 (ChebNet-K=2 as a flag) ---
  H = S_hat @ S_hat @ node_Z_raw      # closed-form 2-hop smoothing  [B,C,F]
  node_Z = ELU(H @ W_g)               # W_g: Linear(F, d=32)   -> node_Z [B,C,d=32]
(ChebNet variant: T0=X,T1=Lx,Tk=2Lx·Tk-1−Tk-2, sum_k Tk@W_k, then ELU.)
Params: W_g (F*d≈512).

--- (4) READOUT -> graph_Z, logits ---
  graph_Z = node_Z.mean(1)            # mean-pool over C nodes -> [B, d=32]   (sum/attn = flags)
  logits  = Linear(d=32, n_classes)(graph_Z)
self.z_dim = d = 32   (graph_Z dim, what the existing harness consumes)

--- FORWARD CONTRACT ---
def forward(self, x):                 # x:[B,C,T]
    ... -> return logits, graph_Z     # 2-tuple = backward-compatible with HookedBackbone callers
def forward_graph(self, x):           # NEW, used by the GNN-aware trainer path
    ... -> return logits, graph_Z, node_Z, edge_logits
    # graph_Z [B,32]; node_Z [B,C,32]; edge_logits [B,C,C]
The 2-tuple `forward` keeps `predict()/embed()/leakage_probe()` working unchanged.

--- SHAPES (SEED C=62) ---
x[B,62,800] -> node_Z_raw[B,62,16] -> A[B,62,62] -> node_Z[B,62,32] -> graph_Z[B,32] -> logits[B,3].

--- #PARAMS (C=62, n_cls=3) ---
temporal ~3.5k + W_e 0.26k + B 3.84k + W_g 0.5k + head 0.1k ≈ ~8–9k trainable. (Tiny → good for small-data LOSO; far below EEGNet.) DEAP C=32 ≈ ~5k.

--- DEPS ---
Plain torch only (Conv2d/Linear/ELU/BN + bmm). NO PyG, NO braindecode. Importable in any env with torch. ~180–220 LOC.

--- INTEGRATION w/ build_backbone ---
build_backbone("GraphCMI", n_chans, n_times, n_classes, device) -> GraphCMINet(...).to(device); add "GraphCMI" to run_loso --backbone choices. Probe z_dim via a dummy forward like HookedBackbone.

---
## NODE/EDGE CMI

PRINCIPLE: keep the existing global term `I(graph_Z;D|Y)=E KL(q_ψ(D|graph_Z,Y)‖π_y(D))` untouched, and ADD two terms estimated by the SAME variational-posterior-KL machinery and the SAME 2-step alternation. Total Step-B regularizer:
  L_reg = λ·I(graph_Z;D|Y)  +  λ_node·Σ_v I(Z_v;D|Y)  +  λ_edge·I(A;D|Y).

=== NODE-LEVEL  Σ_v I(Z_v; D | Y)  (which channels leak subject identity) ===
DEFINITION: for each node v (channel), I(Z_v;D|Y) ≈ E_i KL( q_node(D | Z_{v,i}, y_i) ‖ π_y(D) ), summed/averaged over the C nodes. π_y(D)=p(D|Y=y) is the SAME label-prior already computed by `empirical_priors` (so node-CMI inherits the imbalance correction for free).

q_ψ PER NODE — weight-shared trunk (PGExplainer-style amortization, NOT C separate MLPs):
  class NodePosterior(nn.Module):   # operates on node_Z [B,C,d]
    body = _mlp(d + n_cls, n_dom)    # ONE shared MLP applied to every node (broadcast over C)
  forward(node_Z, y): y_oh broadcast to [B,C,n_cls]; logits = body(cat([node_Z, y_oh], -1)) -> [B,C,n_dom]
This shares parameters across channels (stable on small data, ~the same #params as the existing global head) yet still yields a PER-CHANNEL posterior, hence a per-channel leakage value.

STEP A (fit, detached): la_node = mean over B,C of CE( logits[B,C,n_dom], d.broadcast[B,C] ), added to the existing `post.posterior_loss` total, same opt_post, same n_inner.
STEP B (penalize, grad through node_Z): 
  per_node_kl[B,C] = kl_to_prior_rowwise( body(cat[node_Z, y_oh]) , log_pi_y[y] )   # reuse kl_to_prior, vectorized over C
  L_node = per_node_kl.mean()        # = (1/C) Σ_v Ê KL  -> the Σ_v I(Z_v;D|Y) estimate
POOLING / WEIGHTING: default uniform over channels (mean). Optionally learn λ_v or anneal toward high-leakage channels; but uniform is the clean default and already produces the diagnostic. 
DIAGNOSTIC (free): per_node_kl averaged over the eval split = a length-C "subject-leakage map" — the headline figure (expect frontal/temporal + the gamma-carrying channels to dominate, per BrainPrint). This is exactly what NodeDAT cannot produce (its discriminator is marginal/symmetric). Reuse `leakage_probe` pattern with a frozen per-node head for the honest residual map.
WHY IT BEATS NodeDAT: conditional on Y (does not erase class-correlated-but-subject-varying signal that marginal node-alignment kills) + non-adversarial (no GRL β schedule, stable) + pure DG (never touches target).

=== EDGE / ADJACENCY-LEVEL  I(A; D | Y)  (the learned adjacency is a subject fingerprint) ===
DEFINITION: I(A;D|Y) ≈ E_i KL( q_edge(D | a_i, y_i) ‖ π_y(D) ), where a_i is a compact summary of sample i's adjacency. Two interchangeable representations of `a` (start with the cheap one):
  (R1, default) edge-summary vector: a_i = vec(triu(edge_logits_i, k=1)) compressed by a small Linear to e_a=64 dims (the upper-triangular C(C-1)/2 edges, symmetric A so triu suffices). For C=62 that's 1891 edges -> Linear(1891,64).
  (R2, ablation) per-edge amortized (PGExplainer/BrainIB): treat each edge (u,v) as a unit with feature [node_Z_u ⊕ node_Z_v]; a shared edge-MLP gives per-edge logits; aggregate I over edges = Σ_{u<v} I(e_uv;D|Y). Heavier; use to localize WHICH edges leak.
q_ψ FOR EDGES:
  class EdgePosterior(nn.Module):
    enc = Linear(C*(C-1)//2, e_a=64)   # (R1) compress triu(edge_logits)
    body = _mlp(e_a + n_cls, n_dom)
STEP A (fit, detached edge_logits): la_edge = CE( body(cat[enc(a.detach()), y_oh]), d ), added to opt_post total.
STEP B (penalize, grad through edge_logits→A→W_e, B): 
  L_edge = kl_to_prior( body(cat[enc(a), y_oh]), log_pi_y[y] )    # = Ê KL(q_edge(D|A,Y)‖π_y) ≈ I(A;D|Y)
This makes the *connectivity structure* conditionally subject-invariant: an edge pattern that predicts the subject but not the label is pushed out of A. Pair with RGNN's L1 sparsity + a small BrainIB-style connectivity/consistency term so A stays stable while subject-edges are pruned. Genuinely novel: no EEG-GNN regularizes A's domain-information.

=== HOW THEY COMPOSE WITH THE EXISTING 2-STEP TRAINER ===
A `DomainPosteriors`-like container (extend it, or a sibling `GraphPosteriors`) now holds {q_dzy (global, unchanged), node body, edge enc+body} + the SAME registered log priors (log_pi_y, etc.).
- STEP A: opt_post minimizes  posterior_loss(graph_Z,y,d) [unchanged] + la_node(node_Z,y,d) + la_edge(edge_logits,y,d), all on DETACHED features, for n_inner steps. One optimizer, one backward.
- STEP B: loss = CE + lam_t*( reg("lpc_prior", graph_Z, y) + λ_node*L_node(node_Z,y) + λ_edge*L_edge(edge_logits,y) ) [+ L1 sparsity]. Same warmup lam_t, same sampler (classbal preserves p(D|Y) -> empirical π_y stays the valid KL target for ALL three terms).
KNOBS: a single config string e.g. `graphcmi:λ:λ_node:λ_edge` (parsed like the existing `lpc_supcon:lam:gamma`). Setting λ_node=λ_edge=0 recovers vanilla global LPC-CMI on a GNN backbone -> clean ablation ladder (ERM → +global → +node → +edge). Reuse `kl_to_prior` and `empirical_priors` verbatim; the only new code is the two posterior heads + the broadcast-over-C KL.

---
## INTEGRATION PLAN

1. 1. Add `cmi/models/gnn.py` (plain torch, ~200 LOC): GraphCMINet per the backbone_spec. Implement (a) LGGNet-style temporal+PowerLayer node encoder, (b) per-sample adjacency A(x)=ReLU(sym(prior + (B+Bᵀ) + g gᵀ)) exposing pre-ReLU edge_logits, (c) SGC L=2 (ChebNet-K=2 behind a flag), (d) mean readout. Expose BOTH `forward(x)->(logits, graph_Z)` (backward-compatible) and `forward_graph(x)->(logits, graph_Z, node_Z, edge_logits)`. Set self.z_dim. NO PyG, NO braindecode imports.
2. 2. Wire into `cmi/models/backbones.py::build_backbone`: add branch `if name=='GraphCMI': return GraphCMINet(n_chans,n_times,n_classes,...).to(device)`; probe z_dim with a dummy forward like HookedBackbone. Add 'GraphCMI' to `run_loso.py` --backbone choices and the cross-dataset runner.
3. 3. Smoke-test the backbone alone in env `icml` (/home/infres/yinwang/anaconda3/envs/icml/bin/python): build at SEED (C=62,T=800) and DEAP (C=32) shapes, assert forward returns finite (logits,graph_Z) and forward_graph returns node_Z[B,C,32], edge_logits[B,C,C]; confirm it trains 1 epoch with method='erm' through the EXISTING trainer (since erm ignores node/edge, this verifies the (logits,Z) contract).
4. 4. Extend the CMI container: add `NodePosterior` (shared trunk) and `EdgePosterior` (R1 triu-compress) to `cmi/methods/regularizers.py` (or a new `graph_regularizers.py`), reusing `_mlp`, `kl_to_prior`, `empirical_priors`. Add a vectorized row-wise KL helper for the [B,C] node case. New container exposes `posterior_loss_graph(graph_Z,node_Z,edge_logits,y,d)` (Step A) and `reg_graph(graph_Z,node_Z,edge_logits,y)` (Step B) returning the three additive terms.
5. 5. Add a GNN-aware branch in `cmi/train/trainer.py`: when the backbone has `forward_graph` AND method is the new 'graphcmi' family, Step A fits all three posteriors on detached (graph_Z,node_Z,edge_logits); Step B = CE + lam_t*(global + λ_node*L_node + λ_edge*L_edge) [+ 1e-4*||A||_1]. Parse config `graphcmi:λ:λ_node:λ_edge` in run_loso like the existing lpc_supcon parsing. Register the method in ALL_METHODS. Keep all existing methods/paths byte-identical (the new branch is gated on backbone capability + method name).
6. 6. Add a node/edge leakage diagnostic to `cmi/eval/metrics.py`: `node_leakage_map` (frozen backbone, per-channel residual KL -> length-C vector) and `edge_leakage` (residual I(A;D|Y)); log into the per-target records so run_loso emits a subject-leakage channel map per fold. This is the headline diagnostic figure and costs ~nothing.
7. 7. EXPERIMENTS — primary GNN testbed first (connectivity shift is severe there): SEED then SEED-IV LOSO, ablation ladder per (dataset): `erm:0`, global-only `graphcmi:λ:0:0`, +node `graphcmi:λ:λn:0`, +edge `graphcmi:λ:λn:λe`. Compare against EEGNet/Deep4Net+lpc_prior (existing) and a fair raw-graph baseline (graphcmi λ=0). Sweep λ,λ_node,λ_edge ∈ {0.1,0.3,1.0} small grid; report per-target balanced acc + worst-subject + leakage_kl + node/edge leakage maps. Targets to beat: RGNN LOSO SEED 85.30 / SEED-IV 73.84 (flag DGCNN 90.4 / LGGNet DEAP as within-subject, not DG).
8. 8. Then DEAP (valence/arousal/quadrant) LOSO same ladder; then transfer the SAME backbone (only C changes) to MI 2a/2b/Lee2019 and clinical ADFTD/TUAB to show generality. Optionally region-pool ablation (LGGNet region_list) for the low-channel clinical montages.
9. 9. Cross-dataset arm (cmi/run_cross_dataset.py): SEED->SEED-IV and montage-shared MI transfers; report whether node/edge CMI improves cross-dataset over global-only. (Optional later: graph-MAE SSL pretrain for cross-dataset — keep out of the core.)
10. 10. Honesty + ablations for the paper: (a) NodeDAT re-implemented (~10 LOC GRL + per-node head) as the adversarial comparator to Σ_v I(Z_v;D|Y); (b) chsic (existing) as the kernel CMI foil; (c) matrix-Rényi MI as an estimator-robustness check that the leakage story is estimator-independent; (d) PyG-free claim verified by importing the backbone in a torch-only env.

---
## RECOMMENDATION

BUILD FIRST: `GraphCMINet` = LGGNet-style raw temporal+PowerLayer per-channel node encoder → per-sample learnable adjacency A(x) (RGNN-initializable when a montage is available, else a free symmetric learnable base + a SOGNN-style data-dependent term) → RGNN SGC L=2 propagation → mean readout, in PLAIN PyTorch (~200 LOC, ~8–9k params for SEED). It returns `(logits, graph_Z)` for full backward-compat with the existing harness, and a `forward_graph` that also exposes `node_Z [B,C,d]` and pre-ReLU `edge_logits [B,C,C]`. Implement the per-channel nodes (not region pooling) so the node-CMI diagnostic is per electrode.

WHY THIS ONE: it is the unique published recipe that (1) is raw-signal (no DE, honoring policy), (2) is tiny and SGC-linear (best for small-data LOSO), and (3) natively exposes the THREE granularities our CMI needs — global graph_Z (existing term, zero change), per-node Z_v (node-CMI, the non-adversarial label-conditional generalization of RGNN's NodeDAT), and a per-sample A (edge-CMI, genuinely novel — no EEG-GNN regularizes the adjacency's domain-information, yet the fingerprinting literature shows the learned adjacency is the *strongest* subject discriminator). A single per-sample A is essential: a DGCNN-style shared static A would give edge-CMI nothing subject-varying to act on.

PyG: NOT worth it. It's installed in the run env (icml) but the backbone path imports only torch+braindecode; a fixed dense ≤62-node channel graph is just `bmm` (SGC = `(D^{-1/2}AD^{-1/2})^L X W`), so PyG adds a version-brittle hard dep (RGNN pins PyG 1.2.1) for zero benefit and would break import in leaner envs. Dense `node_Z [B,C,d]` + `A [B,C,C]` is also the cleanest substrate for indexing the node/edge CMI heads. Stay plain-torch; reserve PyG only for future GAT/sparse-montage work.

PAYOFF vs RISK (honest):
- HIGH-confidence payoff: a clean ablation ladder (ERM → +global CMI → +node CMI → +edge CMI) on a raw-signal GNN, plus a per-channel subject-leakage MAP and a per-edge fingerprint diagnostic that NodeDAT/adversarial methods cannot produce. That diagnostic story alone (which channels/edges leak subject identity, conditional on label) is publishable and differentiates us from every adversarial DG-GNN.
- MEDIUM risk on raw accuracy: LGGNet was never validated cross-subject and RGNN/SOGNN's strong LOSO numbers ride on DE features we forbid; our raw stem may underperform DE-based RGNN's 85.30 on SEED before the CMI gains. Mitigation: this is *expected and honest* — we report a fair raw-graph baseline and show the CMI delta, not a raw-vs-DE apples-to-oranges win. The cross-subject GENERALIZATION-GAP reduction (worst-subject, leakage_kl) is the metric we should headline, not absolute SEED accuracy.
- LOW-medium risk on edge-CMI stability: a per-sample A regularized by I(A;D|Y) can collapse/over-sparsify. Mitigation: RGNN L1 + a BrainIB-style connectivity term, warmup on λ_edge, and start with the cheap R1 triu-summary edge head before the heavier per-edge R2.
- Effort: ~1–2 days to a runnable backbone + node CMI (node-CMI is the lowest-risk, highest-novelty-per-effort term — ship it first, edge-CMI second). Validate the (logits,Z) contract with method='erm' through the UNMODIFIED trainer before touching the trainer.

SEQUENCING: backbone + global-CMI passthrough (day 1) → node-CMI + leakage map (day 1–2) → edge-CMI (day 2–3) → SEED/SEED-IV ablation ladder → DEAP → transfer to MI/clinical.