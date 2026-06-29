# CIGL Method Specification

## 1. Notation

- \(X \in \mathbb{R}^{C \times T}\): one EEG trial with \(C\) channels and \(T\) time samples.
- \(Y \in \{1,\ldots,K\}\): task label.
- \(D \in \{1,\ldots,M\}\): domain, normally subject for LOSO or subject-session for cross-session.
- \(Z_v\): node/electrode representation for channel \(v\).
- \(Z_g\): pooled graph representation.
- \(A(X)\): per-sample learned adjacency / connectivity matrix.
- \(\pi_y(D)=p(D\mid Y=y)\): empirical label-conditioned domain prior, Laplace-smoothed.

The core objective is label-conditional domain-leakage suppression:

\[
I(Z_g;D\mid Y),\quad \frac{1}{C}\sum_v I(Z_v;D\mid Y),\quad I(A;D\mid Y).
\]

---

## 2. Backbone

The default backbone is `GraphCMINet`:

\[
X \rightarrow Z^{raw}_{1:C} \rightarrow A(X) \rightarrow Z_{1:C} \rightarrow Z_g \rightarrow \hat{Y}.
\]

### 2.1 Raw temporal node encoder

Each channel is encoded by a shared raw temporal stem:

\[
X_v \mapsto Z_v^{raw}.
\]

The default design uses multi-scale temporal convolutions followed by a differentiable power layer:

\[
\mathrm{PowerLayer}(h)=\log(\mathrm{AvgPool}(h^2)+\epsilon).
\]

This is allowed because it is computed end-to-end from raw EEG, not precomputed DE or handcrafted band-power input.

### 2.2 Per-sample adjacency

The learned adjacency has a static learnable base and a data-dependent similarity term:

\[
S_i = g(Z^{raw}_i)g(Z^{raw}_i)^\top,
\]

\[
A_i^{raw} = B+B^\top+S_i,
\]

\[
A_i = \frac{1}{2}\left[\mathrm{ReLU}(A_i^{raw})+\mathrm{ReLU}(A_i^{raw})^\top\right].
\]

`edge_logits = A_i^{raw}` is the object used by the edge-CMI posterior. The pre-ReLU version is preferred for the edge head because it retains signed evidence before truncation.

### 2.3 Graph propagation

Default graph propagation is lightweight SGC:

\[
\hat{S}=D^{-1/2}(A+I)D^{-1/2},
\]

\[
H = \hat{S}^L Z^{raw},
\]

\[
Z_v = W H_v.
\]

Default \(L=2\). ChebNet is allowed only as an ablation.

### 2.4 Readout

\[
Z_g = \frac{1}{C}\sum_{v=1}^{C} Z_v,
\]

\[
\hat{Y}=h(Z_g).
\]

The backbone must expose two contracts:

```python
forward(x) -> (logits, graph_Z)
forward_graph(x) -> (logits, graph_Z, node_Z, edge_logits)
```

The first contract preserves compatibility with the existing harness. The second contract is used only by CIGL training and diagnostics.

---

## 3. Posterior-KL conditional leakage proxy

For any representation object \(R\), define a posterior:

\[
q_\psi(D\mid R,Y).
\]

The trainable leakage proxy is:

\[
\widehat{I}(R;D\mid Y)
=
\mathbb{E}_i\,
\mathrm{KL}\left(q_\psi(D\mid R_i,Y_i)\;\|\;\pi_{Y_i}(D)\right).
\]

This must be described as a **posterior-KL plug-in leakage proxy**, not an unbiased CMI estimator.

---

## 4. Three CIGL regularizers

### 4.1 Graph-level term

\[
\mathcal{R}_g
=
\mathbb{E}_i\,
\mathrm{KL}\left(q_g(D\mid Z_{g,i},Y_i)\;\|\;\pi_{Y_i}(D)\right).
\]

This reuses the existing `DomainPosteriors` mechanism.

### 4.2 Node-level term

Use a single shared posterior trunk across channels:

\[
q_n(D\mid Z_{v,i},Y_i).
\]

Default node regularizer:

\[
\mathcal{R}_n
=
\frac{1}{C}\sum_{v=1}^{C}
\mathbb{E}_i\,
\mathrm{KL}\left(q_n(D\mid Z_{v,i},Y_i)\;\|\;\pi_{Y_i}(D)\right).
\]

The shared trunk avoids overfitting \(C\) separate domain heads. It also yields a per-channel leakage map:

\[
L_v = \mathbb{E}_i\,
\mathrm{KL}\left(q_n(D\mid Z_{v,i},Y_i)\;\|\;\pi_{Y_i}(D)\right).
\]

### 4.3 Edge-level term

Default training uses a compact summary of the upper triangle of `edge_logits`:

\[
a_i = \mathrm{compress}(\mathrm{vec}(\mathrm{triu}(A^{raw}_i))).
\]

Then:

\[
\mathcal{R}_e
=
\mathbb{E}_i\,
\mathrm{KL}\left(q_e(D\mid a_i,Y_i)\;\|\;\pi_{Y_i}(D)\right).
\]

Per-edge maps are diagnostic-only in the first version. Do not train thousands of per-edge heads in the first implementation.

---

## 5. Total objective

The default training objective is:

\[
\mathcal{L}
=
\mathcal{L}_{CE}
+
\lambda_g \mathcal{R}_g
+
\lambda_n \mathcal{R}_n
+
\lambda_e \mathcal{R}_e
+
\eta \lVert A \rVert_1.
\]

Default first sweep:

- \(\lambda_g \in \{0,0.1,0.3,1.0\}\)
- \(\lambda_n \in \{0,0.1,0.3,1.0\}\)
- \(\lambda_e \in \{0,0.03,0.1,0.3\}\)
- \(\eta=10^{-4}\), only if edge collapse or dense trivial graphs appear.

The ablation ladder is mandatory:

1. GraphCMI-ERM: \(\lambda_g=\lambda_n=\lambda_e=0\)
2. Global only: \(\lambda_g>0,\lambda_n=\lambda_e=0\)
3. Node only or graph+node
4. Edge only or graph+edge
5. Full CIGL

---

## 6. Alternating optimization

Training uses the existing two-step scheme.

### Step A: posterior fitting

Freeze/detach backbone features:

```python
with torch.no_grad():
    logits, graph_Z, node_Z, edge_logits = backbone.forward_graph(x)
```

Update posterior heads by domain classification:

\[
\min_\psi
\mathcal{L}_{post}
=
CE(q_g(D\mid Z_g,Y),D)
+CE(q_n(D\mid Z_v,Y),D)
+CE(q_e(D\mid A,Y),D).
\]

### Step B: backbone update

Update the backbone with task CE and KL-to-prior penalties:

\[
\min_{\theta,h}
CE(h(f_\theta(X)),Y)
+
\lambda_g\mathcal{R}_g
+
\lambda_n\mathcal{R}_n
+
\lambda_e\mathcal{R}_e.
\]

Warm up all three CMI terms. Edge warmup should be at least as conservative as graph/node warmup.

---

## 7. Configuration grammar

Standardize method config. Recommended internal API:

```text
graphcmi:<lambda_g>:<lambda_n>:<lambda_e>
```

Examples:

```text
graphcmi:0:0:0        # GraphCMI-ERM
graphcmi:0.3:0:0      # global only
graphcmi:0.3:0.3:0    # graph + node
graphcmi:0.3:0.3:0.1  # full CIGL
```

Do not overload `gamma` silently for node-CMI in user-facing scripts. If the trainer currently uses `gamma` for node-CMI, the CLI and logs must rename/report it as `lambda_node`.

---

## 8. Diagnostics

Required diagnostics:

1. `graph_leakage_kl`: held-out \(\widehat I(Z_g;D\mid Y)\)
2. `node_leakage_mean`: average held-out \(\widehat I(Z_v;D\mid Y)\)
3. `node_leakage_map`: length-\(C\) vector
4. `edge_leakage_kl`: held-out \(\widehat I(A;D\mid Y)\)
5. `edge_subject_acc`: ability of `edge_logits` to predict subject/domain conditional on label
6. `label_separability`: linear probe or task head metric
7. `worst_subject_bacc`
8. `accuracy_leakage_pareto_rank`

Diagnostics must be computed on held-out source folds for model selection and source validation. The strict DG main table cannot use target labels or target covariates.

---

## 9. Implementation red lines

- No PyG dependency for the first paper version.
- No target-domain adaptation in CIGL main results.
- No precomputed DE feature input in the main raw-input table.
- No hidden mixing of CITA/TTA and strict DG metrics.
- No unlogged loss terms.
- No “method works” claim without leakage and task metrics.
