"""CIGL Phase 2 — probe-only audit of label-conditional domain leakage in learned EEG graph objects.

Answers ONE scientific question, with no target information and no regularization: do the graph
objects produced by a GraphCMINet-ERM encoder carry label-conditional domain leakage?

  graph: I(Z_g;D|Y)            node: (1/C) Σ_v I(Z_v;D|Y)            edge: I(A(X);D|Y)

For each object we fit a held-out conditional domain probe q(D | object, Y) on frozen detached
features and report the posterior-KL leakage proxy  E KL( q(D|·,Y) ‖ π_y(D) )  (NOT an unbiased
CMI; see cmi/methods/regularizers.py). Significance is judged against a WITHIN-LABEL DOMAIN
PERMUTATION null that **retrains the probe** each permutation.

Why retrain: the estimate E KL(q(D|O,Y) ‖ π_y(D)) does NOT depend on the observed D at evaluation
time — it is a function of (O, Y) through the probe and of Y through π_y. Shuffling the validation
domain labels therefore leaves the KL essentially unchanged and yields a FALSE null. The only valid
null breaks the O→D association the probe can learn, i.e. permute D within each label group ON THE
PROBE TRAINING SPLIT and refit, leaving the held-out (validation) D untouched. Restricting the
permutation to the training indices preserves the TRAINING-split π_y(D)=p(D|Y) exactly (only the
per-sample O→D pairing is destroyed), so the prior reference is identical under null and observed
and only the probe's usable signal changes.

This module is diagnostic-only. The per-edge map is a non-neural binned plug-in CMI for
interpretation; it is NOT a training regularizer (CIGL trains a compact edge SUMMARY head, never
per-edge heads). Everything is CPU-friendly.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- small helpers
def _np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _feat_tensor(x, device):
    """Frozen, detached float32 feature tensor on `device`."""
    if torch.is_tensor(x):
        return x.detach().to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def compute_label_domain_prior(y, d, n_classes, n_domains, smoothing=1e-3):
    """π_y(D) = p(D | Y), Laplace/additive-smoothed, as a Tensor[n_classes, n_domains] whose rows sum to 1.

    Missing (label, domain) cells are covered by `smoothing` so log π_y is always finite and every
    row is a valid distribution even for labels absent from the split.
    """
    y = _np(y).astype(np.int64)
    d = _np(d).astype(np.int64)
    counts = np.zeros((int(n_classes), int(n_domains)), dtype=np.float64)
    for yi, di in zip(y, d):
        counts[yi, di] += 1.0
    counts += float(smoothing)
    pi = counts / counts.sum(axis=1, keepdims=True)
    return torch.tensor(pi, dtype=torch.float32)


def conditional_kl_to_prior(probs_d_given_obj_y, y, pi_y):
    """Per-sample KL( q(D|O_i,Y_i) ‖ π_{Y_i}(D) ) for predicted domain posteriors.

    probs_d_given_obj_y: [N, n_dom] (probabilities, NOT logits). y: [N]. pi_y: [n_cls, n_dom].
    Returns a finite, non-negative Tensor[N].
    """
    probs = probs_d_given_obj_y if torch.is_tensor(probs_d_given_obj_y) \
        else torch.as_tensor(probs_d_given_obj_y, dtype=torch.float32)
    probs = probs.to(torch.float32).clamp_min(1e-12)
    pi_y = pi_y if torch.is_tensor(pi_y) else torch.as_tensor(pi_y, dtype=torch.float32)
    pi_y = pi_y.to(torch.float32).clamp_min(1e-12)
    y_idx = torch.as_tensor(_np(y), dtype=torch.long, device=probs.device)
    log_ref = pi_y.to(probs.device).log()[y_idx]                       # [N, n_dom]
    kl = (probs * (probs.log() - log_ref)).sum(dim=1)                  # [N]
    return kl.clamp_min(0.0)


def bootstrap_mean_ci(values, n_boot=200, alpha=0.05, seed=0):
    """Nonparametric bootstrap CI for the mean of `values` (per-sample held-out KL, typically)."""
    v = _np(values).astype(np.float64).ravel()
    if v.size == 0:
        return dict(mean=0.0, ci_low=0.0, ci_high=0.0, std=0.0, n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(int(n_boot), v.size))
    boot_means = v[idx].mean(axis=1)
    return dict(mean=float(v.mean()),
                ci_low=float(np.quantile(boot_means, alpha / 2)),
                ci_high=float(np.quantile(boot_means, 1 - alpha / 2)),
                std=float(boot_means.std()),
                n=int(v.size))


def within_label_permutation(y, d, seed):
    """Permute domain labels D **within each label group** of Y over the WHOLE dataset.

    A general utility that preserves the GLOBAL π_y(D)=p(D|Y). NOTE: for the Phase-2 permutation
    null use `within_label_permutation_on_indices` restricted to the probe training split instead —
    a whole-dataset permutation can move domains across the train/val boundary and so does NOT
    preserve the TRAIN-split π_y(D) that the probe uses as its KL reference.
    """
    y = _np(y).astype(np.int64)
    d = _np(d).astype(np.int64)
    rng = np.random.default_rng(seed)
    out = d.copy()
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if idx.size > 1:
            out[idx] = d[idx][rng.permutation(idx.size)]
    return out


def within_label_permutation_on_indices(y, d, indices, seed):
    """Phase-2 null generator: permute D within each label group, RESTRICTED to `indices`.

    Only the samples in `indices` (the probe TRAINING rows/trials) have their domains shuffled;
    every sample outside `indices` keeps its original D. Because the shuffle is a within-label
    permutation of the training subset, it preserves the TRAINING-split per-label domain multiset
    — hence π_y(D)=p(D|Y) estimated from the training split is exactly preserved — while destroying
    the per-sample O→D pairing the probe could otherwise learn. The held-out (validation) domains
    are left untouched (the leakage KL at eval does not depend on observed D, so the null must not
    alter the held-out split).
    """
    y = _np(y).astype(np.int64)
    d = _np(d).astype(np.int64)
    idx_all = np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    out = d.copy()
    for c in np.unique(y[idx_all]):
        idx = idx_all[y[idx_all] == c]
        if idx.size > 1:
            out[idx] = d[idx][rng.permutation(idx.size)]
    return out


def _trial_split(n, seed, train_frac=0.7):
    """Trial-level (NOT row-level) train/val split so node-rows of the same trial never straddle."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(int(n))
    cut = max(1, int(round(train_frac * n)))
    cut = min(cut, n - 1) if n > 1 else 1
    return np.sort(perm[:cut]), np.sort(perm[cut:])


# --------------------------------------------------------------------------- the conditional probe
def fit_conditional_domain_probe(features, y, d, n_classes, n_domains, *,
                                 train_idx=None, val_idx=None, hidden_dim=64, epochs=100,
                                 lr=1e-3, weight_decay=1e-3, seed=0, device="cpu", smoothing=1e-3):
    """Fit q(D | features, Y) on a train split, evaluate the leakage proxy on a disjoint val split.

    features: [N, F] frozen detached. y, d: [N]. The probe sees Y (one-hot) so the KL-to-π_y(D)
    measures leakage that survives CONDITIONING on the label. π_y is estimated from the TRAIN split
    only (never uses val-D). Returns a result dict; `val_kl` is the per-sample held-out KL so callers
    (e.g. the node map) can re-aggregate it.
    """
    torch.manual_seed(int(seed))
    X = _feat_tensor(features, device)
    N, Fdim = X.shape
    y_np = _np(y).astype(np.int64)
    d_np = _np(d).astype(np.int64)
    if train_idx is None or val_idx is None:
        train_idx, val_idx = _trial_split(N, seed)
    tr = torch.as_tensor(np.asarray(train_idx), dtype=torch.long, device=device)
    va = torch.as_tensor(np.asarray(val_idx), dtype=torch.long, device=device)
    y_t = torch.as_tensor(y_np, dtype=torch.long, device=device)
    d_t = torch.as_tensor(d_np, dtype=torch.long, device=device)
    y_oh = F.one_hot(y_t, int(n_classes)).float()

    probe = nn.Sequential(nn.Linear(Fdim + int(n_classes), int(hidden_dim)), nn.ReLU(),
                          nn.Linear(int(hidden_dim), int(n_domains))).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    inp = torch.cat([X, y_oh], dim=1)
    inp_tr, d_tr = inp[tr], d_t[tr]
    probe.train()
    for _ in range(int(epochs)):
        opt.zero_grad()
        F.cross_entropy(probe(inp_tr), d_tr).backward()
        opt.step()

    # π_y from the TRAIN split (so a permuted-D null preserves it but the probe can't fit D)
    pi_y = compute_label_domain_prior(y_np[_np(tr)], d_np[_np(tr)], n_classes, n_domains, smoothing)
    probe.eval()
    with torch.no_grad():
        probs = F.softmax(probe(inp[va]), dim=1)
        val_kl = conditional_kl_to_prior(probs, y_t[va], pi_y)               # [n_val]
        dom_pred = probs.argmax(1).cpu().numpy()
    d_val = d_np[_np(va)]
    y_val = y_np[_np(va)]
    prior_pred = pi_y.numpy()[y_val].argmax(1)                                # label-only-prior baseline
    dom_acc = float((dom_pred == d_val).mean())
    prior_acc = float((prior_pred == d_val).mean())
    return dict(kl_mean=float(val_kl.mean().item()),
                val_kl=val_kl.cpu().numpy(),
                val_idx=_np(va),
                domain_acc=dom_acc,
                prior_acc=prior_acc,
                leakage_advantage=dom_acc - prior_acc,
                n_train=int(tr.numel()), n_val=int(va.numel()))


def _permutation_null(fit_fn, y, d, n_perm, seed, permute_idx):
    """Retrain `fit_fn(d_perm)` for n_perm nulls; D is permuted within-label ONLY over `permute_idx`
    (the probe training split), leaving validation D unchanged. Returns the null KL array."""
    nulls = []
    for k in range(int(n_perm)):
        d_perm = within_label_permutation_on_indices(y, d, permute_idx, seed=seed + 1 + k)
        nulls.append(float(fit_fn(d_perm)["kl_mean"]))
    return np.asarray(nulls, dtype=np.float64)


def _perm_summary(observed_kl, null_kls):
    """Permutation-test summary: mean/std of the retrained null + the (+1)-smoothed one-sided p-value."""
    null_kls = np.asarray(null_kls, dtype=np.float64)
    n = null_kls.size
    p = (1.0 + float((null_kls >= observed_kl).sum())) / (1.0 + n) if n else float("nan")
    return dict(permutation_mean=float(null_kls.mean()) if n else float("nan"),
                permutation_std=float(null_kls.std()) if n else float("nan"),
                permutation_p=p, n_perm=int(n),
                excess_over_null=float(observed_kl - (null_kls.mean() if n else 0.0)))


def audit_graph_objects(graph_z, node_z, edge_logits, y, d, n_classes, n_domains, *,
                        n_perm=20, seed=0, device="cpu", hidden_dim=64, epochs=100,
                        n_edge_bins=4):
    """Full Phase-2 audit of the three graph objects on a SHARED frozen source split.

    graph_z: [N, Dg]   node_z: [N, C, Dn]   edge_logits: [N, C, C]   y, d: [N]
    Returns {"graph":{...}, "node":{...}, "edge":{...}} where each block carries kl_mean,
    permutation_{mean,std,p}, domain_acc, prior_acc, leakage_advantage and a bootstrap CI; the node
    block adds a length-C `node_leakage_map`; the edge block adds the [C,C] `edge_leakage_map`
    (non-neural binned CMI). All inputs are detached; no target data is involved.
    """
    graph_z = _feat_tensor(graph_z, device)
    node_z = _feat_tensor(node_z, device)
    edge_logits = _feat_tensor(edge_logits, device)
    N, C, _ = edge_logits.shape
    Dn = node_z.shape[2]
    y_np = _np(y).astype(np.int64)
    d_np = _np(d).astype(np.int64)
    tr_trials, va_trials = _trial_split(N, seed)            # shared trial split for all three objects

    # ---- graph: probe q(D | Z_g, Y) ---------------------------------------------------------------
    def fit_graph(d_arr):
        return fit_conditional_domain_probe(graph_z, y_np, d_arr, n_classes, n_domains,
                                            train_idx=tr_trials, val_idx=va_trials,
                                            hidden_dim=hidden_dim, epochs=epochs, seed=seed, device=device)
    g = fit_graph(d_np)
    # null permutes D within-label ONLY over the training trials (preserves train-split π_y)
    g.update(_perm_summary(g["kl_mean"],
                           _permutation_null(fit_graph, y_np, d_np, n_perm, seed, permute_idx=tr_trials)))
    g["kl_ci"] = bootstrap_mean_ci(g.pop("val_kl"), seed=seed)
    g.pop("val_idx", None)

    # ---- node: shared probe over flattened (trial, channel) rows ----------------------------------
    # feature = [z_v, normalized channel id]; y/d repeated over channels; split stays trial-level.
    chan_id = np.tile(np.arange(C), N)                                          # [N*C]
    chan_feat = (chan_id / max(C - 1, 1)).astype(np.float32)[:, None]
    node_feat = torch.cat([node_z.reshape(N * C, Dn),
                           torch.as_tensor(chan_feat, device=device)], dim=1)   # [N*C, Dn+1]
    trial_id = np.repeat(np.arange(N), C)
    y_rep = np.repeat(y_np, C)
    d_rep = np.repeat(d_np, C)
    tr_rows = np.where(np.isin(trial_id, tr_trials))[0]
    va_rows = np.where(np.isin(trial_id, va_trials))[0]

    def fit_node(d_arr):
        d_rep_arr = np.repeat(d_arr, C)
        return fit_conditional_domain_probe(node_feat, y_rep, d_rep_arr, n_classes, n_domains,
                                            train_idx=tr_rows, val_idx=va_rows,
                                            hidden_dim=hidden_dim, epochs=epochs, seed=seed, device=device)
    nres = fit_node(d_np)
    # per-channel leakage map from the held-out per-sample KL
    val_rows = nres["val_idx"]
    val_kl = nres["val_kl"]
    val_chan = chan_id[val_rows]
    node_map = np.zeros(C, dtype=np.float64)
    for c in range(C):
        m = val_chan == c
        node_map[c] = float(val_kl[m].mean()) if m.any() else 0.0
    # permute over TRAIN TRIALS (not rows): fit_node receives a trial-level d_arr and repeats it
    nres.update(_perm_summary(nres["kl_mean"],
                              _permutation_null(fit_node, y_np, d_np, n_perm, seed, permute_idx=tr_trials)))
    nres["kl_ci"] = bootstrap_mean_ci(nres.pop("val_kl"), seed=seed)
    nres.pop("val_idx", None)
    nres["node_leakage_map"] = node_map.tolist()

    # ---- edge: compact upper-triangular summary probe + non-neural per-edge map -------------------
    iu = np.triu_indices(C, k=1)
    edge_vec = edge_logits[:, iu[0], iu[1]]                                      # [N, C(C-1)/2]

    def fit_edge(d_arr):
        return fit_conditional_domain_probe(edge_vec, y_np, d_arr, n_classes, n_domains,
                                            train_idx=tr_trials, val_idx=va_trials,
                                            hidden_dim=hidden_dim, epochs=epochs, seed=seed, device=device)
    e = fit_edge(d_np)
    e.update(_perm_summary(e["kl_mean"],
                           _permutation_null(fit_edge, y_np, d_np, n_perm, seed, permute_idx=tr_trials)))
    e["kl_ci"] = bootstrap_mean_ci(e.pop("val_kl"), seed=seed)
    e.pop("val_idx", None)
    e["edge_leakage_map"] = edge_binned_cmi_map(edge_logits, y_np, d_np, n_classes, n_domains,
                                                n_bins=n_edge_bins).cpu().numpy().tolist()
    return dict(graph=g, node=nres, edge=e)


def edge_binned_cmi_map(edge_logits, y, d, n_classes, n_domains, n_bins=4, smoothing=1e-3):
    """Non-neural per-edge conditional MI map  I(E_ij ; D | Y), Tensor[C, C], for interpretation only.

    Each scalar edge value E_ij is quantile-binned into `n_bins`; the plug-in CMI
      I(E;D|Y) = Σ_y p(y) Σ_{b,m} p(b,m|y) log[ p(b,m|y) / (p(b|y) p(m|y)) ]
    is estimated from smoothed joint counts. The map is symmetric with a zero diagonal. This is a
    DIAGNOSTIC audit of where adjacency leaks subject identity — it is never used as a training loss
    (CIGL regularizes a compact edge summary, not per-edge heads).
    """
    E = _np(edge_logits)
    y = _np(y).astype(np.int64)
    d = _np(d).astype(np.int64)
    N, C, _ = E.shape
    n_bins = int(n_bins)
    K, M = int(n_classes), int(n_domains)
    out = np.zeros((C, C), dtype=np.float64)
    py = np.bincount(y, minlength=K).astype(np.float64)
    py = py / max(py.sum(), 1e-12)
    for i in range(C):
        for j in range(i + 1, C):
            e = E[:, i, j]
            # quantile bin edges (deduped); degenerate (constant) edges -> all in one bin -> CMI 0
            qs = np.quantile(e, np.linspace(0, 1, n_bins + 1))
            edges = np.unique(qs)
            if edges.size < 3:
                continue
            b = np.clip(np.digitize(e, edges[1:-1]), 0, edges.size - 2)
            nb = edges.size - 1
            cmi = 0.0
            for c in range(K):
                mask = y == c
                if mask.sum() < 2:
                    continue
                joint = np.zeros((nb, M), dtype=np.float64)
                for bb, dd in zip(b[mask], d[mask]):
                    joint[bb, dd] += 1.0
                joint += smoothing
                joint /= joint.sum()
                pb = joint.sum(axis=1, keepdims=True)
                pm = joint.sum(axis=0, keepdims=True)
                cmi += py[c] * float((joint * np.log(joint / (pb * pm))).sum())
            out[i, j] = out[j, i] = max(cmi, 0.0)
    return torch.tensor(out, dtype=torch.float32)
