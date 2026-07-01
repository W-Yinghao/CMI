"""Two-step alternating trainer (Step A: fit domain posteriors on detached Z;
Step B: update backbone+task-head with task CE + lambda_t * L_CMI), with KL warm-up.
Mirrors synthetic/sanity_check.py.train_one but for EEG backbones returning (logits, Z).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from cmi.methods.regularizers import DomainPosteriors, empirical_priors, effective_priors, subject_priors
from cmi.methods.contrastive import sup_con_loss, CMI_METHODS, SUPCON_METHODS
from cmi.methods.fmca import FMCA_METHODS, fmca_reg
from cmi.methods import augment as ssl
from cmi.methods.graph_regularizers import NodePosterior, EdgePosterior
from cmi.methods import dg_penalties as dgp

# Every implemented framework. Anything else must fail loudly (not silently train ERM).
ALL_METHODS = {
    "erm", "iib", "graphcmi", "graphdualpc", "dual", "dualc", "dualpc", "dualpc_hinge", "dualpc_marginal",
} | CMI_METHODS | SUPCON_METHODS | dgp.DG_METHODS | FMCA_METHODS | ssl.ALL_SSL


def _make_sampler(y, d, mode):
    """mode: 'raw' (natural source dist, no reweighting) | 'classbal' (balance Y only -> PRESERVES
    p(D|Y), the right default for the conditional-MI objective) | 'domainbal' (balance (class,domain)
    -> uniformizes p(D|Y) in-batch, which is INCONSISTENT with an empirical pi_y target)."""
    if mode == "raw":
        return None
    if mode == "classbal":
        _, inv, cnt = np.unique(y, return_inverse=True, return_counts=True)
    elif mode == "domainbal":
        key = d.astype(np.int64) * (int(y.max()) + 1) + y
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    else:
        raise ValueError(f"unknown sampler '{mode}'")
    w = 1.0 / cnt[inv]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), replacement=True)


def _label_shift_weights(y, d, n_dom, n_cls):
    """GLS (anchor A4) per-sample importance weight w_i = pi*(y_i)/pi_{d_i}(y_i),
    pi* = uniform reference (1/n_cls), pi_d(y) = per-domain class frequency p(Y=y|D=d).
    Estimated (Laplace-smoothed) from the training (y,d) arrays. Reweighting domain d to the
    common reference prior makes I~(Y;D)=0, decoupling the encoder I(Z;D|Y) and decoder
    I(Y;D|Z) CMIs so both can be driven to zero. Returns a float32 [N] array (mean ~ 1)."""
    counts = np.zeros((n_dom, n_cls), dtype=np.float64)
    for yi, di in zip(y, d):
        counts[di, yi] += 1.0
    pi_d = (counts + 1.0) / (counts.sum(1, keepdims=True) + n_cls)   # p(Y=y|D=d), smoothed
    pi_star = 1.0 / n_cls                                            # uniform reference prior
    w = pi_star / pi_d[d, y]
    w = w / w.mean()                                                # normalize so E[w]~1 (stable LR)
    return w.astype("float32")


def _scalar(x):
    return float(x.detach().cpu()) if torch.is_tensor(x) else float(x)


def _unpack_graph(out):
    """forward_graph may return a 4-tuple (logits, graph_z, node_z, edge_logits) or a future 5-tuple
    (..., fused_z). Return (logits, graph_z, node_z, edge_logits, z_dec) where z_dec is the representation
    the decoder-CMI term acts on: fused_z when present, else graph_z (the readout the classifier uses)."""
    if len(out) == 5:
        logits, gz, nz, el, fz = out
    else:
        logits, gz, nz, el = out
        fz = None
    return logits, gz, nz, el, (fz if fz is not None else gz)


def default_dec_margin(method):
    """Method-specific default decoder gate.

    Route-C `dualc` keeps a CE-residual null margin. DualPC / graphdualpc use a JS consistency loss whose
    empirical scale is much smaller, so the paper-facing default must keep the P(Y|Z) side active.
    """
    return 0.0 if method in {"dualpc", "dualpc_hinge", "dualpc_marginal", "graphdualpc"} else 0.02


def resolve_dec_margin(method, dec_margin):
    return default_dec_margin(method) if dec_margin is None else float(dec_margin)


def _balanced_acc(y_true, y_pred, n_cls):
    """Macro-recall (balanced accuracy) over present classes. Pure source-side metric (P3-E)."""
    recs = [float((y_pred[y_true == c] == c).mean()) for c in range(n_cls) if (y_true == c).any()]
    return float(np.mean(recs)) if recs else 0.0


def train_model(backbone, Xtr, ytr, dtr, n_cls, method="lpc_prior", lam=1.0, gamma=0.0,
                lam_edge=0.0, beta=0.0, balance=False, label_correct=False, reweight_dual=False,
                dec_margin=None, epochs=200, bs=64, warmup=40, n_inner=2,
                z_margin=0.0, dec_scale=1.0,
                lr=1e-3, post_lr=2e-3,
                weight_decay=0.0, sampler="classbal", prior_mode="empirical", prior_alpha=1.0,
                early_stop=False, source_val_domains=None,
                device="cpu", seed=0, log_every=0):
    if method not in ALL_METHODS:
        raise ValueError(f"unknown method '{method}'; allowed: {sorted(ALL_METHODS)}")
    torch.manual_seed(seed)
    n_dom = int(dtr.max()) + 1
    # P3-E: OPT-IN source-only early stopping. Hold out one or more SOURCE domains (subjects) as an inner
    # validation set; select on source-val balanced accuracy (tie-break lower source-val CE) and restore the
    # best-epoch weights before returning. Target labels are NEVER touched (val is a held-out SOURCE domain);
    # n_dom is fixed from the FULL dtr so the domain space is stable, then Xtr/ytr/dtr are rebound to the
    # train part so ALL downstream (priors, GLS weights, sampler, dataloader) uses source-train only.
    # Default off (source_val_domains=None) -> zero behavior change for every existing method/run.
    Xval = yval = None
    es_val_domains = []
    if early_stop and source_val_domains is not None:
        _svd = np.asarray(list(source_val_domains))
        Xtr, ytr, dtr = np.asarray(Xtr), np.asarray(ytr), np.asarray(dtr)
        val_mask = np.isin(dtr, _svd)
        if 0 < int(val_mask.sum()) < len(dtr) and int((~val_mask).sum()) > 0:
            Xval, yval = Xtr[val_mask], ytr[val_mask]
            es_val_domains = [int(v) for v in np.unique(dtr[val_mask])]
            Xtr, ytr, dtr = Xtr[~val_mask], ytr[~val_mask], dtr[~val_mask]
    # KL target must be CONSISTENT with the sampler: 'empirical' pi_y for raw/classbal (which
    # preserve p(D|Y)); 'effective' (uniform-over-present) to match a domainbal sampler.
    prior_fn = {"empirical": empirical_priors, "effective": effective_priors, "subject": subject_priors}[prior_mode]
    priors = prior_fn(ytr, dtr, n_dom, n_cls, alpha=prior_alpha)
    post = DomainPosteriors(backbone.z_dim, n_dom, n_cls, priors, device=device)
    post_dec = post   # decoder-residual head (q(Y|Z)/h(Y|Z,D)/h0); ALIASES `post` unless a distinct
    #                   fused_z backbone splits it below (P3-C) — keeps DGCNNGraph behavior byte-for-byte.

    uses_cmi = method in CMI_METHODS
    uses_supcon = method in SUPCON_METHODS
    uses_fmca = method in FMCA_METHODS          # Route 2: closed-form, no Step-A posterior
    uses_ssl = method in ssl.ALL_SSL            # self-supervised contrastive (SimCLR/BYOL [+ our CMI])
    is_lpc_ssl = method in ssl.LPC_SSL_METHODS  # SSL framework hosting our CMI term
    ssl_kind = "byol" if "byol" in method else "simclr"
    uses_graphcmi = method == "graphcmi"        # GNN node/edge CMI (needs backbone.forward_graph)
    uses_graphdualpc = method == "graphdualpc"  # Graph-DualCMI: encoder graph/node/edge I(Z;D|Y) [GLS] + decoder residual I(Y;D|Z)
    uses_graph = uses_graphcmi or uses_graphdualpc
    node_post = edge_post = None
    if uses_graph:                              # global term reuses `post`; add node + edge heads
        # Fail closed: the GNN branch calls backbone.forward_graph(x) -> (logits, graph_Z, node_Z,
        # edge_logits[, fused_z]). A non-graph backbone (EEGNet, TSMNet, ...) lacks it; without this guard
        # the AttributeError would only surface mid-training. Raise a clear, actionable error up front.
        if not callable(getattr(backbone, "forward_graph", None)):
            raise ValueError(
                f"method='{method}' requires a graph backbone exposing "
                f"forward_graph(x) -> (logits, graph_Z, node_Z, edge_logits[, fused_z]); "
                f"{type(backbone).__name__} has no callable forward_graph. "
                f"Use --backbone DGCNNGraph (or another graph backbone), or pick a non-graph method.")
        # For method='graphcmi' the (lam, gamma, lam_edge) knobs ARE (lambda_g, lambda_node,
        # lambda_edge); they are reported under those names in the returned diagnostics.
        # node feature dim may differ from the graph readout dim (z_dim): e.g. the DGCNN adapter reads out
        # graph_z=64 but node_z has hidden=16. Use an explicit node_z_dim when the backbone exposes it;
        # GraphCMINet (node_Z dim == z_dim) has no such attr and falls back to z_dim unchanged.
        node_post = NodePosterior(getattr(backbone, "node_z_dim", backbone.z_dim), n_dom, n_cls, priors,
                                  n_chans=int(Xtr.shape[1])).to(device)   # q(D|Z_v,e_v,Y): node-id-conditioned
        edge_post = EdgePosterior(int(Xtr.shape[1]), n_dom, n_cls, priors).to(device)
        # P3-C: decoder residual acts on the representation the CLASSIFIER reads (fused_z). For DGCNNGraph
        # z_dec == graph_z, so `post` already hosts the decoder probes (post_dec := post, exact CIGL_46
        # behavior). A backbone declaring a DISTINCT fused_z (meta['distinct_fused_z']) gets its OWN
        # DomainPosteriors on the fused_z dim, with INDEPENDENT params — encoder graph-CMI (on graph_z)
        # and decoder residual (on fused_z) then never share heads.
        if uses_graphdualpc and bool(getattr(backbone, "meta", {}).get("distinct_fused_z", False)):
            fused_dim = int(getattr(backbone, "fused_z_dim", backbone.z_dim))
            post_dec = DomainPosteriors(fused_dim, n_dom, n_cls, priors, device=device)
    is_iib = method == "iib"
    is_dual = method == "dual"                   # joint encoder I(Z;D|Y) + decoder I(Y;D|Z) invariance
    is_dualc = method == "dualc"                  # Route C: GLS-reweighted, RESIDUAL (intercept) decoder, gated
    # AAAI candidate: factorized reference-P(Z) control + gated JS-consistency P(Y|Z).
    # Under GLS, Y and D have a common reference prior; driving I_w(Z;D|Y) to zero therefore
    # aligns the induced reference mixture P_w(Z|D)=sum_y pi*(y)P(Z|Y=y,D) without the label-erasure
    # risk of a direct marginal I_w(Z;D) penalty.
    is_dualpc = method == "dualpc"
    is_dualpc_hinge = method == "dualpc_hinge"
    is_dualpc_marginal = method == "dualpc_marginal"  # direct I_w(Z;D) ablation; kept after negative synthetic tests
    dec_margin = resolve_dec_margin(method, dec_margin)
    # class-balanced (BER) CE weights for the task CE; GLS CMI weights are handled separately below.
    ce_weight = None
    if balance:
        cnt = np.bincount(ytr, minlength=n_cls).astype("float32")
        ce_weight = torch.tensor((cnt.sum() / (n_cls * np.maximum(cnt, 1))), dtype=torch.float32, device=device)
    cmi_method = "lpc_prior" if method == "lpc_supcon" else method
    is_adv = method in dgp.ADV_METHODS
    is_pen = method in dgp.PENALTY_METHODS
    is_gdro = method == "groupdro"

    # adversarial frameworks own a discriminator, optimized jointly via gradient reversal
    disc = dgp.make_discriminator(backbone.z_dim, n_dom, method == "cdann", n_cls).to(device) \
        if is_adv else None
    # self-supervised contrastive heads (SimCLR: projector; BYOL: + predictor + EMA target)
    ssl_proj = ssl_pred = tgt_bb = tgt_proj = None
    if uses_ssl:
        ssl_proj = ssl.MLPHead(backbone.z_dim).to(device)
        if ssl_kind == "byol":
            import copy
            ssl_pred = ssl.MLPHead(128, out=128).to(device)
            tgt_bb, tgt_proj = copy.deepcopy(backbone), copy.deepcopy(ssl_proj)
            for p in list(tgt_bb.parameters()) + list(tgt_proj.parameters()):
                p.requires_grad_(False)
    main_params = (list(backbone.parameters()) + (list(disc.parameters()) if disc else [])
                   + (list(ssl_proj.parameters()) if ssl_proj else [])
                   + (list(ssl_pred.parameters()) if ssl_pred else []))
    # SPD/manifold backbones (e.g. TSMNet's Stiefel BiMap) need a Riemannian optimizer.
    try:
        import geoopt
        has_manifold = any(isinstance(p, geoopt.ManifoldParameter) for p in main_params)
    except Exception:
        has_manifold = False
    if has_manifold:
        import geoopt
        opt_main = geoopt.optim.RiemannianAdam(main_params, lr=lr, weight_decay=weight_decay)
    else:
        opt_main = torch.optim.AdamW(main_params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=epochs)
    post_params = list(post.parameters())
    if uses_graph:
        post_params += list(node_post.parameters()) + list(edge_post.parameters())
    if post_dec is not post:                         # separate decoder head -> optimize its params too
        post_params += list(post_dec.parameters())
    opt_post = torch.optim.Adam(post_params, lr=post_lr)
    gdro_q = torch.ones(n_dom, device=device) / n_dom   # GroupDRO domain weights

    # GLS label-shift weights (anchor A4): per-sample w_i = pi*(y_i)/pi_{d_i}(y_i); all-ones if off.
    # Carried as a 4th dataset column so it follows the sampler's row permutation exactly.
    # Needed by --label_correct (CE/decoder weighting) AND --reweight_dual (Route B: both CMIs).
    rw_dual = reweight_dual and is_dual    # Route B only meaningful for the joint 'dual' objective
    if label_correct or rw_dual or is_dualc or is_dualpc or is_dualpc_hinge or is_dualpc_marginal or uses_graphdualpc:
        wtr = _label_shift_weights(ytr, dtr, n_dom, n_cls)   # graphdualpc: GLS weights on encoder+decoder CMI terms
    else:
        wtr = np.ones(len(ytr), dtype="float32")
    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(dtr),
                       torch.tensor(wtr))
    effective_sampler = "raw" if (is_dualpc or is_dualpc_hinge or is_dualpc_marginal or uses_graphdualpc) else sampler
    # dualpc variants apply explicit GLS weights to the P(Z) and P(Y|Z) CMI estimators. A class/domain-
    # balanced sampler would impose a second, implicit reference distribution and break those semantics.
    smp = _make_sampler(ytr, dtr, effective_sampler)
    # P2.4: drop_last only when the tail batch would be size 1 (which breaks BatchNorm); otherwise KEEP it so
    # rare domain×class cells are not dropped (CMI is sensitive to them).
    drop_last = (len(ds) % bs == 1)
    dl = DataLoader(ds, batch_size=bs, sampler=smp, shuffle=smp is None, drop_last=drop_last)

    diag = dict(stepA_dom_correct=0, stepA_dom_total=0, inloop_reg=[],
                sampler=effective_sampler,  # q_psi diagnostics
                # graphcmi-only: per-component leakage breakdown + per-head Step-A critic quality
                # (empty/unused for other methods)
                inloop_ce=[], inloop_reg_graph=[], inloop_reg_node=[], inloop_reg_edge=[],
                stepA_graph_correct=0, stepA_node_correct=0, stepA_edge_correct=0,
                stepA_graph_total=0, stepA_node_total=0, stepA_edge_total=0,
                stepA_graph_loss=[], stepA_node_loss=[], stepA_edge_loss=[])
    backbone.train(); post.train()
    if post_dec is not post:
        post_dec.train()
    _es_active = Xval is not None                     # P3-E source-only early stopping active this run?
    best_state = None; best_vbacc = -1.0; best_vce = float("inf"); best_ep = -1; last_vbacc = float("nan")
    for ep in range(epochs):
        lam_t = lam * min(1.0, ep / max(1, warmup))
        gamma_t = gamma * min(1.0, ep / max(1, warmup))   # decoder-term warmup (dual)
        last_epoch = ep == epochs - 1
        for xb, yb, db, wb in dl:
            xb, yb, db, wb = xb.to(device), yb.to(device), db.to(device), wb.to(device)
            if uses_ssl:                                  # self-supervised contrastive framework
                if is_lpc_ssl:                            # Step A: fit CMI posterior on detached clean z
                    with torch.no_grad():
                        _, zc = backbone(xb)
                    for _ in range(n_inner):
                        la = post.posterior_loss(zc, yb, db)
                        opt_post.zero_grad(); la.backward(); opt_post.step()
                x1, x2 = ssl.two_views(xb)
                logits, z1 = backbone(x1)
                _, z2 = backbone(x2)
                loss = F.cross_entropy(logits, yb)        # joint: CE + γ·SSL [+ λ·I(Z;D|Y)]
                ssl_w = gamma if is_lpc_ssl else lam_t     # SSL weight (γ for hybrid, λ for pure)
                if ssl_kind == "simclr":
                    loss = loss + ssl_w * ssl.simclr_loss(ssl_proj(z1), ssl_proj(z2))
                else:                                     # BYOL (EMA target, no negatives)
                    with torch.no_grad():
                        _, zt1 = tgt_bb(x1); _, zt2 = tgt_bb(x2)
                        t1, t2 = tgt_proj(zt1), tgt_proj(zt2)
                    p1, p2 = ssl_pred(ssl_proj(z1)), ssl_pred(ssl_proj(z2))
                    loss = loss + ssl_w * ssl.byol_loss(p1, t2, p2, t1)
                if is_lpc_ssl:                            # OUR CMI term on the SSL representation
                    r = post.reg("lpc_prior", z1, yb)
                    loss = loss + lam_t * r
                    if last_epoch:
                        diag["inloop_reg"].append(_scalar(r))
                opt_main.zero_grad(); loss.backward(); opt_main.step()
                if ssl_kind == "byol":                    # EMA update of the target network
                    with torch.no_grad():
                        for o, t in zip(list(backbone.parameters()) + list(ssl_proj.parameters()),
                                        list(tgt_bb.parameters()) + list(tgt_proj.parameters())):
                            t.mul_(0.99).add_(o, alpha=0.01)
                continue
            if uses_graphcmi:                             # GNN: graph(global) + node + edge CMI
                warm = min(1.0, ep / max(1, warmup))
                with torch.no_grad():                     # Step A: fit all 3 posteriors on detached graph features
                    _, gz, nz, el = backbone.forward_graph(xb)
                has_edge = el is not None                  # static-adjacency backbones (DGCNN adapter) -> edge_logits=None
                if lam_edge != 0 and not has_edge:         # fail closed: cannot apply an edge term without an edge object
                    raise ValueError("method='graphcmi' with lambda_edge!=0 needs per-sample edge_logits; "
                                     "this backbone yields edge_logits=None (static/shared adjacency).")
                for _ in range(n_inner):
                    la = post.posterior_loss(gz, yb, db) + node_post.step_a_loss(nz, yb, db)
                    if has_edge:
                        la = la + edge_post.step_a_loss(el, yb, db)
                    opt_post.zero_grad(); la.backward(); opt_post.step()
                logits, gz, nz, el = backbone.forward_graph(xb)   # Step B (grad to encoder)
                # Three named leakage proxies; weights (lam, gamma, lam_edge) == (lambda_g, lambda_node,
                # lambda_edge). Each term computed into its own variable so it can be logged separately;
                # the loss math is byte-identical to the prior inline form.
                r_graph = post.reg("lpc_prior", gz, yb)   # lambda_g    : I(Z_g;D|Y)
                r_node = node_post.reg(nz, yb)            # lambda_node : (1/C) Σ_v I(Z_v;D|Y)
                r_edge = edge_post.reg(el, yb) if has_edge else None   # lambda_edge : I(A;D|Y) (None -> no edge object)
                ce = F.cross_entropy(logits, yb)
                loss = ce + lam * warm * r_graph + gamma * warm * r_node
                if r_edge is not None:
                    loss = loss + lam_edge * warm * r_edge
                opt_main.zero_grad(); loss.backward(); opt_main.step()
                if last_epoch:
                    diag["inloop_reg"].append(_scalar(r_graph))   # back-compat: graph term == inloop_reg
                    diag["inloop_ce"].append(_scalar(ce))
                    diag["inloop_reg_graph"].append(_scalar(r_graph))
                    diag["inloop_reg_node"].append(_scalar(r_node))
                    if r_edge is not None:
                        diag["inloop_reg_edge"].append(_scalar(r_edge))
                    with torch.no_grad():   # Step-A critic quality per head (diagnostic only; no grad/loss effect)
                        gpred = post.q_dzy(torch.cat([gz, F.one_hot(yb, n_cls).float()], 1)).argmax(1)
                        npred = node_post._logits(nz, yb).argmax(-1)   # [B,C] per-channel domain pred
                        diag["stepA_graph_correct"] += int((gpred == db).sum()); diag["stepA_graph_total"] += int(db.numel())
                        diag["stepA_node_correct"] += int((npred == db.unsqueeze(1)).sum()); diag["stepA_node_total"] += int(npred.numel())
                        diag["stepA_graph_loss"].append(_scalar(post.posterior_loss(gz, yb, db)))
                        diag["stepA_node_loss"].append(_scalar(node_post.step_a_loss(nz, yb, db)))
                        if has_edge:                            # edge head only exists for per-sample edge objects
                            epred = edge_post._logits(el, yb).argmax(1)    # [B]
                            diag["stepA_edge_correct"] += int((epred == db).sum()); diag["stepA_edge_total"] += int(db.numel())
                            diag["stepA_edge_loss"].append(_scalar(edge_post.step_a_loss(el, yb, db)))
                continue
            if uses_graphdualpc:                          # Graph-DualCMI: encoder graph/node/edge I(Z;D|Y) [GLS] + decoder residual I(Y;D|Z)
                warm = min(1.0, ep / max(1, warmup))
                # Four explicit, paper-facing weights (see docs/CIGL_46): lambda_g=lam, lambda_node=beta,
                # lambda_edge=lam_edge, gamma_dec=gamma. Encoder terms use the DualPC GLS-reweighted
                # reference="marginal" (post-GLS domain marginal); decoder term is the JS residual.
                lambda_g, lambda_node, lambda_edge_w, gamma_dec = lam, beta, lam_edge, gamma
                with torch.no_grad():                     # Step A: fit posteriors + decoder probes on detached features
                    _og = backbone.forward_graph(xb)
                    # Fail closed on an UNDECLARED distinct fused_z: a separate post_dec is only allocated when
                    # the backbone declares meta['distinct_fused_z']. If a backbone returns a distinct fused_z
                    # WITHOUT declaring it, post_dec still aliases `post` (sized for graph_z) -> refuse rather
                    # than silently fit the decoder probes on the wrong Z (see docs/CIGL_47 P3-C).
                    if len(_og) == 5 and _og[4] is not None and _og[4] is not _og[1] and post_dec is post:
                        raise NotImplementedError(
                            "graphdualpc got a distinct fused_z but no separate decoder head was allocated; "
                            "the backbone must declare meta['distinct_fused_z']=True and expose fused_z_dim "
                            "so post_dec is sized on the fused_z dim (see docs/CIGL_47).")
                    _, gz_d, nz_d, el_d, zdec_d = _unpack_graph(_og)
                has_edge = el_d is not None                # static-adjacency (DGCNN adapter) -> edge_logits=None
                if lambda_edge_w != 0 and not has_edge:    # fail closed: no edge term without a per-sample edge object
                    raise ValueError("method='graphdualpc' with lambda_edge!=0 needs per-sample edge_logits; "
                                     "this backbone yields edge_logits=None (static/shared adjacency).")
                for _ in range(n_inner):
                    # Encoder head fits q(D|Z_g,Y) on graph_z; decoder head (post_dec) fits its q(Y|Z)/h0
                    # (posterior_loss) and h(Y|Z,D) (iib_ce_h) on fused_z. When post_dec IS post (DGCNNGraph,
                    # z_dec == graph_z), the extra posterior_loss is skipped -> byte-identical to CIGL_46.
                    la = post.posterior_loss(gz_d, yb, db, weight=wb)       # q(D|Z_g,Y)      [GLS] (encoder)
                    if post_dec is not post:
                        la = la + post_dec.posterior_loss(zdec_d, yb, db, weight=wb)  # q(Y|fused),h0 [GLS] (decoder)
                    la = (la + post_dec.iib_ce_h(zdec_d, yb, db, weight=wb)  # h(Y|Z,D) probe  [GLS] (decoder)
                          + node_post.step_a_loss(nz_d, yb, db, weight=wb))  # q(D|Z_v,e_v,Y)  [GLS]
                    if has_edge:
                        la = la + edge_post.step_a_loss(el_d, yb, db, weight=wb)
                    opt_post.zero_grad(); la.backward(); opt_post.step()
                logits, gz, nz, el, zdec = _unpack_graph(backbone.forward_graph(xb))   # Step B (grad to encoder)
                if label_correct:                                          # optional GLS-weighted task CE (off by default)
                    per = F.cross_entropy(logits, yb, weight=ce_weight, reduction="none")
                    ce = (wb * per).sum() / wb.sum().clamp(min=1e-8)
                else:
                    ce = F.cross_entropy(logits, yb, weight=ce_weight)     # default: GLS on the CMI side, not on task CE
                r_graph = post.reg("lpc_prior", gz, yb, weight=wb, reference="marginal")     # lambda_g   : I~(Z_g;D|Y)
                r_node = node_post.reg(nz, yb, weight=wb, reference="marginal")              # lambda_node: (1/C) Σ_v I~(Z_v;D|Y)
                r_edge = edge_post.reg(el, yb, weight=wb, reference="marginal") if has_edge else None  # lambda_edge: I~(A;D|Y)
                r_dec_js = post_dec.dec_js_residual(zdec, db, weight=wb)       # gamma_dec  : decoder residual JS (training loss) [on fused_z]
                r_dec_ce = post_dec.dec_cmi_residual(zdec, yb, db, weight=wb)  # decoder residual CE (diagnostic only) [on fused_z]
                loss = ce + lambda_g * warm * r_graph + lambda_node * warm * r_node
                if r_edge is not None:
                    loss = loss + lambda_edge_w * warm * r_edge
                # P3-D: scale THEN threshold — [dec_scale·JS − dec_margin]_+ (matches the CIGL_47 objective).
                # For graphdualpc dec_margin defaults to 0.0 and JS≥0, so this is identical to the CIGL_46
                # form (dec_scale·relu(JS)) at dec_scale=1.0 → byte-compatible with the G2 runs.
                r_dec_gate = F.relu(dec_scale * r_dec_js - dec_margin)
                loss = loss + gamma_dec * warm * r_dec_gate
                opt_main.zero_grad(); loss.backward(); opt_main.step()
                if last_epoch:
                    diag["inloop_reg"].append(_scalar(r_graph))           # back-compat: graph term == inloop_reg
                    diag["inloop_ce"].append(_scalar(ce))
                    diag["inloop_reg_graph"].append(_scalar(r_graph))
                    diag["inloop_reg_node"].append(_scalar(r_node))
                    if r_edge is not None:
                        diag["inloop_reg_edge"].append(_scalar(r_edge))
                    diag.setdefault("inloop_dec_js", []).append(_scalar(r_dec_js))
                    diag.setdefault("inloop_dec_ce", []).append(_scalar(r_dec_ce))
                    # P3-D decoder-activation diagnostics: is the [dec_scale*JS - dec_margin]_+ gate firing?
                    # (CIGL_46 the decoder term was dormant: dec_js_res~3e-4, loss_dec_over_ce << 1%.)
                    diag.setdefault("inloop_dec_js_scaled", []).append(_scalar(dec_scale * r_dec_js))
                    diag.setdefault("inloop_loss_dec", []).append(gamma_dec * warm * float(r_dec_gate))
                    diag.setdefault("inloop_dec_gate", []).append(
                        1.0 if float(dec_scale * r_dec_js) > dec_margin else 0.0)
                    with torch.no_grad():   # Step-A critic quality per head (GLS diagnostic; probe underfit check)
                        gpred = post.q_dzy(torch.cat([gz, F.one_hot(yb, n_cls).float()], 1)).argmax(1)
                        npred = node_post._logits(nz, yb).argmax(-1)   # [B,C] per-channel domain pred
                        diag["stepA_graph_correct"] += int((gpred == db).sum()); diag["stepA_graph_total"] += int(db.numel())
                        diag["stepA_node_correct"] += int((npred == db.unsqueeze(1)).sum()); diag["stepA_node_total"] += int(npred.numel())
                        diag["stepA_graph_loss"].append(_scalar(post.posterior_loss(gz, yb, db, weight=wb)))
                        diag["stepA_node_loss"].append(_scalar(node_post.step_a_loss(nz, yb, db, weight=wb)))
                        if has_edge:
                            epred = edge_post._logits(el, yb).argmax(1)    # [B]
                            diag["stepA_edge_correct"] += int((epred == db).sum()); diag["stepA_edge_total"] += int(db.numel())
                            diag["stepA_edge_loss"].append(_scalar(edge_post.step_a_loss(el, yb, db, weight=wb)))
                continue
            # Step A: fit auxiliary predictor(s) on detached Z (CMI posteriors, or IIB's h)
            fits_qdzy = uses_cmi or is_dual or is_dualc or is_dualpc or is_dualpc_hinge or is_dualpc_marginal
            if fits_qdzy or is_iib:
                with torch.no_grad():
                    _, z = backbone(xb)
                for _ in range(n_inner):
                    if is_dual or is_dualc or is_dualpc or is_dualpc_hinge or is_dualpc_marginal:
                        # dualpc variants/dualc fit all auxiliary probes on the GLS-reweighted measure.
                        w = wb if (rw_dual or is_dualc or is_dualpc or is_dualpc_hinge or is_dualpc_marginal) else None
                        la = post.posterior_loss(z, yb, db, weight=w) + post.iib_ce_h(z, yb, db, weight=w)
                    elif is_iib:
                        la = post.iib_ce_h(z, yb, db)
                    else:
                        la = post.posterior_loss(z, yb, db)
                    opt_post.zero_grad(); la.backward(); opt_post.step()
                if last_epoch and fits_qdzy:           # how well does q_psi predict D from (Z,Y)?
                    with torch.no_grad():
                        pr = post.q_dzy(torch.cat([z, F.one_hot(yb, n_cls).float()], 1)).argmax(1)
                    diag["stepA_dom_correct"] += int((pr == db).sum()); diag["stepA_dom_total"] += len(db)
            # Step B: backbone (+ task head, + framework term)
            logits, z = backbone(xb)
            if is_gdro:                                  # worst-domain reweighting (GroupDRO)
                doms = [i for i in range(n_dom) if (db == i).sum() > 0]
                li = {i: F.cross_entropy(logits[db == i], yb[db == i]) for i in doms}
                with torch.no_grad():                    # update domain weights from detached risks
                    for i in doms:
                        gdro_q[i] = (gdro_q[i] * torch.exp(lam * li[i])).clamp(max=1e8)
                    gdro_q.div_(gdro_q.sum())
                loss = sum(gdro_q[i].item() * li[i] for i in doms)  # detached weights
            else:
                if label_correct:
                    # GLS (A4): per-sample importance weight w_i = pi*(y_i)/pi_{d_i}(y_i) on the CE /
                    # H(Y|Z) term. reduction='none' then weighted mean; reduces EXACTLY to the plain
                    # F.cross_entropy when label_correct is off (wb==1), so legacy 'dual'/'balance' are untouched.
                    per = F.cross_entropy(logits, yb, weight=ce_weight, reduction="none")
                    ce_q = (wb * per).sum() / wb.sum().clamp(min=1e-8)
                else:
                    ce_q = F.cross_entropy(logits, yb, weight=ce_weight)   # balanced (BER) if balance=True
                loss = ce_q
                if beta > 0 and getattr(backbone, "last_kl", None) is not None:
                    loss = loss + beta * backbone.last_kl   # VIB: + beta * E KL(q(z|x)||N(0,I)) >= beta*I(X;Z)
                    if last_epoch:
                        diag["inloop_reg"].append(_scalar(backbone.last_kl))
                if is_iib:                               # I(Y;D|Z) = CE_q - CE_h (predictor invariance)
                    loss = loss + lam_t * (ce_q - post.iib_ce_h(z, yb, db))
                if is_dual:                              # DUAL: encoder I(Z;D|Y) + decoder I(Y;D|Z)
                    # decoder term uses the SEPARATE frozen probe q_yz (report §8.1) so gamma does not
                    # silently rescale the task CE; r_dec = H(Y|Z)-H(Y|Z,D) via post.dec_cmi.
                    if rw_dual:
                        # Route B (reweighted-dual): apply the GLS weight w_i=pi*(y_i)/pi_{d_i}(y_i)
                        # to BOTH CMI estimators so the label-shift component is removed from EACH and
                        # the two terms genuinely decouple (I~(Y;D)=0). Encoder = WEIGHTED-mean KL vs the
                        # post-GLS domain marginal; decoder = WEIGHTED H(Y|Z)-H(Y|Z,D) on the separate probe.
                        r_enc = post.reg("lpc_prior", z, yb, weight=wb, reference="marginal")
                        r_dec = post.dec_cmi(z, yb, db, weight=wb)
                    else:
                        r_enc = post.reg("lpc_prior", z, yb)         # I(Z;D|Y): invariant p(z|y)
                        r_dec = post.dec_cmi(z, yb, db)             # I(Y;D|Z): invariant p(y|z), separate probe
                    loss = loss + lam_t * r_enc + gamma_t * r_dec
                    if last_epoch:
                        diag["inloop_reg"].append(_scalar(r_enc)); diag.setdefault("inloop_dec", []).append(_scalar(r_dec))
                if is_dualc:                             # ROUTE C: GLS reweight -> encoder CMI + GATED residual decoder
                    r_enc = post.reg("lpc_prior", z, yb, weight=wb, reference="marginal")   # vs domain marginal p(D)
                    r_dec_res = post.dec_cmi_residual(z, yb, db, weight=wb)   # CE(h0)-CE(h): boundary-change only
                    r_dec_gated = F.relu(r_dec_res - dec_margin)             # gate: penalize only above the null margin
                    loss = loss + lam_t * r_enc + gamma_t * r_dec_gated
                    if last_epoch:
                        diag["inloop_reg"].append(_scalar(r_enc))
                        diag.setdefault("inloop_dec", []).append(_scalar(r_dec_res))
                if is_dualpc or is_dualpc_hinge:
                    # AAAI candidate objective:
                    #   task CE on the empirical source risk (or GLS CE only when --label_correct is explicit)
                    #   + λ · I~(Z;D|Y)          -> aligns the reference mixture P(Z) without erasing labels
                    #   + γ · [JS(h_full,h0)-τ]+ -> aligns P(Y|Z) only when domain-conditioned predictions
                    #                                  exceed the intercept/calibration-only null model
                    # This is the factorized "optimize P(z) and P(y|Z)" counterpart to naive dual-CMI.
                    r_z = post.reg("lpc_prior", z, yb, weight=wb, reference="marginal")
                    r_z_loss = F.relu(r_z - z_margin) if is_dualpc_hinge else r_z
                    r_dec_res = post.dec_cmi_residual(z, yb, db, weight=wb)   # CE residual diagnostic
                    r_dec_loss = post.dec_js_residual(z, db, weight=wb)       # distributional training loss
                    r_dec_gated = dec_scale * F.relu(r_dec_loss - dec_margin)
                    loss = loss + lam_t * r_z_loss + gamma_t * r_dec_gated
                    if last_epoch:
                        diag["inloop_reg"].append(_scalar(r_z))
                        diag.setdefault("inloop_reg_loss", []).append(_scalar(r_z_loss))
                        diag.setdefault("inloop_dec", []).append(_scalar(r_dec_res))
                        diag.setdefault("inloop_dec_loss", []).append(_scalar(r_dec_loss))
                if is_dualpc_marginal:
                    # Direct marginal P(Z) penalty ablation. CPU synthetic validation showed this can hurt
                    # target accuracy and source-only selection, so it is retained as a negative/control variant.
                    r_z = post.reg("marginal", z, yb, weight=wb, reference="ref_marginal")
                    r_dec_res = post.dec_cmi_residual(z, yb, db, weight=wb)
                    r_dec_loss = post.dec_js_residual(z, db, weight=wb)
                    r_dec_gated = F.relu(r_dec_loss - dec_margin)
                    loss = loss + lam_t * r_z + gamma_t * r_dec_gated
                    if last_epoch:
                        diag["inloop_reg"].append(_scalar(r_z))
                        diag.setdefault("inloop_dec", []).append(_scalar(r_dec_res))
                        diag.setdefault("inloop_dec_loss", []).append(_scalar(r_dec_loss))
                if uses_cmi:
                    r = post.reg(cmi_method, z, yb)
                    loss = loss + lam_t * r
                    if last_epoch:
                        diag["inloop_reg"].append(_scalar(r))
                if uses_fmca:                            # Route 2: FMCA(Z,S) closed-form on z
                    r = fmca_reg(method, z, yb, db, n_cls, n_dom)
                    loss = loss + lam_t * r
                    if last_epoch:
                        diag["inloop_reg"].append(_scalar(r))
                if uses_supcon:
                    loss = loss + gamma * sup_con_loss(z, yb, db, cross_domain=True)
                if is_pen:
                    pen = {"coral": lambda: dgp.coral(z, db, n_dom),
                           "mmd": lambda: dgp.mmd(z, db, n_dom),
                           "irm": lambda: dgp.irm(logits, yb, db, n_dom),
                           "vrex": lambda: dgp.vrex(logits, yb, db, n_dom),
                           "chsic": lambda: dgp.chsic(z, yb, db, n_cls, n_dom),
                           "scldgn": lambda: dgp.scldgn(z, yb, db, n_cls, n_dom)}[method]()
                    loss = loss + lam_t * pen
                if is_adv:
                    # canonical DANN: discriminator loss at weight 1; the GRL coefficient (lam_t,
                    # i.e. lam warmed up) scales ONLY the encoder's reversed gradient. lam is the
                    # adversarial-strength knob. (The old code also multiplied the term by lam ->
                    # encoder gradient ~ lam*lam_t = lam^2; at lam=1 it coincided with this canonical
                    # form, so the lam=1 runs are unaffected; only lam!=1 sweep points were distorted.)
                    loss = loss + dgp.adv_penalty(disc, z, yb, db, n_cls, method == "cdann", alpha=lam_t)
            opt_main.zero_grad(); loss.backward(); opt_main.step()
        sched.step()
        if _es_active:                                # P3-E: source-only validation + best-epoch tracking
            _vprob = predict(backbone, Xval, device)  # SOURCE held-out domain only; sets backbone.eval()
            backbone.train()
            _vp = _vprob.argmax(1)
            last_vbacc = _balanced_acc(yval, _vp, n_cls)
            _vce = float(-np.log(np.clip(_vprob[np.arange(len(yval)), yval], 1e-8, 1.0)).mean())
            better = (last_vbacc > best_vbacc + 1e-9) or \
                     (abs(last_vbacc - best_vbacc) <= 1e-9 and _vce < best_vce - 1e-9)
            if better:
                best_vbacc, best_vce, best_ep = last_vbacc, _vce, ep
                best_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
        if log_every and (ep + 1) % log_every == 0:
            print(f"  ep {ep+1}/{epochs} lam_t={lam_t:.3f} loss={loss.item():.4f}", flush=True)
    es_out = {}
    if _es_active:                                    # P3-E: finalize source-only early stopping
        _tr_pred = predict(backbone, Xtr, device).argmax(1); backbone.train()   # final-epoch source-train bAcc
        es_out = dict(source_val_subjects=es_val_domains, best_epoch=int(best_ep),
                      best_source_val_bacc=float(best_vbacc), final_val_source_bacc=float(last_vbacc),
                      final_train_source_bacc=float(_balanced_acc(np.asarray(ytr), _tr_pred, n_cls)))
        if best_state is not None:
            backbone.load_state_dict(best_state)      # restore best-source-val weights before returning
    out = dict(stepA_dom_acc=diag["stepA_dom_correct"] / max(1, diag["stepA_dom_total"]),
               inloop_reg=float(np.mean(diag["inloop_reg"])) if diag["inloop_reg"] else 0.0,
               sampler=diag["sampler"],
               dec_margin=float(dec_margin),
               z_margin=float(z_margin),
               dec_scale=float(dec_scale))
    if uses_graphcmi:
        # User-facing CIGL diagnostics: report the three weights under their spec names
        # (lambda_g/lambda_node/lambda_edge), NOT the internal lam/gamma/lam_edge, plus the
        # per-component held-in leakage breakdown (loss_ce / reg_graph / reg_node / reg_edge).
        _mean = lambda k: float(np.mean(diag[k])) if diag[k] else 0.0
        graph_dom_acc = diag["stepA_graph_correct"] / max(1, diag["stepA_graph_total"])
        out.update(lambda_g=float(lam), lambda_node=float(gamma), lambda_edge=float(lam_edge),
                   loss_ce=_mean("inloop_ce"), reg_graph=_mean("inloop_reg_graph"),
                   reg_node=_mean("inloop_reg_node"), reg_edge=_mean("inloop_reg_edge"),
                   # spec-named per-head Step-A critic quality (replaces the undefined legacy 0.0)
                   stepA_graph_dom_acc=graph_dom_acc,
                   stepA_node_dom_acc=diag["stepA_node_correct"] / max(1, diag["stepA_node_total"]),
                   stepA_edge_dom_acc=diag["stepA_edge_correct"] / max(1, diag["stepA_edge_total"]),
                   stepA_graph_loss=_mean("stepA_graph_loss"), stepA_node_loss=_mean("stepA_node_loss"),
                   stepA_edge_loss=_mean("stepA_edge_loss"),
                   stepA_dom_acc=graph_dom_acc)   # override legacy field (was a fake 0.0 for graphcmi)
    if uses_graphdualpc:
        # Paper-facing Graph-DualCMI diagnostics: four explicit weights + GLS encoder terms + decoder residuals
        # + Step-A critic quality (probe-underfit check: reg_*_gls is only interpretable if the critics fit).
        _mean = lambda k: float(np.mean(diag[k])) if diag.get(k) else 0.0
        out.update(lambda_g=float(lam), lambda_node=float(beta), lambda_edge=float(lam_edge),
                   gamma_dec=float(gamma),
                   reg_graph_gls=_mean("inloop_reg_graph"), reg_node_gls=_mean("inloop_reg_node"),
                   reg_edge_gls=_mean("inloop_reg_edge"),
                   dec_js_res=_mean("inloop_dec_js"), dec_ce_res=_mean("inloop_dec_ce"),
                   loss_ce=_mean("inloop_ce"),
                   # P3-D decoder-activation diagnostics (target: loss_dec_over_ce ~ 1-10%; <0.1% == dormant)
                   dec_js_res_raw=_mean("inloop_dec_js"),
                   dec_js_res_scaled=_mean("inloop_dec_js_scaled"),
                   loss_dec=_mean("inloop_loss_dec"),
                   loss_dec_over_ce=(_mean("inloop_loss_dec") / max(_mean("inloop_ce"), 1e-8)),
                   dec_gate_active_frac=_mean("inloop_dec_gate"),
                   stepA_graph_dom_acc_gls=diag["stepA_graph_correct"] / max(1, diag["stepA_graph_total"]),
                   stepA_node_dom_acc_gls=diag["stepA_node_correct"] / max(1, diag["stepA_node_total"]),
                   stepA_edge_dom_acc_gls=diag["stepA_edge_correct"] / max(1, diag["stepA_edge_total"]),
                   stepA_graph_loss_gls=_mean("stepA_graph_loss"), stepA_node_loss_gls=_mean("stepA_node_loss"),
                   stepA_edge_loss_gls=_mean("stepA_edge_loss"))
    if "inloop_reg_loss" in diag:
        out["inloop_reg_loss"] = float(np.mean(diag["inloop_reg_loss"]))
    if "inloop_dec" in diag:
        out["inloop_dec"] = float(np.mean(diag["inloop_dec"]))
    if "inloop_dec_loss" in diag:
        out["inloop_dec_loss"] = float(np.mean(diag["inloop_dec_loss"]))
    if es_out:                                        # P3-E source-only early-stopping metadata
        out.update(es_out)
    return backbone, post, out


@torch.no_grad()
def predict(backbone, X, device="cpu", bs=512):
    backbone.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs]).to(device)
        out.append(backbone(xb)[0].softmax(1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def embed(backbone, X, device="cpu", bs=512):
    backbone.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs]).to(device)
        out.append(backbone(xb)[1].cpu().numpy())
    return np.concatenate(out)
