"""Two-step alternating H2-CMI trainer.

Per batch:
  Step A  fit the hierarchical conditional-domain critics on DETACHED z_c.
  Step B  update encoder + hybrid head with
            L = hybrid(CE + density NLL + JS consistency)
              + sum_j lambda_j * I_hat_j            (hierarchical CMI budget, P0-2)
              + align * reference-prior marginal     (P0-4 corrected alignment)
              + rho  * disentangle(z_c, z_n)
              + ssl  * (masked recon + VICReg)       (anti-collapse)
              + trust * canonicaliser deviation
  per epoch  primal-dual update lambda_j <- [lambda_j + eta(I_hat_j - eps_j)]_+.

Honors the review's code-correctness notes: seeds set BEFORE model construction; single
forward per batch (no double BatchNorm update); drop_last defaults False so rare
domain-class cells survive; the effective training distribution (raw source) is declared.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from h2cmi.config import H2Config
from h2cmi.domains import DomainDAG, DomainLabels
from h2cmi.models.encoder import H2Encoder
from h2cmi.density.student_t_mixture import HybridHead
from h2cmi.cmi.hierarchical import HierarchicalCMI, DualBudget
from h2cmi.align.reference_marginal import ReferenceMarginalAlignment
from h2cmi.disentangle.penalties import DisentangleLoss
from h2cmi.ssl.aux import SSLAux


def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reference_prior(y: np.ndarray, n_classes: int, mode: str) -> np.ndarray:
    if mode == "source_marginal":
        c = np.bincount(y, minlength=n_classes).astype(np.float64)
        return c / c.sum()
    return np.full(n_classes, 1.0 / n_classes)


class H2Model(nn.Module):
    """Encoder + hybrid (discriminative + class-conditional density) head."""

    def __init__(self, cfg: H2Config, pi_star: np.ndarray):
        super().__init__()
        self.cfg = cfg
        self.encoder = H2Encoder(cfg.encoder, cfg.n_classes)
        self.head = HybridHead(cfg.encoder.z_c_dim, cfg.n_classes, cfg.density,
                               reference_prior=torch.tensor(pi_star, dtype=torch.float32))

    def forward(self, x):
        z_c, z_n = self.encoder(x)
        return z_c, z_n, self.head.disc_logits(z_c)

    @torch.no_grad()
    def embed(self, X, device="cpu", bs=256):
        self.eval()
        out = []
        for i in range(0, len(X), bs):
            xb = torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=device)
            out.append(self.encoder(xb)[0].cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.cfg.encoder.z_c_dim))

    @torch.no_grad()
    def predict_proba(self, X, device="cpu", prior=None, mode="blend", bs=256):
        self.eval()
        out = []
        prior_t = None if prior is None else torch.as_tensor(prior, dtype=torch.float32, device=device)
        for i in range(0, len(X), bs):
            xb = torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=device)
            z_c, _ = self.encoder(xb)
            out.append(self.head.predict_proba(z_c, prior_t, mode).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.cfg.n_classes))


def train_h2(X: np.ndarray, y: np.ndarray, domains: DomainLabels, dag: DomainDAG,
             cfg: H2Config, align_factor: str = "site", verbose: bool = False):
    """Train H2-CMI. Returns (model, hcmi, dual, history)."""
    set_seed(cfg.train.seed)                       # seeds BEFORE model construction
    dev = cfg.train.device
    n_cls = cfg.n_classes
    pi_star = reference_prior(y, n_cls, cfg.align.reference_prior)

    model = H2Model(cfg, pi_star).to(dev)
    hcmi = HierarchicalCMI(cfg.encoder.z_c_dim, n_cls, dag, domains, y, cfg.cmi).to(dev)
    dual = DualBudget(hcmi.budgets, cfg.cmi)
    align = ReferenceMarginalAlignment(cfg.align, n_cls, pi_star) if cfg.align.enabled else None
    n_dom_align = int(domains.factor(align_factor).max()) + 1
    disent = DisentangleLoss(cfg.encoder.z_c_dim, cfg.encoder.z_n_dim, n_cls, n_dom_align,
                             cfg.disentangle.eta_zn_y, cfg.disentangle.kappa_zn_d,
                             cfg.disentangle.method).to(dev) if cfg.disentangle.enabled else None
    sslaux = SSLAux(cfg.encoder.z_c_dim, cfg.encoder.n_chans, cfg.encoder.n_times,
                    mask_ratio=cfg.ssl.mask_ratio, vicreg=cfg.ssl.vicreg,
                    masked_recon=cfg.ssl.masked_recon).to(dev) if cfg.ssl.enabled else None

    main_params = list(model.parameters())
    if disent is not None:
        main_params += list(disent.parameters())
    if sslaux is not None:
        main_params += list(sslaux.parameters())
    opt_main = torch.optim.AdamW(main_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    opt_critic = torch.optim.Adam(hcmi.parameters(), lr=cfg.cmi.critic_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=cfg.train.epochs)

    align_dom = domains.factor(align_factor).astype(np.int64)
    idx_all = np.arange(len(y))
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                       torch.as_tensor(y, dtype=torch.long),
                       torch.as_tensor(align_dom, dtype=torch.long),
                       torch.as_tensor(idx_all, dtype=torch.long))
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True,
                    drop_last=cfg.train.drop_last)

    history = []
    for ep in range(cfg.train.epochs):
        model.train()
        if disent: disent.train()
        if sslaux: sslaux.train()
        ep_ihat = {f: [] for f in hcmi.factors}
        ep_log = dict(hybrid=[], cmi=[], align=[], disent=[], ssl=[])
        lam_warm = min(1.0, (ep + 1) / max(1, cfg.train.warmup))
        for xb, yb, db, ib in dl:
            xb, yb, db = xb.to(dev), yb.to(dev), db.to(dev)
            ib_np = ib.numpy()
            lev, pk = hcmi.batch_context(domains, ib_np)

            # ---- single forward (grad) ----
            z_c, z_n, logits = model(xb)

            # ---- Step A: fit critics on detached z_c ----
            z_det = z_c.detach()
            for _ in range(cfg.cmi.critic_inner):
                la = hcmi.critic_loss(z_det, yb, lev, pk)
                opt_critic.zero_grad(); la.backward(); opt_critic.step()

            # ---- Step B: encoder + head + penalties ----
            loss, info = model.head.loss(z_c, yb)
            ep_log["hybrid"].append(float(loss.detach()))

            terms = hcmi.estimate(z_c, yb, lev, pk)         # frozen critics (opt_main excludes them)
            lambdas = dual.as_tensors(dev)
            cmi_pen = hcmi.total_penalty(terms, lambdas) * lam_warm
            loss = loss + cmi_pen
            for f, v in terms.items():
                ep_ihat[f].append(float(v.detach()))
            ep_log["cmi"].append(float(cmi_pen.detach()))

            if align is not None:
                a, _ = align(z_c, yb, db, n_cls)
                loss = loss + cfg.align.weight * a
                ep_log["align"].append(float(a.detach()))
            if disent is not None:
                dloss, _ = disent(z_c, z_n, yb, db)
                loss = loss + cfg.disentangle.rho * dloss
                ep_log["disent"].append(float(dloss.detach()))
            if sslaux is not None:
                sloss, _ = sslaux(z_c, xb)
                loss = loss + cfg.ssl.weight * sloss
                ep_log["ssl"].append(float(sloss.detach()))
            loss = loss + 1e-3 * model.encoder.canon_penalty()

            opt_main.zero_grad(); loss.backward()
            if cfg.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(main_params, cfg.train.grad_clip)
            opt_main.step()
            if cfg.density.ema > 0:
                model.head.density.ema_update(z_c.detach(), yb, cfg.density.ema)

        # ---- per-epoch primal-dual update ----
        ihat_mean = {f: float(np.mean(v)) if v else 0.0 for f, v in ep_ihat.items()}
        dual.step(ihat_mean)
        sched.step()
        rec = dict(epoch=ep, ihat=ihat_mean, lambda_=dual.state(),
                   **{k: (float(np.mean(v)) if v else 0.0) for k, v in ep_log.items()})
        history.append(rec)
        if verbose or cfg.train.log_every and (ep + 1) % cfg.train.log_every == 0:
            print(f"ep {ep+1}/{cfg.train.epochs} hybrid={rec['hybrid']:.3f} "
                  f"cmi={rec['cmi']:.3f} align={rec['align']:.3f} "
                  f"Ihat={ {f: round(ihat_mean[f],3) for f in hcmi.factors} } "
                  f"lam={ {f: round(dual.state()[f],3) for f in hcmi.factors} }", flush=True)
    return model, hcmi, dual, history
