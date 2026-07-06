"""CITA — target-unlabeled offline transductive CMI adaptation (CITA_01).

NEW INFORMATION REGIME (distinct from the closed source-only CMI line, see docs/CIGL_70): the held-out LOSO
target subject's UNLABELED X is available during an offline adaptation phase; the target LABELS are FORBIDDEN
everywhere (training, adaptation, model selection, pseudo-label selection, early stopping, hyperparameter
selection) and are touched only for the final reported metric.

Three methods (minimal attribution set) — all start from the SAME source-ERM model M0 (trained separately via
train_model(method='erm')):
  - erm_no_adapt : M0 evaluated on target, no adaptation (the pre-adaptation baseline).
  - tta_control  : adapt a copy of M0 with  L = CE(source_replay) + tau*H(p_target)
                                              + mu*KL(mean p_target || source_label_prior).
                   (target-unlabeled entropy/confidence adaptation + anti-collapse label balance;
                   NOT a CMI method -- the control that isolates "is adaptation itself useful?")
  - cita_cmi     : tta_control  +  lam * L_cond_domain, a label-conditional source/target domain-confusion
                   (posterior-KL proxy): a conditional domain posterior q(D|z,y) is fit to distinguish
                   source(D=0) vs target(D=1) GIVEN y, and the encoder is penalized so q(D|z,y) -> the domain
                   prior (i.e. D _||_ z | Y). Y for source = one-hot true label; Y for target = DETACHED soft
                   pseudo-label softmax(logits_t) -- no confidence threshold, no target label.

The primary attribution comparison is CITA-CMI vs TTA-Control (does the CMI term add anything beyond
target-unlabeled adaptation?), not CITA vs ERM.

FIREWALL: the adaptation entry point `adapt()` takes ONLY (model, Xs, ys, Xt, ...). It has NO target-label
parameter, so target y CANNOT enter adaptation by construction. Target pseudo-labels are recomputed from the
current model on target X and DETACHED before any use. Source replay CE uses source labels only.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CITA_METHODS = ("erm_no_adapt", "tta_control", "cita_cmi")
ADAPT_METHODS = ("tta_control", "cita_cmi")     # methods that actually run the adaptation loop


class ConditionalDomainPosterior(nn.Module):
    """q(D in {source=0, target=1} | z, y): small MLP over [z, y_probs] -> logit P(target). Estimates
    label-conditional source/target distinguishability; the CITA penalty drives its output toward the domain
    prior so that D is uninformative given (z, Y)."""
    def __init__(self, z_dim, n_cls, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(int(z_dim) + int(n_cls), hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z, y_probs):
        return self.net(torch.cat([z, y_probs], dim=1)).squeeze(1)      # [B] logit P(target)


def _entropy(p):
    return -(p * p.clamp_min(1e-8).log()).sum(1).mean()


def _source_label_prior(ys, n_cls):
    c = torch.bincount(torch.as_tensor(ys).long(), minlength=int(n_cls)).float()
    return (c / c.sum()).clamp_min(1e-6)


def _n_cls_of(model):
    return int(model.head.out_features)


def adapt(model, Xs, ys, Xt, method, *, steps=50, bs=64, lr=1e-3, tau=1.0, mu=1.0,
          lam_cita=0.010, cond_inner=1, device="cpu", seed=0):
    """Offline transductive target-unlabeled adaptation of `model` (a source-ERM model). Returns
    (adapted_model, diag). `erm_no_adapt` returns the model unchanged. NO target labels are accepted or used.

    Firewall by construction: signature has no target-y; target pseudo-labels are detached; source CE uses ys
    only; target entropy uses Xt only. `adaptation_uses_all_target_X` (offline transductive): every step samples
    from the full Xt."""
    is_adapt = method in ADAPT_METHODS
    # adaptation budget (steps/lr/bs/tau/mu are SHARED across tta_control & cita_cmi; only lambda_cita differs)
    budget = dict(adapt_steps=int(steps), adapt_lr=float(lr), adapt_batch_size_source=int(bs),
                  adapt_batch_size_target=int(bs), tau_entropy=float(tau), mu_label_balance=float(mu),
                  lambda_cita=(float(lam_cita) if method == "cita_cmi" else 0.0), cond_inner=int(cond_inner),
                  source_label_prior_source="source_train_only", pseudo_label_mode="detached_soft")
    # model-mode policy during adaptation (IDENTICAL for tta_control & cita_cmi: same model.train() code path)
    mode_policy = dict(adaptation_train_mode=is_adapt, dropout_active_during_adapt=is_adapt,
                       batchnorm_updates_during_adapt=is_adapt)
    diag = {"method": method, "adapted": is_adapt, "adaptation_mode": "offline_transductive",
            "adaptation_uses_all_target_X": True, "target_y_used": False,
            "cond_domain_active": method == "cita_cmi", "budget": budget, "mode_policy": mode_policy}
    if method == "erm_no_adapt":
        model.eval()                                              # return inference-ready (deterministic)
        return model, diag
    if method not in ADAPT_METHODS:
        raise ValueError(f"unknown CITA method '{method}'")

    n_cls = _n_cls_of(model)
    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    Xs = np.asarray(Xs, dtype="float32"); ys = np.asarray(ys).astype("int64"); Xt = np.asarray(Xt, dtype="float32")
    src_prior = _source_label_prior(ys, n_cls).to(device)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    qcd = opt_q = None
    if method == "cita_cmi":
        qcd = ConditionalDomainPosterior(model.z_dim, n_cls).to(device)
        opt_q = torch.optim.Adam(qcd.parameters(), lr=lr)

    ce_l = ent_l = bal_l = cond_l = None
    for step in range(int(steps)):
        si = rng.integers(0, len(Xs), size=min(bs, len(Xs)))
        ti = rng.integers(0, len(Xt), size=min(bs, len(Xt)))               # samples the FULL target X (transductive)
        xs = torch.as_tensor(Xs[si], device=device); yss = torch.as_tensor(ys[si], device=device)
        xt = torch.as_tensor(Xt[ti], device=device)
        logits_s, z_s = model(xs)
        logits_t, z_t = model(xt)
        ce = F.cross_entropy(logits_s, yss)                                # SOURCE replay CE (source labels only)
        p_t = F.softmax(logits_t, 1)
        ent = _entropy(p_t)                                                # TARGET entropy (target X only)
        mean_pt = p_t.mean(0).clamp_min(1e-8)
        bal = (mean_pt * (mean_pt.log() - src_prior.log())).sum()          # KL(mean p_target || source prior)
        loss = ce + tau * ent + mu * bal
        cond = None
        if method == "cita_cmi":
            y_s = F.one_hot(yss, n_cls).float()                            # source: one-hot TRUE label
            y_t = p_t.detach()                                             # target: DETACHED soft pseudo-label
            z_all = torch.cat([z_s, z_t], 0)
            y_all = torch.cat([y_s, y_t], 0).detach()
            d_all = torch.cat([torch.zeros(len(z_s), device=device),
                               torch.ones(len(z_t), device=device)])       # 0 source, 1 target
            for _ in range(int(cond_inner)):                               # inner: fit q(D|z,y) on detached z
                lq = F.binary_cross_entropy_with_logits(qcd(z_all.detach(), y_all), d_all)
                opt_q.zero_grad(); lq.backward(); opt_q.step()
            q_t = torch.sigmoid(qcd(z_all, y_all))                         # outer: z carries grad, q weights fixed
            pi = d_all.mean().detach().clamp(1e-4, 1 - 1e-4)
            cond = (q_t * (q_t / pi).clamp_min(1e-8).log()
                    + (1 - q_t) * ((1 - q_t) / (1 - pi)).clamp_min(1e-8).log()).mean()   # KL(q(D|z,y)||prior)
            loss = loss + lam_cita * cond
        if step == 0:                                                      # BEFORE adaptation (initial M0 forward)
            ce_0, ent_0, bal_0 = float(ce), float(ent), float(bal)
        if step == int(steps) - 1:                                         # AFTER adaptation — loss-scale diagnostics
            ce_l, ent_l, bal_l = float(ce), float(ent), float(bal)         # (computed BEFORE the step frees graph)
            cond_l = float(cond) if cond is not None else None
            total_l = float(loss)
            lam_cond_l = float(lam_cita * cond) if cond is not None else 0.0
            frac_l = (lam_cond_l / total_l) if (cond is not None and abs(total_l) > 1e-12) else 0.0
            cond_gn = None
            if method == "cita_cmi" and cond is not None:                  # grad-norm of the CMI term on the encoder
                gs = torch.autograd.grad(lam_cita * cond, [p for p in model.parameters() if p.requires_grad],
                                         retain_graph=True, allow_unused=True)
                cond_gn = float(torch.sqrt(sum((g.detach() ** 2).sum() for g in gs if g is not None)))
        opt.zero_grad(); loss.backward(); opt.step()
    diag.update(source_replay_ce_before=ce_0, source_replay_ce_after=ce_l,
                target_entropy_before=ent_0, target_entropy_after=ent_l,
                target_label_balance_before=bal_0, target_label_balance_after=bal_l,
                final_source_ce=ce_l, final_target_entropy=ent_l, final_label_balance_kl=bal_l,
                final_cond_domain=cond_l, lambda_times_cond_domain=lam_cond_l,
                final_total_loss=total_l, cond_domain_fraction_of_total_loss=frac_l,
                cond_domain_gradient_norm=cond_gn, lambda_cita_used=float(lam_cita), steps=int(steps),
                # legacy aliases kept for any earlier reader
                final_ce=ce_l, final_entropy=ent_l, final_balance=bal_l)
    model.eval()                                                  # return inference-ready (deterministic; dropout off)
    return model, diag
