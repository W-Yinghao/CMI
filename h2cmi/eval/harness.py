"""Three-setting evaluation harness (review section 10.1): strict DG, offline transductive
TTA, online streaming TTA -- reported separately, never under one 'DG accuracy' header.

Also trains the source-only safety gate via inner leave-one-source-domain-out, and applies
it to gate offline TTA (adapt vs identity fallback per target domain), reporting the
review's selective-risk panel (coverage, avoided-harm, missed-benefit, AUROC).
"""
from __future__ import annotations

import numpy as np
import torch

from h2cmi.config import H2Config
from h2cmi.domains import DomainLabels
from h2cmi.tta.class_conditional import ClassConditionalTTA
from h2cmi.gate.safety_gate import SafetyGate, gate_features
from h2cmi.eval.metrics import metric_panel, panel_delta, cluster_bootstrap_ci
from sklearn.metrics import balanced_accuracy_score


# ----------------------------------------------------------------- prediction helpers
def _json_safe(x):
    """Coerce numpy scalars / arrays (and nested containers) to plain JSON-serialisable types."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):        # numpy scalars (floating / integer / bool_) -> python
        return x.item()
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    return x


def _embed(model, X, device):
    return torch.as_tensor(model.embed(X, device=device), dtype=torch.float32, device=device)


@torch.no_grad()
def _predict_generative(model, U: torch.Tensor, prior: np.ndarray) -> np.ndarray:
    log_prior = torch.log(torch.as_tensor(prior, dtype=torch.float32, device=U.device).clamp_min(1e-8))
    return model.head.density.class_posterior(U, log_prior).cpu().numpy()


@torch.no_grad()
def _predict_transform(model, U: torch.Tensor, transform, pi_T: np.ndarray) -> np.ndarray:
    z = transform.apply(U)
    log_prior = torch.log(torch.as_tensor(pi_T, dtype=torch.float32, device=U.device).clamp_min(1e-8))
    return model.head.density.class_posterior(z, log_prior).cpu().numpy()


# ----------------------------------------------------------------- strict DG
def evaluate_strict_dg(model, X, y, domain, device="cpu", prior=None, mode="blend") -> dict:
    """No target data touched: inference only (the strict-DG unit is the unseen site)."""
    proba = model.predict_proba(X, device=device, prior=prior, mode=mode)
    panel = metric_panel(proba, y, domain)
    panel["setting"] = "strict_dg"
    return panel


# ----------------------------------------------------------------- safety gate training
def train_safety_gate(model, X, y, site, cfg: H2Config, pseudo_unit_levels: np.ndarray,
                      source_prior: np.ndarray, device="cpu") -> tuple[SafetyGate, dict]:
    """Inner leave-one-source-domain-out: for each pseudo-target source domain, run TTA,
    measure the TRUE gain (we have labels), collect (gate features, gain), fit the gate."""
    tta = ClassConditionalTTA(model.head.density, source_prior, cfg.tta, cfg.n_classes, device)
    feats, gains = [], []
    for u in np.unique(pseudo_unit_levels):
        m = pseudo_unit_levels == u
        if m.sum() < cfg.tta.min_target:
            continue
        U = _embed(model, X[m], device)
        yu = y[m]
        res = tta.fit(U, pseudo_labels=_predict_generative(model, U, source_prior).argmax(1))
        p_id = _predict_generative(model, U, source_prior)
        p_ad = _predict_transform(model, U, res.transform, res.pi_T)
        gain = balanced_accuracy_score(yu, p_ad.argmax(1)) - balanced_accuracy_score(yu, p_id.argmax(1))
        feats.append(gate_features(res.diagnostics)); gains.append(gain)
    info = dict(n_pseudo=len(gains))
    gate = SafetyGate(cfg.gate.model, cfg.gate.harm_delta, cfg.gate.risk_threshold, cfg.gate.min_evidence)
    if len(gains) >= 2:
        feats = np.stack(feats); gains = np.asarray(gains)
        gate.fit(feats, gains)
        info["harm_metrics"] = gate.harm_detection_metrics(feats, gains)
        info["mean_pseudo_gain"] = float(gains.mean())
    return gate, info


# ----------------------------------------------------------------- offline transductive TTA
def evaluate_offline_tta(model, X, y, domain, cfg: H2Config, source_prior: np.ndarray,
                         gate: SafetyGate | None = None, device="cpu") -> dict:
    """Per target domain: fit class-conditional TTA, optionally GATE it, compare to identity.

    Returns identity & adapted panels, per-domain gain, a domain-clustered bootstrap CI on
    the gain, and the selective-adaptation panel (coverage / avoided-harm / missed-benefit).
    """
    tta = ClassConditionalTTA(model.head.density, source_prior, cfg.tta, cfg.n_classes, device)
    proba_id = np.zeros((len(y), cfg.n_classes))
    proba_ad = np.zeros((len(y), cfg.n_classes))
    proba_sel = np.zeros((len(y), cfg.n_classes))
    per_dom_gain, per_dom_decision = {}, {}
    per_dom_pi_T, per_dom_diag = {}, {}          # evidence exported for the audited eval bridge
    for d in np.unique(domain):
        m = domain == d
        U = _embed(model, X[m], device)
        p_id = _predict_generative(model, U, source_prior)
        res = tta.fit(U, pseudo_labels=p_id.argmax(1))
        p_ad = _predict_transform(model, U, res.transform, res.pi_T)
        per_dom_pi_T[int(d)] = _json_safe(res.pi_T)                 # estimated target prior pi_T
        per_dom_diag[int(d)] = _json_safe(res.diagnostics)         # TTA diagnostics (density-NLL, etc.)
        proba_id[m] = p_id; proba_ad[m] = p_ad
        # gate decision (selective)
        do_adapt = res.adapted
        if gate is not None and res.adapted:
            g = gate_features(res.diagnostics)
            do_adapt = gate.should_adapt(g, res.diagnostics.get("delta_density_nll"))
        proba_sel[m] = p_ad if do_adapt else p_id
        per_dom_decision[int(d)] = bool(do_adapt)
        gid = balanced_accuracy_score(y[m], p_id.argmax(1))
        gad = balanced_accuracy_score(y[m], p_ad.argmax(1))
        per_dom_gain[int(d)] = float(gad - gid)

    panel_id = metric_panel(proba_id, y, domain); panel_id["setting"] = "offline_tta_identity"
    panel_ad = metric_panel(proba_ad, y, domain); panel_ad["setting"] = "offline_tta_adapt"
    panel_sel = metric_panel(proba_sel, y, domain); panel_sel["setting"] = "offline_tta_selective"
    boot = cluster_bootstrap_ci(per_dom_gain)
    # selective-risk panel
    adapted_doms = [d for d, v in per_dom_decision.items() if v]
    skipped = [d for d, v in per_dom_decision.items() if not v]
    avoided_harm = float(np.mean([max(0.0, -per_dom_gain[d]) for d in skipped])) if skipped else 0.0
    missed_benefit = float(np.mean([max(0.0, per_dom_gain[d]) for d in skipped])) if skipped else 0.0
    selective_gain = float(np.mean([per_dom_gain[d] for d in adapted_doms])) if adapted_doms else 0.0
    return dict(identity=panel_id, adapt=panel_ad, selective=panel_sel,
                delta_adapt=panel_delta(panel_ad, panel_id),
                delta_selective=panel_delta(panel_sel, panel_id),
                per_domain_gain=per_dom_gain, gain_bootstrap=boot,
                gate_decisions=per_dom_decision,
                per_domain_pi_T=per_dom_pi_T,                       # exported evidence (audit bridge)
                per_domain_tta_diagnostics=per_dom_diag,
                selective_risk=dict(coverage=len(adapted_doms) / max(1, len(per_dom_decision)),
                                    avoided_harm=avoided_harm, missed_benefit=missed_benefit,
                                    selective_gain=selective_gain))


# ----------------------------------------------------------------- online streaming TTA
def evaluate_online_tta(model, X, y, domain, cfg: H2Config, source_prior: np.ndarray,
                        device="cpu", batch=32) -> dict:
    """Per target domain, stream trials in order; predict each batch under the running
    (EMA) prior/transform BEFORE seeing it (no peeking at future samples)."""
    tta = ClassConditionalTTA(model.head.density, source_prior, cfg.tta, cfg.n_classes, device)
    proba = np.zeros((len(y), cfg.n_classes))
    for d in np.unique(domain):
        m = np.where(domain == d)[0]
        U = _embed(model, X[m], device)
        pi_run = source_prior.copy()
        from h2cmi.tta.class_conditional import Transform
        T = Transform(U.shape[1], "diag_affine", device=device)
        for i in range(0, len(m), batch):
            sl = slice(i, i + batch)
            Ub = U[sl]
            proba[m[sl]] = _predict_transform(model, Ub, T, pi_run)   # predict BEFORE update
            with torch.no_grad():                                     # then EMA-update prior
                z = T.apply(Ub)
                log_prior = torch.log(torch.as_tensor(pi_run, dtype=torch.float32, device=device).clamp_min(1e-8))
                r = model.head.density.class_posterior(z, log_prior)
                counts = r.sum(0).cpu().numpy()
                pi_b = (counts + 1e-3) / (counts.sum() + 1e-3 * cfg.n_classes)
                pi_run = cfg.tta.online_ema * pi_run + (1 - cfg.tta.online_ema) * pi_b
    panel = metric_panel(proba, y, domain); panel["setting"] = "online_tta"
    return panel


# ----------------------------------------------------------------- orchestrator
def run_three_settings(model, X_tgt, y_tgt, domain_tgt, cfg: H2Config, source_prior: np.ndarray,
                       X_src=None, y_src=None, gate_pseudo_levels=None, device="cpu") -> dict:
    """Run all three settings (+ optional safety-gate training) on one held-out target set."""
    out = {}
    out["strict_dg"] = evaluate_strict_dg(model, X_tgt, y_tgt, domain_tgt, device, prior=None)
    gate, gate_info = (None, {})
    if cfg.gate.enabled and X_src is not None and gate_pseudo_levels is not None:
        gate, gate_info = train_safety_gate(model, X_src, y_src, None, cfg,
                                            gate_pseudo_levels, source_prior, device)
    out["gate_info"] = gate_info
    out["offline_tta"] = evaluate_offline_tta(model, X_tgt, y_tgt, domain_tgt, cfg,
                                              source_prior, gate, device)
    out["online_tta"] = evaluate_online_tta(model, X_tgt, y_tgt, domain_tgt, cfg, source_prior, device)
    return out
