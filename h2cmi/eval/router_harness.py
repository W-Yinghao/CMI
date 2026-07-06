"""Project B — router integration harness (Step-2E).

First real integration: a frozen H2-CMI model + ClassConditionalTTA + the Step-2D
RefusalFirstRouter. This module does NOT train the model, modify the TTA, or touch the old
harness/gate; it only wires them together and evaluates the router's per-domain decisions.

Hard rules (enforced by construction):
  - Target labels `y` are used ONLY for post-hoc evaluation, AFTER a RouterDecision is produced;
    never to build diagnostics, choose an action, or calibrate a threshold.
  - IDENTITY diagnostics are identity-neutral (delta_density_nll=0, transform_norm=0,
    condition_number=1, pred_disagreement=0) but keep the shared support/prior diagnostics — a
    TTA-evidence failure must not pollute the identity decision.
  - `cmi_residual` (a dead 0.0 field in the TTA diagnostics) is dropped at route time, so the
    router emits OACI_LEAKAGE_RESIDUAL_UNAVAILABLE rather than trusting a fake I(Z;D|Y).
  - Source calibration (support threshold + pseudo harm gains) uses source data only.
"""
from __future__ import annotations

import dataclasses
from collections import Counter

import numpy as np
import torch
from scipy.special import logsumexp
from sklearn.metrics import balanced_accuracy_score

from h2cmi.tta.class_conditional import ClassConditionalTTA
from h2cmi.eval.harness import _embed, _predict_generative, _predict_transform
from h2cmi.eval.metrics import metric_panel, panel_delta
from h2cmi.router.router import RefusalFirstRouter, RouterConfig, RouterAction
from h2cmi.router.features import (
    RouterFeatureConfig, build_router_features, assess_acar_harm_calibration, CalibrationState,
)
from h2cmi.router.reasons import OACIReason, normalize_reasons

_PANEL_KEYS = ("balanced_acc", "macro_f1", "nll", "brier", "ece",
              "worst_domain_bacc", "domain_cvar25")


def _nan_panel(setting: str) -> dict:
    p = {k: float("nan") for k in _PANEL_KEYS}
    p["per_domain_bacc"] = {}
    p["setting"] = setting
    return p


# ------------------------------------------------------------------ prior-decoupled support
def prior_decoupled_density_diagnostics(density, U, source_prior, *, pi_target=None, eps: float = 1e-3) -> dict:
    """Support/OOD diagnostics that separate label-prior shift from genuine support mismatch.

    Uses the frozen class-conditional density only (no TTA, no labels). Returns the shared,
    action-independent support features used by both the identity and offline-TTA bundles.
    """
    with torch.no_grad():
        logp = density.log_prob_all(U).detach().cpu().numpy().astype(np.float64)   # [B, K]
    pi_S = np.asarray(source_prior, dtype=np.float64)
    pi_S = pi_S / pi_S.sum()
    K = pi_S.size
    log_piS = np.log(np.clip(pi_S, 1e-8, None))

    a = logp + log_piS[None, :]
    lse_s = logsumexp(a, axis=1)
    nll_source = float(-np.mean(lse_s))
    r = np.exp(a - lse_s[:, None])                                # responsibilities under source prior

    if pi_target is None:
        pi_hat = (r.sum(0) + eps) / (r.sum() + eps * K)
    else:
        pi_hat = np.asarray(pi_target, dtype=np.float64)
        pi_hat = pi_hat / pi_hat.sum()
    log_piT = np.log(np.clip(pi_hat, 1e-8, None))
    nll_target = float(-np.mean(logsumexp(logp + log_piT[None, :], axis=1)))

    col_sum = r.sum(0)
    col_sq = (r ** 2).sum(0)
    ess_per_class = np.where(col_sq > 0.0, col_sum ** 2 / np.clip(col_sq, 1e-12, None), 0.0)
    return dict(
        ess=float(ess_per_class.min()),
        ood_score=float(nll_source),
        density_nll_source_prior=nll_source,
        density_nll_target_prior=nll_target,
        support_gap=float(nll_source - nll_target),
        prior_shift=float(np.abs(pi_hat - pi_S).sum()),
        min_class_responsibility=float(r.mean(0).min()),
    )


# ------------------------------------------------------------------ source calibration
def calibrate_router_feature_config_from_source(
    model, X_src, y_src, source_pseudo_levels, cfg, source_prior,
    *, device: str = "cpu", support_quantile: float = 0.95,
) -> "RouterFeatureConfig | None":
    """Source-calibrate ONLY the target-prior density-NLL support threshold (q95 over source
    pseudo-domains). ood_score / support_gap thresholds are left OFF in v1 because they would
    misfire on benign label-prior shift. Returns None if too few source pseudo-domains."""
    levels = np.asarray(source_pseudo_levels)
    nlls = []
    for u in np.unique(levels):
        m = levels == u
        if int(m.sum()) < cfg.tta.min_target:
            continue
        U = _embed(model, X_src[m], device)
        sd = prior_decoupled_density_diagnostics(model.head.density, U, source_prior)
        nlls.append(sd["density_nll_target_prior"])
    if len(nlls) < 2:
        return None
    q = float(np.quantile(np.asarray(nlls, dtype=np.float64), support_quantile))
    return RouterFeatureConfig(
        min_target_n=max(20, int(cfg.tta.min_target)),
        min_ess=8.0,
        max_density_nll_target_prior=q,
        max_ood_score=None,
        max_support_gap_abs=None,
    )


def collect_source_pseudo_tta_gains(
    model, X_src, y_src, source_pseudo_levels, cfg, source_prior, *, device: str = "cpu",
) -> dict:
    """Per-source-pseudo-domain TTA gain (bAcc_adapt - bAcc_identity). Reuses the train_safety_gate
    idea WITHOUT fitting a gate. These gains drive the router's ACAR-harm degeneracy detection."""
    tta = ClassConditionalTTA(model.head.density, source_prior, cfg.tta, cfg.n_classes, device)
    levels = np.asarray(source_pseudo_levels)
    y_src = np.asarray(y_src)
    gains = []
    for u in np.unique(levels):
        m = levels == u
        if int(m.sum()) < cfg.tta.min_target:
            continue
        U = _embed(model, X_src[m], device)
        yu = y_src[m]
        p_id = _predict_generative(model, U, source_prior)
        res = tta.fit(U, pseudo_labels=p_id.argmax(1))
        p_ad = _predict_transform(model, U, res.transform, res.pi_T)
        gains.append(float(balanced_accuracy_score(yu, p_ad.argmax(1))
                           - balanced_accuracy_score(yu, p_id.argmax(1))))
    arr = np.asarray(gains, dtype=np.float64)
    harmed = arr <= -0.02
    return dict(
        gains=arr.tolist(), n_pseudo=int(arr.size),
        gain_min=(float(arr.min()) if arr.size else float("nan")),
        gain_mean=(float(arr.mean()) if arr.size else float("nan")),
        gain_max=(float(arr.max()) if arr.size else float("nan")),
        n_harm_margin_002=int(harmed.sum()),
        n_nonharm_margin_002=int((~harmed).sum()),
        has_two_classes_margin_002=bool(harmed.any() and (~harmed).any()),
    )


# ------------------------------------------------------------------ action bundles
def build_identity_bundle(support_diag, n_target, feature_cfg, acar_harm_gains):
    diag = dict(support_diag)                    # shared support/prior features
    diag["n_target"] = float(n_target)
    diag["delta_density_nll"] = 0.0              # identity-neutral action features
    diag["transform_norm"] = 0.0
    diag["condition_number"] = 1.0
    diag["pred_disagreement"] = 0.0
    diag.pop("cmi_residual", None)               # never trust the dead field at route time
    return build_router_features(diag, config=feature_cfg, acar_harm_gains=acar_harm_gains)


def build_offline_tta_bundle(res, support_diag, n_target, feature_cfg, acar_harm_gains):
    diag = dict(res.diagnostics)                 # TTA action features
    diag.pop("cmi_residual", None)
    diag["n_target"] = float(n_target)
    diag.update(support_diag)                     # override ess/ood/prior_shift + add density NLLs
    bundle = build_router_features(diag, config=feature_cfg, acar_harm_gains=acar_harm_gains)
    if not res.adapted:                          # TTA fell back to identity -> block this TTA action
        new_reasons = normalize_reasons(list(bundle.reason_codes) + [OACIReason.OACI_TTA_IDENTITY_FALLBACK])
        new_diag = dict(bundle.diagnostics)
        new_diag["reason_codes"] = [r.value for r in new_reasons]
        bundle = dataclasses.replace(bundle, reason_codes=new_reasons, diagnostics=new_diag)
    return bundle


# ------------------------------------------------------------------ main entry
def evaluate_router_offline_tta(
    model, X, y, domain, cfg, source_prior,
    *, router: "RefusalFirstRouter | None" = None, X_src=None, y_src=None,
    source_pseudo_levels=None, device: str = "cpu",
    calibrate_source_support: bool = True, support_quantile: float = 0.95,
) -> dict:
    y = np.asarray(y)
    domain = np.asarray(domain)
    n_classes = cfg.n_classes

    # --- source-only calibration (support threshold + pseudo harm gains) ---
    feature_cfg = None
    if calibrate_source_support and X_src is not None and y_src is not None and source_pseudo_levels is not None:
        feature_cfg = calibrate_router_feature_config_from_source(
            model, X_src, y_src, source_pseudo_levels, cfg, source_prior,
            device=device, support_quantile=support_quantile)
    support_cal_available = feature_cfg is not None
    if feature_cfg is None:
        feature_cfg = RouterFeatureConfig(min_target_n=max(20, int(cfg.tta.min_target)))

    gains_info = None
    acar_harm_gains = None
    if X_src is not None and y_src is not None and source_pseudo_levels is not None:
        gains_info = collect_source_pseudo_tta_gains(
            model, X_src, y_src, source_pseudo_levels, cfg, source_prior, device=device)
        acar_harm_gains = gains_info["gains"]
    src_acar_state = assess_acar_harm_calibration(
        acar_harm_gains, min_examples=feature_cfg.min_calibration_examples,
        harm_margin=feature_cfg.harm_margin).state.value

    if router is None:
        router = RefusalFirstRouter(RouterConfig(feature_config=feature_cfg))

    # --- per-domain routing (labels touched only in the post-hoc block) ---
    tta = ClassConditionalTTA(model.head.density, source_prior, cfg.tta, n_classes, device)
    proba_id = np.zeros((len(y), n_classes))
    proba_tta = np.zeros((len(y), n_classes))
    proba_sel = np.full((len(y), n_classes), np.nan)
    per_domain = {}
    reason_hist: Counter = Counter()
    tta_block_hist: Counter = Counter()
    action_counts: Counter = Counter()

    for d in np.unique(domain):
        m = domain == d
        U = _embed(model, X[m], device)
        p_id = _predict_generative(model, U, source_prior)
        res = tta.fit(U, pseudo_labels=p_id.argmax(1))
        p_ad = _predict_transform(model, U, res.transform, res.pi_T)
        proba_id[m] = p_id
        proba_tta[m] = p_ad

        support_diag = prior_decoupled_density_diagnostics(model.head.density, U, source_prior)
        n_target = int(m.sum())
        id_bundle = build_identity_bundle(support_diag, n_target, feature_cfg, acar_harm_gains)
        tta_bundle = build_offline_tta_bundle(res, support_diag, n_target, feature_cfg, acar_harm_gains)

        decision = router.route_diagnostics(
            {"identity": id_bundle, "offline_tta": tta_bundle},
            mode="offline", acar_harm_gains=acar_harm_gains)

        # -------- post-hoc evaluation (y first used HERE) --------
        yd = y[m]
        bacc_id = float(balanced_accuracy_score(yd, p_id.argmax(1)))
        bacc_ad = float(balanced_accuracy_score(yd, p_ad.argmax(1)))
        raw_gain = bacc_ad - bacc_id
        act = decision.action
        if act == RouterAction.OFFLINE_TTA:
            proba_sel[m] = p_ad
            sel_bacc = bacc_ad
        elif act == RouterAction.IDENTITY:
            proba_sel[m] = p_id
            sel_bacc = bacc_id
        else:
            sel_bacc = float("nan")               # REFUSE: no prediction counted
        sel_gain = (sel_bacc - bacc_id) if not np.isnan(sel_bacc) else float("nan")

        action_counts[act.value] += 1
        reason_hist.update(r.value for r in decision.reason_codes)
        tta_block_hist.update(decision.action_scores["offline_tta"]["blocking_reason_codes"])

        per_domain[int(d)] = dict(
            n=n_target, decision_action=act.value, accepted=bool(decision.accepted),
            reason_codes=[r.value for r in decision.reason_codes],
            identity_bacc=bacc_id, offline_tta_bacc=bacc_ad, raw_gain=raw_gain,
            selected_bacc=sel_bacc, selected_gain_vs_identity=sel_gain,
            action_scores=decision.action_scores, conformal_bounds=decision.conformal_bounds,
            diagnostics_identity=id_bundle.diagnostics, diagnostics_offline_tta=tta_bundle.diagnostics,
            support=dict(support_diag),
        )

    # --- panels ---
    panel_id = metric_panel(proba_id, y, domain); panel_id["setting"] = "identity"
    panel_tta = metric_panel(proba_tta, y, domain); panel_tta["setting"] = "offline_tta_raw"
    acc_mask = ~np.isnan(proba_sel[:, 0])
    if acc_mask.any():
        panel_sel = metric_panel(proba_sel[acc_mask], y[acc_mask], domain[acc_mask])
        panel_sel["setting"] = "router_selected"
    else:
        panel_sel = _nan_panel("router_selected")

    # --- summary ---
    n_domains = len(per_domain)
    doms = list(per_domain.values())
    n_off = action_counts.get("offline_tta", 0)
    n_id = action_counts.get("identity", 0)
    n_ref = action_counts.get("refuse", 0)
    n_accepted = sum(1 for dv in doms if dv["accepted"])
    not_adapted = [dv for dv in doms if dv["decision_action"] != "offline_tta"]

    def _mean(xs):
        xs = [x for x in xs if not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(xs)) if xs else float("nan")

    summary = dict(
        n_domains=n_domains, n_accepted_domains=n_accepted,
        coverage=(n_accepted / n_domains if n_domains else float("nan")),
        refusal_rate=(n_ref / n_domains if n_domains else float("nan")),
        identity_rate=(n_id / n_domains if n_domains else float("nan")),
        offline_tta_rate=(n_off / n_domains if n_domains else float("nan")),
        accepted_bacc=float(panel_sel["balanced_acc"]),
        raw_tta_mean_gain=_mean([dv["raw_gain"] for dv in doms]),
        selected_mean_gain_vs_identity=_mean([dv["selected_gain_vs_identity"] for dv in doms]),
        avoided_harm=(_mean([max(0.0, -dv["raw_gain"]) for dv in not_adapted]) if not_adapted else 0.0),
        missed_benefit=(_mean([max(0.0, dv["raw_gain"]) for dv in not_adapted]) if not_adapted else 0.0),
        source_support_calibration_available=bool(support_cal_available),
        source_support_threshold_nll_target_prior=(
            None if feature_cfg.max_density_nll_target_prior is None
            else float(feature_cfg.max_density_nll_target_prior)),
        source_acar_harm_calibration_state=src_acar_state,
        source_pseudo_gains=(gains_info if gains_info is not None else None),
        reason_hist=dict(reason_hist), tta_block_reason_hist=dict(tta_block_hist),
        action_counts=dict(action_counts),
    )

    return dict(
        setting="router_offline_tta",
        identity=panel_id, offline_tta_raw=panel_tta, router_selected=panel_sel,
        delta_raw_offline_tta=panel_delta(panel_tta, panel_id),
        router_summary=summary, per_domain=per_domain,
    )


if __name__ == "__main__":
    # tiny end-to-end integration self-test (small config, not a scientific run)
    from h2cmi.config import H2Config
    from h2cmi.data.eeg_simulator import EEGSimulator, ShiftSpec, train_target_split
    from h2cmi.train.trainer import train_h2, reference_prior

    shift = ShiftSpec(cov=1.0, prior=0.4, concept=0.0, concept_site_frac=0.0, montage=0.2, noise=0.3)
    sim = EEGSimulator(3, 16, 128, 128.0, shift=shift, seed=0).sample(5, 3, 2, 40)
    src_idx, tgt_idx = train_target_split(sim, n_target_sites=1, seed=0)
    cfg = H2Config(n_classes=3).small()
    cfg.encoder.n_chans = 16; cfg.encoder.n_times = 128; cfg.encoder.fs = 128.0
    Xs, ys = sim.X[src_idx], sim.y[src_idx]
    src_dom = sim.domains.subset(src_idx)
    model, *_ = train_h2(Xs, ys, src_dom, sim.dag, cfg, align_factor="site", verbose=False)
    pi_star = reference_prior(ys, 3, cfg.align.reference_prior)

    Xt, yt = sim.X[tgt_idx], sim.y[tgt_idx]
    tgt_unit = sim.domains.subset(tgt_idx).factor("subject")
    src_unit = src_dom.factor("subject")
    rep = evaluate_router_offline_tta(
        model, Xt, yt, tgt_unit, cfg, pi_star,
        X_src=Xs, y_src=ys, source_pseudo_levels=src_unit, device=cfg.train.device)

    assert set(("identity", "offline_tta_raw", "router_selected", "router_summary", "per_domain")) <= set(rep)
    s = rep["router_summary"]
    assert sum(s["action_counts"].values()) == s["n_domains"]
    assert 0.0 <= s["coverage"] <= 1.0
    assert np.isfinite(rep["identity"]["balanced_acc"]) and np.isfinite(rep["offline_tta_raw"]["balanced_acc"])
    for dv in rep["per_domain"].values():
        assert "identity" in dv["action_scores"] and "offline_tta" in dv["action_scores"]
        assert np.all(np.isfinite(np.asarray(dv["diagnostics_identity"]["reason_codes"], dtype=object) != None))
    # default conservative router must not select OFFLINE_TTA when source ACAR harm is degenerate/unavailable
    if s["source_acar_harm_calibration_state"] in ("degenerate", "unavailable"):
        assert s["action_counts"].get("offline_tta", 0) == 0, s["action_counts"]
    print("router_harness self-test passed:",
          "actions=", dict(s["action_counts"]), "acar=", s["source_acar_harm_calibration_state"])
