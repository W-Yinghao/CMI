"""Evaluation metrics: balanced accuracy, macro-F1, ECE, NLL, conditional domain
leakage probe (frozen backbone), and label separability."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

from cmi.methods.regularizers import _mlp, kl_to_prior, empirical_priors
from cmi.train.trainer import embed


def domain_class_span_stats(y, d, n_cls, prefix="decoder"):
    """Summarize whether each domain spans enough classes for a decoder concept probe.

    A decoder I(Y;D|Z) or JS(h_full,h0) probe is concept-interpretable only when each
    domain contains at least two classes. Otherwise the domain-conditioned decoder can
    exploit single-class domain priors and the number should be reported as invalid.
    """
    y = np.asarray(y)
    d = np.asarray(d)
    spans = []
    counts = []
    for dom in sorted(np.unique(d)):
        yy = y[d == dom]
        spans.append(int(len(np.unique(yy))))
        counts.append(int(len(yy)))
    spans_arr = np.asarray(spans, dtype=np.int64)
    needed = min(2, int(n_cls))
    min_span = int(spans_arr.min()) if len(spans_arr) else 0
    valid = bool(len(spans_arr) >= 2 and min_span >= needed)
    return {
        f"{prefix}_valid": valid,
        f"{prefix}_n_domains": int(len(spans)),
        f"{prefix}_min_domain_classes": min_span,
        f"{prefix}_mean_domain_classes": float(spans_arr.mean()) if len(spans_arr) else 0.0,
        f"{prefix}_single_class_frac": float((spans_arr < needed).mean()) if len(spans_arr) else 1.0,
        f"{prefix}_domain_class_spans": spans,
        f"{prefix}_domain_counts": counts,
    }


DECODER_SUMMARY_KEYS = (
    "decoder_cmi", "decoder_cmi_rw",
    "decoder_cmi_res", "decoder_cmi_res_rw",
    "decoder_js_res", "decoder_js_res_rw",
    "decoder_cmi_res_null_q", "decoder_cmi_res_excess",
    "decoder_cmi_res_rw_null_q", "decoder_cmi_res_rw_excess",
    "decoder_js_res_null_q", "decoder_js_res_excess",
    "decoder_js_res_rw_null_q", "decoder_js_res_rw_excess",
)


def add_decoder_valid_means(summary, records, valid_key="decoder_valid"):
    """Add valid-only decoder probe means to a runner summary dict.

    Raw decoder means are kept for backward compatibility, but paper-facing tables should
    use the *_valid_mean fields so invalid domain splits cannot silently affect P(Y|Z)
    evidence. Fields are JSON null when no valid folds are available.
    """
    valid_records = [r for r in records if bool(r.get(valid_key, False))]
    summary["decoder_valid_n"] = int(len(valid_records))
    for key in DECODER_SUMMARY_KEYS:
        vals = [r[key] for r in valid_records if key in r]
        summary[f"{key}_valid_mean"] = float(np.mean(vals)) if vals else None
    return summary


def classification_metrics(prob, y_true):
    pred = prob.argmax(1)
    p = np.clip(prob[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    return dict(
        balanced_acc=float(balanced_accuracy_score(y_true, pred)),
        macro_f1=float(f1_score(y_true, pred, average="macro")),
        nll=float(-np.log(p).mean()),
        ece=float(_ece(prob, y_true)),
    )


def _ece(prob, y_true, n_bins=15):
    conf = prob.max(1); pred = prob.argmax(1); acc = (pred == y_true)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def leakage_probe(backbone, Xprobe, yprobe, dprobe, Xeval, yeval, deval, n_cls,
                  device="cpu", epochs=150, lr=2e-3, seed=0, reweight=False):
    """Freeze backbone, fit a fresh q_probe(D|Z,Y) on a held-out source split, then on a
    second source split report mean KL(q_probe || pi_y) (residual I(Z;D|Y)) and the
    conditional-domain-prediction advantage over the label-only prior baseline.

    With reweight=True this matches DualPC's GLS reference measure: q(D|Z,Y) is fit with
    w_i=pi*(y_i)/pi_d_i(y_i), and the KL reference is the weighted domain marginal p_w(D),
    which is the same target used by train_model(... method="dualpc") for I_w(Z;D|Y).
    """
    torch.manual_seed(seed)
    n_dom = int(max(dprobe.max(), deval.max())) + 1
    Zp = torch.tensor(embed(backbone, Xprobe, device)).to(device)
    Ze = torch.tensor(embed(backbone, Xeval, device)).to(device)
    yp = torch.tensor(yprobe).to(device); dp = torch.tensor(dprobe).to(device)
    if reweight:
        from cmi.train.trainer import _label_shift_weights
        wp = torch.tensor(_label_shift_weights(yprobe, dprobe, n_dom, n_cls), dtype=torch.float32, device=device)
        we = torch.tensor(_label_shift_weights(yeval, deval, n_dom, n_cls), dtype=torch.float32, device=device)
        wd = torch.zeros(n_dom, device=device)
        wd.scatter_add_(0, dp, wp)
        pd_ref = (wd + 1e-6) / (wd.sum() + 1e-6 * n_dom)
        log_ref = torch.log(pd_ref)
    else:
        pi_y, _, _ = empirical_priors(yprobe, dprobe, n_dom, n_cls)
        log_pi_y = torch.log(torch.tensor(pi_y, dtype=torch.float32, device=device))
        wp = torch.ones(len(dprobe), dtype=torch.float32, device=device)
        we = torch.ones(len(deval), dtype=torch.float32, device=device)
        log_ref = None
    q = _mlp(backbone.z_dim + n_cls, n_dom).to(device)
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    inp_p = torch.cat([Zp, F.one_hot(yp, n_cls).float()], 1)
    for _ in range(epochs):
        per = F.cross_entropy(q(inp_p), dp, reduction="none")
        loss = (wp * per).sum() / wp.sum().clamp(min=1e-8)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ye = torch.tensor(yeval).to(device)
        logits = q(torch.cat([Ze, F.one_hot(ye, n_cls).float()], 1))
        if reweight:
            kl = kl_to_prior(logits, log_ref, weight=we).item()
            prior_pred = np.full(len(deval), int(log_ref.argmax().cpu()))
        else:
            kl = kl_to_prior(logits, log_pi_y[ye]).item()
            prior_pred = pi_y[yeval].argmax(1)
        dom_acc = float((logits.argmax(1).cpu().numpy() == deval).mean())
        prior_acc = float((prior_pred == deval).mean())
    return dict(leakage_kl=kl, cond_dom_acc=dom_acc, prior_dom_acc=prior_acc,
                leakage_advantage=dom_acc - prior_acc)


def marginal_leakage_probe(backbone, Xprobe, yprobe, dprobe, Xeval, yeval, deval, n_cls,
                           device="cpu", epochs=150, lr=2e-3, seed=0, reweight=False):
    """Freeze backbone, fit q(D|Z), and report marginal domain leakage.

    This is the direct P(Z) diagnostic used by DualPC. With reweight=False it estimates raw
    I(Z;D)-style leakage against the source domain marginal. With reweight=True it estimates
    I_w(Z;D) under the GLS reference-prior measure, using w_i=pi*(y_i)/pi_d_i(y_i) and the
    corresponding weighted domain marginal. It complements `leakage_probe`, which measures
    conditional leakage I(Z;D|Y).
    """
    torch.manual_seed(seed)
    n_dom = int(max(dprobe.max(), deval.max())) + 1
    Zp = torch.tensor(embed(backbone, Xprobe, device)).to(device)
    Ze = torch.tensor(embed(backbone, Xeval, device)).to(device)
    dp = torch.tensor(dprobe).to(device)

    if reweight:
        from cmi.train.trainer import _label_shift_weights
        wp = torch.tensor(_label_shift_weights(yprobe, dprobe, n_dom, n_cls), device=device)
        we = torch.tensor(_label_shift_weights(yeval, deval, n_dom, n_cls), device=device)
    else:
        wp = torch.ones(len(dprobe), device=device)
        we = torch.ones(len(deval), device=device)

    q = _mlp(backbone.z_dim, n_dom).to(device)
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    for _ in range(epochs):
        per = F.cross_entropy(q(Zp), dp, reduction="none")
        loss = (wp * per).sum() / wp.sum().clamp(min=1e-8)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        de = torch.tensor(deval).to(device)
        wd = torch.zeros(n_dom, device=device)
        wd.scatter_add_(0, dp, wp)
        pd_ref = (wd + 1e-6) / (wd.sum() + 1e-6 * n_dom)
        logits = q(Ze)
        kl = kl_to_prior(logits, torch.log(pd_ref), weight=we).item()
        pred = logits.argmax(1).cpu().numpy()
        dom_acc = float((pred == deval).mean())
        prior_acc = float((np.full(len(deval), int(pd_ref.argmax().cpu())) == deval).mean())
    return dict(marginal_leakage_kl=kl,
                marginal_dom_acc=dom_acc,
                marginal_prior_acc=prior_acc,
                marginal_leakage_advantage=dom_acc - prior_acc)


def decoder_leakage_probe(backbone, Xprobe, yprobe, dprobe, Xeval, yeval, deval, n_cls,
                          device="cpu", epochs=150, lr=2e-3, seed=0, reweight=False,
                          null_perms=0, null_quantile=0.95):
    """Held-out estimate of the DECODER CMI I(Y;D|Z) = H(Y|Z) - H(Y|Z,D).
    Freeze backbone; fit q(Y|Z) and h(Y|Z,D) on one source split; report CE_q - CE_h on a disjoint
    source split. >0 means the predictor still NEEDS the domain to predict Y given Z = residual
    CONCEPT shift (the z->y rule differs across domains). ~0 means the labeling rule is domain-invariant.
    NOTE: when D determines Y (SCPS, D=subject) this degenerates to ~H(Y|Z); use a D where each domain
    spans both classes (D=cohort/site) for a meaningful concept estimate.
    If reweight=True, apply GLS importance weights w_d(y)=pi*(y)/pi_d(y) to the CEs -> the LABEL-SHIFT-
    CORRECTED decoder CMI Itilde(Y;D|Z): a positive value here is genuine concept shift, separated from a
    posterior difference caused purely by per-domain label-prior shift (report BOTH plain and reweighted)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(10000 + seed)
    n_dom = int(max(dprobe.max(), deval.max())) + 1
    validity = domain_class_span_stats(np.concatenate([yprobe, yeval]),
                                       np.concatenate([dprobe, deval]), n_cls)
    Zp = torch.tensor(embed(backbone, Xprobe, device)).to(device)
    Ze = torch.tensor(embed(backbone, Xeval, device)).to(device)
    yp = torch.tensor(yprobe).to(device)
    ye = torch.tensor(yeval).to(device)

    def _permute_domain_within_class(d, y):
        out = np.array(d, copy=True)
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            if len(idx) > 1:
                out[idx] = rng.permutation(out[idx])
        return out

    def _weights(y_arr, d_arr):
        if reweight:
            from cmi.train.trainer import _label_shift_weights
            return torch.tensor(_label_shift_weights(y_arr, d_arr, n_dom, n_cls)).to(device)
        return torch.ones(len(y_arr), device=device)

    def wce(logits, t, w):
        return (F.cross_entropy(logits, t, reduction="none") * w).sum() / w.sum()

    def wjs(logits_a, logits_b, w):
        pa = F.softmax(logits_a, 1).clamp_min(1e-8)
        pb = F.softmax(logits_b, 1).clamp_min(1e-8)
        m = (0.5 * (pa + pb)).clamp_min(1e-8)
        js = 0.5 * (pa * (pa.log() - m.log())).sum(1) + 0.5 * (pb * (pb.log() - m.log())).sum(1)
        return (js * w).sum() / w.sum().clamp(min=1e-8)

    def _fit_eval(dprobe_arr, deval_arr, seed_offset=0):
        torch.manual_seed(seed + seed_offset)
        wp, we = _weights(yprobe, dprobe_arr), _weights(yeval, deval_arr)
        dp_idx = torch.tensor(dprobe_arr).to(device)
        de_idx = torch.tensor(deval_arr).to(device)
        dp_oh = F.one_hot(dp_idx, n_dom).float()
        qY = _mlp(backbone.z_dim, n_cls).to(device)               # domain-blind decoder a(Y|Z)
        hY = _mlp(backbone.z_dim + n_dom, n_cls).to(device)       # FULL domain decoder h(Y|Z,D)
        uY = _mlp(backbone.z_dim, n_cls).to(device)               # intercept-only: shared u(Z) ...
        bD = torch.zeros(n_dom, n_cls, device=device, requires_grad=True)   # ... + per-domain bias b_D
        opt = torch.optim.Adam(list(qY.parameters()) + list(hY.parameters()) + list(uY.parameters()) + [bD], lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            (wce(qY(Zp), yp, wp) + wce(hY(torch.cat([Zp, dp_oh], 1)), yp, wp)
             + wce(uY(Zp) + bD[dp_idx], yp, wp)).backward()
            opt.step()
        de_oh = F.one_hot(de_idx, n_dom).float()
        with torch.no_grad():
            lh = hY(torch.cat([Ze, de_oh], 1))
            l0 = uY(Ze) + bD[de_idx]
            ce_q = wce(qY(Ze), ye, we).item()                      # H(Y|Z)
            ce_h = wce(lh, ye, we).item()                          # H(Y|Z,D) full
            ce_0 = wce(l0, ye, we).item()                          # H(Y|Z,D) intercept-only
            js = wjs(lh, l0, we).item()                            # JS(h_full || h0), DualPC training diagnostic
        return ce_q, ce_h, ce_0, js

    ce_q, ce_h, ce_0, js = _fit_eval(dprobe, deval)
    res = max(ce_0 - ce_h, 0.0)
    out = dict(decoder_cmi=max(ce_q - ce_h, 0.0),             # raw I(Y;D|Z) = CE(a)-CE(h)
               decoder_cmi_res=res,                          # RESIDUAL = CE(h0)-CE(h): boundary-only (Route C)
               decoder_js_res=max(js, 0.0),                   # JS(h_full,h0): DualPC P(Y|Z) consistency diagnostic
               ce_q=ce_q, ce_h=ce_h, ce_0=ce_0)
    out.update(validity)
    if null_perms:
        vals, js_vals = [], []
        for k in range(int(null_perms)):
            dp_null = _permute_domain_within_class(dprobe, yprobe)
            de_null = _permute_domain_within_class(deval, yeval)
            _, ce_hn, ce_0n, jsn = _fit_eval(dp_null, de_null, seed_offset=1000 + k)
            vals.append(max(ce_0n - ce_hn, 0.0))
            js_vals.append(max(jsn, 0.0))
        vals = np.asarray(vals, dtype=np.float64)
        js_vals = np.asarray(js_vals, dtype=np.float64)
        q = float(np.quantile(vals, null_quantile))
        js_q = float(np.quantile(js_vals, null_quantile))
        out.update(decoder_cmi_res_null_mean=float(vals.mean()),
                   decoder_cmi_res_null_std=float(vals.std()),
                   decoder_cmi_res_null_q=q,
                   decoder_cmi_res_excess=max(res - q, 0.0),
                   decoder_cmi_res_null_perms=int(null_perms),
                   decoder_js_res_null_mean=float(js_vals.mean()),
                   decoder_js_res_null_std=float(js_vals.std()),
                   decoder_js_res_null_q=js_q,
                   decoder_js_res_excess=max(js - js_q, 0.0),
                   decoder_js_res_null_perms=int(null_perms))
    return out


def label_separability(backbone, Xtr, ytr, Xte, yte, device="cpu"):
    Ztr, Zte = embed(backbone, Xtr, device), embed(backbone, Xte, device)
    lr = LogisticRegression(max_iter=1000).fit(Ztr, ytr)
    return float(lr.score(Zte, yte))
