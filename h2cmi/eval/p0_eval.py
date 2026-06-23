"""REVIEW_P0 corrected evaluation core (section A): separate FIT prior from DECISION prior.

For one target unit, fit the joint EM EXACTLY ONCE -> (T_J geometry, pi_J estimated prior), then decode
four branches from the SAME (T_J, pi_J): {identity, joint-geometry} x {uniform, pi_J}. Balanced-accuracy
PRIMARY results always use the UNIFORM decision prior. Also evaluate the comparator operators
(fixed-iterative geometry, fixed-reference one-shot, pooled, Latent-IM-Diag, source-recolored EA) under
the uniform decision prior. Records full metrics (bAcc, acc, macro-F1, NLL, Brier, ECE), transform
parameters, prediction hashes, and the exact G + P + Interaction decomposition with a float-tolerance
identity check. Reused by the W1/W2/V2P P0 runners; backward compatible (adds, never reinterprets).
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score

from h2cmi.eval.harness import _embed, _predict_generative, _predict_transform
from h2cmi.tta.class_conditional import B1A_VARIANTS_BY_NAME
from h2cmi.eval.spdim import spdim_fit
from h2cmi.eval.ea import reference_cov, ea_transform, apply_ea


def _nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def _brier(p, y, K):
    oh = np.eye(K)[y]
    return float(((p - oh) ** 2).sum(1).mean())


def _ece(p, y, n_bins=10):
    conf = p.max(1); pred = p.argmax(1); correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for b in range(n_bins):
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if m.sum() == 0:
            continue
        e += (m.mean()) * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def _pred_hash(pred):
    return hashlib.sha256(np.asarray(pred, dtype=np.int64).tobytes()).hexdigest()[:16]


def _transform_fields(T):
    if T is None:
        return dict(a=None, b=None, transform_norm=0.0)
    a = [float(x) for x in T.a.detach().cpu().numpy()] if hasattr(T, "a") else None
    b = [float(x) for x in T.b.detach().cpu().numpy()] if hasattr(T, "b") else None
    A = T.matrix(); I = torch.eye(A.shape[0], device=A.device)
    tn = float(((A - I) ** 2).sum().sqrt().detach().cpu())
    return dict(a=a, b=b, transform_norm=tn)


def _record(p, y, K, T=None, keep_probs=False, keep_preds=False):
    pred = p.argmax(1)
    r = dict(bacc=float(balanced_accuracy_score(y, pred)), acc=float(accuracy_score(y, pred)),
             macro_f1=float(f1_score(y, pred, average="macro")), nll=_nll(p, y),
             brier=_brier(p, y, K), ece=_ece(p, y), pred_hash=_pred_hash(pred),
             occupancy=[float((pred == c).mean()) for c in range(K)])
    r.update(_transform_fields(T))
    if keep_probs:
        r["probs"] = p.tolist()
    if keep_preds:
        r["preds"] = [int(x) for x in pred]
    return r


def eval_unit_p0(model, tta, pooled_ref, R_src, Xa, Xe, Ua, Ue, ye, device, K, tta_seed,
                 keep_probs=False, keep_preds=False):
    """Returns (branches: dict[name->record], decomposition: dict). All decoded with the UNIFORM
    decision prior except the two *_joint_prior branches (which use pi_J for the decomposition)."""
    uni = np.full(K, 1.0 / K)
    V = B1A_VARIANTS_BY_NAME
    out = {}

    # ---- joint EM fit ONCE -> (T_J, pi_J) ; four (geometry x decision-prior) branches ----
    fj = tta.fit_variant(Ua, V["joint_iterative_diag"], tta_seed=tta_seed)
    Tj = fj.transform
    piJ = np.asarray(fj.pi_T.cpu().numpy() if torch.is_tensor(fj.pi_T) else fj.pi_T, dtype=float)
    out["identity_uniform"] = _record(_predict_generative(model, Ue, uni), ye, K, None, keep_probs, keep_preds)
    out["identity_joint_prior"] = _record(_predict_generative(model, Ue, piJ), ye, K, None, keep_probs, keep_preds)
    out["joint_geometry_uniform"] = _record(_predict_transform(model, Ue, Tj, uni), ye, K, Tj, keep_probs, keep_preds)
    out["joint_geometry_joint_prior"] = _record(_predict_transform(model, Ue, Tj, piJ), ye, K, Tj, keep_probs, keep_preds)

    # ---- comparator operators (uniform decision) ----
    for name, vkey, needs_pool in (
            ("fixed_iterative_geometry_uniform", "gen_iterative_diag", False),
            ("fixed_reference_oneshot_uniform", "gen_oneshot_diag", False),
            ("pooled_uniform", "pooled_empirical_diag", True)):
        f = tta.fit_variant(Ua, V[vkey], pooled_ref=(pooled_ref if needs_pool else None), tta_seed=tta_seed)
        out[name] = _record(_predict_transform(model, Ue, f.transform, uni), ye, K, f.transform, keep_probs, keep_preds)
    Ts = spdim_fit(model.head.density, Ua, uni, device)
    out["latent_im_diag_uniform"] = _record(_predict_transform(model, Ue, Ts, uni), ye, K, Ts, keep_probs, keep_preds)
    M = ea_transform(R_src, reference_cov(Xa))
    Ue_ea = _embed(model, apply_ea(Xe, M), device)
    out["source_recolored_ea"] = _record(_predict_generative(model, Ue_ea, uni), ye, K, None, keep_probs, keep_preds)

    # ---- exact prediction decomposition (balanced accuracy) ----
    B_iu = out["identity_uniform"]["bacc"]; B_ij = out["identity_joint_prior"]["bacc"]
    B_ju = out["joint_geometry_uniform"]["bacc"]; B_jj = out["joint_geometry_joint_prior"]["bacc"]
    G = B_ju - B_iu
    P = B_ij - B_iu
    inter = (B_jj - B_ju) - (B_ij - B_iu)
    full = B_jj - B_iu
    decomp = dict(G=G, P=P, interaction=inter, full_joint_delta=full,
                  residual=float(full - (G + P + inter)),
                  prior_m_step_geometry=out["fixed_iterative_geometry_uniform"]["bacc"] - B_ju,
                  pi_J=[float(x) for x in piJ])
    assert abs(decomp["residual"]) < 1e-9, f"decomposition identity violated: {decomp['residual']}"
    return out, decomp
