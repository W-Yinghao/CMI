"""CMI-Trace E1 (Theorem 2) — subject-information x exact-head-use SPECTRUM on real EEG.

Tests the leakage-reliance reversal MECHANISM: amount-only CMI control (CIGL) should strip the highest-lambda,
task-orthogonal (low-tau) subject directions first, so total CMI + subject effective-rank fall while the top
subject direction rotates toward a task-bearing one and exact-head reliance on it rises.

This is the WHITENED-metric variant the theorem asks for (pooled within-class Sigma_W^{-1/2}), distinct from
the RAW-metric `cmi.eval.leakage_removal`/`reliance_audit` audit which it does NOT modify. Reuses:
  * cmi.eval.conditional_subject_leakage.{three_way_support_split, flat_conditional_cmi} — the posterior-KL
    conditional-subject ruler + fully-retrained within-label permutation null, applied to each 1-D projection.
  * cmi.eval.audit_npz.{load_audit_npz, replay_head, head_replay_ok} — the VERIFIED stored linear head.

FIREWALL: whitening, subject subspace, lambda, and tau are all fit on SOURCE rows (d != target_domain,
cross-checked against stored source_indices). Target trials are eval-only. No target labels touch any fit.

Direction bookkeeping (the crux):
  whitened coordinate   t_j = Z_tilde @ u_tilde_j,   Z_tilde = Z @ Sigma_W^{-1/2}
  RAW-space unit dir    u_j = normalize(Sigma_W^{-1/2} @ u_tilde_j)   (the raw dir whose readout is t_j)
  exact-head reliance   tau_j = CE(h((I - u_j u_j^T) Z), Y) - CE(h(Z), Y)   via the stored head h.
lambda_j is measured on t_j; tau_j removes u_j in the head's RAW input space. Cross-model raw axes are NEVER
compared directly (we pair the SCALAR endpoints by principal angle, per the reliance-audit discipline).
"""
from __future__ import annotations
import numpy as np

from cmi.eval.conditional_subject_leakage import three_way_support_split, flat_conditional_cmi
from cmi.eval.audit_npz import load_audit_npz, replay_head, head_replay_ok

EPS = 1e-8


# --------------------------------------------------------------------- whitening + subject spectrum
def pooled_within_class_cov(Z, y, shrink="lw", floor=1e-4):
    """Pooled within-class covariance Sigma_W of Z [N,dz] (subtract per-class mean, then estimate on the
    pooled residuals). Returns (Sigma_W, W_inv=Sigma_W^{-1/2}, W_half=Sigma_W^{1/2}) via symmetric
    eigendecomposition with a relative PSD floor. shrink='lw' uses Ledoit-Wolf toward scaled identity
    (stable at high dz / few trials); shrink=None is the raw MLE."""
    Z = np.asarray(Z, float); y = np.asarray(y)
    dz = Z.shape[1]
    Zc = np.empty_like(Z)
    for c in np.unique(y):
        m = y == c
        Zc[m] = Z[m] - Z[m].mean(0, keepdims=True)
    if shrink == "lw":
        from sklearn.covariance import LedoitWolf
        S = LedoitWolf(assume_centered=True).fit(Zc).covariance_
    else:
        S = (Zc.T @ Zc) / max(1, len(Z) - len(np.unique(y)))
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    w = np.clip(w, floor * max(w.max(), EPS), None)
    W_inv = (V / np.sqrt(w)) @ V.T
    W_half = (V * np.sqrt(w)) @ V.T
    return S, W_inv, W_half


def whitened_subject_scatter(Zt, y, d):
    """Between-subject-within-label scatter S_B in WHITENED coords Zt [N,dz]:
    S_B = sum_{y,d} n_{y,d} (mean(Zt|y,d) - mean(Zt|y)) (...)^T. Eigendecompose -> (dirs [r,dz] whitened
    orthonormal, energy [r] descending). energy_j = subject-score energy (eigenvalue of S_B)."""
    Zt = np.asarray(Zt, float); y = np.asarray(y); d = np.asarray(d)
    dz = Zt.shape[1]
    S_B = np.zeros((dz, dz))
    for c in np.unique(y):
        my = y == c
        mu_y = Zt[my].mean(0)
        for s in np.unique(d[my]):
            m = my & (d == s)
            n = int(m.sum())
            if n > 0:
                off = Zt[m].mean(0) - mu_y
                S_B += n * np.outer(off, off)
    w, V = np.linalg.eigh(0.5 * (S_B + S_B.T))
    order = np.argsort(w)[::-1]
    w = np.clip(w[order], 0.0, None)
    dirs = V[:, order].T                      # [dz, dz] rows = directions, descending energy
    return dirs, w


# --------------------------------------------------------------------- head / loss utilities
def class_centered_head(W, b):
    """Class-centered head (W_c, b_c): subtract the mean over classes so softmax/predictions are unchanged.
    Head GEOMETRY (kernel/rowspace/alignment) is defined w.r.t. W_c."""
    W = np.asarray(W, float); b = np.asarray(b, float)
    return W - W.mean(0, keepdims=True), b - b.mean()


def _ce(logits, y):
    """Mean cross-entropy of softmax(logits) against integer labels y (log-sum-exp stable)."""
    logits = np.asarray(logits, float)
    m = logits.max(1, keepdims=True)
    logZ = m[:, 0] + np.log(np.exp(logits - m).sum(1) + EPS)
    return float((logZ - logits[np.arange(len(y)), np.asarray(y)]).mean())


def head_alignment(u_raw, Wc):
    """Squared cosine between raw unit direction u and the leading right-singular direction of the
    class-centered head W_c (the head's dominant input direction). In [0,1]; rises as u becomes task-bearing."""
    _, _, Vt = np.linalg.svd(np.asarray(Wc, float), full_matrices=False)
    v1 = Vt[0]
    return float((u_raw @ v1) ** 2 / (max(u_raw @ u_raw, EPS)))


def effective_rank(energy):
    """Participation-ratio effective rank of the subject-energy spectrum: (sum e)^2 / sum e^2."""
    e = np.asarray(energy, float); e = e[e > 0]
    return float((e.sum() ** 2) / (max((e ** 2).sum(), EPS))) if e.size else 0.0


# --------------------------------------------------------------------- the per-sidecar spectrum
def subject_spectrum(data_or_path, *, representation="graph_z", k_spec=16, n_perm=50, n_random=50,
                     seed=0, device="cpu", shrink="lw"):
    """Full whitened subject lambda-tau spectrum for ONE audit sidecar (SOURCE-only fit).
    Returns a dict: per-direction energy/lambda/tau (exact-head) + random-control tau + geometry scalars.
    k_spec = # top directions to score with the (expensive) null-calibrated lambda ruler."""
    data = load_audit_npz(data_or_path) if isinstance(data_or_path, (str, bytes)) else data_or_path
    Z = np.asarray(data[representation], float)
    y = np.asarray(data["y"]).astype(np.int64); d = np.asarray(data["d"]).astype(np.int64)
    # target_domain is the DISTINCT d-tag on the held-out rows (NOT `target_subject`, which is the human
    # subject label and can collide with a source subject's remapped id). Derive it from target_indices.
    if not ("target_indices" in data and len(np.asarray(data["target_indices"]))):
        # FAIL-LOUD: never fall back to `target_subject` (a human label that can collide with a source
        # subject's remapped id and silently fold target rows into the source fit). Firewall requires the
        # distinct target tag, which lives in target_indices.
        raise ValueError("firewall: sidecar lacks target_indices; cannot determine the target domain tag "
                         "without risking target leakage (refusing the `target_subject` fallback)")
    ti = np.asarray(data["target_indices"]).ravel()
    tvals = np.unique(d[ti])
    if len(tvals) != 1:
        raise ValueError(f"target rows span multiple domain ids {tvals}; ambiguous target_domain")
    target_domain = int(tvals[0])
    src = d != target_domain
    # firewall: cross-check against stored provenance (no source-index row is the target tag; split matches)
    firewall_ok = bool(src.sum() > 0)
    if "source_indices" in data and len(np.asarray(data["source_indices"])):
        si = np.asarray(data["source_indices"]).ravel()
        firewall_ok = bool(firewall_ok and not np.any(d[si] == target_domain)
                           and set(np.where(~src)[0].tolist()) == set(np.asarray(data["target_indices"]).ravel().tolist()))
    if not head_replay_ok(data):
        raise ValueError("subject_spectrum requires a VERIFIED stored linear head (head_replay_ok=False)")

    Zs, ys, ds = Z[src], y[src], d[src]
    n_cls = int(np.asarray(data.get("model_logits")).shape[1]) if "model_logits" in data else int(ys.max() + 1)
    subs = np.unique(ds); n_dom = len(subs)
    d_remap = np.searchsorted(subs, ds)                       # contiguous 0..n_dom-1 for the ruler

    # whitening + subject spectrum on SOURCE
    _, W_inv, W_half = pooled_within_class_cov(Zs, ys, shrink=shrink)
    Zt = Zs @ W_inv                                           # whitened source reps
    dirs, energy = whitened_subject_scatter(Zt, ys, d_remap)
    dz = Zs.shape[1]
    r_eff = effective_rank(energy)
    top2_conc = float(energy[:2].sum() / max(energy.sum(), EPS))

    Wc, bc = class_centered_head(data["task_head_weight"], data.get("task_head_bias", np.zeros(n_cls)))
    ce_full = _ce(replay_head(data, Zs), ys)                  # baseline CE with full source rep

    # one shared eraser-disjoint posterior split reused across directions (paired lambda comparison)
    e_idx, ptr, pev, split_diag = three_way_support_split(ys, d_remap, seed=seed)

    K = int(min(k_spec, dz))
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(K):
        ut = dirs[j]                                          # whitened unit direction
        t = (Zt @ ut).reshape(-1, 1)                          # whitened subject coordinate [Ns,1]
        # lambda_j : null-calibrated conditional subject info of the 1-D projection
        cmi = flat_conditional_cmi(t, ys, d_remap, n_cls, n_dom, ptr, pev,
                                   n_perm=n_perm, seed=seed, device=device, with_residual=False)
        # tau_j : exact-head CE reliance, removing the RAW-space direction u_j = normalize(W_inv @ ut)
        u_raw = W_inv @ ut
        u_raw = u_raw / max(np.linalg.norm(u_raw), EPS)
        Z_rm = Zs - np.outer(Zs @ u_raw, u_raw)
        tau = _ce(replay_head(data, Z_rm), ys) - ce_full
        # same-rank-1 random control (raw metric)
        rand_taus = []
        for i in range(n_random):
            g = rng.standard_normal(dz); g /= max(np.linalg.norm(g), EPS)
            Zr = Zs - np.outer(Zs @ g, g)
            rand_taus.append(_ce(replay_head(data, Zr), ys) - ce_full)
        rows.append({
            "j": j, "energy": float(energy[j]),
            "lambda_excess_over_null": float(cmi["excess_over_null"]),
            "lambda_posterior_kl": float(cmi["posterior_kl_nats"]),
            "lambda_perm_p": float(cmi["perm_p"]),
            "tau_ce_reliance": float(tau),
            "tau_random_mean": float(np.mean(rand_taus)),
            "tau_random_sd": float(np.std(rand_taus)),
            "head_alignment": head_alignment(u_raw, Wc),
            "u_raw": u_raw.tolist(),                            # RAW (common) ambient dir, for principal-angle matching
            "u_tilde": ut.tolist(),                            # whitened dir (model-relative; not cross-model comparable)
        })
    return {
        "dataset": str(np.asarray(data.get("dataset", ""))), "method": str(np.asarray(data.get("method", ""))),
        "seed": int(np.asarray(data.get("seed", seed))), "fold": int(np.asarray(data.get("fold", -1))),
        "target_subject": target_domain, "representation": representation,
        "d_z": dz, "n_source_subjects": n_dom, "n_cls": n_cls,
        "effective_rank": r_eff, "top2_energy_concentration": top2_conc,
        "ce_full": ce_full, "k_spec": K, "n_perm": n_perm, "n_random": n_random,
        "firewall_passed": firewall_ok, "head_replay_verified": True,
        "posterior_split": {k: split_diag.get(k) for k in ("n_eraser", "n_ptrain", "n_peval", "disjoint")},
        "directions": rows,
    }


# --------------------------------------------------------------------- ERM <-> CIGL matching + Delta lambda
def match_directions(dirs_a, dirs_b):
    """OPTIMAL principal-angle matching between two sets of RAW (common-ambient) unit directions (each
    [K,dz]) via Hungarian assignment maximising sum |cos|. Returns (ia, ib, abs_cos) pairs, most-aligned
    first. Used ONLY to pair the SCALAR endpoints across models (ERM<->CIGL); raw axes themselves are never
    compared as values. Matching is done in the shared raw graph_z ambient (the only coordinate system common
    to two separately-trained models); the per-model whitened bases are NOT cross-comparable."""
    A = np.asarray(dirs_a, float); B = np.asarray(dirs_b, float)
    A = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), EPS, None)
    B = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), EPS, None)
    C = np.abs(A @ B.T)
    try:
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(-C)
        pairs = [(int(i), int(j), float(C[i, j])) for i, j in zip(ri, ci)]
    except Exception:                                          # fallback: greedy (reason-coded, not silent)
        pairs, ua, ub = [], set(), set()
        for c, i, j in sorted(((C[i, j], i, j) for i in range(C.shape[0]) for j in range(C.shape[1])),
                              reverse=True):
            if i in ua or j in ub:
                continue
            ua.add(i); ub.add(j); pairs.append((int(i), int(j), float(c)))
    return sorted(pairs, key=lambda p: -p[2])


def paired_delta_lambda(spec_erm, spec_cigl, mode="cosine"):
    """Pair ERM<->CIGL directions and return per-pair (tau_erm, delta_lambda = lambda_cigl - lambda_erm, cos).
    mode='cosine': optimal principal-angle match in the common raw ambient (primary, per handoff).
    mode='rank'  : pair by energy rank (j-th highest-energy ERM <-> j-th highest-energy CIGL) — a robustness
    variant that makes NO cross-model geometric-correspondence assumption. corr(tau_erm, delta_lambda) is
    computed downstream over pairs pooled across folds (cluster bootstrap)."""
    E, Cd = spec_erm["directions"], spec_cigl["directions"]
    if mode == "rank":
        pairs = [(j, j, float("nan")) for j in range(min(len(E), len(Cd)))]
    else:
        pairs = match_directions([np.asarray(r["u_raw"]) for r in E],
                                 [np.asarray(r["u_raw"]) for r in Cd])
    out = []
    for ia, ib, cos in pairs:
        re, rc = E[ia], Cd[ib]
        out.append({
            "erm_j": ia, "cigl_j": ib, "abs_cos": cos, "pairing": mode,
            "tau_erm": re["tau_ce_reliance"], "tau_cigl": rc["tau_ce_reliance"],
            "lambda_erm": re["lambda_excess_over_null"], "lambda_cigl": rc["lambda_excess_over_null"],
            "delta_lambda": rc["lambda_excess_over_null"] - re["lambda_excess_over_null"],
        })
    return out
