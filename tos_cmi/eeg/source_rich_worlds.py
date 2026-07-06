"""Fork 2 Phase 1A --- source-rich World A (the constructive witness for Proposition 2). Unlike V2 World A
(target-beneficial shift is source-INVISIBLE -> gate must abstain), here the target's shift regime IS
REPRESENTED among the source environments, so a leave-one-ENVIRONMENT-out benefit can see it -> gate should
ACCEPT (safely). The contrast: E0 leave-one-SUBJECT-out misses it (subjects within a regime share the shortcut);
E_oracle leave-one-REGIME-out sees it; E2/E4/E5 test whether source-only environment discovery recovers it.

Construction. Each source subject gets a REGIME r_s in {aligned, reversed, noisy} (fractions imbalanced toward
aligned so the pooled head USES the shortcut). Trial shortcut z: aligned -> z=y, reversed -> z=1-y, noisy ->
random. Injected nuisance N = alpha*(2z-1)*u + noise, appended to z-scored Z. The eraser erases D_nuis = z.
Target subject regime = reversed (REPRESENTED in source) -> the aligned-majority head misleads the target ->
erasing D_nuis helps the target, and holding out the reversed regime reproduces that on source.

NOTE: the erased concept `z_src` is the KNOWN INJECTED nuisance factor D_nuis = z, NOT the original EEG subject
identity (D = subject). This tests source-rich certification WHEN THE NUISANCE IS KNOWN; it does NOT solve
nuisance discovery in real EEG. Environments E0-E5 are SEPARATE from D_nuis. See SOURCE_RICH_PHASE1A_VERDICT.md.
"""
from __future__ import annotations
import numpy as np

from tos_cmi.eeg.source_environments import env_labels as _env_labels, random_partition_matched

REGIMES = ["aligned", "reversed", "noisy"]


def inject_source_rich(Zs, ys, subj, Zt, yt, alpha=1.0, m=4, noise=0.1, seed=0,
                       frac=(0.5, 0.3, 0.2), target_regime="reversed"):
    """Return dict(Zs2, Zt2, z_src [shortcut=domain to erase], regime_src [per-trial regime id], z_tgt,
    grp_subj [subject], target_regime). frac = (aligned, reversed, noisy) subject fractions."""
    rng = np.random.default_rng(20_000 + seed)
    ys = ys.astype(int); yt = yt.astype(int)
    mu = Zs.mean(0, keepdims=True); sd = Zs.std(0, keepdims=True) + 1e-8
    Zs_n = (Zs - mu) / sd; Zt_n = (Zt - mu) / sd
    u = np.ones(m) / np.sqrt(m)
    subs = sorted(set(subj.tolist()))
    order = list(rng.permutation(subs))
    n_al = int(round(frac[0] * len(order))); n_rev = int(round(frac[1] * len(order)))
    reg = {}
    for i, s in enumerate(order):
        reg[s] = 0 if i < n_al else (1 if i < n_al + n_rev else 2)     # 0 aligned / 1 reversed / 2 noisy
    regime_src = np.array([reg[s] for s in subj], int)
    z_src = np.empty(len(ys), int)
    for i in range(len(ys)):
        r = regime_src[i]
        z_src[i] = ys[i] if r == 0 else (1 - ys[i] if r == 1 else int(rng.integers(0, 2)))
    tr = {"aligned": 0, "reversed": 1, "noisy": 2}[target_regime]
    z_tgt = (yt if tr == 0 else (1 - yt if tr == 1 else rng.integers(0, 2, len(yt)))).astype(int)

    def block(z, n):
        return (alpha * (2 * z - 1)).astype(float)[:, None] * u[None, :] + noise * rng.standard_normal((n, m))
    Zs2 = np.concatenate([Zs_n, block(z_src, len(ys))], axis=1)
    Zt2 = np.concatenate([Zt_n, block(z_tgt, len(yt))], axis=1)
    return {"Zs2": Zs2, "Zt2": Zt2, "z_src": z_src, "regime_src": regime_src, "z_tgt": z_tgt,
            "grp_subj": subj, "target_regime": tr, "params": {"alpha": alpha, "m": m, "noise": noise, "frac": frac}}


def _augmentation_env(Zs2, subj, k, seed):
    """E5: source-only augmentation-defined environments -- cluster subjects by their feature statistics under
    a fixed covariance-scaling perturbation (no target). A proxy for augmentation-defined source environments."""
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(500 + seed)
    scale = 1.0 + 0.5 * rng.standard_normal(Zs2.shape[1])          # fixed per-feature scaling (the "augmentation")
    Zp = Zs2 * scale
    subs = sorted(set(subj.tolist()))
    F = np.array([np.concatenate([Zp[subj == s].mean(0), Zp[subj == s].std(0)]) for s in subs])
    F = (F - F.mean(0)) / (F.std(0) + 1e-8)
    kk = int(min(k, len(subs)))
    if kk < 2:
        return None, "too few subjects"
    lab = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit_predict(F)
    s2c = {s: int(lab[i]) for i, s in enumerate(subs)}
    return np.array([s2c[s] for s in subj], int), None


def smoke_environments(name, Zs2, ys, subj, regime_src, k=8, seed=0):
    """Return (per-trial env labels, reason). Deployable source-only: subject/covariance_cluster/margin_cluster/
    augmentation_shift/random. Diagnostic: oracle (= ground-truth regime; uses injected knowledge)."""
    if name == "subject":
        return subj.copy(), None                                   # E0
    if name == "oracle":
        return regime_src.copy(), None                             # E_oracle (DIAGNOSTIC: ground-truth regime)
    if name in ("covariance_cluster", "margin_cluster"):
        return _env_labels(name, Zs2, ys, subj, session=None, k=k, seed=seed)   # E2 / E4
    if name == "augmentation_shift":
        return _augmentation_env(Zs2, subj, k, seed)               # E5
    if name == "random":
        base, _ = smoke_environments("oracle", Zs2, ys, subj, regime_src, k, seed)  # match #environments
        return random_partition_matched(base, seed), None
    raise ValueError("unknown environment '%s'" % name)


DEPLOYABLE_ENVS = ["subject", "covariance_cluster", "margin_cluster", "augmentation_shift", "random"]
DIAGNOSTIC_ENVS = ["oracle"]
