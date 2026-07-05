"""V2 semi-synthetic latents: take a REAL frozen EEG latent (Z_source/target) and APPEND an m-dim nuisance
block driven by an injected binary nuisance variable z, whose relationship to the label y and to the
source->target shift is ground-truth-controlled per world (see notes/V2_SEMI_SYNTHETIC_DESIGN.md).

The eraser family erases D = z (we return z as the domain array). Real subjects are returned separately for
leave-one-source-subject-out (LOSO) benefit grouping. Target y/z are used ONLY for the post-hoc audit.

inject(world, Zs, ys, subj, Zt, yt, alpha, beta, phi, seed, m, noise) ->
    dict(Zs2, Zt2, z_src, z_tgt, grp_subj, world, ground_truth)
"""
from __future__ import annotations
import numpy as np


def _reliability(subs, phi, rng):
    """Assign each source subject aligned(z=y) or reversed(z=1-y). Returns {subject: +1 aligned / -1 reversed}."""
    order = list(rng.permutation(subs))
    n_rev = int(round(phi * len(order)))
    rev = set(order[:n_rev])
    return {s: (-1 if s in rev else +1) for s in subs}


def inject(world, Zs, ys, subj, Zt, yt, alpha=1.0, beta=1.0, phi=0.35, seed=0, m=4, noise=0.1,
           variantA="aligned_noise"):
    rng = np.random.default_rng(10_000 + seed)
    ys = ys.astype(int); yt = yt.astype(int)
    # z-score the real block so alpha is on a comparable scale (fit on source, apply to both)
    mu = Zs.mean(0, keepdims=True); sd = Zs.std(0, keepdims=True) + 1e-8
    Zs_n = (Zs - mu) / sd; Zt_n = (Zt - mu) / sd
    u = np.ones(m) / np.sqrt(m)
    subs = sorted(set(subj.tolist()))
    sy = (2 * ys - 1).astype(float); ty = (2 * yt - 1).astype(float)

    if world == "A":                               # beneficial spurious nuisance (see variantA)
        gt = "beneficial"
        if variantA == "reversed":                 # [DEPRECATED] symmetric -> net LOSO benefit <=0 (design review)
            rel = _reliability(subs, phi, rng)
            z_src = np.array([ys[i] if rel[subj[i]] > 0 else 1 - ys[i] for i in range(len(ys))], int)
            z_tgt = (1 - yt).astype(int)
        elif variantA in ("aligned_noise", "aligned_noise_flip"):
            # A MINORITY (f_align) of source subjects carry the spurious shortcut (z=y); for the majority the
            # nuisance is NOISE (z independent of y). Pooled corr(z,y)=f_align>0 so the head USES u, but for the
            # noise-majority the head injects w_u*noise -> mispredicts held-out noise subjects -> erasing u helps
            # them. Net source-LOSO benefit is positive when f_align is small (asymmetric: noise-majority gain vs
            # aligned-minority loss). Target: noise (aligned_noise) or reversed (aligned_noise_flip) -> misleading.
            f_align = phi
            order = list(rng.permutation(subs)); n_al = max(1, int(round(f_align * len(order))))
            aligned = set(order[:n_al])
            z_src = np.array([ys[i] if subj[i] in aligned else int(rng.integers(0, 2))
                              for i in range(len(ys))], int)
            z_tgt = ((1 - yt) if variantA == "aligned_noise_flip"
                     else rng.integers(0, 2, len(yt))).astype(int)
        else:
            raise ValueError("variantA=%s" % variantA)
    elif world == "B":                             # unsafe: nuisance == true label -> erasing z destroys task
        z_src = ys.copy(); z_tgt = yt.copy()
        gt = "unsafe"
    elif world == "C":                             # useless: z independent of y (same law source & target)
        z_src = rng.integers(0, 2, len(ys)); z_tgt = rng.integers(0, 2, len(yt))
        gt = "neutral"
    else:
        raise ValueError(world)

    def block(z, n):
        return (alpha * (2 * z - 1)).astype(float)[:, None] * u[None, :] + noise * rng.standard_normal((n, m))

    # World B optionally strengthens the confound with an explicit beta*sy term on the same direction
    Ns = block(z_src, len(ys)); Nt = block(z_tgt, len(yt))
    if world == "B":
        Ns = Ns + beta * sy[:, None] * u[None, :]
        Nt = Nt + beta * ty[:, None] * u[None, :]

    Zs2 = np.concatenate([Zs_n, Ns], axis=1)
    Zt2 = np.concatenate([Zt_n, Nt], axis=1)
    return {"Zs2": Zs2, "Zt2": Zt2, "z_src": z_src, "z_tgt": z_tgt,
            "grp_subj": subj, "world": world, "ground_truth": gt,
            "params": {"alpha": alpha, "beta": beta, "phi": phi, "m": m, "noise": noise, "variantA": variantA}}
