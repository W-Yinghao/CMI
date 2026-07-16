"""CMI-Trace — Spurious-task-bearing synthetic DGP (the third synthetic world, for the DG-erasure oracle).

The exact-head-null oracle + TTE V1 showed that the safely-removable subject leakage sits mostly in the task
head's NULLSPACE (functionally unused) — so removing it cannot improve DG. The DG-relevant object is
different: a subject-rich subspace that IS used by the source predictor but is UNSTABLE across subjects
(hence harmful to an unseen subject). This DGP builds exactly that, to prove `safe-erasure oracle` ≠
`DG-erasure oracle`.

  Z = [ Z_inv | Z_spur | Z_id ]
   * Z_inv  : shifts with Y the SAME way in every domain -> a STABLE, invariant task predictor.
   * Z_spur : shifts with Y but with a per-DOMAIN sign s_d in {+1,-1} -> predicts Y in (most) source domains
              yet FLIPS on other subjects (the held-out target) -> a subject-unstable task-bearing SHORTCUT.
   * Z_id   : shifts with the DOMAIN only (NOT Y) -> pure subject identity, not used by the task.

Expected oracle behaviour (the whole point):
   - safe / exact-null / CMI-only selector deletes Z_id  -> task unchanged, TARGET unchanged (no DG gain);
   - DG-erasure oracle deletes Z_spur                    -> source task may dip, TARGET IMPROVES;
   - a source-only meta selector (source-LOSO) should pick Z_spur (its instability is visible within source).
Pure numpy.
"""
from __future__ import annotations
import numpy as np


def make_spurious_task_dgp(n_domains=8, per_domain=200, n_cls=2, seed=0,
                           d_inv=2, d_spur=2, d_id=2, d_noise=2,
                           inv_strength=0.3, spur_strength=3.0, id_strength=3.0, noise=0.8,
                           target_dom=None, n_minority_source=2):
    """Return dict with Z [N,D], y [N], d [N] (domain=subject), and block index lists inv/spur/id/noise.

    Spurious sign s_d: the SOURCE has a CONSISTENT majority sign (+1) so a pooled source head LEANS on the
    (stronger) shortcut Z_spur; the TARGET is FLIPPED (-1) so the shortcut is harmful there. `n_minority_source`
    source subjects also carry -1 so the instability is DETECTABLE by source-LOSO. spur_strength > inv_strength
    makes the shortcut the model's preferred (easier) feature — the classic spurious-correlation regime."""
    rng = np.random.default_rng(seed)
    D = d_inv + d_spur + d_id + d_noise
    i_inv = list(range(0, d_inv))
    i_spur = list(range(d_inv, d_inv + d_spur))
    i_id = list(range(d_inv + d_spur, d_inv + d_spur + d_id))
    i_noise = list(range(d_inv + d_spur + d_id, D))
    target_dom = (n_domains - 1) if target_dom is None else int(target_dom)
    s_dom = np.ones(n_domains)                                # source majority +1
    s_dom[target_dom] = -1.0                                  # target flipped (harmful shortcut)
    src = [x for x in range(n_domains) if x != target_dom]
    for mv in rng.choice(src, size=min(n_minority_source, len(src)), replace=False):
        s_dom[mv] = -1.0                                      # a few source subjects flipped -> source-LOSO detectable
    id_off = rng.standard_normal((n_domains, d_id)) * id_strength   # per-domain identity offset
    Z, y, d = [], [], []
    for dom in range(n_domains):
        yy = rng.integers(0, n_cls, per_domain)
        ysign = (2.0 * yy - (n_cls - 1)) / max(1, n_cls - 1)  # in [-1,1]
        z = noise * rng.standard_normal((per_domain, D))
        for j in i_inv:
            z[:, j] += inv_strength * ysign                  # stable invariant predictor
        for j in i_spur:
            z[:, j] += spur_strength * s_dom[dom] * ysign    # subject-unstable shortcut
        for jj, j in enumerate(i_id):
            z[:, j] += id_off[dom, jj]                        # pure subject identity (no Y)
        Z.append(z); y.append(yy); d.append(np.full(per_domain, dom))
    return {"Z": np.vstack(Z), "y": np.concatenate(y), "d": np.concatenate(d),
            "inv": i_inv, "spur": i_spur, "id": i_id, "noise": i_noise, "n_cls": n_cls,
            "n_domains": n_domains, "spurious_sign": s_dom.tolist(), "D": D, "target_dom": target_dom}
