"""Cross-fitted hierarchical conditional-leakage estimate on a FROZEN encoder (review P0-7).

Two correctness requirements the random-split version violated:

1. **Grouped split.** EEG windows from the same recording/session are autocorrelated, so a
   plain trial-level split lets the critic memorise recording-specific structure that
   "generalises" to adjacent windows in the other fold, inflating apparent leakage. We
   split by a GROUP factor chosen so the measured factor D_j still appears in both folds
   but via DIFFERENT recordings:
       site leakage    -> group by subject (train/eval subjects disjoint within a site)
       subject leakage -> group by session
       session leakage -> within-session contiguous temporal halves (its own child unit)

2. **Refit-under-permutation null.** The permutation null must re-train the critic on
   permuted labels, otherwise it is not the estimator's null distribution. We permute D_j
   within (Y, Pa) in BOTH folds, refit, and evaluate -- the full null of the cross-fitted
   estimator.

Reports per factor the SIGNED held-out I_hat_j = H(D_j|Y,Pa) - CE_psi, the null 95%
quantile and the excess above it (no truncation of negatives -> no upward bias).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from h2cmi.cmi.hierarchical import ConditionalDomainCritic, reference_conditional_entropy
from h2cmi.domains import DomainDAG, DomainLabels


# ----------------------------------------------------------------- grouping
def _child_factor(dag: DomainDAG, name: str) -> str | None:
    """The factor whose parents include `name` (the finer nesting unit), if any."""
    for f in dag.factors:
        if name in f.parents:
            return f.name
    return None


def _grouped_folds(domains: DomainLabels, f_name: str, dag: DomainDAG, rng) -> np.ndarray:
    """Return a fold id in {0,1} per sample, grouped so D_{f_name} appears in both folds
    but no child-recording is split across them."""
    n = domains.n
    child = _child_factor(dag, f_name)
    if child is not None:
        groups = domains.factor(child)
        fac = domains.factor(f_name)
        fold = np.zeros(n, dtype=int)
        # STRATIFIED by the measured factor: for each f-level, split ITS child-groups
        # across folds, so every f-level appears in both folds (no f-level vanishes ->
        # avoids pathological held-out CE) while no child-recording is split.
        for lev in np.unique(fac):
            cg = np.unique(groups[fac == lev])
            cg = cg[rng.permutation(len(cg))]
            half = max(1, len(cg) // 2)
            fold1 = set(cg[half:].tolist()) if len(cg) > 1 else set()
            for g in cg:
                fold[(groups == g)] = 1 if g in fold1 else 0
        return fold
    # finest factor: within-(f-level) contiguous temporal halves
    fac = domains.factor(f_name)
    fold = np.zeros(n, dtype=int)
    for lev in np.unique(fac):
        pos = np.where(fac == lev)[0]                  # array order ~ temporal order
        fold[pos[len(pos) // 2:]] = 1
    return fold


def _perm_within(d: np.ndarray, ctx: np.ndarray, rng) -> np.ndarray:
    out = d.copy()
    for c in np.unique(ctx):
        idx = np.where(ctx == c)[0]
        if len(idx) > 1:
            out[idx] = rng.permutation(out[idx])
    return out


# ----------------------------------------------------------------- critic fit/eval
def _fit_critic(Z, y, pk, d, n_levels, n_cls, n_pa, device, epochs, lr, seed):
    torch.manual_seed(seed)
    crit = ConditionalDomainCritic(Z.shape[1], n_levels, n_cls, n_pa).to(device)
    opt = torch.optim.Adam(crit.parameters(), lr=lr)
    Zt = torch.as_tensor(Z, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.long, device=device)
    pkt = torch.as_tensor(pk, dtype=torch.long, device=device)
    dt = torch.as_tensor(d, dtype=torch.long, device=device)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(crit(Zt, yt, pkt), dt).backward()
        opt.step()
    return crit


@torch.no_grad()
def _signed(crit, Z, y, pk, d, n_levels, n_cls, n_pa, device):
    logits = crit(torch.as_tensor(Z, dtype=torch.float32, device=device),
                  torch.as_tensor(y, dtype=torch.long, device=device),
                  torch.as_tensor(pk, dtype=torch.long, device=device))
    ce = float(F.cross_entropy(logits, torch.as_tensor(d, dtype=torch.long, device=device)))
    h_ref = reference_conditional_entropy(d, y, pk, n_levels, n_cls, n_pa)
    acc = float((logits.argmax(1).cpu().numpy() == d).mean())
    return h_ref - ce, ce, h_ref, acc


def crossfit_conditional_leakage(Z: np.ndarray, y: np.ndarray, domains: DomainLabels,
                                 dag: DomainDAG, n_classes: int, device: str = "cpu",
                                 epochs: int = 120, lr: float = 2e-3, n_perm: int = 0,
                                 null_epochs: int | None = None, seed: int = 0) -> dict:
    """Per-factor grouped cross-fitted leakage with a refit-under-permutation null."""
    rng = np.random.default_rng(seed)
    null_epochs = null_epochs if null_epochs is not None else max(40, epochs // 2)
    out = {}
    for f in dag.penalised_factors():
        n_levels = f.n_levels
        pidx = dag.parent_indices(f.name)
        n_pa = int(np.prod([dag.factors[j].n_levels for j in pidx])) if pidx else 1
        d_all = domains.factor(f.name)
        pk_all = domains.parent_key(f.name)
        ctx_all = y.astype(np.int64) * n_pa + pk_all.astype(np.int64)
        fold = _grouped_folds(domains, f.name, dag, rng)

        def _crossfit(d_use, ep, sd):
            vals = []
            for fit_fold in (0, 1):
                fit = fold == fit_fold
                ev = fold != fit_fold
                if fit.sum() < 4 or ev.sum() < 4:
                    continue
                crit = _fit_critic(Z[fit], y[fit], pk_all[fit], d_use[fit],
                                   n_levels, n_classes, n_pa, device, ep, lr, sd + fit_fold)
                vals.append(_signed(crit, Z[ev], y[ev], pk_all[ev], d_use[ev],
                                    n_levels, n_classes, n_pa, device))
            return vals

        real = _crossfit(d_all, epochs, seed)
        if not real:
            out[f.name] = dict(I_hat=float("nan"), budget=f.budget); continue
        I_hat = float(np.mean([v[0] for v in real]))
        rec = dict(I_hat=I_hat, ce=float(np.mean([v[1] for v in real])),
                   h_ref=float(np.mean([v[2] for v in real])),
                   cond_dom_acc=float(np.mean([v[3] for v in real])),
                   budget=f.budget, group_unit=_child_factor(dag, f.name) or "temporal_half")
        if n_perm:
            nulls = []
            for k in range(int(n_perm)):
                d_perm = _perm_within(d_all, ctx_all, rng)        # break Z<->D, keep (Y,Pa)
                vals = _crossfit(d_perm, null_epochs, seed + 1000 + k)  # REFIT on permuted
                if vals:
                    nulls.append(float(np.mean([v[0] for v in vals])))
            if nulls:
                nulls = np.asarray(nulls)
                q = float(np.quantile(nulls, 0.95))
                rec.update(null_q95=q, null_mean=float(nulls.mean()),
                           null_std=float(nulls.std()), n_perm=len(nulls),
                           excess=float(max(I_hat - q, 0.0)))
        out[f.name] = rec
    return out


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    from h2cmi.data.eeg_simulator import EEGSimulator, ShiftSpec
    from h2cmi.domains import compact_domain_labels
    sim = EEGSimulator(3, 8, 64, shift=ShiftSpec(cov=1.0, prior=0.3), seed=0).sample(4, 3, 4, 30)
    dag, dom, _ = compact_domain_labels(sim.domains)
    rng = np.random.default_rng(0)
    site = dom.factor("site")
    n_site = int(site.max()) + 1

    def build_Z(site_strength):
        Z = np.zeros((sim.n, 12), dtype=np.float32)
        Z[np.arange(sim.n), sim.y] = 2.0                       # class signal
        # SITE-shared signal (same dim for all subjects of a site) -> subject-generalising
        Z[np.arange(sim.n), 3 + site] += site_strength
        return Z + 0.3 * rng.standard_normal(Z.shape).astype(np.float32)

    print("--- planted site-shared leakage (should be DETECTED on subject-grouped split) ---")
    leak = crossfit_conditional_leakage(build_Z(2.0), sim.y, dom, dag, 3, n_perm=10, seed=0)
    for fct, r in leak.items():
        print(f"{fct:8s} I_hat={r['I_hat']:+.3f} group={r.get('group_unit')} "
              f"null_q95={r.get('null_q95', float('nan')):+.3f} excess={r.get('excess', 0):.3f}")
    assert leak["site"]["excess"] > 0.05, "failed to detect planted site leakage"

    print("--- control: no site signal (excess should be ~0) ---")
    leak0 = crossfit_conditional_leakage(build_Z(0.0), sim.y, dom, dag, 3, n_perm=10, seed=1)
    print(f"site excess (control) = {leak0['site']['excess']:.3f}")
    assert leak0["site"]["excess"] < 0.3, "false-positive site leakage in control"
    print("leakage group-split + refit-null self-test PASSED")
