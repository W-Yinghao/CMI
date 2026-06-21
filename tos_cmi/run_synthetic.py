"""End-to-end demonstration of TOS-CMI on the controllable synthetic world.

Three things, all on the simulator (no GPU, no EEG):

  1. overlap sweep   -- as task/domain entanglement grows, the selected nuisance
                        subspace shrinks and eventually the method REFUSES (identity);
                        at low overlap it recovers the planted subspace and the
                        Bayes-risk-preservation proposition holds.
  2. stability gate  -- the termination criterion: is the selected subspace the same
                        across seeds? (principal-angle overlap)
  3. train compare   -- ERM vs GLOBAL LPC (erase all I(Z;D|Y)) vs SELECTIVE: global LPC
                        over-erases and drops label accuracy once subspaces touch;
                        selective preserves it. This is the synthetic analog of the
                        "global LPC collapses TSMNet" counterexample.

    conda run -n icml python -m tos_cmi.run_synthetic
"""
from __future__ import annotations
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FisherConfig, SubspaceConfig, PenaltyConfig
from .data.synthetic import SynthSpec, make
from .subspace import SubspaceSelector
from .selective_cmi import SelectivePenalty
from .eval.proposition import bayes_risk_check
from .eval.stability import subspace_overlap, selection_stability


def overlap_sweep(overlaps, seed=0):
    print("\n=== 1. overlap sweep (selection + proposition) ===")
    print(f"{'ovlp':>5} {'k':>3} {'ident':>6} {'recov':>6} "
          f"{'acc_full':>8} {'acc_task':>8} {'drop':>6} "
          f"{'leak_full':>9} {'leak_nuis':>9} {'leak_task':>9}")
    for ov in overlaps:
        data = make(SynthSpec(overlap=ov), seed=seed)
        sel = SubspaceSelector(data["Z"].shape[1], data["spec"].n_cls, data["spec"].n_dom)
        sel.refresh(torch.tensor(data["Z"]), torch.tensor(data["y"]), torch.tensor(data["d"]))
        recov = subspace_overlap(sel.report.basis, data["nuisance_basis"]) if not sel.is_identity else 0.0
        p = bayes_risk_check(data, sel, seed=seed)
        print(f"{ov:5.2f} {p['k']:3d} {str(p['is_identity']):>6} {recov:6.2f} "
              f"{p['acc_full']:8.3f} {p['acc_task']:8.3f} {p['acc_drop']:6.3f} "
              f"{p['leak_full']:9.3f} {p['leak_nuis']:9.3f} {p['leak_task']:9.3f}")


def stability_gate(overlap=0.0, seeds=(0, 1, 2, 3, 4)):
    print(f"\n=== 2. stability gate (overlap={overlap}, {len(seeds)} sample draws of one world) ===")
    bases = []
    for s in seeds:
        data = make(SynthSpec(overlap=overlap), seed=s, struct_seed=0)  # fixed world, vary draw
        sel = SubspaceSelector(data["Z"].shape[1], data["spec"].n_cls, data["spec"].n_dom)
        sel.refresh(torch.tensor(data["Z"]), torch.tensor(data["y"]), torch.tensor(data["d"]))
        bases.append(sel.report.basis)
    st = selection_stability(bases)
    print(f"  mean pairwise subspace overlap = {st['mean_overlap']:.3f} "
          f"(min {st['min_overlap']:.3f})   k = {st['k_values']}   n_identity={st['n_identity']}")
    print("  GATE:", "STABLE -> proceed" if st['mean_overlap'] > 0.8 and st['n_identity'] == 0
          else "UNSTABLE -> terminate direction")


# --- tiny trainable encoder so we can contrast erasure regimes -------------------
class LinearEnc(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.enc = nn.Linear(d, k)
        self.head = None
        self.k = k

    def build_head(self, n_cls):
        self.head = nn.Linear(self.k, n_cls)
        return self


def _train_compare(data, mode, lam=1.0, epochs=120, seed=0):
    """mode in {erm, global, selective}. Returns (test_bAcc, leak_full_on_Z)."""
    torch.manual_seed(seed)
    Z, y, d = data["Z"], data["y"], data["d"]
    n_cls, n_dom = data["spec"].n_cls, data["spec"].n_dom
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Z)); cut = int(0.6 * len(Z))
    tr, te = idx[:cut], idx[cut:]
    Xtr = torch.tensor(Z[tr]); ytr = torch.tensor(y[tr]); dtr = torch.tensor(d[tr])
    Xte = torch.tensor(Z[te]); yte = torch.tensor(y[te])

    enc = LinearEnc(Z.shape[1], Z.shape[1]).build_head(n_cls)
    sp = SelectivePenalty(enc.k, n_cls, n_dom, priors=(y[tr], d[tr]),
                          pcfg=PenaltyConfig(lam=lam))
    opt = torch.optim.Adam(list(enc.parameters()) + list(sp.critic.parameters()), lr=1e-2)

    for ep in range(epochs):
        zt = enc.enc(Xtr)
        if mode in ("global", "selective") and ep % 5 == 0:
            sp.refresh(zt.detach(), ytr, dtr)
        # Step A: fit critic on detached features (full Z for global, P_N Z for selective)
        opt.zero_grad()
        la = None
        if mode == "global":
            la = F.cross_entropy(sp.critic(zt.detach(), ytr), dtr)
        elif mode == "selective":
            la = sp.posterior_loss(zt.detach(), ytr, dtr)
        if la is not None and la.requires_grad:      # erm / identity have no Step-A grad
            la.backward(); opt.step()
        # Step B: task CE + penalty (critic frozen via no grad on its params here)
        opt.zero_grad()
        zt = enc.enc(Xtr)
        loss = F.cross_entropy(enc.head(zt), ytr)
        if mode == "global":
            logits = sp.critic(zt, ytr)
            logq = F.log_softmax(logits, 1)
            loss = loss + lam * (logq.exp() * (logq - sp.log_pi_y[ytr])).sum(1).mean()
        elif mode == "selective":
            loss = loss + sp.penalty(zt, ytr, dtr)
        loss.backward()
        for p_ in sp.critic.parameters():            # don't let Step B move the critic
            p_.grad = None
        opt.step()

    with torch.no_grad():
        pred = enc.head(enc.enc(Xte)).argmax(1).numpy()
    bacc = np.mean([(pred[(yte.numpy() == c)] == c).mean()
                    for c in range(n_cls) if (yte.numpy() == c).sum() > 0])
    return float(bacc)


def train_compare(overlap=0.5, lam=1.0, seed=0):
    print(f"\n=== 3. ERM vs GLOBAL-LPC vs SELECTIVE (overlap={overlap}, lam={lam}) ===")
    data = make(SynthSpec(overlap=overlap), seed=seed)
    for mode in ("erm", "global", "selective"):
        bacc = _train_compare(data, mode, lam=lam, seed=seed)
        print(f"  {mode:>9} : test bAcc = {bacc:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=1.0)
    args = ap.parse_args()
    overlap_sweep([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], seed=args.seed)
    stability_gate(overlap=0.0)
    train_compare(overlap=args.overlap, lam=args.lam, seed=args.seed)


if __name__ == "__main__":
    main()
