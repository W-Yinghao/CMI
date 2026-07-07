"""End-to-end demonstration of TOS-CMI on the controllable synthetic world (no GPU, no EEG).

  1. overlap sweep   -- leakage-free projection ablation (P_N estimated on selector-train,
                        metrics on a disjoint probe-test): label accuracy preserved while the
                        linear conditional-domain advantage is removed, degrading to identity
                        as task/domain entangle. Reports precision/recall vs the planted span.
  2. stability gate  -- selection consistency across SAMPLE DRAWS of one fixed world, gated on
                        the dimension-sensitive projection distance (not containment cos^2).
  3. train compare   -- ERM vs GLOBAL LPC (erase all I(Z;D|Y)) vs SELECTIVE, over MULTIPLE
                        seeds and lambdas, reporting BOTH test bAcc AND post-training linear
                        domain advantage. Synthetic analog of "global LPC over-erases".

Writes a machine-readable artifact (env + seeds + all numbers) via --out.

    conda run -n icml python -m tos_cmi.run_synthetic --out results/tos_cmi_synthetic.json

CAVEATS (see THEORY.md): the Fishers are first-moment (mean-scatter) proxies, not CMI; this
is a SYNTHETIC proof-of-concept aligned with the method's linear-mean assumptions; it is not
real-EEG evidence and `domadv_*` is a linear probe advantage, not mutual information.
"""
from __future__ import annotations
import argparse
import json
import os
import platform
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy

from .config import PenaltyConfig
from .data.synthetic import SynthSpec, make
from .subspace import SubspaceSelector
from .selective_cmi import SelectivePenalty
from .eval.projection_ablation import linear_probe_projection_ablation, _cond_domain_adv
from .eval.stability import selection_stability, precision_recall


def _env():
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def overlap_sweep(overlaps, seed=0):
    print("\n=== 1. overlap sweep (leakage-free projection ablation) ===")
    print(f"{'ovlp':>5} {'k':>3} {'ident':>6} {'prec':>5} {'rec':>5} "
          f"{'acc_full':>8} {'acc_task':>8} {'drop':>6} "
          f"{'dadv_full':>9} {'dadv_nuis':>9} {'dadv_task':>9}")
    rows = []
    for ov in overlaps:
        data = make(SynthSpec(overlap=ov), seed=seed)
        p, sel = linear_probe_projection_ablation(data, seed=seed)
        pr = (precision_recall(sel.report.basis, data["nuisance_basis"])
              if not sel.is_identity else {"precision": 0.0, "recall": 0.0})
        row = {"overlap": ov, **p, "precision": pr["precision"], "recall": pr["recall"]}
        rows.append(row)
        print(f"{ov:5.2f} {p['k']:3d} {str(p['is_identity']):>6} "
              f"{pr['precision']:5.2f} {pr['recall']:5.2f} "
              f"{p['acc_full']:8.3f} {p['acc_task']:8.3f} {p['acc_drop']:6.3f} "
              f"{p['domadv_full']:9.3f} {p['domadv_nuis']:9.3f} {p['domadv_task']:9.3f}")
    return rows


def stability_gate(overlap=0.0, seeds=(0, 1, 2, 3, 4)):
    print(f"\n=== 2. stability gate (overlap={overlap}, {len(seeds)} sample draws of one world) ===")
    bases = []
    for s in seeds:
        data = make(SynthSpec(overlap=overlap), seed=s, struct_seed=0)  # fixed world, vary draw
        sel = SubspaceSelector(data["spec"].d, data["spec"].n_cls, data["spec"].n_dom)
        sel.refresh(torch.tensor(data["Z"]), torch.tensor(data["y"]), torch.tensor(data["d"]))
        bases.append(sel.report.basis)
    st = selection_stability(bases)
    print(f"  proj_dist_max = {st['proj_dist_max']:.3f} (flicker magnitude)  "
          f"nested cos2_min = {st['cos2_similarity_min']:.3f}  k = {st['k_values']}  "
          f"n_identity = {st['n_identity']}")
    print(f"  CORE gate: {'STABLE' if st['passed'] else 'UNSTABLE'}    "
          f"STRICT proj-dist gate: {'PASS' if st['proj_dist_strict_pass'] else 'NOT MET (eigengap/hysteresis TODO)'}")
    return st


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
    """mode in {erm, global, selective}. Returns dict(bAcc, domadv_full_posttrain)."""
    torch.manual_seed(seed)
    Z, y, d = data["Z"], data["y"], data["d"]
    n_cls, n_dom = data["spec"].n_cls, data["spec"].n_dom
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Z)); cut = int(0.6 * len(Z))
    tr, te = idx[:cut], idx[cut:]
    Xtr = torch.tensor(Z[tr]); ytr = torch.tensor(y[tr]); dtr = torch.tensor(d[tr])
    Xte = torch.tensor(Z[te]); yte = torch.tensor(y[te])

    enc = LinearEnc(Z.shape[1], Z.shape[1]).build_head(n_cls)
    sp = SelectivePenalty(enc.k, n_cls, n_dom, priors=(y[tr], d[tr]), pcfg=PenaltyConfig(lam=lam))
    opt = torch.optim.Adam(list(enc.parameters()) + list(sp.critic.parameters()), lr=1e-2)

    for ep in range(epochs):
        zt = enc.enc(Xtr)
        if mode in ("global", "selective") and ep % 5 == 0:
            sp.refresh(zt.detach(), ytr, dtr)
        opt.zero_grad()
        la = None
        if mode == "global":
            la = F.cross_entropy(sp.critic(zt.detach(), ytr), dtr)
        elif mode == "selective":
            la = sp.posterior_loss(zt.detach(), ytr, dtr)
        if la is not None and la.requires_grad:          # erm / identity have no Step-A grad
            la.backward(); opt.step()
        opt.zero_grad()
        zt = enc.enc(Xtr)
        loss = F.cross_entropy(enc.head(zt), ytr)
        if mode == "global":
            logq = F.log_softmax(sp.critic(zt, ytr), 1)
            loss = loss + lam * (logq.exp() * (logq - sp.log_pi_y[ytr])).sum(1).mean()
        elif mode == "selective":
            loss = loss + sp.penalty(zt, ytr, dtr)
        loss.backward()
        for p_ in sp.critic.parameters():                # don't let Step B move the critic
            p_.grad = None
        opt.step()

    with torch.no_grad():
        ztr = enc.enc(Xtr).numpy(); zte = enc.enc(Xte).numpy()
        pred = enc.head(torch.tensor(zte)).argmax(1).numpy()
    bacc = float(np.mean([(pred[yte.numpy() == c] == c).mean()
                          for c in range(n_cls) if (yte.numpy() == c).sum() > 0]))
    # post-training leakage on the LEARNED representation (linear domain advantage)
    domadv = _cond_domain_adv(ztr, y[tr], d[tr], zte, y[te], d[te], n_cls, n_dom, seed)
    return {"bAcc": bacc, "domadv_full_posttrain": domadv}


def train_compare(overlap=0.5, lams=(0.3, 1.0, 3.0), seeds=(0, 1, 2, 3, 4)):
    print(f"\n=== 3. ERM vs GLOBAL-LPC vs SELECTIVE (overlap={overlap}, "
          f"{len(seeds)} seeds, lambdas={lams}) ===")
    print(f"{'mode':>10} {'lam':>5} {'bAcc(mean+-std)':>18} {'domadv_posttrain':>17}")
    out = []
    # ERM once (lambda-independent)
    erm = [_train_compare(make(SynthSpec(overlap=overlap), seed=s), "erm", seed=s) for s in seeds]
    erm_b = np.array([r["bAcc"] for r in erm]); erm_l = np.array([r["domadv_full_posttrain"] for r in erm])
    print(f"{'erm':>10} {'-':>5} {erm_b.mean():8.3f} +- {erm_b.std():.3f}     {erm_l.mean():17.3f}")
    out.append({"mode": "erm", "lam": None, "bAcc_mean": float(erm_b.mean()),
                "bAcc_std": float(erm_b.std()), "domadv_mean": float(erm_l.mean())})
    for mode in ("global", "selective"):
        for lam in lams:
            res = [_train_compare(make(SynthSpec(overlap=overlap), seed=s), mode, lam=lam, seed=s)
                   for s in seeds]
            b = np.array([r["bAcc"] for r in res]); l = np.array([r["domadv_full_posttrain"] for r in res])
            print(f"{mode:>10} {lam:5.1f} {b.mean():8.3f} +- {b.std():.3f}     {l.mean():17.3f}")
            out.append({"mode": mode, "lam": lam, "bAcc_mean": float(b.mean()),
                        "bAcc_std": float(b.std()), "domadv_mean": float(l.mean())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", type=str, default="results/tos_cmi_synthetic.json")
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))
    env = _env()
    print("env:", env)
    results = {
        "env": env, "seed": args.seed, "seeds": list(seeds), "overlap_traincmp": args.overlap,
        "overlap_sweep": overlap_sweep([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], seed=args.seed),
        "stability_gate": stability_gate(overlap=0.0, seeds=seeds),
        "train_compare": train_compare(overlap=args.overlap, seeds=seeds),
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nartifact written: {args.out}")


if __name__ == "__main__":
    main()
