#!/usr/bin/env python
"""FSR Phase 4D — counterfactual / task-protected repair of the injected PC1 shortcut (CPU-only).

Reuses PC1's injection construction (token = unit(normalize(u)+normalize(v_class)), source-margin scale) so
the induced harm is of the same construction/magnitude as FSR_20. A small SOURCE-ONLY adapter A on the frozen
spatial_z branch (backbone head3(_fuse3(graph_z,temporal_z,.)) frozen) is trained to be INVARIANT to the token
family while preserving the source label -- a task-protected repair, NOT subspace erasure. See FSR_21.

Design fixes from the FSR_21 design red-team (wtbd1lg62):
  * Augmentation token CLASS is drawn UNIFORM over classes, INDEPENDENT of y (so the CE+consistency objective
    enforces token-INVARIANCE, not token-EXPLOITATION). Augmentation token DIRECTION u is a fresh random unit
    vector (the token family = {random-direction u + genuine class v_c}); the deployment target token
    u_tsub + v_{c_target} is in this family's distribution.
  * D3a control differs from D1 ONLY in that its perturbations are pure random directions (NO class-v
    structure) -- identical arch/loss/epochs/selection -- isolating whether the token subspace structure helps.
  * Epoch SELECTION = source-val injected-task bAcc where val samples get their REAL held-out subject u_d + a
    y-DECORRELATED (uniform) class. This is simultaneously (i) a correct invariance metric and (ii) the u-
    generalization diagnostic (unseen source-subject directions). A target fail with a LOW val-inj bAcc is
    attributable to the token-shift / u-generalization gap, not to the repair idea.
  * class-directed V computed over ALL source samples (nsamp=len) -> deterministic (no RNG-order dependence);
    the SAME V drives both the injection v_{c_target} and D2's task subspace T, so "protect T => protect harm"
    holds by construction.
  * PYTHONHASHSEED pinned by the run wrapper (recorded in manifest) so the hash-based u/c_target/cd are
    reproducible across processes. Induced harm is reported and checked against PC1's +0.04/+0.07 range.

FIREWALL: adapter fit, consistency targets, subspace estimation, epoch/arm selection use SOURCE only; alpha is
fixed; target labels are read ONLY through TargetScorer.score() (final bAcc). Source-val selection holds out
SOURCE SUBJECTS (not samples).

    PYTHONHASHSEED=0 <icml python> scripts/run_phase4d_repair.py [--alphas 1.0 2.0] [--folds N]

Fixed config (no sweep): K views=4, hidden=64, epochs=150, lr=1e-3, lam_cons=1.0, lam_l2=1.0,
val_subject_frac=0.3, k_subspace=2, seed=0.
"""
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

# The branch latents are tiny (32-dim); torch's default multi-thread dispatch spends ~100x more time on
# thread sync than on the microsecond matmuls. Pin to 1 thread -> forward 292ms -> 2.9ms (measured).
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))  # repo root so `cmi` imports
sys.path.insert(0, _HERE)                                        # scripts/ so pc1 imports
import run_pc1_subject_token as pc1  # reuse PC1 injection helpers

OUT = Path("results/fsr_phase4d_repair")
LAT = Path("results/fsr_rq4_refit/latents")
CK = Path("results/fsr_rq4_refit/ckpt")
PRIMARY = "spatial_z"
BR_SLOT = 2
CFG = dict(K=4, hidden=64, epochs=150, lr=1e-3, lam_cons=1.0, lam_l2=1.0,
           val_subject_frac=0.3, k_subspace=2, seed=0)
RNG = np.random.default_rng(0)


def _bacc(y, logits):
    return float(balanced_accuracy_score(y, logits.argmax(1)))


class TargetScorer:
    """By-construction firewall guard: target labels are readable ONLY via .score() (final bAcc scoring)."""
    def __init__(self, y):
        self._y = np.asarray(y)
        self.n_score_reads = 0

    def score(self, logits):
        self.n_score_reads += 1
        return _bacc(self._y, logits)


class Adapter(nn.Module):
    """A(z) = z + MLP(z); one hidden layer; last layer zero-init so A starts at identity."""
    def __init__(self, d, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, d))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        return z + self.net(z)


def _aug_tokens(N, dim, V, ncls, K, rng, mode):
    """K per-sample UNIT tokens.
       counterfactual: unit( unit(random_u) + unit(V[c]) ), c ~ Uniform(classes) INDEPENDENT of y.
       random        : pure random unit direction (no class structure) -- D3a control."""
    out = []
    for _ in range(K):
        u = rng.standard_normal((N, dim)); u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-9)
        if mode == "counterfactual":
            c = rng.integers(0, ncls, N)
            out.append(pc1.unit(pc1.unit(u) + pc1.unit(V[c])))
        else:
            out.append(u)
    return np.stack(out)  # [K,N,dim]


def _val_inject(S, subj, U, V, ncls, alpha, scale, rng):
    """Val-selection injection: real held-out subject u_d + y-DECORRELATED (uniform) class. Fixed once."""
    N = S.shape[0]
    c = rng.integers(0, ncls, N)
    U_rows = np.stack([U[int(d)] for d in subj])
    tok = pc1.unit(pc1.unit(U_rows) + pc1.unit(V[c]))
    return S + alpha * scale * tok


def train_adapter(bb, G, T, S, y, subj, val_mask, base_logits, U, V, ncls, alpha, scale, mode, cfg, seed):
    """Train A source-only; select epoch by source-val injected-task bAcc (held-out subjects, y-decorrelated
    class); return (A, selection dict). Target labels are NOT touched here."""
    torch.manual_seed(seed)
    rng_aug = np.random.default_rng(seed + (0 if mode == "counterfactual" else 777))
    rng_val = np.random.default_rng(seed + 999)
    Gt, Tt, St = (torch.as_tensor(a, dtype=torch.float32) for a in (G, T, S))
    yt = torch.as_tensor(y, dtype=torch.long)
    base_t = torch.as_tensor(base_logits, dtype=torch.float32)
    tr = torch.as_tensor(~val_mask); va = val_mask
    N, dim = S.shape
    for p in bb.parameters():
        p.requires_grad_(False)
    A = Adapter(dim, cfg["hidden"])
    opt = torch.optim.Adam(A.parameters(), lr=cfg["lr"])

    toks = _aug_tokens(N, dim, V, ncls, cfg["K"], rng_aug, mode)
    views = torch.as_tensor(np.stack([S + alpha * scale * toks[k] for k in range(cfg["K"])]),
                            dtype=torch.float32)
    val_inj = torch.as_tensor(_val_inject(S, subj, U, V, ncls, alpha, scale, rng_val), dtype=torch.float32)

    def fwd(sz):
        return bb.head3(bb._fuse3(Gt, Tt, sz))

    with torch.no_grad():
        vv = torch.stack([fwd(views[k]) for k in range(cfg["K"])])
        cons_pre = float(vv[:, torch.as_tensor(va)].var(0).mean())

    best = dict(vb=-1.0, state=None, epoch=-1)
    for ep in range(cfg["epochs"]):
        A.train(); opt.zero_grad()
        lv = [fwd(A(views[k])) for k in range(cfg["K"])]
        stacked = torch.stack(lv)
        ce = torch.stack([nn.functional.cross_entropy(lv[k][tr], yt[tr]) for k in range(cfg["K"])]).mean()
        cons = stacked[:, tr].var(0).mean()
        l2 = nn.functional.mse_loss(fwd(A(St))[tr], base_t[tr])
        (ce + cfg["lam_cons"] * cons + cfg["lam_l2"] * l2).backward()
        opt.step()
        A.eval()
        with torch.no_grad():
            vb = _bacc(y[va], fwd(A(val_inj)).numpy()[va])   # SOURCE y (val subjects) -- selection metric
        if vb > best["vb"]:
            best = dict(vb=vb, state={k: v.detach().clone() for k, v in A.state_dict().items()}, epoch=ep)
    A.load_state_dict(best["state"]); A.eval()
    with torch.no_grad():
        clean_a = _bacc(y[va], fwd(A(St)).numpy()[va])
        clean_f = _bacc(y[va], base_logits[va])
        vv2 = torch.stack([fwd(A(views[k])) for k in range(cfg["K"])])
        cons_post = float(vv2[:, torch.as_tensor(va)].var(0).mean())
    sel = dict(sel_epoch=best["epoch"], val_inj_task_bacc=round(best["vb"], 4),
               val_clean_frozen=round(clean_f, 4), val_clean_adapter=round(clean_a, 4),
               val_clean_task_drop=round(clean_f - clean_a, 4),
               consistency_pre=round(cons_pre, 6), consistency_post=round(cons_post, 6),
               consistency_gain=round(cons_pre - cons_post, 6))
    return A, sel


def apply_adapter(bb, A, g, t, s):
    with torch.no_grad():
        sz = A(torch.as_tensor(s, dtype=torch.float32)).numpy()
    return pc1.recompose(bb, g, t, sz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--folds", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    hashseed = os.environ.get("PYTHONHASHSEED", "UNSET")
    man_rows, sel_rows, cons_rows, rec_rows, rc_rows, fw_rows = ([] for _ in range(6))
    seed = CFG["seed"]

    mans = sorted(glob.glob(str(LAT / "*_latent_dump_manifest.json")))
    if args.folds:
        mans = mans[:args.folds]
    for mp in mans:
        man = json.load(open(mp))
        ds, tag, tsub = man["dataset"], man["tag"], man["target_subject"]
        src = np.load(LAT / f"{tag}_source_latents.npz")
        tgt = np.load(LAT / f"{tag}_target_latents.npz")
        sg, stz, ss = src["src_graph_z"], src["src_temporal_z"], src["src_spatial_z"]
        tg, ttz, ts_ = tgt["tgt_graph_z"], tgt["tgt_temporal_z"], tgt["tgt_spatial_z"]
        sy, sd = src["y"].astype(int), src["d"].astype(int)
        scorer = TargetScorer(tgt["y"].astype(int))   # target labels locked behind .score()
        ncls = int(src["src_logits"].shape[1])
        ck = torch.load(CK / f"{tag}_ckpt_best.pt", map_location="cpu", weights_only=False)
        bb = pc1.load_model(ds, ck["config"], ncls)
        bb.load_state_dict(ck["state_dict"], strict=True); bb.eval()

        cd = pc1.assign_cd(sy, sd, ncls)
        c_target = abs(hash(("ct", str(tsub)))) % ncls
        base_src = pc1.recompose(bb, sg, stz, ss)
        mtop = np.partition(base_src, -2, axis=1)
        margin = max(float(np.median(base_src[np.arange(len(sy)), sy] - mtop[:, -2])), 0.2)
        bacc_orig = scorer.score(pc1.recompose(bb, tg, ttz, ts_))
        V = pc1.class_dirs(bb, sg, stz, ss, BR_SLOT, ncls, nsamp=len(ss))   # deterministic (all samples)
        subj_ids = np.unique(sd)
        U = {int(d): pc1.uvec(d, ss.shape[1]) for d in subj_ids}
        U_t = pc1.uvec(tsub, ss.shape[1])
        tok_src = pc1.token_matrix([int(x) for x in sd], [cd[int(x)] for x in sd], U, V)
        tok_tgt = pc1.token_matrix([tsub] * len(sd) if False else [tsub] * len(tgt["y"]),
                                   [c_target] * len(tgt["y"]), {tsub: U_t}, V)
        shift = pc1.recompose(bb, sg, stz, ss + tok_src)
        s_shift = float(np.mean(shift[np.arange(len(sy)), [cd[int(x)] for x in sd]] -
                                base_src[np.arange(len(sy)), [cd[int(x)] for x in sd]]))
        scale = margin / s_shift if s_shift > 1e-6 else 0.0

        vr = np.random.default_rng(seed + 13)
        n_val = max(1, int(round(len(subj_ids) * CFG["val_subject_frac"])))
        val_subj = set(int(x) for x in vr.choice(subj_ids, n_val, replace=False))
        val_mask = np.array([int(d) in val_subj for d in sd])

        man_rows.append(dict(dataset=ds, target_subject=tsub, branch=PRIMARY, ncls=ncls,
                             n_src_subjects=len(subj_ids), n_val_subjects=n_val,
                             val_subject_ids=";".join(str(x) for x in sorted(val_subj)),
                             margin=round(margin, 4), unit_class_shift=round(s_shift, 4),
                             scale=round(scale, 4), c_target=c_target, pythonhashseed=hashseed,
                             K=CFG["K"], hidden=CFG["hidden"], epochs=CFG["epochs"], lr=CFG["lr"],
                             lam_cons=CFG["lam_cons"], lam_l2=CFG["lam_l2"], k_subspace=CFG["k_subspace"]))

        for a in args.alphas:
            inj_src_sz = ss + a * scale * tok_src
            inj_tgt_sz = ts_ + a * scale * tok_tgt
            bacc_inj = scorer.score(pc1.recompose(bb, tg, ttz, inj_tgt_sz))
            denom = bacc_orig - bacc_inj

            b_d0 = scorer.score(pc1.recompose(bb, tg, ttz, inj_tgt_sz - a * scale * tok_tgt))
            A1, sel1 = train_adapter(bb, sg, stz, ss, sy, sd, val_mask, base_src, U, V, ncls, a, scale,
                                     "counterfactual", CFG, seed)
            b_d1 = scorer.score(apply_adapter(bb, A1, tg, ttz, inj_tgt_sz))
            S_sub = pc1.subj_subspace(inj_src_sz, sd, k=CFG["k_subspace"])
            QT, _ = np.linalg.qr(V.T)
            S_perp = S_sub - (S_sub @ QT) @ QT.T
            b_d2 = scorer.score(pc1.recompose(bb, tg, ttz, pc1.erase(inj_tgt_sz, S_perp)))
            A3, sel3 = train_adapter(bb, sg, stz, ss, sy, sd, val_mask, base_src, U, V, ncls, a, scale,
                                     "random", CFG, seed)
            b_d3a = scorer.score(apply_adapter(bb, A3, tg, ttz, inj_tgt_sz))
            rb = RNG.standard_normal((CFG["k_subspace"], ss.shape[1]))
            b_d3b = scorer.score(pc1.recompose(bb, tg, ttz, pc1.erase(inj_tgt_sz, rb)))

            def rec(b):
                return round(float((b - bacc_inj) / denom), 4) if abs(denom) > 1e-4 else None
            rec_rows.append(dict(dataset=ds, target_subject=tsub, alpha=a,
                                 bacc_orig=round(bacc_orig, 4), bacc_injected=round(bacc_inj, 4),
                                 induced_harm=round(denom, 4),
                                 D0_exact_bacc=round(b_d0, 4), D0_recovery=rec(b_d0),
                                 D1_cf_adapter_bacc=round(b_d1, 4), D1_recovery=rec(b_d1),
                                 D2_taskorth_erase_bacc=round(b_d2, 4), D2_recovery=rec(b_d2),
                                 D3a_rand_adapter_bacc=round(b_d3a, 4), D3a_recovery=rec(b_d3a),
                                 D3b_randk_erase_bacc=round(b_d3b, 4), D3b_recovery=rec(b_d3b),
                                 D4_injected_bacc=round(bacc_inj, 4)))
            rc_rows.append(dict(dataset=ds, target_subject=tsub, alpha=a,
                                D1_bacc=round(b_d1, 4), D3a_bacc=round(b_d3a, 4),
                                D1_minus_D3a_bacc=round(b_d1 - b_d3a, 4)))
            for arm, sel in (("D1_counterfactual", sel1), ("D3a_random", sel3)):
                sel_rows.append(dict(dataset=ds, target_subject=tsub, alpha=a, arm=arm, **sel))
            cons_rows.append(dict(dataset=ds, target_subject=tsub, alpha=a,
                                  D1_consistency_pre=sel1["consistency_pre"],
                                  D1_consistency_post=sel1["consistency_post"],
                                  D1_consistency_gain=sel1["consistency_gain"],
                                  D1_val_inj_task_bacc=sel1["val_inj_task_bacc"],
                                  D1_val_clean_task_drop=sel1["val_clean_task_drop"]))
            fw_rows.append(dict(dataset=ds, target_subject=tsub, alpha=a,
                                target_scorer_reads=scorer.n_score_reads,
                                target_labels_used_for_fit=False, target_labels_used_for_selection=False,
                                alpha_selection_used_target=False, subspace_est_used_target=False,
                                token_assignment_used_target=False,
                                target_labels_used_for_final_eval_only=True))
            print(f"[4d] {tag} a={a}: orig={bacc_orig:.3f} inj={bacc_inj:.3f} harm={denom:+.3f} "
                  f"D1={b_d1:.3f}(rec {rec(b_d1)}) D2={b_d2:.3f} D3a={b_d3a:.3f} D0={b_d0:.3f} "
                  f"val_inj_bacc={sel1['val_inj_task_bacc']}", flush=True)

    pc1._w(OUT / "phase4d_repair_manifest.csv", man_rows)
    pc1._w(OUT / "phase4d_source_val_selection.csv", sel_rows)
    pc1._w(OUT / "phase4d_counterfactual_consistency.csv", cons_rows)
    pc1._w(OUT / "phase4d_target_recovery.csv", rec_rows)
    pc1._w(OUT / "phase4d_random_control.csv", rc_rows)
    (OUT / "phase4d_target_label_firewall.json").write_text(json.dumps(
        dict(n_folds=len(fw_rows), pythonhashseed=hashseed, rows=fw_rows,
             target_labels_used_for_fit=False, target_labels_used_for_selection=False,
             alpha_selection_used_target=False, subspace_est_used_target=False,
             target_labels_used_for_final_eval_only=True,
             note="target labels are read only via TargetScorer.score(); target_scorer_reads counts those."),
        indent=2) + "\n")
    print(f"wrote Phase 4D CSVs over {len(man_rows)} folds (PYTHONHASHSEED={hashseed})")


if __name__ == "__main__":
    main()
