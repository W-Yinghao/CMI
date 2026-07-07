#!/usr/bin/env python
"""FSR Phase 7B — head-only learned reliance (CPU-only). See docs/FSR_38.

Freeze the Phase-4B branch latents; retrain ONLY a linear head on the concatenated branch latents
[graph_z|temporal_z|spatial_z] (96-dim, giving access to spatial's strong subject signal) under a source
subject<->class skew. PRIMARY question: can source-side subject-class skew weaponize the naturally present EEG
subject signal into target-harmful LEARNED reliance at the head level? Repair is secondary.

STAGED + FAIL-CLOSED (PM): --stage gate runs the Q5a LEARNABILITY / POWER GATE first; --stage full runs the
confirmatory dose-response + repair ONLY if the gate passed. The skew is induced by per-sample REWEIGHTING (not
subsampling), which holds global P(y) exactly and keeps effective-n = full -- eliminating the data-loss confound.

    <icml python> scripts/run_head_only_learned_reliance.py --stage gate [--seeds ...] [--folds N]

No GPU, no backbone retrain, no CMI/fbdualpc, no target-label fit, no rho change after seeing target harm.
"""
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))
sys.path.insert(0, _HERE)
import run_pc1_subject_token as pc1
import run_phase4e_token_centering as p4e

OUT = Path("results/fsr_head_only_learned_reliance")
LAT = Path("results/fsr_rq4_refit/latents")
CONFIRM_SEEDS = [20260721, 20260722, 20260723, 20260724, 20260725, 20260726, 20260727, 20260728]
RHOS = [0.0, 0.5, 0.8]
K_SUBJ = 2                      # subject-subspace rank for L5 replay / erasure
DECODE_MARGIN = 0.20           # subject-decode bAcc must beat chance by this for the learnability gate
N_SHUFFLE = 5                  # H2 null-band size
EPOCHS, LR, HID = 250, 5e-3, 0  # HID=0 -> linear head (head3-class); >0 -> MLP


def bacc(y, lg):
    return float(balanced_accuracy_score(y, lg.argmax(1)))


def concat(npz, pref):
    return np.concatenate([npz[f"{pref}graph_z"], npz[f"{pref}temporal_z"], npz[f"{pref}spatial_z"]], axis=1)


def assign_cd(sd, ncls, seed):
    """Complementary spurious class per subject (cyclic by rank) so the per-subject boost cancels in P(y)."""
    subj = np.unique(sd)
    order = sorted(int(d) for d in subj)
    return {d: (order.index(d) % ncls) for d in order}


def skew_weights(sy, sd, cd, rho, ncls):
    """Per-sample weights making the within-subject class fraction of c_d = rho; then a per-class global rescale
    restores global P(y) EXACTLY. eff-n unchanged (reweight, not subsample)."""
    w = np.ones(len(sy))
    if rho > 0:
        for d in np.unique(sd):
            m = sd == d; yd = sy[m]; c = cd[int(d)]
            cnt = np.bincount(yd, minlength=ncls).astype(float)
            frac = np.where(np.arange(ncls) == c, rho, (1 - rho) / max(ncls - 1, 1))
            wc = np.where(cnt > 0, frac * m.sum() / np.maximum(cnt, 1), 0.0)
            w[m] = wc[yd]
    # global per-class rescale to hold P(y): weighted class mass -> original class mass
    p0 = np.bincount(sy, minlength=ncls).astype(float)
    wm = np.array([w[sy == c].sum() for c in range(ncls)])
    scale = np.where(wm > 0, p0 / np.maximum(wm, 1e-9), 0.0)
    w = w * scale[sy]
    return w * (len(sy) / w.sum())


def train_head(X, y, w, ncls, seed):
    torch.manual_seed(seed)
    Xt = torch.as_tensor(X, dtype=torch.float32); yt = torch.as_tensor(y, dtype=torch.long)
    wt = torch.as_tensor(w, dtype=torch.float32)
    d = X.shape[1]
    H = nn.Linear(d, ncls) if HID == 0 else nn.Sequential(nn.Linear(d, HID), nn.ReLU(), nn.Linear(HID, ncls))
    opt = torch.optim.Adam(H.parameters(), lr=LR, weight_decay=1e-4)
    for _ in range(EPOCHS):
        opt.zero_grad()
        ce = (nn.functional.cross_entropy(H(Xt), yt, reduction="none") * wt).mean()
        ce.backward(); opt.step()
    H.eval()
    return H


def head_logits(H, X):
    with torch.no_grad():
        return H(torch.as_tensor(X, dtype=torch.float32)).numpy()


def subj_decode(X, d):
    d = np.asarray(d)
    if len(np.unique(d)) < 2 or np.min(np.bincount(d)) < 5:
        return float("nan"), 1.0 / max(len(np.unique(d)), 1)
    skf = StratifiedKFold(5, shuffle=True, random_state=0); pr = np.zeros_like(d)
    for tr, te in skf.split(X, d):
        pr[te] = LogisticRegression(max_iter=200).fit(X[tr], d[tr]).predict(X[te])
    return float(balanced_accuracy_score(d, pr)), 1.0 / len(np.unique(d))


def l5_reliance(H, X, y, S):
    """Head reliance on the subject subspace = accuracy drop when S is erased from the head input."""
    a0 = bacc(y, head_logits(H, X))
    a1 = bacc(y, head_logits(H, pc1.erase(X, S)))
    return a0 - a1


def cd_pred_rate(H, X, sd, cd):
    """Fraction of samples the head predicts as their OWN subject's spurious class c_d (learned subject->c_d
    bias, whether via representation or prior). Baseline for a c_d-agnostic head is ~ mean class prior at c_d."""
    pred = head_logits(H, X).argmax(1)
    return float(np.mean([pred[i] == cd[int(sd[i])] for i in range(len(sd))]))


def l4_align(H, S):
    """|cos| of head weight rows with the subject subspace (mean over classes x directions)."""
    W = (H.weight.detach().numpy() if HID == 0 else H[-1].weight.detach().numpy())
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    Sn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-9)
    return float(np.mean(np.abs(Wn @ Sn.T))) if S.shape[0] else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gate", "full"], default="gate")
    ap.add_argument("--seeds", type=int, nargs="+", default=CONFIRM_SEEDS)
    ap.add_argument("--folds", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    mans = sorted(glob.glob(str(LAT / "*_latent_dump_manifest.json")))
    if args.folds:
        mans = mans[:args.folds]

    gate_rows, man_rows, dose_rows, harm_rows, rep_rows, fw_rows = ([] for _ in range(6))
    for mp in mans:
        M = json.load(open(mp)); ds, tag, tsub = M["dataset"], M["tag"], M["target_subject"]
        src = np.load(LAT / f"{tag}_source_latents.npz"); tgt = np.load(LAT / f"{tag}_target_latents.npz")
        Xs = concat(src, "src_"); Xt = concat(tgt, "tgt_")
        sy, sd = src["y"].astype(int), src["d"].astype(int); ty = tgt["y"].astype(int)
        ncls = int(src["src_logits"].shape[1])
        scorer = p4e.TargetScorer(ty)
        subj_bacc, chance = subj_decode(Xs, sd)          # linear (head3-class) subject decodability
        S_subj = pc1.subj_subspace(Xs, sd, k=K_SUBJ)

        for seed in args.seeds:
            cd = assign_cd(sd, ncls, seed)
            # source held-in split (hold out ~30% source subjects) for the learnability gate
            vr = np.random.default_rng(p4e.seed_int(seed, "hi", tsub))
            subj = np.unique(sd); n_hi = max(1, int(round(len(subj) * 0.3)))
            hi = set(int(x) for x in vr.choice(subj, n_hi, replace=False))
            hi_m = np.array([int(d) in hi for d in sd]); tr_m = ~hi_m
            # learnability gate: compare the skewed head H1(rho) to the balanced baseline H0 on SOURCE HELD-IN.
            # "did-learn" evidence = H1 relies on the subject subspace OR predicts the held-in subject's c_d
            # MORE than H0 (a learned subject->c_d shortcut, whether via representation or prior).
            H0 = train_head(Xs[tr_m], sy[tr_m], skew_weights(sy[tr_m], sd[tr_m], cd, 0.0, ncls), ncls, seed)
            l5_0 = l5_reliance(H0, Xs[hi_m], sy[hi_m], S_subj)
            cdrate_0 = cd_pred_rate(H0, Xs[hi_m], sd[hi_m], cd)
            for rho in RHOS:
                w = skew_weights(sy[tr_m], sd[tr_m], cd, rho, ncls)
                py0 = np.bincount(sy[tr_m], minlength=ncls) / tr_m.sum()
                pyw = np.array([w[sy[tr_m] == c].sum() for c in range(ncls)]); pyw = pyw / pyw.sum()
                py_match = float(np.max(np.abs(py0 - pyw)))
                H = train_head(Xs[tr_m], sy[tr_m], w, ncls, seed)
                l5_hi = l5_reliance(H, Xs[hi_m], sy[hi_m], S_subj)
                cdrate = cd_pred_rate(H, Xs[hi_m], sd[hi_m], cd)
                l4 = l4_align(H, S_subj)
                if args.stage == "gate":
                    gate_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, rho=rho,
                                          subj_decode_bacc=round(subj_bacc, 4), chance=round(chance, 4),
                                          subj_decodable=bool(subj_bacc - chance > DECODE_MARGIN),
                                          py_match_max=round(py_match, 5), eff_n_frac=1.0,
                                          n_subj_diversity=len(np.unique(sd[tr_m])),
                                          l4_head_subj_align=round(l4, 4),
                                          l5_heldin_reliance=round(l5_hi, 4),
                                          l5_minus_H0=round(l5_hi - l5_0, 4),
                                          cd_pred_rate=round(cdrate, 4),
                                          cd_pred_rate_minus_H0=round(cdrate - cdrate_0, 4)))
            if args.stage == "gate":
                continue

            # ---- FULL stage (only reached if PM/gate approved) ----
            def train_at(rho, shuffle_seed=None, reg=False):
                cdx = cd
                if shuffle_seed is not None:
                    perm = np.random.default_rng(p4e.seed_int(shuffle_seed, "shuf", tsub)).permutation(
                        [cd[int(d)] for d in np.unique(sd)])
                    cdx = {int(d): int(perm[i]) for i, d in enumerate(np.unique(sd))}
                w = skew_weights(sy, sd, cdx, rho, ncls)
                Xtr = pc1.erase(Xs, S_subj) if reg else Xs      # H1_reg: subject-subspace-erased inputs (source-only)
                return train_head(Xtr, sy, w, ncls, seed)
            H0 = train_at(0.0)
            for rho in RHOS:
                H1 = train_at(rho)
                l5t = l5_reliance(H1, Xt, ty, S_subj); l4 = l4_align(H1, S_subj)
                harm = scorer.score(head_logits(H0, Xt)) - scorer.score(head_logits(H1, Xt))
                dose_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, rho=rho,
                                      l4_head_subj_align=round(l4, 4), l5_target_reliance=round(l5t, 4)))
                harm_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, rho=rho,
                                      H0_tgt_bacc=round(scorer.score(head_logits(H0, Xt)), 4),
                                      H1_tgt_bacc=round(scorer.score(head_logits(H1, Xt)), 4),
                                      target_harm=round(harm, 4)))
            # H2 null band + repair at rho=0.8 (strongest skew)
            rho = 0.8; H1 = train_at(rho); Hreg = train_at(rho, reg=True)
            h1_b = scorer.score(head_logits(H1, Xt)); h0_b = scorer.score(head_logits(H0, Xt))
            h2 = [scorer.score(head_logits(train_at(rho, shuffle_seed=s), Xt)) for s in range(N_SHUFFLE)]
            mu_s = p4e.balanced_mu(Xs, sy, ncls); Cs = np.cov(Xs.T); Ci = np.cov(Xt.T)
            e4 = scorer.score(head_logits(H1, Xt - 1.0 * (Xt.mean(0) - mu_s)))
            from run_phase4g_second_moment import excess_dirs, shrink_along
            e4b = scorer.score(head_logits(H1, shrink_along(Xt, excess_dirs(Ci, Cs, 2), Ci, Cs, 1.0)))
            erase_b = scorer.score(head_logits(H1, pc1.erase(Xt, S_subj)))
            hreg_b = scorer.score(head_logits(Hreg, Xt))
            rep_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, rho=rho,
                                 H0_bacc=round(h0_b, 4), H1_bacc=round(h1_b, 4),
                                 H2_band_mean=round(float(np.mean(h2)), 4), H2_band_hi=round(float(np.max(h2)), 4),
                                 E4_bacc=round(e4, 4), E4b_bacc=round(e4b, 4), ERASE_bacc=round(erase_b, 4),
                                 H1reg_bacc=round(hreg_b, 4)))
            fw_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, target_scorer_reads=scorer.n,
                                target_labels_used_for_fit=False, target_labels_used_for_selection=False,
                                target_labels_used_for_final_eval_only=True))
        print(f"[7b:{args.stage}] {tag} done", flush=True)

    if args.stage == "gate":
        pc1._w(OUT / "head_learnability_gate.csv", gate_rows)
        pc1._w(OUT / "head_skew_manifest.csv", [dict(
            dataset=r["dataset"], target_subject=r["target_subject"], token_seed=r["token_seed"], rho=r["rho"],
            py_match_max=r["py_match_max"], eff_n_frac=r["eff_n_frac"], n_subj_diversity=r["n_subj_diversity"])
            for r in gate_rows])
        print(f"wrote gate CSVs over {len(gate_rows)} fold-seed-rho")
    else:
        pc1._w(OUT / "head_reliance_dose_response.csv", dose_rows)
        pc1._w(OUT / "head_target_harm.csv", harm_rows)
        pc1._w(OUT / "head_repair_results.csv", rep_rows)
        (OUT / "head_target_label_firewall.json").write_text(json.dumps(
            dict(n=len(fw_rows), rows=fw_rows, target_labels_used_for_fit=False,
                 target_labels_used_for_selection=False, target_labels_used_for_final_eval_only=True),
            indent=2) + "\n")
        print(f"wrote FULL CSVs over {len(harm_rows)} fold-seed-rho")


if __name__ == "__main__":
    main()
