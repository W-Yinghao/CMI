#!/usr/bin/env python
"""FSR Phase 7C — head-only TASK-CONFLICT weaponization (CPU-only). See docs/FSR_40.

7B showed prevalence-reweighting (true labels) does not weaponize the head. 7C injects a subject-correlated,
TASK-CONFLICTING training label (label corruption that FIGHTS the true task) via deterministic PAIRED LABEL SWAPS
that hold the global label histogram EXACTLY unchanged, and asks whether the head weaponizes the naturally present
subject signal into a cross-subject TRANSFERABLE, target-harmful learned reliance. PRIMARY = weaponization; repair
secondary. Staged fail-closed:
  --stage gate : Q7C-a HELD-IN learnability -- on TRAINING subjects, does the linear head SATISFY the conflict
                 labels (fit on the relabeled subset beyond a task-only floor), monotone in gamma? (memorization)
  --stage full : Q7C-b TRANSFER (held-out source subjects, structured beats shuffle/random controls) + target
                 harm + secondary repair. Same Hconflict head as the gate (deterministic RNG keys).

    <icml python> scripts/run_head_only_label_conflict.py --stage gate [--seeds ...] [--folds N]
No GPU/backbone-retrain/CMI/fbdualpc/new-primitive/target-label-fit/post-hoc-gamma.
"""
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np
import torch

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
import run_head_only_learned_reliance as h7b   # reuse concat/train_head/head_logits/subj_decode/l5/l4

OUT = Path("results/fsr_head_only_label_conflict")
LAT = Path("results/fsr_rq4_refit/latents")
CONFIRM_SEEDS = [20260721, 20260722, 20260723, 20260724, 20260725, 20260726, 20260727, 20260728]
GAMMAS = [0.0, 0.2, 0.4]
PRIMARY_GAMMA = 0.4
K_SUBJ = 2
N_SHUFFLE = 5


def bacc(y, lg):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, lg.argmax(1)))


def acc_on(H, X, y):
    """Plain accuracy of head H's argmax over the given (sub)set — used for conflict-label fit on relabeled rows."""
    if len(y) == 0:
        return float("nan")
    return float((h7b.head_logits(H, X).argmax(1) == y).mean())


def arng(seed, arm, tsub, g, extra=0):
    """Symmetric per-(seed, arm, fold, gamma) RNG so every arm is derived the same fold-dependent way."""
    return np.random.default_rng(p4e.seed_int(seed, arm, tsub, int(round(g * 100)), extra))


def label_conflict(sy, sd, cd, gamma, rng, shuffle_cd=False):
    """Subject-correlated TASK-CONFLICT labels via deterministic PAIRED SWAPS (global histogram exact).
    For a pair (d,d') with c_d=a != b=c_d': swap k of d's TRUE-b samples (-> a=c_d) with k of d''s TRUE-a
    samples (-> b=c_d'). Each swap is an a<->b exchange => histogram net zero. Per-subject budget = gamma*N_d.
    Returns (yt, c_d_map, n_swaps, conflict_mask) where conflict_mask marks the relabeled (task-conflicting) rows."""
    subj = list(np.unique(sd)); cdx = dict((int(d), int(cd[int(d)])) for d in subj)
    if shuffle_cd:
        vals = [cdx[int(d)] for d in subj]; rng.shuffle(vals); cdx = {int(d): int(vals[i]) for i, d in enumerate(subj)}
    yt = sy.copy(); used = np.zeros(len(sy), bool)
    budget = {int(d): int(round(gamma * (sd == d).sum())) for d in subj}
    pairs = [(int(d), int(e)) for d in subj for e in subj if int(d) < int(e) and cdx[int(d)] != cdx[int(e)]]
    rng.shuffle(pairs); nsw = 0
    for d, e in pairs:
        a, b = cdx[d], cdx[e]
        pd = np.where((sd == d) & (sy == b) & ~used)[0]        # d truly-b -> a (=c_d)
        pe = np.where((sd == e) & (sy == a) & ~used)[0]        # e truly-a -> b (=c_e)
        k = int(min(len(pd), len(pe), budget[d], budget[e]))
        if k > 0:
            i1 = rng.choice(pd, k, replace=False); i2 = rng.choice(pe, k, replace=False)
            yt[i1] = a; yt[i2] = b; used[i1] = True; used[i2] = True
            budget[d] -= k; budget[e] -= k; nsw += k
    return yt, cdx, nsw, used


def random_corrupt(sy, n_swaps, rng):
    """Matched-rate BALANCED label corruption WITHOUT subject structure (histogram exact via distinct class-pair
    swaps). used-tracked so exactly 2*n_swaps distinct labels change (net), matching label_conflict's 2*nsw."""
    yt = sy.copy(); ncls = int(sy.max()) + 1; used = np.zeros(len(sy), bool); done = 0
    guard = 0
    while done < int(n_swaps) and guard < 50 * (int(n_swaps) + 1):
        guard += 1
        p, q = rng.choice(ncls, 2, replace=False)
        ip = np.where((yt == p) & ~used)[0]; iq = np.where((yt == q) & ~used)[0]
        if len(ip) and len(iq):
            i, j = int(rng.choice(ip)), int(rng.choice(iq))
            yt[i], yt[j] = q, p; used[i] = used[j] = True; done += 1
    return yt, used


def hist_delta(sy, yt, ncls):
    return int(np.max(np.abs(np.bincount(sy, minlength=ncls) - np.bincount(yt, minlength=ncls))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gate", "full"], default="gate")
    ap.add_argument("--seeds", type=int, nargs="+", default=CONFIRM_SEEDS)
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--outdir", default=str(OUT), help="shard dir for process-parallel runs; merged to canonical later")
    args = ap.parse_args()
    od = Path(args.outdir); od.mkdir(parents=True, exist_ok=True)
    gate_rows, hist_rows, man_rows, trans_rows, dose_rows, harm_rows, rep_rows, fw_rows = ([] for _ in range(8))
    mans = sorted(glob.glob(str(LAT / "*_latent_dump_manifest.json")))
    if args.folds:
        mans = mans[:args.folds]

    for mp in mans:
        M = json.load(open(mp)); ds, tag, tsub = M["dataset"], M["tag"], M["target_subject"]
        src = np.load(LAT / f"{tag}_source_latents.npz"); tgt = np.load(LAT / f"{tag}_target_latents.npz")
        Xs = h7b.concat(src, "src_"); Xt = h7b.concat(tgt, "tgt_")
        sy, sd = src["y"].astype(int), src["d"].astype(int); ty = tgt["y"].astype(int)
        ncls = int(src["src_logits"].shape[1]); scorer = p4e.TargetScorer(ty)
        S_all = pc1.subj_subspace(Xs, sd, k=K_SUBJ)                    # subject subspace for repair / target erase

        for seed in args.seeds:
            cd = h7b.assign_cd(sd, ncls, seed)
            vr = np.random.default_rng(p4e.seed_int(seed, "hi", tsub))
            subj = np.unique(sd); n_hi = max(1, int(round(len(subj) * 0.3)))
            hi = set(int(x) for x in vr.choice(subj, n_hi, replace=False))
            him = np.array([int(d) in hi for d in sd]); trm = ~him
            S_trm = pc1.subj_subspace(Xs[trm], sd[trm], k=K_SUBJ)      # reliance subspace: fit on TRAIN only

            Xtr, sytr, sdtr = Xs[trm], sy[trm], sd[trm]
            Xhi, syhi = Xs[him], sy[him]
            H0 = h7b.train_head(Xtr, sytr, np.ones(trm.sum()), ncls, seed)
            task0_tr = bacc(sytr, h7b.head_logits(H0, Xtr))            # clean head true-task on TRAIN
            task0_hi = bacc(syhi, h7b.head_logits(H0, Xhi))            # clean head true-task on HELD-OUT
            l5_0_hi = h7b.l5_reliance(H0, Xhi, syhi, S_trm)

            for g in GAMMAS:
                yc, cdx, nsw, conf = label_conflict(sytr, sdtr, cd, g, arng(seed, "conflict", tsub, g))
                hd = hist_delta(sytr, yc, ncls); achieved = round(2.0 * nsw / max(1, len(sytr)), 4)
                Hc = h7b.train_head(Xtr, yc, np.ones(trm.sum()), ncls, seed)
                yr, _ = random_corrupt(sytr, nsw, arng(seed, "random", tsub, g))
                Hr = h7b.train_head(Xtr, yr, np.ones(trm.sum()), ncls, seed)
                Hsh, sh_nsw = [], []
                for i in range(N_SHUFFLE):
                    ysh, _, nsh, _ = label_conflict(sytr, sdtr, cd, g, arng(seed, "shuffle", tsub, g, i), shuffle_cd=True)
                    Hsh.append(h7b.train_head(Xtr, ysh, np.ones(trm.sum()), ncls, seed)); sh_nsw.append(nsh)
                man_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, gamma=g, c_d=str(cdx),
                                     n_swaps=nsw, achieved_conflict_frac=achieved, global_Py_delta=hd,
                                     shuffle_nsw_mean=round(float(np.mean(sh_nsw)), 2)))
                hist_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, gamma=g,
                                      max_abs_delta_global_Py=hd, n_swaps=nsw))

                if args.stage == "gate":
                    # Q7C-a HELD-IN LEARNABILITY (on TRAIN): did the head SATISFY the conflict labels?
                    cfit = acc_on(Hc, Xtr[conf], yc[conf])            # Hconflict fit on the relabeled rows
                    floor = acc_on(H0, Xtr[conf], yc[conf])           # task-only floor (~0 by construction)
                    l5_c = h7b.l5_reliance(Hc, Xtr, sytr, S_trm)      # memorized subject reliance on TRAIN
                    l5_r = h7b.l5_reliance(Hr, Xtr, sytr, S_trm)
                    l5_sh = [h7b.l5_reliance(h, Xtr, sytr, S_trm) for h in Hsh]
                    task_c_tr = bacc(sytr, h7b.head_logits(Hc, Xtr))
                    gate_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, gamma=g,
                                          global_Py_delta=hd, achieved_conflict_frac=achieved,
                                          conflict_fit=round(cfit, 4), conflict_fit_floor=round(floor, 4),
                                          conflict_fit_minus_floor=round(cfit - floor, 4),
                                          l5_conflict_train=round(l5_c, 4), l5_random_train=round(l5_r, 4),
                                          l5_shuffle_train_mean=round(float(np.mean(l5_sh)), 4),
                                          l5_shuffle_train_max=round(float(np.max(l5_sh)), 4),
                                          heldin_task_drop_train=round(task0_tr - task_c_tr, 4)))
                    continue

                # ---- FULL stage: Q7C-b TRANSFER (held-out) + target harm ----
                # PRIMARY transfer signal = held-out TRUE-TASK harm of the STRUCTURED conflict head beyond BOTH
                # controls (subject-shuffle band + random-noise). L4/L5 reliance are mechanism DIAGNOSTICS (the
                # subspace-erasure L5 is a weak instrument once true-task is degraded, so it does not hard-veto).
                pt_task_c = bacc(syhi, h7b.head_logits(Hc, Xhi))
                pt_task_r = bacc(syhi, h7b.head_logits(Hr, Xhi))
                pt_task_sh = float(np.mean([bacc(syhi, h7b.head_logits(h, Xhi)) for h in Hsh]))
                pt_l5_c = h7b.l5_reliance(Hc, Xhi, syhi, S_trm)
                pt_l5_r = h7b.l5_reliance(Hr, Xhi, syhi, S_trm)
                pt_l5_sh = [h7b.l5_reliance(h, Xhi, syhi, S_trm) for h in Hsh]
                pt_l4_c = h7b.l4_align(Hc, S_trm)                                # label-free head-weight alignment
                pt_l4_sh = float(np.mean([h7b.l4_align(h, S_trm) for h in Hsh]))
                trans_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, gamma=g,
                                       pt_task_drop_vs_H0=round(task0_hi - pt_task_c, 4),
                                       pt_task_drop_vs_shuffle=round(pt_task_sh - pt_task_c, 4),
                                       pt_task_drop_vs_random=round(pt_task_r - pt_task_c, 4),
                                       pt_l5_conflict=round(pt_l5_c, 4),
                                       pt_l5_conflict_minus_shuffle=round(pt_l5_c - float(np.mean(pt_l5_sh)), 4),
                                       pt_l5_conflict_minus_random=round(pt_l5_c - pt_l5_r, 4),
                                       pt_l4_conflict_minus_shuffle=round(pt_l4_c - pt_l4_sh, 4),
                                       pt_l5_shuffle_max=round(float(np.max(pt_l5_sh)), 4)))
                # target harm: FULL-source heads (same conflict RNG keys) scored on TARGET via scorer only
                ycf, _, _, _ = label_conflict(sy, sd, cd, g, arng(seed, "conflict_full", tsub, g))
                Hc_full = h7b.train_head(Xs, ycf, np.ones(len(sy)), ncls, seed)
                yshf, _, _, _ = label_conflict(sy, sd, cd, g, arng(seed, "shuffle_full", tsub, g, 0), shuffle_cd=True)
                Hsh_full = h7b.train_head(Xs, yshf, np.ones(len(sy)), ncls, seed)
                H0_full = h7b.train_head(Xs, sy, np.ones(len(sy)), ncls, seed)
                h0b = scorer.score(h7b.head_logits(H0_full, Xt))
                hcb = scorer.score(h7b.head_logits(Hc_full, Xt))
                dose_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, gamma=g,
                                      l4_conflict_target=round(h7b.l4_align(Hc_full, S_all), 4),  # label-free
                                      l5_conflict_src_heldout=round(pt_l5_c, 4)))                 # source-label only
                harm_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, gamma=g,
                                      H0_tgt=round(h0b, 4), Hconflict_tgt=round(hcb, 4),
                                      Hshuffle_tgt=round(scorer.score(h7b.head_logits(Hsh_full, Xt)), 4),
                                      target_harm=round(h0b - hcb, 4)))
                if abs(g - PRIMARY_GAMMA) < 1e-9:
                    # secondary repair on the primary-gamma weaponized full-source head
                    Hreg = h7b.train_head(pc1.erase(Xs, S_all), ycf, np.ones(len(sy)), ncls, seed)
                    mu_s = p4e.balanced_mu(Xs, sy, ncls); Cs = np.cov(Xs.T); Ci = np.cov(Xt.T)
                    from run_phase4g_second_moment import excess_dirs, shrink_along
                    rep_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, gamma=g,
                                         H0_tgt=round(h0b, 4), Hconflict_tgt=round(hcb, 4),
                                         E4_tgt=round(scorer.score(h7b.head_logits(Hc_full, Xt - 1.0 * (Xt.mean(0) - mu_s))), 4),
                                         E4b_tgt=round(scorer.score(h7b.head_logits(Hc_full, shrink_along(Xt, excess_dirs(Ci, Cs, 2), Ci, Cs, 1.0))), 4),
                                         ERASE_tgt=round(scorer.score(h7b.head_logits(Hc_full, pc1.erase(Xt, S_all))), 4),
                                         Hreg_tgt=round(scorer.score(h7b.head_logits(Hreg, Xt)), 4)))
            if args.stage == "full":
                fw_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, target_scorer_reads=scorer.n,
                                    target_labels_used_for_fit=False, target_labels_used_for_selection=False,
                                    target_labels_used_for_final_eval_only=True))
        print(f"[7c:{args.stage}] {tag} done", flush=True)

    pc1._w(od / "label_conflict_manifest.csv", man_rows)
    pc1._w(od / "global_label_histogram_check.csv", hist_rows)
    if args.stage == "gate":
        pc1._w(od / "heldin_learnability_gate.csv", gate_rows)
        print(f"wrote gate CSVs over {len(gate_rows)} fold-seed-gamma")
    else:
        pc1._w(od / "pseudo_target_transferability_gate.csv", trans_rows)
        pc1._w(od / "dose_response_reliance.csv", dose_rows)
        pc1._w(od / "target_harm.csv", harm_rows)
        pc1._w(od / "repair_secondary_results.csv", rep_rows)
        (od / "target_label_firewall.json").write_text(json.dumps(
            dict(n=len(fw_rows), rows=fw_rows, target_labels_used_for_fit=False,
                 target_labels_used_for_selection=False, target_labels_used_for_final_eval_only=True,
                 note="all target reads via p4e.TargetScorer; l4_conflict_target is label-free (head-weight/subspace "
                      "alignment); dose-response reliance reported on SOURCE held-out (source labels)."), indent=2) + "\n")
        print(f"wrote FULL CSVs over {len(harm_rows)} fold-seed-gamma")


if __name__ == "__main__":
    main()
