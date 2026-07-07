#!/usr/bin/env python
"""FSR Phase 7C — head-only TASK-CONFLICT weaponization (CPU-only). See docs/FSR_40.

7B showed prevalence-reweighting (true labels) does not weaponize the head. 7C injects a subject-correlated,
TASK-CONFLICTING training label (label corruption that FIGHTS the true task) via deterministic PAIRED LABEL SWAPS
that hold the global label histogram EXACTLY unchanged, and asks whether the head weaponizes the naturally present
subject signal into a cross-subject TRANSFERABLE, target-harmful learned reliance. PRIMARY = weaponization; repair
secondary. Staged fail-closed: --stage gate = Q7C-a held-in learnability; --stage full = Q7C-b transferability +
target harm + secondary repair (only if gate passes).

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
import run_head_only_learned_reliance as h7b   # reuse concat/train_head/head_logits/subj_decode/l5/l4/cd_rate

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


def label_conflict(sy, sd, cd, gamma, rng, shuffle_cd=False):
    """Subject-correlated TASK-CONFLICT labels via deterministic PAIRED SWAPS (global histogram exact).
    For a pair (d,d') with c_d=a != b=c_d': swap k of d's TRUE-b samples (-> a=c_d) with k of d''s TRUE-a
    samples (-> b=c_d'). Each swap is an a<->b exchange => histogram net zero. Per-subject budget = gamma*N_d."""
    subj = list(np.unique(sd)); cdx = dict((int(d), cd[int(d)]) for d in subj)
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
    return yt, cdx, nsw


def random_corrupt(sy, n_swaps, rng):
    """Matched-rate BALANCED label corruption WITHOUT subject structure (histogram exact via class-pair swaps)."""
    yt = sy.copy(); ncls = int(sy.max()) + 1
    for _ in range(int(n_swaps)):
        p, q = rng.choice(ncls, 2, replace=False)
        ip = np.where(yt == p)[0]; iq = np.where(yt == q)[0]
        if len(ip) and len(iq):
            i, j = rng.choice(ip), rng.choice(iq); yt[i], yt[j] = q, p
    return yt


def hist_delta(sy, yt, ncls):
    return int(np.max(np.abs(np.bincount(sy, minlength=ncls) - np.bincount(yt, minlength=ncls))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gate", "full"], default="gate")
    ap.add_argument("--seeds", type=int, nargs="+", default=CONFIRM_SEEDS)
    ap.add_argument("--folds", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gate_rows, hist_rows, man_rows, trans_rows, dose_rows, harm_rows, rep_rows, rnd_rows, shuf_rows, fw_rows = ([] for _ in range(10))
    mans = sorted(glob.glob(str(LAT / "*_latent_dump_manifest.json")))
    if args.folds:
        mans = mans[:args.folds]

    for mp in mans:
        M = json.load(open(mp)); ds, tag, tsub = M["dataset"], M["tag"], M["target_subject"]
        src = np.load(LAT / f"{tag}_source_latents.npz"); tgt = np.load(LAT / f"{tag}_target_latents.npz")
        Xs = h7b.concat(src, "src_"); Xt = h7b.concat(tgt, "tgt_")
        sy, sd = src["y"].astype(int), src["d"].astype(int); ty = tgt["y"].astype(int)
        ncls = int(src["src_logits"].shape[1]); scorer = p4e.TargetScorer(ty)
        S_subj = pc1.subj_subspace(Xs, sd, k=K_SUBJ)

        for seed in args.seeds:
            cd = h7b.assign_cd(sd, ncls, seed)
            vr = np.random.default_rng(p4e.seed_int(seed, "hi", tsub))
            subj = np.unique(sd); n_hi = max(1, int(round(len(subj) * 0.3)))
            hi = set(int(x) for x in vr.choice(subj, n_hi, replace=False))
            him = np.array([int(d) in hi for d in sd]); trm = ~him
            rc = np.random.default_rng(p4e.seed_int(seed, "lc", tsub))

            # H0 baseline on train subjects
            H0 = h7b.train_head(Xs[trm], sy[trm], np.ones(trm.sum()), ncls, seed)
            l5_0 = h7b.l5_reliance(H0, Xs[him], sy[him], S_subj); task_0 = bacc(sy[him], h7b.head_logits(H0, Xs[him]))
            for g in GAMMAS:
                yc, cdx, nsw = label_conflict(sy[trm], sd[trm], cd, g, rc)
                hd = hist_delta(sy[trm], yc, ncls)
                # Hconflict head on task-conflict labels
                Hc = h7b.train_head(Xs[trm], yc, np.ones(trm.sum()), ncls, seed)
                # controls at matched rate: random label noise + subject-shuffled conflict
                yr = random_corrupt(sy[trm], nsw, np.random.default_rng(p4e.seed_int(seed, "rnd", int(g * 100))))
                Hr = h7b.train_head(Xs[trm], yr, np.ones(trm.sum()), ncls, seed)
                ysh, _, _ = label_conflict(sy[trm], sd[trm], cd, g, np.random.default_rng(p4e.seed_int(seed, "shuf", int(g * 100))), shuffle_cd=True)
                Hsh = h7b.train_head(Xs[trm], ysh, np.ones(trm.sum()), ncls, seed)
                # held-in learnability (Q7C-a): reliance on subject + corrupted-label fit + task collapse
                l5_c = h7b.l5_reliance(Hc, Xs[him], sy[him], S_subj)
                l5_r = h7b.l5_reliance(Hr, Xs[him], sy[him], S_subj)
                l5_sh = h7b.l5_reliance(Hsh, Xs[him], sy[him], S_subj)
                task_c = bacc(sy[him], h7b.head_logits(Hc, Xs[him]))
                # corrupted-label fit on TRAIN (did the head satisfy the task-conflicting labels?)
                fit_c = bacc(yc, h7b.head_logits(Hc, Xs[trm]))
                man_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, gamma=g, c_d=str(cdx),
                                     n_swaps=nsw, global_Py_delta=hd))
                hist_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, gamma=g,
                                      max_abs_delta_global_Py=hd, n_swaps=nsw))
                if args.stage == "gate":
                    gate_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, gamma=g,
                                          global_Py_delta=hd, corrupt_label_fit=round(fit_c, 4),
                                          l5_conflict=round(l5_c, 4), l5_conflict_minus_H0=round(l5_c - l5_0, 4),
                                          l5_random=round(l5_r, 4), l5_shuffle=round(l5_sh, 4),
                                          l5_conflict_minus_shuffle=round(l5_c - l5_sh, 4),
                                          l5_conflict_minus_random=round(l5_c - l5_r, 4),
                                          heldin_task_drop=round(task_0 - task_c, 4)))
            if args.stage == "gate":
                continue

            # ---- FULL stage: Q7C-b transferability + target harm + secondary repair ----
            # Q7C-b: pseudo-target = held-in source subjects scored on TRUE labels (train Hconflict on the REST)
            for g in GAMMAS:
                yc, cdx, nsw = label_conflict(sy[trm], sd[trm], cd, g, np.random.default_rng(p4e.seed_int(seed, "lc2", int(g * 100))))
                Hc = h7b.train_head(Xs[trm], yc, np.ones(trm.sum()), ncls, seed)
                ysh, _, _ = label_conflict(sy[trm], sd[trm], cd, g, np.random.default_rng(p4e.seed_int(seed, "sh2", int(g * 100))), shuffle_cd=True)
                Hsh = h7b.train_head(Xs[trm], ysh, np.ones(trm.sum()), ncls, seed)
                pt_task_c = bacc(sy[him], h7b.head_logits(Hc, Xs[him]))     # pseudo-target TRUE-label bAcc
                pt_task_sh = bacc(sy[him], h7b.head_logits(Hsh, Xs[him]))
                l5_c = h7b.l5_reliance(Hc, Xs[him], sy[him], S_subj)
                trans_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, gamma=g,
                                       pt_task_drop_vs_H0=round(task_0 - pt_task_c, 4),
                                       pt_task_drop_vs_shuffle=round(pt_task_sh - pt_task_c, 4),
                                       pt_l5_conflict=round(l5_c, 4)))
                # target harm: full-source-trained heads scored on TARGET true labels
                Hc_full = h7b.train_head(Xs, label_conflict(sy, sd, cd, g, np.random.default_rng(p4e.seed_int(seed, "lcf", int(g * 100))))[0], np.ones(len(sy)), ncls, seed)
                Hsh_full = h7b.train_head(Xs, label_conflict(sy, sd, cd, g, np.random.default_rng(p4e.seed_int(seed, "shf", int(g * 100))), shuffle_cd=True)[0], np.ones(len(sy)), ncls, seed)
                H0_full = h7b.train_head(Xs, sy, np.ones(len(sy)), ncls, seed)
                h0b = scorer.score(h7b.head_logits(H0_full, Xt))
                dose_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, gamma=g,
                                      l4_conflict=round(h7b.l4_align(Hc_full, S_subj), 4),
                                      l5_conflict=round(h7b.l5_reliance(Hc_full, Xt, ty, S_subj), 4)))
                harm_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, gamma=g,
                                      H0_tgt=round(h0b, 4),
                                      Hconflict_tgt=round(scorer.score(h7b.head_logits(Hc_full, Xt)), 4),
                                      Hshuffle_tgt=round(scorer.score(h7b.head_logits(Hsh_full, Xt)), 4),
                                      target_harm=round(h0b - scorer.score(h7b.head_logits(Hc_full, Xt)), 4)))
            # secondary repair on Hconflict at PRIMARY_GAMMA (full-source head)
            g = PRIMARY_GAMMA
            Hc_full = h7b.train_head(Xs, label_conflict(sy, sd, cd, g, np.random.default_rng(p4e.seed_int(seed, "lcr", 1)))[0], np.ones(len(sy)), ncls, seed)
            H0_full = h7b.train_head(Xs, sy, np.ones(len(sy)), ncls, seed)
            Hreg = h7b.train_head(pc1.erase(Xs, S_subj), label_conflict(sy, sd, cd, g, np.random.default_rng(p4e.seed_int(seed, "lcr", 2)))[0], np.ones(len(sy)), ncls, seed)
            mu_s = p4e.balanced_mu(Xs, sy, ncls); Cs = np.cov(Xs.T); Ci = np.cov(Xt.T)
            from run_phase4g_second_moment import excess_dirs, shrink_along
            h0b = scorer.score(h7b.head_logits(H0_full, Xt)); hcb = scorer.score(h7b.head_logits(Hc_full, Xt))
            rep_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, gamma=g,
                                 H0_tgt=round(h0b, 4), Hconflict_tgt=round(hcb, 4),
                                 E4_tgt=round(scorer.score(h7b.head_logits(Hc_full, Xt - 1.0 * (Xt.mean(0) - mu_s))), 4),
                                 E4b_tgt=round(scorer.score(h7b.head_logits(Hc_full, shrink_along(Xt, excess_dirs(Ci, Cs, 2), Ci, Cs, 1.0))), 4),
                                 ERASE_tgt=round(scorer.score(h7b.head_logits(Hc_full, pc1.erase(Xt, S_subj))), 4),
                                 Hreg_tgt=round(scorer.score(h7b.head_logits(Hreg, Xt)), 4)))
            fw_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, target_scorer_reads=scorer.n,
                                target_labels_used_for_fit=False, target_labels_used_for_selection=False,
                                target_labels_used_for_final_eval_only=True))
        print(f"[7c:{args.stage}] {tag} done", flush=True)

    pc1._w(OUT / "label_conflict_manifest.csv", man_rows)
    pc1._w(OUT / "global_label_histogram_check.csv", hist_rows)
    if args.stage == "gate":
        pc1._w(OUT / "heldin_learnability_gate.csv", gate_rows)
        print(f"wrote gate CSVs over {len(gate_rows)} fold-seed-gamma")
    else:
        pc1._w(OUT / "pseudo_target_transferability_gate.csv", trans_rows)
        pc1._w(OUT / "dose_response_reliance.csv", dose_rows)
        pc1._w(OUT / "target_harm.csv", harm_rows)
        pc1._w(OUT / "repair_secondary_results.csv", rep_rows)
        (OUT / "target_label_firewall.json").write_text(json.dumps(
            dict(n=len(fw_rows), rows=fw_rows, target_labels_used_for_fit=False,
                 target_labels_used_for_selection=False, target_labels_used_for_final_eval_only=True), indent=2) + "\n")
        print(f"wrote FULL CSVs over {len(harm_rows)} fold-seed-gamma")


if __name__ == "__main__":
    main()
