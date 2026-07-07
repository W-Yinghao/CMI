#!/usr/bin/env python
"""FSR Phase 4G — controlled second-moment repair positive control (CPU-only). See FSR_29.

Inject a strictly MEAN-NULLED second-moment shortcut into spatial_z; test whether E4b (covariance/CORAL
alignment) repairs it where E4 (first-moment mean alignment) is INSUFFICIENT by construction. Reuses 4E/4F
operators + arm scaffolding; target-X-only + source; target labels score only.

Injection types: 'varmod' (class-directed variance modulation) primary; 'covtoken' (subject x class rank-2
covariance token) secondary. Strict mean-null: P -= P.mean(0). alpha_star by source-only stress rule (per-sample
class-directed logit STD >= FRAC*margin). E4b = shrinkage-CORAL, (lambda,eps) selected source-heldout.

    <icml python> scripts/run_phase4g_second_moment.py [--seeds ...] [--folds N]
"""
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np
import torch
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

OUT = Path("results/fsr_phase4g_second_moment")
LAT, CK = p4e.LAT, p4e.CK
DEV_SEED = 0
CONFIRM_SEEDS = [20260721, 20260722, 20260723, 20260724, 20260725, 20260726, 20260727, 20260728]
ALPHAS = [1.0, 2.0, 3.0]
FRAC = 1.0
LAM_GRID = [0.5, 1.0]
K_GRID = [1, 2]
SAFE_DROP = 0.01
N_PSEUDO = 3
INJ_TYPES = ["varmod", "covtoken"]


def bacc(y, lg):
    return float(balanced_accuracy_score(y, lg.argmax(1)))


def excess_dirs(Ci, Cs, k):
    """Top-k directions where injected target variance most EXCEEDS source variance (the inflated directions)."""
    D = (Ci - Cs); D = (D + D.T) / 2
    w, V = np.linalg.eigh(D)
    idx = np.argsort(w)[::-1][:k]        # most-inflated first
    return V[:, idx].T                   # [k, d] unit rows


def shrink_along(z, dirs, Ci, Cs, lam):
    """Rescale the deviation along each unit direction from target-var down to source-var level (shrink only).
    E4b = dirs from excess_dirs (targeted); E3 = random dirs (control). First moment untouched (deviation only)."""
    mu = z.mean(0); zc = z.copy()
    for q in dirs:
        q = pc1.unit(q)
        st = np.sqrt(max(float(q @ Ci @ q), 1e-9)); sr = np.sqrt(max(float(q @ Cs @ q), 1e-9))
        r = min(sr / st, 1.0)            # shrink toward source variance; never inflate
        proj = (z - mu) @ q
        zc = zc - lam * (1 - r) * np.outer(proj, q)
    return zc


def rand_dirs(d, k, rng):
    A = rng.standard_normal((k, d))
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


def mk_perturb(z, sc_dirs, codes, alpha, scale):
    """P = alpha*scale*sum_j codes[j][:,None]*unit(dir_j); strictly mean-nulled. Returns z+P."""
    P = np.zeros_like(z)
    for dvec, code in zip(sc_dirs, codes):
        P += code[:, None] * pc1.unit(dvec)[None, :]
    P *= alpha * scale
    P -= P.mean(0)                     # STRICT mean-null
    return z + P, P


def rademacher(n, rng):
    return rng.choice([-1.0, 1.0], n).astype(float)


def build_inj(bb, sg, stz, ss, sy, sd, tsub, ncls, token_seed, injtype):
    cd = p4e.assign_cd_seeded(sy, sd, ncls, token_seed)
    c_target = p4e.seed_int(token_seed, "ct", tsub) % ncls
    base = pc1.recompose(bb, sg, stz, ss)
    mtop = np.partition(base, -2, axis=1)
    margin = max(float(np.median(base[np.arange(len(sy)), sy] - mtop[:, -2])), 0.2)
    V = pc1.class_dirs(bb, sg, stz, ss, 2, ncls, nsamp=len(ss))
    U = {int(d): p4e.uvec_seeded(token_seed, int(d), ss.shape[1]) for d in np.unique(sd)}
    U_t = p4e.uvec_seeded(token_seed, tsub, ss.shape[1])
    # calibrate scale: unit varmod on source -> per-sample class-directed logit STD ~ margin
    rng = np.random.default_rng(p4e.seed_int(token_seed, "cal", tsub))
    s_src = rademacher(len(sy), rng)
    dirs_src = [V[cd[int(d)]] for d in sd]
    z1 = ss + (s_src[:, None] * np.stack([pc1.unit(d) for d in dirs_src]))
    z1 -= (z1 - ss).mean(0)
    lg1 = pc1.recompose(bb, sg, stz, z1)
    cds = np.array([cd[int(d)] for d in sd])
    shift_std = float(np.std(lg1[np.arange(len(sy)), cds] - base[np.arange(len(sy)), cds]))
    scale = margin / shift_std if shift_std > 1e-6 else 1.0
    return dict(cd=cd, c_target=c_target, margin=margin, V=V, U=U, U_t=U_t, scale=scale, base=base,
                subj_ids=np.unique(sd), injtype=injtype)


def perturb_source(ss, sy, sd, inj, alpha, rng, mask=None):
    """varmod/covtoken perturbation on source (their own c_d / u_d)."""
    V, U, cd, scale, injtype = inj["V"], inj["U"], inj["cd"], inj["scale"], inj["injtype"]
    idx = np.arange(len(sd)) if mask is None else np.where(mask)[0]
    z = ss.copy()
    if injtype == "varmod":
        code = rademacher(len(idx), rng)
        dirs = [V[cd[int(sd[i])]] for i in idx]
        P = code[:, None] * np.stack([pc1.unit(d) for d in dirs])
    else:  # covtoken
        cu = rademacher(len(idx), rng); cv = rademacher(len(idx), rng)
        P = (cu[:, None] * np.stack([pc1.unit(U[int(sd[i])]) for i in idx]) +
             cv[:, None] * np.stack([pc1.unit(V[cd[int(sd[i])]]) for i in idx]))
    P *= alpha * scale; P -= P.mean(0)
    z[idx] = ss[idx] + P
    return z


def perturb_target(ts, inj, alpha, rng):
    V, U_t, c_target, scale, injtype = inj["V"], inj["U_t"], inj["c_target"], inj["scale"], inj["injtype"]
    n = len(ts)
    if injtype == "varmod":
        return mk_perturb(ts, [V[c_target]], [rademacher(n, rng)], alpha, scale)
    else:
        return mk_perturb(ts, [U_t, V[c_target]], [rademacher(n, rng), rademacher(n, rng)], alpha, scale)


def alpha_star_rule(bb, sg, stz, ss, sy, sd, inj, rng):
    subj = inj["subj_ids"]; ho = set(int(x) for x in rng.choice(subj, max(1, len(subj) // 3), replace=False))
    hom = np.array([int(d) in ho for d in sd]); idx = np.where(hom)[0]
    cds = np.array([inj["cd"][int(sd[i])] for i in idx])
    stds = {}
    for a in ALPHAS:
        z = perturb_source(ss, sy, sd, inj, a, np.random.default_rng(p4e.seed_int(1, "as", a)), mask=hom)
        lg = pc1.recompose(bb, sg, stz, z)
        stds[a] = float(np.std(lg[idx, cds] - inj["base"][idx, cds]))
    thr = FRAC * inj["margin"]
    star = next((a for a in ALPHAS if stds[a] >= thr), None)
    return (star if star is not None else ALPHAS[-1]), (star is None), stds, thr


def select_k_lam(bb, sg, stz, ss, sy, sd, inj, alpha, rng):
    """Source-heldout pseudo-target selection of (k, lambda) for E4b excess-direction shrinkage."""
    subj = inj["subj_ids"]; pts = rng.choice(subj, min(N_PSEUDO, len(subj)), replace=False)
    Cs_full = np.cov(ss.T)
    rows = []
    for k in K_GRID:
        for lam in LAM_GRID:
            recs, drops = [], []
            for pt in pts:
                pm = sd == pt; rest = ~pm
                if pm.sum() < 5 or len(np.unique(sd[rest])) < 2:
                    continue
                Cs = np.cov(ss[rest].T)
                z_cln = ss[pm]
                z_inj = perturb_target(z_cln, {**inj, "c_target": inj["cd"][int(pt)],
                                                "U_t": inj["U"][int(pt)]}, alpha,
                                       np.random.default_rng(p4e.seed_int(2, "pt", int(pt))))[0]
                yb = sy[pm]; Ci = np.cov(z_inj.T); Cc = np.cov(z_cln.T)
                dirs = excess_dirs(Ci, Cs, k)
                b_o = bacc(yb, pc1.recompose(bb, sg[pm], stz[pm], z_cln))
                b_i = bacc(yb, pc1.recompose(bb, sg[pm], stz[pm], z_inj))
                b_ri = bacc(yb, pc1.recompose(bb, sg[pm], stz[pm], shrink_along(z_inj, dirs, Ci, Cs, lam)))
                b_rc = bacc(yb, pc1.recompose(bb, sg[pm], stz[pm], shrink_along(z_cln, excess_dirs(Cc, Cs, k), Cc, Cs, lam)))
                den = b_o - b_i
                if abs(den) > 1e-4:
                    recs.append(((b_ri - b_i) - (b_rc - b_o)) / den)
                drops.append(b_o - b_rc)
            if recs:
                rows.append(dict(k=k, lam=lam, netted_rec=float(np.mean(recs)),
                                 clean_drop=float(np.mean(drops)) if drops else 0.0))
    ok = [r for r in rows if r["clean_drop"] <= SAFE_DROP]
    pool = ok if ok else rows
    best = max(pool, key=lambda r: r["netted_rec"]) if pool else dict(k=1, lam=1.0, netted_rec=0.0, clean_drop=0.0)
    return best, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[DEV_SEED] + CONFIRM_SEEDS)
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--injtypes", type=str, nargs="+", default=INJ_TYPES)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    man, sanity, mnull, covchk, res, fw = ([] for _ in range(6))

    mans = sorted(glob.glob(str(LAT / "*_latent_dump_manifest.json")))
    if args.folds:
        mans = mans[:args.folds]
    for mp in mans:
        M = json.load(open(mp)); ds, tag, tsub = M["dataset"], M["tag"], M["target_subject"]
        src = np.load(LAT / f"{tag}_source_latents.npz"); tgt = np.load(LAT / f"{tag}_target_latents.npz")
        sg, stz, ss = src["src_graph_z"], src["src_temporal_z"], src["src_spatial_z"]
        tg, ttz, ts_ = tgt["tgt_graph_z"], tgt["tgt_temporal_z"], tgt["tgt_spatial_z"]
        sy, sd = src["y"].astype(int), src["d"].astype(int)
        scorer = p4e.TargetScorer(tgt["y"].astype(int)); ncls = int(src["src_logits"].shape[1])
        ck = torch.load(CK / f"{tag}_ckpt_best.pt", map_location="cpu", weights_only=False)
        bb = pc1.load_model(ds, ck["config"], ncls); bb.load_state_dict(ck["state_dict"], strict=True); bb.eval()
        mu_s = p4e.balanced_mu(ss, sy, ncls); Cs = np.cov(ss.T)
        Cs_tr = float(np.trace(Cs))

        for seed in args.seeds:
            for injt in args.injtypes:
                inj = build_inj(bb, sg, stz, ss, sy, sd, tsub, ncls, seed, injt)
                rng = np.random.default_rng(p4e.seed_int(seed, injt, tsub))
                a_star, unmet, stds, thr = alpha_star_rule(bb, sg, stz, ss, sy, sd, inj, rng)
                best, _ = select_k_lam(bb, sg, stz, ss, sy, sd, inj, a_star, rng)
                k, lam = best["k"], best["lam"]
                orig = scorer.score(pc1.recompose(bb, tg, ttz, ts_))
                man.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, injtype=injt,
                                is_confirm=seed in CONFIRM_SEEDS, alpha_star=a_star, stress_unmet=unmet,
                                k_star=k, lam_star=lam, margin=round(inj["margin"], 4),
                                scale=round(inj["scale"], 4), c_target=inj["c_target"],
                                sel_netted_rec=round(best["netted_rec"], 4), sel_clean_drop=round(best["clean_drop"], 4)))
                for a in ALPHAS:
                    z_inj, P = perturb_target(ts_, inj, a, np.random.default_rng(p4e.seed_int(seed, "tg", a)))
                    mean_disp = float(np.linalg.norm(P.mean(0)))
                    b_inj = scorer.score(pc1.recompose(bb, tg, ttz, z_inj))
                    denom = orig - b_inj
                    Ci = np.cov(z_inj.T); Cc = np.cov(ts_.T)
                    vdir = pc1.unit(inj["V"][inj["c_target"]])
                    excess = float(vdir @ (Ci - Cc) @ vdir)   # injected variance along the class direction
                    # E4b targets the ESTIMATED excess-variance directions; E3 shrinks RANDOM directions (same op)
                    dirs_exc = excess_dirs(Ci, Cs, k); dirs_exc_c = excess_dirs(Cc, Cs, k)
                    dirs_rnd = rand_dirs(ss.shape[1], k, np.random.default_rng(p4e.seed_int(seed, "rp", a)))
                    # ORACLE-E4b: shrink along the TRUE injected direction(s) (known; non-deployable bound)
                    dirs_or = (np.array([vdir]) if injt == "varmod"
                               else np.stack([pc1.unit(inj["U_t"]), vdir]))
                    def sc(z):
                        return scorer.score(pc1.recompose(bb, tg, ttz, z))
                    arms = dict(orig=orig, injected=b_inj)
                    arms["E4_inj"] = sc(z_inj - lam * (z_inj.mean(0) - mu_s)); arms["E4_cln"] = sc(ts_ - lam * (ts_.mean(0) - mu_s))
                    arms["E4b_inj"] = sc(shrink_along(z_inj, dirs_exc, Ci, Cs, lam)); arms["E4b_cln"] = sc(shrink_along(ts_, dirs_exc_c, Cc, Cs, lam))
                    arms["E4bO_inj"] = sc(shrink_along(z_inj, dirs_or, Ci, Cs, lam)); arms["E4bO_cln"] = sc(shrink_along(ts_, dirs_or, Cc, Cs, lam))
                    arms["E3_inj"] = sc(shrink_along(z_inj, dirs_rnd, Ci, Cs, lam)); arms["E3_cln"] = sc(shrink_along(ts_, dirs_rnd, Cc, Cs, lam))
                    arms["ERASE_inj"] = sc(pc1.erase(z_inj, dirs_exc)); arms["ERASE_cln"] = sc(pc1.erase(ts_, dirs_exc_c))
                    # interpretability diagnostics (design-red-team wjdzttrhu)
                    wD, _ = np.linalg.eigh((Ci - Cs + (Ci - Cs).T) / 2)
                    topk_mass = float(np.clip(np.sort(wD)[::-1][:k], 0, None).sum())
                    inj_dominance = float(excess / (topk_mass + 1e-9))          # injection var / top-k excess mass
                    vc_overlap = float(np.mean(np.abs(dirs_exc @ vdir)))         # |estimated dir . true v_c|
                    arm_overlap = float(abs(dirs_exc[0] @ dirs_exc_c[0]))        # injected-arm vs clean-arm dir match

                    def netrec(pre):
                        return round(float(((arms[f"{pre}_inj"] - b_inj) - (arms[f"{pre}_cln"] - orig)) / denom), 4) if abs(denom) > 1e-4 else None

                    def rawrec(pre):
                        return round(float((arms[f"{pre}_inj"] - b_inj) / denom), 4) if abs(denom) > 1e-4 else None
                    res.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, injtype=injt,
                                    is_confirm=seed in CONFIRM_SEEDS, alpha=a, is_alpha_star=(a == a_star),
                                    lam=lam, k=k, bacc_orig=round(orig, 4), bacc_injected=round(b_inj, 4),
                                    induced_harm=round(denom, 4), mean_disp=round(mean_disp, 5),
                                    E4_inj_bacc=round(arms["E4_inj"], 4), E4_cln_bacc=round(arms["E4_cln"], 4),
                                    E4_raw_rec=rawrec("E4"), E4_netted_rec=netrec("E4"),
                                    E4b_inj_bacc=round(arms["E4b_inj"], 4), E4b_cln_bacc=round(arms["E4b_cln"], 4),
                                    E4b_raw_rec=rawrec("E4b"), E4b_netted_rec=netrec("E4b"),
                                    E4bO_inj_bacc=round(arms["E4bO_inj"], 4), E4bO_cln_bacc=round(arms["E4bO_cln"], 4),
                                    E4bO_netted_rec=netrec("E4bO"),
                                    inj_dominance=round(inj_dominance, 4), vc_overlap=round(vc_overlap, 4),
                                    arm_overlap=round(arm_overlap, 4),
                                    E3_inj_bacc=round(arms["E3_inj"], 4), E3_cln_bacc=round(arms["E3_cln"], 4),
                                    E3_raw_rec=rawrec("E3"), E3_netted_rec=netrec("E3"),
                                    ERASE_inj_bacc=round(arms["ERASE_inj"], 4), ERASE_cln_bacc=round(arms["ERASE_cln"], 4),
                                    ERASE_raw_rec=rawrec("ERASE"), ERASE_netted_rec=netrec("ERASE")))
                    if a == a_star:
                        mnull.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, injtype=injt,
                                          mean_disp=round(mean_disp, 5), E4_netted_rec=netrec("E4"), harm=round(denom, 4)))
                        covchk.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, injtype=injt,
                                           excess_var_along_vc=round(excess, 5), inj_dominance=round(inj_dominance, 4),
                                           vc_overlap=round(vc_overlap, 4), arm_overlap=round(arm_overlap, 4),
                                           E4bO_netted_rec=netrec("E4bO"), harm=round(denom, 4)))
                        sanity.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, injtype=injt,
                                           alpha_star=a_star, harm=round(denom, 4), stress_unmet=unmet))
                fw.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, injtype=injt,
                               target_scorer_reads=scorer.n, target_labels_used_for_fit=False,
                               target_labels_used_for_selection=False, comparator_veto_set_used_target=False,
                               alpha_selection_used_target=False, target_labels_used_for_final_eval_only=True))
        print(f"[4g] {tag} done", flush=True)

    pc1._w(OUT / "phase4g_manifest.csv", man)
    pc1._w(OUT / "phase4g_injection_sanity.csv", sanity)
    pc1._w(OUT / "phase4g_mean_null_check.csv", mnull)
    pc1._w(OUT / "phase4g_covariance_shift_check.csv", covchk)
    pc1._w(OUT / "phase4g_repair_results.csv", res)
    pc1._w(OUT / "phase4g_random_controls.csv", [dict(
        dataset=r["dataset"], target_subject=r["target_subject"], token_seed=r["token_seed"], injtype=r["injtype"],
        E3_inj_bacc=r["E3_inj_bacc"], E3_netted_rec=r["E3_netted_rec"], ERASE_inj_bacc=r["ERASE_inj_bacc"],
        ERASE_netted_rec=r["ERASE_netted_rec"]) for r in res if r["is_alpha_star"]])
    (OUT / "phase4g_target_label_firewall.json").write_text(json.dumps(
        dict(n=len(fw), dev_seed=DEV_SEED, confirm_seeds=CONFIRM_SEEDS, rows=fw,
             target_labels_used_for_fit=False, target_labels_used_for_selection=False,
             comparator_veto_set_used_target=False, target_labels_used_for_final_eval_only=True), indent=2) + "\n")
    print(f"wrote Phase 4G CSVs over {len(man)} fold-seed-injtypes")


if __name__ == "__main__":
    main()
