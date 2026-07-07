#!/usr/bin/env python
"""FSR Phase 4E — branch-local token-neutralization repair (CPU-only). See docs/FSR_24.

The PC1 injected token is a CONSTANT per-batch offset on spatial_z, so the deployable token remover is
first-moment target-batch mean alignment. Design-red-team (w4adgd9z3) reframing:
  E0   exact token subtraction            (oracle bound, non-deployable, headline-excluded)
  E4   full-space mean alignment  z-λ(mean(z_T)-μ_src)          PRIMARY (deployable)
  E1   subspace-restricted        z-λ P_S(mean(z_T)-μ_src)      SECONDARY (u_tsub is out of S by construction)
  E2   counterfactual marginalization                          exploratory
  E3   random-subspace centering                               control
  ERASE subspace erasure z-P_S(z)  (=PC1 R2)                    named control E4/E1 must beat
  CLEAN-target arms  -> token-specific NETTED effect = (E on injected) - (E on clean)   REQUIRED
Mechanism capture (cos(u_tsub,S), captured_fraction u/v) dumped BEFORE scoring. alpha_star by source-only stress
rule (smallest alpha with source-heldout class-directed logit shift >= FRAC*margin, FRAC=1.0). k,λ selected on a
source-heldout pseudo-target, re-derived per seed. Verdict = multi-seed CONFIRM aggregate, NETTED. Target labels
only via TargetScorer.score(). No GPU/retrain/CMI/fbdualpc.

    <icml python> scripts/run_phase4e_token_centering.py [--seeds dev confirm...] [--folds N]
"""
import argparse, glob, hashlib, json, os, sys
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

OUT = Path("results/fsr_phase4e_token_centering")
LAT = Path("results/fsr_rq4_refit/latents")
CK = Path("results/fsr_rq4_refit/ckpt")
BR_SLOT = 2
DEV_SEED = 0
CONFIRM_SEEDS = [20260707, 20260708, 20260709]
ALPHAS = [0.5, 1.0, 1.5, 2.0]
FRAC = 1.0
K_GRID = [1, 2, 4]
LAM_GRID = [0.5, 1.0]
SAFE_DROP = 0.01
N_PSEUDO = 3


def bacc(y, logits):
    return float(balanced_accuracy_score(y, logits.argmax(1)))


class TargetScorer:
    def __init__(self, y):
        self._y = np.asarray(y); self.n = 0

    def score(self, logits):
        self.n += 1
        return bacc(self._y, logits)


def seed_int(token_seed, *parts):
    s = "|".join([str(token_seed)] + [str(p) for p in parts])
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def uvec_seeded(token_seed, subj, dim):
    v = np.random.default_rng(seed_int(token_seed, "u", subj)).standard_normal(dim)
    return v / (np.linalg.norm(v) + 1e-9)


def assign_cd_seeded(sy, sd, ncls, token_seed):
    cd = {}
    for d in np.unique(sd):
        cnt = np.bincount(sy[sd == d], minlength=ncls)
        cands = np.where(cnt == cnt.max())[0]
        cd[int(d)] = int(cands[seed_int(token_seed, "c", int(d)) % len(cands)])
    return cd


def balanced_mu(Z, y, ncls):
    ms = [Z[y == c].mean(0) for c in range(ncls) if (y == c).any()]
    return np.mean(ms, axis=0)


def proj(v, basis):
    if basis.shape[0] == 0:
        return np.zeros_like(v)
    Q, _ = np.linalg.qr(basis.T)
    return (v @ Q) @ Q.T


def center_full(z_T, mu, lam):
    return z_T - lam * (z_T.mean(0) - mu)


def center_sub(z_T, mu, S, lam):
    return z_T - lam * proj(z_T.mean(0) - mu, S)


def mechanism_capture(u_t, V, c_target, alpha, scale, S):
    tok = pc1.unit(pc1.unit(u_t) + pc1.unit(V[c_target]))
    full = alpha * scale * tok
    def cap(vec):
        return float(np.linalg.norm(proj(vec, S)) / (np.linalg.norm(vec) + 1e-9))
    return dict(cos_u_S=round(cap(pc1.unit(u_t)), 4), captured_fraction=round(cap(full), 4),
                u_capture=round(cap(pc1.unit(u_t)), 4), v_capture=round(cap(pc1.unit(V[c_target])), 4))


def build_seed_injection(bb, sg, stz, ss, sy, sd, tsub, ty_len, ncls, token_seed):
    """All source-only token machinery for one (fold, token_seed)."""
    cd = assign_cd_seeded(sy, sd, ncls, token_seed)
    c_target = seed_int(token_seed, "ct", tsub) % ncls
    base_src = pc1.recompose(bb, sg, stz, ss)
    mtop = np.partition(base_src, -2, axis=1)
    margin = max(float(np.median(base_src[np.arange(len(sy)), sy] - mtop[:, -2])), 0.2)
    V = pc1.class_dirs(bb, sg, stz, ss, BR_SLOT, ncls, nsamp=len(ss))
    subj_ids = np.unique(sd)
    U = {int(d): uvec_seeded(token_seed, int(d), ss.shape[1]) for d in subj_ids}
    U_t = uvec_seeded(token_seed, tsub, ss.shape[1])
    tok_src = pc1.token_matrix([int(x) for x in sd], [cd[int(x)] for x in sd], U, V)
    tok_tgt = pc1.token_matrix([tsub] * ty_len, [c_target] * ty_len, {tsub: U_t}, V)
    shift = pc1.recompose(bb, sg, stz, ss + tok_src)
    s_shift = float(np.mean(shift[np.arange(len(sy)), [cd[int(x)] for x in sd]] -
                            base_src[np.arange(len(sy)), [cd[int(x)] for x in sd]]))
    scale = margin / s_shift if s_shift > 1e-6 else 0.0
    return dict(cd=cd, c_target=c_target, margin=margin, V=V, U=U, U_t=U_t, tok_src=tok_src,
                tok_tgt=tok_tgt, scale=scale, base_src=base_src, subj_ids=subj_ids)


def alpha_star_rule(bb, sg, stz, ss, sy, sd, inj, rng):
    """Smallest alpha with source-heldout class-directed shift >= FRAC*margin (source-only)."""
    cd, margin, scale, tok_src = inj["cd"], inj["margin"], inj["scale"], inj["tok_src"]
    base = inj["base_src"]
    subj_ids = inj["subj_ids"]
    ho = set(int(x) for x in rng.choice(subj_ids, max(1, len(subj_ids) // 3), replace=False))
    ho_mask = np.array([int(d) in ho for d in sd])
    idx = np.where(ho_mask)[0]
    cds = np.array([cd[int(sd[i])] for i in idx])
    shifts = {}
    for a in ALPHAS:
        z = ss.copy(); z[ho_mask] = ss[ho_mask] + a * scale * tok_src[ho_mask]
        lg = pc1.recompose(bb, sg, stz, z)
        shifts[a] = float(np.mean(lg[idx, cds] - base[idx, cds]))
    thr = FRAC * margin
    star = next((a for a in ALPHAS if shifts[a] >= thr), None)
    return (star if star is not None else 2.0), (star is None), shifts, thr


def select_kl(bb, sg, stz, ss, sy, sd, inj, alpha, rng):
    """Source-heldout pseudo-target selection of (k, lambda) for E1 and lambda for E4 (netted recovery)."""
    V, U, scale = inj["V"], inj["U"], inj["scale"]
    ncls = V.shape[0]
    subj_ids = inj["subj_ids"]
    ptargets = rng.choice(subj_ids, min(N_PSEUDO, len(subj_ids)), replace=False)
    rows = []
    for k in K_GRID:
        for lam in LAM_GRID:
            recs, safes = [], []
            for pt in ptargets:
                pm = sd == pt; rest = ~pm
                if pm.sum() < 5 or len(np.unique(sd[rest])) < 2:
                    continue
                c_pt = seed_int(1, "pt", int(pt)) % ncls
                tok_pt = pc1.unit(pc1.unit(U[int(pt)]) + pc1.unit(V[c_pt]))
                # inject rest with their real tokens to build S; pseudo-target injected with its token
                inj_rest = ss[rest] + alpha * scale * inj["tok_src"][rest]
                S = pc1.subj_subspace(inj_rest, sd[rest], k=k)
                mu = balanced_mu(ss[rest], sy[rest], ncls)
                z_clean = ss[pm]; z_inj = z_clean + alpha * scale * tok_pt
                yb = sy[pm]
                b_orig = bacc(yb, pc1.recompose(bb, sg[pm], stz[pm], z_clean))
                b_inj = bacc(yb, pc1.recompose(bb, sg[pm], stz[pm], z_inj))
                b_e1_inj = bacc(yb, pc1.recompose(bb, sg[pm], stz[pm], center_sub(z_inj, mu, S, lam)))
                b_e1_cln = bacc(yb, pc1.recompose(bb, sg[pm], stz[pm], center_sub(z_clean, mu, S, lam)))
                denom = b_orig - b_inj
                netted = (b_e1_inj - b_inj) - (b_e1_cln - b_orig)
                if abs(denom) > 1e-4:
                    recs.append(netted / denom)
                safes.append(b_orig - b_e1_cln)   # clean-task drop
            if recs:
                rows.append(dict(k=k, lam=lam, netted_rec=float(np.mean(recs)),
                                 clean_drop=float(np.mean(safes)) if safes else 0.0))
    ok = [r for r in rows if r["clean_drop"] <= SAFE_DROP]
    pool = ok if ok else rows
    best = max(pool, key=lambda r: r["netted_rec"]) if pool else dict(k=2, lam=1.0, netted_rec=0.0, clean_drop=0.0)
    return best, rows


def run_arms(bb, tg, ttz, ts_, scorer, inj, S, mu, alpha, k, lam, rng):
    """All arms at (alpha,k,lam) on the REAL target: injected + clean -> netted. Returns dict of bAccs."""
    scale, tok_tgt, V, c_target = inj["scale"], inj["tok_tgt"], inj["V"], inj["c_target"]
    ncls = V.shape[0]
    z_clean = ts_
    z_inj = ts_ + alpha * scale * tok_tgt
    Srand = rng.standard_normal((k, ts_.shape[1]))

    def sc(z):
        return scorer.score(pc1.recompose(bb, tg, ttz, z))

    def e2(zc):
        lg = np.mean([pc1.recompose(bb, tg, ttz, zc + alpha * scale * pc1.unit(V[c]))
                      for c in range(ncls)], axis=0)
        return scorer.score(lg)
    out = {}
    out["orig"] = sc(z_clean)
    out["injected"] = sc(z_inj)
    out["E0_inj"] = sc(z_inj - alpha * scale * tok_tgt)
    out["E4_inj"] = sc(center_full(z_inj, mu, lam));   out["E4_cln"] = sc(center_full(z_clean, mu, lam))
    out["E1_inj"] = sc(center_sub(z_inj, mu, S, lam)); out["E1_cln"] = sc(center_sub(z_clean, mu, S, lam))
    out["E3_inj"] = sc(z_inj - lam * proj(z_inj.mean(0) - mu, Srand))
    out["E3_cln"] = sc(z_clean - lam * proj(z_clean.mean(0) - mu, Srand))
    out["ERASE_inj"] = sc(pc1.erase(z_inj, S));        out["ERASE_cln"] = sc(pc1.erase(z_clean, S))
    out["E2_inj"] = e2(center_sub(z_inj, mu, S, lam)); out["E2_cln"] = e2(center_sub(z_clean, mu, S, lam))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[DEV_SEED] + CONFIRM_SEEDS)
    ap.add_argument("--folds", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    man, mech, sel_rows, arule, res_rows, fw = ([] for _ in range(6))

    mans = sorted(glob.glob(str(LAT / "*_latent_dump_manifest.json")))
    if args.folds:
        mans = mans[:args.folds]
    for mp in mans:
        M = json.load(open(mp))
        ds, tag, tsub = M["dataset"], M["tag"], M["target_subject"]
        src = np.load(LAT / f"{tag}_source_latents.npz"); tgt = np.load(LAT / f"{tag}_target_latents.npz")
        sg, stz, ss = src["src_graph_z"], src["src_temporal_z"], src["src_spatial_z"]
        tg, ttz, ts_ = tgt["tgt_graph_z"], tgt["tgt_temporal_z"], tgt["tgt_spatial_z"]
        sy, sd = src["y"].astype(int), src["d"].astype(int)
        scorer = TargetScorer(tgt["y"].astype(int))
        ncls = int(src["src_logits"].shape[1])
        ck = torch.load(CK / f"{tag}_ckpt_best.pt", map_location="cpu", weights_only=False)
        bb = pc1.load_model(ds, ck["config"], ncls); bb.load_state_dict(ck["state_dict"], strict=True); bb.eval()

        for seed in args.seeds:
            rng = np.random.default_rng(seed_int(seed, "sel", tsub))
            inj = build_seed_injection(bb, sg, stz, ss, sy, sd, tsub, len(tgt["y"]), ncls, seed)
            a_star, unmet, shifts, thr = alpha_star_rule(bb, sg, stz, ss, sy, sd, inj, rng)
            best, sel_grid = select_kl(bb, sg, stz, ss, sy, sd, inj, a_star, rng)
            k, lam = best["k"], best["lam"]
            # S, mu from FULL injected source (real pipeline)
            inj_src_full = ss + a_star * inj["scale"] * inj["tok_src"]
            S = pc1.subj_subspace(inj_src_full, sd, k=k)
            mu = balanced_mu(ss, sy, ncls)
            mc = mechanism_capture(inj["U_t"], inj["V"], inj["c_target"], a_star, inj["scale"], S)
            is_dev = (seed == DEV_SEED)
            man.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, is_dev=is_dev,
                            ncls=ncls, alpha_star=a_star, stress_unmet=unmet, k_star=k, lam_star=lam,
                            margin=round(inj["margin"], 4), scale=round(inj["scale"], 4),
                            c_target=inj["c_target"], sel_netted_rec=round(best["netted_rec"], 4),
                            sel_clean_drop=round(best["clean_drop"], 4)))
            mech.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, alpha_star=a_star, **mc))
            arule.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, thr=round(thr, 4),
                              **{f"shift_a{a}": round(shifts[a], 4) for a in ALPHAS}, alpha_star=a_star))
            for r in sel_grid:
                sel_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, **r))
            # score arms at alpha_star (headline) and full grid (appendix)
            for a in ALPHAS:
                Sa = pc1.subj_subspace(ss + a * inj["scale"] * inj["tok_src"], sd, k=k)
                arms = run_arms(bb, tg, ttz, ts_, scorer, inj, Sa, mu, a, k, lam, rng)
                denom = arms["orig"] - arms["injected"]
                def netrec(pre):
                    g_inj = arms[f"{pre}_inj"] - arms["injected"]
                    g_cln = arms[f"{pre}_cln"] - arms["orig"]
                    return round(float(((g_inj - g_cln) / denom)), 4) if abs(denom) > 1e-4 else None
                def rawrec(pre):
                    return round(float((arms[f"{pre}_inj"] - arms["injected"]) / denom), 4) if abs(denom) > 1e-4 else None
                res_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, is_dev=is_dev,
                                     alpha=a, is_alpha_star=(a == a_star), k=k, lam=lam,
                                     bacc_orig=round(arms["orig"], 4), bacc_injected=round(arms["injected"], 4),
                                     induced_harm=round(denom, 4),
                                     E0_recovery=rawrec("E0"),
                                     E4_inj_bacc=round(arms["E4_inj"], 4), E4_cln_bacc=round(arms["E4_cln"], 4),
                                     E4_raw_rec=rawrec("E4"), E4_netted_rec=netrec("E4"),
                                     E1_inj_bacc=round(arms["E1_inj"], 4), E1_cln_bacc=round(arms["E1_cln"], 4),
                                     E1_raw_rec=rawrec("E1"), E1_netted_rec=netrec("E1"),
                                     E2_netted_rec=netrec("E2"),
                                     E3_inj_bacc=round(arms["E3_inj"], 4), E3_netted_rec=netrec("E3"),
                                     ERASE_inj_bacc=round(arms["ERASE_inj"], 4), ERASE_netted_rec=netrec("ERASE")))
            fw.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, target_scorer_reads=scorer.n,
                           target_labels_used_for_fit=False, target_labels_used_for_selection=False,
                           alpha_selection_used_target=False, target_labels_used_for_final_eval_only=True))
        print(f"[4e] {tag} done (seeds {args.seeds})", flush=True)

    pc1._w(OUT / "phase4e_manifest.csv", man)
    pc1._w(OUT / "phase4e_mechanism_capture.csv", mech)
    pc1._w(OUT / "phase4e_source_heldout_selection.csv", sel_rows)
    pc1._w(OUT / "phase4e_alpha_rule.csv", arule)
    pc1._w(OUT / "phase4e_token_centering_results.csv", res_rows)
    (OUT / "phase4e_target_label_firewall.json").write_text(json.dumps(
        dict(n=len(fw), dev_seed=DEV_SEED, confirm_seeds=CONFIRM_SEEDS, rows=fw,
             note="target labels read only via TargetScorer.score(); reads cumulative per fold across seeds/alphas.",
             target_labels_used_for_fit=False, target_labels_used_for_selection=False,
             target_labels_used_for_final_eval_only=True), indent=2) + "\n")
    print(f"wrote Phase 4E CSVs over {len(man)} fold-seeds")


if __name__ == "__main__":
    main()
