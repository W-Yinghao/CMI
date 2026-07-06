#!/usr/bin/env python
"""FSR Phase 4C PC1-S — injected subject-token positive control (CPU-only, frozen dumps + checkpoints).

Inject a KNOWN harmful branch-local shortcut into a frozen branch latent and show the FSR protocol
detects/localizes/repairs it. All source-derived; target labels score only. No GPU, no retrain, no
target-label fit, alpha never chosen by target bAcc. See docs/FSR_19_PC1_SUBJECT_TOKEN_PROTOCOL.md.

Token = normalize(u_d) [subject-unique, L1] + normalize(v_{b,c_d}) [class-directed via frozen head, L5/L6].
Inject z_b' = z_b + alpha*scale*normalize(token); scale calibrated (source-only) so alpha=1 ~ median
source correct-vs-runnerup margin. Sign: task_drop = bAcc_orig - bAcc_erased; <0 (erasing helps) = harmful.

    <icml python> scripts/run_pc1_subject_token.py
"""
import csv, glob, json, os
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from cmi.run_loso import _infer_ch_names
from cmi.models.fb_lgg_dualcmi import central_strip_groups
from cmi.models.backbones import build_backbone

R = Path("results/fsr_pc1_subject_token")
LAT = Path("results/fsr_rq4_refit/latents")
CK = Path("results/fsr_rq4_refit/ckpt")
NCH = {"BNCI2014_001": 22, "BNCI2015_001": 13}
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0]
REPAIR_ALPHAS = [1.0, 2.0]
PRIMARY = "spatial_z"
CONTROLS = ["graph_z", "temporal_z"]
BR2SLOT = {"graph_z": 0, "temporal_z": 1, "spatial_z": 2}
RNG = np.random.default_rng(0)


def load_model(ds, cfg, ncls):
    nch, nt = NCH[ds], int(cfg["resample"] * (cfg["tmax"] - cfg["tmin"])) + 1
    ch, _ = _infer_ch_names(ds, nch)
    g, n, _ = central_strip_groups(ds, ch)
    bb = build_backbone("FBCSPLGGGraph", nch, nt, ncls, device="cpu", ch_names=ch, groups=g,
                        group_names=n, grouping_scheme="central_strip_v1", fusion_floor=0.0)
    return bb


def recompose(bb, gz, tz, sz):
    with torch.no_grad():
        return bb.head3(bb._fuse3(torch.as_tensor(gz, dtype=torch.float32),
                                  torch.as_tensor(tz, dtype=torch.float32),
                                  torch.as_tensor(sz, dtype=torch.float32))).numpy()


def class_dirs(bb, gz, tz, sz, slot, ncls, nsamp=512):
    idx = RNG.choice(len(sz), min(nsamp, len(sz)), replace=False)
    ins = [torch.as_tensor(a[idx], dtype=torch.float32) for a in (gz, tz, sz)]
    ins[slot] = ins[slot].clone().requires_grad_(True)
    logits = bb.head3(bb._fuse3(*ins))
    V = []
    for c in range(ncls):
        if ins[slot].grad is not None:
            ins[slot].grad.zero_()
        logits[:, c].sum().backward(retain_graph=(c < ncls - 1))
        V.append(ins[slot].grad.mean(0).detach().numpy().copy())
    V = np.stack(V)
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)


def uvec(subj, dim):
    r = np.random.default_rng(abs(hash(("u", str(subj)))) % (2 ** 32))
    v = r.standard_normal(dim)
    return v / (np.linalg.norm(v) + 1e-9)


def assign_cd(sy, sd, ncls):
    cd = {}
    for d in np.unique(sd):
        cnt = np.bincount(sy[sd == d], minlength=ncls)
        cands = np.where(cnt == cnt.max())[0]
        cd[int(d)] = int(cands[abs(hash(("c", int(d)))) % len(cands)])
    return cd


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def token_matrix(subjects, classes, U, V):
    """Per-sample unit token = normalize( normalize(u_subj) + normalize(v_class) )."""
    t = unit(np.stack([U[s] for s in subjects])) + unit(np.stack([V[c] for c in classes]))
    return unit(t)


def erase(Z, basis):
    if basis.shape[0] == 0:
        return Z.copy()
    Q, _ = np.linalg.qr(basis.T)
    return Z - (Z @ Q) @ Q.T


def subj_subspace(Z, d, k=2):
    doms = np.unique(d)
    if len(doms) < 2:
        return np.zeros((0, Z.shape[1]))
    mu = Z.mean(0)
    M = np.stack([Z[d == s].mean(0) - mu for s in doms])
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[:min(k, Vt.shape[0])]


def probe(Z, d):
    d = np.asarray(d)
    if len(np.unique(d)) < 2 or np.min(np.bincount(d)) < 5:
        return float("nan")
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    pr = np.zeros_like(d)
    for tr, te in skf.split(Z, d):
        pr[te] = LogisticRegression(max_iter=200).fit(Z[tr], d[tr]).predict(Z[te])
    return float(balanced_accuracy_score(d, pr))


def symkl(a, b):
    pa = torch.softmax(torch.tensor(a), 1).numpy() + 1e-9
    pb = torch.softmax(torch.tensor(b), 1).numpy() + 1e-9
    return float(np.mean(np.sum(pa * np.log(pa / pb) + pb * np.log(pb / pa), 1)))


def main():
    R.mkdir(parents=True, exist_ok=True)
    inj_rows, alpha_rows, san_rows, l1_rows, harm_rows, rep_rows, rk_rows = ([] for _ in range(7))
    for mp in sorted(glob.glob(str(LAT / "*_latent_dump_manifest.json"))):
        man = json.load(open(mp))
        ds, tag, tsub = man["dataset"], man["tag"], man["target_subject"]
        src = np.load(LAT / f"{tag}_source_latents.npz")
        tgt = np.load(LAT / f"{tag}_target_latents.npz")
        sg, stz, ss = src["src_graph_z"], src["src_temporal_z"], src["src_spatial_z"]
        tg, ttz, ts_ = tgt["tgt_graph_z"], tgt["tgt_temporal_z"], tgt["tgt_spatial_z"]
        sy, sd = src["y"].astype(int), src["d"].astype(int)
        ty = tgt["y"].astype(int)
        ncls = int(src["src_logits"].shape[1])
        ck = torch.load(CK / f"{tag}_ckpt_best.pt", map_location="cpu", weights_only=False)
        bb = load_model(ds, ck["config"], ncls)
        bb.load_state_dict(ck["state_dict"], strict=True)
        bb.eval()
        branch_src = {"graph_z": sg, "temporal_z": stz, "spatial_z": ss}
        branch_tgt = {"graph_z": tg, "temporal_z": ttz, "spatial_z": ts_}

        cd = assign_cd(sy, sd, ncls)
        c_target = abs(hash(("ct", str(tsub)))) % ncls
        # source margin (source labels allowed)
        base_src_logits = recompose(bb, sg, stz, ss)
        mtop = np.partition(base_src_logits, -2, axis=1)
        margin = float(np.median(base_src_logits[np.arange(len(sy)), sy] - mtop[:, -2]))
        margin = max(margin, 0.2)
        logit_orig_tgt = recompose(bb, tg, ttz, ts_)
        bacc_orig = balanced_accuracy_score(ty, logit_orig_tgt.argmax(1))

        branches = [PRIMARY] + CONTROLS
        for br in branches:
            slot = BR2SLOT[br]
            V = class_dirs(bb, sg, stz, ss, slot, ncls)              # [ncls, dim] source-only
            U = {int(d): uvec(d, ss.shape[1]) for d in np.unique(sd)}
            U[c_target] = U.get(c_target)  # placeholder
            U_t = uvec(tsub, ss.shape[1])
            tok_src = token_matrix([int(x) for x in sd], [cd[int(x)] for x in sd], U, V)
            tok_tgt = token_matrix([tsub] * len(ty), [c_target] * len(ty), {tsub: U_t}, V)
            # calibrate scale so alpha=1 -> mean source class-directed shift ~ margin
            sc_src1 = branch_src[br] + tok_src
            packs = dict(graph_z=sg, temporal_z=stz, spatial_z=ss); packs[br] = sc_src1
            shift = recompose(bb, packs["graph_z"], packs["temporal_z"], packs["spatial_z"])
            s_shift = float(np.mean(shift[np.arange(len(sy)), [cd[int(x)] for x in sd]] -
                                    base_src_logits[np.arange(len(sy)), [cd[int(x)] for x in sd]]))
            scale = margin / s_shift if s_shift > 1e-6 else 0.0
            san_rows.append(dict(dataset=ds, target_subject=tsub, branch=br, margin=round(margin, 4),
                                 unit_class_shift=round(s_shift, 4), scale=round(scale, 4),
                                 class_shift_positive=bool(s_shift > 0)))
            if br == PRIMARY:
                inj_rows.append(dict(dataset=ds, target_subject=tsub, branch=br, c_target=c_target,
                                     n_source_subjects=len(np.unique(sd)), ncls=ncls,
                                     cd_assignment=";".join(f"{k}:{v}" for k, v in sorted(cd.items()))))

            alist = ALPHAS if br == PRIMARY else [1.0]
            for a in alist:
                inj_src = {**{k: branch_src[k] for k in packs}}
                inj_src[br] = branch_src[br] + a * scale * tok_src
                inj_tgt = {"graph_z": tg, "temporal_z": ttz, "spatial_z": ts_}
                inj_tgt[br] = branch_tgt[br] + a * scale * tok_tgt
                lg_t = recompose(bb, inj_tgt["graph_z"], inj_tgt["temporal_z"], inj_tgt["spatial_z"])
                bacc_inj = balanced_accuracy_score(ty, lg_t.argmax(1))
                l1 = probe(inj_src[br], sd)
                # L5: erase source-estimated subject subspace from injected TARGET branch
                basis = subj_subspace(inj_src[br], sd)
                er = inj_tgt[br].copy()
                er = erase(er, basis)
                p2 = {**inj_tgt}; p2[br] = er
                lg_e = recompose(bb, p2["graph_z"], p2["temporal_z"], p2["spatial_z"])
                rb = RNG.standard_normal((min(2, ss.shape[1]), ss.shape[1]))
                p3 = {**inj_tgt}; p3[br] = erase(inj_tgt[br], rb)
                lg_r = recompose(bb, p3["graph_z"], p3["temporal_z"], p3["spatial_z"])
                td = bacc_inj - balanced_accuracy_score(ty, lg_e.argmax(1))
                tdr = bacc_inj - balanced_accuracy_score(ty, lg_r.argmax(1))
                alpha_rows.append(dict(dataset=ds, target_subject=tsub, branch=br, alpha=a,
                                       bacc_orig=round(bacc_orig, 4), bacc_injected=round(bacc_inj, 4),
                                       induced_harm=round(bacc_orig - bacc_inj, 4),
                                       l1_subject_bacc=round(l1, 4) if l1 == l1 else "",
                                       l5_task_drop=round(td, 4), l5_task_drop_random=round(tdr, 4),
                                       l5_symkl=round(symkl(lg_t, lg_e), 4),
                                       l5_symkl_random=round(symkl(lg_t, lg_r), 4)))
                if br == PRIMARY:
                    l1_rows.append(dict(dataset=ds, target_subject=tsub, alpha=a,
                                        l1_subject_bacc=round(l1, 4) if l1 == l1 else ""))
                    harm_rows.append(dict(dataset=ds, target_subject=tsub, alpha=a,
                                          bacc_orig=round(bacc_orig, 4), bacc_injected=round(bacc_inj, 4),
                                          induced_harm=round(bacc_orig - bacc_inj, 4),
                                          l5_task_drop=round(td, 4), l6_harmful=bool(td < 0)))

            # ---- Repair (primary only) at alpha in REPAIR_ALPHAS ----
            if br == PRIMARY:
                for a in REPAIR_ALPHAS:
                    inj_tgt = {"graph_z": tg, "temporal_z": ttz, "spatial_z": ts_}
                    inj_tgt[br] = branch_tgt[br] + a * scale * tok_tgt
                    inj_src_b = branch_src[br] + a * scale * tok_src
                    lg_inj = recompose(bb, inj_tgt["graph_z"], inj_tgt["temporal_z"], inj_tgt["spatial_z"])
                    bacc_inj = balanced_accuracy_score(ty, lg_inj.argmax(1))
                    # R1 oracle: remove known injected {u_target, v_{c_target}} span
                    oracle = np.stack([unit(U_t), unit(V[c_target])])
                    # R2 source-estimated subject subspace
                    r2b = subj_subspace(inj_src_b, sd)
                    # R3 random-k
                    r3b = RNG.standard_normal((2, ss.shape[1]))
                    res = {}
                    # R0_exact: subtract the known token (attributability upper bound; recovers ~1.0)
                    p0 = {**inj_tgt}; p0[br] = inj_tgt[br] - a * scale * tok_tgt
                    lg0 = recompose(bb, p0["graph_z"], p0["temporal_z"], p0["spatial_z"])
                    b0 = balanced_accuracy_score(ty, lg0.argmax(1))
                    denom0 = (bacc_orig - bacc_inj)
                    res["R0_exact"] = (b0, ((b0 - bacc_inj) / denom0) if abs(denom0) > 1e-6 else float("nan"))
                    for name, basis in (("R1_oracle", oracle), ("R2_source_est", r2b), ("R3_random_k", r3b)):
                        p = {**inj_tgt}; p[br] = erase(inj_tgt[br], basis)
                        lg = recompose(bb, p["graph_z"], p["temporal_z"], p["spatial_z"])
                        b = balanced_accuracy_score(ty, lg.argmax(1))
                        denom = (bacc_orig - bacc_inj)
                        res[name] = (b, ((b - bacc_inj) / denom) if abs(denom) > 1e-6 else float("nan"))
                    rep_rows.append(dict(dataset=ds, target_subject=tsub, alpha=a, branch=br,
                                         bacc_orig=round(bacc_orig, 4), bacc_injected=round(bacc_inj, 4),
                                         R0_exact_bacc=round(res["R0_exact"][0], 4),
                                         R0_recovery=round(res["R0_exact"][1], 4),
                                         R1_oracle_bacc=round(res["R1_oracle"][0], 4),
                                         R1_recovery=round(res["R1_oracle"][1], 4),
                                         R2_source_est_bacc=round(res["R2_source_est"][0], 4),
                                         R2_recovery=round(res["R2_source_est"][1], 4),
                                         R3_random_bacc=round(res["R3_random_k"][0], 4),
                                         R3_recovery=round(res["R3_random_k"][1], 4)))
                    rk_rows.append(dict(dataset=ds, target_subject=tsub, alpha=a,
                                        R1_minus_R3=round(res["R1_oracle"][1] - res["R3_random_k"][1], 4),
                                        R2_minus_R3=round(res["R2_source_est"][1] - res["R3_random_k"][1], 4)))
        print(f"[pc1] {tag} done (bacc_orig={bacc_orig:.3f})", flush=True)

    _w(R / "pc1_injection_manifest.csv", inj_rows)
    _w(R / "pc1_alpha_grid.csv", alpha_rows)
    _w(R / "pc1_token_direction_sanity.csv", san_rows)
    _w(R / "pc1_l1_subject_decode.csv", l1_rows)
    _w(R / "pc1_l5_l6_harm_curve.csv", harm_rows)
    _w(R / "pc1_repair_results.csv", rep_rows)
    _w(R / "pc1_randomk_specificity.csv", rk_rows)
    print(f"wrote PC1 CSVs over {len(inj_rows)} folds")


def _w(p, rows):
    if not rows:
        Path(p).write_text("")
        return
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
