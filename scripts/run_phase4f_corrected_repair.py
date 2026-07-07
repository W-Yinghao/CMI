#!/usr/bin/env python
"""FSR Phase 4F — corrected confirmatory test of first-moment token neutralization (CPU-only). See FSR_26.

New confirmatory experiment (Phase 4E stays none, frozen). Reuses Phase 4E arm operators VERBATIM (import
run_phase4e_token_centering) but on >=5 FRESH confirm token seeds, and lets the aggregator apply the CORRECTED
comparator-eligibility gate (a task-destructive arm cannot veto E4). Dev seed 0 for mechanism only.

    <icml python> scripts/run_phase4f_corrected_repair.py [--seeds ...] [--folds N]
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
import run_phase4e_token_centering as p4e   # reuse arm operators VERBATIM

OUT = Path("results/fsr_phase4f_corrected_repair")
LAT = p4e.LAT
CK = p4e.CK
DEV_SEED = 0
CONFIRM_SEEDS_4F = [20260721, 20260722, 20260723, 20260724, 20260725, 20260726, 20260727, 20260728]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[DEV_SEED] + CONFIRM_SEEDS_4F)
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
        scorer = p4e.TargetScorer(tgt["y"].astype(int))
        ncls = int(src["src_logits"].shape[1])
        ck = torch.load(CK / f"{tag}_ckpt_best.pt", map_location="cpu", weights_only=False)
        bb = pc1.load_model(ds, ck["config"], ncls); bb.load_state_dict(ck["state_dict"], strict=True); bb.eval()

        for seed in args.seeds:
            is_confirm = seed in CONFIRM_SEEDS_4F
            rng = np.random.default_rng(p4e.seed_int(seed, "sel", tsub))
            inj = p4e.build_seed_injection(bb, sg, stz, ss, sy, sd, tsub, len(tgt["y"]), ncls, seed)
            a_star, unmet, shifts, thr = p4e.alpha_star_rule(bb, sg, stz, ss, sy, sd, inj, rng)
            best, sel_grid = p4e.select_kl(bb, sg, stz, ss, sy, sd, inj, a_star, rng)
            k, lam = best["k"], best["lam"]
            mu = p4e.balanced_mu(ss, sy, ncls)
            S = pc1.subj_subspace(ss + a_star * inj["scale"] * inj["tok_src"], sd, k=k)
            mc = p4e.mechanism_capture(inj["U_t"], inj["V"], inj["c_target"], a_star, inj["scale"], S)
            man.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, is_confirm4f=is_confirm,
                            ncls=ncls, alpha_star=a_star, stress_unmet=unmet, k_star=k, lam_star=lam,
                            margin=round(inj["margin"], 4), scale=round(inj["scale"], 4), c_target=inj["c_target"],
                            sel_netted_rec=round(best["netted_rec"], 4), sel_clean_drop=round(best["clean_drop"], 4)))
            mech.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, is_confirm4f=is_confirm,
                             alpha_star=a_star, **mc))
            arule.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, thr=round(thr, 4),
                              **{f"shift_a{a}": round(shifts[a], 4) for a in p4e.ALPHAS}, alpha_star=a_star))
            for r in sel_grid:
                sel_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, **r))
            for a in p4e.ALPHAS:
                Sa = pc1.subj_subspace(ss + a * inj["scale"] * inj["tok_src"], sd, k=k)
                arms = p4e.run_arms(bb, tg, ttz, ts_, scorer, inj, Sa, mu, a, k, lam, rng)
                denom = arms["orig"] - arms["injected"]

                def netrec(pre):
                    gi = arms[f"{pre}_inj"] - arms["injected"]; gc = arms[f"{pre}_cln"] - arms["orig"]
                    return round(float((gi - gc) / denom), 4) if abs(denom) > 1e-4 else None

                def rawrec(pre):
                    return round(float((arms[f"{pre}_inj"] - arms["injected"]) / denom), 4) if abs(denom) > 1e-4 else None
                res_rows.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, is_confirm4f=is_confirm,
                                     alpha=a, is_alpha_star=(a == a_star), k=k, lam=lam,
                                     bacc_orig=round(arms["orig"], 4), bacc_injected=round(arms["injected"], 4),
                                     induced_harm=round(denom, 4), E0_recovery=rawrec("E0"),
                                     E4_inj_bacc=round(arms["E4_inj"], 4), E4_cln_bacc=round(arms["E4_cln"], 4),
                                     E4_raw_rec=rawrec("E4"), E4_netted_rec=netrec("E4"),
                                     E1_inj_bacc=round(arms["E1_inj"], 4), E1_cln_bacc=round(arms["E1_cln"], 4),
                                     E1_raw_rec=rawrec("E1"), E1_netted_rec=netrec("E1"), E2_netted_rec=netrec("E2"),
                                     E3_inj_bacc=round(arms["E3_inj"], 4), E3_cln_bacc=round(arms["E3_cln"], 4),
                                     E3_raw_rec=rawrec("E3"), E3_netted_rec=netrec("E3"),
                                     ERASE_inj_bacc=round(arms["ERASE_inj"], 4), ERASE_cln_bacc=round(arms["ERASE_cln"], 4),
                                     ERASE_raw_rec=rawrec("ERASE"), ERASE_netted_rec=netrec("ERASE")))
            fw.append(dict(dataset=ds, target_subject=tsub, token_seed=seed, is_confirm4f=is_confirm,
                           target_scorer_reads=scorer.n, target_labels_used_for_fit=False,
                           target_labels_used_for_selection=False, comparator_eligibility_used_target=False,
                           alpha_selection_used_target=False, target_labels_used_for_final_eval_only=True))
        print(f"[4f] {tag} done", flush=True)

    pc1._w(OUT / "phase4f_manifest.csv", man)
    pc1._w(OUT / "phase4f_mechanism_capture.csv", mech)
    pc1._w(OUT / "phase4f_source_heldout_selection.csv", sel_rows)
    pc1._w(OUT / "phase4f_alpha_rule.csv", arule)
    pc1._w(OUT / "phase4f_token_centering_results.csv", res_rows)
    (OUT / "phase4f_target_label_firewall.json").write_text(json.dumps(
        dict(n=len(fw), dev_seed=DEV_SEED, confirm_seeds=CONFIRM_SEEDS_4F, rows=fw,
             target_labels_used_for_fit=False, target_labels_used_for_selection=False,
             comparator_eligibility_used_target=False, target_labels_used_for_final_eval_only=True), indent=2) + "\n")
    print(f"wrote Phase 4F CSVs over {len(man)} fold-seeds (confirm seeds {CONFIRM_SEEDS_4F})")


if __name__ == "__main__":
    main()
