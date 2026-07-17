#!/usr/bin/env python
"""Mechanism-Subspace Oracle runner (M0.2). --smoke = Stage-C real-EEG ENGINEERING smoke (2 datasets x 2
backbones x 2 subjects x seed 0 x 4 candidate families; random blocks 2x10). NO scientific verdict from the
smoke. Full M1 (126 cells) is HOLD. Firewall: source-only construction; Y_cal only for non-deployable exhaustive
selection; T_query only for the session-macro outcome. Only the project owner may stop a scientific line.

  python scripts/run_mechanism_subspace_oracle.py --smoke
"""
from __future__ import annotations
import argparse, glob, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from tos_cmi.eeg.relaxation_ladder import feat_from_tos_dump, feat_from_audit_npz, _dense
from tos_cmi.eval import targetx_metric as TM
from tos_cmi.eval import mechanism_subspace as MS
from tos_cmi.eval.targetx_observability import session_split

OUT = REPO / "results" / "cmi_trace_mechanism_subspace"
DATASETS = ["BNCI2014_001", "BNCI2015_001"]
FAMILIES = ["contrast_disagreement", "rule_disagreement", "gradient_disagreement", "B_cond_negative_ref"]


def _cells(ds, bb, seeds):
    if bb == "EEGNet":
        return [(p, "tos") for p in sorted(glob.glob(str(REPO / "tos_cmi/results/tos_cmi_eeg_frozen" /
                f"{ds}_EEGNet_LOSO" / "sub*_erm_lam0_seed*.npz"))) if any(p.endswith(f"_seed{s}.npz") for s in seeds)]
    return [(p, "audit") for p in sorted(glob.glob(str(REPO / "results/cmi_trace_relaxation_ladder" /
            f"dgcnn_graph_z_{ds}" / "*.npz"))) if any(f"seed{s}" in Path(p).name for s in seeds)]


def _build_family(fam, Zs_w, ys, dsc, max_rank):
    if fam == "contrast_disagreement":
        cd = MS.build_contrast_disagreement(Zs_w, ys, dsc)
        if cd["fail_closed"]:
            return None, dict(fail=cd["reason"])
        gm = MS.solve_generalized_mechanism_basis(cd["G_dis"], cd["G_shared"], max_rank=max_rank)
        if gm.get("below_resolution"):
            return None, dict(fail="TASK_MECHANISM_BELOW_RESOLUTION")
        return gm["orthonormal_basis"], dict(numerical_rank=gm["numerical_rank"], gen_eigs=gm["generalized_eigenvalues"][:4],
                                             G_shared_spec=list(np.linalg.svd(cd["G_shared"], compute_uv=False)[:4]),
                                             G_dis_spec=list(np.linalg.svd(cd["G_dis"], compute_uv=False)[:4]), G_shared=cd["G_shared"])
    if fam == "rule_disagreement":
        rr = MS.fit_shared_residual_ridge(Zs_w, ys, dsc)
        if rr.get("fail_closed"):
            return None, dict(fail=rr["reason"])
        gb = MS.basis_from_gram(rr["G_rule"], max_rank=max_rank)
        return gb["orthonormal_basis"], dict(numerical_rank=gb["numerical_rank"], raw_spec=rr["raw_singular_values"][:4])
    if fam == "gradient_disagreement":
        gd = MS.build_class_conditional_gradient_disagreement(Zs_w, ys, dsc)
        if gd.get("fail_closed"):
            return None, dict(fail=gd["reason"])
        gb = MS.basis_from_gram(gd["G_grad"], max_rank=max_rank)
        return gb["orthonormal_basis"], dict(numerical_rank=gb["numerical_rank"], raw_spec=gd["raw_singular_values"][:4])
    B = TM.whitened_cond_basis(Zs_w, ys, dsc, max_rank=max_rank)
    return B, dict(numerical_rank=int(B.shape[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--seeds", nargs="+", default=["0"])
    ap.add_argument("--n_subjects", type=int, default=2); ap.add_argument("--n_random", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=2); ap.add_argument("--pool", type=int, default=500)
    a = ap.parse_args()
    cfg = REPO / "configs/cmi_trace_mechanism_subspace_oracle_v3.yaml"
    cfg_hash = MS.config_hash(cfg) if cfg.exists() else "no_config"
    try:
        sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        sha = "unknown"
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in DATASETS:
        for bb in ["EEGNet", "DGCNN"]:
            cells = _cells(ds, bb, a.seeds)
            if a.smoke:
                cells = cells[: a.n_subjects]
            for cp, kind in cells:
                f = feat_from_tos_dump(cp) if kind == "tos" else feat_from_audit_npz(cp)
                Zs = np.asarray(f["Z_source"], float); ys = np.asarray(f["y_source"]).astype(int); dsc = _dense(f["subj_source"])
                Zt = np.asarray(f["Z_target"], float); yt = np.asarray(f["y_target"]).astype(int)
                if "session_target" not in f:
                    continue
                W = TM.source_whitener(Zs); Zs_w = TM.to_whitened(Zs, W); D = Zs.shape[1]
                cal, qry, _ = session_split(f["session_target"], yt, int(f["seed"]))
                Xcal_w, ycal = TM.to_whitened(Zt[cal], W), yt[cal]
                Xq_w, yq, sq = TM.to_whitened(Zt[qry], W), yt[qry], np.asarray(f["session_target"])[qry]
                for fam in FAMILIES:
                    B, meta = _build_family(fam, Zs_w, ys, dsc, MS.DICT_MAX_RANK)
                    if B is None or B.shape[0] == 0:
                        rows.append(dict(dataset=ds, backbone=bb, subject=str(f["heldout_subject"]), seed=int(f["seed"]),
                                         family=fam, status="skipped", **meta)); continue
                    acts = MS.build_exhaustive_action_family(B.shape[0], MS.MAX_SUBSET_RANK)
                    S_unc = MS.select_on_target_cal(Zs_w, ys, B, acts, Xcal_w, ycal, source_safe=False)
                    S_safe = MS.select_on_target_cal(Zs_w, ys, B, acts, Xcal_w, ycal, source_safe=True, ds=dsc)
                    U_unc = TM._orthonormal(B[S_unc]) if S_unc else np.zeros((0, D))
                    U_safe = TM._orthonormal(B[S_safe]) if S_safe else np.zeros((0, D))
                    dU_unc = MS.score_on_target_query(Zs_w, ys, U_unc, Xq_w, yq, sq)
                    dU_safe = MS.score_on_target_query(Zs_w, ys, U_safe, Xq_w, yq, sq)
                    # random controls (blocks x n_random): each random dict goes through the SAME select-on-cal
                    # exhaustive family then score-on-query (equal budget).
                    def _rand_score(Q):
                        Sr = MS.select_on_target_cal(Zs_w, ys, Q, MS.build_exhaustive_action_family(Q.shape[0], MS.MAX_SUBSET_RANK), Xcal_w, ycal)
                        return MS.score_on_target_query(Zs_w, ys, TM._orthonormal(Q[Sr]) if Sr else np.zeros((0, D)), Xq_w, yq, sq)
                    amb = [_rand_score(Q) for blk in range(a.blocks)
                           for Q in MS.build_ambient_random_dictionaries(D, B.shape[0], a.n_random, blk)]
                    match = None
                    if "G_shared" in meta:
                        mm = MS.build_shared_profile_matched_dictionaries(B, meta["G_shared"], D, B.shape[0], a.n_random, 0, n_pool=a.pool)
                        match = dict(rmse=mm["matching_rmse"], gap=mm["total_overlap_gap"], verdict=mm["verdict"])
                    fw = dict(source_only_construction=True, Ycal_used_for_selection=True, Yquery_used_for_selection=False,
                              Xquery_used_for_selection=False, Yquery_used_for_outcome=True)
                    rows.append(dict(dataset=ds, backbone=bb, subject=str(f["heldout_subject"]), seed=int(f["seed"]),
                                     family=fam, status="ok", numerical_rank=int(B.shape[0]), n_actions=len(acts),
                                     selected_rank_unconstrained=len(S_unc), selected_rank_safe=len(S_safe),
                                     dU_unconstrained=dU_unc, dU_source_safe=dU_safe,
                                     dU_random_ambient_mean=float(np.mean(amb)) if amb else float("nan"),
                                     n_ambient=len(amb), shared_overlap_match=match,
                                     projector_hash_unc=TM._hash(U_unc) if U_unc.shape[0] else "identity",
                                     firewall=fw, config_hash=cfg_hash, git_sha=sha,
                                     **{k: meta[k] for k in meta if k not in ("G_shared", "numerical_rank")}))
                    print(f"  {ds}/{bb} sub{f['heldout_subject']} {fam[:8]}: rank={B.shape[0]} acts={len(acts)} "
                          f"selU={len(S_unc)}(dU={dU_unc:+.3f}) selSafe={len(S_safe)}(dU={dU_safe:+.3f}) "
                          f"rand={np.mean(amb):+.3f} match={match['verdict'] if match else 'n/a'}", flush=True)
    tag = "smoke" if a.smoke else "full"
    with open(OUT / f"mechanism_oracle_rows_{tag}.jsonl", "w") as fh:
        [fh.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n") for r in rows]
    print(f"[mech-{tag}] {len(rows)} rows -> {OUT}; families={FAMILIES}; NO scientific verdict (engineering smoke)")


if __name__ == "__main__":
    main()
