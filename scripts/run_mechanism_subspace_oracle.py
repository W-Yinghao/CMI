#!/usr/bin/env python
"""Mechanism-Subspace Oracle runner (AMENDMENT 03 = shared-null CONDITIONAL estimand). --smoke = real-EEG
ENGINEERING smoke (2ds x 2bb x 2subj x seed0 x 4 families; random blocks 2x10). NO scientific verdict from the
smoke. Full M1 (126 cells) HOLD. Pipeline per fold: ONE shared-null projector N = null(Cbar) shared by all
families; PRIMARY contrast basis = N @ TopEig(N^T G_dis N); rule/grad null-projected; B_cond = negative ref.
PRIMARY specificity control = SHARED_NULL_HAAR (Haar in Gr(r, span N), 2 blocks), ambient Haar = SECONDARY. Both
informed and random report SYMMETRIC safe-vs-safe AND unc-vs-unc. Firewall: source-only construction; Y_cal only
for non-deployable exhaustive selection; T_query only for the session-macro outcome. Only the project owner may
stop a scientific line. Manuscript FROZEN.

  python scripts/run_mechanism_subspace_oracle.py --smoke
"""
from __future__ import annotations
import argparse, glob, json, subprocess, sys
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
    pats = [str(REPO / "results/cmi_trace_relaxation_ladder" / f"dgcnn_graph_z_{ds}" / "*.npz"),
            str(REPO / "results/cmi_trace_p0p1/objective_comparison" / ds / "audit" / "*erm*seed*.audit.npz")]
    out = []
    for pat in pats:
        out += [(p, "audit") for p in sorted(glob.glob(pat)) if any(f"seed{s}" in Path(p).name for s in seeds)]
    return out


def _build_families(Zs_w, ys, dsc):
    """All family bases share ONE fold-level shared-null projector N (A03.3). Returns (bundle, fail_meta)."""
    cd = MS.build_contrast_disagreement(Zs_w, ys, dsc)
    if cd["fail_closed"]:
        return None, dict(fail=cd["reason"])
    bc = MS.build_shared_null_contrast_basis(cd)
    if bc.get("fail_closed"):
        return None, dict(fail=bc["reason"], shared_null_dim=bc.get("shared_null_dim"))
    N = bc["N"]; fams = {}
    fams["contrast_disagreement"] = (bc["orthonormal_basis"], dict(numerical_rank=bc["numerical_rank"],
        shared_null_dim=bc["shared_null_dim"], shared_rank=bc["shared_rank"], gen_eigs=bc["generalized_eigenvalues"][:4]))
    rr = MS.fit_shared_residual_ridge(Zs_w, ys, dsc)
    if not rr.get("fail_closed"):
        br = MS.build_shared_null_gram_basis(rr["G_rule"], N)
        fams["rule_disagreement"] = (br["orthonormal_basis"], dict(numerical_rank=br["numerical_rank"], kkt_residual=rr["kkt_residual"]))
    gd = MS.build_class_conditional_gradient_disagreement(Zs_w, ys, dsc)
    if not gd.get("fail_closed"):
        bg = MS.build_shared_null_gram_basis(gd["G_grad"], N)
        fams["gradient_disagreement"] = (bg["orthonormal_basis"], dict(numerical_rank=bg["numerical_rank"]))
    Bcond = TM.whitened_cond_basis(Zs_w, ys, dsc, max_rank=MS.DICT_MAX_RANK)
    fams["B_cond_negative_ref"] = (Bcond, dict(numerical_rank=int(Bcond.shape[0])))
    return dict(N=N, G_shared=cd["G_shared"], G_dis=cd["G_dis"], families=fams), None


def _select_score(Zs_w, ys, dsc, Q, Xcal_w, ycal, Xq_w, yq, sq, D):
    """Symmetric: returns (unc dict, safe dict) each with selected indices + query dU (P0.1)."""
    acts = MS.build_exhaustive_action_family(Q.shape[0], MS.MAX_SUBSET_RANK)
    S_unc = MS.select_on_target_cal(Zs_w, ys, Q, acts, Xcal_w, ycal, source_safe=False)
    S_safe = MS.select_on_target_cal(Zs_w, ys, Q, acts, Xcal_w, ycal, source_safe=True, ds=dsc)
    U_unc = TM._orthonormal(Q[S_unc]) if S_unc else np.zeros((0, D))
    U_safe = TM._orthonormal(Q[S_safe]) if S_safe else np.zeros((0, D))
    return (dict(sel=S_unc, dU=MS.score_on_target_query(Zs_w, ys, U_unc, Xq_w, yq, sq), U=U_unc),
            dict(sel=S_safe, dU=MS.score_on_target_query(Zs_w, ys, U_safe, Xq_w, yq, sq), U=U_safe))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--seeds", nargs="+", default=["0"])
    ap.add_argument("--n_subjects", type=int, default=2); ap.add_argument("--n_random", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=2)
    a = ap.parse_args()
    cfg = REPO / "configs/cmi_trace_mechanism_subspace_oracle_v4.yaml"
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
                if "session_target" not in f:
                    rows.append(dict(dataset=ds, backbone=bb, subject=str(f.get("heldout_subject")),
                                     seed=int(f.get("seed", -1)), family="ALL", status="skipped",
                                     reason="NO_SESSION_AXIS_FOR_QUERY_SPLIT", feature_object=str(f.get("backbone"))))
                    continue
                Zs = np.asarray(f["Z_source"], float); ys = np.asarray(f["y_source"]).astype(int); dsc = _dense(f["subj_source"])
                Zt = np.asarray(f["Z_target"], float); yt = np.asarray(f["y_target"]).astype(int)
                subj = str(f["heldout_subject"]); sd = int(f["seed"])
                W = TM.source_whitener(Zs); Zs_w = TM.to_whitened(Zs, W); D = Zs.shape[1]
                cal, qry, _ = session_split(f["session_target"], yt, sd)
                Xcal_w, ycal = TM.to_whitened(Zt[cal], W), yt[cal]
                Xq_w, yq, sq = TM.to_whitened(Zt[qry], W), yt[qry], np.asarray(f["session_target"])[qry]
                bundle, fmeta = _build_families(Zs_w, ys, dsc)
                if bundle is None:
                    rows.append(dict(dataset=ds, backbone=bb, subject=subj, seed=sd, family="ALL", status="skipped", **fmeta))
                    continue
                N, G_shared, G_dis = bundle["N"], bundle["G_shared"], bundle["G_dis"]
                for fam in FAMILIES:
                    if fam not in bundle["families"]:
                        rows.append(dict(dataset=ds, backbone=bb, subject=subj, seed=sd, family=fam, status="skipped",
                                         reason="FAMILY_BASIS_UNAVAILABLE")); continue
                    B, meta = bundle["families"][fam]
                    if B is None or B.shape[0] == 0:
                        rows.append(dict(dataset=ds, backbone=bb, subject=subj, seed=sd, family=fam, status="skipped",
                                         reason="EMPTY_BASIS", **meta)); continue
                    r = B.shape[0]
                    inf_unc, inf_safe = _select_score(Zs_w, ys, dsc, B, Xcal_w, ycal, Xq_w, yq, sq, D)
                    # PRIMARY control = SHARED_NULL_HAAR (2 blocks, cell-specific seeds); SECONDARY = ambient Haar.
                    def _controls(builder, ctrl_name):
                        recs = []
                        for blk in range(a.blocks):
                            seed = MS.cell_seed(ds, bb, subj, sd, fam, ctrl_name, blk)
                            for j, Q in enumerate(builder(seed)):
                                ru, rs = _select_score(Zs_w, ys, dsc, Q, Xcal_w, ycal, Xq_w, yq, sq, D)
                                recs.append(dict(block=blk, rid=j, dU_unc=ru["dU"], dU_safe=rs["dU"],
                                                 sel_unc=ru["sel"], sel_safe=rs["sel"],
                                                 shared_overlap=MS.shared_overlap(Q, G_shared),
                                                 subspace_overlap=float(np.linalg.norm(B @ Q.T, "fro") ** 2 / r),
                                                 gdis_capture=MS.gdis_capture_fraction(rs["U"], G_dis, N),
                                                 dict_hash=TM._hash(Q)))
                        return recs
                    null_haar = None
                    if N is not None and N.shape[1] > r:                                   # non-degenerate control space
                        null_haar = _controls(lambda s: MS.build_shared_null_haar_dictionaries(N, r, a.n_random, s), "SHARED_NULL_HAAR")
                    ambient = _controls(lambda s: MS.build_ambient_random_dictionaries(D, r, a.n_random, s), "AMBIENT_HAAR")
                    fw = dict(source_only_construction=True, Ycal_used_for_selection=True, Yquery_used_for_selection=False,
                              Xquery_used_for_selection=False, Yquery_used_for_outcome=True)
                    rows.append(dict(dataset=ds, backbone=bb, subject=subj, seed=sd, family=fam, status="ok",
                                     numerical_rank=r, shared_null_dim=(int(N.shape[1]) if N is not None else None),
                                     dU_informed_unc=inf_unc["dU"], dU_informed_safe=inf_safe["dU"],
                                     selected_unc=inf_unc["sel"], selected_safe=inf_safe["sel"],
                                     informed_shared_overlap=MS.shared_overlap(B, G_shared),
                                     informed_gdis_capture=MS.gdis_capture_fraction(inf_safe["U"], G_dis, N),
                                     primary_control="SHARED_NULL_HAAR" if null_haar is not None else "SHARED_NULL_CONTROL_LOW_DOF",
                                     shared_null_haar=null_haar, ambient=ambient,
                                     projector_hash_safe=TM._hash(inf_safe["U"]) if inf_safe["U"].shape[0] else "identity",
                                     firewall=fw, config_hash=cfg_hash, git_sha=sha,
                                     **{k: meta[k] for k in meta if k not in ("numerical_rank", "shared_null_dim")}))
                    nh = null_haar if null_haar is not None else []
                    nh_safe = np.mean([x["dU_safe"] for x in nh]) if nh else float("nan")
                    print(f"  {ds[:11]}/{bb} sub{subj} {fam[:8]}: r={r} nulldim={N.shape[1] if N is not None else '-'} "
                          f"infSafe={inf_safe['dU']:+.3f} nullHaarSafe={nh_safe:+.3f} "
                          f"cap_inf={rows[-1]['informed_gdis_capture']:.2f} ctrl={rows[-1]['primary_control']}", flush=True)
    tag = "smoke" if a.smoke else "full"
    with open(OUT / f"mechanism_oracle_rows_{tag}.jsonl", "w") as fh:
        [fh.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n") for r in rows]
    print(f"[mech-{tag}] {len(rows)} rows -> {OUT}; families={FAMILIES}; PRIMARY=SHARED_NULL_HAAR; NO scientific verdict (engineering smoke)")


if __name__ == "__main__":
    main()
