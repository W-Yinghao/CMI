#!/usr/bin/env python
"""Mechanism-Subspace Oracle runner (AMENDMENT 03 = shared-null CONDITIONAL estimand).

Two run modes:
  --smoke                         real-EEG ENGINEERING smoke (NO scientific weight): 2ds x 2bb x n_subjects x seed0
                                  x 4 families; writes a single combined mechanism_oracle_rows_smoke.jsonl.
  --cell-index N --out-dir D      run exactly the Nth cell of the (filtered) manifest and write its own
                                  cell_{...}.jsonl + a .done marker (fail-resumable SLURM array; M1-P).
  --list-cells                    print the filtered cell manifest (index -> dataset/backbone/subject/seed) + exit.

EXECUTION-LAYER filters only (--backbone/--family/--seeds); they change WHICH cells run, never any estimator,
threshold, control or gate. Pipeline per fold: ONE shared-null projector N = null(Cbar) shared by all families;
PRIMARY contrast basis = N @ TopEig(N^T G_dis N); rule/grad null-projected; B_cond = negative ref (NOT shared-null
projected -> excluded from the M1-P primary tranche). PRIMARY specificity control = SHARED_NULL_HAAR (Haar in
Gr(r, span N), 2 blocks), ambient Haar = SECONDARY. Informed and random both report SYMMETRIC safe-vs-safe AND
unc-vs-unc. Firewall: source-only construction; Y_cal only for non-deployable exhaustive selection; T_query only
for the session-macro outcome. Only the project owner may stop a scientific line. Manuscript FROZEN.
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
ALL_FAMILIES = ["contrast_disagreement", "rule_disagreement", "gradient_disagreement", "B_cond_negative_ref"]


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


def _enumerate(datasets, backbones, seeds):
    """Deterministic cell manifest: [(dataset, backbone, cp, kind)] over the requested datasets/backbones/seeds."""
    man = []
    for ds in datasets:
        for bb in backbones:
            for cp, kind in _cells(ds, bb, seeds):
                man.append((ds, bb, cp, kind))
    return man


def _feature_hash(cp):
    return hashlib.sha256(open(cp, "rb").read()).hexdigest()[:16]


def _build_families(Zs_w, ys, dsc, want):
    cd = MS.build_contrast_disagreement(Zs_w, ys, dsc)
    if cd["fail_closed"]:
        return None, dict(fail=cd["reason"])
    bc = MS.build_shared_null_contrast_basis(cd)
    if bc.get("fail_closed"):
        return None, dict(fail=bc["reason"], shared_null_dim=bc.get("shared_null_dim"))
    N = bc["N"]; fams = {}
    if "contrast_disagreement" in want:
        fams["contrast_disagreement"] = (bc["orthonormal_basis"], dict(numerical_rank=bc["numerical_rank"],
            shared_null_dim=bc["shared_null_dim"], shared_rank=bc["shared_rank"], gen_eigs=bc["generalized_eigenvalues"][:4]))
    if "rule_disagreement" in want:
        rr = MS.fit_shared_residual_ridge(Zs_w, ys, dsc)
        if not rr.get("fail_closed"):
            br = MS.build_shared_null_gram_basis(rr["G_rule"], N)
            fams["rule_disagreement"] = (br["orthonormal_basis"], dict(numerical_rank=br["numerical_rank"], kkt_residual=rr["kkt_residual"]))
    if "gradient_disagreement" in want:
        gd = MS.build_class_conditional_gradient_disagreement(Zs_w, ys, dsc)
        if not gd.get("fail_closed"):
            bg = MS.build_shared_null_gram_basis(gd["G_grad"], N)
            fams["gradient_disagreement"] = (bg["orthonormal_basis"], dict(numerical_rank=bg["numerical_rank"]))
    if "B_cond_negative_ref" in want:                                # NOT shared-null projected (negative ref only)
        Bcond = TM.whitened_cond_basis(Zs_w, ys, dsc, max_rank=MS.DICT_MAX_RANK)
        fams["B_cond_negative_ref"] = (Bcond, dict(numerical_rank=int(Bcond.shape[0]), shared_null_projected=False))
    return dict(N=N, G_shared=cd["G_shared"], G_dis=cd["G_dis"], families=fams), None


def _select_score(Zs_w, ys, dsc, Q, Xcal_w, ycal, Xq_w, yq, sq, D):
    acts = MS.build_exhaustive_action_family(Q.shape[0], MS.MAX_SUBSET_RANK)
    S_unc = MS.select_on_target_cal(Zs_w, ys, Q, acts, Xcal_w, ycal, source_safe=False)
    S_safe = MS.select_on_target_cal(Zs_w, ys, Q, acts, Xcal_w, ycal, source_safe=True, ds=dsc)
    U_unc = TM._orthonormal(Q[S_unc]) if S_unc else np.zeros((0, D))
    U_safe = TM._orthonormal(Q[S_safe]) if S_safe else np.zeros((0, D))
    return (dict(sel=S_unc, dU=MS.score_on_target_query(Zs_w, ys, U_unc, Xq_w, yq, sq), U=U_unc),
            dict(sel=S_safe, dU=MS.score_on_target_query(Zs_w, ys, U_safe, Xq_w, yq, sq), U=U_safe))


def _run_cell(ds, bb, cp, kind, want, n_random, blocks, cfg_hash, sha):
    """All requested families for one (dataset, backbone, subject, seed) cell -> list of rows."""
    f = feat_from_tos_dump(cp) if kind == "tos" else feat_from_audit_npz(cp)
    subj = str(f.get("heldout_subject")); sd = int(f.get("seed", -1)); fhash = _feature_hash(cp)
    common = dict(dataset=ds, backbone=bb, subject=subj, seed=sd, config_hash=cfg_hash, git_sha=sha, feature_hash=fhash)
    if "session_target" not in f:
        return [dict(**common, family="ALL", status="skipped", reason="NO_SESSION_AXIS_FOR_QUERY_SPLIT",
                     feature_object=str(f.get("backbone")))]
    Zs = np.asarray(f["Z_source"], float); ys = np.asarray(f["y_source"]).astype(int); dsc = _dense(f["subj_source"])
    Zt = np.asarray(f["Z_target"], float); yt = np.asarray(f["y_target"]).astype(int)
    W = TM.source_whitener(Zs); Zs_w = TM.to_whitened(Zs, W); D = Zs.shape[1]
    cal, qry, _ = session_split(f["session_target"], yt, sd)
    Xcal_w, ycal = TM.to_whitened(Zt[cal], W), yt[cal]
    Xq_w, yq, sq = TM.to_whitened(Zt[qry], W), yt[qry], np.asarray(f["session_target"])[qry]
    bundle, fmeta = _build_families(Zs_w, ys, dsc, want)
    if bundle is None:
        return [dict(**common, family="ALL", status="skipped", **fmeta)]
    N, G_shared, G_dis = bundle["N"], bundle["G_shared"], bundle["G_dis"]
    rows = []
    for fam in want:
        if fam not in bundle["families"]:
            rows.append(dict(**common, family=fam, status="skipped", reason="FAMILY_BASIS_UNAVAILABLE")); continue
        B, meta = bundle["families"][fam]
        if B is None or B.shape[0] == 0:
            rows.append(dict(**common, family=fam, status="skipped", reason="EMPTY_BASIS", **meta)); continue
        r = B.shape[0]
        inf_unc, inf_safe = _select_score(Zs_w, ys, dsc, B, Xcal_w, ycal, Xq_w, yq, sq, D)

        def _controls(builder, ctrl_name):
            recs = []
            for blk in range(blocks):
                seed = MS.cell_seed(ds, bb, subj, sd, fam, ctrl_name, blk)
                for j, Q in enumerate(builder(seed)):
                    ru, rs = _select_score(Zs_w, ys, dsc, Q, Xcal_w, ycal, Xq_w, yq, sq, D)
                    recs.append(dict(block=blk, rid=j, dU_unc=ru["dU"], dU_safe=rs["dU"], sel_unc=ru["sel"], sel_safe=rs["sel"],
                                     shared_overlap=MS.shared_overlap(Q, G_shared),
                                     subspace_overlap=float(np.linalg.norm(B @ Q.T, "fro") ** 2 / r),
                                     dictionary_gdis_capture=MS.gdis_capture_fraction(Q, G_dis, N),
                                     selected_safe_gdis_capture=MS.gdis_capture_fraction(rs["U"], G_dis, N),
                                     dict_hash=TM._hash(Q)))
            return recs
        null_haar = None
        if N is not None and N.shape[1] > r:
            null_haar = _controls(lambda s: MS.build_shared_null_haar_dictionaries(N, r, n_random, s), "SHARED_NULL_HAAR")
        ambient = _controls(lambda s: MS.build_ambient_random_dictionaries(D, r, n_random, s), "AMBIENT_HAAR")
        fw = dict(source_only_construction=True, Ycal_used_for_selection=True, Yquery_used_for_selection=False,
                  Xquery_used_for_selection=False, Yquery_used_for_outcome=True)
        rows.append(dict(**common, family=fam, status="ok", numerical_rank=r,
                         shared_null_dim=(int(N.shape[1]) if N is not None else None),
                         dU_informed_unc=inf_unc["dU"], dU_informed_safe=inf_safe["dU"],
                         selected_unc=inf_unc["sel"], selected_safe=inf_safe["sel"],
                         informed_shared_overlap=MS.shared_overlap(B, G_shared),
                         dictionary_gdis_capture=MS.gdis_capture_fraction(B, G_dis, N),          # FULL rank-r dictionary
                         selected_safe_gdis_capture=MS.gdis_capture_fraction(inf_safe["U"], G_dis, N),  # SELECTED <=3 projector
                         primary_control="SHARED_NULL_HAAR" if null_haar is not None else "SHARED_NULL_CONTROL_LOW_DOF",
                         shared_null_haar=null_haar, ambient=ambient,
                         projector_hash_safe=TM._hash(inf_safe["U"]) if inf_safe["U"].shape[0] else "identity",
                         firewall=fw, **{k: meta[k] for k in meta if k not in ("numerical_rank", "shared_null_dim")}))
        nh = null_haar if null_haar is not None else []
        nh_safe = np.mean([x["dU_safe"] for x in nh]) if nh else float("nan")
        print(f"  {ds[:11]}/{bb} sub{subj} s{sd} {fam[:8]}: r={r} nulldim={N.shape[1] if N is not None else '-'} "
              f"infSafe={inf_safe['dU']:+.3f} nullHaarSafe={nh_safe:+.3f} dictCap={rows[-1]['dictionary_gdis_capture']:.2f} "
              f"selCap={rows[-1]['selected_safe_gdis_capture']:.2f} ctrl={rows[-1]['primary_control']}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--seeds", nargs="+", default=["0"])
    ap.add_argument("--n_subjects", type=int, default=2); ap.add_argument("--n_random", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--backbone", nargs="+", default=["EEGNet", "DGCNN"])          # execution-layer filter
    ap.add_argument("--family", nargs="+", default=ALL_FAMILIES)                    # execution-layer filter
    ap.add_argument("--cell-index", type=int, default=None)                         # SLURM array task
    ap.add_argument("--list-cells", action="store_true")
    ap.add_argument("--out-dir", default=None)                                      # per-cell output dir (array mode)
    a = ap.parse_args()
    want = [x for x in ALL_FAMILIES if x in a.family]
    cfg = REPO / "configs/cmi_trace_mechanism_subspace_oracle_v4.yaml"
    cfg_hash = MS.config_hash(cfg) if cfg.exists() else "no_config"
    try:
        sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        sha = "unknown"

    man = _enumerate(DATASETS, a.backbone, a.seeds)
    if a.list_cells:
        for i, (ds, bb, cp, kind) in enumerate(man):
            f = feat_from_tos_dump(cp) if kind == "tos" else feat_from_audit_npz(cp)
            print(f"{i}\t{ds}\t{bb}\tsub{f.get('heldout_subject')}\tseed{f.get('seed')}\t{Path(cp).name}")
        print(f"# {len(man)} cells; families={want}; config_hash={cfg_hash}; git_sha={sha}")
        return

    if a.cell_index is not None:                                   # ---- M1-P fail-resumable array: one cell ----
        assert a.out_dir, "--cell-index requires --out-dir"
        outd = Path(a.out_dir); outd.mkdir(parents=True, exist_ok=True)
        ds, bb, cp, kind = man[a.cell_index]
        print(f"[mech-cell {a.cell_index}/{len(man)}] {ds}/{bb} {Path(cp).name} families={want} git_sha={sha} config_hash={cfg_hash}", flush=True)
        rows = _run_cell(ds, bb, cp, kind, want, a.n_random, a.blocks, cfg_hash, sha)
        subj = str(rows[0].get("subject")); sd = rows[0].get("seed")
        stem = f"cell_{a.cell_index:03d}_{ds}_{bb}_sub{subj}_seed{sd}"
        with open(outd / f"{stem}.jsonl", "w") as fh:
            [fh.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n") for r in rows]
        (outd / f"{stem}.done").write_text(f"{sha}\t{cfg_hash}\t{len(rows)} rows\n")
        print(f"[mech-cell {a.cell_index}] wrote {len(rows)} rows -> {stem}.jsonl (+.done)")
        return

    # ---- combined mode (smoke / dev) ----
    if a.smoke:
        man = [c for c in man if sum(1 for cc in man if cc[0] == c[0] and cc[1] == c[1]) and True]
        # keep first n_subjects per (ds, bb)
        seen = {}; keep = []
        for c in man:
            k = (c[0], c[1]); seen[k] = seen.get(k, 0) + 1
            if seen[k] <= a.n_subjects:
                keep.append(c)
        man = keep
    OUT.mkdir(parents=True, exist_ok=True); rows = []
    for ds, bb, cp, kind in man:
        rows += _run_cell(ds, bb, cp, kind, want, a.n_random, a.blocks, cfg_hash, sha)
    tag = "smoke" if a.smoke else "full"
    with open(OUT / f"mechanism_oracle_rows_{tag}.jsonl", "w") as fh:
        [fh.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n") for r in rows]
    print(f"[mech-{tag}] {len(rows)} rows -> {OUT}; families={want}; PRIMARY=SHARED_NULL_HAAR; NO scientific verdict (engineering smoke)")


if __name__ == "__main__":
    main()
