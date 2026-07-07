"""P1.5 DEPLOYMENT-BRANCH resolution. Audits the ACTUAL deployed CITA+LPC arm (per-cohort nested-selected lambda:
PD->0.1, SCZ->0.3 at seed 0) vs CITA-no-LPC (erm:0), using the IDENTICAL probes, thresholds and decision engine as
the fixed-lambda audit (imported from p15_audit — nothing re-tuned). Consumes ONLY the hash-verified
feat_dump_v4/FEATURE_BOUND_DEPLOYMENT.json: re-hashes every shard's content (hard-stop on any mismatch, no partial
decision). Reports the deployment verdict (RETAIN/DROP_COLLAPSE/INCONCLUSIVE) and, ALONGSIDE, a hash-verified
fixed-lambda=0.1 dose-response panel (does NOT substitute for the deployment-branch verdict).
"""
import os, sys, json, glob, hashlib, tempfile
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import LabelEncoder
from p15_audit import (freeze_gate, domain_probe_audit, representation_metrics, decision_engine, saturation_met,
                       PROBE_TIERS, PROBE_ALT, _canon_hash_full)
from p15_bind_redump import freeze_pred_hash, canon_pred_hash16, expected_cohorts

V4 = "results/feat_dump_v4"
DEP = f"{V4}/FEATURE_BOUND_DEPLOYMENT.json"
STRONG = "mlp256x2"

def _rep_and_leak(o, rng):
    """representation metrics on z_te + grouped, Y-conditioned domain-leakage probes on z_se (D = source cohort)."""
    rep = representation_metrics(o["z_te"], np.asarray(o["y_te"]), np.random.default_rng(0))
    D = LabelEncoder().fit_transform([str(c) for c in o["cohort_id_se"]])
    leak = None
    if len(set(D)) >= 2:
        Y = np.asarray(o["y_se"]); G = np.array([str(g) for g in o["group_id_se"]])
        Zin = np.c_[np.asarray(o["z_se"], float), np.eye(int(Y.max()) + 1)[Y]]     # condition on Y
        leak = {t: v["heldout_bacc"] for t, v in domain_probe_audit(Zin, D, Y, G, PROBE_TIERS + PROBE_ALT, rng).items()}
    return rep, leak

def _load_verified(cond, cohort, lbl, feat_hash_te=None):
    """Load a v4 shard, verify its prob_te reproduces Freeze A1 AND (if given) its z_te content hash. Hard-stop."""
    fn = f"{V4}/audit_{cond}_{cohort}_{lbl.replace(':', '_')}.npz"
    o = np.load(fn, allow_pickle=True)
    if canon_pred_hash16(o["prob_te"]) != freeze_pred_hash(cond, cohort, lbl):
        raise RuntimeError(f"PRED-HASH != Freeze A1 for {fn} — hard stop")
    fh = feat_hash_te if feat_hash_te is not None else str(o["feat_hash_te"])
    if _canon_hash_full(o["z_te"]) != fh or _canon_hash_full(o["z_se"]) != str(o["feat_hash_se"]):
        raise RuntimeError(f"CONTENT HASH MISMATCH {fn} — hard stop, no partial decision")
    return o

def _aggregate(reps, leaks):
    rep = {k: float(np.mean([r[k] for r in reps])) for k in
           ("eff_rank", "stable_rank", "scatter_ratio", "feat_var", "task_bacc")}
    L = [l for l in leaks if l]
    agg = (lambda t: float(np.mean([l[t] for l in L]))) if L else (lambda t: float("nan"))
    return rep, agg

def run_contrast(pairs, rng):
    """pairs: list of (cond, cohort, erm_shard_loaded, lpc_shard_loaded). Returns (decision, predicates, detail)."""
    rep_e, rep_l, leak_e, leak_l = [], [], [], []
    for cond, cohort, oe, ol in pairs:
        re_, le_ = _rep_and_leak(oe, rng); rl_, ll_ = _rep_and_leak(ol, rng)
        rep_e.append(re_); rep_l.append(rl_); leak_e.append(le_); leak_l.append(ll_)
    rep_erm, ae = _aggregate(rep_e, leak_e); rep_lpc, al = _aggregate(rep_l, leak_l)
    tier_ladder = [ae(t) for t in PROBE_TIERS]
    decision, predicates = decision_engine(ae(STRONG), al(STRONG), max(ae("gbt"), ae("knn")), max(al("gbt"), al("knn")),
                                           rep_erm, rep_lpc, saturation_met([abs(x) for x in tier_ladder]))
    detail = dict(leakage=dict(strong_probe=STRONG, erm=ae(STRONG), lpc=al(STRONG),
                               alt_erm=max(ae("gbt"), ae("knn")), alt_lpc=max(al("gbt"), al("knn")),
                               tier_ladder=dict(zip(PROBE_TIERS, tier_ladder))),
                  representation=dict(erm=rep_erm, lpc=rep_lpc))
    return decision, predicates, detail

def main():
    ok, fz, src = freeze_gate()
    if not ok:
        print(f"BLOCKED_ON_FREEZE_A1: {src}"); sys.exit(2)
    if not os.path.exists(DEP):
        print(f"no deployment bind ({DEP}); run p15_bind_deployment after the v4 redump completes"); sys.exit(3)
    bm = json.load(open(DEP)); mh = bm.pop("manifest_hash")
    if hashlib.sha256(json.dumps(bm, sort_keys=True).encode()).hexdigest() != mh:
        print("deployment manifest hash mismatch — hard stop"); sys.exit(4)
    bm["manifest_hash"] = mh
    if bm["freeze_a1_hash"] != fz["hash"]:
        print("deployment bind freeze hash != current Freeze A1 — hard stop"); sys.exit(4)
    rng = np.random.default_rng(0)
    # index the bound shards by (cond,cohort,role) with their verified feat_hash_te
    idx = {(s["cond"], s["cohort"], s["role"]): s for s in bm["shards"]}
    try:
        # ---- (A) DEPLOYMENT branch: erm:0 vs the per-cohort selected lambda ----
        dep_pairs = []
        for key, sel in bm["deployment_selection_seed0"].items():
            cond, cohort = key.split("/")
            es = idx.get((cond, cohort, "CITA-no-LPC")); ls = idx.get((cond, cohort, "CITA+LPC-deployment"))
            if es is None or ls is None:                       # erm-selected cohort (no +LPC contrast) — skip, noted in bind
                continue
            oe = _load_verified(cond, cohort, "erm:0", es["feat_hash_te"])
            ol = _load_verified(cond, cohort, sel, ls["feat_hash_te"])
            dep_pairs.append((cond, cohort, oe, ol))
        if not dep_pairs:
            raise RuntimeError("no deployment +LPC contrasts to audit")
        dep_dec, dep_pred, dep_det = run_contrast(dep_pairs, rng)

        # ---- (B) ALONGSIDE: fixed lambda=0.1 dose-response (hash-verified; NOT the deployment verdict) ----
        dose_pairs = []
        for cond in ("PD", "SCZ"):
            for cohort in expected_cohorts(cond):
                oe = _load_verified(cond, cohort, "erm:0")
                ol = _load_verified(cond, cohort, "lpc_prior:0.1")
                dose_pairs.append((cond, cohort, oe, ol))
        dose_dec, dose_pred, dose_det = run_contrast(dose_pairs, rng)
    except Exception as e:
        print(f"deployment audit aborted ({e}); no partial output left."); sys.exit(5)

    result = dict(
        deployment_verdict=dict(decision=dep_dec, predicates=dep_pred, n_cohorts=len(dep_pairs),
                                deployment_selection_seed0=bm["deployment_selection_seed0"], **dep_det),
        dose_response_fixed_lambda01=dict(decision=dose_dec, predicates=dose_pred, n_cohorts=len(dose_pairs),
                                          note="dose-response only; does NOT decide the deployment branch", **dose_det),
        fixed_lambda03_prior_verdict="DROP_LPC_COLLAPSE (results/p15_audit/4c7f495d8e48b930; FINAL for fixed 0.3)",
        freeze_a1_hash=fz["hash"], bind_manifest_hash=mh, instrumentation_commit=bm["instrumentation_commit"])
    os.makedirs("results/p15_audit_deployment", exist_ok=True)
    out = f"results/p15_audit_deployment/{fz['hash'][:16]}"
    if os.path.exists(out):
        print(f"{out} exists — refusing to overwrite (immutable)"); sys.exit(4)
    tmp = tempfile.mkdtemp(prefix="p15dep_tmp_", dir="results")
    json.dump(result, open(f"{tmp}/deployment_decision.json", "w"), indent=2, default=str)
    os.rename(tmp, out)
    print(f"=== P1.5 DEPLOYMENT verdict: {dep_dec} ({len(dep_pairs)} cohorts: {bm['deployment_selection_seed0']}) ===")
    print(f"    leakage strong-probe erm/lpc: {dep_det['leakage']['erm']:.3f} / {dep_det['leakage']['lpc']:.3f}")
    print(f"    task_bacc erm/lpc: {dep_det['representation']['erm']['task_bacc']:.1f} / {dep_det['representation']['lpc']['task_bacc']:.1f}"
          f" | eff_rank erm/lpc: {dep_det['representation']['erm']['eff_rank']:.1f} / {dep_det['representation']['lpc']['eff_rank']:.1f}")
    print(f"--- alongside fixed-lambda=0.1 dose-response: {dose_dec} (NOT the deployment decision) ---")
    print(f"    -> {out}/deployment_decision.json")

if __name__ == "__main__":
    main()
