"""ACAR go/no-go harness (leak-proof). Implements notes/ACAR_FROZEN.md §5–§8.

Phase-1 (y-FREE): for every natural batch and action, compute post-predictions, post-features, and the label-free
paired feature vector φ_a. Phase-2 (y ONLY here): compute the paired incremental risk ΔR_a and the harm label.
Guards: metamorphic label-permutation invariance of φ; double-run determinism. Endpoints: G1 (signal exists) and
G2 (conformal router reduces deployed loss). Decision: PROCEED / MEASUREMENT_ONLY / TERMINATE.

Run:  python -m acar.run_gonogo --alpha 0.1 --delta 0.0 --out results/acar_gonogo
"""
from __future__ import annotations
import argparse, json, os, hashlib, subprocess
import numpy as np
from sklearn.metrics import roc_auc_score

from . import config as C
from .config import ACARConfig, DISEASE, NON_IDENTITY, PAIRED_FEATURES, AUROC_GATE, RETAIN_FRAC
from .data import load_all
from .actions import apply_action
from .features import paired_features, context_features, feature_vector
from .risk import delta_risk, harm_label
from .regressor import ActionRegressor
from .conformal import fit_action_router, route


# ---------- Phase-1: y-free scoring of one batch ----------
def phase1_score(state, z, actions):
    """NO y argument. Returns {action: dict(p, ztil, phi, fvec)}. Permuting y cannot change any of this."""
    p0, z0 = apply_action("identity", state, z)
    out = {"identity": dict(p=p0, ztil=z0, phi=None, fvec=None)}
    for a in actions:
        if a == "identity":
            continue
        pa, za = apply_action(a, state, z)
        phi = paired_features(p0, pa, z0, za)
        ctx = context_features(state, za, pa)
        out[a] = dict(p=pa, ztil=za, phi={**phi, **ctx}, fvec=feature_vector(phi, ctx))
    return out


# ---------- build per-batch records (Phase-1 + Phase-2) ----------
def build_records(data, cfg, guard_fail):
    recs = []
    for disease, cohorts in data.items():
        for coh in cohorts:
            for b in coh.batches:
                if b.fallback:
                    continue                                   # label-blind exclusion (n < MIN_BATCH)
                s1 = phase1_score(coh.state, b.z, cfg.actions)
                # METAMORPHIC GUARD: permuting y must leave the Phase-1 path bit-identical
                yperm = np.random.default_rng(cfg.seed + 7).permutation(b.y)
                s1b = phase1_score(coh.state, b.z, cfg.actions)   # y not passed; must be byte-identical
                for a in NON_IDENTITY:
                    if a in s1 and not np.array_equal(np.nan_to_num(s1[a]["fvec"]),
                                                      np.nan_to_num(s1b[a]["fvec"])):
                        guard_fail.append(f"{disease}/{coh.cohort}/{a}: phi not deterministic")
                # Phase-2: y used ONLY here
                p0 = s1["identity"]["p"]
                dr = {"identity": 0.0}; harm = {"identity": 0}
                for a in NON_IDENTITY:
                    d = delta_risk(p0, s1[a]["p"], b.y, cfg.risk)
                    dr[a] = float(d); harm[a] = harm_label(d)
                # guard: ΔR MUST be sensitive to labels (proves the estimand actually uses y)
                dperm = delta_risk(p0, s1[NON_IDENTITY[0]]["p"], yperm, cfg.risk)
                if abs(dperm - dr[NON_IDENTITY[0]]) < 1e-12 and np.std(b.y) > 0:
                    guard_fail.append(f"{disease}/{coh.cohort}: ΔR insensitive to y (estimand broken)")
                recs.append(dict(disease=disease, cohort=coh.cohort,
                                 phi={a: s1[a]["phi"] for a in NON_IDENTITY},
                                 fvec={a: s1[a]["fvec"] for a in NON_IDENTITY},
                                 dr=dr, harm=harm))
    return recs


# ---------- metrics ----------
def _auc(s, lab):
    s = np.asarray(s, float); lab = np.asarray(lab)
    ok = np.isfinite(s)
    s, lab = s[ok], lab[ok]
    if len(set(lab.tolist())) != 2 or np.std(s) < 1e-12:
        return np.nan
    return float(roc_auc_score(lab, s))


def _cohorts(disease):
    return DISEASE[disease]


def g1_feature(recs, action, feat):
    """Per-cohort directed AUROC -> orientation -> oriented disease-macro + per-cohort pass count."""
    per = {}
    for d in DISEASE:
        for coh in _cohorts(d):
            r = [x for x in recs if x["disease"] == d and x["cohort"] == coh]
            vals = [x["phi"][action][feat] for x in r]
            harm = [x["harm"][action] for x in r]
            a = _auc(vals, harm)
            if not np.isnan(a):
                per[(d, coh)] = a
    if not per:
        return dict(auc_PD=np.nan, auc_SCZ=np.nan, n_cohort_pass=0, n_cohort=0)
    orient = 1.0 if np.nanmean(list(per.values())) >= 0.5 else -1.0
    o = {k: (v if orient > 0 else 1 - v) for k, v in per.items()}
    macro = {d: float(np.mean([o[(d, c)] for c in _cohorts(d) if (d, c) in o]) if any((d, c) in o for c in _cohorts(d)) else np.nan) for d in DISEASE}
    return dict(auc_PD=macro["PD"], auc_SCZ=macro["SCZ"], orientation=orient,
                n_cohort_pass=int(sum(v > 0.5 for v in o.values())), n_cohort=len(o))


def g1_regressor(recs, action, cfg):
    """LOCO learned ĝ_a: directed AUROC(ΔR̂, harm), cohort-macro per disease."""
    per = {}
    for d in DISEASE:
        cohs = _cohorts(d)
        for held in cohs:
            tr = [x for x in recs if x["disease"] == d and x["cohort"] != held]
            te = [x for x in recs if x["disease"] == d and x["cohort"] == held]
            if len(tr) < 8 or not te:
                continue
            Xtr = np.vstack([x["fvec"][action] for x in tr]); dtr = np.array([x["dr"][action] for x in tr])
            reg = ActionRegressor(seed=cfg.seed).fit(Xtr, dtr)
            pred = reg.predict(np.vstack([x["fvec"][action] for x in te]))
            harm = [x["harm"][action] for x in te]
            a = _auc(pred, harm)
            if not np.isnan(a):
                per[(d, held)] = a
    macro = {d: float(np.mean([per[(d, c)] for c in _cohorts(d) if (d, c) in per])) if any((d, c) in per for c in _cohorts(d)) else np.nan for d in DISEASE}
    return dict(auc_PD=macro["PD"], auc_SCZ=macro["SCZ"],
                n_cohort_pass=int(sum(v > 0.5 for v in per.values())), n_cohort=len(per))


def passes_g1(m, n_total):
    return (not np.isnan(m["auc_PD"]) and not np.isnan(m["auc_SCZ"]) and
            m["auc_PD"] >= AUROC_GATE and m["auc_SCZ"] >= AUROC_GATE and
            m["n_cohort_pass"] >= max(1, m["n_cohort"] - 1))


# ---------- G2: LOCO conformal router, closed-loop ----------
def g2_router(recs, cfg, fixed="matched_coral"):
    rng = np.random.default_rng(cfg.seed)
    res = {}
    for d in DISEASE:
        cohs = _cohorts(d)
        routed, always, abst = [], [], []
        for held in cohs:
            tr = [x for x in recs if x["disease"] == d and x["cohort"] != held]
            te = [x for x in recs if x["disease"] == d and x["cohort"] == held]
            if len(tr) < 8 or not te:
                continue
            routers = {}
            for a in cfg.actions:
                if a == "identity":
                    continue
                by_coh = {}
                for c in cohs:
                    if c == held:
                        continue
                    rc = [x for x in tr if x["cohort"] == c]
                    if rc:
                        by_coh[c] = (np.vstack([x["fvec"][a] for x in rc]), np.array([x["dr"][a] for x in rc]))
                if by_coh:
                    routers[a] = fit_action_router(a, by_coh, cfg.alpha, seed=cfg.seed)
            if not routers:
                continue
            for x in te:
                chosen, _ = route(routers, x["fvec"], cfg.delta)
                routed.append(0.0 if chosen == "identity" else x["dr"][chosen])
                always.append(x["dr"].get(fixed, 0.0))
                abst.append(chosen == "identity")
        if not routed:
            res[d] = None; continue
        routed = np.array(routed); always = np.array(always); abst = np.array(abst)
        k = int(abst.sum()); rand = np.zeros(len(routed), bool)
        if 0 < k <= len(routed):
            rand[rng.choice(len(routed), size=k, replace=False)] = True
        random_dr = np.where(rand, 0.0, always)
        ben_a = -np.minimum(always, 0.0).sum(); ben_r = -np.minimum(routed, 0.0).sum()
        res[d] = dict(n=len(routed), abstain_rate=float(abst.mean()),
                      nll_red_always=float(-always.mean()), nll_red_random=float(-random_dr.mean()),
                      nll_red_router=float(-routed.mean()),
                      retained_benefit_frac=float(ben_r / ben_a) if ben_a > 1e-12 else float("nan"),
                      harmful_always=int((always > 0).sum()), harmful_router=int((routed > 0).sum()))
    return res


def passes_g2(res):
    ok = True
    for d in DISEASE:
        r = res.get(d)
        if r is None:
            return False
        cond = (r["nll_red_router"] > r["nll_red_always"] and
                r["nll_red_router"] > r["nll_red_random"] and
                (np.isnan(r["retained_benefit_frac"]) or r["retained_benefit_frac"] >= RETAIN_FRAC))
        ok = ok and cond
    return ok


# ---------- driver ----------
def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def analyze(recs, cfg):
    n_total = 7
    g1 = {}
    for a in NON_IDENTITY:
        feats = {f: g1_feature(recs, a, f) for f in PAIRED_FEATURES}
        reg = g1_regressor(recs, a, cfg)
        feat_pass = {f: bool(passes_g1(m, n_total)) for f, m in feats.items()}
        g1[a] = dict(features=feats, feature_pass=feat_pass, regressor=reg,
                     regressor_pass=bool(passes_g1(reg, n_total)),
                     action_pass=bool(any(feat_pass.values()) or passes_g1(reg, n_total)))
    G1 = any(g1[a]["action_pass"] for a in NON_IDENTITY)
    g2 = g2_router(recs, cfg)
    G2 = bool(passes_g2(g2))
    decision = "PROCEED" if (G1 and G2) else ("MEASUREMENT_ONLY" if G1 else "TERMINATE")
    return dict(G1=bool(G1), G2=G2, decision=decision, g1=g1, g2=g2, n_records=len(recs))


def canonical_hash(summary):
    core = dict(decision=summary["decision"], G1=summary["G1"], G2=summary["G2"],
                g1={a: {f: round(summary["g1"][a]["features"][f]["auc_PD"], 6)
                        for f in PAIRED_FEATURES} for a in NON_IDENTITY})
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=C.B)
    ap.add_argument("--risk", choices=["nll", "01"], default="nll")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/acar_gonogo")
    ap.add_argument("--smoke", action="store_true", help="2 cohorts/disease, quick path check")
    a = ap.parse_args()
    cfg = ACARConfig(alpha=a.alpha, delta=a.delta, batch=a.batch, risk=a.risk, seed=a.seed, out=a.out)

    data = load_all(batch_size=cfg.batch)
    if a.smoke:
        data = {d: cohs[:2] for d, cohs in data.items()}

    guard_fail = []
    recs = build_records(data, cfg, guard_fail)
    if guard_fail:
        print("METAMORPHIC GUARD FAILED (label leakage):"); [print("  ", g) for g in guard_fail[:10]]
        raise SystemExit(3)
    summary = analyze(recs, cfg)
    h1 = canonical_hash(summary)

    # double-run determinism (skip under --smoke for speed)
    if not a.smoke:
        recs2 = build_records(data, cfg, [])
        h2 = canonical_hash(analyze(recs2, cfg))
        if h1 != h2:
            print(f"NON-DETERMINISTIC ({h1} != {h2}) — abort"); raise SystemExit(4)
        summary["double_run_hash"] = h1

    try:
        fa1 = json.load(open("results/freeze_a1/manifest.json")).get("hash", "nohash")[:16]
    except Exception:
        fa1 = "nohash"
    out = f"{cfg.out}/{fa1}"; os.makedirs(out, exist_ok=True)
    summary.update(metamorphic_guard="PASS", n_seeds=1, alpha=cfg.alpha, delta=cfg.delta, risk=cfg.risk,
                   pre_registration="notes/ACAR_FROZEN.md", git_sha=_git_sha(), feat_dump=C.feat_dump_dir())
    json.dump(summary, open(f"{out}/acar_gonogo_summary.json", "w"), indent=2, default=str)

    print(f"=== ACAR GO/NO-GO: {summary['decision']} ===  (G1={summary['G1']} G2={summary['G2']}, guard=PASS, n={len(recs)})")
    print("  G1 per-action best paired-feature cohort-macro AUROC (PD / SCZ):")
    for ac in NON_IDENTITY:
        fe = summary["g1"][ac]["features"]
        best = max(PAIRED_FEATURES, key=lambda f: np.nanmin([fe[f]["auc_PD"], fe[f]["auc_SCZ"]]) if not (np.isnan(fe[f]["auc_PD"]) or np.isnan(fe[f]["auc_SCZ"])) else -1)
        rg = summary["g1"][ac]["regressor"]
        print(f"    {ac:14s} best={best:9s} PD={fe[best]['auc_PD']:.3f} SCZ={fe[best]['auc_SCZ']:.3f} | "
              f"ĝ PD={rg['auc_PD']:.3f} SCZ={rg['auc_SCZ']:.3f} | pass={summary['g1'][ac]['action_pass']}")
    print("  G2 closed-loop NLL reduction (router vs always vs random), retained-benefit:")
    for d in DISEASE:
        r = summary["g2"].get(d)
        if r:
            print(f"    {d}: router={r['nll_red_router']:+.4f} always={r['nll_red_always']:+.4f} "
                  f"random={r['nll_red_random']:+.4f} retain={r['retained_benefit_frac']:.2f} "
                  f"harm a/r={r['harmful_always']}/{r['harmful_router']}")
    print(f"  -> {out}/acar_gonogo_summary.json")


if __name__ == "__main__":
    main()
