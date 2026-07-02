"""BNCI2014_001 LOSO seeds-[0,1,2] MULTI-SEED aggregation (C8): read all 27 committed fold artifacts
(9 targets × 3 seeds), deep-verify each, and report the native K1 (per fold/level/seed + aggregate counts)
and the real multi-seed K2 (worst-held-out-target endpoints across the 3 seeds → reproducible-gain / stop).

    BNCI2014-001 minimum-seed K1/K2 run. Seeds [0,1,2] meet the configured K2 minimum.
    This is not yet the full 5-seed manifest sweep.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from ..decision.k2_decision import k2_decision
from .aggregate import _protocol_family
from .loso_plan import SUBJECTS


def _rd(p):
    b = json.load(open(p)); return b.get("body", b)


def read_c8_fold(loso_root, seed, target) -> dict:
    """Locate + deep-verify one (seed, target) artifact; extract per-level K1 (from decisions/k1.json) and
    the per-method worst-held-out-target metrics (worst_domain_reference_bacc / worst_domain_nll)."""
    from ..artifacts.decision_codec import read_level_decisions
    from ..artifacts.verify import verify_artifact_tree
    adir = os.path.join(loso_root, f"seed-{int(seed)}", f"target-{int(target):03d}", "artifacts")
    cand = sorted(glob.glob(os.path.join(adir, "*", "COMMITTED.json")))
    if len(cand) != 1:
        raise ValueError(f"seed-{seed}/target-{target:03d}: expected exactly one committed artifact, found {len(cand)}")
    artifact = os.path.dirname(cand[0])
    rep = verify_artifact_tree(artifact, deep=True)
    marker = json.load(open(cand[0]))
    manifest = _rd(os.path.join(artifact, "context", "manifest.json")).get("manifest", {})
    levels = {}
    for ld in sorted(glob.glob(os.path.join(artifact, "levels", "level-*"))):
        L = int(ld.rsplit("-", 1)[-1])
        k1 = read_level_decisions(artifact, L)["k1"]
        worst = {}
        for md in glob.glob(os.path.join(ld, "methods", "*")):
            name = os.path.basename(md)
            ta = _rd(os.path.join(md, "metrics.json"))["roles"]["target_audit"]
            worst[name] = {"bacc": ta.get("worst_domain_reference_bacc"), "nll": ta.get("worst_domain_nll")}
        levels[L] = {"k1": k1, "worst": worst}
    target_fit_empty = all(not _rd(os.path.join(ld, "provenance.json")).get("target_fit_ids")
                           for ld in glob.glob(os.path.join(artifact, "levels", "level-*")))
    return {"seed": int(seed), "target": int(target), "artifact_dir": artifact,
            "deep_verification_ok": bool(rep.ok), "target_fit_empty": target_fit_empty,
            "protocol_family": _protocol_family(manifest.get("protocol_id")),
            "provenance_hash": marker.get("provenance_hash"), "context_hash": marker.get("context_hash"),
            "levels": levels}


def _delta(o, e):
    if o is None or e is None:
        return None
    o, e = float(o), float(e)
    return None if (o != o or e != e) else o - e


def aggregate_c8(fold_records, *, seeds, subjects=SUBJECTS, k2_min_seeds=3, k2_margins=None) -> dict:
    """Verify the 27 folds and build the K1 aggregate + the multi-seed K2. The K2 unit per (seed, level) uses
    the WORST held-out target across the 9 LOSO folds (min bAcc / max NLL), Δ = OACI − ERM."""
    seeds = sorted(int(s) for s in seeds)
    want_t = sorted(int(s) for s in subjects)
    by = {(r["seed"], r["target"]): r for r in fold_records}
    n_exp = len(seeds) * len(want_t)
    if len(fold_records) != n_exp or len(by) != n_exp:
        raise ValueError(f"expected {n_exp} unique (seed,target) folds; got {len(by)}")
    for s in seeds:
        for t in want_t:
            if (s, t) not in by:
                raise ValueError(f"missing fold seed-{s}/target-{t:03d}")
    for r in fold_records:
        if not r["deep_verification_ok"]:
            raise ValueError(f"seed-{r['seed']}/target-{r['target']}: deep verification failed")
        if not r["target_fit_empty"]:
            raise ValueError(f"seed-{r['seed']}/target-{r['target']}: target_fit not empty")
    fams = {r["protocol_family"] for r in fold_records}
    provs = {r["provenance_hash"] for r in fold_records}
    if len(fams) != 1 or None in fams:
        raise ValueError(f"folds are not one protocol family: {fams}")
    if len(provs) != 1 or None in provs:
        raise ValueError(f"folds do not share one code provenance: {provs}")

    levels = sorted({L for r in fold_records for L in r["levels"]})
    # ---- K1: per fold/level/seed + aggregate counts ----
    k1_per_fold, k1_counts = [], {}
    for L in levels:
        det = stop = other = 0
        for s in seeds:
            for t in want_t:
                k1 = by[(s, t)]["levels"][L]["k1"]
                st = k1.get("k1_status")
                k1_per_fold.append({"seed": s, "target": t, "level": L, "k1_status": st,
                                    "observed_delta": k1.get("observed_delta"), "p_lower": k1.get("p_lower"),
                                    "permutation_plan_hash": k1.get("permutation_plan_hash")})
                if st == "leakage_reduction_detected":
                    det += 1
                elif st == "stop_no_detectable_heldout_leakage_reduction":
                    stop += 1
                else:
                    other += 1
        k1_counts[L] = {"leakage_reduction_detected": det,
                        "stop_no_detectable_heldout_leakage_reduction": stop, "other": other, "n": det + stop + other}
    # ---- K2: worst held-out target across the 9 LOSO folds, per (seed, level) ----
    margins = k2_margins or {"worst_domain_bacc": 0.0, "worst_domain_nll": 0.0}
    units = []
    for s in seeds:
        for L in levels:
            wl = [by[(s, t)]["levels"][L]["worst"] for t in want_t]
            eb = [w["ERM"]["bacc"] for w in wl if w["ERM"]["bacc"] is not None]
            ob = [w["OACI"]["bacc"] for w in wl if w["OACI"]["bacc"] is not None]
            en = [w["ERM"]["nll"] for w in wl if w["ERM"]["nll"] is not None]
            on = [w["OACI"]["nll"] for w in wl if w["OACI"]["nll"] is not None]
            db = _delta(min(ob) if ob else None, min(eb) if eb else None)       # worst target: min bAcc
            dn = _delta(max(on) if on else None, max(en) if en else None)       # worst target: max NLL
            units.append({"seed": s, "level": L, "deltas": {"worst_domain_bacc": db, "worst_domain_nll": dn}})
    k2 = k2_decision(units, endpoints=["worst_domain_bacc", "worst_domain_nll"], min_seeds=int(k2_min_seeds),
                     level_policy="both_levels", margins=margins)
    return {"n_folds": len(fold_records), "seeds": seeds, "targets": want_t, "levels": levels,
            "protocol_family": next(iter(fams)), "provenance_hash": next(iter(provs)),
            "all_deep_verified": True, "all_target_fit_empty": True,
            "k1_counts": k1_counts, "k1_per_fold": k1_per_fold, "k2": k2, "k2_units": units}


def collect_c8(loso_root, *, seeds, subjects=SUBJECTS) -> list:
    return [read_c8_fold(loso_root, s, t) for s in sorted(int(x) for x in seeds) for t in sorted(int(x) for x in subjects)]


def _f(x, nd=4):
    return "n/a" if x is None else (f"{x:+.{nd}f}" if isinstance(x, (int, float)) else str(x))


def render_c8_report_md(agg) -> str:
    L = [f"# C8 — BNCI2014_001 LOSO seeds {agg['seeds']} (native K1/K2)", "",
         "> BNCI2014-001 minimum-seed K1/K2 run. Seeds [0,1,2] meet the configured K2 minimum. "
         "This is not yet the full 5-seed manifest sweep.", "",
         f"- fold-runs: **{agg['n_folds']}** (9 targets × {len(agg['seeds'])} seeds); all deep-verified: "
         f"**{agg['all_deep_verified']}**; all target_fit ∅: **{agg['all_target_fit_empty']}**",
         f"- protocol_family: `{agg['protocol_family']}` · provenance: `{str(agg['provenance_hash'])[:12]}`", "",
         "## K1 — held-out audit permutation (per fold/level/seed; aggregate counts)"]
    for lvl, c in agg["k1_counts"].items():
        L.append(f"- level {lvl}: **detected {c['leakage_reduction_detected']}/{c['n']}**, "
                 f"stop {c['stop_no_detectable_heldout_leakage_reduction']}/{c['n']}, other {c['other']}/{c['n']}")
    L += ["", "| seed | target | level | K1 | Δ | p_lower |", "|---:|---:|---:|---|---:|---:|"]
    for r in agg["k1_per_fold"]:
        L.append(f"| {r['seed']} | {r['target']} | {r['level']} | {r['k1_status']} | "
                 f"{_f(r['observed_delta'])} | {r.get('p_lower')} |")
    k2 = agg["k2"]
    L += ["", "## K2 — reproducible worst-held-out-target gain across seeds",
          f"- **{k2['k2_status']}** · available_seeds = {k2.get('n_seeds')} · required_min_seeds = {k2.get('min_seeds')}"
          + (f" · reproduced: {k2.get('reproduced_endpoints')}" if k2.get("reproduced_endpoints") else ""), "",
          "| seed | level | Δ worst bAcc | Δ worst NLL |", "|---:|---:|---:|---:|"]
    for u in agg["k2_units"]:
        L.append(f"| {u['seed']} | {u['level']} | {_f(u['deltas']['worst_domain_bacc'])} | "
                 f"{_f(u['deltas']['worst_domain_nll'])} |")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.confirmatory.c8_aggregate")
    ap.add_argument("--loso-root", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args(argv)
    from ..artifacts.canonical_json import canonical_json_bytes
    seeds = [int(s) for s in args.seeds.split(",")]
    agg = aggregate_c8(collect_c8(args.loso_root, seeds=seeds), seeds=seeds)
    for p in (args.out_json, args.out_md):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(args.out_json, "wb") as f:
        f.write(canonical_json_bytes(agg))
    with open(args.out_md, "w") as f:
        f.write(render_c8_report_md(agg))
    print(f"wrote {args.out_json} + {args.out_md}: {agg['n_folds']} folds, K2 {agg['k2']['k2_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
