"""BNCI2014_001 LOSO seed-0 aggregation (C6) -- verifies all nine folds and builds the k1/k2-style
endpoint aggregates. The aggregation LOGIC (checks + k1/k2) operates on per-fold result dicts and is order
invariant; read_fold_artifact extracts those dicts from a deep-verified artifact.

    BNCI2014-001 LOSO seed-0 full-bootstrap staged run.
    This is not the final multi-seed, multi-dataset confirmatory efficacy result.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics

from .loso_plan import SUBJECTS

_METHODS = ("ERM", "OACI", "global_lpc", "uniform")


def _protocol_family(protocol_id) -> str:
    """The cross-fold protocol identity: the materialized protocol_id minus the per-target suffix
    (`...-BNCI2014_001-target002` -> `...-BNCI2014_001`). SHARED across the nine LOSO folds (which differ
    only in the held-out target), unlike the per-fold manifest/context hashes."""
    return re.sub(r"-target\d+$", "", str(protocol_id))


def _q(xs, name):
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return {f"{name}_n": 0}
    return {f"{name}_mean": statistics.fmean(xs), f"{name}_median": statistics.median(xs),
            f"{name}_min": min(xs), f"{name}_max": max(xs), f"{name}_n": len(xs)}


def _delta(level, key, method="OACI", base="ERM"):
    m = {x["method"]: x for x in level["methods"]}
    a, b = m.get(method, {}).get(key), m.get(base, {}).get(key)
    return None if a is None or b is None else float(a) - float(b)


def aggregate_loso(fold_results, *, subjects=SUBJECTS, protocol_family=None, provenance_hash=None) -> dict:
    """Verify the nine folds and compute the descriptive k1/k2 aggregates. ORDER INVARIANT (folds are
    sorted by target). Raises ValueError on any structural failure.

    A coherent sweep shares ONE protocol family (protocol_id minus the per-target tag) and ONE code
    provenance (same commit). The per-fold manifest / context / artifact hashes legitimately DIFFER (each
    fold holds out a different target), so the cross-fold identity is (protocol_family, provenance_hash) --
    NOT the per-fold context_hash. Optional protocol_family / provenance_hash pin the expected values."""
    folds = sorted(fold_results, key=lambda f: int(f["target"]))
    targets = [int(f["target"]) for f in folds]
    want = sorted(int(s) for s in subjects)
    if targets != want:
        raise ValueError(f"LOSO aggregation expects targets {want} exactly once; got {targets}")
    fams = {f.get("protocol_family") for f in folds}
    if len(fams) != 1 or None in fams:
        raise ValueError(f"folds are not one coherent protocol family: {fams}")
    if protocol_family is not None and fams != {protocol_family}:
        raise ValueError(f"folds are not protocol family {protocol_family!r}: {fams}")
    provs = {f.get("provenance_hash") for f in folds}
    if len(provs) != 1 or None in provs:
        raise ValueError(f"folds do not share one code provenance (mixed commits?): {provs}")
    if provenance_hash is not None and provs != {provenance_hash}:
        raise ValueError(f"folds are not code provenance {provenance_hash!r}: {provs}")
    for f in folds:
        if not f.get("deep_verification_ok"):
            raise ValueError(f"target {f['target']}: deep verification did not pass")
        if not f.get("target_fit_empty"):
            raise ValueError(f"target {f['target']}: target_fit_ids is not empty")
        if sorted(f.get("methods_present", [])) != sorted(_METHODS):
            raise ValueError(f"target {f['target']}: methods != {_METHODS} (got {f.get('methods_present')})")

    levels = sorted({l["level"] for f in folds for l in f["levels"]})
    k1, k2 = [], []
    for lvl in levels:
        dlk, dba, dnll, dece = [], [], [], []
        per_fold = []
        for f in folds:
            L = next(l for l in f["levels"] if l["level"] == lvl)
            d_leak = _delta(L, "audit_ucl")
            d_b = _delta(L, "target_bacc"); d_n = _delta(L, "target_nll"); d_e = _delta(L, "target_ece")
            per_fold.append({"target": int(f["target"]), "delta_audit_ucl": d_leak,
                             "delta_target_bacc": d_b, "delta_target_nll": d_n, "delta_target_ece": d_e})
            for acc, v in ((dlk, d_leak), (dba, d_b), (dnll, d_n), (dece, d_e)):
                if v is not None:
                    acc.append(v)
        k1.append({"level": lvl, "statistic": "audit_bootstrap_ucl(OACI) - (ERM)", "per_fold": per_fold,
                   **_q(dlk, "delta_leakage_ucl"), "n_folds_delta_negative": sum(1 for x in dlk if x < 0)})
        k2.append({"level": lvl,
                   "delta_target_bacc": _q(dba, "d"), "n_bacc_improved": sum(1 for x in dba if x > 0),
                   "delta_target_nll": _q(dnll, "d"), "n_nll_improved": sum(1 for x in dnll if x < 0),
                   "delta_target_ece": _q(dece, "d"), "n_ece_improved": sum(1 for x in dece if x < 0)})
    return {"n_folds": len(folds), "targets": targets,
            "protocol_family": next(iter(fams)), "provenance_hash": next(iter(provs)),
            "per_fold_context_hashes": {f"target-{int(f['target']):03d}": f.get("context_hash") for f in folds},
            "all_deep_verified": True, "all_target_fit_empty": True, "levels": levels,
            "k1_descriptive": k1, "k2_descriptive": k2, "per_fold": folds}


# ---- artifact reader (real data; exercised by the run, not CI) ----
def _rd(p):
    b = json.load(open(p)); return b.get("body", b)


def read_fold_artifact(artifact_dir, target) -> dict:
    """Extract a per-fold result dict (deep-verify status + endpoints) from a committed artifact."""
    from ..artifacts.verify import verify_artifact_tree
    rep = verify_artifact_tree(artifact_dir, deep=True)
    levels = []
    for ld in sorted(glob.glob(os.path.join(artifact_dir, "levels/level-*"))):
        L = int(ld.rsplit("-", 1)[-1])
        pay = _rd(os.path.join(ld, "level.json"))["payload"]
        prov = _rd(os.path.join(ld, "provenance.json"))
        methods = []
        for md in sorted(glob.glob(os.path.join(ld, "methods/*"))):
            name = os.path.basename(md)
            sel = _rd(os.path.join(md, "method.json"))["selection"]

            def _leak(kind):
                p = os.path.join(md, kind + "_leakage.json")
                return _rd(p) if os.path.exists(p) else {}
            mt = _rd(os.path.join(md, "metrics.json"))["roles"]
            methods.append({"method": name, "selected_checkpoint": sel["model_hash"],
                            "selected_risk": sel["R_src"], "selected_epoch": sel["selected_epoch"],
                            "selection_ucl": _leak("selection").get("bootstrap_ucl"),
                            "selection_lq": _leak("selection").get("extractable_LQ_ov"),
                            "audit_ucl": _leak("audit").get("bootstrap_ucl"),
                            "audit_lq": _leak("audit").get("extractable_LQ_ov"),
                            "source_audit_bacc": mt["source_audit"].get("pooled_reference_bacc"),
                            "source_audit_nll": mt["source_audit"].get("pooled_nll"),
                            "source_audit_ece": mt["source_audit"].get("pooled_ece"),
                            "target_bacc": mt["target_audit"].get("pooled_reference_bacc"),
                            "target_nll": mt["target_audit"].get("pooled_nll"),
                            "target_ece": mt["target_audit"].get("pooled_ece")})
        levels.append({"level": L, "R_ERM_hat": pay["erm"]["R_ERM_hat"], "tau": pay["erm"]["tau"],
                       "erm_checkpoint": pay["erm"]["checkpoint"], "methods": methods})
    marker = json.load(open(os.path.join(artifact_dir, "COMMITTED.json")))
    manifest = _rd(os.path.join(artifact_dir, "context", "manifest.json")).get("manifest", {})
    return {"target": int(target), "artifact_dir": artifact_dir,
            "deep_verification_ok": bool(rep.ok),
            "target_fit_empty": all(not _rd(os.path.join(ld, "provenance.json")).get("target_fit_ids")
                                    for ld in glob.glob(os.path.join(artifact_dir, "levels/level-*"))),
            "protocol_id": manifest.get("protocol_id"),
            "protocol_family": _protocol_family(manifest.get("protocol_id")),   # SHARED across folds
            "provenance_hash": marker.get("provenance_hash"),                   # SHARED (same commit)
            "context_hash": marker.get("context_hash"),                        # per-fold (differs), for the record
            "artifact_scientific_hash": marker.get("artifact_scientific_hash"),
            "artifact_pure_science_hash": marker.get("artifact_pure_science_hash"),
            "methods_present": sorted({m["method"] for l in levels for m in l["methods"]}), "levels": levels}


def collect_fold_artifacts(loso_root, *, subjects=SUBJECTS) -> list:
    """Locate + read each target's committed artifact under loso_root/target-00N/artifacts/<hash>/. Requires
    EXACTLY one committed artifact per target (raises otherwise -- a missing/duplicate fold is caught here,
    before aggregation)."""
    out = []
    for t in sorted(int(s) for s in subjects):
        adir = os.path.join(loso_root, f"target-{t:03d}", "artifacts")
        cand = sorted(glob.glob(os.path.join(adir, "*", "COMMITTED.json")))
        if len(cand) != 1:
            raise ValueError(f"target-{t:03d}: expected exactly one committed artifact under {adir}, found {len(cand)}")
        out.append(read_fold_artifact(os.path.dirname(cand[0]), t))
    return out


def _f(x, nd=4) -> str:
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def render_report_md(agg, *, title="C6 — BNCI2014_001 LOSO seed-0") -> str:
    """Human-readable k1/k2 report from an aggregate_loso() result."""
    L = [f"# {title}", "",
         "> BNCI2014-001 LOSO seed-0 full-bootstrap staged run. "
         "This is not the final multi-seed, multi-dataset confirmatory efficacy result.", "",
         f"- folds: **{agg['n_folds']}** (targets {agg['targets']})",
         f"- protocol_family: `{agg['protocol_family']}`",
         f"- provenance_hash: `{agg['provenance_hash']}`",
         f"- all_deep_verified: **{agg['all_deep_verified']}**; all_target_fit_empty: **{agg['all_target_fit_empty']}**",
         "- per-fold context hashes (distinct per fold): "
         + ", ".join(f"{t}:`{(h or 'n/a')[:8]}`" for t, h in sorted(agg["per_fold_context_hashes"].items())),
         "",
         "## k1 — leakage UCL: Δ = audit_ucl(OACI) − audit_ucl(ERM)  (lower ⇒ OACI leaks less)"]
    for k in agg["k1_descriptive"]:
        L += [f"### level {k['level']}",
              f"- mean {_f(k.get('delta_leakage_ucl_mean'))} · median {_f(k.get('delta_leakage_ucl_median'))} · "
              f"min {_f(k.get('delta_leakage_ucl_min'))} · max {_f(k.get('delta_leakage_ucl_max'))} · "
              f"n {k.get('delta_leakage_ucl_n')} · folds Δ<0: **{k['n_folds_delta_negative']}/{k.get('delta_leakage_ucl_n', 0)}**",
              "", "| target | Δ audit_ucl |", "|---:|---:|"]
        L += [f"| {pf['target']} | {_f(pf['delta_audit_ucl'])} |" for pf in k["per_fold"]]
        L += [""]
    L += ["## k2 — target metrics: Δ = OACI − ERM  (bAcc ↑ · NLL ↓ · ECE ↓ better)"]
    for k in agg["k2_descriptive"]:
        L += [f"### level {k['level']}"]
        for name, key, imp, arrow in (("bAcc", "delta_target_bacc", "n_bacc_improved", "↑"),
                                      ("NLL", "delta_target_nll", "n_nll_improved", "↓"),
                                      ("ECE", "delta_target_ece", "n_ece_improved", "↓")):
            q = k[key]
            L += [f"- Δ{name} ({arrow}): mean {_f(q.get('d_mean'))} · median {_f(q.get('d_median'))} · "
                  f"min {_f(q.get('d_min'))} · max {_f(q.get('d_max'))} · improved **{k[imp]}/{q.get('d_n', 0)}**"]
        L += [""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.confirmatory.aggregate")
    ap.add_argument("--loso-root", required=True, help="dir holding target-00N/artifacts/<hash>/ for all 9 folds")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--protocol-family", default=None, help="pin the expected protocol family (optional)")
    ap.add_argument("--provenance-hash", default=None, help="pin the expected code provenance (optional)")
    args = ap.parse_args(argv)
    from ..artifacts.canonical_json import canonical_json_bytes
    folds = collect_fold_artifacts(args.loso_root)
    agg = aggregate_loso(folds, protocol_family=args.protocol_family, provenance_hash=args.provenance_hash)
    for p in (args.out_json, args.out_md):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(args.out_json, "wb") as f:
        f.write(canonical_json_bytes(agg))
    with open(args.out_md, "w") as f:
        f.write(render_report_md(agg))
    print(f"wrote {args.out_json} + {args.out_md}: {agg['n_folds']} folds, "
          f"family {agg['protocol_family']}, provenance {agg['provenance_hash'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
