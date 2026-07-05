"""BNCI2014_001 LOSO seeds-[0,1,2] MULTI-SEED aggregation (C8): read all 27 committed fold artifacts
(9 targets × 3 seeds), deep-verify each, and report the native K1 (per fold/level/seed + aggregate counts,
descriptive stats, and a multiplicity-corrected sweep SUMMARY) and the real multi-seed K2 (worst-held-out
target endpoints across the 3 seeds → reproducible_gain / stop_no_reproducible_gain).

    BNCI2014-001 minimum-seed K1/K2 run. Seeds [0,1,2] meet the configured K2 minimum.
    This is not yet the full 5-seed manifest sweep.

Provenance: this run's Phase-B spanned TWO commits (execution-only: staged_b.sh 16->32 CPU + controller
--leakage-jobs flag). Parallelism is not in any scientific hash, so fold science is bit-identical across the
split. The one-provenance guard is therefore replaced by a NARROW approved-provenance rule that accepts <=2
provenance groups ONLY when every scientific-identity invariant matches, and records the transition.

Decision hierarchy (pre-registered / C7): K1 is a PER-FOLD gate (p_lower<alpha); the frozen SWEEP go/no-go is
K2 (both_levels, min_seeds=3). No sweep-level K1 verdict is pre-registered, so the K1 sweep line here is a
DESCRIPTIVE, multiplicity-corrected (Bonferroni + Benjamini-Hochberg) summary — never an uncorrected count.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from ..decision.k1_decision import K1_DETECTED, K1_STOP
from ..decision.k2_decision import k2_decision
from .aggregate import _protocol_family
from .loso_plan import SUBJECTS

# manifest-frozen K1/K2 config (oaci/protocol/confirmatory_v2.yaml) — asserted constant across folds.
_K1_ALPHA = 0.05
_K1_NPERM = 2000
_K1_STAT = "grouped_max_probe_extractable_LQ_ov_OACI_minus_ERM"
_K1_SPLIT = "source_audit"
_K2_ENDPOINTS = ("worst_domain_bacc", "worst_domain_nll")
_K2_MIN_SEEDS = 3
_K2_LEVEL_POLICY = "both_levels"


def _rd(p):
    b = json.load(open(p)); return b.get("body", b)


def _doc(p):
    """Full decision doc (wrapper): keeps schema_version + body."""
    return json.load(open(p))


def read_c8_fold(loso_root, seed, target) -> dict:
    """Locate + deep-verify one (seed, target) artifact; extract the rich per-level K1 payload (from
    decisions/k1.json) and the per-method worst-held-out-target metrics. Asserts the 3 decision files
    (k1.json, k1.npz, k2.json) exist at every level."""
    from ..artifacts.decision_codec import has_level_decisions, read_level_decisions
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
    dec_schema = set()
    for ld in sorted(glob.glob(os.path.join(artifact, "levels", "level-*"))):
        L = int(ld.rsplit("-", 1)[-1])
        if not has_level_decisions(artifact, L):
            raise ValueError(f"seed-{seed}/target-{target:03d} level {L}: missing k1.json/k1.npz/k2.json")
        dec = read_level_decisions(artifact, L)
        k1 = dec["k1"]; k2 = dec["k2"]
        npz = k1.get("npz", {})
        k1c = {"k1_status": k1.get("k1_status"), "observed_delta": k1.get("observed_delta"),
               "p_lower": k1.get("p_lower"), "p_two_sided": k1.get("p_two_sided"),
               "alpha": k1.get("alpha"), "n_permutations": k1.get("n_permutations"),
               "statistic": k1.get("statistic"), "split_role": k1.get("split_role"),
               "permutation_plan_hash": k1.get("permutation_plan_hash"),
               "audit_support_hash": k1.get("audit_support_hash"),
               "audit_population_hash": k1.get("audit_population_hash"),
               "probe_config_hash": k1.get("probe_config_hash"),
               "null_content_hash": npz.get("null", {}).get("array_content_hash"),
               "observed_delta_content_hash": npz.get("observed_delta", {}).get("array_content_hash")}
        k2c = {"min_seeds": k2.get("min_seeds"), "level_policy": k2.get("level_policy"),
               "endpoints": tuple(k2.get("endpoints") or ())}
        worst = {}
        for md in glob.glob(os.path.join(ld, "methods", "*")):
            name = os.path.basename(md)
            ta = _rd(os.path.join(md, "metrics.json"))["roles"]["target_audit"]
            worst[name] = {"bacc": ta.get("worst_domain_reference_bacc"), "nll": ta.get("worst_domain_nll")}
        levels[L] = {"k1": k1c, "k2cfg": k2c, "worst": worst}
        dec_schema.add(_doc(os.path.join(ld, "decisions", "k1.json")).get("schema_version"))
    target_fit_empty = all(not _rd(os.path.join(ld, "provenance.json")).get("target_fit_ids")
                           for ld in glob.glob(os.path.join(artifact, "levels", "level-*")))
    return {"seed": int(seed), "target": int(target), "artifact_dir": artifact,
            "deep_verification_ok": bool(rep.ok), "target_fit_empty": target_fit_empty,
            "protocol_family": _protocol_family(manifest.get("protocol_id")),
            "provenance_hash": marker.get("provenance_hash"), "context_hash": marker.get("context_hash"),
            "artifact_schema_version": marker.get("schema_version"),
            "decision_schema_version": (next(iter(dec_schema)) if len(dec_schema) == 1 else f"MIXED:{sorted(dec_schema)}"),
            "artifact_scientific_hash": marker.get("artifact_scientific_hash"),
            "artifact_pure_science_hash": marker.get("artifact_pure_science_hash"), "levels": levels}


def _delta(o, e):
    if o is None or e is None:
        return None
    o, e = float(o), float(e)
    return None if (o != o or e != e) else o - e


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _median(xs):
    xs = sorted(float(x) for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs); m = n // 2
    return xs[m] if n % 2 else 0.5 * (xs[m - 1] + xs[m])


def _bh_reject(pvals, q):
    """Benjamini-Hochberg: return a boolean list (same order as pvals) of which are rejected at FDR q."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    n = len(pvals); kmax = -1
    for rank, i in enumerate(idx, start=1):
        if pvals[i] <= q * rank / n:
            kmax = rank
    rej = [False] * n
    if kmax >= 1:
        for rank, i in enumerate(idx, start=1):
            if rank <= kmax:
                rej[i] = True
    return rej


def _assert_execution_only_split(fold_records, provs):
    """The NARROW approved-provenance rule. Accept <=2 provenance groups ONLY when every scientific-identity
    invariant matches; >2 groups or any mismatch -> raise. Returns the transition record."""
    if None in provs:
        raise ValueError("a fold is missing provenance_hash")
    if len(provs) > 2:
        raise ValueError(f"more than two provenance groups ({len(provs)}) — not an execution-only split: {provs}")
    # scientific-identity invariants that make a 2-commit split safe
    art_schema = {r["artifact_schema_version"] for r in fold_records}
    dec_schema = {r["decision_schema_version"] for r in fold_records}
    if len(art_schema) != 1:
        raise ValueError(f"artifact schema not identical: {art_schema}")
    if len(dec_schema) != 1 or any(str(s).startswith("MIXED") for s in dec_schema):
        raise ValueError(f"decision schema not identical: {dec_schema}")
    k1_cfg, k2_cfg, probe = set(), set(), set()
    for r in fold_records:
        for L, lv in r["levels"].items():
            k1 = lv["k1"]
            k1_cfg.add((k1["alpha"], k1["n_permutations"], k1["statistic"], k1["split_role"]))
            probe.add(k1["probe_config_hash"])
            k2_cfg.add((lv["k2cfg"]["min_seeds"], lv["k2cfg"]["level_policy"], lv["k2cfg"]["endpoints"]))
    if k1_cfg != {(_K1_ALPHA, _K1_NPERM, _K1_STAT, _K1_SPLIT)}:
        raise ValueError(f"K1 config not the manifest-frozen constant across folds: {k1_cfg}")
    if len(probe) != 1:
        raise ValueError(f"probe_config_hash not identical across folds: {probe}")
    if k2_cfg != {(_K2_MIN_SEEDS, _K2_LEVEL_POLICY, tuple(_K2_ENDPOINTS))}:
        raise ValueError(f"K2 config not the manifest-frozen constant across folds: {k2_cfg}")
    affected = {}
    for p in provs:
        affected[p] = sum(1 for r in fold_records if r["provenance_hash"] == p)
    return {"accepted": True, "n_groups": len(provs), "provenance_hashes": sorted(provs),
            "affected_folds": affected, "probe_config_hash": next(iter(probe)),
            "artifact_schema_version": next(iter(art_schema)), "decision_schema_version": next(iter(dec_schema)),
            "reason": "execution-only two-commit split during live sweep (staged_b.sh cpus 16->32 + "
                      "controller --leakage-jobs); parallelism is in no scientific hash",
            "science_hash_policy": "artifact pure-science identities remain authoritative; execution "
                                   "parallelism / git tree are not inputs to any fold science hash"}


def aggregate_c8(fold_records, *, seeds, subjects=SUBJECTS, k2_min_seeds=3, k2_margins=None,
                 approved_provenance=None, transition_commits=None) -> dict:
    """Verify the 27 folds and build the K1 aggregate (+multiplicity-corrected sweep summary) and the
    multi-seed K2. K2 unit per (seed, level) uses the WORST held-out target across the 9 LOSO folds."""
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
    if len(fams) != 1 or None in fams:
        raise ValueError(f"folds are not one protocol family: {fams}")
    provs = {r["provenance_hash"] for r in fold_records}
    transition = _assert_execution_only_split(fold_records, provs)
    if approved_provenance is not None:
        allow = {p.strip() for p in approved_provenance if p and p.strip()}
        if provs != allow:
            raise ValueError(f"observed provenance {sorted(provs)} != approved allowlist {sorted(allow)}")
        transition["approved_allowlist"] = sorted(allow)
    if transition_commits:
        transition["commits"] = list(transition_commits)

    levels = sorted({L for r in fold_records for L in r["levels"]})
    # ---- K1: per fold/level/seed + counts + descriptive stats ----
    k1_per_fold, k1_counts, k1_stats = [], {}, {}
    all_p, all_delta = [], []
    for L in levels:
        det = stop = other = 0
        deltas = []
        for s in seeds:
            for t in want_t:
                k1 = by[(s, t)]["levels"][L]["k1"]
                st = k1.get("k1_status")
                k1_per_fold.append({"seed": s, "target": t, "level": L, "k1_status": st,
                                    "observed_delta": k1.get("observed_delta"), "p_lower": k1.get("p_lower"),
                                    "p_two_sided": k1.get("p_two_sided"),
                                    "permutation_plan_hash": k1.get("permutation_plan_hash"),
                                    "audit_support_hash": k1.get("audit_support_hash"),
                                    "audit_population_hash": k1.get("audit_population_hash"),
                                    "probe_config_hash": k1.get("probe_config_hash"),
                                    "null_content_hash": k1.get("null_content_hash")})
                deltas.append(k1.get("observed_delta"))
                all_delta.append(k1.get("observed_delta"))
                if k1.get("p_lower") is not None:
                    all_p.append(float(k1["p_lower"]))
                if st == K1_DETECTED:
                    det += 1
                elif st == K1_STOP:
                    stop += 1
                else:
                    other += 1
        n = det + stop + other
        k1_counts[L] = {"leakage_reduction_detected": det, "stop_no_detectable_heldout_leakage_reduction": stop,
                        "other": other, "n": n, "fraction_detected": (det / n if n else None)}
        dnn = [d for d in deltas if d is not None]
        k1_stats[L] = {"mean": _mean(dnn), "median": _median(dnn),
                       "min": (min(dnn) if dnn else None), "max": (max(dnn) if dnn else None)}
    ndet = sum(c["leakage_reduction_detected"] for c in k1_counts.values())
    ntot = sum(c["n"] for c in k1_counts.values())
    dnn_all = [d for d in all_delta if d is not None]
    # multiplicity-corrected SWEEP summary (descriptive; NOT a frozen threshold — frozen sweep = K2)
    bonf = [p < (_K1_ALPHA / len(all_p)) for p in all_p] if all_p else []
    bh = _bh_reject(all_p, _K1_ALPHA) if all_p else []
    n_bonf, n_bh = sum(bonf), sum(bh)
    k1_sweep_status = K1_DETECTED if n_bh >= 1 else K1_STOP
    k1_overall = {"n_tests": ntot, "n_leakage_reduction_detected": ndet,
                  "n_stop_no_detectable_heldout_leakage_reduction": sum(
                      c["stop_no_detectable_heldout_leakage_reduction"] for c in k1_counts.values()),
                  "fraction_detected_uncorrected": (ndet / ntot if ntot else None),
                  "observed_delta_mean": _mean(dnn_all), "observed_delta_median": _median(dnn_all),
                  "observed_delta_min": (min(dnn_all) if dnn_all else None),
                  "observed_delta_max": (max(dnn_all) if dnn_all else None),
                  "multiplicity": {"alpha": _K1_ALPHA, "n_tests": len(all_p),
                                   "bonferroni_threshold": (_K1_ALPHA / len(all_p) if all_p else None),
                                   "n_bonferroni_survive": n_bonf, "n_bh_survive": n_bh,
                                   "bh_survivors": [k1_per_fold[i] for i in range(len(k1_per_fold))
                                                    if i < len(bh) and bh[i]]},
                  "k1_sweep_status": k1_sweep_status,
                  "note": "K1 is pre-registered PER-FOLD; no sweep-level K1 threshold is frozen. This sweep "
                          "line is DESCRIPTIVE (multiplicity-corrected via BH-FDR), not a frozen go/no-go. "
                          "The frozen sweep decision is K2."}
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
    k2 = k2_decision(units, endpoints=list(_K2_ENDPOINTS), min_seeds=int(k2_min_seeds),
                     level_policy=_K2_LEVEL_POLICY, margins=margins)
    # K2 descriptive aggregate per endpoint (across the 6 units)
    k2_agg = {}
    for e in _K2_ENDPOINTS:
        ds = [u["deltas"][e] for u in units if u["deltas"][e] is not None]
        higher = e == "worst_domain_bacc"
        gain = [d for d in ds if (d > margins[e]) if higher] + [d for d in ds if (d < -margins[e]) if not higher]
        worst_fold = (min(ds) if higher else max(ds)) if ds else None      # least favorable unit
        k2_agg[e] = {"mean": _mean(ds), "median": _median(ds), "worst_fold": worst_fold,
                     "n_improved": len(gain), "n_harmed": len(ds) - len(gain), "n_units": len(ds)}

    return {"n_folds": len(fold_records), "seeds": seeds, "targets": want_t, "levels": levels,
            "protocol_family": next(iter(fams)), "provenance_transition": transition,
            "all_deep_verified": True, "all_target_fit_empty": True,
            "k1_counts": k1_counts, "k1_stats": k1_stats, "k1_per_fold": k1_per_fold, "k1_overall": k1_overall,
            "k2": k2, "k2_units": units, "k2_agg": k2_agg}


def collect_c8(loso_root, *, seeds, subjects=SUBJECTS) -> list:
    return [read_c8_fold(loso_root, s, t) for s in sorted(int(x) for x in seeds) for t in sorted(int(x) for x in subjects)]


def _f(x, nd=4):
    return "n/a" if x is None else (f"{x:+.{nd}f}" if isinstance(x, (int, float)) else str(x))


def render_c8_report_md(agg) -> str:
    tr = agg["provenance_transition"]; ko = agg["k1_overall"]; k2 = agg["k2"]
    k1_stop = ko["k1_sweep_status"] == K1_STOP
    k2_stop = k2["k2_status"] != "reproducible_gain"
    L = [f"# C8 — BNCI2014_001 LOSO seeds {agg['seeds']} (native K1/K2)", "",
         "> **BNCI2014-001 minimum-seed K1/K2 run. Seeds [0,1,2] meet the configured K2 minimum. "
         "This is not yet the full 5-seed manifest sweep.**", "",
         "## run", "",
         f"- dataset: **BNCI2014-001** · targets: **9** · seeds: **{agg['seeds']}** · fold-runs: "
         f"**{agg['n_folds']}** · levels: **{agg['levels']}**",
         "- methods: `ERM, OACI, global_lpc, uniform` · bootstrap: **full** · K1 permutations: **2000** · "
         "staged execution: **true**", "",
         "## verification", "",
         f"- artifacts_complete: **{agg['n_folds']}/{agg['n_folds']}** · deep_verified: "
         f"**{agg['all_deep_verified']}** · target_fit_ids_empty: **{agg['all_target_fit_empty']}**",
         f"- decision_payloads_present: **{agg['n_folds']}/{agg['n_folds']} × {len(agg['levels'])} levels** "
         "(k1.json + k1.npz + k2.json each)",
         f"- provenance_groups: **{tr['n_groups']}** · approved_provenance_exception: **{tr['accepted']}**", "",
         "## K1 — held-out audit permutation null (pre-registered PER-FOLD; sweep line is descriptive)", ""]
    for lvl, c in agg["k1_counts"].items():
        s = agg["k1_stats"][lvl]
        L.append(f"- level {lvl}: detected **{c['leakage_reduction_detected']}/{c['n']}**, stop "
                 f"{c['stop_no_detectable_heldout_leakage_reduction']}/{c['n']}, other {c['other']} · "
                 f"Δ mean {_f(s['mean'])}, median {_f(s['median'])}, min {_f(s['min'])}, max {_f(s['max'])}")
    m = ko["multiplicity"]
    L += ["",
          f"- **overall**: n_tests {ko['n_tests']}, detected (uncorrected) {ko['n_leakage_reduction_detected']}, "
          f"stop {ko['n_stop_no_detectable_heldout_leakage_reduction']}, "
          f"fraction_detected {_f(ko['fraction_detected_uncorrected'], 3)}",
          f"- Δ overall: mean {_f(ko['observed_delta_mean'])}, median {_f(ko['observed_delta_median'])}, "
          f"min {_f(ko['observed_delta_min'])}, max {_f(ko['observed_delta_max'])}",
          f"- **multiplicity control** (α={m['alpha']}, {m['n_tests']} one-sided tests): "
          f"Bonferroni survivors **{m['n_bonferroni_survive']}** (thr {_f(m['bonferroni_threshold'], 5)}), "
          f"BH-FDR survivors **{m['n_bh_survive']}**",
          f"- **K1 sweep status (descriptive): `{ko['k1_sweep_status']}`** — {ko['note']}", "",
          "| seed | target | level | K1 | Δ (OACI−ERM) | p_lower | p_two |", "|---:|---:|---:|---|---:|---:|---:|"]
    for r in agg["k1_per_fold"]:
        L.append(f"| {r['seed']} | {r['target']} | {r['level']} | {r['k1_status']} | "
                 f"{_f(r['observed_delta'])} | {r.get('p_lower')} | {r.get('p_two_sided')} |")
    L += ["", "## K2 — reproducible worst-held-out-target gain across seeds (FROZEN sweep go/no-go)", "",
          f"- **`{k2['k2_status']}`** · available_seeds = {k2.get('n_seeds')} · required_min_seeds = "
          f"{k2.get('min_seeds')} · level_policy = {k2.get('level_policy')}"
          + (f" · reproduced: {k2.get('reproduced_endpoints')}" if k2.get("reproduced_endpoints") else ""), ""]
    for e, a in agg["k2_agg"].items():
        L.append(f"- {e}: Δ mean {_f(a['mean'])}, median {_f(a['median'])}, worst-fold {_f(a['worst_fold'])} · "
                 f"improved {a['n_improved']}/{a['n_units']}, harmed {a['n_harmed']}/{a['n_units']}")
    L += ["", "| seed | level | Δ worst bAcc | Δ worst NLL |", "|---:|---:|---:|---:|"]
    for u in agg["k2_units"]:
        L.append(f"| {u['seed']} | {u['level']} | {_f(u['deltas']['worst_domain_bacc'])} | "
                 f"{_f(u['deltas']['worst_domain_nll'])} |")
    L += ["", "## provenance transition", "",
          f"- accepted: **{tr['accepted']}** ({tr['n_groups']} groups) — {tr['reason']}",
          f"- provenance_hashes: {[h[:12] for h in tr['provenance_hashes']]}"
          + (f" · commits: {tr.get('commits')}" if tr.get("commits") else ""),
          f"- affected_folds: {{{', '.join(f'{h[:12]}: {n}' for h, n in tr['affected_folds'].items())}}}",
          f"- probe_config_hash (constant): `{tr['probe_config_hash'][:12]}` · artifact schema "
          f"`{tr['artifact_schema_version']}` · decision schema `{tr['decision_schema_version']}`",
          f"- science-hash policy: {tr['science_hash_policy']}", "",
          "## decision hierarchy & verdict", "",
          f"- **K1 (per-fold gate; multiplicity-corrected sweep summary): `{ko['k1_sweep_status']}`**",
          f"- **K2 (frozen sweep go/no-go): `{k2['k2_status']}`**", ""]
    if k1_stop or k2_stop:
        L += ["> **VERDICT: pause.** K1 shows no multiplicity-surviving held-out leakage reduction and/or K2 "
              "shows no reproducible gain. Per pre-registration: do NOT run seeds [3,4]; do NOT add "
              "BNCI2014_004. Write this up as a negative BNCI001 minimum-seed result."]
    else:
        L += ["> **VERDICT: K1 sweep detected AND K2 not-stop** — extend to seeds [3,4] under the same driver."]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.confirmatory.c8_aggregate")
    ap.add_argument("--loso-root", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--approved-provenance", default=None,
                    help="optional comma-separated provenance_hash allowlist (max narrowness)")
    ap.add_argument("--transition-commits", default=None, help="comma-separated commit refs for the record")
    args = ap.parse_args(argv)
    from ..artifacts.canonical_json import canonical_json_bytes
    seeds = [int(s) for s in args.seeds.split(",")]
    allow = args.approved_provenance.split(",") if args.approved_provenance else None
    commits = args.transition_commits.split(",") if args.transition_commits else None
    agg = aggregate_c8(collect_c8(args.loso_root, seeds=seeds), seeds=seeds,
                       approved_provenance=allow, transition_commits=commits)
    for p in (args.out_json, args.out_md):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(args.out_json, "wb") as f:
        f.write(canonical_json_bytes(agg))
    with open(args.out_md, "w") as f:
        f.write(render_c8_report_md(agg))
    print(f"wrote {args.out_json} + {args.out_md}: {agg['n_folds']} folds, "
          f"K1 sweep {agg['k1_overall']['k1_sweep_status']}, K2 {agg['k2']['k2_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
