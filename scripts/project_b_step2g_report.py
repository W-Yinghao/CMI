"""Project B Step-2G: freeze the Step-2E/2F synthetic router evidence into paper-style tables.

READ-ONLY over the locked Step-2E/2F outputs. Trains nothing, reruns nothing, and does NOT touch
any h2cmi module. Emits ablation tables, a reason-code audit, representative limitation examples,
LaTeX table fragments, a markdown report, and a machine-readable claim-boundary file.

Fails loudly on structural problems (missing inputs / worlds / modes, an OFFLINE_TTA selected under
degenerate ACAR-harm, or R2 nested coverage not exceeding baseline). Does NOT fail on the known
scientific limitations (R2 missed benefit, HF3 concept-degraded identity, H_OOD low-ESS subset).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

WORLDS = ["R2", "HF3", "H_OOD"]
PRIMARY_MODES = ["in_source_subject_q95", "nested_site_excess_q95"]
DEGENERATE = ("degenerate", "unavailable")


def _load_reports(d):
    out = []
    for fp in sorted(glob.glob(os.path.join(d, "*_router_report.json"))):
        out.append(json.load(open(fp)))
    return out


def _mean(xs):
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _write_csv(path, cols, rows):
    def fmt(v):
        if isinstance(v, float):
            return "nan" if math.isnan(v) else f"{v:.6g}"
        if isinstance(v, (list, tuple)):
            return "|".join(str(x) for x in v)
        if isinstance(v, dict):
            return ";".join(f"{k}:{v[k]}" for k in v)
        return "" if v is None else str(v)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([fmt(r.get(c)) for c in cols])


# --------------------------------------------------------------------------- validation
def validate(step2f_dir, reports2f):
    for fn in ("world_summary.csv", "per_domain_decisions.csv"):
        if not os.path.exists(os.path.join(step2f_dir, fn)):
            raise SystemExit(f"[FAIL] missing Step-2F {fn}")
    have_worlds = {r["world"] for r in reports2f}
    if not set(WORLDS) <= have_worlds:
        raise SystemExit(f"[FAIL] missing worlds: {set(WORLDS) - have_worlds}")
    have_modes = {r["support_mode"] for r in reports2f}
    if not set(PRIMARY_MODES) <= have_modes:
        raise SystemExit(f"[FAIL] missing support modes: {set(PRIMARY_MODES) - have_modes}")
    # no OFFLINE_TTA selected under degenerate/unavailable ACAR-harm
    for r in reports2f:
        s = r["report"]["router_summary"]
        if s["source_acar_harm_calibration_state"] in DEGENERATE and s["action_counts"].get("offline_tta", 0) > 0:
            raise SystemExit(f"[FAIL] OFFLINE_TTA selected under degenerate ACAR-harm: {r['world']}/{r['seed']}")
    # R2 nested coverage must exceed R2 baseline coverage (the Step-2F fix)
    def cov(mode):
        return _mean([r["report"]["router_summary"]["coverage"] for r in reports2f
                      if r["world"] == "R2" and r["support_mode"] == mode])
    if not (cov("nested_site_excess_q95") > cov("in_source_subject_q95")):
        raise SystemExit(f"[FAIL] R2 nested coverage {cov('nested_site_excess_q95')} !> baseline {cov('in_source_subject_q95')}")
    print("[validate] structural checks passed "
          f"(R2 coverage baseline={cov('in_source_subject_q95'):.2f} -> nested={cov('nested_site_excess_q95'):.2f})")


# --------------------------------------------------------------------------- table 1
T1_COLS = ["world", "support_mode", "n_seeds", "n_domains", "strict_bacc_mean",
           "raw_offline_delta_bacc_mean", "coverage_mean", "refusal_rate_mean", "identity_rate_mean",
           "offline_tta_rate_mean", "accepted_bacc_mean", "selected_gain_vs_identity_mean",
           "missed_benefit_mean", "avoided_harm_mean", "support_threshold_mean",
           "mean_target_density_nll_target_prior_mean", "support_mismatch_domains_total",
           "low_ess_domains_total", "acar_harm_state"]


def table1(reports2f):
    rows = []
    by = defaultdict(list)
    for r in reports2f:
        by[(r["world"], r["support_mode"])].append(r)
    for world in WORLDS:
        for mode in PRIMARY_MODES:
            rs = by.get((world, mode), [])
            if not rs:
                continue
            ss = [r["report"]["router_summary"] for r in rs]
            per_counts = [len(r["report"]["per_domain"]) for r in rs]

            def sm_low(kind):
                tot = 0
                for r in rs:
                    for dv in r["report"]["per_domain"].values():
                        idr = dv["action_scores"]["identity"]["reason_codes"]
                        tot += 1 if kind in idr else 0
                return tot
            rows.append(dict(
                world=world, support_mode=mode, n_seeds=len(rs), n_domains=sum(per_counts),
                strict_bacc_mean=_mean([r["strict_bacc"] for r in rs]),
                raw_offline_delta_bacc_mean=_mean([r["raw_offline_delta_bacc"] for r in rs]),
                coverage_mean=_mean([s["coverage"] for s in ss]),
                refusal_rate_mean=_mean([s["refusal_rate"] for s in ss]),
                identity_rate_mean=_mean([s["identity_rate"] for s in ss]),
                offline_tta_rate_mean=_mean([s["offline_tta_rate"] for s in ss]),
                accepted_bacc_mean=_mean([s["accepted_bacc"] for s in ss]),
                selected_gain_vs_identity_mean=_mean([s["selected_mean_gain_vs_identity"] for s in ss]),
                missed_benefit_mean=_mean([s["missed_benefit"] for s in ss]),
                avoided_harm_mean=_mean([s["avoided_harm"] for s in ss]),
                support_threshold_mean=_mean([r["threshold"] for r in rs]),
                mean_target_density_nll_target_prior_mean=_mean(
                    [np.mean([dv["support"]["density_nll_target_prior"] for dv in r["report"]["per_domain"].values()]) for r in rs]),
                support_mismatch_domains_total=sm_low("OACI_TOS_SUPPORT_MISMATCH"),
                low_ess_domains_total=sm_low("OACI_TOS_LOW_EFFECTIVE_SAMPLE_SIZE"),
                acar_harm_state=Counter(s["source_acar_harm_calibration_state"] for s in ss).most_common(1)[0][0],
            ))
    return rows


# --------------------------------------------------------------------------- table 2
T2_COLS = ["world", "component", "coverage", "refusal_rate", "identity_rate", "offline_tta_rate",
           "bacc_or_accepted_bacc", "delta_vs_identity", "missed_benefit", "avoided_harm",
           "primary_failure_or_blocker"]


def _top_blocker(reports, world, mode, which="identity"):
    c = Counter()
    for r in reports:
        if r["world"] != world or r["support_mode"] != mode:
            continue
        for dv in r["report"]["per_domain"].values():
            if which == "identity":
                c.update(dv["action_scores"]["identity"]["reason_codes"])
            else:
                c.update(dv["action_scores"]["offline_tta"]["blocking_reason_codes"])
    return c


def table2(reports2f, t1):
    t1i = {(r["world"], r["support_mode"]): r for r in t1}
    rows = []
    for world in WORLDS:
        base = t1i[(world, "in_source_subject_q95")]
        nested = t1i[(world, "nested_site_excess_q95")]
        strict = base["strict_bacc_mean"]
        raw_d = base["raw_offline_delta_bacc_mean"]
        # raw identity
        rows.append(dict(world=world, component="raw_identity", coverage=1.0, refusal_rate=0.0,
                         identity_rate=1.0, offline_tta_rate=0.0, bacc_or_accepted_bacc=strict,
                         delta_vs_identity=0.0, missed_benefit=0.0, avoided_harm=0.0,
                         primary_failure_or_blocker="none (always outputs identity)"))
        # raw offline tta
        rows.append(dict(world=world, component="raw_offline_tta", coverage=1.0, refusal_rate=0.0,
                         identity_rate=0.0, offline_tta_rate=1.0, bacc_or_accepted_bacc=strict + raw_d,
                         delta_vs_identity=raw_d, missed_benefit=0.0,
                         avoided_harm=0.0,
                         primary_failure_or_blocker=("TTA harmful (d<0)" if raw_d < 0 else "TTA neutral/beneficial")))
        # routers
        for mode, t in (("in_source_subject_q95", base), ("nested_site_excess_q95", nested)):
            tta_block = _top_blocker(reports2f, world, mode, "offline")
            id_reasons = _top_blocker(reports2f, world, mode, "identity")
            if mode == "in_source_subject_q95":
                blk = "SUPPORT_MISMATCH over-refusal" if id_reasons.get("OACI_TOS_SUPPORT_MISMATCH") else "refuse"
            else:
                bits = []
                if tta_block.get("OACI_ACAR_HARM_CALIBRATION_DEGENERATE"):
                    bits.append("ACAR_HARM_DEGENERATE blocks TTA")
                if id_reasons.get("OACI_TOS_LOW_EFFECTIVE_SAMPLE_SIZE"):
                    bits.append("LOW_ESS refuses subset")
                if world == "HF3":
                    bits.append("concept-degraded identity may pass support")
                blk = "; ".join(bits) if bits else "identity accepted (support valid)"
            rows.append(dict(world=world, component=f"router_{mode}",
                             coverage=t["coverage_mean"], refusal_rate=t["refusal_rate_mean"],
                             identity_rate=t["identity_rate_mean"], offline_tta_rate=t["offline_tta_rate_mean"],
                             bacc_or_accepted_bacc=t["accepted_bacc_mean"],
                             delta_vs_identity=t["selected_gain_vs_identity_mean"],
                             missed_benefit=t["missed_benefit_mean"], avoided_harm=t["avoided_harm_mean"],
                             primary_failure_or_blocker=blk))
    return rows


# --------------------------------------------------------------------------- table 3
T3_COLS = ["world", "support_mode", "reason_code", "top_level_count", "identity_action_count",
           "offline_tta_action_count", "offline_tta_blocker_count"]


def table3(reports2f):
    agg = defaultdict(lambda: dict(top=0, idc=0, offc=0, offb=0))
    for r in reports2f:
        world, mode = r["world"], r["support_mode"]
        for dv in r["report"]["per_domain"].values():
            for rc in dv["reason_codes"]:
                agg[(world, mode, rc)]["top"] += 1
            for rc in dv["action_scores"]["identity"]["reason_codes"]:
                agg[(world, mode, rc)]["idc"] += 1
            for rc in dv["action_scores"]["offline_tta"]["reason_codes"]:
                agg[(world, mode, rc)]["offc"] += 1
            for rc in dv["action_scores"]["offline_tta"]["blocking_reason_codes"]:
                agg[(world, mode, rc)]["offb"] += 1
    rows = []
    for (world, mode, rc), c in agg.items():
        rows.append(dict(world=world, support_mode=mode, reason_code=rc,
                         top_level_count=c["top"], identity_action_count=c["idc"],
                         offline_tta_action_count=c["offc"], offline_tta_blocker_count=c["offb"]))
    rows.sort(key=lambda r: (r["world"], r["support_mode"], -r["top_level_count"], r["reason_code"]))
    return rows


# --------------------------------------------------------------------------- table 4
T4_COLS = ["world", "seed", "domain_id", "support_mode", "decision_action", "identity_bacc",
           "offline_tta_bacc", "raw_gain", "selected_bacc", "selected_gain_vs_identity", "reason_codes",
           "offline_tta_blocking_reason_codes", "density_nll_target_prior",
           "support_threshold_nll_target_prior", "target_support_excess", "ess", "interpretation"]


def _domain_iter(reports2f):
    for r in reports2f:
        thr = r["threshold"]
        for did, dv in r["report"]["per_domain"].items():
            sup = dv["support"]
            yield dict(world=r["world"], seed=r["seed"], domain_id=did, support_mode=r["support_mode"],
                       decision_action=dv["decision_action"], identity_bacc=dv["identity_bacc"],
                       offline_tta_bacc=dv["offline_tta_bacc"], raw_gain=dv["raw_gain"],
                       selected_bacc=dv["selected_bacc"], selected_gain_vs_identity=dv["selected_gain_vs_identity"],
                       reason_codes=dv["reason_codes"],
                       offline_tta_blocking_reason_codes=dv["action_scores"]["offline_tta"]["blocking_reason_codes"],
                       identity_admissible=dv["action_scores"]["identity"]["admissible"],
                       density_nll_target_prior=sup["density_nll_target_prior"],
                       support_threshold_nll_target_prior=thr,
                       target_support_excess=sup["density_nll_target_prior"] - thr, ess=sup["ess"],
                       low_ess=("OACI_TOS_LOW_EFFECTIVE_SAMPLE_SIZE" in dv["action_scores"]["identity"]["reason_codes"]))


def table4(reports2f):
    doms = list(_domain_iter(reports2f))
    N = "nested_site_excess_q95"
    picks = []

    def pick(pred, interp):
        cand = [d for d in doms if pred(d)]
        if cand:
            d = dict(cand[0]); d["interpretation"] = interp
            picks.append(d)

    pick(lambda d: d["world"] == "R2" and d["support_mode"] == N and d["decision_action"] == "identity" and d["raw_gain"] > 0.02,
         "recoverable: identity accepted, raw TTA would help -> missed benefit (TTA blocked by degenerate ACAR-harm)")
    pick(lambda d: d["world"] == "HF3" and d["support_mode"] == N and d["raw_gain"] < -0.05 and "OACI_ACAR_HARM_CALIBRATION_DEGENERATE" in d["offline_tta_blocking_reason_codes"],
         "harmful: raw TTA hurts, TTA blocked by degenerate ACAR-harm -> avoided harm")
    pick(lambda d: d["world"] == "HF3" and d["support_mode"] == N and d["decision_action"] == "identity" and d["identity_bacc"] < 0.6,
         "support-valid but concept-degraded identity: passes support yet low bAcc -> source-only concept limitation")
    pick(lambda d: d["world"] == "H_OOD" and d["support_mode"] == N and d["low_ess"] and d["decision_action"] == "refuse",
         "H_OOD: density SUPPORT_MISMATCH cleared by nested threshold but LOW_ESS refuses this subject -> density/ESS split")
    for p in picks:
        p.pop("identity_admissible", None); p.pop("low_ess", None)
    return picks


# --------------------------------------------------------------------------- LaTeX
def _tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def paper_tables_tex(t1, t2, t4):
    lines = ["% Project B Step-2 synthetic router tables (auto-generated; \\input-able)"]
    # Table 1
    lines += [r"\begin{table}[t]\centering\small",
              r"\caption{Project B router: world $\times$ support-calibration mode (means over seeds).}",
              r"\label{tab:pb-world-support}",
              r"\begin{tabular}{llrrrrr}\toprule",
              r"world & mode & strict & cov. & id.\ rate & off-TTA & acc.\ bAcc \\ \midrule"]
    for r in t1:
        lines.append(f"{_tex_escape(r['world'])} & {_tex_escape(r['support_mode'])} & "
                     f"{r['strict_bacc_mean']:.3f} & {r['coverage_mean']:.2f} & {r['identity_rate_mean']:.2f} & "
                     f"{r['offline_tta_rate_mean']:.2f} & {r['accepted_bacc_mean']:.3f} \\\\")
    lines += [r"\bottomrule\end{tabular}\end{table}", ""]
    # Table 2
    lines += [r"\begin{table}[t]\centering\small",
              r"\caption{Component ablation per world.}\label{tab:pb-ablation}",
              r"\begin{tabular}{llrrrl}\toprule",
              r"world & component & cov. & off-TTA & bAcc & primary blocker \\ \midrule"]
    for r in t2:
        lines.append(f"{_tex_escape(r['world'])} & {_tex_escape(r['component'])} & {r['coverage']:.2f} & "
                     f"{r['offline_tta_rate']:.2f} & {r['bacc_or_accepted_bacc']:.3f} & "
                     f"{_tex_escape(r['primary_failure_or_blocker'])} \\\\")
    lines += [r"\bottomrule\end{tabular}\end{table}", ""]
    # Table 3 (limitation examples)
    lines += [r"\begin{table}[t]\centering\small",
              r"\caption{Representative limitation examples (nested mode).}\label{tab:pb-limits}",
              r"\begin{tabular}{llll}\toprule",
              r"world & action & id.\ bAcc & interpretation \\ \midrule"]
    for r in t4:
        lines.append(f"{_tex_escape(r['world'])} & {_tex_escape(r['decision_action'])} & "
                     f"{r['identity_bacc']:.3f} & {_tex_escape(r['interpretation'][:60])} \\\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- claim boundary + report
def claim_boundary(t1):
    t1i = {(r["world"], r["support_mode"]): r for r in t1}
    def locked(world):
        b = t1i[(world, "in_source_subject_q95")]; n = t1i[(world, "nested_site_excess_q95")]
        return dict(strict_bacc_mean=round(b["strict_bacc_mean"], 3),
                    raw_offline_delta_bacc_mean=round(b["raw_offline_delta_bacc_mean"], 3),
                    baseline_coverage=round(b["coverage_mean"], 3), nested_coverage=round(n["coverage_mean"], 3),
                    nested_accepted_bacc=round(n["accepted_bacc_mean"], 3),
                    nested_offline_tta_rate=round(n["offline_tta_rate_mean"], 3),
                    nested_low_ess_domains=n["low_ess_domains_total"], acar_harm_state=n["acar_harm_state"])
    return {
        "claimable": [
            "Default Project B v1 never selects OFFLINE_TTA when ACAR-harm calibration is degenerate/unavailable.",
            "Nested source-site support excess calibration fixes the Step-2E all-refuse failure on R2.",
            "Project B can allow support-valid IDENTITY while still refusing low-ESS targets.",
            "OACI reason codes expose whether refusal came from support, ESS, ACAR degeneracy, or TTA evidence.",
        ],
        "not_claimable": [
            "It cannot claim source-only ACAR-harm is generally identifiable.",
            "It cannot claim to recover R2's raw TTA benefit, because default v1 blocks TTA under harm-calibration degeneracy.",
            "It cannot claim support-valid identity is accurate under concept shift; HF3 shows concept-degraded identity can pass support checks.",
            "It cannot claim the density support threshold alone catches H-OOD after nested threshold widening; LOW_ESS is the active blocker for only part of H-OOD.",
            "It cannot claim target-label-tuned thresholds; thresholds are source-only and label-safe.",
        ],
        "locked_worlds": {w: locked(w) for w in WORLDS},
        "primary_support_mode": "nested_site_excess_q95",
        "primary_router_posture": "support-valid identity; no TTA under degenerate ACAR-harm",
        "known_limitations": [
            "R2 missed benefit (TTA blocked under degenerate ACAR-harm)",
            "HF3 concept-degraded identity can pass source-only support checks",
            "H_OOD density threshold cleared after nested calibration; LOW_ESS catches only a subset",
        ],
    }


def report_md(t1, t2, cb):
    t1i = {(r["world"], r["support_mode"]): r for r in t1}
    def line(w):
        b = t1i[(w, "in_source_subject_q95")]; n = t1i[(w, "nested_site_excess_q95")]
        return (f"| {w} | {b['strict_bacc_mean']:.3f} | {b['raw_offline_delta_bacc_mean']:+.3f} | "
                f"{b['coverage_mean']:.2f} | {n['coverage_mean']:.2f} | {n['accepted_bacc_mean']:.3f} | "
                f"{n['offline_tta_rate_mean']:.2f} | {n['low_ess_domains_total']} |")
    md = ["# Project B Step-2 Synthetic Router Report (auto-generated)", "",
          "Generated by `scripts/project_b_step2g_report.py` from the frozen Step-2E/2F outputs.", "",
          "## Main result (baseline vs nested support calibration)", "",
          "| world | strict | raw dTTA | base cov | nested cov | nested acc-bAcc | off-TTA | low-ESS |",
          "|---|---|---|---|---|---|---|---|"]
    md += [line(w) for w in WORLDS]
    md += ["", "## Claimable", ""] + [f"- {c}" for c in cb["claimable"]]
    md += ["", "## NOT claimable", ""] + [f"- {c}" for c in cb["not_claimable"]]
    md += ["", "## Known limitations", ""] + [f"- {c}" for c in cb["known_limitations"]]
    return "\n".join(md) + "\n"


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Project B Step-2G frozen synthetic report")
    ap.add_argument("--step2e", required=True)
    ap.add_argument("--step2f", required=True)
    ap.add_argument("--out", default="/tmp/project_b_step2g_report")
    args = ap.parse_args()

    reports2f = _load_reports(args.step2f)
    if not reports2f:
        raise SystemExit(f"[FAIL] no Step-2F reports in {args.step2f}")
    _ = _load_reports(args.step2e)          # presence cross-check (Step-2E == 2F baseline mode)
    validate(args.step2f, reports2f)

    os.makedirs(args.out, exist_ok=True)
    t1 = table1(reports2f)
    t2 = table2(reports2f, t1)
    t3 = table3(reports2f)
    t4 = table4(reports2f)
    cb = claim_boundary(t1)

    _write_csv(os.path.join(args.out, "table1_world_support_summary.csv"), T1_COLS, t1)
    _write_csv(os.path.join(args.out, "table2_component_ablation.csv"), T2_COLS, t2)
    _write_csv(os.path.join(args.out, "table3_reason_code_audit.csv"), T3_COLS, t3)
    _write_csv(os.path.join(args.out, "table4_limitation_examples.csv"), T4_COLS, t4)
    with open(os.path.join(args.out, "paper_tables.tex"), "w") as f:
        f.write(paper_tables_tex(t1, t2, t4))
    with open(os.path.join(args.out, "project_b_step2_synthetic_report.md"), "w") as f:
        f.write(report_md(t1, t2, cb))
    with open(os.path.join(args.out, "claim_boundary.json"), "w") as f:
        json.dump(cb, f, indent=2)

    print(f"[step2g] wrote 7 artifacts to {args.out}")
    print("\n===== table2 component ablation =====")
    for r in t2:
        print(f"  {r['world']:6s} {r['component']:32s} cov={r['coverage']:.2f} off_tta={r['offline_tta_rate']:.2f} "
              f"bAcc={r['bacc_or_accepted_bacc']:.3f} d={r['delta_vs_identity']:+.3f} | {r['primary_failure_or_blocker']}")


if __name__ == "__main__":
    main()
