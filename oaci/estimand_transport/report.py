"""C22 — assemble the Estimand Transport Mechanism Audit. Locks the C19 config hash (read-only), extracts the
frozen-probe per-candidate scores + epoch/features across regimes, runs the decomposition / offset-scale /
epoch-confound / normalization / feature-shift analyses, and emits the deterministic transport-failure
taxonomy. The epoch/order baseline (Q2) is reported BEFORE any normalization-rescue interpretation. All
normalization is diagnostic-only. No selector, no probe tuning, no external dataset."""
from __future__ import annotations

import argparse
import csv
import json
import os

from ..competence_probe import schema as c19
from ..support_stress import source_signal_recompute as ssr
from ..support_stress.stress_plan import boundary_classes_from_c16
from . import (epoch_confound, estimand_decomposition, feature_shift, normalization_diagnostics, offset_scale,
               schema, score_loader, taxonomy)


def _lock_config():
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C22 requires the frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
    return got


def _writecsv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def _boundary_classes(c16_path):
    return boundary_classes_from_c16(json.load(open(c16_path))["harm_decomposition"]["per_class_recall_delta"])


def run(extract_dir, c10_dir, *, folds=None, n_workers=8, c16_path="oaci/reports/C16_MECHANISM_DEEP_DIVE.json") -> dict:
    cfg_hash = _lock_config()
    bnd = _boundary_classes(c16_path)
    fold_dirs = folds if folds is not None else ssr._list_folds(extract_dir)
    leak = ssr.precompute_all_leakage(extract_dir, boundary_classes=bnd, folds=fold_dirs, n_workers=n_workers,
                                      regimes=list(schema.ALL_REGIMES))
    rows = score_loader.score_table(extract_dir, c10_dir, boundary_classes=bnd, leakage_cache=leak, folds=fold_dirs)
    dec = estimand_decomposition.decompose(rows); dsum = estimand_decomposition.summary(dec)
    off = offset_scale.offset_scale(rows)
    ep = epoch_confound.epoch_confound(rows)                 # Q2 -- computed/reported before rescue
    norm = normalization_diagnostics.normalization_diagnostics(rows)
    fsh = feature_shift.feature_shift(rows)
    tax = taxonomy.transport_taxonomy(ep, dsum, norm, off, fsh)
    return {"config_hash": cfg_hash, "n_score_rows": len(rows), "decomposition": dec, "decomposition_summary": dsum,
            "offset_scale": off, "epoch_confound": ep, "normalization": norm, "feature_shift": fsh,
            "taxonomy": tax, "boundary_classes": list(bnd), "diagnostic_only_non_deployable": True}


def _interpret(res):
    res["taxonomy"] = taxonomy.transport_taxonomy(res["epoch_confound"], res["decomposition_summary"],
                                                  res["normalization"], res["offset_scale"], res["feature_shift"])
    return res


# ---------- tables ----------
def write_tables(res, tdir) -> None:
    os.makedirs(tdir, exist_ok=True)
    dec = res["decomposition"]
    _writecsv(os.path.join(tdir, "pooled_vs_within_estimand_decomposition.csv"),
              [{"group": k, "mode": d["mode"], "regime": d["regime"], "pooled_auc": d["pooled_auc"],
                "within_target_mean_auc": d["within_target_mean_auc"], "pooled_minus_within": d["pooled_minus_within"],
                "within_min": d["within_target_min"], "within_max": d["within_target_max"]} for k, d in dec.items()],
              ["group", "mode", "regime", "pooled_auc", "within_target_mean_auc", "pooled_minus_within",
               "within_min", "within_max"])
    off = res["offset_scale"]
    _writecsv(os.path.join(tdir, "offset_scale_variance_components.csv"),
              [{"mode": m, "target_between_fraction": off[m]["target_between_fraction"],
                "regime_between_fraction": off[m]["regime_between_fraction"],
                "target_offset_vs_baserate_corr": off[m]["target_offset_vs_baserate_corr"]} for m in off],
              ["mode", "target_between_fraction", "regime_between_fraction", "target_offset_vs_baserate_corr"])
    _writecsv(os.path.join(tdir, "target_regime_score_offsets.csv"),
              [{"mode": m, "group_type": "regime", "group": g, "score_offset": v}
               for m in off for g, v in off[m]["regime_offsets"].items()],
              ["mode", "group_type", "group", "score_offset"])
    ep = res["epoch_confound"]
    _writecsv(os.path.join(tdir, "epoch_order_baselines.csv"),
              [{"baseline": b, "within_target_strength": s} for b, s in ep["baseline_within_target_strength"].items()]
              + [{"baseline": "robust_core_probe", "within_target_strength": ep["probe_within_target_strength"]}],
              ["baseline", "within_target_strength"])
    _writecsv(os.path.join(tdir, "residual_signal_after_epoch_control.csv"),
              [{"metric": "partial_spearman_score_label_given_epoch", "value": ep["partial_spearman_score_label_given_epoch"]},
               {"metric": "residual_signal_present", "value": ep["residual_signal_present"]},
               {"metric": "probe_beats_epoch_family", "value": ep["probe_beats_epoch_family"]},
               {"metric": "epoch_confounded", "value": ep["epoch_confounded"]}], ["metric", "value"])
    nm = res["normalization"]["per_mode"]
    _writecsv(os.path.join(tdir, "targetwise_normalization_results.csv"),
              [{"mode": m, "normalization": k, "pooled_auc": v}
               for m in nm for k, v in nm[m]["pooled_auc_by_normalization"].items()],
              ["mode", "normalization", "pooled_auc"])
    _writecsv(os.path.join(tdir, "regimewise_normalization_results.csv"),
              [{"mode": m, "target_normalization_recovers": nm[m]["target_normalization_recovers"],
                "pooled_none": nm[m]["pooled_none"], "best_target_normalized_pooled": nm[m]["best_target_normalized_pooled"]}
               for m in nm], ["mode", "target_normalization_recovers", "pooled_none", "best_target_normalized_pooled"])
    fs = res["feature_shift"]["per_feature"]
    _writecsv(os.path.join(tdir, "feature_shift_by_target_regime.csv"),
              [{"feature": f, "target_between_fraction": v["target_between_fraction"],
                "regime_between_fraction": v["regime_between_fraction"],
                "within_target_label_spearman": v["within_target_label_spearman"],
                "usable_ranking": v["usable_ranking"], "offset_dominated": v["offset_dominated"]}
               for f, v in fs.items()],
              ["feature", "target_between_fraction", "regime_between_fraction", "within_target_label_spearman",
               "usable_ranking", "offset_dominated"])
    _writecsv(os.path.join(tdir, "feature_rank_stability.csv"),
              [{"feature": f, "usable_ranking": v["usable_ranking"], "regime_between_fraction": v["regime_between_fraction"]}
               for f, v in fs.items()], ["feature", "usable_ranking", "regime_between_fraction"])
    t = res["taxonomy"]
    _writecsv(os.path.join(tdir, "score_transport_failure_modes.csv"),
              [{"factor": "epoch_confounded", "value": t["epoch_confounded"]},
               {"factor": "within_target_present", "value": t["within_target_present"]},
               {"factor": "target_normalization_recovers_pooled", "value": t["target_normalization_recovers_pooled"]},
               {"factor": "feature_offset_dominated", "value": t["feature_offset_dominated"]}], ["factor", "value"])
    _writecsv(os.path.join(tdir, "c22_case_taxonomy.csv"),
              [{"primary_case": t["primary_case"], "secondary": ";".join(t["secondary"]),
                "interpretation": t["interpretation"], "next_science": t["next_science"]}],
              ["primary_case", "secondary", "interpretation", "next_science"])


def _f(x):
    return "n/a" if x is None else (f"{x:+.3f}" if isinstance(x, float) else str(x))


def render_md(res) -> str:
    t = res["taxonomy"]; ep = res["epoch_confound"]; ds = res["decomposition_summary"]
    L = [f"# C22 — Estimand Transport Mechanism Audit (frozen C19 `{res['config_hash']}`)", "",
         "> Read-only mechanism audit: WHY does the C19 in-regime source-only competence signal not transport as "
         "a pooled cross-regime estimand? No probe tuning, no selector, no external dataset. Normalization is "
         "diagnostic-only.", "",
         f"- **CASE: `{t['primary_case']}`**" + (f"  ·  secondary: {t['secondary']}" if t['secondary'] else ""),
         f"- {t['interpretation']}", f"- next: {t['next_science']}", "",
         "## Q2 — epoch / trajectory-position confound (reported FIRST, gates everything)", "",
         f"- probe within-target strength **{_f(ep['probe_within_target_strength'])}** vs best epoch-family "
         f"baseline **{_f(ep['best_epoch_family_baseline_strength'])}** → probe beats epoch family: "
         f"**{ep['probe_beats_epoch_family']}**",
         f"- residual (partial Spearman score~label | epoch) = **{_f(ep['partial_spearman_score_label_given_epoch'])}** "
         f"→ residual signal present: **{ep['residual_signal_present']}**",
         f"- **epoch_confounded: {ep['epoch_confounded']}**  "
         + ("(→ T2: the in-regime positive is a trajectory-position diagnostic, NOT competence)"
            if ep['epoch_confounded'] else "(→ within-target signal survives epoch control)"), "",
         "## Q1 — pooled vs within-target decomposition", "",
         f"- mean pooled AUC **{_f(ds['mean_pooled_auc'])}** vs mean within-target AUC **{_f(ds['mean_within_target_auc'])}** "
         f"(gap {_f(ds['mean_pooled_minus_within'])}); within exceeds pooled everywhere: "
         f"**{ds['within_exceeds_pooled_everywhere']}**", "",
         "| group | pooled | within-target mean | gap |", "|---|---:|---:|---:|"]
    for k, d in res["decomposition"].items():
        L.append(f"| {k} | {_f(d['pooled_auc'])} | {_f(d['within_target_mean_auc'])} | {_f(d['pooled_minus_within'])} |")
    nm = res["normalization"]["per_mode"]
    L += ["", "## Q3 — post-hoc normalization diagnostics (MECHANISM only, NON-deployable)", "",
          "| mode | pooled (none) | best target-normalized pooled | target-normalization recovers |",
          "|---|---:|---:|:--:|"]
    for m in nm:
        L.append(f"| {m} | {_f(nm[m]['pooled_none'])} | {_f(nm[m]['best_target_normalized_pooled'])} | "
                 f"{nm[m]['target_normalization_recovers']} |")
    fsh = res["feature_shift"]
    L += ["", "> Target/regime-wise normalization needs the target/regime identity at score time -> NON-deployable; "
          "reported as mechanism only. Recovery => rank-like signal / score-offset problem; no recovery => "
          "regime-specific relationship shift.", "",
          "## Q4 — feature-level offset vs ranking", "",
          f"- {fsh['n_usable_ranking']}/{fsh['n_features']} robust-core features carry within-target ranking; "
          f"{fsh['n_offset_dominated']}/{fsh['n_features']} are offset-dominated "
          f"(fraction {_f(fsh['offset_dominated_fraction'])}).", "",
          "## Boundary of the claim", "",
          "> DIAGNOSTIC-ONLY mechanism audit. Normalization diagnostics are NOT deployable procedures (they use "
          "target/regime identity). No selector is produced; the C19/C20 estimand boundary is explained, not "
          "rescued."]
    return "\n".join(L)


def render_epoch_md(res) -> str:
    ep = res["epoch_confound"]
    return (f"# C22 — Epoch / trajectory-position confound audit\n\n> Reported BEFORE any normalization-rescue "
            f"interpretation (hard gate).\n\n{ep['note']}\n\n"
            f"- probe within-target strength: {_f(ep['probe_within_target_strength'])}\n"
            f"- epoch-family baseline strengths: {ep['baseline_within_target_strength']}\n"
            f"- probe beats epoch family: {ep['probe_beats_epoch_family']}\n"
            f"- partial Spearman(score, label | epoch): {_f(ep['partial_spearman_score_label_given_epoch'])} "
            f"(n_targets {ep['n_targets_partial']})\n- residual signal present: {ep['residual_signal_present']}\n"
            f"- **epoch_confounded: {ep['epoch_confounded']}**\n")


def render_normalization_md(res) -> str:
    nm = res["normalization"]
    lines = [f"# C22 — Score-normalization diagnostics (MECHANISM only, NON-deployable)\n\n> {nm['note']}\n"]
    for mode, d in nm["per_mode"].items():
        lines.append(f"\n## {mode}\n- pooled (none): {_f(d['pooled_none'])}\n- by normalization: "
                     f"{ {k: round(v,3) if isinstance(v,float) else v for k,v in d['pooled_auc_by_normalization'].items()} }\n"
                     f"- target-normalization recovers pooled: {d['target_normalization_recovers']}")
    return "\n".join(lines)


def _guard_forbidden(md) -> None:
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        if s in low:
            raise ValueError(f"forbidden claim in C22 report: {s!r}")


def _write_artifacts(res, out_dir):
    md = render_md(res); _guard_forbidden(md)
    ep = render_epoch_md(res); _guard_forbidden(ep)
    nm = render_normalization_md(res); _guard_forbidden(nm)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C22_ESTIMAND_TRANSPORT_AUDIT.md"), "w").write(md)
    json.dump(res, open(os.path.join(out_dir, "C22_ESTIMAND_TRANSPORT_AUDIT.json"), "w"), indent=2, sort_keys=True, default=str)
    open(os.path.join(out_dir, "C22_EPOCH_CONFOUND_AUDIT.md"), "w").write(ep)
    open(os.path.join(out_dir, "C22_SCORE_NORMALIZATION_DIAGNOSTICS.md"), "w").write(nm)
    write_tables(res, os.path.join(out_dir, "c22_tables"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.estimand_transport.report")
    ap.add_argument("--extract-dir")
    ap.add_argument("--c10-dir")
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--reinterpret", default=None)
    args = ap.parse_args(argv)
    if args.reinterpret:
        res = _interpret(json.load(open(args.reinterpret)))
        _write_artifacts(res, args.out_dir)
        print(f"[C22 reinterpret] case={res['taxonomy']['primary_case']}")
        return 0
    if not (args.extract_dir and args.c10_dir):
        ap.error("full run requires --extract-dir and --c10-dir (or --reinterpret)")
    res = run(args.extract_dir, args.c10_dir, n_workers=args.n_workers)
    _write_artifacts(res, args.out_dir)
    t = res["taxonomy"]
    print(f"[C22] case={t['primary_case']} epoch_confounded={t['epoch_confounded']} "
          f"within_present={t['within_target_present']} norm_recovers={t['target_normalization_recovers_pooled']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
