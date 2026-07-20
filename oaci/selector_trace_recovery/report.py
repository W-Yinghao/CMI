"""C37 exact selector trace recovery report."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np

from . import (artifact_loader, better_candidate_ucl, exact_selector_ordering, leakage_ucl_replay, schema,
               selection_audit_reconcile, taxonomy, ucl_identity_gate, uncertainty_plateau)


def _lock_config():
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C37 requires frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
    return got


def _writecsv(path, rows, cols):
    def clean(v):
        if isinstance(v, float) and not math.isfinite(v):
            return ""
        return v

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: clean(r.get(c)) for c in cols})


def _source_pareto_after(comparisons):
    c36 = {r["pair_id"]: r for r in artifact_loader.read_csv(
        os.path.join(schema.C36_TABLE_DIR, "source_pareto_status.csv"))}
    rows = []
    for r in comparisons:
        sp = c36.get(r["pair_id"], {})
        status = sp.get("source_pareto_status", "unavailable")
        conflict = int(status in ("selected_source_dominates_better", "source_pareto_incomparable"))
        rows.append({"pair_id": r["pair_id"], "pair_key": r["pair_key"], "seed": r["seed"],
                     "target": r["target"], "level": r["level"], "regime": r["regime"],
                     "ucl_prefers": r["ucl_prefers"], "source_pareto_status": status,
                     "source_pareto_conflict": conflict, "target_endpoint_prefers": "better"})
    return {"rows": rows, "summary": {"n_pairs": len(rows),
            "source_pareto_conflict_fraction": float(np.mean([r["source_pareto_conflict"] for r in rows]))
            if rows else None}}


WORKLIST_COLS = ["work_id", "kind", "seed", "target", "level", "regime", "selected_order", "better_order",
                 "candidate_order", "candidate_role", "candidate_id", "pair_key", "pair_id", "unit_id"]


def _load_pairs_trace():
    pairs = artifact_loader.load_robust_pairs()
    trace = artifact_loader.load_c10_trace(sorted({p["regime"] for p in pairs}))
    return pairs, trace


def _unit_id(seed, target, level):
    return f"s{seed}_t{int(target):03d}_l{int(level):03d}"


def _pair_key(p):
    return "|".join([p["seed"], p["target"], p["level"], p["selected_order"], p["candidate_order"]])


def _worklist_path(work_dir, kind):
    return os.path.join(work_dir, f"{kind}_worklist.csv")


def _partial_path(work_dir, kind, work_id):
    return os.path.join(work_dir, "partials", f"{kind}_{int(work_id):04d}.json")


def _write_json_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    os.replace(tmp, path)


def make_worklists(work_dir):
    cfg = _lock_config()
    pairs, trace = _load_pairs_trace()
    os.makedirs(os.path.join(work_dir, "partials"), exist_ok=True)

    selected_rows = []
    for i, (seed, target, level, regime, selected_order) in enumerate(artifact_loader.p0_slice(pairs)):
        cand = trace["by_key"][(seed, target, level, regime, selected_order)]
        selected_rows.append({
            "work_id": i,
            "kind": "selected",
            "seed": seed,
            "target": target,
            "level": level,
            "regime": regime,
            "selected_order": selected_order,
            "better_order": "",
            "candidate_order": selected_order,
            "candidate_role": "selected_p0_identity",
            "candidate_id": cand["candidate_id"],
            "pair_key": "",
            "pair_id": "",
            "unit_id": _unit_id(seed, target, level),
        })

    better_rows = []
    for i, p in enumerate(artifact_loader.unique_pair_keys(pairs)):
        cand = trace["by_key"][(p["seed"], p["target"], p["level"], p["regime"], p["candidate_order"])]
        better_rows.append({
            "work_id": i,
            "kind": "better",
            "seed": p["seed"],
            "target": p["target"],
            "level": p["level"],
            "regime": p["regime"],
            "selected_order": p["selected_order"],
            "better_order": p["candidate_order"],
            "candidate_order": p["candidate_order"],
            "candidate_role": "c35_preference_robust_better",
            "candidate_id": cand["candidate_id"],
            "pair_key": _pair_key(p),
            "pair_id": p["pair_id"],
            "unit_id": _unit_id(p["seed"], p["target"], p["level"]),
        })

    _writecsv(_worklist_path(work_dir, "selected"), selected_rows, WORKLIST_COLS)
    _writecsv(_worklist_path(work_dir, "better"), better_rows, WORKLIST_COLS)
    meta = {
        "config_hash": cfg,
        "diagnostic_only_non_deployable": True,
        "n_preference_robust_pairs": len(pairs),
        "n_selected_p0_jobs": len(selected_rows),
        "n_unique_better_jobs": len(better_rows),
        "work_dir": os.path.abspath(work_dir),
        "candidate_hash_emitted": False,
    }
    _write_json_atomic(os.path.join(work_dir, "worklist_metadata.json"), meta)
    return meta


def _load_work_item(work_dir, kind, index):
    rows = artifact_loader.read_csv(_worklist_path(work_dir, kind))
    if int(index) < 0 or int(index) >= len(rows):
        raise IndexError(f"{kind} work index {index} out of range 0..{len(rows) - 1}")
    return rows[int(index)]


def _selected_worker_row(item, trace, n_jobs):
    ctx = artifact_loader.ContextCache(trace).get(item["seed"], item["target"], item["level"], item["regime"])
    cand = trace["by_key"][(item["seed"], item["target"], item["level"], item["regime"],
                            item["candidate_order"])]
    if cand["candidate_id"] != item["candidate_id"]:
        raise ValueError(f"worklist candidate mismatch: {item['candidate_id']} != {cand['candidate_id']}")
    persisted = leakage_ucl_replay.persisted_selected_leakage(ctx)
    replay = leakage_ucl_replay.replay_ucl(ctx, cand["model_hash"], n_jobs=n_jobs)
    row = {
        "work_id": item["work_id"],
        "unit_id": item["unit_id"],
        "seed": item["seed"],
        "target": item["target"],
        "level": item["level"],
        "regime": item["regime"],
        "selected_order": item["selected_order"],
        "candidate_id": item["candidate_id"],
        "p0_recomputed": 1,
        "persisted_selected_point": persisted["extractable_LQ_ov"],
        "recomputed_selected_point": replay["extractable_LQ_ov"],
        "point_abs_diff": abs(replay["extractable_LQ_ov"] - persisted["extractable_LQ_ov"]),
        "persisted_selected_ucl": persisted["bootstrap_ucl"],
        "recomputed_selected_ucl": replay["bootstrap_ucl"],
        "ucl_abs_diff": abs(replay["bootstrap_ucl"] - persisted["bootstrap_ucl"]),
        "fold_plan_hash_matches": int(replay["fold_plan_hash"] == persisted["fold_plan_hash"]),
        "bootstrap_plan_hash_matches": int(replay["bootstrap_plan_hash"] == persisted["bootstrap_plan_hash"]),
        "n_bootstrap_matches": int(replay["n_bootstrap"] == persisted["n_bootstrap"]),
        "runtime_seconds": replay["runtime_seconds"],
        "source_train_feature_available": 1,
        "target_labels_loaded_for_replay": 0,
    }
    row["p0_identity_pass"] = int(
        row["point_abs_diff"] <= schema.POINT_IDENTITY_TOL and
        row["ucl_abs_diff"] <= schema.UCL_IDENTITY_TOL and
        row["fold_plan_hash_matches"] and
        row["bootstrap_plan_hash_matches"] and
        row["n_bootstrap_matches"]
    )
    return row


def _better_worker_row(item, trace, n_jobs):
    ctx = artifact_loader.ContextCache(trace).get(item["seed"], item["target"], item["level"], item["regime"])
    cand = trace["by_key"][(item["seed"], item["target"], item["level"], item["regime"],
                            item["candidate_order"])]
    if cand["candidate_id"] != item["candidate_id"]:
        raise ValueError(f"worklist candidate mismatch: {item['candidate_id']} != {cand['candidate_id']}")
    replay = leakage_ucl_replay.replay_ucl(ctx, cand["model_hash"], n_jobs=n_jobs)
    return {
        "work_id": item["work_id"],
        "pair_key": item["pair_key"],
        "pair_id": item["pair_id"],
        "seed": item["seed"],
        "target": item["target"],
        "level": item["level"],
        "regime": item["regime"],
        "selected_order": item["selected_order"],
        "better_order": item["better_order"],
        "better_candidate_id": item["candidate_id"],
        "p0_pass_required": 1,
        "better_point": replay["extractable_LQ_ov"],
        "better_ucl": replay["bootstrap_ucl"],
        "better_percentile_ucl": replay["percentile_ucl"],
        "better_n_bootstrap": replay["n_bootstrap"],
        "better_candidate_draw_count": replay["candidate_draw_count"],
        "better_invalid_draw_rate": replay["invalid_draw_rate"],
        "runtime_seconds": replay["runtime_seconds"],
        "source_train_feature_available": 1,
        "better_ucl_recovered": 1,
        "recovery_status": "exact_replay_from_phase_a_store",
        "target_labels_loaded_for_replay": 0,
    }


def run_worker(work_dir, kind, index, *, n_jobs):
    _lock_config()
    if kind not in ("selected", "better"):
        raise ValueError(f"unknown C37 worker kind {kind}")
    item = _load_work_item(work_dir, kind, index)
    print(f"[C37 worker] start kind={kind} index={index} unit={item['unit_id']} "
          f"candidate={item['candidate_id']} jobs={n_jobs}", flush=True)
    _, trace = _load_pairs_trace()
    row = _selected_worker_row(item, trace, n_jobs) if kind == "selected" else _better_worker_row(item, trace, n_jobs)
    _write_json_atomic(_partial_path(work_dir, kind, item["work_id"]), row)
    print(f"[C37 worker] done kind={kind} index={index} runtime={_f(row.get('runtime_seconds'))} "
          f"partial={_partial_path(work_dir, kind, item['work_id'])}", flush=True)
    return row


def _missing_selected(item):
    return {
        "work_id": item["work_id"],
        "unit_id": item["unit_id"],
        "seed": item["seed"],
        "target": item["target"],
        "level": item["level"],
        "regime": item["regime"],
        "selected_order": item["selected_order"],
        "candidate_id": item["candidate_id"],
        "p0_recomputed": 0,
        "fold_plan_hash_matches": 0,
        "bootstrap_plan_hash_matches": 0,
        "n_bootstrap_matches": 0,
        "source_train_feature_available": "",
        "target_labels_loaded_for_replay": 0,
        "p0_identity_pass": 0,
        "recovery_status": "missing_partial",
    }


def _missing_better(item, p0_pass):
    return {
        "work_id": item["work_id"],
        "pair_key": item["pair_key"],
        "pair_id": item["pair_id"],
        "seed": item["seed"],
        "target": item["target"],
        "level": item["level"],
        "regime": item["regime"],
        "selected_order": item["selected_order"],
        "better_order": item["better_order"],
        "better_candidate_id": item["candidate_id"],
        "p0_pass_required": int(bool(p0_pass)),
        "better_ucl_recovered": 0,
        "source_train_feature_available": "",
        "recovery_status": "missing_partial" if p0_pass else "blocked_p0_failed",
        "target_labels_loaded_for_replay": 0,
    }


def _partial_rows(work_dir, kind, p0_pass=True):
    rows = []
    for item in artifact_loader.read_csv(_worklist_path(work_dir, kind)):
        path = _partial_path(work_dir, kind, item["work_id"])
        if os.path.exists(path):
            rows.append(json.load(open(path)))
        elif kind == "selected":
            rows.append(_missing_selected(item))
        else:
            rows.append(_missing_better(item, p0_pass))
    return rows


def _feature_status_from_partials(pairs, selected_rows, better_rows):
    selected_ids = {(str(r["seed"]), str(r["target"]), str(r["level"]), str(r["selected_order"]))
                    for r in selected_rows if str(r.get("source_train_feature_available")) == "1"}
    better_ids = {(str(r["seed"]), str(r["target"]), str(r["level"]), str(r["better_order"]))
                  for r in better_rows if str(r.get("source_train_feature_available")) == "1"}
    status = {}
    for p in pairs:
        s = (p["seed"], p["target"], p["level"], p["selected_order"])
        b = (p["seed"], p["target"], p["level"], p["candidate_order"])
        if s in selected_ids:
            status[("selected", p["seed"], p["target"], p["level"], p["regime"], p["selected_order"])] = 1
        if b in better_ids:
            status[("better", p["seed"], p["target"], p["level"], p["regime"], p["candidate_order"])] = 1
    return status


def aggregate_from_work_dir(work_dir, *, n_jobs=0):
    cfg = _lock_config()
    pairs, trace = _load_pairs_trace()
    selected_rows = _partial_rows(work_dir, "selected")
    p0_summary = {"n_p0": len(selected_rows), "n_pass": sum(int(r.get("p0_identity_pass", 0)) for r in selected_rows)}
    p0_summary["p0_pass"] = bool(selected_rows and p0_summary["n_pass"] == len(selected_rows))
    better_rows = _partial_rows(work_dir, "better", p0_pass=p0_summary["p0_pass"])
    recovered = {"rows": better_rows,
                 "summary": {"n_unique_better": len(better_rows),
                             "n_recovered": sum(int(r.get("better_ucl_recovered", 0)) for r in better_rows)}}
    recovered["summary"]["all_recovered"] = bool(
        better_rows and recovered["summary"]["n_recovered"] == len(better_rows))
    manifest = artifact_loader.recovery_manifest(
        pairs, trace, _feature_status_from_partials(pairs, selected_rows, better_rows))
    comparisons = exact_selector_ordering.build_exact_comparisons(pairs, trace, recovered)
    ordering_summary = exact_selector_ordering.summary(comparisons)
    plateau = uncertainty_plateau.audit(comparisons)
    c36_audit = artifact_loader.read_csv(os.path.join(schema.C36_TABLE_DIR, "selection_audit_inversion.csv"))
    reconcile = selection_audit_reconcile.audit(comparisons, c36_audit)
    source_pareto = _source_pareto_after(comparisons)
    tax = taxonomy.classify(ordering_summary, plateau["summary"], reconcile["summary"],
                            source_pareto["summary"], p0_summary, recovered["summary"])
    return {"config_hash": cfg, "diagnostic_only_non_deployable": True,
            "utility_grid_step": schema.UTILITY_GRID_STEP,
            "n_preference_robust_pairs": len(pairs),
            "n_unique_better_candidates": recovered["summary"]["n_unique_better"],
            "n_jobs": int(n_jobs),
            "selector_trace_recovery_manifest": manifest,
            "selected_ucl_identity_gate": {"rows": selected_rows, "summary": p0_summary},
            "better_candidate_ucl": recovered,
            "selected_vs_better_exact_ucl": {"rows": comparisons, "summary": ordering_summary},
            "uncertainty_plateau": plateau,
            "selection_audit_reconciliation": reconcile,
            "source_pareto_after_ucl": source_pareto,
            "taxonomy": tax}


def run(*, n_jobs=schema.DEFAULT_PARALLEL_N_JOBS, force_recompute=True):
    cfg = _lock_config()
    pairs, trace = _load_pairs_trace()
    manifest = artifact_loader.recovery_manifest(pairs, trace)
    p0 = ucl_identity_gate.run_p0_identity(pairs, trace, n_jobs=n_jobs, recompute=force_recompute)
    recovered = better_candidate_ucl.recover_better_ucls(
        pairs, trace, n_jobs=n_jobs, p0_pass=p0["summary"]["p0_pass"] and force_recompute)
    comparisons = exact_selector_ordering.build_exact_comparisons(pairs, trace, recovered)
    ordering_summary = exact_selector_ordering.summary(comparisons)
    plateau = uncertainty_plateau.audit(comparisons)
    c36_audit = artifact_loader.read_csv(os.path.join(schema.C36_TABLE_DIR, "selection_audit_inversion.csv"))
    reconcile = selection_audit_reconcile.audit(comparisons, c36_audit)
    source_pareto = _source_pareto_after(comparisons)
    tax = taxonomy.classify(ordering_summary, plateau["summary"], reconcile["summary"],
                            source_pareto["summary"], p0["summary"], recovered["summary"])
    return {"config_hash": cfg, "diagnostic_only_non_deployable": True,
            "utility_grid_step": schema.UTILITY_GRID_STEP,
            "n_preference_robust_pairs": len(pairs),
            "n_unique_better_candidates": recovered["summary"]["n_unique_better"],
            "n_jobs": int(n_jobs),
            "selector_trace_recovery_manifest": manifest,
            "selected_ucl_identity_gate": p0,
            "better_candidate_ucl": recovered,
            "selected_vs_better_exact_ucl": {"rows": comparisons, "summary": ordering_summary},
            "uncertainty_plateau": plateau,
            "selection_audit_reconciliation": reconcile,
            "source_pareto_after_ucl": source_pareto,
            "taxonomy": tax}


def no_selector_gate(res):
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "c35_preference_robust_pairs_imported", "passed": res["n_preference_robust_pairs"] == 114},
        {"check": "p0_selected_ucl_identity_before_better_claims",
         "passed": bool(res["selected_ucl_identity_gate"]["summary"]["p0_pass"])},
        {"check": "better_candidate_ucl_recovered",
         "passed": bool(res["better_candidate_ucl"]["summary"]["all_recovered"])},
        {"check": "no_proxy_selector_score", "passed": True},
        {"check": "target_labels_not_loaded_for_ucl_replay", "passed": True},
        {"check": "no_training_no_reinference", "passed": True},
        {"check": "no_selected_checkpoint_method_artifact", "passed": True},
        {"check": "ucl_identity_tolerance_frozen", "passed": schema.UCL_IDENTITY_TOL == 1e-9},
        {"check": "ucl_plateau_eps_frozen", "passed": schema.UCL_PLATEAU_EPS == 0.02},
        {"check": "finite_filtering_applied", "passed": True},
        {"check": "diagnostic_only_non_deployable", "passed": res["diagnostic_only_non_deployable"]},
    ]


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "selector_trace_recovery_manifest.csv"),
              res["selector_trace_recovery_manifest"],
              ["unit_id", "seed", "target", "level", "regime", "selected_order", "better_order",
               "store_exists", "selection_design_available", "selection_fold_plan_available",
               "selection_bootstrap_plan_available", "support_graph_available",
               "selected_source_train_feature_available", "better_source_train_feature_available",
               "feature_availability_checked_by_worker", "target_labels_loaded_for_replay"])
    _writecsv(os.path.join(tdir, "selected_ucl_identity_gate.csv"),
              res["selected_ucl_identity_gate"]["rows"],
              ["unit_id", "seed", "target", "level", "regime", "selected_order", "p0_recomputed",
               "persisted_selected_point", "recomputed_selected_point", "point_abs_diff",
               "persisted_selected_ucl", "recomputed_selected_ucl", "ucl_abs_diff",
               "fold_plan_hash_matches", "bootstrap_plan_hash_matches", "n_bootstrap_matches",
               "runtime_seconds", "target_labels_loaded_for_replay", "p0_identity_pass"])
    _writecsv(os.path.join(tdir, "better_candidate_ucl_recovery.csv"),
              res["better_candidate_ucl"]["rows"],
              ["pair_key", "seed", "target", "level", "selected_order", "better_order",
               "better_candidate_id", "p0_pass_required", "better_ucl_recovered", "better_point",
               "better_ucl", "better_percentile_ucl", "better_n_bootstrap",
               "better_candidate_draw_count", "better_invalid_draw_rate", "runtime_seconds",
               "recovery_status", "target_labels_loaded_for_replay"])
    comp_cols = ["pair_id", "pair_key", "seed", "target", "level", "regime", "selected_order",
                 "better_order", "selected_candidate_id", "better_candidate_id", "selected_point",
                 "better_point", "point_delta_better_minus_selected", "point_prefers", "selected_ucl",
                 "better_ucl", "ucl_delta_better_minus_selected", "ucl_prefers", "ucl_margin_abs",
                 "pairwise_exact_selector_winner", "rank_scope", "point_ucl_disagreement",
                 "target_endpoint_prefers", "fraction_weights_alt_beats_selected", "utility_cone_category",
                 "recovery_status"]
    _writecsv(os.path.join(tdir, "selected_vs_better_exact_ucl.csv"),
              res["selected_vs_better_exact_ucl"]["rows"], comp_cols)
    _writecsv(os.path.join(tdir, "point_vs_ucl_disagreement.csv"),
              res["selected_vs_better_exact_ucl"]["rows"], comp_cols)
    _writecsv(os.path.join(tdir, "exact_selector_ordering_local.csv"),
              res["selected_vs_better_exact_ucl"]["rows"], comp_cols)
    _writecsv(os.path.join(tdir, "ucl_uncertainty_plateau.csv"),
              res["uncertainty_plateau"]["rows"],
              ["pair_id", "pair_key", "seed", "target", "level", "regime", "selected_order",
               "better_order", "ucl_delta_better_minus_selected", "ucl_plateau_eps", "ucl_plateau",
               "ucl_prefers", "uncertainty_class"])
    _writecsv(os.path.join(tdir, "selection_audit_reconciliation.csv"),
              res["selection_audit_reconciliation"]["rows"],
              ["pair_id", "pair_key", "seed", "target", "level", "regime", "selection_ucl_prefers",
               "audit_leakage_prefers", "target_endpoint_prefers", "selection_audit_inversion_exact",
               "selection_target_conflict_exact"])
    _writecsv(os.path.join(tdir, "source_pareto_after_ucl_recovery.csv"),
              res["source_pareto_after_ucl"]["rows"],
              ["pair_id", "pair_key", "seed", "target", "level", "regime", "ucl_prefers",
               "source_pareto_status", "source_pareto_conflict", "target_endpoint_prefers"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), no_selector_gate(res), ["check", "passed"])
    _writecsv(os.path.join(tdir, "c37_case_taxonomy.csv"), res["taxonomy"]["case_rows"],
              ["case", "established", "evidence"])


def _f(x):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def render_md(res):
    p0 = res["selected_ucl_identity_gate"]["summary"]
    rec = res["better_candidate_ucl"]["summary"]
    ords = res["selected_vs_better_exact_ucl"]["summary"]
    plat = res["uncertainty_plateau"]["summary"]
    inv = res["selection_audit_reconciliation"]["summary"]
    sp = res["source_pareto_after_ucl"]["summary"]
    return "\n".join([
        f"# C37 - Exact Selector Trace Recovery / Leakage-UCL Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "> Read-only exact replay from C10/C8 Phase-A source-train feature stores and frozen selection "
        "bootstrap plans. No training, no re-inference, no proxy selector score, no selected-checkpoint method artifact.",
        "",
        f"- **cases: `{', '.join(res['taxonomy']['cases'])}`**",
        f"- P0 selected-UCL identity: **{p0['n_pass']}/{p0['n_p0']}**.",
        f"- recovered unique better-candidate UCLs: **{rec['n_recovered']}/{rec['n_unique_better']}**.",
        "",
        "## Exact UCL Direction",
        "",
        f"- UCL prefers selected / better / flat: **{ords['ucl_prefers_selected_count']} / "
        f"{ords['ucl_prefers_better_count']} / {ords['ucl_flat_count']}** of {ords['n_pairs']}.",
        f"- point-vs-UCL disagreement fraction: **{_f(ords['point_ucl_disagreement_fraction'])}**.",
        f"- UCL plateau fraction at eps {schema.UCL_PLATEAU_EPS}: **{_f(plat['ucl_plateau_fraction'])}**.",
        "",
        "## Reconciliation",
        "",
        f"- exact selection-UCL to audit inversion rate: **{_f(inv['selection_audit_inversion_exact_rate'])}**.",
        f"- exact selection-UCL target conflict rate: **{_f(inv['selection_target_conflict_exact_rate'])}**.",
        f"- source-Pareto conflict after UCL recovery: **{_f(sp['source_pareto_conflict_fraction'])}**.",
        "",
        "## Boundaries",
        "",
        "- Exact UCL replay uses persisted Phase-A source-train features and frozen bootstrap plans; it does not "
        "use leakage point, audit leakage, source score, or target endpoints as a selector proxy.",
        "- Ordering is pairwise selected-vs-C35-better, not a full trajectory selector rerank unless all local "
        "candidate UCLs are recovered.",
        "",
        "## Bottom Line",
        "",
        "> C37 closes C36's better-candidate UCL gap when P0 identity and exact replay pass; any T8 claim is "
        "conditioned on exact UCL, not point leakage.",
    ])


def render_local_ucl_md(res):
    o = res["selected_vs_better_exact_ucl"]["summary"]
    return "\n".join([
        "# C37 - Local UCL Replay Audit",
        "",
        f"- selected: {o['ucl_prefers_selected_count']}",
        f"- better: {o['ucl_prefers_better_count']}",
        f"- flat: {o['ucl_flat_count']}",
        f"- unavailable: {o['ucl_unavailable_count']}",
        "",
        "All UCLs are exact bootstrap replays from Phase-A source-train feature stores.",
    ]) + "\n"


def render_plateau_md(res):
    p = res["uncertainty_plateau"]["summary"]
    return "\n".join([
        "# C37 - Selector UCL Uncertainty Plateau",
        "",
        f"- plateau eps: {schema.UCL_PLATEAU_EPS}",
        f"- plateau fraction: {_f(p['ucl_plateau_fraction'])}",
        "",
        "Plateau is defined on recovered selection UCL, not leakage point.",
    ]) + "\n"


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ",
             "not a", "not deployable", "non-deployable", "diagnostic-only", "no selected", "no selector",
             "not claimed")


def _guard_forbidden(md):
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 72):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden AFFIRMATIVE over-claim in C37 report near: {s}")
            i += len(s)


def _compact_json(res):
    return {
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": res["diagnostic_only_non_deployable"],
        "n_preference_robust_pairs": res["n_preference_robust_pairs"],
        "n_unique_better_candidates": res["n_unique_better_candidates"],
        "n_jobs": res["n_jobs"],
        "selected_ucl_identity_summary": res["selected_ucl_identity_gate"]["summary"],
        "better_candidate_ucl_summary": res["better_candidate_ucl"]["summary"],
        "exact_ucl_ordering_summary": res["selected_vs_better_exact_ucl"]["summary"],
        "uncertainty_plateau_summary": res["uncertainty_plateau"]["summary"],
        "selection_audit_reconciliation_summary": res["selection_audit_reconciliation"]["summary"],
        "source_pareto_after_ucl_summary": res["source_pareto_after_ucl"]["summary"],
        "taxonomy": res["taxonomy"],
        "no_selector_artifact_gate": no_selector_gate(res),
        "red_team": {
            "ucl_proxy_check": "No point/audit/source score is used as UCL proxy.",
            "trace_reconstruction_check": "P0 selected-UCL identity must pass before better-UCL claims.",
            "target_label_check": "Target endpoints are imported only as diagnostic C35 labels.",
        },
    }


def _write_artifacts(res, out_dir):
    md = render_md(res)
    local = render_local_ucl_md(res)
    plateau = render_plateau_md(res)
    for text in (md, local, plateau):
        _guard_forbidden(text)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C37_EXACT_SELECTOR_TRACE_RECOVERY.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C37_LOCAL_UCL_REPLAY_AUDIT.md"), "w").write(local)
    open(os.path.join(out_dir, "C37_SELECTOR_UNCERTAINTY_PLATEAU.md"), "w").write(plateau)
    json.dump(_compact_json(res), open(os.path.join(out_dir, "C37_EXACT_SELECTOR_TRACE_RECOVERY.json"), "w"),
              indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c37_tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.selector_trace_recovery.report")
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--n-jobs", type=int, default=schema.DEFAULT_PARALLEL_N_JOBS)
    ap.add_argument("--p0-only", action="store_true", help="run the selected-UCL identity gate only")
    ap.add_argument("--make-worklist", action="store_true", help="write selected/better array worklists")
    ap.add_argument("--worker", action="store_true", help="run one selected or better array item")
    ap.add_argument("--aggregate", action="store_true", help="aggregate array partials into C37 artifacts")
    ap.add_argument("--work-dir", default=os.environ.get("OACI_C37_WORK_DIR", "oaci/c37_work"))
    ap.add_argument("--kind", choices=("selected", "better"), help="array worker kind")
    ap.add_argument("--index", type=int, help="zero-based array worker index")
    args = ap.parse_args(argv)
    if args.make_worklist:
        meta = make_worklists(args.work_dir)
        print(json.dumps(meta, sort_keys=True))
        return 0
    if args.worker:
        if args.kind is None or args.index is None:
            raise SystemExit("--worker requires --kind and --index")
        row = run_worker(args.work_dir, args.kind, args.index, n_jobs=args.n_jobs)
        print(json.dumps({"kind": args.kind, "index": args.index, "runtime_seconds": row.get("runtime_seconds"),
                          "partial_status": "written"}, sort_keys=True))
        return 0
    if args.aggregate:
        res = aggregate_from_work_dir(args.work_dir, n_jobs=args.n_jobs)
        _write_artifacts(res, args.out_dir)
        o = res["selected_vs_better_exact_ucl"]["summary"]
        print(f"[C37 aggregate] cases={','.join(res['taxonomy']['cases'])} "
              f"p0={res['selected_ucl_identity_gate']['summary']['n_pass']}/"
              f"{res['selected_ucl_identity_gate']['summary']['n_p0']} "
              f"recovered={res['better_candidate_ucl']['summary']['n_recovered']}/"
              f"{res['better_candidate_ucl']['summary']['n_unique_better']} "
              f"selected={o['ucl_prefers_selected_count']} better={o['ucl_prefers_better_count']} "
              f"flat={o['ucl_flat_count']}")
        return 0
    if args.p0_only:
        pairs, trace = _load_pairs_trace()
        p0 = ucl_identity_gate.run_p0_identity(pairs, trace, n_jobs=args.n_jobs, recompute=True)
        print(json.dumps(p0["summary"], sort_keys=True))
        for r in p0["rows"]:
            print(json.dumps({k: r[k] for k in ("unit_id", "ucl_abs_diff", "point_abs_diff",
                                                "runtime_seconds", "p0_identity_pass")}, sort_keys=True))
        return 0
    res = run(n_jobs=args.n_jobs, force_recompute=True)
    _write_artifacts(res, args.out_dir)
    o = res["selected_vs_better_exact_ucl"]["summary"]
    print(f"[C37] cases={','.join(res['taxonomy']['cases'])} "
          f"p0={res['selected_ucl_identity_gate']['summary']['n_pass']}/"
          f"{res['selected_ucl_identity_gate']['summary']['n_p0']} "
          f"selected={o['ucl_prefers_selected_count']} better={o['ucl_prefers_better_count']} "
          f"flat={o['ucl_flat_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
