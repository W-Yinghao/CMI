"""Fork 1 Tier-1 --- dry-run report writer. Renders the task plan + split schema into a human-readable report.
NO result metrics, NO ΔbAcc, NO target performance --- this is a planning artifact only (implementation stage).

  python -m tos_cmi.eeg.report_target_info_tier1        # reads the dry-run plan/schema and (re)writes the report
"""
from __future__ import annotations
import json
import os
from collections import Counter

OUT = "tos_cmi/results/target_info/tier1_driver_dryrun"


def write_report(cfg, plan, schema, token, out=OUT, expanded=None):
    sc = cfg["tier1_scope"]
    by_budget = Counter(r["budget"] for r in plan)
    la = schema["label_access"]
    cp = schema.get("code_path_access", {})
    exp = schema.get("expansion", {})
    n_folds = 5 if sc["folds"] == "first5" else int(sc["folds"])
    L = ["# Fork 1 Tier-1 smoke --- DRY-RUN plan (NO experiments; NO metrics)\n",
         "Implementation stage: `experiments_allowed=%s`, `runs_allowed=%s`, design_lock_hash=`%s`."
         % (cfg["experiments_allowed"], cfg["runs_allowed"], cfg["design_lock_hash"]),
         "Structural gate: **%s**.\n" % token,
         "## Scope",
         "- datasets: %s" % ", ".join(sc["datasets"]),
         "- backbones: %s" % ", ".join(sc["backbones"]),
         "- worlds: %s" % ", ".join(cfg["worlds"]),
         "- seeds: %s ; folds: %s ; target split repeats R = %s" % (sc["seeds"], sc["folds"], sc["repeats_R"]),
         "- budgets: %s" % ", ".join(cfg["budgets"].keys()),
         "- k-grid (B2/B3): %s" % cfg["budgets"]["B2_k_labels_per_class"]["k_grid"],
         "- world_alpha_grid (inner loop): %s" % cfg.get("world_alpha_grid"),
         "- interventions: %s\n" % ", ".join(cfg["interventions"]),
         "## Task plan",
         "- **plan rows = %d**" % len(plan),
         "  (= datasets %d x backbones %d x worlds %d x folds %d x budgets %d x interventions %d)"
         % (len(sc["datasets"]), len(sc["backbones"]), len(cfg["worlds"]),
            n_folds, len(cfg["budgets"]), len(cfg["interventions"])),
         "- per budget (plan rows): %s" % dict(by_budget),
         "- **expanded executable tasks = %s**" % exp.get("expanded_tasks"),
         "  expansion rule: %s" % exp.get("rule"),
         "  expanded per budget family: %s" % exp.get("expanded_by_budget_family"),
         "- inner-loop multipliers: world_alpha_grid %s, k_grid %s, R %s\n"
         % (schema["inner_loops"]["world_alpha_grid"], schema["inner_loops"]["k_grid"],
            schema["inner_loops"]["repeats_R"]),
         "## Calibration / audit split policy",
         "- stratified by class; R = %s repeats; calibration ∩ audit = ∅ (enforced by %s)"
         % (sc["repeats_R"], token),
         "- k UNAVAILABLE policy: %s" % schema["split_policy"]["k_unavailable_policy"],
         "- calibration used by: %s" % ", ".join(schema["split_policy"]["calibration_used_by"]),
         "- audit used by: %s\n" % ", ".join(schema["split_policy"]["audit_used_by"]),
         "## Label-access capability matrix",
         "- source labels -> %s" % ", ".join(la["source_labels"]),
         "- target CALIBRATION labels -> %s" % ", ".join(la["target_calibration_labels"]),
         "- target AUDIT labels -> %s" % ", ".join(la["target_audit_labels"]),
         "- B4 oracle labels -> %s\n" % ", ".join(la["B4_oracle_labels"]),
         "## Code-path label access (which function may read which labels)",
         "- `compute_decision_row` may read: %s" % ", ".join(cp.get("compute_decision_row", [])),
         "- `compute_decision_row` MUST NOT read: %s" % ", ".join(cp.get("compute_decision_row_forbidden", [])),
         "- `calibration_delta_bacc` may read: %s" % ", ".join(cp.get("calibration_delta_bacc", [])),
         "- `audit_scalar` may read: %s\n" % ", ".join(cp.get("audit_scalar", [])),
         "## Expected real-run output schema (for reference; NOT produced now)",
         "- decision rows: %s" % ", ".join(schema["expected_run_output_schema"]["decision_rows"]),
         "- audit reported separately: %s" % ", ".join(schema["expected_run_output_schema"]["audit_reported_separately"]),
         "- accounting: %s" % ", ".join(schema["expected_run_output_schema"]["accounting"]),
         "- B4 excluded from: %s\n" % ", ".join(schema["expected_run_output_schema"]["b4_excluded_from"]),
         "## Hard gates",
         "- B1 accept forbidden: %s ; B4 diagnostic-only: %s" % (schema["b1_accept_forbidden"],
                                                                 schema["b4_diagnostic_only"]),
         "- gates: %s\n" % ", ".join(schema["hard_gates"]),
         "_No result metrics are produced at this stage. Running Tier-1 requires a separate PM go._"]
    os.makedirs(out, exist_ok=True)
    p = "%s/target_info_tier1_dryrun_report.md" % out
    open(p, "w").write("\n".join(L) + "\n")
    return p


def write_preflight_report(manifest_out, split_hash_rows, unavail_rows, out):
    """Render the real-split preflight report. Split-only: trial/class counts, k-availability, disjointness, hash
    summaries. Emits NO ΔbAcc / NLL / accept-reject / gate action / performance metric."""
    from collections import Counter
    all_disjoint = all(r["disjoint"] for r in split_hash_rows)
    per_fold = {}
    for r in split_hash_rows:
        per_fold.setdefault((r["dataset"], r["fold"], r["target_subject"]),
                            (r["n_calibration"], r["n_audit"]))
    L = ["# Fork 1 Tier-1 --- REAL split preflight (split/hash/unavailable-k ONLY; NO metrics)\n",
         "Purpose: %s. No metrics emitted: %s." % (manifest_out["purpose"], manifest_out["no_metrics_emitted"]),
         "split_rng_scheme: **%s** (global_split_seed %s, calib_fraction %s).\n"
         % (manifest_out.get("split_rng_scheme"), manifest_out.get("global_split_seed"),
            manifest_out.get("calib_fraction")),
         "## Datasets / folds checked",
         "- datasets: %s ; backbones: %s ; seed: %s ; folds: %s"
         % (", ".join(manifest_out["datasets"]), ", ".join(manifest_out["backbones"]),
            manifest_out["seed"], manifest_out["folds"]),
         "- dumps checked: %d ; R = %d ; k-grid = %s\n" % (manifest_out["n_dumps"], manifest_out["R"],
                                                           manifest_out["k_grid"]),
         "## Target trial counts per class (per held-out target subject)"]
    for r in [x for x in manifest_out["rows"] if x["split_id"] == 1 and x["k"] == manifest_out["k_grid"][0]]:
        L.append("- %s fold %d (subj %d) class %d: n_target_total = %d"
                 % (r["dataset"], r["fold"], r["target_subject"], r["class"], r["n_target_total"]))
    L += ["", "## Calibration / audit split summary",
          "- calibration+audit index disjoint on ALL %d splits: %s" % (len(split_hash_rows), all_disjoint),
          "- total calibration∩audit overlap across all splits: %d" % manifest_out["calibration_audit_overlap_total"],
          "- n(calibration,audit) per (dataset,fold): %s"
          % {("%s/f%d/s%d" % (k[0], k[1], k[2])): v for k, v in sorted(per_fold.items())}]
    div = manifest_out.get("per_subject_split_diversity", [])
    if div:
        divset = sorted(set(d["distinct_calibration_splits"] for d in div))
        L += ["", "## Per-subject split diversity",
              "- distinct calibration splits per target subject (want = R = %d): values seen = %s over %d subjects"
              % (manifest_out["R"], divset, len(div)),
              "- min %d / max %d distinct splits per subject"
              % (min(d["distinct_calibration_splits"] for d in div),
                 max(d["distinct_calibration_splits"] for d in div))]
    L += ["", "## k availability + nested-k check",
          "- schema rows (dataset x fold x split x k x class): %d" % manifest_out["n_schema_rows"],
          "- UNAVAILABLE (dataset,fold,split,k) entries: %d" % manifest_out["n_unavailable_k"],
          "- nested-k subset checks passed (k=1 subset of k=2 ... subset of k=max, all within calibration pool): %s"
          % manifest_out.get("nested_k_checks_passed")]
    if unavail_rows:
        byk = Counter((u["dataset"], u["k"]) for u in unavail_rows)
        L.append("- unavailable-by (dataset,k): %s" % {("%s,k%d" % kk): n for kk, n in sorted(byk.items())})
    else:
        L.append("- all requested k available on every split (no UNAVAILABLE entries)")
    L += ["", "## Hash summaries",
          "- distinct calibration_idx_hash: %d ; distinct audit_idx_hash: %d"
          % (len({r["calibration_idx_hash"] for r in split_hash_rows}),
             len({r["audit_idx_hash"] for r in split_hash_rows})),
          "- distinct calibration_label_hash: %d ; distinct audit_label_hash: %d"
          % (len({r["calibration_label_hash"] for r in split_hash_rows}),
             len({r["audit_label_hash"] for r in split_hash_rows})),
          "", "## Confirmation",
          "- NO estimator and NO intervention was run: this preflight produced only calibration/audit splits, "
          "per-class trial counts, k-availability, and index/label hashes. No predictive quality, no benefit, no "
          "likelihood, and no decision of any kind was computed. Split, count, availability, and hash provenance "
          "only.\n"]
    p = "%s/preflight_report.md" % out
    open(p, "w").write("\n".join(L) + "\n")
    return p


def write_provider_validation_report(summary, out):
    """Render the provider-validation report: pipeline exercised, shapes, contexts, schema presence, hash counts.
    NO metric VALUES (bAcc/ΔbAcc/NLL/accept-rate/...) and no metric field names -- only counts of redacted fields."""
    L = ["# Fork 1 Tier-1 --- provider-validation (plumbing check; metrics REDACTED; NOT a science result)\n",
         "Exercises `_real_provider` on ONE real dump. Metric code ran internally; every metric VALUE is redacted "
         "from this output. metrics_computed_internally=%s, metrics_redacted=%s.\n"
         % (summary["metrics_computed_internally"], summary["metrics_redacted"]),
         "## Scope (one dump only)",
         "- %s" % summary["scope"],
         "- source_shape %s ; target_shape %s\n" % (summary["source_shape"], summary["target_shape"]),
         "## Pipeline exercised",
         "- decision rows completed: %d ; audit rows completed: %d" % (summary["rows_completed"],
                                                                       summary["audit_rows_completed"]),
         "- contexts constructed: %s" % ", ".join(summary["contexts_constructed"]),
         "- decision_row_schema_present: %s ; audit_row_schema_present: %s"
         % (summary["decision_row_schema_present"], summary["audit_row_schema_present"]),
         "- distinct calibration_idx_hashes: %d\n" % summary["distinct_calibration_idx_hashes"],
         "## Schema (safe field names + redacted metric-field counts)",
         "- decision safe fields: %s" % summary["decision_row_safe_fields"],
         "- decision redacted metric fields (count only): %d" % summary["decision_row_redacted_metric_fields"],
         "- audit safe fields: %s" % summary["audit_row_safe_fields"],
         "- audit redacted metric fields (count only): %d\n" % summary["audit_row_redacted_metric_fields"],
         "## Confirmation",
         "- This is a plumbing/provider check. NO balanced-metric, NO benefit, NO likelihood, NO decision-rate, and "
         "NO source/target quality VALUE was written. The real-dump load path was exercised; its outputs are "
         "redacted. This is NOT a Tier-1 science result.\n"]
    p = "%s/provider_validation_report.md" % out
    open(p, "w").write("\n".join(L) + "\n")
    return p


def write_smoke_report(summary, fig, out, prefix="target_info_tier1_smoke"):
    """Tier-1 smoke / budget-frontier report: deployable budgets (B0/B1/B2/B3) vs diagnostic (B4), B2 k-curve
    (accept / true / false / audit ΔbAcc / bounded-LCB / specificity), B3 label budget, oracle gap, sample-complexity
    thresholds, and the stop-condition audit."""
    sc = summary["scope"]
    L = ["# Fork 1 Tier-1 smoke --- target-information budget curves (semi-synthetic; NOT a final paper claim)\n",
         "Scope: %s x %s x folds %s x worlds %s x budgets %s ; k=%s ; R=%s ; alpha=%s ; split=%s ; n_boot=%s."
         % (sc["datasets"], sc["backbones"], sc["folds"], sc["worlds"], sc["budgets"], sc["k_grid"], sc["R"],
            sc["world_alpha_grid"], sc["split_rng_scheme"], sc["n_boot"]),
         "Decision rows %d ; audit rows %d ; workers %d ; failures %d.\n"
         % (summary["n_decision_rows"], summary["n_audit_rows"], summary.get("n_workers"), summary["n_failures"]),
         "## Stop-condition audit (all must be 0)",
         "```", "\n".join("%-26s %d" % (k, v) for k, v in summary["stop_conditions"].items()), "```", "",
         "## Per-budget action counts (deployable B0/B1/B2/B3 ; diagnostic B4)"]
    for b in sorted(summary["per_budget"]):
        L.append("- %-26s %s" % (b, summary["per_budget"][b]))
    L += ["", "## Deployable (B2+B3) safety summary",
          "- deployable accepts: %d ; false accepts (audit<=0): %d ; harmful (audit<-0.01): %d ; false-accept rate %.3f"
          % (summary["n_deployable_accepts"], summary["n_deployable_false_accepts"],
             summary["n_deployable_harmful_accepts"], summary["deployable_false_accept_rate"]),
          "", "## Sample-complexity thresholds (per world)",
          "```"]
    for w, scx in sorted(summary.get("sample_complexity", {}).items()):
        L.append("%-40s min_k_true_accept=%s  min_k_false<=5%%=%s  any_accept_at_max_k=%s  best_cal_LCB=%s (thr %s)"
                 % (w[:40], scx["min_k_any_true_accept"], scx["min_k_false_rate_le_5pct"],
                    scx["any_accept_at_max_k"],
                    ("%.3f" % scx["best_cal_lcb_over_all_k"]) if scx["best_cal_lcb_over_all_k"] is not None else "n/a",
                    scx["benefit_lcb_threshold"]))
    L += ["```", "",
          "## B2 k-curve (per world): accept rate, true/false/harmful, audit ΔbAcc, bounded cal-LCB, specificity",
          "| world | k | n | acc_rate | true | false | harm | audit_ΔbAcc | cal_LCB_max | spec_cal | spec_aud |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary["b2_k_curve"]:
        L.append("| %s | %s | %d | %.2f | %d | %d | %d | %s | %s | %d | %d |"
                 % (s["world"][:22], s["k"], s["n"], s["accept_rate"], s["true_accept"], s["false_accept"],
                    s["harmful_accept"],
                    ("%.3f" % s["mean_audit_dbacc_accepted"]) if s["mean_audit_dbacc_accepted"] is not None else "n/a",
                    ("%.3f" % s["cal_lcb_max"]) if s.get("cal_lcb_max") is not None else "n/a",
                    s["specific_calibration"], s["specific_audit"]))
    L += ["", "## B3 sequential calibration (hardened bounded LCB)",
          "- actions: %s" % summary["b3_actions"],
          "- accepts: %d ; false accepts: %d ; k=1 accepts (must be 0): %d ; mean label budget (accepted): %s"
          % (summary["b3_accepts"], summary["b3_false_accepts"], summary["b3_k1_accepts"],
             summary["b3_label_budget_mean"]),
          "", "## B4 oracle diagnostic (upper bound; excluded from deployable accept counts)"]
    for w, o in sorted(summary.get("b4_oracle_by_world", {}).items()):
        L.append("- %s: oracle audit ΔbAcc mean %.3f / max %.3f over %d cells"
                 % (w[:30], o["mean_audit_dbacc"], o["max_audit_dbacc"], o["n"]))
    L += ["", "Budget curve: `%s`" % fig,
          "", "## Reading guide",
          "- B0 source-only expected to abstain/reject on source-invisible benefit; B1 must NEVER accept (stop-cond).",
          "- B2/B3 accept is the target-information signal: SAFE only if held-out audit ΔbAcc > +0.01 AND same-k "
          "random does not reproduce it (accepted_specific). accepted_non_specific / false_accept are disclosed.",
          "- Many abstains at small k are EXPECTED (weak calibration LCB), not a failure.\n"]
    p = "%s/%s_report.md" % (out, prefix)
    open(p, "w").write("\n".join(L) + "\n")
    return p


def main():
    plan = json.load(open("%s/target_info_tier1_plan.json" % OUT))["plan"]
    schema = json.load(open("%s/target_info_tier1_schema.json" % OUT))
    from tos_cmi.eeg.run_target_info_tier1_smoke import load_cfg
    cfg = load_cfg()
    p = write_report(cfg, plan, schema, "TARGET_LEAK_STRUCTURAL_PASS", OUT)
    print("report ->", p)


if __name__ == "__main__":
    main()
