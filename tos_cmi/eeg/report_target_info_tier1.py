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


def main():
    plan = json.load(open("%s/target_info_tier1_plan.json" % OUT))["plan"]
    schema = json.load(open("%s/target_info_tier1_schema.json" % OUT))
    from tos_cmi.eeg.run_target_info_tier1_smoke import load_cfg
    cfg = load_cfg()
    p = write_report(cfg, plan, schema, "TARGET_LEAK_STRUCTURAL_PASS", OUT)
    print("report ->", p)


if __name__ == "__main__":
    main()
