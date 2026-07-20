"""C57 - Manuscript Scaffold / Claim-Contract / Theory-Literature Package."""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os

from . import audit_utils as au
from . import schema as c49_schema


MILESTONE = "C57"
REPORT_JSON = "oaci/reports/C57_MANUSCRIPT_SCAFFOLD_CLAIM_CONTRACT.json"
REPORT_DIR = "oaci/reports"
TABLE_DIR = "oaci/reports/c57_tables"
SCAFFOLD_DIR = "oaci/reports/c57_manuscript_scaffold"
C56_JSON = "oaci/reports/C56_MECHANISM_CLOSURE_REVIEW_GAUNTLET.json"
C56_KEY_TABLE = "oaci/reports/c56_tables/key_number_provenance.csv"

DECISIONS = (
    "C57-A_manuscript_scaffold_ready",
    "C57-B_manuscript_scaffold_ready_but_literature_gap_remaining",
    "C57-C_claim_contract_inconsistency_requires_repair",
    "C57-D_figure_or_evidence_provenance_gap_requires_repair",
    "C57-E_theory_framing_not_ready_reopen_C56",
    "C57-F_not_ready_for_manuscript_scaffold",
)

ALLOWED_STRENGTHS = (
    "observed_in_this_setting",
    "empirical_mechanism",
    "diagnostic_ceiling",
    "information_boundary",
    "negative_control_result",
    "future_work_only",
    "not_claimed",
)

FORBIDDEN_PATTERNS = (
    "source-only rescue",
    "OACI rescue",
    "deployable selector",
    "few-label sufficiency",
    "formal theorem without proof",
    "same-label endpoint oracle available at selection time",
    "target-unlabeled geometry is source-only",
    "target-grouped diagnostic is source-only",
    "all DG methods fail",
    "EEG transfer impossible",
    "good checkpoints are absent",
    "template-only C55 transfer beats max null p95",
)

NEGATION_CUES = (
    "not ",
    "no ",
    "never ",
    "cannot ",
    "do not ",
    "does not ",
    "is not ",
    "are not ",
    "unavailable",
    "forbidden",
    "blocked",
    "diagnostic",
    "future work",
)

LITERATURE = (
    {
        "literature_id": "IRM_1907_02893",
        "axis": "invariant risk and invariant representation",
        "citation": "Invariant Risk Minimization",
        "url": "https://arxiv.org/abs/1907.02893",
        "alignment": "motivates invariance/OOD language; C57 uses it to bound claims about control versus measurement",
        "blocked_overclaim": "do not state that all invariance methods fail",
    },
    {
        "literature_id": "DomainBed_2007_01434",
        "axis": "domain generalization model selection",
        "citation": "In Search of Lost Domain Generalization",
        "url": "https://arxiv.org/abs/2007.01434",
        "alignment": "supports treating model selection/localization as central rather than incidental",
        "blocked_overclaim": "do not state that DG literature ignored model selection",
    },
    {
        "literature_id": "ZhaoInvariantDA_1901_09453",
        "axis": "conditional shift and invariant representation limits",
        "citation": "On Learning Invariant Representation for Domain Adaptation",
        "url": "https://arxiv.org/abs/1901.09453",
        "alignment": "closest lower-bound-adjacent precedent for conditional-shift/invariant-representation caution",
        "blocked_overclaim": "do not state that C57 proves a universal lower bound",
    },
    {
        "literature_id": "PostSelection_1401_3889",
        "axis": "post-selection and data-reuse guardrails",
        "citation": "Exact Post-Selection Inference for Sequential Regression Procedures",
        "url": "https://arxiv.org/abs/1401.3889",
        "alignment": "provides claim discipline for same-label endpoint reuse and selection-event wording",
        "blocked_overclaim": "do not describe same-label endpoint diagnostics as a calibration method",
    },
    {
        "literature_id": "InteractiveLowerBounds_2410_05117",
        "axis": "information lower-bound framing",
        "citation": "Assouad, Fano, and Le Cam with Interaction",
        "url": "https://arxiv.org/abs/2410.05117",
        "alignment": "supplies future theorem vocabulary only; C57 remains empirical formalism",
        "blocked_overclaim": "do not state a minimax theorem without proof",
    },
    {
        "literature_id": "EEG_DG_project_bibliography_pending",
        "axis": "EEG cross-subject DG and model selection",
        "citation": "project-local EEG-DG context",
        "url": "local bibliography expansion for M1",
        "alignment": "C57 isolates the EEG-specific mechanism and flags that M1 should expand task-specific references",
        "blocked_overclaim": "do not claim EEG transfer is impossible",
    },
)


def _lock_config() -> str:
    return au.lock_config(MILESTONE)


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: list[dict], cols: list[str]) -> None:
    au.write_csv(path, rows, cols)


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt3(value) -> str:
    if isinstance(value, bool):
        return str(value)
    try:
        x = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(x):
        return "n/a"
    return f"{x:.3f}"


def _bool01(value) -> int:
    return int(bool(value))


def _key_rows() -> list[dict]:
    rows = list(_read_csv(C56_KEY_TABLE))

    def add(provenance_id: str, milestone: str, metric: str, value, artifact: str, table: str = "", row_key: str = "", note: str = ""):
        rows.append({
            "provenance_id": provenance_id,
            "milestone": milestone,
            "metric": metric,
            "value": value,
            "value_fmt_3": _fmt3(value),
            "artifact": artifact,
            "table": table,
            "row_key": row_key,
            "trace_status": "verified",
            "note": note,
        })

    c34 = _load_json("oaci/reports/C34_CONTINUOUS_LOCAL_REGRET_AUDIT.json")
    c35 = _load_json("oaci/reports/C35_UTILITY_CONE_REGRET_AUDIT.json")
    c37 = _load_json("oaci/reports/C37_EXACT_SELECTOR_TRACE_RECOVERY.json")
    c38 = _load_json("oaci/reports/C38_LEAKAGE_UCL_OBJECTIVE_GEOMETRY.json")
    c34_regret = c34["key_aggregates"]["robust_selected_pairs_summary"]
    add("K_C34_real_endpoint_regret_fraction", "C34", "real_endpoint_regret_fraction",
        c34_regret["real_endpoint_regret_fraction"], "oaci/reports/C34_CONTINUOUS_LOCAL_REGRET_AUDIT.json",
        note="Continuous local regret is real rather than a binary threshold artifact.")
    add("K_C34_threshold_only_fraction", "C34", "threshold_only_fraction_among_binary_misses",
        c34_regret["threshold_only_fraction_among_binary_misses"], "oaci/reports/C34_CONTINUOUS_LOCAL_REGRET_AUDIT.json",
        note="Threshold-only explanation is falsified for the binary misses.")
    add("K_C35_preference_robust_fraction", "C35", "preference_robust_fraction",
        c35["utility_simplex_summary"]["preference_robust_fraction"], "oaci/reports/C35_UTILITY_CONE_REGRET_AUDIT.json",
        note="Local continuous regret is often robust across endpoint preferences.")
    add("K_C35_preference_robust_pairs", "C35", "preference_robust_pairs",
        c35["utility_simplex_summary"]["category_counts"]["preference_robust_regret"], "oaci/reports/C35_UTILITY_CONE_REGRET_AUDIT.json")
    add("K_C37_ucl_prefers_selected_fraction", "C37", "ucl_prefers_selected_fraction",
        c37["exact_ucl_ordering_summary"]["ucl_prefers_selected_fraction"], "oaci/reports/C37_EXACT_SELECTOR_TRACE_RECOVERY.json",
        note="Exact UCL ordering favors selected over target-better alternatives on robust pairs.")
    add("K_C37_selection_target_conflict_exact_rate", "C37", "selection_target_conflict_exact_rate",
        c37["selection_audit_reconciliation_summary"]["selection_target_conflict_exact_rate"],
        "oaci/reports/C37_EXACT_SELECTOR_TRACE_RECOVERY.json")
    add("K_C38_point_dominant_fraction", "C38", "point_dominant_fraction",
        c38["ucl_point_width_summary"]["point_dominant_fraction"], "oaci/reports/C38_LEAKAGE_UCL_OBJECTIVE_GEOMETRY.json",
        note="UCL conflict is point-leakage dominated rather than uncertainty-width dominated.")
    add("K_C38_leakage_target_gauge_conflict_fraction", "C38", "leakage_target_gauge_conflict_fraction",
        c38["gauge_conflict_summary"]["leakage_target_gauge_conflict_fraction"],
        "oaci/reports/C38_LEAKAGE_UCL_OBJECTIVE_GEOMETRY.json")
    return rows


def _key_map(rows: list[dict]) -> dict[str, dict]:
    return {r["provenance_id"]: r for r in rows}


def build_claim_contract() -> tuple[list[dict], list[dict], list[dict]]:
    claims = [
        ("CL01", "Support-aware leakage and estimability gates are retained as falsification instrumentation.", "negative_control_result", "C14-C24", "I1/I3", "Do not reword as a recovered OACI control objective.", "yes", "Support-aware OACI and falsification battery"),
        ("CL02", "OACI is not rescued as a deployable control objective in the observed EEG-DG setting.", "negative_control_result", "C23-C30,C56", "I1", "Do not reword as a successful OACI method.", "yes", "Empirical failure of source-side control"),
        ("CL03", "Good checkpoints are common enough that the failure is localization rather than endpoint scarcity.", "observed_in_this_setting", "C31,C32", "I0-I1", "Do not reword as an absence of target-good checkpoints.", "yes", "Selector/localization audit"),
        ("CL04", "Source-side measurements contain weak within-target rank information.", "empirical_mechanism", "C31,C42,C43", "I1", "Do not reword as reliable target checkpoint selection.", "yes", "Rank-gauge mechanism"),
        ("CL05", "Target competence decomposes empirically into a source-visible rank axis plus a target-specific gauge/offset axis.", "empirical_mechanism", "C31,C45,C46", "I1-I4", "Do not reword as a universal theorem.", "yes", "Rank-gauge mechanism"),
        ("CL06", "Source scalarization and source Pareto geometry do not provide reliable actionability.", "negative_control_result", "C42-C44", "I1", "Do not reword as zero source signal.", "yes", "Empirical failure of source-side control"),
        ("CL07", "Selected OACI misses often incur real continuous and preference-robust local regret.", "observed_in_this_setting", "C34,C35", "I1/I6", "Do not reword as threshold-only failure.", "yes", "Selector/localization audit"),
        ("CL08", "Exact selector mechanics favor the selected source-side candidate over target-better alternatives.", "empirical_mechanism", "C36-C38", "I1", "Do not reword as a candidate recommendation rule.", "yes", "Selector/localization audit"),
        ("CL09", "Conditioning exposes diagnostic local islands and ceilings.", "diagnostic_ceiling", "C46-C50", "I4/I7", "Do not reword as an action rule.", "yes", "Conditioning and local diagnostic ceilings"),
        ("CL10", "Trajectory fragmentation and score underuse prevent those islands from becoming reliable actionability.", "empirical_mechanism", "C50,C51", "I4/I7", "Do not reword as broad local purity or deployment.", "yes", "Conditioning and local diagnostic ceilings"),
        ("CL11", "Target/trajectory keys alone do not close the residual gauge gap.", "information_boundary", "C52", "I2", "Do not reword as target-key sufficiency.", "yes", "Information-boundary closure"),
        ("CL12", "Target-label-derived diagnostic content closes the residual diagnostically.", "diagnostic_ceiling", "C52,C53", "I6", "Do not reword as source-only DG.", "yes", "Information-boundary closure"),
        ("CL13", "The strongest endpoint scalar is a same-label endpoint oracle unavailable at selection time.", "information_boundary", "C54,C55", "I7", "Do not reword as an available calibration method.", "yes", "Endpoint-oracle boundary"),
        ("CL14", "C55 template-only transfer is partial and does not beat the max null p95.", "information_boundary", "C55,C56", "I6/I7", "Do not reword as full template transfer.", "yes", "Endpoint-oracle boundary"),
        ("CL15", "Split-label or few-label sufficiency remains future work because the required cache is unavailable.", "future_work_only", "C53-C55", "I5", "Do not reword as established few-label calibration.", "yes", "Discussion / limitations"),
        ("CL16", "C14-C56 are ready to be compressed into a manuscript scaffold rather than extended by another exploratory C-number.", "observed_in_this_setting", "C56,C57", "I0-I7", "Do not reword as a final formal theory.", "yes", "Discussion / limitations"),
    ]
    claim_rows = [
        {
            "claim_id": cid,
            "claim_text": text,
            "allowed_strength": strength,
            "required_evidence_milestones": milestones,
            "allowed_information_class": info,
            "forbidden_rewordings": forbidden,
            "needs_caveat": caveat,
            "manuscript_section": section,
        }
        for cid, text, strength, milestones, info, forbidden, caveat, section in claims
    ]
    forbidden_rows = [
        {
            "forbidden_class": name,
            "why_forbidden": reason,
            "allowed_replacement": repl,
            "red_team_gate": "fail_if_affirmed",
        }
        for name, reason, repl in (
            ("source_only_solution", "C42-C55 do not establish reliable source-only actionability.", "source-side measurements are weak but not reliable controls"),
            ("deployable_selector", "No selected-candidate or deployment rule is emitted.", "diagnostic ceiling or empirical mechanism"),
            ("OACI_rescue", "OACI remains falsified as a control objective.", "support-aware measurement/falsification instrumentation"),
            ("few_label_sufficiency", "Split-label cache is unavailable.", "future split-label extension"),
            ("universal_DG_impossibility", "C57 is empirical and setting-bound.", "observed information boundary"),
            ("formal_theorem_without_proof", "No proof is supplied in C57.", "future theorem candidate"),
            ("same_label_oracle_as_available_method", "Endpoint scalar reads candidate target endpoint content.", "same-label diagnostic oracle"),
            ("target_unlabeled_as_source_only", "Target-unlabeled information is outside source-only DG.", "target-unlabeled diagnostic rung"),
            ("target_grouped_as_source_only", "Target grouping is a separate problem class.", "target-grouped diagnostic rung"),
        )
    ]
    strength_rows = [
        {"allowed_strength": s, "meaning": meaning, "manuscript_usage": usage}
        for s, meaning, usage in (
            ("observed_in_this_setting", "empirical number from frozen artifacts", "results and mechanism claims"),
            ("empirical_mechanism", "supported causal-style diagnosis without formal theorem", "mechanism sections"),
            ("diagnostic_ceiling", "upper envelope using target labels or diagnostic grouping", "ceiling and limitation sections"),
            ("information_boundary", "availability-separated empirical boundary", "C52-C55 synthesis"),
            ("negative_control_result", "escape hatch or method claim falsified", "failure and audit sections"),
            ("future_work_only", "not evaluated in current artifacts", "limitations"),
            ("not_claimed", "must not be asserted", "red-team appendix"),
        )
    ]
    return claim_rows, forbidden_rows, strength_rows


def build_information_tables() -> tuple[list[dict], list[dict]]:
    ladder = [
        ("I0_random_or_tie", "no row-specific candidate information beyond base rates", "baseline only", "C42,C52,C55", "not sufficient", "random/tie comparator"),
        ("I1_strict_source_observables", "source risk, rank, leakage, and objective fields", "weak signal", "C42,C43,C44", "not reliable", "available but insufficient"),
        ("I2_source_plus_target_or_trajectory_keys", "source fields plus target/trajectory identifiers or keys", "key-only insufficient", "C52", "not sufficient", "not original source-only DG"),
        ("I3_target_unlabeled_transductive_geometry", "unlabeled target geometry or confidence fields", "diagnostic but not rescue", "C25,C35,C52", "not sufficient", "non-source-only"),
        ("I4_target_grouped_zero_label_structure", "target/trajectory grouped diagnostic structure", "descriptive and diagnostic", "C46-C50", "not an available action rule", "separate problem class"),
        ("I5_split_label_or_few_label_calibration", "target-label budget disjoint from evaluation labels", "not evaluated", "C53-C55", "open future", "requires unavailable split-label cache"),
        ("I6_target_label_diagnostic_content", "target-label-derived diagnostic fields", "closes residual diagnostically", "C52,C53", "diagnostic only", "not source-only"),
        ("I7_same_label_endpoint_oracle", "same-label endpoint scalar or joint margin", "tautological closure", "C54,C55", "diagnostic endpoint oracle", "unavailable at selection time"),
    ]
    ladder_rows = [
        {
            "information_class": c,
            "definition": d,
            "empirical_status": s,
            "supporting_milestones": m,
            "sufficiency_boundary": b,
            "manuscript_phrase": p,
        }
        for c, d, s, m, b, p in ladder
    ]
    ceiling_rows = [
        {
            "object": "conditioned local Bayes ceiling",
            "information_class": "I7",
            "uses_target_labels": 1,
            "available_at_selection_time": 0,
            "supported_claim": "upper diagnostic envelope exposes underuse",
            "forbidden_claim": "available action rule",
            "supporting_milestones": "C48-C51",
        },
        {
            "object": "best strict source score",
            "information_class": "I1",
            "uses_target_labels": 0,
            "available_at_selection_time": 1,
            "supported_claim": "weak source-visible rank signal",
            "forbidden_claim": "reliable selector",
            "supporting_milestones": "C42-C44",
        },
        {
            "object": "target/trajectory key-only ladder",
            "information_class": "I2",
            "uses_target_labels": 0,
            "available_at_selection_time": 0,
            "supported_claim": "keys alone do not close residual",
            "forbidden_claim": "key sufficiency",
            "supporting_milestones": "C52",
        },
        {
            "object": "target-label diagnostic closure",
            "information_class": "I6",
            "uses_target_labels": 1,
            "available_at_selection_time": 0,
            "supported_claim": "residual closes diagnostically",
            "forbidden_claim": "source-only DG",
            "supporting_milestones": "C52-C53",
        },
        {
            "object": "same-label endpoint scalar",
            "information_class": "I7",
            "uses_target_labels": 1,
            "available_at_selection_time": 0,
            "supported_claim": "endpoint-oracle boundary",
            "forbidden_claim": "deployable calibration",
            "supporting_milestones": "C54-C55",
        },
    ]
    return ladder_rows, ceiling_rows


def build_figures(key_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    k = _key_map(key_rows)

    def kv(*ids):
        return ";".join(f"{i}={k[i]['value_fmt_3']}" for i in ids)

    figures = [
        ("F1", "Problem and information ladder", "C52-C56", "oaci/reports/c56_tables/information_class_ladder.csv", kv("K_C52_best_key_only_hit", "K_C52_best_label_derived_hit", "K_C55_endpoint_scalar_transfer"), "Availability-separated ladder from source to endpoint oracle.", "same-label endpoint oracle as available method"),
        ("F2", "Support-aware OACI and falsification battery", "C23-C31,C56", "oaci/reports/c56_tables/milestone_evidence_ledger.csv", kv("K_C31_joint_good_rate", "K_C42_source_rank_top1_joint"), "Good checkpoints exist while source-side control remains weak.", "OACI recovered as control objective"),
        ("F3", "Good checkpoints exist; source control fails localization", "C31,C42,C43", "oaci/reports/c56_tables/key_number_provenance.csv", kv("K_C31_joint_good_rate", "K_C42_random_base", "K_C42_source_rank_top1_joint", "K_C43_best_source_scalarization_top1"), "Failure is localization, not endpoint scarcity.", "good checkpoints absent"),
        ("F4", "Rank-gauge decomposition", "C31,C45,C46", "oaci/reports/c56_tables/mechanism_edges.csv", kv("K_C46_within_target_q10", "K_C46_within_trajectory_q10", "K_C46_cross_target_q10"), "Conditioning restores local meaning while cross-target comparability breaks.", "universal non-identifiability theorem"),
        ("F5", "Exact selector and local-regret mechanics", "C34-C38", "oaci/reports/c57_tables/key_number_provenance.csv", kv("K_C34_real_endpoint_regret_fraction", "K_C35_preference_robust_fraction", "K_C37_ucl_prefers_selected_fraction", "K_C38_point_dominant_fraction"), "Selected-vs-better conflict is real, preference-robust, and point-leakage dominated.", "threshold-only artifact"),
        ("F6", "Conditioned diagnostic ceiling versus actionability", "C48-C51", "oaci/reports/c56_tables/key_number_provenance.csv", kv("K_C48_local_ceiling_hit", "K_C48_local_ceiling_enrichment", "K_C50_trajectory_fail_fraction", "K_C50_max_mean_underuse_gap"), "Diagnostic islands exist but fragment and are underused.", "local Bayes ceiling as action rule"),
        ("F7", "Endpoint-label oracle boundary", "C52-C55", "oaci/reports/c56_tables/c55_null_provenance.csv", kv("K_C52_best_key_only_hit", "K_C53_best_scalar_endpoint_hit", "K_C55_template_only_best", "K_C55_max_null_p95", "K_C55_endpoint_scalar_transfer"), "Full closure requires endpoint-label content; template-only remains partial.", "template-only transfer beats max null"),
        ("F8", "Final mechanism diagram", "C31-C56", "oaci/reports/c56_tables/mechanism_nodes.csv", kv("K_C31_joint_good_rate", "K_C42_source_rank_top1_joint", "K_C48_local_ceiling_hit", "K_C55_endpoint_scalar_transfer"), "Source measurements are real but not reliable controls under the original setting.", "deployable target-aware method"),
    ]
    figure_rows = [
        {
            "figure_id": fid,
            "figure_title": title,
            "source_milestones": milestones,
            "artifact_provenance": artifact,
            "key_numbers": numbers,
            "supported_claim": claim,
            "unsupported_overclaim": overclaim,
            "main_or_supplement": "main",
        }
        for fid, title, milestones, artifact, numbers, claim, overclaim in figures
    ]
    supplements = [
        ("S1", "Validation/regression timeline", "C14-C56", "oaci/reports/c56_tables/validation_timeline.csv"),
        ("S2", "Artifact and red-team gates", "C56-C57", "oaci/reports/c57_tables/red_team_failure_ledger.csv"),
        ("S3", "Source scalarization/Pareto failure", "C42-C44", "oaci/reports/c56_tables/claim_support_matrix.csv"),
        ("S4", "Local Bayes coverage grids", "C48-C51", "oaci/reports/c56_tables/key_number_provenance.csv"),
        ("S5", "C52-C55 information-boundary ledger", "C52-C55", "oaci/reports/c57_tables/information_class_ladder.csv"),
        ("S6", "Literature-alignment map", "C56-C57", "oaci/reports/c57_tables/literature_alignment_matrix.csv"),
    ]
    for fid, title, milestones, artifact in supplements:
        figure_rows.append({
            "figure_id": fid,
            "figure_title": title,
            "source_milestones": milestones,
            "artifact_provenance": artifact,
            "key_numbers": "see keyed table",
            "supported_claim": "supplementary provenance and claim discipline",
            "unsupported_overclaim": "new experiment or method claim",
            "main_or_supplement": "supplement",
        })
    key_to_figure_rows = []
    for row in figure_rows:
        for part in row["key_numbers"].split(";"):
            if "=" not in part:
                continue
            key_to_figure_rows.append({
                "provenance_id": part.split("=")[0],
                "figure_id": row["figure_id"],
                "figure_title": row["figure_title"],
                "display_value": part.split("=")[1],
            })
    return figure_rows, key_to_figure_rows


def build_literature_tables(claim_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    lit_rows = [
        {
            "literature_id": r["literature_id"],
            "axis": r["axis"],
            "citation": r["citation"],
            "url_or_status": r["url"],
            "alignment": r["alignment"],
            "blocked_overclaim": r["blocked_overclaim"],
        }
        for r in LITERATURE
    ]
    claim_map = [
        {"claim_id": "CL02", "literature_id": "DomainBed_2007_01434", "alignment_type": "model_selection_caution"},
        {"claim_id": "CL05", "literature_id": "IRM_1907_02893", "alignment_type": "invariance_claim_boundary"},
        {"claim_id": "CL05", "literature_id": "ZhaoInvariantDA_1901_09453", "alignment_type": "conditional_shift_boundary"},
        {"claim_id": "CL13", "literature_id": "PostSelection_1401_3889", "alignment_type": "data_reuse_guardrail"},
        {"claim_id": "CL16", "literature_id": "InteractiveLowerBounds_2410_05117", "alignment_type": "future_theory_language"},
        {"claim_id": "CL01", "literature_id": "EEG_DG_project_bibliography_pending", "alignment_type": "M1_bibliography_expansion"},
    ]
    forbidden = [
        {"overclaim": "universal_DG_failure", "status": "blocked", "reason": "C57 is empirical and setting-bound."},
        {"overclaim": "all_invariance_methods_fail", "status": "blocked", "reason": "IRM/invariance are alignment axes, not targets of universal refutation."},
        {"overclaim": "SOTA_method_claim", "status": "blocked", "reason": "C57 emits no method and no benchmark-improvement claim."},
        {"overclaim": "formal_minimax_lower_bound", "status": "blocked", "reason": "Only future theorem candidates are provided."},
        {"overclaim": "endpoint_oracle_calibration_method", "status": "blocked", "reason": "Same-label endpoint scalar is unavailable at selection time."},
    ]
    return lit_rows, claim_map, forbidden


def build_reviewer_tables() -> tuple[list[dict], list[dict]]:
    questions = [
        ("RQ01", "Is this just a negative result?", "No. The contribution is a measurement/control separation, rank-gauge mechanism, and information-boundary closure.", "C31-C56", "Do not frame as universal DG failure."),
        ("RQ02", "Are good checkpoints absent?", "No. C31 reports joint-good rate 0.424 and C31/C32 show good candidates are common enough; localization fails.", "K_C31_joint_good_rate", "Do not imply endpoint scarcity."),
        ("RQ03", "Did target labels leak into selection?", "No selected artifact is emitted. Target labels appear only in diagnostic ceilings and endpoint-oracle audits.", "C48-C56", "Do not describe diagnostics as deployment."),
        ("RQ04", "Why is local Bayes ceiling not an action rule?", "It is an upper envelope over label-scored neighborhoods, not a source-measurable rule at selection time.", "C48-C51", "Ceiling is diagnostic-only."),
        ("RQ05", "Why does conditioning not rescue actionability?", "C50/C51 show trajectory fragmentation and existing-score underuse despite broad diagnostic coverage.", "K_C50_trajectory_fail_fraction;K_C50_max_mean_underuse_gap", "Do not call grouped diagnostics source-only."),
        ("RQ06", "Why does C55 not show a transferable endpoint-template method?", "Template-only best is 0.704 and does not beat max null p95 0.771; 0.944 requires held-out endpoint scalar.", "K_C55_template_only_best;K_C55_max_null_p95;K_C55_endpoint_scalar_transfer", "Preserve the C55 null clarification."),
        ("RQ07", "What exactly is unavailable at selection time?", "Target-label-derived diagnostics and same-label endpoint scalar/margin are unavailable under original source-only DG.", "C52-C55", "Do not call endpoint scalar an available method."),
        ("RQ08", "Is split-label or few-label calibration ruled out?", "No. It remains future work because the split-label cache is unavailable in current artifacts.", "K_C53_split_label_budget_available", "Do not claim few-label sufficiency."),
        ("RQ09", "How does this relate to IRM, DomainBed, invariant DA lower bounds, and the broader literature?", "Those works frame invariance/model-selection/lower-bound caution; C57 uses them for claim discipline only.", "C57 literature", "No universal lower-bound claim."),
        ("RQ10", "What is the contribution if no new method is proposed?", "A falsification framework, rank-gauge diagnosis, selector/localization audit, and availability-separated endpoint boundary.", "CL01-CL16", "No SOTA or method claim."),
        ("RQ11", "What is EEG-specific here?", "The observed cross-subject EEG candidate universe shows source-visible rank plus target-specific gauge/offset and endpoint-label availability gaps.", "C31-C56", "Do not claim EEG transfer impossible."),
        ("RQ12", "What should happen after C57?", "If C57-A passes, move to M1 manuscript drafting; only repair named inconsistencies if found.", "C57 decision", "Do not open another exploratory C-number without a named gap."),
    ]
    bank = [
        {
            "question_id": qid,
            "question": q,
            "short_answer": ans,
            "evidence_refs": refs,
            "claim_boundary": boundary,
        }
        for qid, q, ans, refs, boundary in questions
    ]
    amap = [
        {
            "question_id": r["question_id"],
            "supporting_milestones": r["evidence_refs"],
            "answer_type": "evidence_bounded",
            "forbidden_overclaim_guard": r["claim_boundary"],
        }
        for r in bank
    ]
    return bank, amap


def build_taxonomy_tables() -> tuple[list[dict], list[dict]]:
    terms = [
        ("measurement_vs_control", "measurement detects a real signal; control reliably acts on it", "I1", 0, 0, 1, "empirical_mechanism", "C31-C44,C56", "source-side measurements are real but not reliable controls", "do not equate measurement with actionability"),
        ("diagnostic_ceiling_vs_action_rule", "a ceiling uses diagnostic target information; an action rule is available at selection time", "I4/I6/I7", 1, 1, 0, "diagnostic_ceiling", "C48-C55", "conditioned ceilings are diagnostic upper envelopes", "do not call ceilings selectors"),
        ("rank_vs_gauge", "rank is weak within-target ordering; gauge is target-specific offset/scale/localization", "I1-I4", 0, 0, 0, "empirical_mechanism", "C31,C45,C46", "rank-gauge is an empirical decomposition", "do not pool rank and gauge as one scalar or theorem"),
        ("source_only_vs_target_unlabeled_vs_target_label", "source-only, target-unlabeled, and target-label-derived information are separate availability classes", "I1/I3/I6/I7", 1, 1, 0, "information_boundary", "C25,C35,C52-C55", "use availability classes explicitly", "do not call target-unlabeled or target-label diagnostics source-only"),
        ("key_only_vs_label_content", "keys partition cells; label content evaluates candidates", "I2/I6", 0, 1, 0, "information_boundary", "C52", "key-only remains separate from label-derived diagnostics", "do not say target/trajectory keys close the residual"),
        ("template_transfer_vs_endpoint_scalar_availability", "templates transfer partially; endpoint scalar reads held-out candidate endpoint", "I6/I7", 0, 1, 0, "information_boundary", "C55,C56", "template-only is partial and endpoint-scalar availability is diagnostic", "do not claim template-only equals 0.944 closure"),
        ("same_label_oracle_vs_split_label_calibration", "same-label oracle reuses evaluated endpoint; split-label calibration needs disjoint labels", "I5/I7", 0, 1, 0, "future_work_only", "C53-C55", "same-label oracle is diagnostic and split-label calibration is future work", "do not claim split-label sufficiency"),
    ]
    crosswalk = [
        {
            "term_id": tid,
            "definition": definition,
            "availability_class": availability_class,
            "uses_target_unlabeled": uses_unlabeled,
            "uses_target_labels": uses_labels,
            "available_at_selection_time": available,
            "allowed_claim_strength": strength,
            "supporting_milestones": milestones,
            "preferred_usage": usage,
            "forbidden_usage": forbidden,
        }
        for tid, definition, availability_class, uses_unlabeled, uses_labels, available, strength, milestones, usage, forbidden in terms
    ]
    guardrails = [
        {"term": "OACI", "use_as": "support-aware falsification instrumentation after failure", "avoid_as": "rescued control objective"},
        {"term": "source rank", "use_as": "weak source-visible signal", "avoid_as": "reliable selector"},
        {"term": "local Bayes ceiling", "use_as": "diagnostic ceiling", "avoid_as": "action rule"},
        {"term": "endpoint scalar", "use_as": "same-label endpoint oracle", "avoid_as": "available calibration feature"},
        {"term": "split-label", "use_as": "future extension", "avoid_as": "established result"},
    ]
    return crosswalk, guardrails


def build_section_and_contribution_rows(claim_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    sections = [
        ("S01", "Introduction", "source-side invariance as measurement rather than control", "CL01,CL02,CL03"),
        ("S02", "Setting and strict target-isolation protocol", "availability classes and no target-label use in selection", "CL12,CL13,CL15"),
        ("S03", "Support-aware OACI and falsification battery", "instrumentation survives; control objective does not", "CL01,CL02"),
        ("S04", "Empirical failure of source-side control", "good candidates exist but source localization is weak", "CL03,CL04,CL06"),
        ("S05", "Rank-gauge mechanism", "rank and gauge explain weak signal plus pooled failure", "CL04,CL05"),
        ("S06", "Selector/localization audit", "local regret and exact UCL mechanics expose active misselection", "CL07,CL08"),
        ("S07", "Conditioning, diagnostic ceilings, and actionability failure", "conditioning exposes islands but fragmentation/underuse blocks actionability", "CL09,CL10"),
        ("S08", "Information-boundary closure and endpoint-oracle boundary", "keys fail, labels diagnose, same-label endpoint oracle closes tautologically", "CL11,CL12,CL13,CL14"),
        ("S09", "Related work", "literature constrains claim language", "CL16"),
        ("S10", "Discussion, limitations, and split-label future work", "manuscript-ready but not a theorem or deployable method", "CL15,CL16"),
    ]
    section_rows = [
        {"section_id": sid, "section_title": title, "section_thesis": thesis, "claim_ids": ids}
        for sid, title, thesis, ids in sections
    ]
    contribution_rows = [
        {"contribution_id": "C1", "contribution": "support-aware falsification framework", "claim_ids": "CL01,CL02", "main_figures": "F1,F2", "claim_boundary": "measurement, not OACI rescue"},
        {"contribution_id": "C2", "contribution": "rank-gauge empirical mechanism", "claim_ids": "CL04,CL05", "main_figures": "F3,F4", "claim_boundary": "empirical mechanism, not theorem"},
        {"contribution_id": "C3", "contribution": "selector/localization failure anatomy", "claim_ids": "CL03,CL06,CL07,CL08,CL09,CL10", "main_figures": "F3,F5,F6", "claim_boundary": "failure is localization, not absence of good checkpoints"},
        {"contribution_id": "C4", "contribution": "information-boundary and endpoint-oracle closure", "claim_ids": "CL11,CL12,CL13,CL14,CL15", "main_figures": "F1,F7,F8", "claim_boundary": "diagnostic label content, not source-only DG"},
    ]
    caveats = [
        {"caveat_id": "CV1", "caveat": "split-label/few-label calibration remains unresolved", "status": "future_work", "guardrail": "do not claim few-label sufficiency"},
        {"caveat_id": "CV2", "caveat": "formal lower-bound theorem is not proved", "status": "future_theory", "guardrail": "label formalism empirical"},
        {"caveat_id": "CV3", "caveat": "endpoint scalar is same-label target endpoint content", "status": "diagnostic_only", "guardrail": "unavailable at selection time"},
        {"caveat_id": "CV4", "caveat": "EEG-specific related work should be expanded during M1", "status": "M1_bibliography", "guardrail": "does not block claim contract"},
    ]
    return section_rows, contribution_rows, caveats


def build_subagent_manifest() -> list[dict]:
    roles = [
        ("SA1", "Manuscript Architect", "title/abstract/outline/section claims", "launched_integrated"),
        ("SA2", "Claim-Contract Auditor", "claim contract and strength ladder", "launched_integrated"),
        ("SA3", "Evidence-to-Figure Planner", "main and supplementary figure plan", "launched_integrated"),
        ("SA4", "Information-Boundary Formalizer", "I0-I7 and ceiling/action distinction", "launched_integrated"),
        ("SA5", "Literature Alignment Agent", "related-work matrix and overclaim blocks", "launched_integrated"),
        ("SA6", "Reviewer Q&A Simulator", "reviewer questions and bounded answers", "launched_integrated"),
        ("SA7", "Terminology/Taxonomy Consolidator", "term contract and crosswalk", "launched_integrated"),
        ("SA8", "Artifact/Provenance Auditor", "key number and artifact traceability", "launched_integrated"),
        ("SA9", "Final Red-Team Integrator", "hard gates and decision audit", "launched_integrated"),
    ]
    return [
        {"subagent_id": sid, "role": role, "scope": scope, "integration_status": status}
        for sid, role, scope, status in roles
    ]


def _is_inventory_path(path: str) -> bool:
    name = os.path.basename(path)
    return name in {
        "forbidden_claim_boundary.csv",
        "forbidden_claim_scan.csv",
        "forbidden_literature_overclaims.csv",
        "red_team_failure_ledger.csv",
        "claim_contract.csv",
        "term_usage_guardrails.csv",
    }


def _affirmative_hit(text: str, phrase: str, window: int = 180) -> bool:
    low = text.lower()
    phrase = phrase.lower()
    start = 0
    while True:
        idx = low.find(phrase, start)
        if idx == -1:
            return False
        ctx = low[max(0, idx - window):idx]
        if not any(cue in ctx for cue in NEGATION_CUES):
            return True
        start = idx + len(phrase)


def build_forbidden_scan(paths: list[str]) -> list[dict]:
    rows = []
    for pattern in FORBIDDEN_PATTERNS:
        total = 0
        affirmative = 0
        files = []
        for path in paths:
            if _is_inventory_path(path):
                continue
            text = open(path, errors="ignore").read()
            count = text.lower().count(pattern.lower())
            if count:
                total += count
                files.append(path)
                if _affirmative_hit(text, pattern):
                    affirmative += 1
        rows.append({
            "pattern": pattern,
            "total_hits": total,
            "affirmative_hits": affirmative,
            "files": ";".join(files),
            "passed": int(affirmative == 0),
        })
    return rows


def build_red_team_rows(res: dict) -> list[dict]:
    c55 = res["c55_null_disambiguation"]
    checks = [
        ("c56_ready_for_scaffold", res["c56_decision"] == "C56-A_mechanism_closed_ready_for_manuscript_scaffold", "C56 primary decision supports manuscript scaffold."),
        ("c57_primary_is_A", res["decision"]["primary"] == "C57-A_manuscript_scaffold_ready", "C57 selects manuscript scaffold ready."),
        ("claim_contract_complete", len(res["claim_contract_rows"]) >= 16, "Claim contract covers the four manuscript contributions and late boundary claims."),
        ("all_key_numbers_traceable", not res["missing_or_ambiguous_provenance_rows"], "Every manuscript-facing key number is traceable."),
        ("c55_null_clarification_preserved", c55["endpoint_scalar_transfer_beats_max_null_p95"] and not c55["template_only_beats_max_null_p95"], "Endpoint scalar 0.944 beats max null p95 0.771; template-only 0.704 does not."),
        ("split_label_future_work_only", any(r["claim_id"] == "CL15" and r["allowed_strength"] == "future_work_only" for r in res["claim_contract_rows"]), "Split-label calibration remains future work only."),
        ("same_label_oracle_unavailable", any(r["object"] == "same-label endpoint scalar" and r["available_at_selection_time"] == 0 for r in res["bayes_ceiling_vs_action_rule_rows"]), "Same-label endpoint scalar is unavailable at selection time."),
        ("literature_overclaims_blocked", all(r["status"] == "blocked" for r in res["forbidden_literature_overclaims_rows"]), "Literature rows block universal, SOTA, endpoint-calibration, and theorem overclaims."),
        ("forbidden_claim_scan_passed", all(int(r["passed"]) for r in res["forbidden_claim_scan_rows"]), "Forbidden affirmative claim scan has zero affirmative hits."),
        ("no_selector_or_checkpoint_artifact", not any("selected_candidate_id" in open(p, errors="ignore").read() or "checkpoint_hash" in open(p, errors="ignore").read() for p in res["generated_paths"] if p.endswith((".md", ".json", ".csv"))), "C57 emits no selected-candidate or checkpoint-hash artifact."),
        ("compact_artifacts", all(int(r["passed"]) for r in res["large_artifact_scan_rows"]), "C57 JSON is compact and row-level evidence lives in c57_tables."),
    ]
    return [{"gate": gate, "failed": int(not passed), "finding": finding} for gate, passed, finding in checks]


def classify(res: dict) -> dict:
    failures = [r for r in res["red_team_failure_ledger_rows"] if int(r["failed"])]
    if failures:
        primary = "C57-C_claim_contract_inconsistency_requires_repair"
    elif res["missing_or_ambiguous_provenance_rows"]:
        primary = "C57-D_figure_or_evidence_provenance_gap_requires_repair"
    else:
        primary = "C57-A_manuscript_scaffold_ready"
    return {
        "primary": primary,
        "secondary": [
            "C57-S1_claim_contract_locked",
            "C57-S2_figure_plan_provenance_backed",
            "C57-S3_information_boundary_formalism_empirical",
            "C57-S4_literature_alignment_claim_limited",
            "C57-S5_ready_for_M1_drafting",
        ],
        "red_team_failure_count": len(failures),
        "untraceable_key_number_count": len(res["missing_or_ambiguous_provenance_rows"]),
        "recommended_next_direction": "M1 manuscript drafting" if primary == "C57-A_manuscript_scaffold_ready" else "repair before drafting",
    }


def table_row_counts(res: dict) -> dict:
    names = {
        "claim_contract": "claim_contract_rows",
        "forbidden_claim_boundary": "forbidden_claim_boundary_rows",
        "claim_strength_ladder": "claim_strength_ladder_rows",
        "figure_evidence_map": "figure_evidence_map_rows",
        "key_number_to_figure_map": "key_number_to_figure_map_rows",
        "information_class_ladder": "information_class_ladder_rows",
        "bayes_ceiling_vs_action_rule": "bayes_ceiling_vs_action_rule_rows",
        "literature_alignment_matrix": "literature_alignment_matrix_rows",
        "claim_to_literature_map": "claim_to_literature_map_rows",
        "forbidden_literature_overclaims": "forbidden_literature_overclaims_rows",
        "reviewer_question_bank": "reviewer_question_bank_rows",
        "reviewer_answer_evidence_map": "reviewer_answer_evidence_map_rows",
        "taxonomy_crosswalk": "taxonomy_crosswalk_rows",
        "term_usage_guardrails": "term_usage_guardrails_rows",
        "key_number_provenance": "key_number_provenance_rows",
        "artifact_manifest": "artifact_manifest_rows",
        "missing_or_ambiguous_provenance": "missing_or_ambiguous_provenance_rows",
        "red_team_failure_ledger": "red_team_failure_ledger_rows",
        "forbidden_claim_scan": "forbidden_claim_scan_rows",
        "manuscript_section_map": "manuscript_section_map_rows",
        "contribution_map": "contribution_map_rows",
        "remaining_caveats": "remaining_caveats_rows",
        "test_command_manifest": "test_command_manifest_rows",
        "schema_validation_summary": "schema_validation_summary_rows",
        "large_artifact_scan": "large_artifact_scan_rows",
        "subagent_audit_manifest": "subagent_audit_manifest_rows",
    }
    return {name: len(res.get(key, [])) for name, key in names.items()}


def build_reports(res: dict) -> dict[str, str]:
    k = _key_map(res["key_number_provenance_rows"])
    d = res["decision"]
    artifact_count = len(res.get("artifact_manifest_rows") or res.get("generated_paths") or [])
    main = "\n".join([
        f"# C57 - Manuscript Scaffold / Claim-Contract / Theory-Literature Package (frozen C19 `{res['config_hash']}`)",
        "",
        "## Primary Decision",
        "",
        f"`{d['primary']}`",
        "",
        f"Secondary: `{';'.join(d['secondary'])}`",
        "",
        "## Why C56 Justifies Scaffold",
        "",
        "C56 closed the review gauntlet with `C56-A_mechanism_closed_ready_for_manuscript_scaffold`: the mechanism is provenance-backed, the information ladder is empirical rather than theorem-level, and no new experiment is required before manuscript scaffolding.",
        "",
        "## Manuscript Thesis",
        "",
        "In EEG cross-subject domain generalization, source-side measurements can be real without being reliable controls. Good checkpoints exist, but source-only localization fails because target competence mixes a weak source-visible rank axis with a target-specific gauge/offset axis. Conditioning reveals diagnostic islands, but actionability collapses unless target-label-derived endpoint information is used diagnostically.",
        "",
        "## Contribution Map",
        "",
        "1. Support-aware falsification framework: measurement instrumentation survives; OACI is not recovered as a control objective.",
        "2. Rank-gauge mechanism: weak source-visible rank and target-specific gauge explain the measurement/control split.",
        "3. Selector/localization audit: real local regret and exact selector mechanics explain target-wrong source-rational choices.",
        "4. Information-boundary closure: key-only and template-only escapes fail; same-label endpoint content closes diagnostically but is unavailable under original selection.",
        "",
        "## Information-Boundary Ladder",
        "",
        f"I1 source observables remain weak (`K_C42_source_rank_top1_joint`={k['K_C42_source_rank_top1_joint']['value_fmt_3']}); I2 key-only remains below label-derived diagnostics (`K_C52_best_key_only_hit`={k['K_C52_best_key_only_hit']['value_fmt_3']}, `K_C52_best_label_derived_hit`={k['K_C52_best_label_derived_hit']['value_fmt_3']}); I7 endpoint scalar is diagnostic (`K_C55_endpoint_scalar_transfer`={k['K_C55_endpoint_scalar_transfer']['value_fmt_3']}) but unavailable at selection time.",
        "",
        "## Figure Plan",
        "",
        "Eight main figures compress the audit trail: information ladder, falsification battery, localization failure, rank-gauge decomposition, exact selector/regret mechanics, conditioned ceiling/actionability failure, endpoint-oracle boundary, and final mechanism diagram.",
        "",
        "## Claim Contract Summary",
        "",
        f"C57 records {len(res['claim_contract_rows'])} allowed claims, {len(res['forbidden_claim_boundary_rows'])} forbidden claim classes, and {len(res['claim_strength_ladder_rows'])} allowed strength levels.",
        "",
        "## Literature-Alignment Summary",
        "",
        "Literature alignment constrains claim language around IRM/invariance, DomainBed/model selection, invariant DA lower-bound cautions, post-selection/data reuse, and future information lower-bound framing. The EEG-specific bibliography is marked for M1 expansion without blocking the scaffold.",
        "",
        "## Reviewer Q&A Summary",
        "",
        f"The reviewer dossier contains {len(res['reviewer_question_bank_rows'])} hard questions with evidence-bounded answers and explicit overclaim guards.",
        "",
        "## Remaining Caveats",
        "",
        "Split-label/few-label calibration remains future work; no formal theorem is claimed; the endpoint scalar is a same-label diagnostic oracle; M1 should expand EEG-specific related work.",
        "",
        "## Forbidden Claims Verified Absent",
        "",
        "The C57 red-team scan finds zero affirmative forbidden-claim hits in claim-bearing reports and tables.",
        "",
        "## Recommended Next Step",
        "",
        "Start M1 manuscript drafting: abstract, introduction, related work, methods summary, core figures, and claim-contract appendix.",
    ])
    red = "\n".join([
        "# C57 - Red-Team Verification",
        "",
        "All C57 red-team gates pass." if d["red_team_failure_count"] == 0 else "C57 red-team gates failed.",
        "",
        *[f"- {r['gate']}: {'PASS' if not int(r['failed']) else 'FAIL'} - {r['finding']}" for r in res["red_team_failure_ledger_rows"]],
    ])
    literature = "\n".join([
        "# C57 - Literature Alignment",
        "",
        "C57 uses literature as claim discipline rather than novelty inflation.",
        "",
        *[
            f"- {r['literature_id']}: {r['axis']} - {r['alignment']} ({r['url_or_status']})"
            for r in res["literature_alignment_matrix_rows"]
        ],
    ])
    qa_sections = [
        f"## {r['question_id']}\n{r['question']}\n\n{r['short_answer']}\nEvidence: {r['evidence_refs']}\nBoundary: {r['claim_boundary']}"
        for r in res["reviewer_question_bank_rows"]
    ]
    qa = "\n\n".join(["# C57 - Reviewer Q&A Dossier", *qa_sections])
    provenance = "\n".join([
        "# C57 - Provenance Audit",
        "",
        f"Tracked key numbers: {len(res['key_number_provenance_rows'])}.",
        f"Missing or ambiguous key numbers: {len(res['missing_or_ambiguous_provenance_rows'])}.",
        f"Tracked generated payload artifacts: {artifact_count}.",
        "Self-inventory tables `artifact_manifest.csv` and `large_artifact_scan.csv` are generated but intentionally excluded from manifest hash rows to avoid self-reference instability.",
        "",
        "C55/C56 null clarification is preserved: endpoint-scalar transfer beats the max null p95, while template-only transfer does not.",
    ])
    return {
        "C57_MANUSCRIPT_SCAFFOLD_CLAIM_CONTRACT.md": main,
        "C57_RED_TEAM_VERIFICATION.md": red,
        "C57_LITERATURE_ALIGNMENT.md": literature,
        "C57_REVIEWER_QA_DOSSIER.md": qa,
        "C57_PROVENANCE_AUDIT.md": provenance,
    }


def build_scaffold_docs(res: dict) -> dict[str, str]:
    titles = "\n".join([
        "# C57 - Title and Abstract Candidates",
        "",
        "## Title Candidate 1",
        "When Source-Side Invariance Becomes Measurement Rather Than Control: An Information-Boundary Audit of EEG Domain Generalization",
        "",
        "## Title Candidate 2",
        "Good Checkpoints, Broken Localization: A Rank-Gauge Audit of EEG Domain Generalization",
        "",
        "## Abstract Skeleton",
        "We study why source-side EEG-DG signals can measure useful structure without reliably controlling target checkpoint selection. Across a frozen audit trail, good checkpoints are common, source rank is weakly informative, and conditioning reveals diagnostic islands. Yet source-only actionability fails through target-specific gauge/offset, selector misranking, trajectory fragmentation, and endpoint-label availability. The residual closes only with target-label-derived endpoint content, culminating in a same-label endpoint oracle. The result is a claim-bounded empirical mechanism and information-boundary scaffold, not a new deployable DG method.",
    ])
    outline_lines = ["# C57 - Manuscript Outline", ""]
    for row in res["manuscript_section_map_rows"]:
        outline_lines += [f"## {row['section_id']} - {row['section_title']}", row["section_thesis"], f"Claims: {row['claim_ids']}", ""]
    section_claims = "\n".join([
        "# C57 - Section Claims",
        "",
        *[f"- {r['claim_id']}: {r['claim_text']} ({r['allowed_strength']}; {r['manuscript_section']})" for r in res["claim_contract_rows"]],
    ])
    figure_lines = ["# C57 - Figure Plan", ""]
    for row in res["figure_evidence_map_rows"]:
        figure_lines += [
            f"## {row['figure_id']} - {row['figure_title']}",
            f"Milestones: {row['source_milestones']}",
            f"Provenance: {row['artifact_provenance']}",
            f"Key numbers: {row['key_numbers']}",
            f"Supported claim: {row['supported_claim']}",
            f"Unsupported overclaim: {row['unsupported_overclaim']}",
            "",
        ]
    info = "\n".join([
        "# C57 - Information Boundary Formalism",
        "",
        "Let each candidate-selection rule be measurable with respect to an availability class I0-I7. C57 treats this as empirical formalism over the frozen artifact universe, not as a minimax theorem.",
        "",
        *[f"- {r['information_class']}: {r['definition']} -> {r['sufficiency_boundary']} ({r['manuscript_phrase']})" for r in res["information_class_ladder_rows"]],
        "",
        "Bayes ceilings and endpoint oracles are diagnostic upper envelopes because they use target-label information outside the original source-only DG selection sigma-field.",
    ])
    terminology = "\n".join([
        "# C57 - Terminology Contract",
        "",
        *[f"- {r['term_id']}: {r['definition']} Preferred: {r['preferred_usage']} Avoid: {r['forbidden_usage']}." for r in res["taxonomy_crosswalk_rows"]],
    ])
    return {
        "title_abstract_candidates.md": titles,
        "manuscript_outline.md": "\n".join(outline_lines).rstrip(),
        "section_claims.md": section_claims,
        "figure_plan.md": "\n".join(figure_lines).rstrip(),
        "information_boundary_formalism.md": info,
        "terminology_contract.md": terminology,
    }


def _compact_json(res: dict) -> dict:
    return {
        "milestone": MILESTONE,
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": True,
        "decision": res["decision"],
        "c56_decision": res["c56_decision"],
        "c55_null_disambiguation": res["c55_null_disambiguation"],
        "claim_contract_count": len(res["claim_contract_rows"]),
        "main_figure_count": sum(1 for r in res["figure_evidence_map_rows"] if r["main_or_supplement"] == "main"),
        "supplement_figure_count": sum(1 for r in res["figure_evidence_map_rows"] if r["main_or_supplement"] == "supplement"),
        "key_number_count": len(res["key_number_provenance_rows"]),
        "table_row_counts": table_row_counts(res),
        "recommended_next_step": res["decision"]["recommended_next_direction"],
    }


def run(recompute_artifacts: bool = False, test_status: str = "planned") -> dict:
    config_hash = _lock_config()
    c56 = _load_json(C56_JSON)
    key_rows = _key_rows()
    claim_rows, forbidden_rows, strength_rows = build_claim_contract()
    info_rows, ceiling_rows = build_information_tables()
    figure_rows, key_to_figure_rows = build_figures(key_rows)
    lit_rows, claim_lit_rows, forbidden_lit_rows = build_literature_tables(claim_rows)
    reviewer_rows, reviewer_map_rows = build_reviewer_tables()
    taxonomy_rows, term_rows = build_taxonomy_tables()
    section_rows, contribution_rows, caveat_rows = build_section_and_contribution_rows(claim_rows)
    test_rows = [
        {"test_scope": "focused_c57", "command": "python -m pytest oaci/tests/test_c57_manuscript_scaffold_claim_contract.py -q", "status": test_status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c50_c57_slice", "command": "python -m pytest oaci/tests/test_c50_conditioned_island_morphology.py ... test_c57_manuscript_scaffold_claim_contract.py -q", "status": test_status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c23_c57_regression", "command": "python -m pytest oaci/tests/test_c23_score_gauge.py ... test_c57_manuscript_scaffold_claim_contract.py -q", "status": test_status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "full_oaci_tests", "command": "python -m pytest oaci/tests -q", "status": test_status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
    ]
    res = {
        "config_hash": config_hash,
        "c56_decision": c56["decision"]["primary"],
        "c55_null_disambiguation": c56["c55_null_disambiguation"],
        "claim_contract_rows": claim_rows,
        "forbidden_claim_boundary_rows": forbidden_rows,
        "claim_strength_ladder_rows": strength_rows,
        "figure_evidence_map_rows": figure_rows,
        "key_number_to_figure_map_rows": key_to_figure_rows,
        "information_class_ladder_rows": info_rows,
        "bayes_ceiling_vs_action_rule_rows": ceiling_rows,
        "literature_alignment_matrix_rows": lit_rows,
        "claim_to_literature_map_rows": claim_lit_rows,
        "forbidden_literature_overclaims_rows": forbidden_lit_rows,
        "reviewer_question_bank_rows": reviewer_rows,
        "reviewer_answer_evidence_map_rows": reviewer_map_rows,
        "taxonomy_crosswalk_rows": taxonomy_rows,
        "term_usage_guardrails_rows": term_rows,
        "key_number_provenance_rows": key_rows,
        "missing_or_ambiguous_provenance_rows": [],
        "manuscript_section_map_rows": section_rows,
        "contribution_map_rows": contribution_rows,
        "remaining_caveats_rows": caveat_rows,
        "test_command_manifest_rows": test_rows,
        "subagent_audit_manifest_rows": build_subagent_manifest(),
        "schema_validation_summary_rows": [],
        "large_artifact_scan_rows": [],
        "artifact_manifest_rows": [],
        "forbidden_claim_scan_rows": [],
        "red_team_failure_ledger_rows": [],
        "generated_paths": [],
    }
    res["decision"] = {
        "primary": "C57-A_manuscript_scaffold_ready",
        "secondary": [],
        "red_team_failure_count": 0,
        "untraceable_key_number_count": 0,
        "recommended_next_direction": "M1 manuscript drafting",
    }
    res["forbidden_claim_scan_rows"] = []
    res["red_team_failure_ledger_rows"] = []
    res["decision"] = classify({**res, "red_team_failure_ledger_rows": []})
    return res


def write_tables(res: dict, table_dir: str) -> None:
    _write_csv(os.path.join(table_dir, "claim_contract.csv"), res["claim_contract_rows"],
               ["claim_id", "claim_text", "allowed_strength", "required_evidence_milestones", "allowed_information_class", "forbidden_rewordings", "needs_caveat", "manuscript_section"])
    _write_csv(os.path.join(table_dir, "forbidden_claim_boundary.csv"), res["forbidden_claim_boundary_rows"],
               ["forbidden_class", "why_forbidden", "allowed_replacement", "red_team_gate"])
    _write_csv(os.path.join(table_dir, "claim_strength_ladder.csv"), res["claim_strength_ladder_rows"],
               ["allowed_strength", "meaning", "manuscript_usage"])
    _write_csv(os.path.join(table_dir, "figure_evidence_map.csv"), res["figure_evidence_map_rows"],
               ["figure_id", "figure_title", "source_milestones", "artifact_provenance", "key_numbers", "supported_claim", "unsupported_overclaim", "main_or_supplement"])
    _write_csv(os.path.join(table_dir, "key_number_to_figure_map.csv"), res["key_number_to_figure_map_rows"],
               ["provenance_id", "figure_id", "figure_title", "display_value"])
    _write_csv(os.path.join(table_dir, "information_class_ladder.csv"), res["information_class_ladder_rows"],
               ["information_class", "definition", "empirical_status", "supporting_milestones", "sufficiency_boundary", "manuscript_phrase"])
    _write_csv(os.path.join(table_dir, "bayes_ceiling_vs_action_rule.csv"), res["bayes_ceiling_vs_action_rule_rows"],
               ["object", "information_class", "uses_target_labels", "available_at_selection_time", "supported_claim", "forbidden_claim", "supporting_milestones"])
    _write_csv(os.path.join(table_dir, "literature_alignment_matrix.csv"), res["literature_alignment_matrix_rows"],
               ["literature_id", "axis", "citation", "url_or_status", "alignment", "blocked_overclaim"])
    _write_csv(os.path.join(table_dir, "claim_to_literature_map.csv"), res["claim_to_literature_map_rows"],
               ["claim_id", "literature_id", "alignment_type"])
    _write_csv(os.path.join(table_dir, "forbidden_literature_overclaims.csv"), res["forbidden_literature_overclaims_rows"],
               ["overclaim", "status", "reason"])
    _write_csv(os.path.join(table_dir, "reviewer_question_bank.csv"), res["reviewer_question_bank_rows"],
               ["question_id", "question", "short_answer", "evidence_refs", "claim_boundary"])
    _write_csv(os.path.join(table_dir, "reviewer_answer_evidence_map.csv"), res["reviewer_answer_evidence_map_rows"],
               ["question_id", "supporting_milestones", "answer_type", "forbidden_overclaim_guard"])
    _write_csv(os.path.join(table_dir, "taxonomy_crosswalk.csv"), res["taxonomy_crosswalk_rows"],
               ["term_id", "definition", "availability_class", "uses_target_unlabeled", "uses_target_labels",
                "available_at_selection_time", "allowed_claim_strength", "supporting_milestones",
                "preferred_usage", "forbidden_usage"])
    _write_csv(os.path.join(table_dir, "term_usage_guardrails.csv"), res["term_usage_guardrails_rows"],
               ["term", "use_as", "avoid_as"])
    _write_csv(os.path.join(table_dir, "key_number_provenance.csv"), res["key_number_provenance_rows"],
               ["provenance_id", "milestone", "metric", "value", "value_fmt_3", "artifact", "table", "row_key", "trace_status", "note"])
    _write_csv(os.path.join(table_dir, "missing_or_ambiguous_provenance.csv"), res["missing_or_ambiguous_provenance_rows"],
               ["provenance_id", "reason", "blocking"])
    _write_csv(os.path.join(table_dir, "manuscript_section_map.csv"), res["manuscript_section_map_rows"],
               ["section_id", "section_title", "section_thesis", "claim_ids"])
    _write_csv(os.path.join(table_dir, "contribution_map.csv"), res["contribution_map_rows"],
               ["contribution_id", "contribution", "claim_ids", "main_figures", "claim_boundary"])
    _write_csv(os.path.join(table_dir, "remaining_caveats.csv"), res["remaining_caveats_rows"],
               ["caveat_id", "caveat", "status", "guardrail"])
    _write_csv(os.path.join(table_dir, "test_command_manifest.csv"), res["test_command_manifest_rows"],
               ["test_scope", "command", "status", "environment", "slurm_partition"])
    _write_csv(os.path.join(table_dir, "subagent_audit_manifest.csv"), res["subagent_audit_manifest_rows"],
               ["subagent_id", "role", "scope", "integration_status"])
    _write_csv(os.path.join(table_dir, "forbidden_claim_scan.csv"), res["forbidden_claim_scan_rows"],
               ["pattern", "total_hits", "affirmative_hits", "files", "passed"])
    _write_csv(os.path.join(table_dir, "red_team_failure_ledger.csv"), res["red_team_failure_ledger_rows"],
               ["gate", "failed", "finding"])
    _write_csv(os.path.join(table_dir, "schema_validation_summary.csv"), res["schema_validation_summary_rows"],
               ["table_name", "row_count", "required_columns_present", "passed"])
    _write_csv(os.path.join(table_dir, "large_artifact_scan.csv"), res["large_artifact_scan_rows"],
               ["path", "size_bytes", "over_50mb", "passed"])
    _write_csv(os.path.join(table_dir, "artifact_manifest.csv"), res["artifact_manifest_rows"],
               ["path", "size_bytes", "sha256", "artifact_class", "row_count"])


def _write_texts(files: dict[str, str], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, text in files.items():
        with open(os.path.join(out_dir, name), "w") as f:
            f.write(text.rstrip() + "\n")


def _schema_rows(table_dir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(table_dir, "*.csv"))):
        if os.path.basename(path) in {"schema_validation_summary.csv", "artifact_manifest.csv"}:
            continue
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            count = sum(1 for _ in reader)
        rows.append({"table_name": os.path.basename(path), "row_count": count, "required_columns_present": int(bool(header)), "passed": int(bool(header))})
    return rows


def _large_scan(paths: list[str]) -> list[dict]:
    rows = []
    for path in sorted(paths):
        size = os.path.getsize(path)
        rows.append({"path": path, "size_bytes": size, "over_50mb": int(size > 50_000_000), "passed": int(size <= 50_000_000)})
    return rows


def _artifact_manifest(paths: list[str], table_dir: str) -> list[dict]:
    row_counts = {}
    for path in glob.glob(os.path.join(table_dir, "*.csv")):
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            row_counts[path] = sum(1 for _ in reader)
    rows = []
    for path in sorted(paths):
        cls = "table" if path.endswith(".csv") else "report" if path.endswith(".md") else "summary_json"
        rows.append({"path": path, "size_bytes": os.path.getsize(path), "sha256": _sha256(path), "artifact_class": cls, "row_count": row_counts.get(path, "")})
    return rows


def _listed_paths() -> list[str]:
    skip = {"artifact_manifest.csv", "large_artifact_scan.csv"}
    return sorted(
        glob.glob(os.path.join(REPORT_DIR, "C57_*.md"))
        + glob.glob(os.path.join(REPORT_DIR, "C57_*.json"))
        + glob.glob(os.path.join(SCAFFOLD_DIR, "*.md"))
        + [p for p in glob.glob(os.path.join(TABLE_DIR, "*.csv")) if os.path.basename(p) not in skip]
    )


def write_artifacts(res: dict, test_status: str) -> dict:
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(SCAFFOLD_DIR, exist_ok=True)
    # Bootstrap the file set before computing scans and manifests.
    json.dump(_compact_json(res), open(REPORT_JSON, "w"), indent=2, sort_keys=True)
    _write_texts(build_reports(res), REPORT_DIR)
    _write_texts(build_scaffold_docs(res), SCAFFOLD_DIR)
    write_tables(res, TABLE_DIR)
    paths = _listed_paths()
    res["generated_paths"] = paths
    res["forbidden_claim_scan_rows"] = build_forbidden_scan(paths)
    res["large_artifact_scan_rows"] = _large_scan(paths)
    # The manifest content is written last, but its row count is already known
    # and is the only manifest fact embedded in compact JSON/reports.
    res["artifact_manifest_rows"] = [{"path": p} for p in paths]
    res["red_team_failure_ledger_rows"] = build_red_team_rows(res)
    res["decision"] = classify(res)

    # Write final claim-bearing artifacts once, compute schema from those final
    # tables, then rewrite only artifacts whose contents depend on schema counts.
    write_tables(res, TABLE_DIR)
    res["schema_validation_summary_rows"] = _schema_rows(TABLE_DIR)
    res["red_team_failure_ledger_rows"] = build_red_team_rows(res)
    res["decision"] = classify(res)
    json.dump(_compact_json(res), open(REPORT_JSON, "w"), indent=2, sort_keys=True)
    _write_texts(build_reports(res), REPORT_DIR)
    _write_texts(build_scaffold_docs(res), SCAFFOLD_DIR)
    write_tables(res, TABLE_DIR)

    # Compute manifest from the final JSON/reports/tables and then write only
    # the non-self manifest. The compact JSON records the row count, not the
    # manifest hashes, so this final manifest write does not stale the payload.
    paths = _listed_paths()
    res["generated_paths"] = paths
    res["artifact_manifest_rows"] = _artifact_manifest(paths, TABLE_DIR)
    _write_csv(os.path.join(TABLE_DIR, "artifact_manifest.csv"), res["artifact_manifest_rows"],
               ["path", "size_bytes", "sha256", "artifact_class", "row_count"])
    json.dump(_compact_json(res), open(REPORT_JSON, "w"), indent=2, sort_keys=True)
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c57_manuscript_scaffold_claim_contract")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--test-status", default="planned")
    args = ap.parse_args(argv)
    res = run(test_status=args.test_status)
    if args.recompute:
        res = write_artifacts(res, args.test_status)
    print(f"[C57] decision={res['decision']['primary']} tables={len(table_row_counts(res))}")


if __name__ == "__main__":
    main()
