"""Single lock-bound execution path for the frozen-field C85E bridge."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .c85e_action_geometry import context_geometry
from .c85e_policy_use import (
    DETERMINISTIC_METHODS, REFERENCE_METHODS, exact_equivalence_scopes,
    summarize_stochastic_q0_context,
)
from .c85e_rank_topk_regret import (
    geometry_regret_associations, join_measurement_and_geometry,
    leave_one_target_sign_stability, target_equal_rows,
)
from .c85e_result_manifest import REGISTERED_TABLES, publish_result_bundle
from .c85e_robust_risk import (
    CVAR_ALPHA_GRID, robust_risk_profile, target_equal_regrets,
)
from .c85e_runtime_guard import (
    ValidatedC85EExecutionContext, create_validated_execution_context,
    require_registered_path, revalidate_execution_context, sha256_file,
)
from .c85e_theorem_bridge import (
    assumption_identification_ledger, forbidden_transfer_claims,
    theorem_applicability_matrix,
)


C85U_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c85u-candidate-utility-v2/"
    "c85u-v2-77382c16a593f7c2-91a428488a634268"
)
SELECTION_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c84s-analysis-v5/"
    "stage_b_selection_freeze"
)
RESULT_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c84s-analysis-v5/"
    "stage_c_scientific_result"
)
CONTEXT_FIELDS = (
    "dataset", "target_subject_id", "panel", "training_seed", "level",
)
SCORE_METHODS = ("S1", "U5", "U7", "U11", "U13", "U14", "U15")
FIXED_METHODS = ("B1", "B2", "B3", "B4O", "B4S")


class C85EExecutionError(RuntimeError):
    """Raised on any input, analysis, or publication contract violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C85EExecutionError(message)


def _identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in CONTEXT_FIELDS}


def _key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in CONTEXT_FIELDS)


@dataclass(frozen=True)
class UtilityContext:
    context_id: str
    dataset: str
    target_subject_id: str
    panel: str
    training_seed: int
    level: int
    utility: np.ndarray
    candidate_ids: tuple[str, ...]
    regimes: tuple[str, ...]

    def identity(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset, "target_subject_id": self.target_subject_id,
            "panel": self.panel, "training_seed": self.training_seed, "level": self.level,
        }


@dataclass(frozen=True)
class FrozenQ0Actions:
    context_id: str
    method_id: str
    selected_actions: np.ndarray


@dataclass(frozen=True)
class C85EInputBundle:
    contexts: Mapping[str, UtilityContext]
    method_rows: tuple[dict[str, Any], ...]
    deterministic_actions: Mapping[str, Mapping[str, int]]
    q0_actions: tuple[FrozenQ0Actions, ...]
    compact_tables: Mapping[str, tuple[dict[str, str], ...]]
    input_replay_sha256: str


def _read_csv_registered(
    context: ValidatedC85EExecutionContext, path: Path,
) -> list[dict[str, str]]:
    require_registered_path(context, path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_utility_contexts(
    context: ValidatedC85EExecutionContext,
) -> dict[str, UtilityContext]:
    manifest_path = C85U_ROOT / "stage_u1_candidate_utility_v2/C85U_CANDIDATE_UTILITY_MANIFEST.json"
    require_registered_path(context, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("context_artifacts")
    _require(isinstance(rows, list) and len(rows) == 944, "C85E utility context coverage drift")
    output: dict[str, UtilityContext] = {}
    for row in rows:
        path = C85U_ROOT / "stage_u1_candidate_utility_v2" / str(row["path"])
        require_registered_path(context, path)
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "context_id", "dataset", "target_subject_id", "panel", "training_seed",
                "level", "candidate_index", "candidate_id", "regime", "composite_utility",
            }
            _require(required.issubset(archive.files), "C85E utility context field drift")
            context_id = str(archive["context_id"].item())
            candidate_index = np.asarray(archive["candidate_index"], dtype=np.int64)
            utility = np.asarray(archive["composite_utility"], dtype=np.float64)
            candidate_ids = tuple(map(str, archive["candidate_id"].tolist()))
            regimes = tuple(map(str, archive["regime"].tolist()))
            _require(np.array_equal(candidate_index, np.arange(81)) and utility.shape == (81,) and
                     np.all(np.isfinite(utility)) and len(candidate_ids) == len(regimes) == 81,
                     "C85E utility context action identity drift")
            item = UtilityContext(
                context_id=context_id, dataset=str(archive["dataset"].item()),
                target_subject_id=str(archive["target_subject_id"].item()),
                panel=str(archive["panel"].item()),
                training_seed=int(archive["training_seed"].item()),
                level=int(archive["level"].item()), utility=utility.copy(),
                candidate_ids=candidate_ids, regimes=regimes,
            )
        _require(context_id == str(row["context_id"]) and context_id not in output,
                 "C85E utility context identity drift")
        output[context_id] = item
    return output


def _load_actions(
    context: ValidatedC85EExecutionContext,
    contexts: Mapping[str, UtilityContext],
) -> tuple[dict[str, dict[str, int]], tuple[FrozenQ0Actions, ...]]:
    actions: dict[str, dict[str, int]] = {context_id: {} for context_id in contexts}
    ranks = _read_csv_registered(context, SELECTION_ROOT / "candidate_ranks.csv")
    for row in ranks:
        if int(row["rank"]) != 1:
            continue
        context_id, method = row["context_id"], row["method_id"]
        candidate = int(row["candidate_index"])
        _require(context_id in contexts and method in SCORE_METHODS and
                 contexts[context_id].candidate_ids[candidate] == row["candidate_id"],
                 "C85E rank-one action identity drift")
        actions[context_id][method] = candidate
    fixed = _read_csv_registered(context, SELECTION_ROOT / "fixed_default_selections.csv")
    for row in fixed:
        context_id, method = row["context_id"], row["method_id"]
        candidate = int(row["selected_candidate_index"])
        _require(context_id in contexts and method in FIXED_METHODS and
                 contexts[context_id].candidate_ids[candidate] == row["selected_candidate_id"],
                 "C85E fixed action identity drift")
        actions[context_id][method] = candidate
    q0_index = _read_csv_registered(context, SELECTION_ROOT / "q0_selection_shard_index.csv")
    q0_actions: list[FrozenQ0Actions] = []
    for row in q0_index:
        context_id = row["context_id"]
        _require(context_id in contexts, "C85E Q0 context identity drift")
        path = SELECTION_ROOT / row["path"]
        require_registered_path(context, path)
        with np.load(path, allow_pickle=False) as archive:
            candidate_ids = tuple(value.decode("ascii") for value in archive["candidate_ids"].tolist())
            _require(candidate_ids == contexts[context_id].candidate_ids,
                     "C85E Q0 candidate identity drift")
            full_order = np.asarray(archive["FULL_candidate_order"], dtype=np.uint8)
            _require(full_order.shape == (1, 81), "C85E Q0 FULL shape drift")
            actions[context_id]["Q0_FULL"] = int(full_order[0, 0])
            codes = np.asarray(archive["finite_budget_code"], dtype=np.uint8)
            orders = np.asarray(archive["finite_candidate_order"], dtype=np.uint8)
            for budget in sorted(set(map(int, codes.tolist()))):
                selected = orders[codes == budget, 0].copy()
                _require(selected.shape == (2048,), "C85E finite Q0 chain coverage drift")
                q0_actions.append(FrozenQ0Actions(
                    context_id=context_id, method_id=f"Q0_B{budget}",
                    selected_actions=selected,
                ))
    _require(len(actions) == 944 and all(set(value) == set(DETERMINISTIC_METHODS)
                                        for value in actions.values()),
             "C85E deterministic action coverage drift")
    return actions, tuple(q0_actions)


def _parse_method_rows(
    context: ValidatedC85EExecutionContext,
) -> tuple[dict[str, Any], ...]:
    rows = _read_csv_registered(context, RESULT_ROOT / "method_context_decisions.csv")
    output: list[dict[str, Any]] = []
    float_fields = (
        "standardized_regret", "selected_utility", "source_relative_regret_gain",
        "top1", "top5", "top10", "coverage",
    )
    optional_float = (
        "Spearman", "Kendall", "pairwise_ordering_accuracy", "accuracy_estimation_MAE",
    )
    for row in rows:
        item: dict[str, Any] = dict(row)
        item["training_seed"] = int(item["training_seed"])
        item["level"] = int(item["level"])
        for field in float_fields:
            item[field] = float(item[field])
        item["rank_measurement_applicable"] = int(item["rank_measurement_applicable"])
        item["performance_estimate_applicable"] = int(item["performance_estimate_applicable"])
        for field in optional_float:
            item[field] = None if item[field] == "" else float(item[field])
        if not item["rank_measurement_applicable"]:
            _require(all(item[field] is None for field in optional_float[:3]),
                     "C85E inapplicable rank measurement is not null")
        output.append(item)
    _require(len(output) == 18_432, "C85E method-context row coverage drift")
    return tuple(output)


def load_frozen_input_bundle(context: ValidatedC85EExecutionContext) -> C85EInputBundle:
    """Load only the lock-bound post-C84S derived and compact artifacts."""
    revalidate_execution_context(context)
    contexts = _load_utility_contexts(context)
    actions, q0_actions = _load_actions(context, contexts)
    method_rows = _parse_method_rows(context)
    compact_names = (
        "label_budget_frontier.csv", "level_specific_Q1_Q2.csv",
        "target_level_method_effects.csv", "panel_seed_stability.csv",
    )
    compact = {
        name: tuple(_read_csv_registered(context, RESULT_ROOT / name))
        for name in compact_names
    }
    digest = hashlib.sha256()
    for item in sorted(context.inputs, key=lambda value: value.object_id):
        digest.update(item.object_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.sha256.encode("ascii"))
        digest.update(b"\n")
    return C85EInputBundle(
        contexts=contexts, method_rows=method_rows, deterministic_actions=actions,
        q0_actions=q0_actions, compact_tables=compact,
        input_replay_sha256=digest.hexdigest(),
    )


def _method_row_map(bundle: C85EInputBundle) -> dict[tuple[Any, ...], dict[str, Any]]:
    rows = {
        (*_key(row), str(row["method_id"])): dict(row)
        for row in bundle.method_rows
    }
    _require(len(rows) == len(bundle.method_rows), "duplicate method-context identity")
    return rows


def _deterministic_rows(
    bundle: C85EInputBundle,
) -> dict[str, list[dict[str, Any]]]:
    historical = _method_row_map(bundle)
    output = {method: [] for method in DETERMINISTIC_METHODS}
    for context_id, context in bundle.contexts.items():
        identity = context.identity()
        for method, selected in bundle.deterministic_actions[context_id].items():
            row = historical[(*_key(identity), method)]
            output[method].append({
                **identity, "method_id": method,
                "selected_candidate_index": int(selected),
                "selected_regime": context.regimes[int(selected)],
                "standardized_regret": float(row["standardized_regret"]),
            })
    return output


def _geometry_aggregates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "best_second_raw_utility_gap", "best_fifth_raw_utility_gap",
        "best_tenth_raw_utility_gap", "utility_range", "exact_comaximizer_count",
    )
    targets: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        targets[(str(row["dataset"]), str(row["target_subject_id"]), int(row["level"]))].append(row)
    target_rows = []
    for (dataset, target, level), group in sorted(targets.items()):
        _require(len(group) == 4, "C85E target geometry repeat drift")
        target_rows.append({
            "dataset": dataset, "target_subject_id": target, "level": level,
            **{field: float(np.mean([float(row[field]) for row in group])) for field in fields},
        })
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in target_rows:
        grouped[(str(row["dataset"]), int(row["level"]))].append(row)
    return [{
        "dataset": dataset, "level": level, "target_count": len(group),
        **{f"mean_{field}": float(np.mean([float(row[field]) for row in group])) for field in fields},
        "weighting": "TARGET_EQUAL", "result_tag": "POST_C84S_EXPLORATORY",
    } for (dataset, level), group in sorted(grouped.items())]


def _panel_seed_geometry(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = ("best_second_raw_utility_gap", "utility_range", "exact_comaximizer_count")
    grouped: dict[tuple[str, str, int], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[(str(row["dataset"]), str(row["panel"]), int(row["training_seed"]))][
            str(row["target_subject_id"])
        ].append(row)
    output = []
    for (dataset, panel, seed), targets in sorted(grouped.items()):
        target_values = []
        for group in targets.values():
            _require(len(group) == 2, "C85E panel/seed target level coverage drift")
            target_values.append({field: np.mean([float(row[field]) for row in group]) for field in fields})
        output.append({
            "dataset": dataset, "panel": panel, "training_seed": seed,
            "target_count": len(target_values),
            **{f"mean_{field}": float(np.mean([row[field] for row in target_values])) for field in fields},
            "weighting": "TARGET_EQUAL_AFTER_LEVEL_MEAN",
            "result_tag": "POST_C84S_EXPLORATORY",
        })
    return output


def _support_and_frontier_rows(
    policy_rows: Sequence[Mapping[str, Any]],
    frontier_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep policy-use and frozen frontier components in one typed long table."""
    fields = (
        "object_type", "dataset", "level", "method_id", "reference_id",
        "action_divergence_rate", "exact_collapse", "budget", "mean_effect",
        "maxT_pvalue", "direct_qualification", "closure_qualification", "Bstar",
        "level0_Bstar", "level1_Bstar", "result_tag",
    )
    output: list[dict[str, Any]] = []
    for row in policy_rows:
        value = {
            "object_type": "REALIZED_POLICY_USE", "dataset": row["dataset"],
            "level": row["level"], "method_id": row["method_id"],
            "reference_id": row["reference_id"],
            "action_divergence_rate": row["action_divergence_rate"],
            "exact_collapse": row["exact_collapse"], "budget": None,
            "mean_effect": None, "maxT_pvalue": None,
            "direct_qualification": None, "closure_qualification": None,
            "Bstar": None, "level0_Bstar": None, "level1_Bstar": None,
            "result_tag": "POST_C84S_EXPLORATORY",
        }
        output.append({field: value[field] for field in fields})
    for row in frontier_rows:
        value = {
            "object_type": "FROZEN_LABEL_FRONTIER_COMPONENT",
            "dataset": row["dataset"], "level": "ALL", "method_id": "Q0",
            "reference_id": "S1", "action_divergence_rate": None,
            "exact_collapse": None, "budget": row["budget"],
            "mean_effect": float(row["mean_effect"]),
            "maxT_pvalue": float(row["maxT_pvalue"]),
            "direct_qualification": int(row["direct_qualification"]),
            "closure_qualification": int(row["closure_qualification"]),
            "Bstar": row["Bstar"], "level0_Bstar": row["level0_Bstar"],
            "level1_Bstar": row["level1_Bstar"],
            "result_tag": "POST_C84S_EXPLORATORY",
        }
        output.append({field: value[field] for field in fields})
    return output


def _q0_rows(
    bundle: C85EInputBundle,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for frozen in bundle.q0_actions:
        context = bundle.contexts[frozen.context_id]
        b1 = int(bundle.deterministic_actions[frozen.context_id]["B1"])
        regimes = [context.regimes[int(action)] for action in frozen.selected_actions]
        row = summarize_stochastic_q0_context(
            selected_actions=frozen.selected_actions, selected_regimes=regimes,
            reference_action=b1,
            identity={**context.identity(), "method_id": frozen.method_id, "aggregation": "CONTEXT"},
        )
        output.append(row)
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in output:
        grouped[(str(row["dataset"]), str(row["target_subject_id"]), str(row["method_id"]))].append(row)
    aggregate = []
    for (dataset, target, method), group in sorted(grouped.items()):
        _require(len(group) == 8, "C85E Q0 target context coverage drift")
        action_keys = sorted({
            key for row in group for key in row["action_distribution"]
        }, key=int)
        regime_keys = sorted({
            key for row in group for key in row["regime_distribution"]
        })
        aggregate.append({
            "dataset": dataset, "target_subject_id": target, "panel": "ALL",
            "training_seed": "ALL", "level": "ALL", "method_id": method,
            "aggregation": "TARGET_EQUAL_CONTEXT_MEAN", "chains": 2048,
            "probability_action_differs_from_reference": float(np.mean([
                row["probability_action_differs_from_reference"] for row in group
            ])),
            "action_entropy": float(np.mean([row["action_entropy"] for row in group])),
            "action_distribution": {
                key: float(np.mean([
                    row["action_distribution"].get(key, 0.0) for row in group
                ])) for key in action_keys
            },
            "regime_distribution": {
                key: float(np.mean([
                    row["regime_distribution"].get(key, 0.0) for row in group
                ])) for key in regime_keys
            },
            "stochastic_policy_preserved": True, "chains_are_scientific_sample": False,
            "result_tag": "POST_C84S_EXPLORATORY",
        })
    return output + aggregate


def _risk_tables(bundle: C85EInputBundle) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    full = target_equal_regrets(bundle.method_rows, level=None)
    levels = target_equal_regrets(bundle.method_rows, level=0) + target_equal_regrets(
        bundle.method_rows, level=1
    )
    targets = full + levels
    profiles = robust_risk_profile(targets)
    target_lookup = {
        (row["dataset"], row["target_subject_id"], str(row["level"]), row["method_id"]): row
        for row in targets
    }
    for profile in profiles:
        dataset, method, level = profile["dataset"], profile["method_id"], str(profile["level"])
        methods = [
            row for key, row in target_lookup.items()
            if key[0] == dataset and key[2] == level and key[3] == method
        ]
        for reference in ("S1", "B1"):
            improvements = []
            for row in methods:
                ref = target_lookup[(dataset, row["target_subject_id"], level, reference)]
                improvements.append(
                    float(ref["standardized_target_regret"]) - float(row["standardized_target_regret"])
                )
            profile[f"mean_target_improvement_vs_{reference}"] = float(np.mean(improvements))
    cvar_rows = [{
        "dataset": row["dataset"], "method_id": row["method_id"], "level": row["level"],
        "target_count": row["target_count"], "alpha": alpha,
        "upper_loss_empirical_CVaR": row[f"CVaR_{alpha:.2f}"],
        "risk_scale": "HISTORICAL_C84_STANDARDIZED_REGRET",
        "fractional_boundary_mass": True, "result_tag": "POST_C84S_EXPLORATORY",
    } for row in profiles for alpha in CVAR_ALPHA_GRID]
    return profiles, cvar_rows


def build_analysis_tables(bundle: C85EInputBundle) -> dict[str, list[dict[str, Any]]]:
    """Build every registered table from a prevalidated real or shadow bundle."""
    geometry_summary: list[dict[str, Any]] = []
    near_rows: list[dict[str, Any]] = []
    multiplicity_rows: list[dict[str, Any]] = []
    for context in bundle.contexts.values():
        geometry = context_geometry(context.utility, identity=context.identity())
        geometry_summary.append(dict(geometry["summary"]))
        near_rows.extend(map(dict, geometry["near_optimal"]))
        multiplicity_rows.extend(map(dict, geometry["multiplicity"]))

    deterministic = _deterministic_rows(bundle)
    policy_rows: list[dict[str, Any]] = []
    for method in DETERMINISTIC_METHODS:
        for reference in REFERENCE_METHODS:
            policy_rows.extend(exact_equivalence_scopes(
                deterministic[method], deterministic[reference],
                method_id=method, reference_id=reference,
            ))
    exact_rows = [{
        key: row[key] for key in (
            "scope", "dataset", "level", "method_id", "reference_id", "contexts",
            "exact_equivalence_contexts", "exact_collapse", "T3_exactly_applicable", "result_tag",
        )
    } for row in policy_rows]
    entropy_rows = [{
        "scope": row["scope"], "dataset": row["dataset"], "level": row["level"],
        "method_id": row["method_id"], "reference_id": row["reference_id"],
        "canonical_action_entropy": row["canonical_action_entropy"],
        "selected_regime_distribution": row["selected_regime_distribution"],
        "result_tag": row["result_tag"],
    } for row in policy_rows]
    divergent_rows = [{
        "scope": row["scope"], "dataset": row["dataset"], "level": row["level"],
        "method_id": row["method_id"], "reference_id": row["reference_id"],
        "mean_standardized_regret_difference": row["mean_standardized_regret_difference"],
        "divergent_context_regret_difference": row["divergent_context_regret_difference"],
        "divergent_context_risk_contribution_fraction": row["divergent_context_risk_contribution_fraction"],
        "risk_scale": row["risk_scale"], "result_tag": row["result_tag"],
    } for row in policy_rows]

    geometry_by_level = _geometry_aggregates(geometry_summary)
    historical_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bundle.method_rows:
        historical_by_method[str(row["method_id"])].append(row)
    joined: list[dict[str, Any]] = []
    for method, rows in historical_by_method.items():
        divergence = {}
        if method in deterministic:
            reference = {_key(row): row for row in deterministic["B1"]}
            divergence = {
                _key(row): float(int(row["selected_candidate_index"] != reference[_key(row)]["selected_candidate_index"]))
                for row in deterministic[method]
            }
        joined.extend(join_measurement_and_geometry(
            rows, geometry_summary,
            divergence_by_context=divergence if divergence else None,
        ))
    target_joined = target_equal_rows(
        joined,
        value_fields=(
            "top1", "top5", "top10", "selected_standardized_regret", "selected_utility",
            "action_divergence", "best_second_raw_utility_gap", "best_fifth_raw_utility_gap",
            "best_tenth_raw_utility_gap", "utility_range", "Spearman", "Kendall",
            "pairwise_ordering_accuracy",
        ),
    )
    for row in target_joined:
        method = row["method_id"]
        reference = next(
            value for value in target_joined
            if value["dataset"] == row["dataset"] and
            value["target_subject_id"] == row["target_subject_id"] and
            value["level"] == row["level"] and value["method_id"] == "B1"
        )
        row["regret_effect_vs_B1"] = float(reference["selected_standardized_regret"]) - float(
            row["selected_standardized_regret"]
        )
        row["method_id"] = method
    associations = geometry_regret_associations(
        target_joined,
        geometry_fields=(
            "best_second_raw_utility_gap", "best_fifth_raw_utility_gap",
            "best_tenth_raw_utility_gap", "utility_range",
        ),
    )
    leave_target = leave_one_target_sign_stability(
        target_joined, effect_field="regret_effect_vs_B1",
    )
    profiles, cvar_rows = _risk_tables(bundle)
    q0_rows = _q0_rows(bundle)
    exact_scopes = [row for row in policy_rows if row["exact_collapse"]]
    theorem_rows = theorem_applicability_matrix(exact_scopes)

    support_policy_rows = [
        row for row in policy_rows if row["scope"] == "dataset_x_level"
    ]
    support_rows = _support_and_frontier_rows(
        support_policy_rows, bundle.compact_tables["label_budget_frontier.csv"],
    )
    heterogeneity_limits = [{
        "proposed_explanation": explanation, "status": "NOT_CAUSALLY_IDENTIFIED",
        "reason": reason, "result_tag": "POST_C84S_EXPLORATORY",
    } for explanation, reason in (
        ("support_deletion_changes_multiplicity", "Descriptive association cannot establish mediation."),
        ("candidate_geometry_causes_policy_success", "Held-evaluation geometry is unavailable at selection time."),
        ("dataset_identity_causes_tail_failure", "Dataset contrasts are observational and post-outcome."),
    )]
    active_requirements = [{
        "requirement": name, "required_for_future_protocol": True,
        "available_in_C85E": available, "execution_authorized": False,
        "result_tag": "POST_C84S_EXPLORATORY",
    } for name, available in (
        ("trial_level_full_loss_vector_variance", False),
        ("untouched_target_population", False),
        ("prospective_primary_decision_endpoints", False),
        ("observation_and_stopping_policy", False),
        ("importance_weighting_contract", False),
        ("frozen_candidate_zoo_identity", True),
    )]
    untouched = [{
        "option": option, "currently_untouched": status,
        "selection_permitted_in_C85E": False,
        "notes": notes, "result_tag": "POST_C84S_EXPLORATORY",
    } for option, status, notes in (
        ("new_external_cohort", "UNKNOWN_REQUIRES_PROSPECTIVE_REGISTRATION", "Must not be selected from C84 outcomes."),
        ("new_EEG_paradigm", "UNKNOWN_REQUIRES_PROSPECTIVE_REGISTRATION", "Requires separate harmonization and PM review."),
        ("held_future_targets", "NOT_CREATED", "Must be fixed before active-policy development."),
    )]
    tables: dict[str, list[dict[str, Any]]] = {
        "realized_action_divergence.csv": policy_rows,
        "exact_policy_equivalence_classes.csv": exact_rows,
        "action_entropy_and_regime_distribution.csv": entropy_rows,
        "divergent_context_risk_contribution.csv": divergent_rows,
        "q0_action_distribution_use.csv": q0_rows,
        "candidate_gap_geometry.csv": geometry_summary,
        "near_optimal_set_grid.csv": near_rows,
        "effective_multiplicity_grid.csv": multiplicity_rows,
        "geometry_by_dataset_level.csv": geometry_by_level,
        "rank_topk_regret_geometry_separation.csv": target_joined,
        "geometry_regret_descriptive_association.csv": associations,
        "leave_target_geometry_stability.csv": leave_target,
        "target_robust_risk_profile.csv": profiles,
        "cvar_grid.csv": cvar_rows,
        "cott_mean_tail_profile.csv": [row for row in profiles if row["method_id"] == "U13"],
        "mano_policy_collapse_risk_profile.csv": [row for row in profiles if row["method_id"] == "U11"],
        "level_robust_risk_interaction.csv": [row for row in profiles if str(row["level"]) != "ALL"],
        "dataset_level_geometry_matrix.csv": geometry_by_level,
        "panel_seed_geometry_stability.csv": _panel_seed_geometry(geometry_summary),
        "support_level_policy_use_profile.csv": support_rows,
        "heterogeneity_explanation_limits.csv": heterogeneity_limits,
        "theorem_empirical_applicability_matrix.csv": theorem_rows,
        "assumption_identification_ledger.csv": assumption_identification_ledger(),
        "forbidden_theorem_transfer_claims.csv": forbidden_transfer_claims(),
        "future_active_acquisition_requirements.csv": active_requirements,
        "untouched_population_options.csv": untouched,
    }
    _require(set(tables) == set(REGISTERED_TABLES) and all(tables.values()),
             "C85E registered analysis table coverage drift")
    return tables


def synthesis_markdown() -> str:
    return """# C85E Restricted Policy and Information Value Synthesis

All rows are `POST_C84S_EXPLORATORY`. The analysis distinguishes the available
information experiment, the registered policy class, its realized action map,
held-evaluation action-set geometry, and the chosen target-risk functional.

A richer-input registered policy may collapse to a fixed action without
implying that its information experiment is uninformative. Dense near-optimal
sets can make low top-1 localization compatible with low selected regret, while
mean improvement can coexist with adverse target tails. Fixed passive-label
policies measure realized registered-policy use, not unrestricted label value.

The analysis does not identify a Blackwell order, minimax optimality, the T4 or
T7 assumptions, or an information-theoretic lower bound. It does not change
C84-D, C84-L4, or any C85 theorem status.
"""


def run_locked(
    *, execution_lock: Path, authorization_record: Path, output_root: Path,
) -> dict[str, Any]:
    context = create_validated_execution_context(
        execution_lock=execution_lock, authorization_record=authorization_record,
        output_root=output_root,
    )
    bundle = load_frozen_input_bundle(context)
    tables = build_analysis_tables(bundle)
    return publish_result_bundle(
        context=context, tables=tables, synthesis_markdown=synthesis_markdown(),
        input_replay_sha256=bundle.input_replay_sha256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-locked")
    run.add_argument("--execution-lock", type=Path, required=True)
    run.add_argument("--authorization-record", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_locked(
        execution_lock=args.execution_lock.resolve(),
        authorization_record=args.authorization_record.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(result["gate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "C85EExecutionError", "C85EInputBundle", "FrozenQ0Actions", "UtilityContext",
    "build_analysis_tables", "load_frozen_input_bundle", "run_locked",
    "synthesis_markdown",
]
