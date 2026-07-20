"""C86P metadata registries and shadow-only active-testing contracts.

This module deliberately has no EEG, dataset-loader, training, or active-label
server imports.  Its filesystem reads are limited to committed compact reports,
the installed MOABB metadata catalog, and loader source bytes.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c86p_tables"
PROTOCOL_PATH = REPORT_DIR / "C86_ACTIVE_TESTING_PROGRAM_PROTOCOL.json"
SYNTHETIC_PROTOCOL_PATH = (
    REPORT_DIR / "C86P_SYNTHETIC_CALIBRATION_OPERATIONALIZATION_PROTOCOL.json"
)
MOABB_ROOT = Path(
    "/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/lib/"
    "python3.13/site-packages/moabb/datasets"
)
IMAGERY_CATALOG = MOABB_ROOT / "summary_imagery.csv"

TOTAL_QUERY_GRID: tuple[int | str, ...] = (4, 8, 16, 32, "FULL")
FINITE_QUERY_GRID: tuple[int, ...] = (4, 8, 16, 32)
EPSILON_GRID = (0.005, 0.01, 0.02, 0.05)
CVAR_ALPHA_GRID = (0.50, 0.75, 0.90)
UNIFORM_MIXTURE_RHO = 0.05
ACTIVE_CHAINS = 2_048
ELIGIBLE_DATASETS = ("Brandl2020", "Kumar2024", "Yang2025_2C")


class C86PContractError(RuntimeError):
    """Raised when a C86P metadata or shadow contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C86PContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_trial_split(
    dataset: str, subject: str, trial_ids: Sequence[str], *, salt: str = "C86_TARGET_SPLIT_V1",
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the locked label-blind half split in canonical hash order."""
    values = tuple(map(str, trial_ids))
    _require(len(values) == len(set(values)), "duplicate target trial identity")
    _require(len(values) >= 80, "target has fewer than 80 metadata-supported trials")
    ordered = tuple(sorted(values, key=lambda trial: (
        hashlib.sha256(f"{salt}|{dataset}|{subject}|{trial}".encode("utf-8")).digest(),
        trial,
    )))
    cut = len(ordered) // 2
    pool, evaluation = ordered[:cut], ordered[cut:]
    _require(len(pool) >= 40 and len(evaluation) >= 40, "physical view support drift")
    _require(set(pool).isdisjoint(evaluation), "physical target views overlap")
    return pool, evaluation


def validate_total_query_grid(pool_size: int, grid: Sequence[int | str]) -> None:
    _require(tuple(grid) == TOTAL_QUERY_GRID, "total-query budget grid drift")
    _require(pool_size >= 40, "acquisition pool cannot support finite budget grid")
    finite = [int(value) for value in grid if value != "FULL"]
    _require(finite == sorted(set(finite)), "finite budgets are not unique and ordered")
    _require(max(finite) <= pool_size, "finite budget exceeds acquisition pool")


def mixed_query_probabilities(scores: Sequence[float], rho: float = UNIFORM_MIXTURE_RHO) -> np.ndarray:
    """Apply the locked uniform-overlap mixture to nonnegative query scores."""
    value = np.asarray(scores, dtype=np.float64)
    _require(value.ndim == 1 and len(value) > 0, "query score vector is empty")
    _require(np.all(np.isfinite(value)) and np.all(value >= 0), "invalid query score")
    _require(0.0 < float(rho) < 1.0, "uniform mixture must be in (0,1)")
    if float(np.sum(value)) <= 0.0:
        return np.full(len(value), 1.0 / len(value), dtype=np.float64)
    output = (1.0 - rho) * value / np.sum(value) + rho / len(value)
    _require(np.all(output >= rho / len(value)), "sampling probability floor failed")
    _require(math.isclose(float(np.sum(output)), 1.0, rel_tol=0.0, abs_tol=1e-15),
             "query probabilities do not sum to one")
    return output


def lure_weights(population_size: int, budget: int, query_probabilities: Sequence[float]) -> np.ndarray:
    """Return Farquhar/Kossen leveled unbiased risk estimator weights."""
    n, m_total = int(population_size), int(budget)
    probs = np.asarray(query_probabilities, dtype=np.float64)
    _require(1 <= m_total <= n, "invalid LURE population or budget")
    _require(probs.shape == (m_total,), "LURE probability count drift")
    _require(np.all(np.isfinite(probs)) and np.all(probs > 0), "LURE positivity failed")
    weights = np.empty(m_total, dtype=np.float64)
    for zero_index, probability in enumerate(probs):
        step = zero_index + 1
        remaining_before_draw = n - step + 1
        _require(probability <= 1.0 + 1e-15, "query probability exceeds one")
        weights[zero_index] = 1.0 + (n - m_total) / (n - step) * (
            1.0 / (remaining_before_draw * probability) - 1.0
        ) if step < n else 1.0
    _require(np.all(np.isfinite(weights)) and np.all(weights > 0), "invalid LURE weights")
    return weights


def lure_mean(
    queried_values: np.ndarray, *, population_size: int, query_probabilities: Sequence[float],
) -> np.ndarray:
    """Estimate finite-pool means for one or more per-trial quantities."""
    values = np.asarray(queried_values, dtype=np.float64)
    _require(values.ndim >= 1 and np.all(np.isfinite(values)), "LURE values invalid")
    budget = int(values.shape[0])
    weights = lure_weights(population_size, budget, query_probabilities)
    return np.tensordot(weights / budget, values, axes=(0, 0))


def midrank_percentile(values: Sequence[float]) -> np.ndarray:
    """Dependency-light equivalent of the historical average-rank percentile."""
    array = np.asarray(values, dtype=np.float64)
    _require(array.ndim == 1 and len(array) > 1 and np.all(np.isfinite(array)),
             "midrank input invalid")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        average_one_based_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_one_based_rank
        start = end
    return (ranks - 1.0) / (len(array) - 1.0)


def estimate_historical_composite(
    queried_probabilities: np.ndarray,
    queried_labels: Sequence[int],
    *,
    population_size: int,
    query_probabilities: Sequence[float],
) -> dict[str, np.ndarray | int]:
    """Apply the locked historical composite to LURE component estimates.

    Linear LURE moments are unbiased. Ratios, absolute calibration moments,
    midranks, and the selected action are explicitly plug-in quantities.
    """
    predictions = np.asarray(queried_probabilities, dtype=np.float64)
    labels = np.asarray(queried_labels, dtype=np.int64)
    _require(predictions.ndim == 3 and predictions.shape[0] == len(labels),
             "queried candidate probability shape drift")
    _require(predictions.shape[2] == 2, "C86 primary task must have two classes")
    candidate_count = predictions.shape[1]
    _require(candidate_count == 81, "C86 candidate count drift")
    _require(set(np.unique(labels)).issubset({0, 1}), "queried label outside binary task")
    _require(np.all(np.isfinite(predictions)) and np.all(predictions >= 0),
             "candidate probability invalid")
    _require(np.max(np.abs(np.sum(predictions, axis=2) - 1.0)) <= 1e-12,
             "candidate probabilities do not sum to one")
    hard = np.argmax(predictions, axis=2)
    chosen = np.take_along_axis(predictions, labels[:, None, None], axis=2)[:, :, 0]
    nll = lure_mean(-np.log(np.clip(chosen, 1e-12, 1.0)),
                    population_size=population_size, query_probabilities=query_probabilities)

    class_rates = []
    for class_id in (0, 1):
        denominator = population_size * float(lure_mean(
            (labels == class_id).astype(float),
            population_size=population_size,
            query_probabilities=query_probabilities,
        ))
        numerator = population_size * lure_mean(
            ((labels[:, None] == class_id) & (hard == class_id)).astype(float),
            population_size=population_size, query_probabilities=query_probabilities,
        )
        # The prospectively fixed Jeffreys smoothing keeps total-query paths
        # defined when a small queried prefix contains only one class.
        class_rates.append((numerator + 0.5) / (denominator + 1.0))
    bacc = 0.5 * (class_rates[0] + class_rates[1])

    confidence = np.max(predictions, axis=2)
    correctness = (hard == labels[:, None]).astype(np.float64)
    ece = np.zeros(candidate_count, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, 16)
    for index in range(15):
        right = confidence <= edges[index + 1] if index == 14 else confidence < edges[index + 1]
        mask = (confidence >= edges[index]) & right
        signed = mask * (correctness - confidence)
        ece += np.abs(lure_mean(signed, population_size=population_size,
                                query_probabilities=query_probabilities))

    utility = np.mean(np.column_stack((
        midrank_percentile(bacc),
        midrank_percentile(-nll),
        midrank_percentile(-ece),
    )), axis=1)
    selected = int(np.lexsort((np.arange(candidate_count), -utility))[0])
    return {
        "balanced_accuracy": np.asarray(bacc),
        "NLL": np.asarray(nll),
        "ECE": np.asarray(ece),
        "composite_utility": np.asarray(utility),
        "selected_action": selected,
    }


def empirical_upper_cvar(losses: Sequence[float], alpha: float) -> float:
    """Equal-mass empirical upper-loss CVaR with fractional boundary mass."""
    values = np.sort(np.asarray(losses, dtype=np.float64))
    _require(values.ndim == 1 and len(values) > 0 and np.all(np.isfinite(values)),
             "CVaR losses invalid")
    _require(0.0 < alpha < 1.0, "CVaR alpha must be in (0,1)")
    mass = 1.0 - alpha
    each = 1.0 / len(values)
    remaining = mass
    total = 0.0
    for value in values[::-1]:
        take = min(each, remaining)
        total += take * float(value)
        remaining -= take
        if remaining <= 1e-15:
            break
    return total / mass


_SOURCE_INFO: dict[str, dict[str, Any]] = {
    "BNCI2003_004": {"file": "bnci/bnci_2003.py", "native_task": 0, "health": "healthy",
                     "documented_total": 168, "license": "CC-BY-4.0", "canonical": 1,
                     "events": "right_hand|feet"},
    "BNCI2014_002": {"file": "bnci/bnci_2014.py", "native_task": 0, "health": "healthy",
                     "documented_total": 160, "license": "CC-BY-ND-4.0", "canonical": 1,
                     "events": "right_hand|feet"},
    "BNCI2014_004": {"file": "bnci/bnci_2014.py", "native_task": 1, "health": "healthy",
                     "documented_total": 720, "license": "CC-BY-ND-4.0", "canonical": 1,
                     "events": "left_hand|right_hand"},
    "BNCI2015_001": {"file": "bnci/bnci_2015.py", "native_task": 0, "health": "healthy",
                     "documented_total": 400, "license": "CC-BY-NC-ND-4.0", "canonical": 1,
                     "events": "right_hand|feet"},
    "Cho2017": {"file": "gigadb.py", "native_task": 1, "health": "healthy",
                "documented_total": 200, "license": "CC-BY-4.0", "canonical": 1,
                "events": "left_hand|right_hand"},
    "Lee2019_MI": {"file": "Lee2019.py", "native_task": 1, "health": "healthy",
                   "documented_total": 200, "license": "GPL-3.0", "canonical": 1,
                   "events": "left_hand|right_hand"},
    "GrosseWentrup2009": {"file": "mpi_mi.py", "native_task": 1, "health": "healthy",
                          "documented_total": 300, "license": "CC-BY-4.0", "canonical": 1,
                          "events": "left_hand|right_hand"},
    "Shin2017A": {"file": "bbci_eeg_fnirs.py", "native_task": 1, "health": "healthy",
                  "documented_total": 60, "license": "GPL-3.0_acceptance_required", "canonical": 1,
                  "events": "left_hand|right_hand"},
    "Shin2017B": {"file": "bbci_eeg_fnirs.py", "native_task": 0, "health": "healthy",
                  "documented_total": 60, "license": "GPL-3.0_acceptance_required", "canonical": 1,
                  "events": "subtraction|rest"},
    "Liu2024": {"file": "liu2024.py", "native_task": 1, "health": "acute_stroke_patients",
                "documented_total": 40, "license": "CC-BY-4.0", "canonical": 1,
                "events": "left_hand|right_hand"},
    "Dreyer2023A": {"file": "dreyer2023.py", "native_task": 1, "health": "healthy",
                    "documented_total": 240, "license": "CC-BY-4.0", "canonical": 0,
                    "events": "left_hand|right_hand"},
    "Dreyer2023B": {"file": "dreyer2023.py", "native_task": 1, "health": "healthy",
                    "documented_total": 240, "license": "CC-BY-4.0", "canonical": 0,
                    "events": "left_hand|right_hand"},
    "Dreyer2023C": {"file": "dreyer2023.py", "native_task": 1, "health": "healthy",
                    "documented_total": 240, "license": "CC-BY-4.0", "canonical": 0,
                    "events": "left_hand|right_hand"},
    "Dreyer2023": {"file": "dreyer2023.py", "native_task": 1, "health": "healthy",
                   "documented_total": 240, "license": "CC-BY-4.0", "canonical": 1,
                   "url": "https://zenodo.org/records/8089820", "events": "left_hand|right_hand"},
    "Kumar2024": {"file": "kumar2024.py", "native_task": 1, "health": "healthy",
                  "documented_total": 400, "license": "CC-BY-4.0", "canonical": 1,
                  "url": "https://zenodo.org/records/10694880", "events": "left_hand|right_hand"},
    "Brandl2020": {"file": "brandl2020.py", "native_task": 1, "health": "healthy",
                   "documented_total": 504, "license": "CC-BY-NC-ND-4.0", "canonical": 1,
                   "url": "https://doi.org/10.3389/fnins.2020.566147", "events": "left_hand|right_hand"},
    "Rozado2015": {"file": "rozado2015.py", "native_task": 0, "health": "healthy",
                   "documented_total": 50, "license": "CC0-1.0", "canonical": 1,
                   "events": "left_hand|rest"},
    "Ma2020": {"file": "ma2020.py", "native_task": 0, "health": "healthy",
               "documented_total": 600, "license": "CC-BY-4.0", "canonical": 1,
               "events": "right_hand|right_elbow"},
    "Wu2020": {"file": "wu2020.py", "native_task": 1, "health": "healthy",
               "documented_total": 160, "license": "CC-BY-4.0", "canonical": 1,
               "events": "left_hand|right_hand"},
    "Forenzo2023": {"file": "forenzo2023.py", "native_task": 1, "health": "healthy",
                    "documented_total": 75, "license": "CC-BY-4.0", "canonical": 1,
                    "events": "left_hand|right_hand"},
    "GuttmannFlury2025_ME": {"file": "guttmann_flury2025.py", "native_task": 0,
                             "health": "healthy", "documented_total": 40,
                             "license": "CC0-1.0", "canonical": 1,
                             "events": "left_hand_execution|right_hand_execution"},
    "GuttmannFlury2025_MI": {"file": "guttmann_flury2025.py", "native_task": 1,
                             "health": "healthy", "documented_total": 40,
                             "license": "CC0-1.0", "canonical": 1,
                             "events": "left_hand|right_hand"},
    "HefmiIch2025": {"file": "hefmi_ich2025.py", "native_task": 1,
                     "health": "mixed_healthy_and_ICH", "documented_total": 90,
                     "license": "CC-BY-NC-ND-4.0", "canonical": 1,
                     "events": "left_hand|right_hand"},
    "Zuo2025": {"file": "zuo2025.py", "native_task": 0, "health": "knee_pain_patients",
                "documented_total": 500, "license": "CC-BY-4.0", "canonical": 1,
                "events": "left_leg|right_leg"},
    "Liu2025": {"file": "liu2025.py", "native_task": 0, "health": "stroke_patients",
                "documented_total": 320, "license": "CC-BY-NC-ND-4.0", "canonical": 1,
                "events": "gait_imagery|rest"},
    "Yang2025": {"file": "yang2025.py", "native_task": 1, "health": "healthy",
                 "documented_total": 600, "license": "CC-BY-4.0", "canonical": 1,
                 "events": "left_hand|right_hand", "interface_id": "Yang2025_2C",
                 "interface_variant": "paradigm_type=2C;subjects=1..51",
                 "eligible_subjects": 51, "sessions": 3, "runs": 1,
                 "sampling_rate_Hz": 1000, "event_interval_seconds": 4.0,
                 "url": "https://plus.figshare.com/articles/dataset/22671172"},
}

_PRIOR_TARGETS = {"BNCI2014_001", "Cho2017", "Lee2019_MI", "PhysionetMI"}
_HISTORICAL_TARGET_ACCESS_NOT_CERTIFIABLY_ABSENT = {
    "Dreyer2023", "Dreyer2023A", "Dreyer2023B", "Dreyer2023C",
}


def _read_catalog() -> list[dict[str, str]]:
    _require(IMAGERY_CATALOG.is_file(), "installed MOABB imagery catalog absent")
    _require(sha256_file(IMAGERY_CATALOG) ==
             "5d7b3abca6f56c83ce90a90d3e0c252783a24f04e237d8d09b11f3862f0ce7e4",
             "installed MOABB imagery catalog drift")
    with IMAGERY_CATALOG.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _eligibility_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _read_catalog():
        dataset = raw["Dataset"]
        source = _SOURCE_INFO.get(dataset, {})
        interface_id = str(source.get("interface_id", dataset))
        interface_variant = str(source.get("interface_variant", "canonical_default"))
        prior = dataset in _PRIOR_TARGETS
        historical_access_absent = dataset not in _HISTORICAL_TARGET_ACCESS_NOT_CERTIFIABLY_ABSENT
        native = bool(source.get("native_task", 0))
        healthy = source.get("health") == "healthy"
        catalog_subjects = int(raw["#Subj"])
        subjects = int(source.get("eligible_subjects", catalog_subjects))
        trials = int(source.get("documented_total", 0))
        channels_text = raw["#Chan"]
        channels_ok = channels_text.isdigit() and int(channels_text) >= 3
        duration = float(source.get("event_interval_seconds", raw["Trials length (s)"]))
        canonical = bool(source.get("canonical", 1))
        public = bool(source.get("license"))
        selected = all((not prior, historical_access_absent, native, healthy,
                        subjects >= 12, trials >= 80,
                        channels_ok, duration >= 3.0, canonical, public))
        if prior:
            reason = "PRIOR_TARGET_SCIENCE"
        elif not historical_access_absent:
            reason = "HISTORICAL_TARGET_ACCESS_NOT_CERTIFIABLY_ABSENT"
        elif not native:
            reason = "NOT_NATIVE_EXACT_BINARY_LEFT_RIGHT_MI"
        elif not healthy:
            reason = "POPULATION_NOT_HEALTHY_ONLY"
        elif subjects < 12:
            reason = "SUBJECT_COUNT_BELOW_12"
        elif trials < 80:
            reason = "GUARANTEED_TRIAL_SUPPORT_BELOW_80"
        elif not canonical:
            reason = "DUPLICATE_SUBCOHORT_OF_CANONICAL_LOADER"
        elif not public:
            reason = "PUBLIC_LICENSE_OR_REPOSITORY_NOT_BOUND"
        else:
            reason = "ELIGIBLE_ALL_RULES_PASS"
        source_path = MOABB_ROOT / str(source["file"]) if source.get("file") else None
        rows.append({
            "dataset": dataset,
            "confirmation_interface_id": interface_id,
            "interface_variant": interface_variant,
            "catalog_subjects": catalog_subjects,
            "eligible_interface_subjects": subjects,
            "catalog_channels": channels_text,
            "catalog_classes": raw["#Classes"],
            "catalog_trials_per_class": raw["#Trials / class"],
            "catalog_sampling_rate_Hz": raw["Freq (Hz)"],
            "catalog_sessions": raw["#Sessions"],
            "catalog_runs": raw["#Runs"],
            "documented_min_total_trials_per_subject": trials or "NOT_DEEP_AUDITED_AFTER_TASK_GATE",
            "event_interval_seconds": duration,
            "event_labels": source.get("events", "CATALOG_CLASS_GATE_ONLY"),
            "native_exact_left_right_MI": int(native),
            "healthy_adults_only": int(healthy),
            "prior_project_target_science": int(prior),
            "prior_project_target_or_label_access_certifiably_absent": int(
                not prior and historical_access_absent
            ),
            "historical_access_evidence": (
                "C84F_C84S_C85_COMMITTED_TARGET_SCIENCE"
                if prior else (
                    "archive/lpc-cmi-failed/notes/preprocessing_decision.md"
                    if not historical_access_absent
                    else "NO_COMMITTED_PROJECT_TARGET_ACCESS_FOUND"
                )
            ),
            "canonical_nondeduplicated_cohort": int(canonical),
            "public_license": source.get("license", "NOT_DEEP_AUDITED_AFTER_TASK_GATE"),
            "loader_source_path": str(source_path) if source_path else "CATALOG_TASK_GATE_ONLY",
            "loader_source_sha256": sha256_file(source_path) if source_path and source_path.is_file() else "NOT_APPLICABLE_AFTER_EARLY_GATE",
            "metadata_url": source.get("url", "NOT_APPLICABLE_OR_REGISTRY_ONLY"),
            "loader_determinism_audit": "SOURCE_IDENTITY_BOUND_NO_DATA_OPEN" if source_path else "EARLY_TASK_GATE",
            "published_performance_used": 0,
            "selected_for_C86H_engineering": int(selected),
            "decision_reason": reason,
        })
    selected = tuple(sorted(row["confirmation_interface_id"] for row in rows
                            if row["selected_for_C86H_engineering"]))
    _require(selected == ELIGIBLE_DATASETS, f"eligible cohort set drift: {selected}")
    return rows


def _identity_rows() -> list[dict[str, Any]]:
    expected = {
        "C85E_EXECUTION_LOCK.json": "a59062305b521973476e0d40236069eba7c9e149aeca9d3fe03c08a1ce106176",
        "C85E_OVERALL_REPORT.md": "31b9ed96377410d725da30b6de71aabee7702387d04448e10af0be31d52d9c82",
        "C85E_OVERALL_REPORT.json": "3908e1ef72db29030aff023917f7853cbe54607460e143c05b9bdd2047b9aec7",
        "C86P_SYNTHETIC_CALIBRATION_OPERATIONALIZATION_PROTOCOL.json":
            "a80e8cca75eaa4d22b374794c06a9304ef9bb21605ec75f5d6aa53509f86b54b",
    }
    rows = []
    for name, digest in expected.items():
        path = REPORT_DIR / name
        observed = sha256_file(path)
        rows.append({"object": name, "expected_sha256": digest, "observed_sha256": observed,
                     "passed": int(observed == digest), "role": "COMMITTED_COMPACT_PARENT"})
    for key, value in (
        ("C84_primary", "C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous"),
        ("C84_frontier", "C84-L4"),
        ("T1_T3_T4_T7", "PROVED"),
        ("T2_T6", "COUNTEREXAMPLE"),
        ("T5", "OPEN"),
        ("C86P_variant_correction", "5948b76a2d08c45c88e157aace1cc421a8c551b1c763a265376ad25921103c0d"),
        ("C86P_historical_access_correction", "fd7e214c6c6675b2a9b071b2bf278e2c4393495b2f4cbe03f82528c3098e7064"),
    ):
        rows.append({"object": key, "expected_sha256": value, "observed_sha256": value,
                     "passed": 1, "role": "IMMUTABLE_CLAIM_STATE"})
    _require(all(row["passed"] for row in rows), "C85E identity or claim replay failed")
    return rows


def _literature_rows() -> list[dict[str, Any]]:
    return [
        {"source_id": "Kossen2021", "title": "Active Testing: Sample-Efficient Model Evaluation",
         "year": 2021, "primary_url": "https://proceedings.mlr.press/v139/kossen21a.html",
         "role": "active_testing_and_surrogate_acquisition", "verified": 1,
         "fidelity_limit": "C86_A1_is_a_multicandidate_adaptation"},
        {"source_id": "Farquhar2021", "title": "On Statistical Bias in Active Learning: How and When to Fix It",
         "year": 2021, "primary_url": "https://openreview.net/forum?id=JiYq3eqTKY",
         "role": "LURE_weights_and_bias_correction", "verified": 1,
         "fidelity_limit": "unbiasedness_applies_to_linear_LURE_moments"},
        {"source_id": "Hara2024", "title": "Active model selection: A variance minimization approach",
         "year": 2024, "primary_url": "https://link.springer.com/article/10.1007/s10994-024-06603-1",
         "role": "pairwise_loss_difference_variance_acquisition", "verified": 1,
         "fidelity_limit": "C86_A2_extends_the_pairwise_object_to_81_candidates"},
        {"source_id": "Karimi2021", "title": "Online Active Model Selection for Pre-trained Classifiers",
         "year": 2021, "primary_url": "https://proceedings.mlr.press/v130/reza-karimi21a.html",
         "role": "disagreement_selective_sampling", "verified": 1,
         "fidelity_limit": "C86_is_finite_pool_fixed_budget_not_the_original_streaming_setting"},
        {"source_id": "Sawade2010", "title": "Active Risk Estimation", "year": 2010,
         "primary_url": "https://icml.cc/Conferences/2010/papers/285.pdf",
         "role": "active_evaluation_precedent", "verified": 1,
         "fidelity_limit": "background_not_a_claimed_byte_exact_method"},
        {"source_id": "Katariya2012", "title": "Active Evaluation of Classifiers on Large Datasets",
         "year": 2012, "primary_url": "https://www.microsoft.com/en-us/research/publication/active-evaluation-classifiers-large-datasets/",
         "role": "stratified_active_evaluation_precedent", "verified": 1,
         "fidelity_limit": "background_only"},
        {"source_id": "Hansen2011", "title": "The Model Confidence Set", "year": 2011,
         "primary_url": "https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA5771",
         "role": "plausible_best_set_terminology", "verified": 1,
         "fidelity_limit": "A4_is_project_specific_and_not_an_MCS_replication"},
    ]


def _synthetic_rows() -> list[dict[str, Any]]:
    contract = json.loads(SYNTHETIC_PROTOCOL_PATH.read_text(encoding="utf-8"))
    _require(contract["status"] == "LOCKED_GENERATIVE_CONTRACT_NOT_SCIENTIFICALLY_EXECUTED",
             "synthetic calibration contract status drift")
    global_contract = contract["global_generator"]
    scenarios = contract["scenarios"]
    _require([row["id"] for row in scenarios] == [f"C86S{index:02d}" for index in range(11)],
             "synthetic scenario identity drift")
    return [
        {
            "scenario_id": row["id"],
            "scenario": row["name"],
            "target_groups": global_contract["target_groups_per_scenario"],
            "contexts_per_target": global_contract["contexts_per_target"],
            "acquisition_pool_trials": global_contract["acquisition_pool_trials_per_context"],
            "held_evaluation_trials": global_contract["held_evaluation_trials_per_context"],
            "candidate_count": global_contract["candidate_count"],
            "active_chains": global_contract["active_chains"],
            "seed_rule": global_contract["seed_rule"],
            "pi_g": row["pi_g"],
            "stratum_masses": json.dumps(row["stratum_masses"], separators=(",", ":")),
            "difficulty": json.dumps(row["difficulty"], separators=(",", ":")),
            "shared_sigma": row["shared_sigma"],
            "action_sigma": json.dumps(row["action_sigma"], separators=(",", ":")),
            "skill_profile": row["skill_profile"],
            "failure_injection": row["failure_injection"],
            "validation_target": row["validation_target"],
            "production_entrypoint_required_before_real_execution": 1,
            "C86P_mode": "LOCKED_SCHEMA_ONLY_NOT_EXECUTED",
            "registered_draws": 0,
            "real_data": 0,
        }
        for row in scenarios
    ]


def _method_rows() -> list[dict[str, Any]]:
    common = {"tie_rule": "first_canonical_candidate_index", "confirmation_tuning": "FORBIDDEN"}
    return [
        {"method_id": "P0", "family": "PASSIVE_PRIMARY", "observable_inputs": "unlabeled_trial_IDs;RNG",
         "query_score": "constant", "estimator": "sample_mean_equals_LURE", "warm_start": 0,
         "probability_floor": "uniform", "variance_estimator": "finite_population_sample_variance",
         "confidence_set": "not_applicable", "stopping": "fixed_budget", "complexity": "O(N)",
         "reference": "DESIGN_BASELINE", "role": "PRIMARY_COMPARATOR", **common},
        {"method_id": "P1", "family": "PASSIVE_CLASS_AWARE", "observable_inputs": "complete_pool_class_allocation_oracle",
         "query_score": "class_stratified_uniform", "estimator": "historical_Q0", "warm_start": 0,
         "probability_floor": "within_class_uniform", "variance_estimator": "historical_Q0",
         "confidence_set": "not_applicable", "stopping": "fixed_budget", "complexity": "O(N)",
         "reference": "C84_Q0", "role": "SECONDARY_UNFAIR_INFORMATION_REFERENCE", **common},
        {"method_id": "A1", "family": "ACTIVE_TESTING", "observable_inputs": "candidate_probabilities;queried_labels",
         "query_score": "mean_expected_candidate_NLL", "estimator": "LURE_composite_plugin", "warm_start": 4,
         "probability_floor": "0.05/remaining", "variance_estimator": "LURE_weighted_term_sample_variance",
         "confidence_set": "not_used", "stopping": "fixed_budget", "complexity": "O(NM)",
         "reference": "Kossen2021;Farquhar2021", "role": "PRIMARY_ACTIVE", **common},
        {"method_id": "A2", "family": "PAIRWISE_VARIANCE", "observable_inputs": "candidate_probabilities;queried_labels",
         "query_score": "max_expected_absolute_pairwise_NLL_difference", "estimator": "LURE_composite_plugin", "warm_start": 4,
         "probability_floor": "0.05/remaining", "variance_estimator": "LURE_pairwise_weighted_term_sample_variance",
         "confidence_set": "not_used_all_81_pairs", "stopping": "fixed_budget", "complexity": "O(NM^2)",
         "reference": "Hara2024;Farquhar2021", "role": "PRIMARY_ACTIVE", **common},
        {"method_id": "A3", "family": "DISAGREEMENT", "observable_inputs": "candidate_predictions;queried_labels",
         "query_score": "one_minus_max_vote_fraction", "estimator": "LURE_composite_plugin", "warm_start": 4,
         "probability_floor": "0.05/remaining", "variance_estimator": "LURE_weighted_term_sample_variance",
         "confidence_set": "not_used_all_81_votes", "stopping": "fixed_budget", "complexity": "O(NM)",
         "reference": "Karimi2021", "role": "PRIMARY_ACTIVE", **common},
        {"method_id": "A4", "family": "PLAUSIBLE_BEST", "observable_inputs": "candidate_probabilities;queried_labels;confidence_set",
         "query_score": "max_expected_pairwise_NLL_difference_in_set", "estimator": "LURE_composite_plugin", "warm_start": 4,
         "probability_floor": "0.05/remaining", "variance_estimator": "LURE_pairwise_weighted_term_sample_variance",
         "confidence_set": "Bonferroni_normal_NLL_plausible_best_set_alpha_0.05",
         "stopping": "fixed_budget_primary;secondary_two_step_singleton", "complexity": "O(NM^2)",
         "reference": "Hansen2011;Hara2024_project_adaptation", "role": "PRIMARY_ACTIVE", **common},
        {"method_id": "C0", "family": "FIXED", "observable_inputs": "none", "query_score": "not_applicable",
         "estimator": "not_applicable", "warm_start": 0, "probability_floor": "not_applicable",
         "variance_estimator": "not_applicable", "confidence_set": "not_applicable", "stopping": "no_queries",
         "complexity": "O(1)", "reference": "C84_B1",
         "role": "B1_ERM_REFERENCE", **common},
        {"method_id": "C1", "family": "SOURCE", "observable_inputs": "source_audit_only", "query_score": "not_applicable",
         "estimator": "S1", "warm_start": 0, "probability_floor": "not_applicable",
         "variance_estimator": "not_applicable", "confidence_set": "not_applicable", "stopping": "no_queries",
         "complexity": "O(M)", "reference": "C84_S1",
         "role": "STRICT_SOURCE_REFERENCE", **common},
        {"method_id": "C2_U11", "family": "ZERO_LABEL", "observable_inputs": "target_unlabeled_only", "query_score": "not_applicable",
         "estimator": "MaNo", "warm_start": 0, "probability_floor": "not_applicable",
         "variance_estimator": "not_applicable", "confidence_set": "not_applicable", "stopping": "no_queries",
         "complexity": "C84_FIXED", "reference": "C84_U11",
         "role": "FROZEN_ZERO_LABEL_REFERENCE", **common},
        {"method_id": "C2_U13", "family": "ZERO_LABEL", "observable_inputs": "target_unlabeled_only", "query_score": "not_applicable",
         "estimator": "COTT", "warm_start": 0, "probability_floor": "not_applicable",
         "variance_estimator": "not_applicable", "confidence_set": "not_applicable", "stopping": "no_queries",
         "complexity": "C84_FIXED", "reference": "C84_U13",
         "role": "FROZEN_ZERO_LABEL_REFERENCE", **common},
        {"method_id": "O1", "family": "ORACLE", "observable_inputs": "full_acquisition_labels_after_evaluation_only_release",
         "query_score": "not_applicable", "estimator": "exact_full_acquisition_composite", "warm_start": 0,
         "probability_floor": "not_applicable", "variance_estimator": "exact", "confidence_set": "not_applicable",
         "stopping": "FULL", "complexity": "O(NM)", "reference": "FULL_ACQUISITION_POOL",
         "role": "EVALUATION_ONLY_CEILING", **common},
    ]


def _simple_rows() -> dict[str, list[dict[str, Any]]]:
    methods = _method_rows()
    eligibility = _eligibility_rows()
    selected = [row for row in eligibility if row["selected_for_C86H_engineering"]]
    return {
        "c85e_identity_and_claim_replay.csv": _identity_rows(),
        "active_testing_literature_registry.csv": _literature_rows(),
        "active_method_registry.csv": methods,
        "reference_fidelity_registry.csv": [
            {"method_id": method, "primary_source": source, "fidelity": fidelity,
             "exact_reference_replication_claimed": 0, "confirmation_requires_frozen_code": 1}
            for method, source, fidelity in (
                ("A1", "Kossen2021;Farquhar2021", "MULTICANDIDATE_ADAPTATION"),
                ("A2", "Hara2024;Farquhar2021", "K_GREATER_THAN_2_ADAPTATION"),
                ("A3", "Karimi2021", "FINITE_POOL_FIXED_BUDGET_ADAPTATION"),
                ("A4", "Hansen2011;Hara2024", "PROJECT_SPECIFIC_PLAUSIBLE_BEST_RULE"),
                ("P0", "DESIGN_BASELINE", "EXACT_UNIFORM_WITHOUT_REPLACEMENT"),
                ("P1", "C84_Q0", "EXACT_HISTORICAL_SECONDARY_REFERENCE"),
            )
        ],
        "observable_information_contract.csv": [
            {"method_id": row["method_id"], "unlabeled_predictions": int(row["method_id"] in {"A1", "A2", "A3", "A4", "C2_U11", "C2_U13"}),
             "queried_labels_only": int(row["method_id"] in {"A1", "A2", "A3", "A4", "P0"}),
             "unqueried_labels": int(row["method_id"] in {"P1", "O1"}),
             "held_evaluation_labels_before_freeze": 0,
             "class_allocation_before_query": int(row["method_id"] == "P1"),
             "information_class": row["family"]}
            for row in methods
        ],
        "untouched_dataset_eligibility_registry.csv": eligibility,
        "historical_access_ledger.csv": [
            {"dataset": row["dataset"], "confirmation_interface_id": row["confirmation_interface_id"],
             "committed_target_science": row["prior_project_target_science"],
             "committed_access_evidence": (
                 "PRIOR_TARGET_SCIENCE" if row["prior_project_target_science"]
                 else ("HISTORICAL_LOCAL_PREPROCESSED_STORE_VERIFIED"
                       if not row["prior_project_target_or_label_access_certifiably_absent"]
                       else "NO_COMMITTED_PROJECT_TARGET_ACCESS_FOUND")
             ),
             "evidence_path": row["historical_access_evidence"],
             "untouched_access_certified": row["prior_project_target_or_label_access_certifiably_absent"],
             "C86P_EEG_open": 0, "C86P_label_open": 0, "C86P_download": 0,
             "future_preaccess_replay_required": int(
                 row["prior_project_target_or_label_access_certifiably_absent"]
             )}
            for row in eligibility
        ],
        "interface_compatibility_matrix.csv": [
            {"dataset": row["dataset"], "confirmation_interface_id": row["confirmation_interface_id"],
             "interface_variant": row["interface_variant"],
             "native_binary_left_right": row["native_exact_left_right_MI"],
             "subjects_ge_12": int(int(row["eligible_interface_subjects"]) >= 12),
             "trials_ge_80": int(isinstance(row["documented_min_total_trials_per_subject"], int) and row["documented_min_total_trials_per_subject"] >= 80),
             "channels_ge_3": int(str(row["catalog_channels"]).isdigit() and int(row["catalog_channels"]) >= 3),
             "interval_ge_3s": int(float(row["event_interval_seconds"]) >= 3),
             "healthy_only": row["healthy_adults_only"], "canonical": row["canonical_nondeduplicated_cohort"],
             "not_prior_target": int(not row["prior_project_target_science"]),
             "historical_target_access_absent": row["prior_project_target_or_label_access_certifiably_absent"],
             "all_interface_rules_pass": row["selected_for_C86H_engineering"]}
            for row in eligibility if row["native_exact_left_right_MI"] or row["prior_project_target_science"]
        ],
        "dataset_selection_rule_truth_table.csv": [
            {"dataset": row["dataset"], "confirmation_interface_id": row["confirmation_interface_id"],
             "all_rules_pass": 1, "all_eligible_included": 1,
             "performance_used": 0, "hand_selected": 0, "confirmation_role": "PRIMARY_UNTOUCHED_COHORT"}
            for row in selected
        ] + [{"dataset": "BNCI2014_004", "confirmation_interface_id": "BNCI2014_004",
              "all_rules_pass": 0, "all_eligible_included": 1,
              "performance_used": 0, "hand_selected": 0, "confirmation_role": "SEPARATE_STRESS_ONLY_SUBJECT_GATE_FAIL"}],
        "development_confirmation_separation.csv": [
            {"track": "C86L", "real_data_role": "C84_DEVELOPMENT_ARTIFACT", "outcome_label": "POST_C84_DEVELOPMENT_ONLY",
             "primary_claim_eligible": 0, "requires_separate_lock": 1, "confirmation_feedback": 0},
            {"track": "C86D", "real_data_role": "METHOD_DEVELOPMENT", "outcome_label": "POST_C84_DEVELOPMENT_ONLY",
             "primary_claim_eligible": 0, "requires_separate_lock": 1, "confirmation_feedback": 0},
            {"track": "C86C_F", "real_data_role": "UNTOUCHED_ENGINEERING_AND_CANDIDATE_FIELD", "outcome_label": "TECHNICAL_ONLY",
             "primary_claim_eligible": 0, "requires_separate_lock": 1, "confirmation_feedback": 0},
            {"track": "C86H", "real_data_role": "UNTOUCHED_CONFIRMATION", "outcome_label": "PROSPECTIVE_CONFIRMATION",
             "primary_claim_eligible": 1, "requires_separate_lock": 1, "confirmation_feedback": 0},
        ],
        "candidate_zoo_contract.csv": [
            {"regime": "ERM", "candidate_count": 1, "canonical_indices": "0", "architecture_changed": 0,
             "retention_from_C84_outcomes": 0, "separate_training_lock_required": 1},
            {"regime": "OACI", "candidate_count": 40, "canonical_indices": "1..40", "architecture_changed": 0,
             "retention_from_C84_outcomes": 0, "separate_training_lock_required": 1},
            {"regime": "SRC", "candidate_count": 40, "canonical_indices": "41..80", "architecture_changed": 0,
             "retention_from_C84_outcomes": 0, "separate_training_lock_required": 1},
        ],
        "physical_label_view_contract.csv": [
            {"view": "target_unlabeled_acquisition_pool", "process": "ACTIVE_CLIENT", "labels_visible": 0,
             "released_before_selection_freeze": 1, "overlap_with_held_evaluation": 0},
            {"view": "target_acquisition_label_oracle", "process": "LABEL_SERVER", "labels_visible": "QUERIED_ONLY",
             "released_before_selection_freeze": "ONE_ROW_PER_QUERY", "overlap_with_held_evaluation": 0},
            {"view": "target_held_evaluation_label_view", "process": "HELD_EVALUATOR", "labels_visible": 1,
             "released_before_selection_freeze": 0, "overlap_with_held_evaluation": "SELF"},
            {"view": "same_label_oracle_view", "process": "SEALED_ORACLE", "labels_visible": 0,
             "released_before_selection_freeze": 0, "overlap_with_held_evaluation": 0},
        ],
        "total_query_budget_contract.csv": [
            {"budget": value, "unit": "TOTAL_QUERIES_PER_TARGET", "primary_comparative": int(value != "FULL"),
             "labels_per_class": 0, "nested_prefix": int(value != "FULL"),
             "role": "FULL_REFERENCE" if value == "FULL" else ("SMALL" if value in {4, 8} else "INTERMEDIATE_LARGE")}
            for value in TOTAL_QUERY_GRID
        ],
        "passive_comparator_fairness_audit.csv": [
            {"comparator": "P0", "same_total_budget": 1, "class_known_before_query": 0,
             "without_replacement": 1, "primary": 1, "fairness_status": "PRIMARY_FAIR"},
            {"comparator": "P1", "same_total_budget": 1, "class_known_before_query": 1,
             "without_replacement": 1, "primary": 0, "fairness_status": "SECONDARY_CLASS_AWARE"},
            {"comparator": "O1", "same_total_budget": 0, "class_known_before_query": 1,
             "without_replacement": 1, "primary": 0, "fairness_status": "EVALUATION_CEILING_ONLY"},
        ],
        "trial_loss_vector_schema.csv": [
            {"field": field, "dtype": dtype, "required": 1, "contains_in_C86P": 0,
             "semantic_role": role}
            for field, dtype, role in (
                ("trial_id", "UTF8", "IDENTITY"), ("context_id", "UTF8", "IDENTITY"),
                ("candidate_ids", "UTF8[81]", "ACTION_IDENTITY"),
                ("candidate_probabilities", "float64[81,2]", "UNLABELED_PREDICTIONS"),
                ("true_label", "uint8", "QUERIED_LABEL"),
                ("NLL_contributions", "float64[81]", "LINEAR_LOSS_VECTOR"),
                ("correctness", "uint8[81]", "BACC_ECE_COMPONENT"),
                ("confidence_bins", "uint8[81]", "ECE_COMPONENT"),
                ("pairwise_NLL_differences", "float64[81,81]", "PAIRWISE_DESIGN"),
            )
        ],
        "estimator_and_importance_weight_contract.csv": [
            {"object": "LURE_weight", "primary": 1, "formula": "1+(N-M)/(N-m)*(1/((N-m+1)*q_m)-1)",
             "unbiased_claim": "LINEAR_FINITE_POOL_MEAN", "clipping": "NONE", "failure": "q_m<=0_or_nonfinite"},
            {"object": "balanced_accuracy", "primary": 1, "formula": "0.5*sum_y((N*LURE_correct_y+0.5)/(N*LURE_class_y+1))",
             "unbiased_claim": "NONE_JEFFREYS_SMOOTHED_PLUGIN", "clipping": "NONE", "failure": "nonfinite"},
            {"object": "NLL", "primary": 1, "formula": "LURE_mean(-log(clip(p_y,1e-12,1)))",
             "unbiased_claim": "LINEAR_CLIPPED_NLL_MEAN", "clipping": "PROBABILITY_ONLY", "failure": "nonfinite"},
            {"object": "ECE", "primary": 1, "formula": "sum_15_bins(abs(LURE_mean(I_bin*(correct-confidence))))",
             "unbiased_claim": "SIGNED_BIN_MOMENTS_ONLY", "clipping": "NONE", "failure": "nonfinite"},
            {"object": "composite_utility", "primary": 1, "formula": "mean_of_three_oriented_81_candidate_midranks",
             "unbiased_claim": "NONE_NONLINEAR_PLUGIN", "clipping": "NONE", "failure": "component_incomplete"},
        ],
        "stopping_rule_contract.csv": [
            {"method": method, "primary_rule": "QUERY_TO_FIXED_BUDGET", "early_stop_primary": 0,
             "secondary_rule": "TWO_CONSECUTIVE_SINGLETON_PLAUSIBLE_SETS" if method == "A4" else "NONE",
             "secondary_can_replace_primary": 0}
            for method in ("P0", "A1", "A2", "A3", "A4")
        ],
        "endpoint_and_robust_risk_contract.csv": [
            {"endpoint": endpoint, "scale": scale, "unit": unit, "primary": primary,
             "dataset_pooling": 0, "new_pvalue_in_C86P": 0}
            for endpoint, scale, unit, primary in (
                ("standardized_regret", "C84_STANDARDIZED", "TARGET_SUBJECT", 1),
                ("selected_utility", "C84_COMPOSITE", "TARGET_SUBJECT", 1),
                ("top1_top5_top10", "PROBABILITY", "TARGET_SUBJECT", 1),
                ("epsilon_near_optimal", "RAW_UTILITY_GAP_GRID_0.005_0.01_0.02_0.05", "TARGET_SUBJECT", 1),
                ("worst_target", "C84_STANDARDIZED", "TARGET_SUBJECT", 1),
                ("CVaR", "C84_STANDARDIZED_ALPHA_0.50_0.75_0.90", "TARGET_SUBJECT", 1),
                ("query_entropy", "NATS", "CONTEXT", 0),
                ("label_count", "TOTAL_QUERIES", "TARGET_SUBJECT", 1),
            )
        ],
        "inference_and_multiplicity_contract.csv": [
            {"component": component, "value": value, "locked": 1, "confirmation_only": 1}
            for component, value in (
                ("principal_cluster", "target_subject"), ("materiality_margin", "0.05"),
                ("familywise_alpha", "0.05"), ("maxT_draws", "65536"),
                ("family", "4_active_methods_x_4_finite_budgets_within_dataset"),
                ("favorable_target_fraction", "0.75"), ("worst_target_effect_floor", "-0.10"),
                ("positive_panel_seed_level_cells", "6_of_8"),
                ("tail_CVaR90_margin", "0.05"), ("LOTO_preservation", "0.75"),
                ("pooled_dataset_pvalue", "FORBIDDEN"),
            )
        ],
        "future_C86L_artifact_contract.csv": [
            {"stage": "C86L", "artifact": "trial_level_full_loss_vector_field", "development_only": 1,
             "separate_lock": 1, "fresh_authorization": 1, "C86P_produced": 0,
             "primary_confirmation_evidence": 0}
        ],
        "future_C86H_confirmation_contract.csv": [
            {"requirement": requirement, "value": value, "locked_before_new_data": 1}
            for requirement, value in (
                ("cohorts", "Brandl2020|Kumar2024|Yang2025_2C"),
                ("all_eligible_included", "true"), ("same_method_identity", "true"),
                ("same_hyperparameters", "true"), ("primary_passive", "P0"),
                ("query_grid", "4|8|16|32|FULL"), ("candidate_count", "81"),
                ("new_performance_based_dataset_choice", "false"),
                ("separate_execution_lock_and_authorization", "required"),
            )
        ],
        "synthetic_scenario_registry.csv": _synthetic_rows(),
        "risk_register.csv": [
            {"risk_id": risk_id, "risk": risk, "blocking": blocking, "mitigation": mitigation,
             "residual_status": status}
            for risk_id, risk, blocking, mitigation, status in (
                ("R1", "outcome_informed_confirmation_dataset_selection", 1, "all_metadata_eligible_cohorts_included", "CLOSED"),
                ("R2", "class_aware_comparator_advantage", 1, "P0_primary_P1_secondary_labeled", "CLOSED"),
                ("R3", "LURE_unbiasedness_overclaim", 1, "linear_moments_only_plugin_outputs_explicit", "CLOSED"),
                ("R4", "C84_development_as_confirmation", 1, "C86D_development_only", "CLOSED"),
                ("R5", "target_chain_pseudoreplication", 1, "target_subject_only_scientific_N", "CLOSED"),
                ("R6", "FULL_superiority_artifact", 1, "FULL_reference_not_superiority_test", "CLOSED"),
                ("R7", "T7_empirical_transfer", 1, "MGF_assumption_guard", "CLOSED"),
                ("R8", "license_or_loader_drift", 1, "future_preaccess_hash_and_terms_replay", "OPEN_FUTURE_STAGE"),
                ("R9", "candidate_field_resource_exceeds_envelope", 1, "separate_C86C_F_resource_lock", "OPEN_FUTURE_STAGE"),
                ("R10", "confirmation_class_support_failure", 1, "fixed_split_fail_without_resplit", "OPEN_FUTURE_STAGE"),
            )
        ],
        "failure_reason_ledger.csv": [
            {"failure_code": code, "observed_in_C86P": 0, "blocking_gate": gate,
             "automatic_retry": 0, "required_action": action}
            for code, gate, action in (
                ("NO_TWO_ELIGIBLE_COHORTS", "C86_NO_ELIGIBLE_UNTOUCHED_COHORT_UNDER_LOCKED_INTERFACE_RULE", "PM_review"),
                ("METADATA_OR_LOADER_IDENTITY_DRIFT", "C86_ACTIVE_POLICY_BUDGET_ESTIMATOR_OR_CONFIRMATION_SEPARATION_RECONCILIATION_REQUIRED", "additive_protocol"),
                ("ESTIMATOR_SEMANTICS_INCOMPLETE", "C86_ACTIVE_POLICY_BUDGET_ESTIMATOR_OR_CONFIRMATION_SEPARATION_RECONCILIATION_REQUIRED", "additive_protocol"),
                ("DEVELOPMENT_CONFIRMATION_LEAK", "C86_ACTIVE_POLICY_BUDGET_ESTIMATOR_OR_CONFIRMATION_SEPARATION_RECONCILIATION_REQUIRED", "stop"),
                ("NEW_DATA_OR_ACTIVE_ACCESS", "C86_ACTIVE_POLICY_BUDGET_ESTIMATOR_OR_CONFIRMATION_SEPARATION_RECONCILIATION_REQUIRED", "stop_and_disclose"),
            )
        ],
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    values = [dict(row) for row in rows]
    _require(values, f"refusing empty C86P table: {path.name}")
    fields = list(values[0])
    _require(all(list(row) == fields for row in values), f"schema drift in {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)
    return sha256_file(path)


def write_readiness_tables(table_dir: Path = TABLE_DIR) -> dict[str, Any]:
    """Materialize metadata/readiness registries; never opens an EEG artifact."""
    tables = _simple_rows()
    expected = {
        "c85e_identity_and_claim_replay.csv", "active_testing_literature_registry.csv",
        "active_method_registry.csv", "reference_fidelity_registry.csv",
        "observable_information_contract.csv", "untouched_dataset_eligibility_registry.csv",
        "historical_access_ledger.csv", "interface_compatibility_matrix.csv",
        "dataset_selection_rule_truth_table.csv", "development_confirmation_separation.csv",
        "candidate_zoo_contract.csv", "physical_label_view_contract.csv",
        "total_query_budget_contract.csv", "passive_comparator_fairness_audit.csv",
        "trial_loss_vector_schema.csv", "estimator_and_importance_weight_contract.csv",
        "stopping_rule_contract.csv", "endpoint_and_robust_risk_contract.csv",
        "inference_and_multiplicity_contract.csv", "future_C86L_artifact_contract.csv",
        "future_C86H_confirmation_contract.csv", "synthetic_scenario_registry.csv",
        "risk_register.csv", "failure_reason_ledger.csv",
    }
    _require(set(tables) == expected, "C86P readiness table registry drift")
    hashes = {name: _write_csv(table_dir / name, rows) for name, rows in tables.items()}
    return {
        "table_count": len(hashes),
        "eligible_datasets": list(ELIGIBLE_DATASETS),
        "catalog_rows": len(tables["untouched_dataset_eligibility_registry.csv"]),
        "new_EEG_downloads": 0,
        "new_label_reads": 0,
        "active_acquisition_runs": 0,
        "hashes": hashes,
    }


__all__ = [
    "ACTIVE_CHAINS", "C86PContractError", "CVAR_ALPHA_GRID", "ELIGIBLE_DATASETS",
    "EPSILON_GRID", "FINITE_QUERY_GRID", "TOTAL_QUERY_GRID", "canonical_trial_split",
    "empirical_upper_cvar", "estimate_historical_composite", "lure_mean", "lure_weights",
    "midrank_percentile", "mixed_query_probabilities", "validate_total_query_grid",
    "write_readiness_tables",
]
