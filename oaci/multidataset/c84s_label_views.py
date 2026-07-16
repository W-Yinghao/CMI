"""Physically isolated target-label view provisioning for future C84S.

This module has no candidate-artifact dependency. C84SL exercises it only on
synthetic rows; real loader-label access remains authorization-gated elsewhere.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .c84s_common import (
    C84SContractError, atomic_publish_directory, canonical_sha256, require,
    sha256_file, write_csv, write_json,
)


SPLIT_SALT = "C84_TARGET_SPLIT_V1"
CANONICAL_CLASSES = {"left_hand": 0, "right_hand": 1}
IDENTITY_FIELDS = (
    "dataset", "target_subject_id", "target_trial_id", "session", "run",
)
LABEL_FIELDS = IDENTITY_FIELDS + ("canonical_class_label", "split_identity")
FORBIDDEN_LABEL_VIEW_FIELDS = {
    "X", "EEG", "logits", "probabilities", "z", "candidate_id", "method_id",
    "score", "rank", "utility", "regret",
}
FROZEN_TARGET_REGISTRY_PATH = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c84-full-field-target-replay-v2/"
    "lock_f0c369ee273352b47e36/C84F_TARGET_UNLABELED_TRIAL_REGISTRY.json"
)


@dataclass(frozen=True)
class LabelViewDescriptor:
    kind: str
    root: str
    manifest_sha256: str
    row_count: int
    descriptor_sha256: str


def _identity(row: Mapping[str, Any]) -> tuple[str, ...]:
    missing = set(IDENTITY_FIELDS) - set(row)
    if missing:
        raise C84SContractError(f"label identity fields missing: {sorted(missing)}")
    return tuple(str(row[field]) for field in IDENTITY_FIELDS)


def split_order_digest(dataset: str, subject: str | int, trial_id: str) -> str:
    key = f"{SPLIT_SALT}|{dataset}|{subject}|{trial_id}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def align_and_split_labels(
    frozen_registry_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    *,
    minimum_per_split_class: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Align labels exactly and create the locked class-stratified hash split."""
    require(len(frozen_registry_rows) == len(label_rows), "target label row count drift")
    registry = {_identity(row): row for row in frozen_registry_rows}
    labels = {_identity(row): row for row in label_rows}
    require(len(registry) == len(frozen_registry_rows), "frozen trial identities are not unique")
    require(len(labels) == len(label_rows), "label trial identities are not unique")
    require(set(registry) == set(labels), "target label alignment differs from frozen trial registry")

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for identity in sorted(registry):
        source = labels[identity]
        forbidden = FORBIDDEN_LABEL_VIEW_FIELDS & set(source)
        require(not forbidden, f"label input contains forbidden candidate/EEG fields: {sorted(forbidden)}")
        raw_label = source.get("canonical_class_label")
        if isinstance(raw_label, str) and raw_label in CANONICAL_CLASSES:
            label = CANONICAL_CLASSES[raw_label]
        else:
            label = int(raw_label)
        require(label in (0, 1), "C84S label is outside the frozen binary mapping")
        row = {field: source[field] for field in IDENTITY_FIELDS}
        row["canonical_class_label"] = label
        key = (str(row["dataset"]), str(row["target_subject_id"]), label)
        grouped[key].append(row)

    construction: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    cell_counts: list[dict[str, Any]] = []
    subjects = {(dataset, subject) for dataset, subject, _ in grouped}
    for dataset, subject in sorted(subjects):
        class_ids = {class_id for ds, sub, class_id in grouped if ds == dataset and sub == subject}
        require(class_ids == {0, 1}, f"target subject lacks a frozen class: {dataset}/{subject}")
    for (dataset, subject, class_id), rows in sorted(grouped.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                split_order_digest(dataset, subject, str(row["target_trial_id"])),
                str(row["target_trial_id"]),
            ),
        )
        cut = len(ordered) // 2
        require(cut >= minimum_per_split_class, f"construction support below minimum: {dataset}/{subject}/{class_id}")
        require(len(ordered) - cut >= minimum_per_split_class, f"evaluation support below minimum: {dataset}/{subject}/{class_id}")
        left = [{**row, "split_identity": "construction"} for row in ordered[:cut]]
        right = [{**row, "split_identity": "evaluation"} for row in ordered[cut:]]
        construction.extend(left)
        evaluation.extend(right)
        cell_counts.append({
            "dataset": dataset, "target_subject_id": subject,
            "canonical_class_label": class_id, "total": len(ordered),
            "construction": len(left), "evaluation": len(right),
        })

    construction_ids = {_identity(row) for row in construction}
    evaluation_ids = {_identity(row) for row in evaluation}
    require(not construction_ids & evaluation_ids, "construction/evaluation trial overlap")
    require(construction_ids | evaluation_ids == set(registry), "split does not cover frozen registry")
    audit = {
        "schema_version": "c84s_label_split_audit_v1",
        "registry_rows": len(registry),
        "construction_rows": len(construction),
        "evaluation_rows": len(evaluation),
        "overlap": 0,
        "cell_count": len(cell_counts),
        "cell_counts_sha256": canonical_sha256(cell_counts),
        "split_salt": SPLIT_SALT,
        "minimum_per_split_class": minimum_per_split_class,
    }
    return construction, evaluation, audit


def _publish_one_view(root: Path, kind: str, rows: Sequence[Mapping[str, Any]]) -> LabelViewDescriptor:
    expected = set(LABEL_FIELDS)
    require(all(set(row) == expected for row in rows), f"{kind} label schema drift")
    require(all(row["split_identity"] == kind for row in rows), f"{kind} split identity drift")
    table_sha = write_csv(root / "labels.csv", rows)
    manifest = {
        "schema_version": "c84s_physical_label_view_v1",
        "kind": kind,
        "row_count": len(rows),
        "table": {"path": "labels.csv", "sha256": table_sha},
        "fields": list(LABEL_FIELDS),
        "candidate_artifacts": 0,
        "EEG_arrays": 0,
    }
    manifest_sha = write_json(root / "manifest.json", manifest)
    descriptor = {
        "kind": kind, "root": str(root), "manifest_sha256": manifest_sha,
        "row_count": len(rows),
    }
    descriptor_sha = canonical_sha256(descriptor)
    return LabelViewDescriptor(**descriptor, descriptor_sha256=descriptor_sha)


def publish_physical_label_views(
    base_root: str | Path,
    construction: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
) -> tuple[LabelViewDescriptor, LabelViewDescriptor]:
    """Publish two sibling roots without ever creating a combined label file."""
    base = Path(base_root)
    construction_root = base / "target_construction_label_view"
    evaluation_root = base / "target_evaluation_label_view"
    require(not construction_root.exists() and not evaluation_root.exists(), "label view root already exists")
    holders: dict[str, LabelViewDescriptor] = {}
    atomic_publish_directory(
        construction_root,
        lambda staging: holders.setdefault("construction", _publish_one_view(staging, "construction", construction)),
    )
    # The descriptor root must name the published path, not the staging path.
    construction_descriptor = LabelViewDescriptor(
        "construction", str(construction_root), sha256_file(construction_root / "manifest.json"),
        len(construction), canonical_sha256({
            "kind": "construction", "root": str(construction_root),
            "manifest_sha256": sha256_file(construction_root / "manifest.json"),
            "row_count": len(construction),
        }),
    )
    try:
        atomic_publish_directory(
            evaluation_root,
            lambda staging: holders.setdefault("evaluation", _publish_one_view(staging, "evaluation", evaluation)),
        )
    except BaseException:
        import shutil
        shutil.rmtree(construction_root)
        raise
    evaluation_descriptor = LabelViewDescriptor(
        "evaluation", str(evaluation_root), sha256_file(evaluation_root / "manifest.json"),
        len(evaluation), canonical_sha256({
            "kind": "evaluation", "root": str(evaluation_root),
            "manifest_sha256": sha256_file(evaluation_root / "manifest.json"),
            "row_count": len(evaluation),
        }),
    )
    return construction_descriptor, evaluation_descriptor


def assert_physical_disjointness(
    construction_rows: Iterable[Mapping[str, Any]],
    evaluation_rows: Iterable[Mapping[str, Any]],
) -> None:
    left = {_identity(row) for row in construction_rows}
    right = {_identity(row) for row in evaluation_rows}
    require(not left & right, "construction/evaluation overlap")


def load_frozen_trial_registry(path: str | Path = FROZEN_TARGET_REGISTRY_PATH) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    require(payload["schema_version"] == "c84f_target_unlabeled_trial_registry_v1",
            "frozen target trial-registry schema drift")
    require(payload["complete_gate"]["complete"] is True, "frozen target trial registry is incomplete")
    rows = [dict(row) for row in payload["trials"]]
    require(len(rows) == 9621, "frozen target trial-registry row count drift")
    return rows


def label_rows_from_loader_result(result: Any, *, dataset: str, subject: int) -> list[dict[str, Any]]:
    """Read slots 1 and 2 only; the returned Epochs slot is never indexed."""
    require(isinstance(result, tuple) and len(result) == 3, "label loader result must have three slots")
    y_slot = result[1]
    metadata = result[2]
    columns = {str(column).strip().lower() for column in getattr(metadata, "columns", ())}
    forbidden_metadata = {
        column for column in columns
        if column in {"y", "target", "targets"}
        or any(token in column for token in ("label", "class", "event"))
    }
    require(not forbidden_metadata, "target metadata contains label-like fields")
    y_values = list(y_slot)
    count = len(metadata)
    require(len(y_values) == count, "loader label/metadata row count drift")

    def column(name: str, default: str) -> list[Any]:
        if hasattr(metadata, "columns") and name in metadata.columns:
            return metadata[name].tolist()
        return [default] * count

    subjects = [int(value) for value in column("subject", str(subject))]
    sessions = [str(value) for value in column("session", "0")]
    runs = [str(value) for value in column("run", "0")]
    require(set(subjects) == {int(subject)}, "label loader subject identity drift")
    rows = []
    for index, value in enumerate(y_values):
        text = str(value)
        require(text in CANONICAL_CLASSES, f"loader class mapping drift: {text}")
        rows.append({
            "dataset": dataset, "target_subject_id": subjects[index],
            "target_trial_id": (
                f"{dataset}|subject={subjects[index]}|session={sessions[index]}|"
                f"run={runs[index]}|trial={index:05d}"
            ),
            "session": sessions[index], "run": runs[index],
            "canonical_class_label": CANONICAL_CLASSES[text],
        })
    require(len({_identity(row) for row in rows}) == len(rows), "label loader trial IDs are not unique")
    return rows


def protected_loader_classes() -> tuple[Any, Any, Any, Any]:
    """Import loaders lazily; callers must first pass the C84S runtime guard."""
    datasets = importlib.import_module("moabb.datasets")
    paradigms = importlib.import_module("moabb.paradigms")
    return (
        datasets.Lee2019_MI, datasets.Cho2017, datasets.PhysionetMI,
        paradigms.MotorImagery,
    )


def provision_real_label_views(
    *,
    guard_receipt: Mapping[str, Any],
    output_root: str | Path,
    loader_classes: tuple[Any, Any, Any, Any] | None = None,
) -> tuple[LabelViewDescriptor, LabelViewDescriptor, dict[str, Any]]:
    """Future authorized Stage A entrypoint; unavailable during C84SL."""
    require(guard_receipt.get("C84S_authorized") is True, "C84S label provisioning lacks authorization receipt")
    require(guard_receipt.get("authorized_stage") == "C84S", "authorization receipt stage drift")
    from . import c84_dataset_registry_v2 as registry

    Lee2019_MI, Cho2017, PhysionetMI, MotorImagery = loader_classes or protected_loader_classes()
    factories = {
        "Lee2019_MI": lambda: Lee2019_MI(train_run=True, test_run=False),
        "Cho2017": Cho2017,
        "PhysionetMI": lambda: PhysionetMI(imagined=True, executed=False),
    }
    label_rows: list[dict[str, Any]] = []
    get_data_calls = 0
    for dataset_code, spec in registry.DATASETS.items():
        dataset = factories[dataset_code]()
        paradigm = MotorImagery(
            n_classes=2, events=["left_hand", "right_hand"], fmin=4.0, fmax=38.0,
            tmin=0.0, tmax=3.0,
            channels=[
                "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
                "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
            ],
            resample=160,
        )
        target_subjects = tuple(int(value) for value in registry.partition_subjects(spec)["targets"])
        for subject in target_subjects:
            result = paradigm.get_data(dataset=dataset, subjects=[subject], return_epochs=True)
            get_data_calls += 1
            label_rows.extend(label_rows_from_loader_result(result, dataset=dataset_code, subject=subject))
    frozen = load_frozen_trial_registry()
    construction, evaluation, audit = align_and_split_labels(frozen, label_rows)
    construction_descriptor, evaluation_descriptor = publish_physical_label_views(
        output_root, construction, evaluation,
    )
    return construction_descriptor, evaluation_descriptor, {
        **audit, "loader_calls": get_data_calls, "target_subjects": 118,
        "candidate_artifact_access": 0, "EEG_array_index_operations": 0,
        "same_label_oracle": 0,
    }
