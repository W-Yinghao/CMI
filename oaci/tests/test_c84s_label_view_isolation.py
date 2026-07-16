from __future__ import annotations

import ast
from pathlib import Path

import pytest

from oaci.multidataset import c84s_label_views as views
from oaci.multidataset.c84s_common import C84SContractError, sha256_file


def fixture_rows(subjects: int = 2, per_class: int = 20):
    registry, labels = [], []
    for subject in range(subjects):
        for class_id in (0, 1):
            for index in range(per_class):
                row = {
                    "dataset": "SyntheticMI", "target_subject_id": subject,
                    "target_trial_id": f"s{subject}-c{class_id}-t{index}",
                    "session": "0", "run": str(index % 2),
                }
                registry.append(row)
                labels.append({**row, "canonical_class_label": class_id})
    return registry, labels


def test_hash_split_is_exact_disjoint_and_deterministic() -> None:
    registry, labels = fixture_rows()
    left, right, audit = views.align_and_split_labels(registry, labels)
    left2, right2, audit2 = views.align_and_split_labels(registry, list(reversed(labels)))
    assert left == left2 and right == right2 and audit == audit2
    assert len(left) == len(right) == 40
    assert audit["overlap"] == 0
    views.assert_physical_disjointness(left, right)


def test_missing_or_extra_trial_identity_fails() -> None:
    registry, labels = fixture_rows()
    labels[0]["target_trial_id"] = "drift"
    with pytest.raises(C84SContractError, match="alignment"):
        views.align_and_split_labels(registry, labels)


def test_low_support_fails_closed_without_alternative_split() -> None:
    registry, labels = fixture_rows(per_class=14)
    with pytest.raises(C84SContractError, match="support below minimum"):
        views.align_and_split_labels(registry, labels)


def test_candidate_or_eeg_field_in_label_input_fails() -> None:
    registry, labels = fixture_rows()
    labels[0]["logits"] = [0.0, 1.0]
    with pytest.raises(C84SContractError, match="forbidden"):
        views.align_and_split_labels(registry, labels)


def test_physical_views_publish_as_separate_roots(tmp_path: Path) -> None:
    registry, labels = fixture_rows()
    construction, evaluation, _ = views.align_and_split_labels(registry, labels)
    left, right = views.publish_physical_label_views(tmp_path, construction, evaluation)
    assert Path(left.root).name == "target_construction_label_view"
    assert Path(right.root).name == "target_evaluation_label_view"
    assert Path(left.root) != Path(right.root)
    assert sha256_file(Path(left.root) / "manifest.json") == left.manifest_sha256
    assert sha256_file(Path(right.root) / "manifest.json") == right.manifest_sha256
    assert not (tmp_path / "all_labels.csv").exists()


def test_label_provisioner_has_no_candidate_module_import() -> None:
    path = Path(views.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any("selector" in name or "evaluation" in name for name in imports)


def test_label_loader_never_touches_epochs_slot() -> None:
    class ExplodingEpochs:
        def __getattribute__(self, name):
            raise AssertionError(f"epochs slot touched: {name}")

    class Column:
        def __init__(self, values):
            self.values = values
        def tolist(self):
            return list(self.values)

    class Metadata:
        columns = ("subject", "session", "run")
        def __init__(self):
            self.data = {
                "subject": [7] * 4, "session": ["0"] * 4,
                "run": ["0", "0", "1", "1"],
            }
        def __len__(self):
            return 4
        def __getitem__(self, key):
            return Column(self.data[key])

    rows = views.label_rows_from_loader_result(
        (ExplodingEpochs(), ["left_hand", "right_hand"] * 2, Metadata()),
        dataset="SyntheticMI", subject=7,
    )
    assert len(rows) == 4
    assert [row["canonical_class_label"] for row in rows] == [0, 1, 0, 1]
