"""Fail-closed signature and split firewall checks."""

import inspect
from typing import Callable, Dict, Iterable

from star_eeg.data.faced_split_contract import contract_payload
from star_eeg.objectives.task_anchor import source_task_anchor_step, ssl_step


FORBIDDEN_TRAINING_PARAMETERS = {
    "target_y",
    "target_labels",
    "target_metrics",
    "target_class_distribution",
    "source_val_y",
    "source_val_labels",
}


def signature_parameters(function: Callable[..., object]) -> Iterable[str]:
    return inspect.signature(function).parameters.keys()


def inspect_training_signatures() -> Dict[str, object]:
    functions = {
        "ssl_step": ssl_step,
        "source_task_anchor_step": source_task_anchor_step,
    }
    signatures = {name: str(inspect.signature(function)) for name, function in functions.items()}
    violations = {
        name: sorted(set(signature_parameters(function)) & FORBIDDEN_TRAINING_PARAMETERS)
        for name, function in functions.items()
    }
    violations = {name: values for name, values in violations.items() if values}
    split = contract_payload()
    checks = {
        "target_y_absent_from_training_signatures": not violations,
        "anchor_api_has_source_y": "source_y" in signature_parameters(source_task_anchor_step),
        "source_train_labels_only_for_gradient": split["source_train_labels_allowed_for_anchor"] is True,
        "source_val_labels_for_gradient_forbidden": split["source_val_labels_allowed_for_gradient"] is False,
        "target_labels_before_final_scoring_forbidden": split["target_test_labels_allowed_before_final_scoring"] is False,
        "target_class_distribution_during_training_forbidden": split["target_class_distribution_allowed_during_training"] is False,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "training_function_signatures": signatures,
        "violations": violations,
        "checks": checks,
        "training_reads": ["TUEG_unlabeled", "FACED_source_train_X", "FACED_source_train_y"],
        "training_forbidden_reads": [
            "FACED_source_val_labels_for_gradient",
            "FACED_target_test_labels",
            "target_metrics",
            "target_class_distribution",
        ],
        "target_labels_final_scoring_only_in_future_star01": True,
    }
