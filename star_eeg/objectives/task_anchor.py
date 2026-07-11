"""Source-only task-anchor scaffold and dependency-light synthetic smoke.

The toy adapter proves loss/gradient plumbing without importing CBraMod or
reading EEG. The future real path is frozen to the existing CBraMod encoder
feature extraction (`encoder(patch_embedding(x)).mean(patches).flatten()`) and
a temporary 6400-to-9 linear head, discarded before frozen evaluation.
"""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass
class TinyEncoderAdapter:
    scale: float = 0.8
    bias: float = 0.1

    def features(self, source_x: Sequence[float]) -> List[float]:
        return [self.scale * float(value) + self.bias for value in source_x]


@dataclass
class TemporaryTaskHead:
    weights: List[float]
    biases: List[float]

    @classmethod
    def deterministic(cls, n_classes: int = 3) -> "TemporaryTaskHead":
        return cls(
            weights=[0.05 * (index + 1) for index in range(n_classes)],
            biases=[-0.01 * index for index in range(n_classes)],
        )


def _finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def ssl_step(adapter: TinyEncoderAdapter, source_x: Sequence[float], learning_rate: float = 1e-2) -> Dict[str, object]:
    """One toy reconstruction update; no dataset or scientific metric."""
    if not source_x:
        raise ValueError("synthetic SSL batch is empty")
    values = [float(value) for value in source_x]
    predictions = adapter.features(values)
    residuals = [prediction - target for prediction, target in zip(predictions, values)]
    loss = sum(value * value for value in residuals) / len(residuals)
    grad_scale = 2.0 * sum(residual * value for residual, value in zip(residuals, values)) / len(values)
    grad_bias = 2.0 * sum(residuals) / len(values)
    adapter.scale -= learning_rate * grad_scale
    adapter.bias -= learning_rate * grad_bias
    return {
        "loss": loss,
        "gradients": [grad_scale, grad_bias],
        "finite_loss": math.isfinite(loss),
        "finite_gradients": _finite([grad_scale, grad_bias]),
    }


def source_task_anchor_step(
    adapter: TinyEncoderAdapter,
    task_head: TemporaryTaskHead,
    source_x: Sequence[float],
    source_y: Sequence[int],
    learning_rate: float = 1e-2,
) -> Dict[str, object]:
    """One toy source-label cross-entropy update with encoder gradients."""
    if len(source_x) != len(source_y) or not source_x:
        raise ValueError("source_x/source_y batch mismatch")
    n_classes = len(task_head.weights)
    if n_classes < 2 or len(task_head.biases) != n_classes:
        raise ValueError("invalid temporary task head")
    if any(int(label) not in range(n_classes) for label in source_y):
        raise ValueError("source_y outside toy class range")

    grad_w = [0.0] * n_classes
    grad_b = [0.0] * n_classes
    grad_scale = 0.0
    grad_encoder_bias = 0.0
    loss = 0.0
    features = adapter.features(source_x)
    for raw_x, feature, label in zip(source_x, features, source_y):
        logits = [weight * feature + bias for weight, bias in zip(task_head.weights, task_head.biases)]
        shift = max(logits)
        exp_values = [math.exp(value - shift) for value in logits]
        denom = sum(exp_values)
        probabilities = [value / denom for value in exp_values]
        loss -= math.log(max(probabilities[int(label)], 1e-12))
        grad_feature = 0.0
        for class_index in range(n_classes):
            grad_logit = probabilities[class_index] - (1.0 if class_index == int(label) else 0.0)
            grad_w[class_index] += grad_logit * feature
            grad_b[class_index] += grad_logit
            grad_feature += grad_logit * task_head.weights[class_index]
        grad_scale += grad_feature * float(raw_x)
        grad_encoder_bias += grad_feature

    batch = float(len(source_x))
    loss /= batch
    grad_w = [value / batch for value in grad_w]
    grad_b = [value / batch for value in grad_b]
    grad_scale /= batch
    grad_encoder_bias /= batch
    task_head.weights = [value - learning_rate * grad for value, grad in zip(task_head.weights, grad_w)]
    task_head.biases = [value - learning_rate * grad for value, grad in zip(task_head.biases, grad_b)]
    adapter.scale -= learning_rate * grad_scale
    adapter.bias -= learning_rate * grad_encoder_bias
    gradients = [grad_scale, grad_encoder_bias, *grad_w, *grad_b]
    return {
        "loss": loss,
        "gradients": gradients,
        "finite_loss": math.isfinite(loss),
        "finite_gradients": _finite(gradients),
    }


def synthetic_training_step_smoke() -> Dict[str, object]:
    adapter = TinyEncoderAdapter()
    task_head = TemporaryTaskHead.deterministic(n_classes=3)
    ssl = ssl_step(adapter, [0.25, -0.5, 1.0, 0.75])
    anchor = source_task_anchor_step(
        adapter,
        task_head,
        source_x=[-0.75, 0.2, 0.9, 0.4],
        source_y=[0, 1, 2, 1],
    )
    return {
        "status": "PASS" if all([
            ssl["finite_loss"], ssl["finite_gradients"],
            anchor["finite_loss"], anchor["finite_gradients"],
        ]) else "FAIL",
        "adapter": "tiny_cbramod_compatible_mock",
        "ssl_steps": 1,
        "task_anchor_steps": 1,
        "ssl_loss": ssl["loss"],
        "task_anchor_loss": anchor["loss"],
        "finite_losses": bool(ssl["finite_loss"] and anchor["finite_loss"]),
        "finite_gradients": bool(ssl["finite_gradients"] and anchor["finite_gradients"]),
        "real_eeg_used": False,
        "scientific_metric_computed": False,
        "temporary_task_head_discarded": True,
    }
