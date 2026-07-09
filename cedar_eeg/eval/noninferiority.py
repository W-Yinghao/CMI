"""Task non-inferiority checks for frozen feature surgery."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cedar_eeg.probes.crossfit_grouped import make_folds


@dataclass(frozen=True)
class NonInferiorityResult:
    before: float
    after: float
    margin: float
    passed: bool

    @property
    def drop(self) -> float:
        return float(self.before - self.after)

    def to_dict(self) -> dict[str, float | bool]:
        out = asdict(self)
        out["drop"] = self.drop
        return out


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> float:
    y_true = np.asarray(y_true).astype(np.int64, copy=False)
    y_pred = np.asarray(y_pred).astype(np.int64, copy=False)
    if n_classes is None:
        labels = np.unique(y_true)
    else:
        labels = np.arange(n_classes)
    recalls = []
    for label in labels:
        idx = y_true == label
        if idx.sum() == 0:
            continue
        recalls.append(float((y_pred[idx] == label).mean()))
    if not recalls:
        return float("nan")
    return float(np.mean(recalls))


def _task_readout(seed: int, max_iter: int = 500):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=seed),
    )


def crossfit_task_bacc(
    z: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    n_classes: int | None = None,
    n_splits: int = 3,
    seed: int = 0,
) -> float:
    """Cross-fit a linear source readout on frozen features."""

    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y).astype(np.int64, copy=False)
    if n_classes is None:
        n_classes = int(y.max()) + 1
    preds = np.full(len(y), -1, dtype=np.int64)
    for fold_id, (tr, ev) in enumerate(make_folds(len(y), groups=groups, n_splits=n_splits, seed=seed)):
        if len(np.unique(y[tr])) < 2:
            continue
        preds[ev] = _task_readout(seed + fold_id).fit(z[tr], y[tr]).predict(z[ev])
    ok = preds >= 0
    if ok.sum() == 0:
        raise ValueError("no valid task folds")
    return balanced_accuracy(y[ok], preds[ok], n_classes)


def _aligned_proba(model, z_eval: np.ndarray, n_classes: int) -> np.ndarray:
    raw = model.predict_proba(z_eval)
    out = np.zeros((len(z_eval), n_classes), dtype=np.float64)
    for raw_idx, label in enumerate(model.classes_):
        label_int = int(label)
        if 0 <= label_int < n_classes:
            out[:, label_int] = raw[:, raw_idx]
    missing = out.sum(axis=1) <= 0.0
    if np.any(missing):
        out[missing, :] = 1.0 / float(n_classes)
    return out


def _nll(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    return float(log_loss(y_true, proba, labels=np.arange(n_classes)))


def crossfit_task_metrics(
    z: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    n_classes: int | None = None,
    n_splits: int = 3,
    seed: int = 0,
) -> dict[str, float | int]:
    """Cross-fit a linear source readout and report bAcc plus CE/NLL."""

    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y).astype(np.int64, copy=False)
    if n_classes is None:
        n_classes = int(y.max()) + 1
    preds = np.full(len(y), -1, dtype=np.int64)
    probas = np.full((len(y), n_classes), np.nan, dtype=np.float64)
    n_folds = 0
    for fold_id, (tr, ev) in enumerate(make_folds(len(y), groups=groups, n_splits=n_splits, seed=seed)):
        if len(np.unique(y[tr])) < 2:
            continue
        model = _task_readout(seed + fold_id).fit(z[tr], y[tr])
        preds[ev] = model.predict(z[ev])
        probas[ev] = _aligned_proba(model, z[ev], n_classes)
        n_folds += 1
    ok = preds >= 0
    if ok.sum() == 0:
        raise ValueError("no valid task folds")
    nll = _nll(y[ok], probas[ok], n_classes)
    return {
        "bacc": balanced_accuracy(y[ok], preds[ok], n_classes),
        "ce": nll,
        "nll": nll,
        "n_eval": int(ok.sum()),
        "n_folds": int(n_folds),
    }


def fit_source_eval_target_bacc(
    z_source: np.ndarray,
    y_source: np.ndarray,
    z_target: np.ndarray,
    y_target: np.ndarray,
    *,
    n_classes: int | None = None,
    seed: int = 0,
) -> float:
    """Train a source readout and evaluate on held-out target labels.

    This is evaluation-only and must not be used to choose masks.
    """

    z_source = np.asarray(z_source, dtype=np.float64)
    y_source = np.asarray(y_source).astype(np.int64, copy=False)
    z_target = np.asarray(z_target, dtype=np.float64)
    y_target = np.asarray(y_target).astype(np.int64, copy=False)
    if n_classes is None:
        n_classes = int(max(y_source.max(initial=0), y_target.max(initial=0))) + 1
    pred = _task_readout(seed).fit(z_source, y_source).predict(z_target)
    return balanced_accuracy(y_target, pred, n_classes)


def fit_source_eval_target_metrics(
    z_source: np.ndarray,
    y_source: np.ndarray,
    z_target: np.ndarray,
    y_target: np.ndarray,
    *,
    n_classes: int | None = None,
    seed: int = 0,
) -> dict[str, float | int]:
    """Train source readout and evaluate target diagnostics only."""

    z_source = np.asarray(z_source, dtype=np.float64)
    y_source = np.asarray(y_source).astype(np.int64, copy=False)
    z_target = np.asarray(z_target, dtype=np.float64)
    y_target = np.asarray(y_target).astype(np.int64, copy=False)
    if n_classes is None:
        n_classes = int(max(y_source.max(initial=0), y_target.max(initial=0))) + 1
    model = _task_readout(seed).fit(z_source, y_source)
    pred = model.predict(z_target)
    proba = _aligned_proba(model, z_target, n_classes)
    nll = _nll(y_target, proba, n_classes)
    return {
        "bacc": balanced_accuracy(y_target, pred, n_classes),
        "ce": nll,
        "nll": nll,
        "n_eval": int(len(y_target)),
    }


def noninferiority(before: float, after: float, margin: float) -> NonInferiorityResult:
    return NonInferiorityResult(
        before=float(before),
        after=float(after),
        margin=float(margin),
        passed=bool((before - after) <= margin),
    )
