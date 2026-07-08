"""Label-conditional domain heads for frozen CEDAR audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ProbeResult:
    probe: str
    domain_bacc: float
    prior_bacc: float
    advantage: float
    n_train: int
    n_eval: int

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _as_int_array(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return arr.astype(np.int64, copy=False)


def label_conditioned_features(z: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """Concatenate frozen features with a one-hot task label."""

    z = np.asarray(z, dtype=np.float64)
    y = _as_int_array(y, "y")
    if z.ndim != 2:
        raise ValueError("z must be a 2D array")
    if len(z) != len(y):
        raise ValueError("z and y length mismatch")
    if n_classes <= int(y.max(initial=0)):
        raise ValueError("n_classes is too small for y")
    y_oh = np.eye(n_classes, dtype=np.float64)[y]
    return np.concatenate([z, y_oh], axis=1)


def conditional_prior_predict(
    y_train: np.ndarray,
    d_train: np.ndarray,
    y_eval: np.ndarray,
    n_classes: int,
    n_domains: int,
) -> np.ndarray:
    """Predict the most common domain inside each source class."""

    y_train = _as_int_array(y_train, "y_train")
    d_train = _as_int_array(d_train, "d_train")
    y_eval = _as_int_array(y_eval, "y_eval")
    counts = np.zeros((n_classes, n_domains), dtype=np.float64)
    for yi, di in zip(y_train, d_train):
        counts[yi, di] += 1.0
    global_domain = int(np.argmax(counts.sum(axis=0)))
    per_class = counts.argmax(axis=1)
    empty = counts.sum(axis=1) == 0
    per_class[empty] = global_domain
    return per_class[y_eval]


def _balanced_accuracy_present(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(y_true)
    recalls = []
    for label in labels:
        idx = y_true == label
        if idx.sum():
            recalls.append(float((y_pred[idx] == label).mean()))
    if not recalls:
        return float("nan")
    return float(np.mean(recalls))


def _make_probe(kind: str, seed: int, max_iter: int, hidden: tuple[int, ...]):
    if kind == "linear":
        model = LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            random_state=seed,
            solver="lbfgs",
        )
    elif kind == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=hidden,
            max_iter=max_iter,
            early_stopping=True,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown probe kind '{kind}'")
    return make_pipeline(StandardScaler(), model)


def fit_conditional_domain_probe(
    z_train: np.ndarray,
    y_train: np.ndarray,
    d_train: np.ndarray,
    z_eval: np.ndarray,
    y_eval: np.ndarray,
    d_eval: np.ndarray,
    *,
    n_classes: int,
    n_domains: int,
    probe: str = "linear",
    seed: int = 0,
    max_iter: int = 500,
    hidden: tuple[int, ...] = (64,),
) -> ProbeResult:
    """Fit q(D | Z, Y) and report advantage over q(D | Y)."""

    y_train = _as_int_array(y_train, "y_train")
    d_train = _as_int_array(d_train, "d_train")
    y_eval = _as_int_array(y_eval, "y_eval")
    d_eval = _as_int_array(d_eval, "d_eval")
    if len(np.unique(d_train)) < 2 or len(np.unique(d_eval)) < 2:
        raise ValueError("domain probe requires at least two domains in train and eval")
    f_train = label_conditioned_features(z_train, y_train, n_classes)
    f_eval = label_conditioned_features(z_eval, y_eval, n_classes)
    pred = _make_probe(probe, seed, max_iter, hidden).fit(f_train, d_train).predict(f_eval)
    prior = conditional_prior_predict(y_train, d_train, y_eval, n_classes, n_domains)
    domain_bacc = _balanced_accuracy_present(d_eval, pred)
    prior_bacc = _balanced_accuracy_present(d_eval, prior)
    return ProbeResult(
        probe=probe,
        domain_bacc=domain_bacc,
        prior_bacc=prior_bacc,
        advantage=domain_bacc - prior_bacc,
        n_train=int(len(y_train)),
        n_eval=int(len(y_eval)),
    )
