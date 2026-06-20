"""Source-free test-time adaptation BASELINES, evaluated under the SAME protocol as CITA (same frozen source
checkpoint, same target batch, no target labels, no source examples at deployment). Each consumes a serialized
`SourceState` (cmi/eval/source_state.py) + the unlabeled target features.

Implemented:
  - T3A  (Iwasawa & Matsuo, NeurIPS 2021): test-time classifier adjustment — per-class support sets initialized
    with the source linear templates, augmented by low-entropy pseudo-labeled target features; classify by the
    adjusted support centroids. Backbone frozen, no gradient, label-free, source-free.
  - SPDIM placeholder lives in this module too once implemented.

Order of implementation (per protocol): T3A first (validates the serialized-state / target-only path), then SPDIM.
"""
import numpy as np


def _softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x); return e / e.sum(1, keepdims=True)


def _class_templates(state):
    """Per-class linear templates from the frozen source readout (T3A's initial support). Binary LR stores a
    single weight row -> templates (-w, +w); multiclass stores one row per class."""
    clf = state["clf"]; n_cls = state["n_cls"]
    W, b = np.atleast_2d(clf.coef_), np.atleast_1d(clf.intercept_)
    if W.shape[0] == 1 and n_cls == 2:
        return np.stack([-W[0], W[0]]), np.array([-b[0], b[0]])
    return W, b


def t3a_predict(state, z_tgt, filter_M=20, use_prototype=False):
    """T3A on the alignment embedding. filter_M = support-set size cap per class (lowest entropy kept).
    Source-free: reads only `state` (frozen readout + class moments) and the unlabeled target features."""
    z_tgt = np.asarray(z_tgt, float)
    clf, n_cls = state["clf"], state["n_cls"]
    cls = clf.classes_
    prob = np.zeros((len(z_tgt), n_cls)); prob[:, cls] = clf.predict_proba(z_tgt)
    pred = prob.argmax(1)
    ent = -(prob * np.log(np.clip(prob, 1e-12, 1))).sum(1)
    templ, _ = _class_templates(state)
    init = state["mu_y"] if use_prototype else templ        # prototype variant uses source class means
    centroids = []
    for k in range(n_cls):
        idx = np.where(pred == k)[0]
        if len(idx):
            idx = idx[np.argsort(ent[idx])[:filter_M]]      # keep the most confident pseudo-labeled targets
            supp = np.vstack([init[k][None], z_tgt[idx]])
        else:
            supp = init[k][None]
        centroids.append(supp.mean(0))
    centroids = np.stack(centroids)
    logits = z_tgt @ centroids.T                            # adjusted nearest-template classification
    return _softmax(logits)
