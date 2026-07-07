"""C24 R3/R4 — target-UNLABELED feature construction. Stage-1 ships (a) the LABEL-FREE confidence-geometry
PRIMITIVE that operates on target logits ONLY (never touches y), (b) the planned R3 feature registry, and (c)
an availability-GATED gauge builder that returns REQUIRES_REINFERENCE when per-candidate target logits are not
yet produced -- it NEVER proxies from method-final checkpoints (wrong population). Stage-3 (after the P0 replay-
identity gate) supplies per-candidate target-unlabeled summaries via the sidecar and this same builder computes
the real R3/R4 gauge. Forbidden: target labels, target bAcc/NLL/ECE, target-centering, target identity."""
from __future__ import annotations

import numpy as np

from . import schema


def _has_forbidden(name) -> bool:
    """Word-boundary aware: a short token (<4 chars, e.g. 'y') must appear as a whole underscore-delimited word;
    a longer token may substring-match. Prevents 'y' from matching inside 'entropy' while still catching
    'target_bacc' inside 'target_bacc_good'."""
    low = name.lower(); parts = set(low.split("_"))
    for tok in schema.FORBIDDEN_TARGET_UNLABELED_INPUTS:
        if len(tok) < 4:
            if tok in parts:
                return True
        elif tok in low:
            return True
    return False


def assert_no_target_labels(feature_names) -> None:
    bad = [f for f in feature_names if _has_forbidden(f)]
    if bad:
        raise ValueError(f"R3/R4 target-unlabeled features contain forbidden target-label/identity tokens: {bad}")


def _softmax(z):
    z = np.asarray(z, float); z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def label_free_confidence_geometry(logits, n_classes=None) -> dict:
    """Per-CANDIDATE target-unlabeled summary from target logits ONLY (N x C). NEVER receives or reads labels.
    entropy / confidence / margin / logit_norm moments (mean, std) + predicted-class proportions."""
    z = np.asarray(logits, float)
    if z.ndim != 2 or len(z) == 0:
        raise ValueError("label_free_confidence_geometry expects a non-empty (N, C) target logits array")
    C = z.shape[1] if n_classes is None else n_classes
    p = _softmax(z)
    ent = -(p * np.log(np.clip(p, 1e-9, 1.0))).sum(1)
    conf = p.max(1)
    srt = np.sort(p, 1); margin = srt[:, -1] - srt[:, -2]
    lnorm = np.linalg.norm(z, axis=1)
    pred = p.argmax(1)
    out = {}
    for name, v in (("entropy", ent), ("confidence", conf), ("margin", margin), ("logit_norm", lnorm)):
        out[f"target_{name}_mean"] = float(v.mean()); out[f"target_{name}_std"] = float(v.std())
    for c in range(C):
        out[f"target_pred_prop_c{c}"] = float((pred == c).mean())
    return out


def target_unlabeled_feature_names(n_classes=4) -> list:
    names = []
    for s in ("entropy", "confidence", "margin", "logit_norm"):
        names += [f"target_{s}_mean", f"target_{s}_std"]
    names += [f"target_pred_prop_c{c}" for c in range(n_classes)]
    return names


def build_target_unlabeled_gauge(rows, availability, mode="in_regime", sidecar=None) -> dict:
    """Per-target target-unlabeled gauge for R3. Returns {status, ...}. Stage-1: REQUIRES_REINFERENCE (no proxy).
    Stage-3: consumes per-candidate target-unlabeled summaries from `sidecar` and aggregates per target."""
    names = target_unlabeled_feature_names()
    assert_no_target_labels(names)
    if not availability.get("per_candidate_target_unlabeled_ready"):
        return {"status": schema.STATUS_REQUIRES_REINFERENCE, "gauge_table": None,
                "reason": availability["method_final_note"],
                "planned_feature_names": names,
                "note": ("R3/R4 not computed: per-candidate target logits require a scoped NO-RETRAINING target-"
                         "audit re-inference behind the P0 replay-identity gate. No method-final proxy is used.")}
    # Stage-3 path: aggregate per-candidate target-unlabeled summaries (already label-free) into a per-target gauge
    import statistics as st
    percand = {(c["seed"], c["target"], c["level"], c["model_hash"]): c["target_unlabeled"]
               for c in sidecar["per_candidate"]}
    by_t = {}
    for r in rows:
        if r["mode"] != mode:
            continue
        key = (r["seed"], r["target"], r["level"], r["model_hash"])
        if key in percand:
            by_t.setdefault(r["target"], []).append((r, percand[key]))
    table = {}
    for t, items in by_t.items():
        gv = {}
        for n in names:
            vals = [tu[n] for _, tu in items if tu.get(n) is not None]
            gv[n] = float(np.mean(vals)) if vals else 0.0
        table[t] = {"gauge": gv, "offset": st.mean([r["score"] for r, _ in items]), "n": len(items)}
    return {"status": schema.STATUS_OK, "gauge_table": table, "feature_names": names}
