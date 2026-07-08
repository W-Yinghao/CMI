"""C27 — FROZEN logit factor families + per-candidate feature computation from per-sample target logits. Every
family is a fixed deterministic function of a candidate's logits (NO labels, NO selection). Declared before
analysis; only pre-declared family unions are evaluated (not feature search)."""
from __future__ import annotations

import numpy as np

from . import schema


def _softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def candidate_features(L) -> dict:
    """All FROZEN logit factor-family features for one candidate's per-sample target logits L (Nsamp x C)."""
    L = np.asarray(L, dtype=np.float64)
    p = _softmax(L); pred = p.argmax(1); conf = p.max(1)
    srt = np.sort(p, 1); margin = srt[:, -1] - srt[:, -2]
    ent = -(p * np.log(np.clip(p, 1e-9, 1.0))).sum(1); lnorm = np.linalg.norm(L, axis=1)
    C = L.shape[1]; out = {}
    for k in range(C):
        mk = pred == k
        out[f"occ_c{k}"] = float(mk.mean())
        out[f"conf_c{k}"] = float(conf[mk].mean()) if mk.any() else 0.0        # class-conditioned confidence
        out[f"margin_c{k}"] = float(margin[mk].mean()) if mk.any() else 0.0
        out[f"bias_c{k}"] = float(L[:, k].mean())                              # class-bias (mean logit per class)
        out[f"occ_x_conf_c{k}"] = out[f"occ_c{k}"] * out[f"conf_c{k}"]         # occupancy x class-cond confidence
    for name, v in (("conf", conf), ("entropy", ent), ("margin", margin), ("logit_norm", lnorm)):
        out[f"{name}_mean"] = float(v.mean()); out[f"{name}_std"] = float(v.std())
    return out


def family_feature_names(*families) -> list:
    names = []
    for fam in families:
        names += list(schema.FAMILIES[fam])
    return names


def select(*families):
    """A feature_fn that selects the given families' features, reusing a candidate's cached `feats` when present
    (report precomputes candidate_features once per candidate)."""
    names = family_feature_names(*families)

    def fn(c):
        f = c.get("feats") if c.get("feats") is not None else candidate_features(c["L"])
        return {n: f[n] for n in names}
    return fn


def all_feature_names() -> list:
    seen, out = set(), []
    for fam in schema.FAMILIES:
        for f in schema.FAMILIES[fam]:
            if f not in seen:
                seen.add(f); out.append(f)
    return out


def feature_family_rows() -> list:
    return [{"feature": f, "family": fam} for fam, feats in schema.FAMILIES.items() for f in feats]
