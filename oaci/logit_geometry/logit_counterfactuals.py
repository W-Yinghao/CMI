"""C27-B — logit-space counterfactuals. Apply deterministic transforms to the FROZEN candidate target logits
(NO retraining) and ask WHICH transform DESTROYS the C24/C26 offset recovery (baseline = the full R3 gauge =
occupancy + global-confidence, +0.491). Per-sample transforms: temperature (softens confidence, preserves
occupancy), class-bias centering (removes per-class logit bias -> changes occupancy), logit-norm normalization
(removes scale), class-uniformization (pushes occupancy toward uniform). Gauge-level shuffles break the per-
target coupling between occupancy and confidence (confidence_shuffle / class_shuffle) -> tests whether the
per-target COUPLING is required."""
from __future__ import annotations

import numpy as np

from . import artifact_loader, factor_registry, schema

_FULL = ("occupancy", "global_confidence")


def _full_feats(c, transform=None):
    L = c["L"] if transform is None else transform(c["L"])
    f = factor_registry.candidate_features(L)
    return {n: f[n] for n in factor_registry.family_feature_names(*_FULL)}


def _temperature(L):
    return L / schema.TEMPERATURE


def _class_bias_center(L):
    return L - L.mean(0, keepdims=True)


def _logit_norm_normalize(L):
    return L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-9)


def _class_uniformize(L):
    p = factor_registry._softmax(L); pred = p.argmax(1); C = L.shape[1]
    occ = np.array([(pred == k).mean() for k in range(C)])
    return L - (np.log(occ + 1e-6) - np.log(1.0 / C))[None, :]


_TRANSFORMS = {"temperature": _temperature, "class_bias_center": _class_bias_center,
               "logit_norm_normalize": _logit_norm_normalize, "class_uniformize": _class_uniformize}


def _shuffle_table(table, names, shuffle_names, seed):
    rng = np.random.RandomState(seed); targets = sorted(table); perm = rng.permutation(len(targets))
    out = {}
    for i, t in enumerate(targets):
        src = targets[perm[i]]
        g = {n: (table[src]["gauge"][n] if n in shuffle_names else table[t]["gauge"][n]) for n in names}
        out[t] = {"gauge": g, "offset": table[t]["offset"]}
    return out


def counterfactuals(logit_cands, score_rows, mode, raw, oracle) -> dict:
    base = artifact_loader.recover(logit_cands, score_rows, mode, raw, oracle, _full_feats)
    results = {"raw": base}
    for name, tf in _TRANSFORMS.items():
        results[name] = artifact_loader.recover(logit_cands, score_rows, mode, raw, oracle, lambda c, tf=tf: _full_feats(c, tf))
    # gauge-level shuffles: break per-target occupancy<->confidence coupling
    table, names = artifact_loader.build_gauge(logit_cands, score_rows, mode, _full_feats)
    conf_names = factor_registry.family_feature_names("global_confidence")
    occ_names = factor_registry.family_feature_names("occupancy")
    results["confidence_shuffle"] = artifact_loader.recover_table(
        _shuffle_table(table, names, conf_names, schema.PERM_SEED), names, score_rows, mode, raw, oracle)
    results["class_shuffle"] = artifact_loader.recover_table(
        _shuffle_table(table, names, occ_names, schema.PERM_SEED + 1), names, score_rows, mode, raw, oracle)
    bg = base["gap_closed"]
    for name, r in results.items():
        r["destroys_recovery"] = bool(name != "raw" and bg is not None and r["gap_closed"] is not None
                                      and r["gap_closed"] < bg * (1 - schema.DESTROYS_FRACTION))
    destroyers = [n for n, r in results.items() if r.get("destroys_recovery")]
    return {"baseline_gap": bg, "per_intervention": results, "destroyers": destroyers,
            "note": ("interventions that destroy the offset recovery: %s -> the recovery depends on the factor(s) "
                     "they remove" % (", ".join(destroyers) if destroyers else "none (recovery robust to all tested transforms)"))}
