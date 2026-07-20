"""C23 — read-only loader for the C22 score sidecar (per-candidate frozen-probe score + 16 source observables
+ target/regime/epoch/order/R_src). Asserts the C19 config hash is unchanged. Nothing is re-scored or refit."""
from __future__ import annotations

import hashlib
import json
import math

from ..competence_probe import schema as c19
from . import schema


def _finite(v):
    return v is not None and not (isinstance(v, float) and not math.isfinite(v))


def load(sidecar_path=None) -> list:
    d = json.load(open(sidecar_path or schema.C22_SCORE_SIDECAR))
    got = hashlib.sha256(json.dumps(c19.frozen_config(), sort_keys=True).encode()).hexdigest()[:16]
    if d.get("config_hash") != schema.LOCKED_C19_CONFIG_HASH or got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C23 requires the frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; "
                         f"sidecar={d.get('config_hash')} recomputed={got}")
    return d["score_table"]


def by_target(rows, mode):
    out = {}
    for r in rows:
        if r["mode"] == mode:
            out.setdefault(r["target"], []).append(r)
    return out


def per_target_offset(rows, mode) -> dict:
    """Per-target score OFFSET = mean frozen-probe score of the target's candidates (the pooling-killer)."""
    import statistics as st
    return {t: st.mean([r["score"] for r in g]) for t, g in by_target(rows, mode).items()}
