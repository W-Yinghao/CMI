"""C34 read-only artifact loader."""
from __future__ import annotations

from ..local_boundary import artifact_loader as c33_loader
from . import endpoint_utility, schema


def load_rows(scores_sidecar=None, c10_dir=None, reinfer_sidecar=None, mode="in_regime", margin=None):
    margin = schema.PRIMARY_MARGIN if margin is None else margin
    rows, tu = c33_loader.load_rows(scores_sidecar, c10_dir, reinfer_sidecar, mode=mode, margin=margin)
    rows = [r for r in rows if all(endpoint_utility.finite(r.get(k))
                                   for k in ("bacc_delta", "nll_improve", "ece_improve", "score", "epoch"))]
    endpoint_utility.attach_endpoint_utilities(rows)
    return rows, tu
