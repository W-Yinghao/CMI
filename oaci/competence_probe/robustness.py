"""C19 — regime robustness. Runs the frozen probe (robust-core primary + endpoint-augmented secondary) on the
cell-PRESENT regimes S0/S2/S3 (the regimes where the C18 signal was preserved). Leakage is recomputed per
regime via the C18 row-consistent estimator (restricted to these regimes). Deletion regimes are NOT scored
here — they are the endpoint-nonestimability domain and are reported by the estimability gate / C18."""
from __future__ import annotations

from ..support_stress import source_signal_recompute as ssr
from . import feature_registry, schema, validation


def run_robustness(extract_dir, c10_dir, *, boundary_classes, folds=None, n_perm=schema.N_PERM, n_workers=8) -> dict:
    regimes = list(schema.ROBUSTNESS_REGIMES)
    cache = ssr.precompute_all_leakage(extract_dir, boundary_classes=boundary_classes, folds=folds,
                                       n_workers=n_workers, regimes=regimes)

    def lookup(regime):
        return lambda s, t, l, mh: cache.get((s, t, l, regime, mh), (None, None))

    out = {}
    for regime in regimes:
        rows = feature_registry.build_atlas(extract_dir, c10_dir, regime, boundary_classes=boundary_classes,
                                            leakage_lookup=lookup(regime), folds=folds)
        out[regime] = validation.evaluate_regime(rows, n_perm=n_perm)
    return out
