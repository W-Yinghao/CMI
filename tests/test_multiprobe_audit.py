"""CIGL R1 — multi-probe stress audit: >=5/7 probes detect a planted within-label domain leak; a no-leak
representation clears (few/no probes detect) under FDR."""
import numpy as np

from cmi.eval.multiprobe_audit import multiprobe_leakage_audit, PROBE_NAMES


def _synth(leak, N=120, n_cls=2, n_dom=3, F=8, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_cls, N)
    d = rng.integers(0, n_dom, N)
    Z = 0.5 * np.eye(n_cls)[y] @ rng.standard_normal((n_cls, F))       # class signal
    Z = Z + 0.3 * rng.standard_normal((N, F))
    if leak:                                                            # plant domain info WITHIN label
        Z = Z + 1.5 * np.eye(n_dom)[d] @ rng.standard_normal((n_dom, F))
    return Z.astype("float32"), y.astype("int64"), d.astype("int64")


def test_multiprobe_detects_planted_leak():
    Z, y, d = _synth(leak=True)
    # n_perm>=20 so the minimum exact p 1/(n_perm+1) can fall below alpha -> BH-FDR can reject
    r = multiprobe_leakage_audit(Z, y, d, 2, 3, n_perm=25, seed=0, min_agree=5)
    assert r["n_probes"] == 7 and set(r["probes"]) <= set(PROBE_NAMES)
    assert r["n_detect_fdr"] >= 5 and r["leakage_exists"] is True
    # every reported probe carries a valid exact p in (0,1]
    for p, v in r["per_probe"].items():
        assert 0.0 < v["exact_p"] <= 1.0 and "fdr_rejected" in v


def test_multiprobe_clears_on_no_leak():
    Z, y, d = _synth(leak=False)
    r = multiprobe_leakage_audit(Z, y, d, 2, 3, n_perm=12, seed=0, min_agree=5)
    assert r["n_detect_fdr"] < 5 and r["leakage_exists"] is False       # no probe-family agreement
