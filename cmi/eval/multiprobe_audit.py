"""CIGL R1 — multi-probe stress audit. Wires the EXISTING 7-probe suite (cmi/eval/leakage_audit.audit:
linear, mlp_s, mlp_l, rf, hgbm, hsic, knn_cmi) over frozen features with a per-probe within-label
permutation null + BH-FDR across the probe family, and reports the agreement/envelope. Answers the reviewer's
"is the leakage probe-family-robust?" — success = >= 5/7 probe families detect leakage under FDR. The
phase-3A audit today runs only a single 1-layer MLP; this makes the claim robust to probe choice.
"""
from __future__ import annotations
import numpy as np

from cmi.eval.leakage_audit import audit as _probe_audit
from cmi.eval.graph_leakage import within_label_permutation
from cmi.eval.evidence_hardening import exact_permutation_pvalue, benjamini_hochberg

# keys as returned by cmi.eval.leakage_audit.audit: classifier probes carry an "_adv" (advantage over the
# label-conditional prior) suffix; hsic/knn_cmi are dependence measures. All "higher = more leakage".
PROBE_NAMES = ("linear_adv", "mlp_s_adv", "mlp_l_adv", "rf_adv", "hgbm_adv", "hsic", "knn_cmi")


def multiprobe_leakage_audit(Z, y, d, n_cls, n_dom, *, n_perm=100, alpha=0.05, seed=0, groups=None,
                             hsic_cap=1500, min_agree=5):
    """Run the 7-probe suite on frozen features Z [N,F] and test each probe against a within-label
    permutation null (shuffle D within Y so label-marginal structure is preserved). Returns per-probe
    observed leakage, null mean, exact p, BH-FDR rejection, and the agreement count. `y,d` int labels/domains.

    Leakage = the probe decodes D from Z beyond what the within-label null allows. Each classifier probe's
    value is balanced D-accuracy; hsic/knn_cmi are dependence measures — all "higher = more leakage".
    """
    y = np.asarray(y).astype(np.int64)
    d = np.asarray(d).astype(np.int64)
    observed = _probe_audit(Z, y, d, n_cls, n_dom, seed=seed, groups=groups, hsic_cap=hsic_cap)
    probes = [p for p in PROBE_NAMES if p in observed]
    # per-probe permutation null: shuffle D WITHIN label, re-run the whole probe suite
    null = {p: [] for p in probes}
    for j in range(n_perm):
        d_perm = within_label_permutation(y, d, seed=seed + 1 + j)
        res = _probe_audit(Z, y, d_perm, n_cls, n_dom, seed=seed, groups=groups, hsic_cap=hsic_cap)
        for p in probes:
            null[p].append(res[p])
    per_probe = {}
    pvals = []
    for p in probes:
        pv = exact_permutation_pvalue(observed[p], null[p], tail="greater")
        per_probe[p] = {"observed": float(observed[p]), "null_mean": float(np.mean(null[p])),
                        "null_std": float(np.std(null[p])), "exact_p": pv}
        pvals.append(pv)
    bh = benjamini_hochberg(pvals, alpha=alpha)
    for i, p in enumerate(probes):
        per_probe[p]["fdr_rejected"] = bool(bh["rejected"][i])
        per_probe[p]["adjusted_p"] = float(bh["adjusted_p"][i])
    n_detect = int(bh["n_rejected"])
    return {"per_probe": per_probe, "probes": probes, "n_probes": len(probes),
            "n_detect_fdr": n_detect, "min_agree": min_agree,
            "leakage_exists": n_detect >= min_agree,
            "envelope": {"observed_min": float(min(observed[p] for p in probes)),
                         "observed_max": float(max(observed[p] for p in probes))},
            "n_perm": n_perm, "alpha": alpha}
