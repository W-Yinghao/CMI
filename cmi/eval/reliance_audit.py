"""CMI-Trace P1.4 — fully-specified, hardened exact-head reliance audit.

Wraps cmi.eval.leakage_removal.evaluate_reliance so every reliance row carries COMPLETE, machine-readable
metadata (subspace construction, centering rule, metric/whitening, rank k, fit split, loss, source-only
firewall, number of random spans, replay max-abs error + tolerance). Adds:
  * random_span_control: average >= 50 same-rank random spans (same metric/rank) -> mean + bootstrap CI;
  * reliance_rank_sensitivity: a FIXED predeclared rank sequence (default k=1..7) x conditioning.

Primary endpoint stays the pre-registered k=2 label_conditional exact-head reliance R_rel. k is NEVER chosen
by target performance. Cross-MODEL raw subspace axes are never compared directly (each model's audit is
self-contained; assert_no_cross_model_axis_compare guards accidental misuse).
"""
from __future__ import annotations
import numpy as np

from cmi.eval.leakage_removal import evaluate_reliance, PRIMARY_K, PRIMARY_CONDITIONING, CONDITIONINGS
from cmi.eval.audit_npz import load_audit_npz, head_replay_ok, DEFAULT_REPLAY_TOL

# machine-readable description of the reliance algorithm (emitted alongside every audit)
RELIANCE_ALGORITHM = {
    "estimand": "R_rel(k) = task balanced-accuracy drop when the source-fit k-dim subject subspace is "
                "removed from the frozen representation (head-replay counterfactual).",
    "subspace_construction": {
        "label_conditional": "rows sqrt(n_{y,d})*(mean(z|y,d)-mu_y); top-k right singular vectors",
        "marginal_domain": "rows sqrt(n_d)*(mean(z|d)-mu); top-k right singular vectors (control)",
        "random_subspace": "top-k right singular vectors of a seeded Gaussian matrix (control)",
    },
    "within_label_centering_rule": "subtract the per-label mean mu_y before the subject offset "
                                   "(label_conditional); global mean (marginal_domain); none (random_subspace)",
    "metric_whitening_rule": "raw Euclidean (no whitening); orthonormal top-k right singular vectors; "
                             "removal P = I - dirs^T dirs applied in the same raw metric",
    "rank_k_primary": PRIMARY_K,
    "rank_sensitivity_sequence": [1, 2, 3, 4, 5, 6, 7],
    "loss_for_R_rel": "balanced_accuracy_drop (task bAcc before - after removal)",
    "fit_split": "source_only (domain != held-out target domain); target trials eval-only",
    "removal_mode_preference": "head_replay (verified linear head) else source-fit probe fallback",
    "n_random_spans_default": 50,
    "replay_tolerance": DEFAULT_REPLAY_TOL,
    "cross_model_axis_comparison": "FORBIDDEN (subspace axes are model-relative; compare drops, not axes)",
}


def _load(data_or_path):
    return load_audit_npz(data_or_path) if isinstance(data_or_path, (str, bytes)) else data_or_path


def reliance_audit_row(data_or_path, target_domain, k=PRIMARY_K, conditioning=PRIMARY_CONDITIONING,
                       seed=0, representation="graph_z", replay_tol=DEFAULT_REPLAY_TOL):
    """One fully-annotated reliance row. Adds all P1.4-required metadata to evaluate_reliance's output."""
    data = _load(data_or_path)
    row = evaluate_reliance(data, target_domain, k=k, conditioning=conditioning, seed=seed,
                            representation=representation)
    row.update(
        subspace_construction_method=conditioning,
        within_label_centering_rule=RELIANCE_ALGORITHM["within_label_centering_rule"],
        metric_whitening_rule=RELIANCE_ALGORITHM["metric_whitening_rule"],
        loss_for_R_rel=RELIANCE_ALGORITHM["loss_for_R_rel"],
        fit_split=RELIANCE_ALGORITHM["fit_split"],
        replay_max_abs_error=float(np.asarray(data.get("task_head_replay_max_abs_diff", np.nan))),
        replay_tolerance=float(replay_tol),
        head_replay_verified=bool(head_replay_ok(data)),
        source_only_firewall=True,
    )
    return row


def _boot_ci(vals, n_boot=2000, seed=0):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(1)
    return float(v.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def random_span_control(data_or_path, target_domain, k=PRIMARY_K, n_spans=50, seed=0,
                        representation="graph_z"):
    """Average n_spans (>=50 for the final interval) SAME-RANK random subspaces (same metric/rank), each a
    distinct seeded random_subspace removal. Returns mean task_drop + bootstrap CI + the per-span drops."""
    data = _load(data_or_path)
    drops = [evaluate_reliance(data, target_domain, k=k, conditioning="random_subspace",
                               seed=seed + i, representation=representation)["task_drop"]
             for i in range(int(n_spans))]
    mean, lo, hi = _boot_ci(drops, seed=seed)
    return {"k": int(k), "n_spans": int(n_spans), "random_task_drop_mean": mean,
            "random_task_drop_ci_lo": lo, "random_task_drop_ci_hi": hi, "per_span": drops}


def reliance_rank_sensitivity(data_or_path, target_domain, ks=(1, 2, 3, 4, 5, 6, 7),
                              conditioning=PRIMARY_CONDITIONING, seed=0, representation="graph_z",
                              n_random_spans=50):
    """Reliance across a FIXED predeclared rank sequence (default k=1..7), each with its same-rank random
    control. k is never selected by target performance. Returns a list of annotated rows."""
    rows = []
    for k in ks:
        r = reliance_audit_row(data_or_path, target_domain, k=k, conditioning=conditioning, seed=seed,
                               representation=representation)
        rc = random_span_control(data_or_path, target_domain, k=k, n_spans=n_random_spans, seed=seed,
                                 representation=representation)
        r.update(random_task_drop_mean=rc["random_task_drop_mean"],
                 random_task_drop_ci_lo=rc["random_task_drop_ci_lo"],
                 random_task_drop_ci_hi=rc["random_task_drop_ci_hi"],
                 n_random_spans=int(n_random_spans),
                 rank_sequence=list(ks))
        rows.append(r)
    return rows


def assert_no_cross_model_axis_compare(model_a, model_b):
    """Guard: subspace AXES are model-relative and must NEVER be compared across models (only the scalar
    reliance DROPS are comparable). Raises if two different models' raw axes are passed for direct comparison."""
    if model_a != model_b:
        raise ValueError("cross-model raw subspace axis comparison is forbidden; compare R_rel DROPS "
                         f"(scalars), not axes ({model_a!r} vs {model_b!r}).")
    return True
