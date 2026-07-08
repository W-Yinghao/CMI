"""P0 frozen-latent CEDAR audit.

This runner consumes saved feature dumps. It must not train EEG backbones. For
real feature dumps, submit it through Slurm rather than running a large audit on
the login node.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cedar_eeg.config import DEFAULT_P0_THRESHOLDS, DEFAULT_PROBE, P0Thresholds, parse_drop_fractions
from cedar_eeg.eval.leakage_atlas import latent_scores_to_atlas
from cedar_eeg.eval.noninferiority import crossfit_task_bacc, fit_source_eval_target_bacc
from cedar_eeg.probes.crossfit_grouped import crossfit_conditional_domain_probe
from cedar_eeg.surgery.latent_mask import (
    apply_diagonal_mask,
    candidate_drop_sets,
    latent_dimension_scores,
    mask_from_drop_dims,
    rank_latent_dimensions,
)
from cedar_eeg.surgery.selection import SurgeryCandidate, decide_p0, score_candidate, target_eval_warnings


def _get_first(data: np.lib.npyio.NpzFile, keys: tuple[str, ...], required: bool = True):
    for key in keys:
        if key in data:
            return data[key]
    if required:
        raise KeyError(f"feature npz missing one of {keys}")
    return None


def load_feature_npz(path: str | Path) -> dict[str, np.ndarray | None]:
    data = np.load(path, allow_pickle=False)
    return {
        "z": _get_first(data, ("z", "Z", "features")),
        "y": _get_first(data, ("y", "labels")),
        "d": _get_first(data, ("domain", "d", "domains")),
        "groups": _get_first(data, ("groups", "recording", "session"), required=False),
        "z_target": _get_first(data, ("z_target", "Z_target", "target_z"), required=False),
        "y_target": _get_first(data, ("y_target", "target_y"), required=False),
    }


def _positive_adv(report: dict[str, object]) -> float:
    return max(0.0, float(report["advantage_mean"]))


def _random_drop_control(
    z: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    groups: np.ndarray | None,
    *,
    n_classes: int,
    n_domains: int,
    base_leakage: float,
    k_drop: int,
    exclude_dims: tuple[int, ...],
    repeats: int,
    seed: int,
    probe: str,
    n_splits: int,
) -> float:
    if base_leakage <= 1e-8:
        return 0.0
    rng = np.random.default_rng(seed)
    excluded = set(int(x) for x in exclude_dims)
    pool = np.asarray([i for i in range(z.shape[1]) if i not in excluded], dtype=np.int64)
    if len(pool) < k_drop:
        return 0.0
    leakages = []
    for rep in range(max(1, repeats)):
        drop = rng.choice(pool, size=k_drop, replace=False)
        keep = mask_from_drop_dims(z.shape[1], drop)
        z_rand = apply_diagonal_mask(z, keep)
        leakages.append(
            _positive_adv(
                crossfit_conditional_domain_probe(
                    z_rand,
                    y,
                    d,
                    n_classes=n_classes,
                    n_domains=n_domains,
                    groups=groups,
                    n_splits=n_splits,
                    probe=probe,
                    seed=seed + 17 + rep,
                )
            )
        )
    leak = float(np.mean(leakages))
    return float((base_leakage - leak) / max(base_leakage, 1e-8))


def run(args: argparse.Namespace) -> dict[str, object]:
    arrays = load_feature_npz(args.feature_npz)
    z = np.asarray(arrays["z"], dtype=np.float64)
    y = np.asarray(arrays["y"]).astype(np.int64, copy=False)
    d = np.asarray(arrays["d"]).astype(np.int64, copy=False)
    groups = arrays["groups"]
    if groups is not None:
        groups = np.asarray(groups)
    n_classes = args.n_classes or int(y.max()) + 1
    n_domains = args.n_domains or int(d.max()) + 1
    fractions = parse_drop_fractions(args.drop_fractions)
    thresholds = P0Thresholds(
        min_leakage_drop_frac=args.min_leakage_drop_frac,
        max_source_bacc_drop=args.max_source_bacc_drop,
        max_target_bacc_drop=args.max_target_bacc_drop,
        max_r3_delta=args.max_r3_delta,
        max_random_control_drop_frac=args.max_random_control_drop_frac,
        min_stability=args.min_stability,
    )

    base_leak_report = crossfit_conditional_domain_probe(
        z,
        y,
        d,
        n_classes=n_classes,
        n_domains=n_domains,
        groups=groups,
        n_splits=args.n_splits,
        probe=args.probe,
        seed=args.seed,
    )
    perm_report = crossfit_conditional_domain_probe(
        z,
        y,
        d,
        n_classes=n_classes,
        n_domains=n_domains,
        groups=groups,
        n_splits=args.n_splits,
        probe=args.probe,
        seed=args.seed,
        permutation=True,
    )
    base_leakage = _positive_adv(base_leak_report)
    base_source_bacc = crossfit_task_bacc(
        z,
        y,
        groups=groups,
        n_classes=n_classes,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    z_target = arrays["z_target"]
    y_target = arrays["y_target"]
    base_target_bacc = None
    if z_target is not None and y_target is not None:
        base_target_bacc = fit_source_eval_target_bacc(
            z,
            y,
            np.asarray(z_target, dtype=np.float64),
            np.asarray(y_target).astype(np.int64, copy=False),
            n_classes=n_classes,
            seed=args.seed,
        )

    scores = latent_dimension_scores(z, y, d)
    ranked = rank_latent_dimensions(z, y, d)
    candidates = []
    for cand_id, (name, drop_dims) in enumerate(candidate_drop_sets(ranked, fractions)):
        keep = mask_from_drop_dims(z.shape[1], drop_dims)
        z_masked = apply_diagonal_mask(z, keep)
        leak_report = crossfit_conditional_domain_probe(
            z_masked,
            y,
            d,
            n_classes=n_classes,
            n_domains=n_domains,
            groups=groups,
            n_splits=args.n_splits,
            probe=args.probe,
            seed=args.seed + 100 + cand_id,
        )
        source_bacc = crossfit_task_bacc(
            z_masked,
            y,
            groups=groups,
            n_classes=n_classes,
            n_splits=args.n_splits,
            seed=args.seed + 100 + cand_id,
        )
        target_bacc = None
        if z_target is not None and y_target is not None and base_target_bacc is not None:
            target_bacc = fit_source_eval_target_bacc(
                z_masked,
                y,
                apply_diagonal_mask(np.asarray(z_target, dtype=np.float64), keep),
                np.asarray(y_target).astype(np.int64, copy=False),
                n_classes=n_classes,
                seed=args.seed,
            )
        random_drop = _random_drop_control(
            z,
            y,
            d,
            groups,
            n_classes=n_classes,
            n_domains=n_domains,
            base_leakage=base_leakage,
            k_drop=len(drop_dims),
            exclude_dims=tuple(int(x) for x in drop_dims),
            repeats=args.random_control_repeats,
            seed=args.seed + 1000 + cand_id,
            probe=args.probe,
            n_splits=args.n_splits,
        )
        cand = SurgeryCandidate(
            name=name,
            dropped_units=tuple(int(x) for x in drop_dims),
            leakage_before=base_leakage,
            leakage_after=_positive_adv(leak_report),
            source_bacc_before=base_source_bacc,
            source_bacc_after=source_bacc,
            target_bacc_before=base_target_bacc,
            target_bacc_after=target_bacc,
            r3_before=0.0,
            r3_after=max(0.0, base_source_bacc - source_bacc),
            stability=max(0.0, 1.0 - float(leak_report["advantage_std"])),
            random_control_drop_frac=random_drop,
        )
        decision, reasons = decide_p0(cand, thresholds)
        candidates.append(
            {
                "candidate": cand.to_dict(),
                "decision": decision.value,
                "reasons": reasons,
                "target_eval_warnings": target_eval_warnings(cand, thresholds),
                "utility": score_candidate(cand),
                "leakage_report": leak_report,
            }
        )

    candidates = sorted(candidates, key=lambda r: float(r["utility"]), reverse=True)
    best_accept = next((c for c in candidates if c["decision"] == "ACCEPT"), None)
    dropped = set(best_accept["candidate"]["dropped_units"]) if best_accept else set()
    atlas = [r.to_dict() for r in latent_scores_to_atlas(scores, dropped=dropped)]
    result = {
        "project": "CEDAR-EEG",
        "phase": "P0_frozen_latent",
        "feature_npz": str(args.feature_npz),
        "probe": args.probe,
        "n_splits": args.n_splits,
        "groups_present": groups is not None,
        "thresholds": thresholds.__dict__,
        "baseline": {
            "leakage": base_leak_report,
            "permutation_null": perm_report,
            "source_bacc": base_source_bacc,
            "target_bacc_eval_only": base_target_bacc,
        },
        "candidates": candidates,
        "selected": best_accept,
        "atlas": atlas,
        "claim_boundary": (
            "Mask selection is source-side. Target metrics, when present, are evaluation-only. "
            "Leakage reduction is not a target-generalization guarantee."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe", default=DEFAULT_PROBE.probe, choices=("linear", "mlp"))
    ap.add_argument("--n-splits", type=int, default=DEFAULT_PROBE.n_splits)
    ap.add_argument("--seed", type=int, default=DEFAULT_PROBE.seed)
    ap.add_argument("--n-classes", type=int, default=0)
    ap.add_argument("--n-domains", type=int, default=0)
    ap.add_argument("--drop-fractions", default="0.05,0.10,0.20,0.30")
    ap.add_argument("--min-leakage-drop-frac", type=float, default=DEFAULT_P0_THRESHOLDS.min_leakage_drop_frac)
    ap.add_argument("--max-source-bacc-drop", type=float, default=DEFAULT_P0_THRESHOLDS.max_source_bacc_drop)
    ap.add_argument("--max-target-bacc-drop", type=float, default=DEFAULT_P0_THRESHOLDS.max_target_bacc_drop)
    ap.add_argument("--max-r3-delta", type=float, default=DEFAULT_P0_THRESHOLDS.max_r3_delta)
    ap.add_argument(
        "--max-random-control-drop-frac",
        type=float,
        default=DEFAULT_P0_THRESHOLDS.max_random_control_drop_frac,
    )
    ap.add_argument("--random-control-repeats", type=int, default=5)
    ap.add_argument("--min-stability", type=float, default=DEFAULT_P0_THRESHOLDS.min_stability)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
