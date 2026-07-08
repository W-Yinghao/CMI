#!/usr/bin/env python
"""S2P Phase 9D high-budget feasibility relaxation audit.

CPU/metadata-only. This script does not train, does not submit SLURM jobs, and
does not inspect downstream labels. It audits which protocol relaxation could
make a >=2000 h high-budget calibration feasible after the strict 9C v2
19-common exact high-coverage NO-GO.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

S2P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S2P / "s2p" / "scripts"))
import tueg_subject_loader as L


TUEG = "/projects/EEG-foundation-model/datalake/processed/4704743c/TUEG"
AUDIT_BUDGETS = [500, 1000, 2000, 3000, 4000]
MIN_EXPOSURE_H = 0.25
MIN_W = L.cap_windows_for(MIN_EXPOSURE_H)
N_VAL = 128
VAL_CAP_W = 24
BASE_200H_WALL_H = 4.5
WALLTIME_LIMIT_H = 96.0
WALLTIME_SAFETY = 1.15


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def dump_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def gini(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    return float((2.0 * np.sum((np.arange(1, n + 1) * arr)) / (n * np.sum(arr))) - ((n + 1.0) / n))


def channel_set(chstr):
    try:
        return set(json.loads(chstr))
    except Exception:
        return set()


def load_subject_windows(corpus):
    meta = pd.read_parquet(f"{TUEG}/metadata.parquet")
    if corpus == "19common":
        want = set(L.COMMON19)
        meta = meta[meta["channels"].map(lambda s: want <= channel_set(s))].copy()
    elif corpus == "full_processed":
        meta = meta.copy()
    elif corpus == "exact33":
        meta = meta[meta["n_channels"] == 33].copy()
    else:
        raise ValueError(corpus)
    meta["avail_w"] = (meta["n_timepoints"] // L.WLEN).astype(int)
    meta = meta[meta["avail_w"] > 0].copy()
    return meta.groupby("subject")["avail_w"].sum().sort_index()


def fixed_val_subjects(sw):
    cand = np.sort(sw[sw >= VAL_CAP_W].index.to_numpy())
    if len(cand) < N_VAL:
        raise RuntimeError(f"not enough val subjects: {len(cand)} < {N_VAL}")
    rng = np.random.default_rng(920000)
    return np.sort(rng.choice(cand, N_VAL, replace=False))


def train_sw_after_val(sw):
    val = fixed_val_subjects(sw)
    return sw.drop(labels=[s for s in val if s in sw.index], errors="ignore"), val


def estimate_wall_h(hours, channel_multiplier=1.0):
    return BASE_200H_WALL_H * (float(hours) / 200.0) * WALLTIME_SAFETY * channel_multiplier


def exact_highcoverage_plan(sw, hours):
    wt = int(round(float(hours) * 120))
    train_sw, val = train_sw_after_val(sw)
    eligible_min = train_sw[train_sw >= MIN_W]
    upper_n = min(int(len(eligible_min)), int(wt // MIN_W))
    for n_subjects in range(upper_n, 0, -1):
        base = wt // n_subjects
        rem = wt - base * n_subjects
        need_w = base + (1 if rem else 0)
        pool = np.sort(train_sw[train_sw >= need_w].index.to_numpy())
        if len(pool) >= n_subjects:
            alloc = np.full(n_subjects, base, dtype=int)
            if rem:
                alloc[:rem] += 1
            return {
                "window_feasible": True,
                "n_subjects": int(n_subjects),
                "train_windows": wt,
                "base_windows": int(base),
                "plus1_subjects": int(rem),
                "need_w": int(need_w),
                "pool_size": int(len(pool)),
                "val_subjects": int(len(val)),
                "alloc": alloc,
                "reason": "ok",
            }
    return {
        "window_feasible": False,
        "n_subjects": None,
        "train_windows": wt,
        "reason": "no N satisfies exact high-coverage allocation",
    }


def waterfill_integer(lower, caps, total):
    alloc = np.full(len(caps), int(lower), dtype=int)
    caps = np.asarray(caps, dtype=int)
    rem = int(total - alloc.sum())
    while rem > 0:
        eligible = np.flatnonzero(alloc < caps)
        if eligible.size == 0:
            return None
        q = max(1, rem // int(eligible.size))
        inc = np.minimum(q, caps[eligible] - alloc[eligible])
        if int(inc.sum()) > rem:
            order = eligible[np.argsort(alloc[eligible])]
            for idx in order:
                if rem <= 0:
                    break
                if alloc[idx] < caps[idx]:
                    alloc[idx] += 1
                    rem -= 1
        else:
            alloc[eligible] += inc
            rem -= int(inc.sum())
    return alloc


def bounded_imbalance_plan(sw, hours, gini_cap, ratio_cap=2.0):
    wt = int(round(float(hours) * 120))
    train_sw, val = train_sw_after_val(sw)
    eligible = train_sw[train_sw >= MIN_W].sort_values(ascending=False)
    upper_n = min(int(len(eligible)), int(wt // MIN_W))
    if upper_n <= 0:
        return {
            "window_feasible": False,
            "n_subjects": None,
            "train_windows": wt,
            "reason": "no eligible subjects at min exposure",
        }

    # Route A keeps the high-coverage contract: choose the maximum subject
    # coverage implied by the min-exposure rule, then ask whether a bounded
    # imbalance can absorb the extra windows. It does not silently reduce N to
    # rescue feasibility, because that would change the claim again.
    n_subjects = upper_n
    lower = MIN_W
    pool = eligible.head(n_subjects)
    caps = np.minimum(pool.to_numpy(dtype=int), int(math.floor(ratio_cap * lower)))
    if int(n_subjects * lower) <= wt <= int(caps.sum()):
        alloc = waterfill_integer(lower, caps, wt)
        if alloc is not None:
            alloc_gini = gini(alloc)
            alloc_ratio = float(alloc.max() / max(1, alloc.min()))
            if alloc_gini <= gini_cap and alloc_ratio <= ratio_cap + 1e-12:
                return {
                    "window_feasible": True,
                    "n_subjects": int(n_subjects),
                    "train_windows": wt,
                    "min_windows": int(alloc.min()),
                    "median_windows": float(np.median(alloc)),
                    "max_windows": int(alloc.max()),
                    "max_min_ratio": round(alloc_ratio, 6),
                    "contribution_gini": round(float(alloc_gini), 6),
                    "pool_size": int(len(pool)),
                    "val_subjects": int(len(val)),
                    "alloc": alloc,
                    "reason": "ok",
                }
    return {
        "window_feasible": False,
        "n_subjects": None,
        "train_windows": wt,
        "reason": (
            f"max-coverage N={upper_n} cannot satisfy bounded imbalance "
            f"with min_w={MIN_W}, gini<={gini_cap}, max/min<={ratio_cap}"
        ),
    }


def data_volume_plan(sw, hours):
    wt = int(round(float(hours) * 120))
    train_sw, val = train_sw_after_val(sw)
    total_available = int(train_sw.sum())
    feasible = total_available >= wt
    expected = train_sw.to_numpy(dtype=float) * (wt / total_available) if total_available else np.asarray([])
    return {
        "window_feasible": bool(feasible),
        "n_subjects": int((train_sw > 0).sum()) if feasible else None,
        "train_windows": wt,
        "total_available_train_windows": total_available,
        "covered_window_fraction": round(float(wt / total_available), 6) if feasible and total_available else None,
        "availability_gini": round(float(gini(train_sw.to_numpy(dtype=float))), 6) if total_available else None,
        "expected_contribution_gini_if_uniform_window_sample": round(float(gini(expected)), 6) if feasible else None,
        "val_subjects": int(len(val)),
        "reason": "ok" if feasible else "not enough 19-common windows after fixed val exclusion",
    }


def diag_row(route, budget_h, corpus, plan, notes):
    alloc = plan.get("alloc")
    if alloc is not None:
        contrib = np.asarray(alloc, dtype=float)
        contribution_gini = round(float(gini(contrib)), 6)
        cmin = float(contrib.min())
        cmed = float(np.median(contrib))
        cp90 = float(np.percentile(contrib, 90))
        cmax = float(contrib.max())
    else:
        contribution_gini = plan.get("expected_contribution_gini_if_uniform_window_sample")
        cmin = cmed = cp90 = cmax = None
    return {
        "route": route,
        "budget_h": budget_h,
        "corpus": corpus,
        "window_feasible": plan.get("window_feasible"),
        "n_subjects": plan.get("n_subjects"),
        "train_windows": plan.get("train_windows"),
        "total_available_train_windows": plan.get("total_available_train_windows"),
        "covered_window_fraction": plan.get("covered_window_fraction"),
        "contribution_gini": contribution_gini,
        "contribution_min_windows": cmin,
        "contribution_median_windows": cmed,
        "contribution_p90_windows": cp90,
        "contribution_max_windows": cmax,
        "availability_gini": plan.get("availability_gini"),
        "notes": notes,
    }


def feasibility_row(route, constraint_relaxed, corpus, allocation, budget_h, plan, channel_multiplier=1.0):
    est = estimate_wall_h(budget_h, channel_multiplier=channel_multiplier)
    return {
        "route": route,
        "constraint_relaxed": constraint_relaxed,
        "corpus": corpus,
        "allocation_contract": allocation,
        "budget_h": budget_h,
        "window_feasible": bool(plan.get("window_feasible")),
        "compute_budget_acceptable": bool(est <= WALLTIME_LIMIT_H),
        "overall_metadata_feasible": bool(plan.get("window_feasible") and est <= WALLTIME_LIMIT_H),
        "n_subjects": plan.get("n_subjects"),
        "train_windows": plan.get("train_windows"),
        "estimated_wall_h": round(est, 2),
        "planned_time_h": WALLTIME_LIMIT_H,
        "reason": plan.get("reason"),
    }


def budgets_feasible(rows, route):
    vals = [int(r["budget_h"]) for r in rows if r["route"] == route and r["overall_metadata_feasible"]]
    return ";".join(str(v) for v in vals) if vals else "none"


def route_feasible_ge2000(rows, route):
    return any(r["route"] == route and r["overall_metadata_feasible"] and int(r["budget_h"]) >= 2000 for r in rows)


def route_id_for_gini(gcap):
    return f"A_gini_{gcap:.2f}".replace(".", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/s2p_budget_floor_calibration_v2/relaxation_audit")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sw19 = load_subject_windows("19common")
    sw33 = load_subject_windows("exact33")

    rows = []
    diag = []

    for gcap in [0.05, 0.10]:
        route = route_id_for_gini(gcap)
        for h in AUDIT_BUDGETS:
            plan = bounded_imbalance_plan(sw19, h, gini_cap=gcap, ratio_cap=2.0)
            rows.append(feasibility_row(
                route=route,
                constraint_relaxed=f"exact_equal_windows_to_bounded_imbalance_gini<={gcap}_maxmin<=2",
                corpus="TUEG_19_common",
                allocation="high_coverage_min0p25h_bounded_imbalance_no_reuse",
                budget_h=h,
                plan=plan,
            ))
            diag.append(diag_row(route, h, "TUEG_19_common", plan, "Route A bounded-imbalance allocation diagnostic."))

    for h in AUDIT_BUDGETS:
        plan = data_volume_plan(sw33, h)
        rows.append(feasibility_row(
            route="B",
            constraint_relaxed="19common_canonical_subset_to_fixed_33ch_cbramod_only_substrate",
            corpus="TUEG_processed_exact_33ch",
            allocation="data_volume_no_reuse_subject_coverage_not_controlled",
            budget_h=h,
            plan=plan,
            channel_multiplier=33.0 / 19.0,
        ))
        diag.append(diag_row("B", h, "TUEG_processed_exact_33ch", plan,
                             "Route B uses fixed 33-channel CBraMod-only substrate; it is not a 19-common canonical curve."))

    for h in AUDIT_BUDGETS:
        plan = data_volume_plan(sw19, h)
        rows.append(feasibility_row(
            route="C",
            constraint_relaxed="high_coverage_subject_allocation_to_total_data_volume_sampling",
            corpus="TUEG_19_common",
            allocation="data_volume_no_reuse_subject_coverage_not_controlled",
            budget_h=h,
            plan=plan,
        ))
        diag.append(diag_row("C", h, "TUEG_19_common", plan,
                             "Route C feasibility only; an actual sampler must be specified before training."))

    route_a_005 = route_id_for_gini(0.05)
    route_a_010 = route_id_for_gini(0.10)
    route_a_feasible = route_feasible_ge2000(rows, route_a_005) or route_feasible_ge2000(rows, route_a_010)
    route_b_feasible = route_feasible_ge2000(rows, "B")
    route_c_feasible = route_feasible_ge2000(rows, "C")

    smoke_path = out / "route_b_33ch_smoke.json"
    route_b_smoke_passed = None
    if smoke_path.exists():
        try:
            route_b_smoke_passed = bool(json.loads(smoke_path.read_text()).get("smoke_passed"))
        except Exception:
            route_b_smoke_passed = False

    if route_b_feasible and route_b_smoke_passed is not False:
        recommendation = "B"
    elif route_c_feasible:
        recommendation = "C"
    elif route_a_feasible:
        recommendation = "A"
    else:
        recommendation = "none"

    claim_rows = [
        {
            "route": "A",
            "constraint_relaxed": "exact per-subject equal-window allocation",
            "budgets_feasible": f"gini<=0.05:{budgets_feasible(rows, route_a_005)} | gini<=0.10:{budgets_feasible(rows, route_a_010)}",
            "model_scope": "CBraMod on TUEG 19-common",
            "claim_allowed": "high-budget calibration under bounded-imbalance 19-common high-coverage allocation",
            "claim_forbidden": "exact high-coverage calibration; pure subject-diversity/depth decomposition",
            "recommended": recommendation == "A",
        },
        {
            "route": "B",
            "constraint_relaxed": "19-common canonical channel constraint",
            "budgets_feasible": budgets_feasible(rows, "B"),
            "model_scope": "CBraMod-only fixed 33-channel processed substrate",
            "claim_allowed": "CBraMod 33-channel full-corpus budget calibration",
            "claim_forbidden": "direct CodeBrain 19-common scaling comparison; patching the 19-common primary curve",
            "recommended": recommendation == "B",
        },
        {
            "route": "C",
            "constraint_relaxed": "subject high-coverage allocation control",
            "budgets_feasible": budgets_feasible(rows, "C"),
            "model_scope": "CBraMod on TUEG 19-common",
            "claim_allowed": "19-common total data-volume scaling test of whether 200h was too short",
            "claim_forbidden": "subject-coverage causal claim; exact high-coverage allocation claim",
            "recommended": recommendation == "C",
        },
    ]

    strict_go_path = out.parent / "budget_floor_v2_go_nogo.json"
    strict_ge2000 = False
    if strict_go_path.exists():
        strict = json.loads(strict_go_path.read_text())
        strict_ge2000 = bool(strict.get("GO") and strict.get("h_high_selected_h") and strict["h_high_selected_h"] >= 2000)

    rec = {
        "phase": "9D_high_budget_feasibility_relaxation_audit",
        "strict_19common_exact_highcoverage_feasible_ge2000": strict_ge2000,
        "route_A_bounded_imbalance_feasible": route_a_feasible,
        "route_B_33ch_full_corpus_feasible": route_b_feasible,
        "route_B_33ch_smoke_passed": route_b_smoke_passed,
        "route_C_19common_data_volume_feasible": route_c_feasible,
        "recommended_next_training_design": recommendation,
        "recommendation_basis": "PM_updated_preference_B_then_C_then_A; Route_B_training_requires_33ch_smoke_and_PM_approval",
        "training_requires_pm_approval": True,
        "target_labels_used": False,
        "training_launched": False,
    }

    write_csv(out / "relaxation_route_feasibility.csv", rows)
    write_csv(out / "relaxation_claim_matrix.csv", claim_rows)
    write_csv(out / "relaxation_population_shift_diagnostics.csv", diag)
    dump_json(out / "relaxation_recommendation.json", rec)

    print(json.dumps(rec, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
