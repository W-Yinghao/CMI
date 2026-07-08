#!/usr/bin/env python
"""S2P P1 D1 frontier summary. Consumes the D1 audit raw CSV (15 P1 cells + random, patch-norm) + the D0.5 released
reference (patch + window) + population diagnostics, and produces the PM-required outputs: L1 subject-structure
allocation frontier (PRIMARY), target transfer (pre-registered floor/null), task-gated L4/L5/L6, population frontier,
and p1_frontier_summary.json with the six carried caveats. No target labels in any selection (final scoring only)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

R = Path("results/s2p_p1_downstream")
D05 = R / "d0p5_decodability_sanity"
NGRID = [128, 256, 512, 1024, 2048]
TASK_GATE = 0.58

CAVEATS = [
    "SHU-MI is weak / low-ceiling (raw CSP within-subject 0.54 misses the strict 0.60 gate); decodability is genuine but small.",
    "Released-checkpoint sanity is above the random floor but with limited margin (~0.05); single seed, no own CI, wide Phase-8B band.",
    "Released reference reported under BOTH window (0.590) and patch (0.553) norm; window-norm lifts even a random encoder (+0.032).",
    "Target subjects 21-25 are near chance individually (CSP within mean 0.528); pooled transfer is a near-chance-but-significant margin.",
    "Released-checkpoint pretraining provenance is not certified here; upstream overlap with SHU-MI target subjects cannot be ruled out (probe firewall IS clean).",
    "D1 PRIMARY endpoint is the L1 subject-structure frontier; target-MI-transfer slope is a pre-registered floor/null, not estimable.",
]


def slope_stats(N, vals):
    """log2(N) linear slope + quadratic curvature + leave-one-N-out sign stability on per-N means."""
    x = np.log2(np.asarray(N, float)); y = np.asarray(vals, float)
    b1 = np.polyfit(x, y, 1)[0]
    curv = np.polyfit(x, y, 2)[0] if len(x) >= 3 else np.nan
    signs = []
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        signs.append(np.sign(np.polyfit(x[m], y[m], 1)[0]))
    return dict(slope_per_log2N=float(b1), curvature=float(curv),
                leave_one_N_out_sign_consistent=bool(len(set(signs)) == 1), per_N=list(map(float, y)))


def main():
    raw = pd.read_csv(R / "p1_task_and_frontier_raw.csv")
    p1 = raw[raw.tag.str.startswith("N")].copy()
    p1["N"] = p1.tag.str.extract(r"N(\d+)_").astype(int); p1["seed"] = p1.tag.str.extract(r"_s(\d+)").astype(int)
    rand = raw[raw.tag == "random"]
    L1 = "l1_l1_pairwise_bacc_mean"

    # ---- released reference (patch + window) from D0.5 ----
    def rel(norm):
        f = D05 / f"rel_{norm}" / "p1_task_and_frontier_raw.csv"
        if f.exists():
            d = pd.read_csv(f); r = d[d.tag == "released"]
            return dict(norm=norm, target_bacc=float(r.target_bacc.iloc[0]), source_val_bacc=float(r.source_val_bacc.iloc[0]),
                        l1=float(r[L1].iloc[0]), l4=float(r.l4_alignment.iloc[0]), l5_z=float(r.l5_l5_reliance_z.iloc[0]))
        return None
    released = [x for x in [rel("patch"), rel("window")] if x]

    # ---- per-N aggregation ----
    agg = p1.groupby("N").agg(l1_mean=(L1, "mean"), l1_sd=(L1, "std"),
                              target_mean=("target_bacc", "mean"), target_sd=("target_bacc", "std"),
                              srcval_mean=("source_val_bacc", "mean"), l4_mean=("l4_alignment", "mean"),
                              l5z_mean=("l5_l5_reliance_z", "mean")).reindex(NGRID).reset_index()
    randL1 = float(rand[L1].iloc[0]) if len(rand) else np.nan
    randT = float(rand.target_bacc.iloc[0]) if len(rand) else np.nan

    # ---- PRIMARY: L1 subject-structure frontier ----
    l1_front = slope_stats(NGRID, agg.l1_mean.values)
    tgt_front = slope_stats(NGRID, agg.target_mean.values)

    # ---- task gate ----
    p1["task_gate_pass"] = p1.source_val_bacc >= TASK_GATE
    gate_frac = float(p1.task_gate_pass.mean()); n_gate = int(p1.task_gate_pass.sum())

    # ---- write required CSVs ----
    p1.to_csv(R / "p1_task_performance.csv", index=False)
    p1[["tag", "N", "seed", L1, "l1_l1_pairwise_bacc_sd", "l1_n_pairs"]].to_csv(R / "p1_pairwise_subject_separability.csv", index=False)
    p1[["tag", "N", "seed", "l4_alignment", "subject_subspace_var_frac", "task_gate_pass"]].to_csv(R / "p1_l4_task_alignment.csv", index=False)
    p1[["tag", "N", "seed", "l5_l5_reliance_z", "l5_subject_removal_drop", "l5_null_drop_mean", "l5_l5_beats_null", "task_gate_pass"]].to_csv(R / "p1_l5_replay.csv", index=False)
    p1[["tag", "N", "seed", "target_bacc", "target_macro_f1", "target_nll", "task_gate_pass"]].to_csv(R / "p1_l6_target_consequence.csv", index=False)
    rand.to_csv(R / "p1_random_init_control.csv", index=False)
    pd.DataFrame(released).to_csv(R / "p1_released_checkpoint_reference.csv", index=False)
    # population frontier (join protocol diagnostics)
    popf = Path("results/s2p_p1_protocol/p1_population_diagnostics.csv")
    if popf.exists():
        pop = pd.read_csv(popf); pop.merge(agg[["N", "l1_mean", "target_mean"]], on="N", how="left").to_csv(R / "p1_population_frontier_diagnostics.csv", index=False)
    pd.DataFrame([dict(cell="P1", norm="patch"), dict(cell="released", norm="patch+window"),
                  dict(cell="random", norm="patch")]).to_csv(R / "p1_downstream_norm_manifest.csv", index=False)
    pd.DataFrame(dict(tag=raw.tag, checkpoint=raw.checkpoint)).to_csv(R / "p1_downstream_run_manifest.csv", index=False)

    summary = dict(
        d1_allowed=True, gate_substitution_accepted=True,
        primary_endpoint="l1_subject_structure_frontier",
        target_transfer_endpoint="pre_registered_floor_null",
        primary_norm="p1_matched_patch_norm",
        released_reference_reported_with_window_and_patch_norm=True,
        task_gate_for_l4_l5_l6=TASK_GATE,
        target_labels_used_for_selection=False,
        # ---- PRIMARY result ----
        l1_frontier=l1_front, l1_random_floor=randL1,
        l1_pretrained_above_random=bool(agg.l1_mean.min() > randL1),
        # ---- transfer null ----
        target_transfer_frontier=tgt_front, target_random_floor=randT,
        target_transfer_verdict="pre-registered FLOOR/NULL: P1 target bAcc ~= random floor, not estimable as a slope",
        # ---- task gate ----
        task_gate_pass_fraction=gate_frac, n_cells_gate_pass=n_gate, n_cells=int(len(p1)),
        l4_l5_l6_interpretable=bool(gate_frac >= 0.5),
        l4_l5_l6_status=("interpretable" if gate_frac >= 0.5 else "WEAK_TASK_DIAGNOSTIC_ONLY (most cells fail source-val>=0.58 gate)"),
        # ---- released reference ----
        released_reference=released,
        # ---- population ----
        population_confound="deep endpoint N=128 clinical (pool 201) -> N=2048 general (6388); disclosed, carried as covariate",
        required_caveats=CAVEATS,
        p2_recommended=None,   # set after reading the frontier
    )
    json.dump(summary, open(R / "p1_frontier_summary.json", "w"), indent=2)
    # console
    print("=== L1 subject-structure frontier (PRIMARY) ===")
    print(agg[["N", "l1_mean", "l1_sd", "srcval_mean", "target_mean", "l4_mean", "l5z_mean"]].to_string(index=False))
    print(f"\nL1 slope/log2N = {l1_front['slope_per_log2N']:+.4f}  curvature={l1_front['curvature']:+.4f}  LOO-sign-stable={l1_front['leave_one_N_out_sign_consistent']}")
    print(f"L1 random floor = {randL1:.3f}; pretrained above random = {agg.l1_mean.min() > randL1}")
    print(f"target transfer slope/log2N = {tgt_front['slope_per_log2N']:+.4f} (per-N {[round(v,3) for v in tgt_front['per_N']]}); random floor {randT:.3f}")
    print(f"task-gate (src-val>=0.58) pass: {n_gate}/{len(p1)} = {gate_frac:.2f} -> L4/L5/L6 {'interpretable' if gate_frac>=0.5 else 'WEAK_TASK_DIAGNOSTIC_ONLY'}")
    print("released ref:", released)


if __name__ == "__main__":
    main()
