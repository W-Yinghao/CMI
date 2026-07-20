"""C42 deterministic taxonomy."""
from __future__ import annotations

from . import artifact_loader as al
from . import schema


def classify(ctx, gap, top1, stability, gauge, conflict):
    sr = top1["summary"]["C30_source_rank_score"]
    oaci = top1["summary"]["actual_oaci_selector"]
    random_joint = top1["summary"]["random_trajectory_conditioned"]["top1_joint_good_rate"]
    src_gap = gap["summary"][("C30_source_rank_score", "top1", "primary_joint_good")]
    leakage_gap = gap["summary"][("selection_leakage_point", "top1", "primary_joint_good")]
    source_auc = al.as_float(src_gap["mean_pairwise_auc_vs_target_utility"])
    leakage_auc = al.as_float(leakage_gap["mean_pairwise_auc_vs_target_utility"])
    top1_gain_oaci = al.as_float(sr["top1_joint_good_rate"]) - al.as_float(oaci["top1_joint_good_rate"])
    top1_gain_random = al.as_float(sr["top1_joint_good_rate"]) - al.as_float(random_joint)
    reliable = (
        al.as_float(sr["top1_joint_good_rate"]) >= schema.TOP1_RELIABLE_JOINT_GOOD_GATE and
        al.as_float(src_gap["mean_enrichment_ratio"]) >= schema.TOP1_RELIABLE_ENRICHMENT_GATE)
    established = {
        schema.R1: source_auc >= schema.SOURCE_RANK_PAIRWISE_SIGNAL_GATE and source_auc > leakage_auc,
        schema.R2: source_auc >= schema.SOURCE_RANK_PAIRWISE_SIGNAL_GATE and not reliable,
        schema.R3: top1_gain_oaci > 0 and not reliable,
        schema.R4: reliable,
        schema.R5: bool(gauge["summary"]["gauge_breaks_source_rank_actionability"]),
        schema.R6: random_joint >= 0.40 and top1_gain_random < schema.TOP1_MODEST_GAIN_GATE,
        schema.R7: bool(stability["summary"]["top_region_plateau_or_instability"]),
        schema.R8: bool(conflict["summary"]["leakage_blocks_rank_better_candidates"]),
        schema.R9: not reliable,
        schema.R10: reliable,
    }
    evidence = {
        schema.R1: f"source_rank_auc={source_auc}, selection_leakage_auc={leakage_auc}",
        schema.R2: (
            f"source_rank_top1_joint={sr['top1_joint_good_rate']}, "
            f"enrichment={src_gap['mean_enrichment_ratio']}, reliable_gate={schema.TOP1_RELIABLE_JOINT_GOOD_GATE}"),
        schema.R3: f"top1_joint_gain_vs_oaci={top1_gain_oaci}, top1_joint_gain_vs_random={top1_gain_random}",
        schema.R4: f"top1_joint={sr['top1_joint_good_rate']}, enrichment={src_gap['mean_enrichment_ratio']}",
        schema.R5: f"max_centered_gain={gauge['summary']['max_centered_top1_joint_good_gain_vs_raw']}",
        schema.R6: f"random_base={random_joint}, source_gain_vs_random={top1_gain_random}",
        schema.R7: (
            f"mean_plateau_size={stability['summary']['mean_plateau_size']}, "
            f"low_margin_fraction={stability['summary']['low_margin_fraction']}"),
        schema.R8: (
            f"leakage_blocks_fraction={conflict['summary']['leakage_blocks_rank_better_fraction']}, "
            f"rank_better_fraction={conflict['summary']['rank_top_target_better_than_oaci_fraction']}"),
        schema.R9: "source-rank top1/top-k remains below reliability gates; diagnostic only",
        schema.R10: "only active if R4 passes; it does not under current gates",
    }
    rows = [{"case": c, "established": int(bool(established[c])), "evidence": evidence[c]}
            for c in schema.ALL_CASES]
    return {"cases": [c for c in schema.ALL_CASES if established[c]], "case_rows": rows,
            "established": established, "evidence": evidence}
