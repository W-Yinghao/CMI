"""CIGL R2a — baseline registry + same-backbone contract. Defines the R2a method set that must run on the
SAME task-capable static-adjacency DGCNN adapter under the SAME source-only LOSO protocol, so the
leakage-vs-task Pareto comparison is apples-to-apples (not a leaderboard across architectures). PM R2a scope:
ERM, CIGL graph-only / node-only / graph+node, DANN, conditional-DANN, CDAN. The rest are registry
placeholders, deferred (do not front-load the whole baseline zoo).
"""
from __future__ import annotations

BACKBONE = "dgcnn_forward_graph_adapter"   # static-adjacency, task-capable (phase-3A); graph_z + node_z audit
LAMBDA = 0.010                             # the pre-registered fixed CIGL strength (graph_node_010)

# every R2a run MUST share these — enforced by the contract test
SAME_BACKBONE_CONTRACT = {
    "backbone": BACKBONE,
    "builder": "cmi.models.graph_task_backbones.build_graph_task_backbone",
    "protocol": "loso",
    "source_only": True,
    "source_val_early_stop": True,
    "target_firewall": "eval_only",         # target labels never in training/selection/probe-fit
    "audit_objects": ("graph_z", "node_z"),
    "null": "within_label_permutation",     # domain permuted WITHIN label
}

# active R2a methods: label -> {config string, trainer method, family}
R2A_METHODS = {
    "erm":             {"config": "erm:0",                             "method": "erm",      "family": "baseline"},
    "cigl_graph":      {"config": f"graphcmi:{LAMBDA:.3f}:0.000:0.000", "method": "graphcmi", "family": "cigl"},
    "cigl_node":       {"config": f"graphcmi:0.000:{LAMBDA:.3f}:0.000", "method": "graphcmi", "family": "cigl"},
    "cigl_graph_node": {"config": f"graphcmi:{LAMBDA:.3f}:{LAMBDA:.3f}:0.000", "method": "graphcmi", "family": "cigl"},
    "dann":            {"config": "dann:1.0",  "method": "dann",  "family": "adversarial_marginal"},
    "cond_dann":       {"config": "cdann:1.0", "method": "cdann", "family": "adversarial_conditional"},
    "cdan":            {"config": "cdan:1.0",  "method": "cdan",  "family": "adversarial_multilinear"},
}

# deferred — registry placeholders only; NOT run in R2a (added in a later R2 round, PM gate)
DEFERRED_METHODS = ("coral", "label_coral", "mmd", "label_mmd", "irm", "vrex", "groupdro", "nodedat", "eeg_dg")


def build_contract_backbone(n_chans, n_times, n_classes):
    """Build the ONE backbone every R2a method shares (the same-backbone contract)."""
    from cmi.models.graph_task_backbones import build_graph_task_backbone
    return build_graph_task_backbone(BACKBONE, n_chans, n_times, n_classes)


def validate_registry():
    """Every active method resolves to a known trainer method; configs are well-formed; no active method is
    also deferred. Returns the list of unknown methods (empty = valid)."""
    from cmi.train.trainer import ALL_METHODS
    unknown = []
    for label, spec in R2A_METHODS.items():
        m = spec["config"].split(":")[0]
        if m != spec["method"] or m not in ALL_METHODS:
            unknown.append(label)
    assert not (set(R2A_METHODS) & set(DEFERRED_METHODS)), "an active method is also marked deferred"
    return unknown
