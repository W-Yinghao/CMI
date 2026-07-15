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

# deferred — registry placeholders only; NOT run as objective rows (activated in a later round if needed)
DEFERRED_METHODS = ("mmd", "label_mmd", "groupdro", "nodedat", "eeg_dg")

# ---------------------------------------------------------------------------------------------------------
# CMI-Trace P0.1 objective comparison (the main-table method set). Closes the promise–evidence gap: the
# manuscript names CORAL and IRM/V-REx as core invariance families but the old table only had ERM /
# encoder-CMI / DANN / cond-DANN / CDAN. This set runs ALL of them on the SAME DGCNN adapter under the SAME
# source-only LOSO protocol. The already-completed adversarial rows (cond_dann/cdan) are KEPT — not rerun —
# unless a protocol/config hash differs.
#
# Fixed-anchor rows use the pre-registered lambda 0.010 on the (graph, node) objects for the moment/CMI
# families; IRM/V-REx use the DomainBed-canonical unit penalty weight. `select: True` rows are chosen by
# SOURCE-ONLY nested leave-one-source-domain validation over a frozen grid (configs/cmi_trace_p0p1.yaml) —
# no target labels ever. The fixed 0.010 encoder-CMI row is the pre-registered anchor; cigl_nested is the
# secondary nested-selected sensitivity row required by P0.1.
OBJECTIVE_METHODS = {
    "erm":                {"config": "erm:0",                                     "method": "erm",         "family": "baseline",                "select": False},
    "cigl_graph_node":    {"config": f"graphcmi:{LAMBDA:.3f}:{LAMBDA:.3f}:0.000", "method": "graphcmi",    "family": "encoder_cmi",             "select": False},
    "cigl_nested":        {"config": "graphcmi:SELECT",                           "method": "graphcmi",    "family": "encoder_cmi",             "select": True},
    "coral":              {"config": f"coral:{LAMBDA:.3f}:{LAMBDA:.3f}",          "method": "coral",       "family": "moment_marginal",         "select": False},
    "label_coral":        {"config": f"label_coral:{LAMBDA:.3f}:{LAMBDA:.3f}",    "method": "label_coral", "family": "moment_conditional",      "select": False},
    "irm":                {"config": "irm:1.000",                                 "method": "irm",         "family": "risk_invariance",         "select": False},
    "vrex":               {"config": "vrex:1.000",                                "method": "vrex",        "family": "risk_invariance",         "select": False},
    "cond_dann":          {"config": "cdann:1.000",                               "method": "cdann",       "family": "adversarial_conditional", "select": False},
    "cdan":               {"config": "cdan:1.000",                                "method": "cdan",        "family": "adversarial_multilinear", "select": False},
    # nested source-domain-selected sensitivity rows (grid frozen in configs/cmi_trace_p0p1.yaml)
    "coral_nested":       {"config": "coral:SELECT",                              "method": "coral",       "family": "moment_marginal",         "select": True},
    "label_coral_nested": {"config": "label_coral:SELECT",                        "method": "label_coral", "family": "moment_conditional",      "select": True},
    "irm_nested":         {"config": "irm:SELECT",                                "method": "irm",         "family": "risk_invariance",         "select": True},
    "vrex_nested":        {"config": "vrex:SELECT",                               "method": "vrex",        "family": "risk_invariance",         "select": True},
}

# the primary (non-sensitivity) main-table rows required by P0.1
OBJECTIVE_PRIMARY = ("erm", "cigl_graph_node", "cigl_nested", "coral", "label_coral", "irm", "vrex", "cond_dann")


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


def validate_objective_registry():
    """Every OBJECTIVE_METHODS row resolves to a known trainer method; fixed-config rows are well-formed;
    no objective method is also deferred. Returns the list of unknown method labels (empty = valid)."""
    from cmi.train.trainer import ALL_METHODS
    unknown = []
    for label, spec in OBJECTIVE_METHODS.items():
        m = spec["config"].split(":")[0]
        if m != spec["method"] or m not in ALL_METHODS:
            unknown.append(label)
        # fixed-config (non-select) rows must carry a parseable numeric config (except erm:0)
        if not spec["select"] and label != "erm":
            tail = spec["config"].split(":")[1:]
            if not tail or any(t == "SELECT" for t in tail):
                unknown.append(label)
    assert not ({s["method"] for s in OBJECTIVE_METHODS.values()} & set(DEFERRED_METHODS)), \
        "an objective method is also marked deferred"
    assert set(OBJECTIVE_PRIMARY) <= set(OBJECTIVE_METHODS), "OBJECTIVE_PRIMARY references an unknown row"
    return unknown
