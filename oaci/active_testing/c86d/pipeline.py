"""C86D selection freeze, held evaluator, and gated execution entrypoint.

Selection is frozen BEFORE any held-evaluation access. The held evaluator opens the
C85U utility field only after receiving a finalized freeze, and it holds no server
handle or oracle. Real execution is gated on a direct '授权 C86D'.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from .core import CONTEXTS_PER_TRIAL, verify_c85u_identity
from .policies import acquisition_path, budget_prefix, composite_select

_CONTRIB_FIELDS = ("nll", "correct", "confidence", "conf_bin", "signed_calibration")
_PLUGIN_FIELDS = ("nll", "correct", "confidence", "conf_bin")


class C86DNotAuthorized(RuntimeError):
    pass


class C86DOrderingError(RuntimeError):
    pass


AUTHORIZATION_PHRASE = "授权 C86D"


@dataclass
class SelectionFreeze:
    target: tuple
    method: str
    budget: object
    seed: int
    query_sequence: list
    lure_weights: list
    receipts: list                      # per-query (trial, label) receipts
    selected_by_context: dict           # context -> selected candidate index
    per_context_estimates: dict         # context -> composite vector
    frozen: bool = False

    def lock(self):
        self.frozen = True
        return self


def run_selection(client, target_pool, target, method, budget, seed) -> SelectionFreeze:
    """Acquire labels via the sealed server, estimate, and FREEZE the selection.

    Uses the nested-prefix acquisition path (uniform warm start + budget-specific
    LURE weights). FULL uses the exact (unsmoothed) construction plugin.
    """
    order, q_seq = acquisition_path(target_pool, method, seed)
    pre, weights = budget_prefix(order, q_seq, len(order), budget)
    full = (budget == "FULL")
    attempt = client.open_attempt(target, "FULL")
    per_ctx, receipts = {}, []
    for trial in pre:
        label, contexts = client.query(attempt, trial)
        receipts.append((trial, int(label)))
        for ctx, row in contexts.items():
            d = per_ctx.setdefault(ctx, {"labels": [], **{f: [] for f in _PLUGIN_FIELDS}})
            d["labels"].append(int(label))
            for f in _PLUGIN_FIELDS:
                d[f].append(np.asarray(row[f]))
    selected, ests = {}, {}
    for ctx, d in per_ctx.items():
        contribs = {f: np.array(d[f]) for f in _PLUGIN_FIELDS}
        sel, metrics = composite_select(d["labels"], contribs, weights, full=full)
        selected[ctx] = sel
        ests[ctx] = metrics["composite"]
    fr = SelectionFreeze(target=tuple(target), method=method, budget=budget, seed=seed,
                         query_sequence=list(pre), lure_weights=list(np.asarray(weights)),
                         receipts=receipts, selected_by_context=selected,
                         per_context_estimates=ests)
    fr.lock()
    return fr


class HeldEvaluator:
    """DIAGNOSTIC-ONLY raw-utility-gap evaluator for shadow tests.

    RETIRED as the C86D primary path: the primary development risk is the held
    STANDARDIZED regret computed in ``run_d2`` (which also enforces freeze-verify-
    before-C85U-open and holds no server handle). This class only computes the raw
    composite-utility gap for the epsilon geometry / shadow ordering tests, and must
    not be used as the primary risk.
    """

    def __init__(self, held_by_ctx: dict, verify_identity: bool = False):
        self._held = held_by_ctx
        self.identity = verify_c85u_identity() if verify_identity else {"verified": False}

    def evaluate(self, freeze: SelectionFreeze) -> dict:
        if not freeze.frozen:
            raise C86DOrderingError("selection must be frozen before held evaluation")
        gap = {}
        for ctx, sel in freeze.selected_by_context.items():
            util = np.asarray(self._held[(freeze.target, ctx)])
            gap[ctx] = float(util.max() - util[sel])
        return {"target": freeze.target, "method": freeze.method, "budget": freeze.budget,
                "context_raw_gap": gap, "target_raw_gap_diagnostic": float(np.mean(list(gap.values()))),
                "n_contexts": len(gap), "primary_risk": "see run_d2 standardized regret"}


def execute(authorization: str | None = None, output_root: str = "", **kwargs):
    """Real C86D active-policy execution — REFUSES without a direct '授权 C86D'."""
    if authorization != AUTHORIZATION_PHRASE:
        raise C86DNotAuthorized(
            "C86D real execution requires a separate direct '授权 C86D'; the protocol/"
            "client-server implementation does not authorize it"
        )
    if not output_root:
        raise ValueError("authorized C86D execution requires an output_root")
    # D1 (selection, no C85U) and D2 (held evaluation, C85U) run as SEPARATE processes.
    import subprocess
    import sys
    d1_root = output_root + "_d1"
    subprocess.run([sys.executable, "-m", "oaci.active_testing.c86d.run_d1",
                    "--output-root", d1_root, "--authorization", AUTHORIZATION_PHRASE], check=True)
    subprocess.run([sys.executable, "-m", "oaci.active_testing.c86d.run_d2",
                    "--d1-root", d1_root, "--output-root", output_root,
                    "--authorization", AUTHORIZATION_PHRASE], check=True)
    return {"d1_root": d1_root, "output_root": output_root, "stages": ["D1", "D2"]}
