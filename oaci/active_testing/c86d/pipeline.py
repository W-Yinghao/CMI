"""C86D selection freeze, held evaluator, and gated execution entrypoint.

Selection is frozen BEFORE any held-evaluation access. The held evaluator opens the
C85U utility field only after receiving a finalized freeze, and it holds no server
handle or oracle. Real execution is gated on a direct '授权 C86D'.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from .core import CONTEXTS_PER_TRIAL, verify_c85u_identity
from .policies import composite_select, select_query_sequence

_CONTRIB_FIELDS = ("nll", "correct", "confidence", "conf_bin", "signed_calibration")


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
    """Acquire labels via the sealed server, estimate, and FREEZE the selection."""
    attempt = client.open_attempt(target, budget)
    order, weights = select_query_sequence(target_pool, method, budget, seed)
    # collect queried rows per context
    per_ctx = {}                        # context -> {field: [m,81]}, labels
    receipts = []
    for trial in order:
        label, contexts = client.query(attempt, trial)
        receipts.append((trial, int(label)))
        for ctx, row in contexts.items():
            d = per_ctx.setdefault(ctx, {"labels": [], **{f: [] for f in _CONTRIB_FIELDS}})
            d["labels"].append(int(label))
            for f in _CONTRIB_FIELDS:
                d[f].append(np.asarray(row[f]))
    selected, ests = {}, {}
    for ctx, d in per_ctx.items():
        contribs = {f: np.array(d[f]) for f in _CONTRIB_FIELDS}
        sel, metrics = composite_select(d["labels"], contribs, weights)
        selected[ctx] = sel
        ests[ctx] = metrics["composite"]
    fr = SelectionFreeze(target=tuple(target), method=method, budget=budget, seed=seed,
                         query_sequence=list(order), lure_weights=list(np.asarray(weights)),
                         receipts=receipts, selected_by_context=selected,
                         per_context_estimates=ests)
    fr.lock()
    return fr


class HeldEvaluator:
    """Opens C85U only after a finalized freeze. No server handle, no oracle."""

    def __init__(self, held_by_ctx: dict, verify_identity: bool = True):
        # held_by_ctx: (target, context) -> util[81] (the frozen C85U candidate-utility field)
        self._held = held_by_ctx
        self.identity = verify_c85u_identity() if verify_identity else {"verified": False}

    def evaluate(self, freeze: SelectionFreeze) -> dict:
        if not freeze.frozen:
            raise C86DOrderingError("selection must be frozen before held evaluation")
        ctx_regret = {}
        for ctx, sel in freeze.selected_by_context.items():
            util = np.asarray(self._held[(freeze.target, ctx)])
            ctx_regret[ctx] = float(util.max() - util[sel])
        target_regret = float(np.mean(list(ctx_regret.values())))   # equal-weight 8-context mean
        return {"target": freeze.target, "method": freeze.method, "budget": freeze.budget,
                "context_regret": ctx_regret, "target_regret": target_regret,
                "n_contexts": len(ctx_regret)}


def execute(authorization: str | None = None, output_root: str = "", **kwargs):
    """Real C86D active-policy execution — REFUSES without a direct '授权 C86D'."""
    if authorization != AUTHORIZATION_PHRASE:
        raise C86DNotAuthorized(
            "C86D real execution requires a separate direct '授权 C86D'; the protocol/"
            "client-server implementation does not authorize it"
        )
    if not output_root:
        raise ValueError("authorized C86D execution requires an output_root")
    from .run import run_c86d
    return run_c86d(output_root)
