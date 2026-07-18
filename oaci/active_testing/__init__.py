"""C86LP — C84 construction-pool trial-contribution development field.

Development-only, shadow-only query infrastructure for active testing.  Opens no
real C84 construction labels, target predictions, Q0 shards, C84S result tables,
or C85U utility values (see ``constants.DEVELOPMENT_ONLY_BOUNDARY``).

Isolation gradient enforced here:
  unlabeled acquisition pool (client)  <  sealed label oracle (server)  <  held
  C85U development outcome (identity-bound only, never opened).

One query reveals one trial's label + registered linear contribution vector.
Balanced accuracy / ECE / midranks / composite utility / selected action /
target regret remain nonlinear plugins with no unbiasedness claim.
"""
from __future__ import annotations

import numpy as np

from . import constants as K
from .constants import (  # noqa: F401
    LINEAR_MOMENTS,
    NONLINEAR_PLUGINS,
    DEVELOPMENT_ONLY_BOUNDARY,
)
from .contribution import (  # noqa: F401
    C86LPClaimError,
    ContributionRow,
    assert_linear_claim,
    compute_contribution,
    pairwise_nll_differences,
    unbiasedness_claim,
)
from .field import (  # noqa: F401
    C86LPFieldError,
    DevelopmentField,
    DevelopmentFieldManifest,
    LabelOracleRow,
    UnlabeledPoolRow,
    build_manifest,
)
from .query_server import (  # noqa: F401
    C86LPInputUnavailable,
    C86LPQueryError,
    QueryReceipt,
    QueryResponse,
    QueryServer,
)


def build_shadow_field(
    *, n_targets: int = 3, n_contexts: int = 4, n_trials_per_context: int = 5,
    n_candidates: int = 81, n_eval_trials: int = 7, seed: int = 0,
) -> tuple[DevelopmentField, dict[tuple[str, object], bool]]:
    """Build a small deterministic synthetic field + budget-availability map.

    Construction and evaluation trial-ID namespaces are disjoint by construction.
    No real data is touched; probabilities are RNG-seeded.
    """
    rng = np.random.default_rng(seed)
    construction_ids: list[str] = []
    plan: list[tuple[str, str, str, int, np.ndarray]] = []
    for ti in range(n_targets):
        target = f"T{ti}"
        for ci in range(n_contexts):
            context = f"{target}/ctx{ci}"
            for j in range(n_trials_per_context):
                trial_id = f"con-{target}-c{ci}-t{j}"
                construction_ids.append(trial_id)
                label = int(rng.integers(0, 2))
                p1 = rng.uniform(0.02, 0.98, size=n_candidates)
                probs = np.stack([1.0 - p1, p1], axis=1)
                plan.append((target, context, trial_id, label, probs))
    eval_ids = frozenset(f"eval-{i}" for i in range(n_eval_trials))

    field = DevelopmentField(
        declared_contexts=n_targets * n_contexts,
        construction_trial_ids=frozenset(construction_ids),
        evaluation_trial_ids=eval_ids,
    )
    candidate_ids = tuple(f"cand{k}" for k in range(n_candidates))
    for target, context, trial_id, label, probs in plan:
        row = UnlabeledPoolRow(
            dataset="ShadowDS", target=target, context=context, trial_id=trial_id,
            candidate_ids=candidate_ids, candidate_probs=probs,
            hard_preds=np.argmax(probs, axis=1), confidence=np.max(probs, axis=1),
            session_run="shadow-s0-r0", input_digest="shadow",
        )
        field.add_trial(row, true_label=label, construction_view_identity="shadow-construction-view")

    availability = {(f"T{ti}", b): True for ti in range(n_targets) for b in K.BUDGET_GRID}
    return field, availability


def validate() -> dict:
    """Cheap self-check that the shadow field + server obey the isolation contract."""
    field, availability = build_shadow_field()
    manifest = build_manifest(field)
    gate = manifest.publish()
    server = QueryServer(field, availability)
    server.open_attempt("a0", "T0", 4)
    trial = next(iter(t for t in field.construction_trial_ids
                      if field._oracle[t].target == "T0"))
    resp = server.query("a0", trial)
    return {
        "gate": gate,
        "n_construction_trials": manifest.n_construction_trials,
        "coverage_complete": manifest.coverage_complete,
        "first_query_trial": resp.trial_id,
        "development_only_boundary": DEVELOPMENT_ONLY_BOUNDARY,
    }
