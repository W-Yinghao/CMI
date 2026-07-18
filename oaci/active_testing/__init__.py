"""C86LP — C84 construction-pool trial-contribution development field.

Development-only, shadow-only query instrument for active testing.  Opens no real
C84 construction labels, target predictions, Q0 shards, C84S result tables, or
C85U utility values (see ``constants.DEVELOPMENT_ONLY_BOUNDARY``).

Isolation gradient (LOGICAL/API only at C86LP; real C86L would use separate
processes/filesystem roots):
  unlabeled acquisition pool (client)  <  sealed label oracle (server)  <  held
  C85U development outcome (identity-bound only, never opened).

Semantics B: a physical trial appears in several contexts; one physical-label
query reveals one label + one contribution row per context, and the budget counts
physical labels per target (a label is never double-billed across its contexts).

The instrument's purpose is the label-complexity probe, not a paper: measure how
much queried target-label information is minimally required to identify the best
candidate, and whether adaptive acquisition beats passive uniform on mean AND
target-tail risk across cohorts (see the pre-registered taxonomy in ``pilot``).
"""
from __future__ import annotations

import numpy as np

from . import constants as K
from .constants import (  # noqa: F401
    BOUNDARY_TAXONOMY,
    DEVELOPMENT_ONLY_BOUNDARY,
    GATE_INSTRUMENT,
    ISOLATION_LEVEL,
    LINEAR_MOMENTS,
    NONLINEAR_PLUGINS,
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
    *, n_targets: int = 3, n_contexts_per_target: int = 8, n_physical_trials: int = 6,
    n_candidates: int = 81, n_eval_trials: int = 7, seed: int = 0,
) -> tuple[DevelopmentField, dict[tuple[str, object], bool]]:
    """Build a production-equivalent synthetic field + budget-availability map.

    Each physical trial appears in every context of its target (default 8), each
    context with its own [K,2] probabilities.  Construction and evaluation trial-ID
    namespaces are disjoint.  No real data is touched.
    """
    rng = np.random.default_rng(seed)
    construction_ids: list[str] = []
    plan: list[tuple[str, str, int, dict[str, np.ndarray]]] = []
    for ti in range(n_targets):
        target = f"T{ti}"
        contexts = [f"{target}/ctx{ci}" for ci in range(n_contexts_per_target)]
        for j in range(n_physical_trials):
            trial_id = f"con-{target}-t{j}"
            construction_ids.append(trial_id)
            label = int(rng.integers(0, 2))
            context_probs = {}
            for ctx in contexts:
                p1 = rng.uniform(0.02, 0.98, size=n_candidates)
                context_probs[ctx] = np.stack([1.0 - p1, p1], axis=1)
            plan.append((target, trial_id, label, context_probs))
    eval_ids = frozenset(f"eval-{i}" for i in range(n_eval_trials))

    field = DevelopmentField(
        declared_contexts=n_targets * n_contexts_per_target,
        construction_trial_ids=frozenset(construction_ids),
        evaluation_trial_ids=eval_ids,
    )
    for target, trial_id, label, context_probs in plan:
        field.add_physical_trial(target, trial_id, label, context_probs)

    availability = {(f"T{ti}", b): True for ti in range(n_targets) for b in K.BUDGET_GRID}
    return field, availability


def validate() -> dict:
    """Cheap self-check that the shadow field + server obey the isolation contract."""
    field, availability = build_shadow_field()
    manifest = build_manifest(field)
    gate = manifest.publish()
    server = QueryServer(field, availability)
    server.open_attempt("a0", "T0", 4)
    trial = next(t for t in field.construction_trial_ids if field._target_of[t] == "T0")
    resp = server.query("a0", trial)
    return {
        "gate": gate,
        "isolation_level": ISOLATION_LEVEL,
        "n_construction_trials": manifest.n_construction_trials,
        "contexts_per_trial": len(resp.contributions),
        "coverage_complete": manifest.coverage_complete,
        "first_query_trial": resp.trial_id,
        "development_only_boundary": DEVELOPMENT_ONLY_BOUNDARY,
    }
