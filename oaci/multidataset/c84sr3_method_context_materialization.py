"""C84SR3 method-context materialization under feasible dataset budgets."""
from __future__ import annotations

from typing import Any

from .c84sr1_method_context_materialization import (
    METHOD_CONTEXT_FIELDS_V2, PERFORMANCE_ESTIMATE_METHODS, Q0_MC_FIELDS,
    Q0_REGIME_FIELDS, materialize_context as historical_materialize_context,
)
from .c84sr3_common import expected_methods, finite_budgets


METHOD_CONTEXT_FIELDS_V3 = METHOD_CONTEXT_FIELDS_V2


def materialize_context(**kwargs: Any):
    return historical_materialize_context(
        **kwargs, budget_provider=finite_budgets, method_provider=expected_methods,
    )
