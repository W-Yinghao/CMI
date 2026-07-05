"""ACAR V5 Stage-2A selection-runner READINESS (dry-run only; fail-closed; pure/stdlib).

Stage-2 is the FIRST label-consuming step of V5. This runner proves READINESS without doing selection:

  * it ADMITS + re-validates the real Stage-1B package (via `stage2_package_intake`);
  * it BINDS the frozen 22-row candidate manifest (joint PD/SCZ) via `stage2_selection_manifest`;
  * it enumerates ONLY the 10 canonical selection refs (seed 20260711); seeds 20260712/20260713 are S1-robustness only and
    can NEVER influence candidate identity;
  * it confirms the routing/scalarization path is LABEL-FREE (no label parameter is even reachable);
  * it exposes the G1/G3/G4 certifier Stage-2 will use — `ltt.gate_disease`, in which G4 (harm_among_adapted) is conditional on
    ADAPTED subjects and a candidate for which NO subject adapts is NON-EVALUABLE and FAILS.

It SELECTS NOTHING: `dry_run_selection_readiness` returns a report with `selected_candidate = None`, and
`run_binding_selection` FAILS CLOSED (`Stage2BNotAuthorizedError`) because binding real-DEV selection is Stage-2B — a SEPARATE
authorization + reviewed code change that is not issued at this stage. G1–G5 evaluation on real data, threshold fitting on real
data, and any label read are all out of scope.
"""
from __future__ import annotations
import inspect
from acar.v5 import protocol as P
from acar.v5 import deploy as DEPLOY
from acar.v5 import scalarization as SCAL
from acar.v5 import ltt as LTT
from acar.v5.substrate import plan as PLAN
from acar.v5.substrate import feature_dump_schema as FS
from acar.v5 import stage2_package_intake as INTAKE
from acar.v5 import stage2_selection_manifest as MANIFEST


class Stage2RunnerError(RuntimeError):
    pass


class Stage2BNotAuthorizedError(Stage2RunnerError):
    """Raised on ANY attempt to run binding candidate selection in Stage-2A. Binding real-DEV selection is Stage-2B — a SEPARATE
    authorization and reviewed code change — and none is issued here, so the selection output path stays dry-run only."""


# Stage-2A is READINESS ONLY. Binding selection (Stage-2B) is a separate, later authorization + reviewed code change; until then
# this flag stays False and `run_binding_selection` cannot select.
_STAGE2B_ENABLED = False

GATE_CERTIFIER = LTT.gate_disease     # G1/G3/G4 certifier; G4 = conditional-on-adapted, no-adapt ⇒ non-evaluable ⇒ FAIL
ROUTING_FN = DEPLOY.route             # label-free routing (candidate, batch, thresholds) — no label parameter

# names that must never appear as a routing/scalarization parameter (label firewall)
_LABEL_PARAM_NAMES = set(FS.FORBIDDEN_FIELDS) | {"label", "labels", "y_true", "y_pred", "outcome"}


def assert_label_free_routing():
    """PROP 8: no label is visible to routing/scalarization code. The routing entry points take ONLY the label-free action-indexed
    feature batch (+ candidate + FIT thresholds); none exposes a label/y/target/... parameter. Fail-closed."""
    for fn in (DEPLOY.route, SCAL.decide, SCAL.proposed_action, SCAL.fit_quantiles):
        params = set(inspect.signature(fn).parameters)
        leak = params & _LABEL_PARAM_NAMES
        if leak:
            raise Stage2RunnerError(f"{fn.__module__}.{fn.__name__} exposes label-like parameter(s) {sorted(leak)}")
    return True


def assert_selection_input_allowed(ref):
    """PROP 3/4: only a seed-20260711 ref may be a Stage-2 selection input; a seed-20260712/20260713 (S1-robustness) ref raises."""
    try:
        seed = int(str(ref).rsplit("seed", 1)[1])
    except (IndexError, ValueError) as e:
        raise Stage2RunnerError(f"cannot parse seed from ref {ref!r}: {e}")
    if seed != P.SELECTION_SEED:
        raise Stage2RunnerError(
            f"{ref}: seed {seed} is S1-robustness only and may not influence Stage-2 candidate identity "
            f"(selection seed is {P.SELECTION_SEED})")
    PLAN.assert_seed_role(seed, PLAN.SELECTION_ROLE)          # belt-and-suspenders (raises for 12/13)
    return True


def selection_refs(view):
    """PROP 3/4: the EXACT 10 canonical selection refs (seed 20260711). Every one is re-checked as an allowed selection input."""
    refs = list(view.selection_refs)
    if len(refs) != 10:
        raise Stage2RunnerError(f"expected 10 selection refs, got {len(refs)}")
    for r in refs:
        assert_selection_input_allowed(r)
    return refs


class Stage2ReadinessReport:
    """Immutable Stage-2A readiness report. `selected_candidate` is ALWAYS None (Stage-2A selects nothing)."""

    def __init__(self, **kw):
        self._d = dict(kw)

    def __getitem__(self, k):
        return self._d[k]

    def get(self, k, default=None):
        return self._d.get(k, default)

    def as_dict(self):
        return dict(self._d)


def dry_run_selection_readiness(output_root, run_id, *, check_feature_dumps=False):
    """Prove Stage-2 readiness WITHOUT selecting: admit+validate the package, bind the frozen 22-row joint manifest, enumerate
    the 10 selection refs, confirm label-free routing. Reads NO label, computes NO score, fits NO threshold, selects NO candidate.
    If check_feature_dumps=True, also validates the 10 selection feat_dump HEADERS (no embeddings). Returns a Stage2ReadinessReport."""
    view = INTAKE.admit_and_validate_registry(output_root, run_id)         # PROPS 1, 2, 7
    manifest = MANIFEST.selection_manifest()                               # PROPS 5, 6
    MANIFEST.assert_joint_disease_scope(manifest)                          # PROP 6 (explicit)
    refs = selection_refs(view)                                            # PROPS 3, 4
    assert_label_free_routing()                                            # PROP 8
    fd_headers = INTAKE.validate_selection_feature_dumps(view) if check_feature_dumps else None
    return Stage2ReadinessReport(
        admitted=True,
        n_refs=len(view.all_refs),
        selection_refs=list(refs),
        robustness_excluded_refs=list(view.robustness_only_refs),
        candidate_ids=list(MANIFEST.selection_candidate_ids()),
        n_candidates=len(manifest),
        family_counts=MANIFEST.family_counts(),
        joint_disease_scope=True,
        forbidden_tokens_absent=True,
        label_free_routing=True,
        g4_semantics="conditional_on_adapted; no_subject_adapts=non_evaluable=fail",
        gate_certifier=f"{GATE_CERTIFIER.__module__}.{GATE_CERTIFIER.__name__}",
        feature_dump_headers=fd_headers,
        selected_candidate=None,                                           # PROP 10: Stage-2A selects nothing
        stage2b_authorized=bool(_STAGE2B_ENABLED),                         # False in Stage-2A
    )


def run_binding_selection(*args, **kwargs):
    """FAIL-CLOSED (PROP 10): real candidate selection is Stage-2B, a SEPARATE authorization + reviewed code change. Stage-2A
    issues no Stage-2B authorization (`_STAGE2B_ENABLED` is False and no selection logic exists here), so this ALWAYS raises —
    the Stage-2 output path is dry-run only."""
    raise Stage2BNotAuthorizedError(
        "Stage-2 binding candidate selection is not authorized in Stage-2A (readiness only). Real DEV selection (Stage-2B) "
        "requires a separate authorization pinned to a reviewed implementation SHA; no Stage-2B authorization is issued here.")
