"""ACAR V5 deployment-time guards (label firewall + external gate + fixed-candidate). Pure/stdlib+numpy.

- `route` is LABEL-FREE: it takes only (candidate, batch, thresholds) and reads only action-indexed features via scalarization.
  There is no label/y/target parameter; the DeploymentBatch carries no label. (Guarded by test_no_label_in_route.)
- `external_read_gate` is FAIL-CLOSED: no held-out/external read without a valid external authorization bound to the protocol
  tag (Step-3 has none, so it always raises here). (Guarded by test_no_external_before_tag.)
- `FixedCandidate` locks the selected candidate_id: after `fix`, any Stage-4/Stage-5 use of a different id raises. (Guarded by
  test_fixed_candidate_no_reselection.)
"""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import scalarization as S

PROTOCOL_TAG = "acar-v5-protocol"
REQUIRED_EXTERNAL_STATEMENT = (
    "Authorize ACAR-V5 single-site held-out external read exactly under ACAR_FROZEN_v5.md after Stage-4 robustness pass"
)


class ExternalNotAuthorizedError(RuntimeError):
    """Raised on any held-out/external read attempt without a valid, tag-bound external authorization."""


class ReselectionError(RuntimeError):
    """Raised when a fixed candidate_id is changed after Stage-2 selection (no reselection)."""


def route(candidate, batch, thresholds):
    """Label-free routing decision for one batch → an action in P.ACTIONS or P.IDENTITY (abstain). No label parameter exists."""
    return S.decide(candidate, batch, thresholds)


def external_read_gate(external_authorization=None, *, stage4_passed=False):
    """Fail-closed external gate. Requires (1) Stage-4 robustness passed, and (2) a valid external authorization dict binding the
    protocol tag + the exact statement + a concrete site. In the Step-3 synthetic scaffold no such authorization is issued, so
    this raises — proving external data cannot be read by the scaffold."""
    if not stage4_passed:
        raise ExternalNotAuthorizedError("external read requires a passed Stage-4 robustness gate (none in Step 3)")
    a = external_authorization
    if not isinstance(a, dict):
        raise ExternalNotAuthorizedError("no external authorization supplied")
    if a.get("protocol_tag") != PROTOCOL_TAG:
        raise ExternalNotAuthorizedError("external authorization not bound to acar-v5-protocol")
    if a.get("statement") != REQUIRED_EXTERNAL_STATEMENT:
        raise ExternalNotAuthorizedError("external authorization statement mismatch")
    site = a.get("site")
    if site not in P.EXTERNAL_PRIMARY.values():
        raise ExternalNotAuthorizedError(f"external site {site!r} is not a pre-registered primary held-out site")
    return True                                               # (unreachable in Step 3 — no authorization is created)


class FixedCandidate:
    """Locks the jointly-selected candidate_id after Stage-2. Stage-4/5 must use the SAME id (no reselection)."""

    def __init__(self):
        self._id = None

    def fix(self, candidate_id):
        if candidate_id not in P.CANDIDATE_IDS:
            raise ValueError(f"{candidate_id!r} is not in the pinned 22-row manifest")
        if self._id is not None and self._id != candidate_id:
            raise ReselectionError(f"candidate already fixed to {self._id}; cannot re-fix to {candidate_id}")
        self._id = candidate_id
        return self._id

    @property
    def candidate_id(self):
        if self._id is None:
            raise ReselectionError("no candidate fixed yet")
        return self._id

    def assert_candidate(self, candidate_id):
        if self._id is None:
            raise ReselectionError("no candidate fixed yet")
        if candidate_id != self._id:
            raise ReselectionError(f"reselection blocked: fixed {self._id}, got {candidate_id}")
        return True
