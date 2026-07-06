"""Project B router — action contract (Step-2B).

A lightweight, side-effect-free action enum plus normalisation/priority helpers. No model,
no TTA, no numpy — just the deployment action vocabulary the router will choose among.
"""
from __future__ import annotations

from enum import Enum


class RouterAction(str, Enum):
    REFUSE = "refuse"
    IDENTITY = "identity"
    OFFLINE_TTA = "offline_tta"
    ONLINE_TTA = "online_tta"

    @property
    def is_prediction(self) -> bool:
        """True for any action that emits a class prediction (everything but REFUSE)."""
        return self is not RouterAction.REFUSE

    @property
    def is_tta(self) -> bool:
        """True for the test-time-adaptation actions."""
        return self in (RouterAction.OFFLINE_TTA, RouterAction.ONLINE_TTA)

    @property
    def is_refusal(self) -> bool:
        return self is RouterAction.REFUSE


def normalize_action(action: "str | RouterAction") -> RouterAction:
    """Coerce a member, a value ('offline_tta'), or a name ('OFFLINE_TTA') to a RouterAction.

    Raises ValueError on anything unknown (never a silent fallback)."""
    if isinstance(action, RouterAction):
        return action
    if isinstance(action, str):
        try:
            return RouterAction(action)          # by value, e.g. "offline_tta"
        except ValueError:
            pass
        try:
            return RouterAction[action]          # by name, e.g. "OFFLINE_TTA"
        except KeyError:
            pass
    raise ValueError(f"unknown router action: {action!r}")


_CANDIDATES: dict[str, tuple[RouterAction, ...]] = {
    "offline": (RouterAction.IDENTITY, RouterAction.OFFLINE_TTA),
    "online": (RouterAction.IDENTITY, RouterAction.ONLINE_TTA),
    "both": (RouterAction.IDENTITY, RouterAction.OFFLINE_TTA, RouterAction.ONLINE_TTA),
    "identity": (RouterAction.IDENTITY,),
}


def candidate_actions(mode: str) -> tuple[RouterAction, ...]:
    """Candidate PREDICTION actions for a routing mode (REFUSE is always implicit).

    Unknown mode raises ValueError (no silent fallback)."""
    if mode not in _CANDIDATES:
        raise ValueError(f"unknown routing mode: {mode!r} (expected one of {sorted(_CANDIDATES)})")
    return _CANDIDATES[mode]


# Interventional ordering for a later router (higher = more interventional).
_PRIORITY: dict[RouterAction, int] = {
    RouterAction.REFUSE: 0,
    RouterAction.IDENTITY: 1,
    RouterAction.ONLINE_TTA: 2,
    RouterAction.OFFLINE_TTA: 3,
}


def action_priority(action: "str | RouterAction") -> int:
    return _PRIORITY[normalize_action(action)]


if __name__ == "__main__":
    # properties
    assert RouterAction.REFUSE.is_refusal and not RouterAction.REFUSE.is_prediction
    assert not RouterAction.REFUSE.is_tta
    assert RouterAction.IDENTITY.is_prediction and not RouterAction.IDENTITY.is_tta
    assert not RouterAction.IDENTITY.is_refusal
    assert RouterAction.OFFLINE_TTA.is_prediction and RouterAction.OFFLINE_TTA.is_tta
    assert RouterAction.ONLINE_TTA.is_prediction and RouterAction.ONLINE_TTA.is_tta

    # normalize_action: value, name, member; unknown raises
    assert normalize_action("offline_tta") is RouterAction.OFFLINE_TTA
    assert normalize_action("OFFLINE_TTA") is RouterAction.OFFLINE_TTA
    assert normalize_action(RouterAction.REFUSE) is RouterAction.REFUSE
    for bad in ("garbage", "Offline", "", 123):
        try:
            normalize_action(bad)  # type: ignore[arg-type]
            raise AssertionError(f"normalize_action({bad!r}) should have raised")
        except ValueError:
            pass

    # candidate_actions
    assert candidate_actions("offline") == (RouterAction.IDENTITY, RouterAction.OFFLINE_TTA)
    assert candidate_actions("online") == (RouterAction.IDENTITY, RouterAction.ONLINE_TTA)
    assert candidate_actions("both") == (RouterAction.IDENTITY, RouterAction.OFFLINE_TTA,
                                         RouterAction.ONLINE_TTA)
    assert candidate_actions("identity") == (RouterAction.IDENTITY,)
    for bad in ("nope", "Offline", "", "OFFLINE"):
        try:
            candidate_actions(bad)
            raise AssertionError(f"candidate_actions({bad!r}) should have raised")
        except ValueError:
            pass

    # priority
    assert action_priority(RouterAction.REFUSE) == 0
    assert action_priority(RouterAction.IDENTITY) == 1
    assert action_priority(RouterAction.ONLINE_TTA) == 2
    assert action_priority(RouterAction.OFFLINE_TTA) == 3
    assert action_priority("offline_tta") == 3

    # value uniqueness
    assert len({a.value for a in RouterAction}) == len(list(RouterAction))
    print("actions self-test passed")
