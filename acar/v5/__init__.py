"""ACAR V5 — Substrate-Robust Constrained-Utility Router (SYNTHETIC-ONLY scaffold).

Protocol frozen at tag `acar-v5-protocol` @ 4278435 (notes/ACAR_FROZEN_v5.md + the 3 companions). This package is the Step-3
scaffold: the pinned protocol constants + the fail-closed GUARDS that prove the code cannot violate the protocol. It performs NO
DEV embedding, NO real cohort read, NO substrate training, NO candidate selection, NO compatibility replay, NO external/held-out
read, and consumes NO lockbox — everything here runs on synthetic inputs only (enforced by the guard tests).
"""
