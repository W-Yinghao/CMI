"""H2-CMI : Hierarchical Class-Conditional Mutual-Information Adaptation for EEG.

A self-contained research package implementing the post-review redesign described in
``notes`` (and the ICLR-direction review).  It is deliberately ISOLATED from the AAAI
``cmi`` package: nothing here mutates or imports-with-side-effects ``cmi`` internals
(a couple of read-only data loaders are reused via plain function calls).  The four
theory corrections (P0-2 .. P0-5) and the H2-CMI method (encoder + class-conditional
latent density + hierarchical-domain CMI + selective probabilistic TTA + safety gate)
all live under this namespace.

The package runs end-to-end without real EEG via ``h2cmi.data.eeg_simulator`` so every
component is exercised by ``h2cmi/tests/test_smoke.py``.

See ``h2cmi/THEORY.md`` for the corrected derivations and ``h2cmi/README.md`` for the
component map.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
