"""Self-supervised / reconstruction auxiliaries for H2-CMI.

Decoupled anti-collapse auxiliaries (review 5.2 / 5.6) that operate only on the
class latent ``z_c`` and the raw input ``x`` -- never on encoder internals.
"""
from __future__ import annotations

from .aux import SSLAux, vicreg_penalty

__all__ = ["vicreg_penalty", "SSLAux"]
