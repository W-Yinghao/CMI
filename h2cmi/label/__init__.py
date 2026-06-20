"""Site-specific label-mechanism subpackage (review section 8).

Models the observed clinical label ``Ytilde`` as a noisy, site-dependent
corruption of a latent true state ``Ystar`` via hierarchically-shrunk per-site
confusion matrices, and provides EM estimation of those matrices from the EEG
model's soft posteriors.
"""
from __future__ import annotations

from .site_mechanism import SiteLabelMechanism, estimate_confusion_em

__all__ = ["SiteLabelMechanism", "estimate_confusion_em"]
