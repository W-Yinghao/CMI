"""C28 Q2 — source<->target factor homology. Per candidate, compare the SOURCE class-conditioned confidence
vector (conf_c0..c3 on source units) to the TARGET one (same definition, on target units). If they align, the
carrier has a source-side homologue; if not, the carrier lives only in target decision occupancy."""
from __future__ import annotations

import numpy as np

from . import schema


def _vecs(cands, role):
    S = np.array([[c[f"src_{role}_feats"][n] for n in schema.CARRIER_NAMES] for c in cands], dtype=np.float64)
    T = np.array([[c["tgt_feats"][n] for n in schema.CARRIER_NAMES] for c in cands], dtype=np.float64)
    return S, T


def homology(cands, role) -> dict:
    S, T = _vecs(cands, role)
    # per-candidate cosine similarity of the two 4-vectors
    cos = [float(np.dot(S[i], T[i]) / (np.linalg.norm(S[i]) * np.linalg.norm(T[i]) + 1e-12)) for i in range(len(S))]
    cos_mean = float(np.mean(cos))
    # class-wise correlation across candidates (per class k)
    cw = [float(np.corrcoef(S[:, k], T[:, k])[0, 1]) for k in range(schema.N_CLASSES)
          if S[:, k].std() > 1e-9 and T[:, k].std() > 1e-9]
    classwise = float(np.mean(cw)) if cw else None
    # centered-vector alignment (subtract per-class mean, then cosine)
    Sc, Tc = S - S.mean(0), T - T.mean(0)
    centered = float(np.mean([np.dot(Sc[i], Tc[i]) / (np.linalg.norm(Sc[i]) * np.linalg.norm(Tc[i]) + 1e-12)
                              for i in range(len(S))]))
    sign_agree = float(np.mean(np.sign(Sc) == np.sign(Tc)))
    # the RAW cosine of two positive similar-magnitude confidence vectors is dominated by the shared MEAN
    # structure (always high); the offset-relevant axis is the CENTERED alignment / class-wise correlation.
    informative = max(abs(centered), abs(classwise or 0.0))
    raw_cosine_mean_dominated = bool(abs(cos_mean) >= schema.ALIGN_STRONG and informative < schema.ALIGN_STRONG)
    aligned = bool(informative >= schema.ALIGN_STRONG)              # informative (centered) alignment, NOT raw cosine
    misaligned = bool(informative < schema.ALIGN_STRONG)            # not strongly aligned on the offset-relevant axis
    return {"role": role, "cosine_mean": cos_mean, "classwise_corr": classwise, "centered_alignment": centered,
            "sign_agreement": sign_agree, "informative_alignment": informative,
            "raw_cosine_mean_dominated": raw_cosine_mean_dominated, "aligned": aligned, "misaligned": misaligned,
            "note": ("source and target class-conditioned confidence ALIGN on the informative (centered) axis -> "
                     "the carrier has a source-side homologue" if aligned else
                     "raw cosine %.3f is high but CENTERED alignment is only %.3f (mean-structure artifact) -> the "
                     "source and target factors share the confidence baseline but their OFFSET-RELEVANT variation "
                     "is weakly aligned; the carrier lives in target decision occupancy, not source logit geometry"
                     % (cos_mean, centered))}
