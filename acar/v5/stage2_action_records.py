"""ACAR V5 Stage-2 ACTION-RECORD computation (numpy imported LAZILY). Turns a subject's frozen substrate embeddings into the
label-free action-indexed batch the scalarization rules consume: {"batch_id", "features": {action: {feature: value}}}.

Pinned by the user (2026-07-05):
  * f_0 = the shared-covariance LDA READOUT of the frozen source_state (class means μ_k, shared pooled covariance Σ, priors π_k):
        f0(z) = softmax_k( w_k·z + b_k ),  w_k = Σ⁻¹ μ_k,  b_k = log π_k − 0.5 μ_kᵀ Σ⁻¹ μ_k.
    Same readout for pre/post:  p0 = f0(z_pre),  p_a = f0(z_after_action_a). (EEGNet head is diagnostic-only, never used here.)
  * z-space actions: identity / matched_coral / spdim / t3a. The paired features are the frozen 7 (acar.features.paired_features).

The action provider is PLUGGABLE. The DEFAULT production provider reuses the FROZEN acar.actions (matched_coral/spdim/t3a) via a
v5→old source-state adapter, LAZY-imported (torch/cmi.eval) so it is never loaded in the label-free Stage-2B0 suite; identity is
served directly by the LDA. A torch-free SYNTHETIC provider is included for fixtures/tests. Fail-closed throughout.
"""
from __future__ import annotations
from acar.v5 import protocol as P


class Stage2ActionError(RuntimeError):
    """Raised on a malformed source_state / non-invertible Σ / bad f_0 output, or when a real action cannot be applied."""


class _LdaClf:
    """Duck-typed binary linear classifier over the LDA readout (predict_proba/predict/classes_/coef_/intercept_) so the frozen
    acar.actions (which expect an sklearn-LR-like clf) can drive the real action transforms at real-run time."""

    def __init__(self, W, b, means):
        import numpy as np
        self._W = W                                              # [2, D] row k = w_k = Σ⁻¹ μ_k
        self._b = b                                              # [2]
        self.classes_ = np.asarray([0, 1], dtype=np.int64)
        self.coef_ = (W[1] - W[0]).reshape(1, -1)                # sklearn binary convention: [1, D]
        self.intercept_ = np.asarray([b[1] - b[0]], dtype=float) # [1]

    def _scores(self, Z):
        import numpy as np
        return np.asarray(Z, float) @ self._W.T + self._b

    def predict_proba(self, Z):
        import numpy as np
        s = self._scores(Z)
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, Z):
        import numpy as np
        return self.classes_[np.asarray(self._scores(Z)).argmax(axis=1)]


class SourceLDA:
    """The frozen-source LDA f_0 built from a v5 source_state dict {means[2,D], cov[D,D], priors[2], classes[2]}. Fail-closed."""

    def __init__(self, source_state):
        import numpy as np
        for k in ("means", "cov", "priors", "classes"):
            if k not in source_state:
                raise Stage2ActionError(f"source_state missing {k!r}")
        means = np.asarray(source_state["means"], float)
        cov = np.asarray(source_state["cov"], float)
        priors = np.asarray(source_state["priors"], float)
        classes = np.asarray(source_state["classes"]).astype(np.int64)
        if means.ndim != 2 or means.shape[0] != 2:
            raise Stage2ActionError(f"means must be [2, D], got {means.shape}")
        D = int(means.shape[1])
        if cov.shape != (D, D):
            raise Stage2ActionError(f"cov must be [D, D]=[{D},{D}], got {cov.shape}")
        if priors.shape != (2,):
            raise Stage2ActionError(f"priors must be [2], got {priors.shape}")
        if classes.tolist() != [0, 1]:
            raise Stage2ActionError(f"class order must be [0,1]={{control,case}}, got {classes.tolist()}")
        if not (np.isfinite(means).all() and np.isfinite(cov).all() and np.isfinite(priors).all()):
            raise Stage2ActionError("source_state has non-finite means/cov/priors")
        if not (priors > 0).all():
            raise Stage2ActionError("priors must be strictly positive (LDA log-prior)")
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError as e:
            raise Stage2ActionError(f"source covariance is not invertible (no pinned fallback): {e}")
        W = means @ cov_inv                                      # row k = μ_kᵀ Σ⁻¹ = w_kᵀ  (Σ symmetric)
        b = np.log(priors) - 0.5 * (W * means).sum(axis=1)       # b_k = log π_k − 0.5 μ_kᵀ Σ⁻¹ μ_k
        if not (np.isfinite(W).all() and np.isfinite(b).all()):
            raise Stage2ActionError("LDA readout produced non-finite weights")
        self.D = D
        self.clf = _LdaClf(W, b, means)
        self.old_state = {"clf": self.clf, "n_cls": 2, "mu_y": means,
                          "mu_pool": (priors[:, None] * means).sum(axis=0),
                          "Sig_pool0": cov, "Sig_y0": [cov, cov], "pi_S": priors,
                          "d": D, "rho": 0.1, "eps": 1e-3}

    def predict_proba(self, Z):
        import numpy as np
        p = self.clf.predict_proba(Z)
        if p.ndim != 2 or p.shape[1] != 2:
            raise Stage2ActionError(f"f_0 output must be [batch, 2], got {p.shape}")
        if not np.isfinite(p).all():
            raise Stage2ActionError("f_0 produced NaN/Inf probabilities")
        return p


def load_source_lda(source_state_or_blob):
    """Build the SourceLDA f_0 from a v5 source_state dict OR its serialized bytes (numpy lazy). Fail-closed."""
    if isinstance(source_state_or_blob, (bytes, bytearray)):
        from acar.v5.substrate import source_state as SS
        state = SS.load_source_state(source_state_or_blob)
    else:
        state = source_state_or_blob
    return SourceLDA(state)


def production_action_provider(name, source_lda, Z):
    """The FROZEN action provider for the REAL Stage-2B run: identity via the LDA; matched_coral/spdim/t3a via the frozen
    acar.actions.apply_action + the v5→old source-state adapter (LAZY torch/cmi.eval import). To be validated against the frozen
    method at real-run time; NEVER invoked in the label-free/synthetic Stage-2B0 suite."""
    import numpy as np
    Z = np.asarray(Z, float)
    if name == "identity":
        return source_lda.predict_proba(Z), Z
    if name not in P.ACTIONS:
        raise Stage2ActionError(f"unknown action {name!r}")
    try:
        from acar.actions import apply_action
    except Exception as e:  # noqa: BLE001 — torch/cmi.eval not available (e.g. py3.9 suite): real actions cannot run here
        raise Stage2ActionError(f"real action provider requires acar.actions (torch/cmi.eval): {e}")
    pa, z_post = apply_action(name, source_lda.old_state, Z)
    return np.asarray(pa, float), (None if z_post is None else np.asarray(z_post, float))


def synthetic_action_provider(name, source_lda, Z):
    """SYNTHETIC (fixtures/tests ONLY — NOT the frozen action math): deterministic, torch-free z-space transforms so the
    record-assembly + scalarization + gate pipeline can be exercised end-to-end without torch. identity is exact (LDA)."""
    import numpy as np
    Z = np.asarray(Z, float)
    if name == "identity":
        return source_lda.predict_proba(Z), Z
    if name == "matched_coral":
        z_post = Z + 0.10
        return source_lda.predict_proba(z_post), z_post
    if name == "spdim":
        z_post = Z * 1.05
        return source_lda.predict_proba(z_post), z_post
    if name == "t3a":
        z_post = Z - 0.05                                        # give t3a a well-defined z so post_sep/bures are finite in tests
        return source_lda.predict_proba(z_post), z_post
    raise Stage2ActionError(f"unknown action {name!r}")


_PROTOCOL_FROM_PHI = {"d_entropy": "d_entropy", "d_margin": "d_margin", "flip_rate": "flip_rate",
                      "js": "JS", "bures": "Bures", "post_sep": "post_sep", "n_eff": "n_eff"}


def _to_protocol_features(phi):
    """Map acar.features.paired_features keys (lowercase js/bures) to the protocol.FEATURES names (JS/Bures); NaN preserved."""
    out = {}
    for src, dst in _PROTOCOL_FROM_PHI.items():
        v = phi[src]
        out[dst] = float(v) if v is not None else float("nan")
    if set(out) != set(P.FEATURES):
        raise Stage2ActionError(f"feature set mismatch {sorted(out)} != {sorted(P.FEATURES)}")
    return out


def subject_action_outputs(Z, source_lda, *, action_provider=production_action_provider):
    """Label-free per-action outputs for one batch of embeddings: {"identity": (p0[n,2], z0), "matched_coral": (pa, z_post),
    "spdim": (pa, z_post), "t3a": (pa, z_post_or_None)}. Computed from embeddings + f_0 only — no label."""
    import numpy as np
    Z = np.asarray(Z, float)
    out = {"identity": action_provider("identity", source_lda, Z)}
    for a in P.ACTIONS:
        out[a] = action_provider(a, source_lda, Z)
    return out


def batch_from_outputs(subject_key, outputs):
    """Assemble the scalarization batch {"batch_id", "features": {action: {feature: value}}} from precomputed, label-free action
    outputs — the 3 non-identity actions' 7 paired features against the identity reference p0 = f_0(z)."""
    from acar.features import paired_features                    # lazy (numpy-only)
    p0, z0 = outputs["identity"]
    feats = {}
    for a in P.ACTIONS:
        pa, z_post = outputs[a]
        feats[a] = _to_protocol_features(paired_features(p0, pa, z0, z_post))
    return {"batch_id": str(subject_key), "features": feats}


def build_subject_batch(subject_key, Z, source_lda, *, action_provider=production_action_provider):
    """One batch = one subject's windows → the label-free scalarization batch. Convenience over subject_action_outputs +
    batch_from_outputs."""
    return batch_from_outputs(subject_key, subject_action_outputs(Z, source_lda, action_provider=action_provider))


def build_batches(by_subject, subject_keys, source_lda, *, action_provider=production_action_provider):
    """Build one batch per subject_key (in the given order) from a feature-loader `by_subject` map. Returns a list of batches."""
    batches = []
    for sk in subject_keys:
        rec = by_subject.get(sk)
        if rec is None:
            raise Stage2ActionError(f"subject {sk!r} not present in the loaded feature map")
        batches.append(build_subject_batch(sk, rec["embedding"], source_lda, action_provider=action_provider))
    return batches
