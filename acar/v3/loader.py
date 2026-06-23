"""ACAR v3 structural real loader (DEV-engineering, NON-BINDING). SYNTHETIC FIXTURES ONLY at this stage — NO real DEV
cohort values are read until `acar-v3-dev-design-v1` is tagged. Mirrors the erm_0 dump schema
(`z_ev,y_ev,z_te,y_te,subject_id_te,recording_id_te,window_index_te`).

Binding invariants (Amendment-8 loader-binding correction):
  * FIELD-SEPARATED provenance: `full_dump` (audit only) / `source_fit` (z_ev,y_ev) / `deployment_input` (z_te + keys)
    / `label` (WindowKey-aligned y_te) / `subject_list`. `source_state_ref` and every deployment identity depend ONLY
    on `source_fit` + `deployment_input` — never on `full_dump` (which contains y_te). So flipping y_te cannot move a
    batch digest, an execution hash, a feature, U, or a routing decision.
  * IMMUTABLE `SourceStateArtifact` carries disease / embedding_dim / hashes / ref. `assert_compatible(batch)` (ref,
    disease, dim) runs BEFORE any forward pass. DEV fits via `fit_source_state_artifact*`; the external/deployment path
    rebuilds a FROZEN artifact and never calls `fit_source_state`.
  * CANONICAL ROW IDENTITY: `DeploymentBatch` stores ONE canonical row order; the execution record binds a
    `canonical_row_digest`, so probabilities produced in one order can't be re-paired with labels in another.
  * SINGLE EXECUTION: `SourceStateArtifact.execute(batch)` runs identity + the 3 actions EXACTLY once into a
    `BatchActionExecutionRecord`, from which BOTH the WindowActionSets (predictor features) and the LabeledRiskRecord
    (ΔR) are derived — they share one `execution_sha256` / `action_outputs_sha256`.

All SHA-256 are full lowercase 64-hex.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
import struct
import numpy as np

from cmi.eval.source_state import fit_source_state
from acar.config import N_CLS, RHO
from acar.actions import apply_action
from .set_features import (WindowKey, ACTION_VOCAB, NON_IDENTITY, ACTION_GEOMETRY, _build_was, _validate_proba)
from .data import (DeploymentBatch, LabeledRiskRecord, SubjectKey, deployment_batch_digest, canonical_row_digest,
                   build_deployment_batches, canon_subject, _is_hex64)
from ._util import frozen_array
from .predictors import SCHEMA_VERSION, env_versions

LOADER_SCHEMA = "acar-v3-loader/1"
PROB_SCHEMA = f"n_cls={N_CLS};class_order=ascending;rowsum=1;dtype=<f8"
REQUIRED_DEPLOY = ("z_te", "subject_id_te", "recording_id_te", "window_index_te")


# =============================================================================== strict field readers + field hashes
def _str_list(a, name):
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    k = arr.dtype.kind
    if k == "U":
        out = [str(x) for x in arr.tolist()]
    elif k == "S":
        out = [x.decode("utf-8") for x in arr.tolist()]
    elif k == "O":
        out = arr.tolist()
        if not all(isinstance(x, str) for x in out):
            raise ValueError(f"{name} object array must contain only str (no coercion)")
    else:
        raise ValueError(f"{name} must be a string array (got dtype kind {k!r}; numeric ids are not coerced)")
    if any((not isinstance(x, str)) or x == "" for x in out):
        raise ValueError(f"{name} entries must be non-empty str")
    return out


def _int_window(a, name):
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if arr.dtype.kind == "b" or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must be an integer array (got dtype {arr.dtype}; bool/float/str not coerced)")
    return [int(x) for x in arr.tolist()]


def _int_labels(a, name):
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if arr.dtype.kind == "b" or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must be an integer label array (got dtype {arr.dtype})")
    out = [int(x) for x in arr.tolist()]
    if any(not (0 <= y < N_CLS) for y in out):
        raise ValueError(f"{name} labels must be in [0, {N_CLS})")
    return out


def _z2d(a, name):
    arr = np.ascontiguousarray(np.asarray(a, float))
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty 2-D [n, d] array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} has non-finite entries")
    return arr


def _digest(tag, items) -> str:
    """Injective length-prefixed SHA-256 over a list of bytes objects."""
    h = hashlib.sha256(); h.update(tag)
    for b in items:
        h.update(struct.pack(">Q", len(b))); h.update(b)
    return h.hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canon_window(wk: WindowKey) -> str:
    return json.dumps([str(wk.dataset_id), str(wk.subject_id), str(wk.recording_id), int(wk.window_index)],
                      separators=(",", ":"))


def hash_source_fit(zev, yev) -> str:
    """Covers ONLY the source-readout training data (z_ev, y_ev) — never y_te."""
    return _digest(b"SRCFIT/1", [LOADER_SCHEMA.encode(),
                                 str(zev.shape).encode(), np.ascontiguousarray(zev, "<f8").tobytes(),
                                 str(np.asarray(yev).shape).encode(), np.ascontiguousarray(yev, "<i8").tobytes()])


def hash_deployment_input(batches) -> str:
    """Permutation-insensitive over (canonical WindowKey, z-row); content-sensitive. No labels."""
    rows = []
    for b in batches:
        for i, wk in enumerate(b.window_keys):
            rows.append((_canon_window(wk), np.ascontiguousarray(b.z[i], "<f8").tobytes()))
    rows.sort(key=lambda t: t[0])
    items = [LOADER_SCHEMA.encode()]
    for k, zb in rows:
        items.append(k.encode()); items.append(zb)
    return _digest(b"DEPIN/1", items)


def hash_labels(labels_by_window) -> str:
    items = [LOADER_SCHEMA.encode()]
    for wk in sorted(labels_by_window, key=_canon_window):
        items.append(_canon_window(wk).encode()); items.append(struct.pack(">q", int(labels_by_window[wk])))
    return _digest(b"LBL/1", items)


def hash_subject_list(subject_keys) -> str:
    """Permutation-insensitive; sensitive to add/remove of a dataset-aware SubjectKey."""
    canon = sorted({canon_subject(s) for s in subject_keys})
    return _digest(b"SUBJ/1", [LOADER_SCHEMA.encode()] + [c.encode() for c in canon])


def _read_arrays(path, keys, *, forbidden=()):
    """Open with a context manager (handle is closed on exit), materialize the requested arrays into memory, and assert
    none of the `forbidden` keys are touched. allow_pickle=False so a poisoned object array cannot execute code."""
    with np.load(path, allow_pickle=False) as o:
        have = set(o.files)
        for k in keys:
            if k not in have:
                raise ValueError(f"dump missing {k}")
        return {k: np.array(o[k], copy=True) for k in keys}, have


# ============================================================================== source-state: full immutable bytes artifact
def _canon_env(env) -> tuple:
    """Sorted, NO-DUPLICATE, (str,str) tuple — silent dict-collapse of duplicate keys is rejected."""
    items = tuple(env.items()) if isinstance(env, dict) else tuple(env)
    seen = set()
    for k, v in items:
        if not (isinstance(k, str) and isinstance(v, str)):
            raise ValueError("env entries must be (str, str)")
        if k in seen:
            raise ValueError(f"duplicate env key {k!r}")
        seen.add(k)
    return tuple(sorted(items))


def _source_state_items(coef, intercept, classes, mu_y, sig_y0, mu_pool, sig_pool0, pi_s,
                        n_cls, d, rho, eps, env, source_fit_sha256):
    """Injective length-prefixed byte items covering EVERY field that determines f_0 / the actions / ΔR — including
    `classes_` (the predict_proba→class map) and the canonical env."""
    items = [SCHEMA_VERSION.encode()] + [a.encode() for a in ACTION_VOCAB] + [PROB_SCHEMA.encode(),
             source_fit_sha256.encode()]
    for name, arr, dt in (("coef", coef, "<f8"), ("intercept", intercept, "<f8"), ("classes", classes, "<i8"),
                          ("mu_y", mu_y, "<f8"), ("mu_pool", mu_pool, "<f8"), ("Sig_pool0", sig_pool0, "<f8"),
                          ("pi_S", pi_s, "<f8")):
        a = np.ascontiguousarray(np.asarray(arr), dt)
        items += [name.encode(), str(a.shape).encode(), a.tobytes()]
    items += [b"Sig_y0", str(len(sig_y0)).encode()]
    for s in sig_y0:
        a = np.ascontiguousarray(np.asarray(s, float), "<f8"); items += [str(a.shape).encode(), a.tobytes()]
    items += [np.array([n_cls, d], "<i8").tobytes(), np.array([rho, eps], "<f8").tobytes()]
    for k, v in env:
        items += [k.encode(), v.encode()]
    return items


def _v3_source_state_ref(source_state_sha256, source_fit_sha256) -> str:
    return _digest(b"V3SS/2", [SCHEMA_VERSION.encode()] + [a.encode() for a in ACTION_VOCAB] +
                   [PROB_SCHEMA.encode(), source_state_sha256.encode(), source_fit_sha256.encode()])


@dataclass(frozen=True, slots=True)
class SourceStateArtifact:
    """Immutable BYTES artifact for the frozen source readout f_0 — peer of FittedCandidateArtifact. No mutable sklearn
    object is stored or exposed; predictions go through a private EPHEMERAL reconstruction. `source_state_sha256` covers
    coef/intercept/**classes_**/all moments/priors/n_cls,d,rho,eps/schema/vocab/prob-schema/source_fit/env. tampering
    any byte (incl. `classes_` or env) fails `verify_integrity()`. assert_compatible() gates ref/disease/dim before any
    forward pass."""
    disease: str
    embedding_dim: int
    source_fit_sha256: str
    coef: np.ndarray
    intercept: np.ndarray
    classes: np.ndarray
    mu_y: np.ndarray
    sig_y0: tuple
    mu_pool: np.ndarray
    sig_pool0: np.ndarray
    pi_s: np.ndarray
    n_cls: int
    d: int
    rho: float
    eps: float
    env: tuple
    source_state_sha256: str = field(default="", init=False)
    source_state_ref: str = field(default="", init=False)

    def __post_init__(self):
        if self.disease not in ("PD", "SCZ"):
            raise ValueError("disease must be PD or SCZ")
        if isinstance(self.embedding_dim, bool) or not isinstance(self.embedding_dim, int) or self.embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive int")
        if not _is_hex64(self.source_fit_sha256):
            raise ValueError("source_fit_sha256 must be full lowercase hex SHA-256")
        if int(self.n_cls) != N_CLS or int(self.d) != self.embedding_dim:
            raise ValueError("n_cls/d mismatch")
        object.__setattr__(self, "env", _canon_env(self.env))
        for nm in ("coef", "intercept", "mu_y", "mu_pool", "sig_pool0", "pi_s"):
            object.__setattr__(self, nm, frozen_array(np.ascontiguousarray(np.asarray(getattr(self, nm), float))))
        object.__setattr__(self, "classes", frozen_array(np.ascontiguousarray(np.asarray(self.classes, np.int64))))
        object.__setattr__(self, "sig_y0",
                           tuple(frozen_array(np.ascontiguousarray(np.asarray(s, float))) for s in self.sig_y0))
        sha = self._sha()
        object.__setattr__(self, "source_state_sha256", sha)
        object.__setattr__(self, "source_state_ref", _v3_source_state_ref(sha, self.source_fit_sha256))

    def _sha(self):
        return _digest(b"SRCSTATE/1", _source_state_items(self.coef, self.intercept, self.classes, self.mu_y,
                       self.sig_y0, self.mu_pool, self.sig_pool0, self.pi_s, self.n_cls, self.d, self.rho, self.eps,
                       self.env, self.source_fit_sha256))

    def verify_integrity(self):
        if self._sha() != self.source_state_sha256 \
                or _v3_source_state_ref(self.source_state_sha256, self.source_fit_sha256) != self.source_state_ref:
            raise ValueError("SourceStateArtifact integrity failure")

    def _ephemeral_state(self):
        """Private: rebuild an EPHEMERAL state dict (incl. a fresh classifier) that shares no mutable object with the
        artifact. NEVER cached."""
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
        clf.coef_ = np.array(self.coef, float); clf.intercept_ = np.array(self.intercept, float)
        clf.classes_ = np.array(self.classes, np.int64); clf.n_features_in_ = int(self.d)
        return dict(clf=clf, mu_y=np.array(self.mu_y, float), Sig_y0=[np.array(s, float) for s in self.sig_y0],
                    mu_pool=np.array(self.mu_pool, float), Sig_pool0=np.array(self.sig_pool0, float),
                    pi_S=np.array(self.pi_s, float), n_cls=int(self.n_cls), d=int(self.d),
                    rho=float(self.rho), eps=float(self.eps))

    def assert_compatible(self, batch: DeploymentBatch):
        self.verify_integrity()
        if self.source_state_ref != batch.source_state_ref:
            raise ValueError("batch.source_state_ref does not match this source artifact")
        if self.disease != batch.disease:
            raise ValueError(f"source artifact disease {self.disease} != batch disease {batch.disease}")
        if self.embedding_dim != int(np.asarray(batch.z).shape[1]):
            raise ValueError("embedding dimension mismatch between source artifact and batch")

    def execute(self, batch: DeploymentBatch) -> "BatchActionExecutionRecord":
        """ONE forward pass: identity + the 3 actions, on the batch's canonical row order (ephemeral state)."""
        self.assert_compatible(batch)
        if batch.fallback:
            raise ValueError("fallback batch is forced to identity; it has no action execution")
        state = self._ephemeral_state()
        z = np.ascontiguousarray(np.asarray(batch.z, float))
        p0, z0 = apply_action("identity", state, z)
        p0 = np.ascontiguousarray(np.asarray(p0, float)); z0 = np.ascontiguousarray(np.asarray(z0, float))
        _validate_proba(p0, z.shape[0], N_CLS, "p0")
        if z0.shape != z.shape or not np.all(np.isfinite(z0)):
            raise ValueError("identity embedding malformed")
        per = []
        for a in NON_IDENTITY:
            pa, za = apply_action(a, state, z)
            pa = np.ascontiguousarray(np.asarray(pa, float))
            _validate_proba(pa, z.shape[0], N_CLS, f"pa[{a}]")
            za = None if za is None else np.ascontiguousarray(np.asarray(za, float))
            per.append((a, za, pa))
        return BatchActionExecutionRecord(self.source_state_ref, batch.disease, deployment_batch_digest(batch),
                                          canonical_row_digest(batch), batch.window_keys, z0, p0, tuple(per))


def _artifact_from_state(state, disease, source_fit_sha256, env) -> SourceStateArtifact:
    clf = state["clf"]
    return SourceStateArtifact(disease, int(state["d"]), source_fit_sha256,
                               np.asarray(clf.coef_, float), np.asarray(clf.intercept_, float),
                               np.asarray(clf.classes_, np.int64), np.asarray(state["mu_y"], float),
                               tuple(np.asarray(s, float) for s in state["Sig_y0"]), np.asarray(state["mu_pool"], float),
                               np.asarray(state["Sig_pool0"], float), np.asarray(state["pi_S"], float),
                               int(state["n_cls"]), int(state["d"]), float(state["rho"]), float(state["eps"]),
                               _canon_env(env))


def fit_source_state_artifact(zev, yev, disease, source_fit_sha256, env) -> SourceStateArtifact:
    """DEV entry — the ONLY function that fits a source readout. External deployment uses load_frozen_*."""
    zev = _z2d(zev, "z_ev")
    state = fit_source_state(zev, np.asarray(yev, int), N_CLS, rho=RHO)
    return _artifact_from_state(state, disease, source_fit_sha256, env)


def load_source_artifact_from_dump(path, *, disease, env=None) -> SourceStateArtifact:
    """DEV convenience: fit f_0 from a dump's SOURCE split (z_ev,y_ev). Reads y_ev (it trains f_0) but NOT y_te."""
    env = env or env_versions()
    arr, _have = _read_arrays(path, ("z_ev", "y_ev"))
    zev = _z2d(arr["z_ev"], "z_ev"); yev = _int_labels(arr["y_ev"], "y_ev")
    if len(yev) != zev.shape[0]:
        raise ValueError("z_ev/y_ev length mismatch")
    return fit_source_state_artifact(zev, np.asarray(yev, int), disease, hash_source_fit(zev, yev), env)


def freeze_source_state_artifact(art: SourceStateArtifact) -> dict:
    """Serialize to plain numpy arrays (no pickle, no z_ev/y_ev), CARRYING the artifact's own hash + ref + env."""
    blob = {"disease": art.disease, "embedding_dim": art.embedding_dim, "source_fit_sha256": art.source_fit_sha256,
            "source_state_sha256": art.source_state_sha256, "source_state_ref": art.source_state_ref,
            "env_keys": [k for k, _ in art.env], "env_vals": [v for _, v in art.env],
            "coef": np.asarray(art.coef, float), "intercept": np.asarray(art.intercept, float),
            "classes": np.asarray(art.classes, np.int64), "mu_y": np.asarray(art.mu_y, float),
            "mu_pool": np.asarray(art.mu_pool, float), "Sig_pool0": np.asarray(art.sig_pool0, float),
            "pi_S": np.asarray(art.pi_s, float), "n_cls": int(art.n_cls), "d": int(art.d),
            "rho": float(art.rho), "eps": float(art.eps), "n_Sig_y0": len(art.sig_y0)}
    for i, s in enumerate(art.sig_y0):
        blob[f"Sig_y0_{i}"] = np.asarray(s, float)
    return blob


def load_frozen_source_state_artifact(blob) -> SourceStateArtifact:
    """External/deployment entry — rebuilds f_0 from frozen params WITHOUT calling fit_source_state, then VERIFIES the
    recomputed hash + ref against the blob's stored values (a new ref is NOT silently minted from the current env)."""
    env = tuple(zip([str(k) for k in blob["env_keys"]], [str(v) for v in blob["env_vals"]]))
    art = SourceStateArtifact(str(blob["disease"]), int(blob["embedding_dim"]), str(blob["source_fit_sha256"]),
                              np.asarray(blob["coef"], float), np.asarray(blob["intercept"], float),
                              np.asarray(blob["classes"], np.int64), np.asarray(blob["mu_y"], float),
                              tuple(np.asarray(blob[f"Sig_y0_{i}"], float) for i in range(int(blob["n_Sig_y0"]))),
                              np.asarray(blob["mu_pool"], float), np.asarray(blob["Sig_pool0"], float),
                              np.asarray(blob["pi_S"], float), int(blob["n_cls"]), int(blob["d"]),
                              float(blob["rho"]), float(blob["eps"]), env)
    if art.source_state_sha256 != str(blob["source_state_sha256"]) \
            or art.source_state_ref != str(blob["source_state_ref"]):
        raise ValueError("frozen source-state blob fails its own stored hash/ref (tamper or env drift)")
    return art


class SourceStateRegistry:
    """Per-disease registry of cohort SourceStateArtifacts. A pooled-disease DEV run holds several cohort source states
    (e.g. 3 PD or 4 SCZ); each batch resolves to the UNIQUE artifact matching its source_state_ref. An unregistered ref
    fails before any adapter; refs are unique; all artifacts share the registry's disease."""
    def __init__(self, disease):
        if disease not in ("PD", "SCZ"):
            raise ValueError("disease must be PD or SCZ")
        self.disease = disease
        self._by_ref = {}

    def add(self, art: SourceStateArtifact):
        if not isinstance(art, SourceStateArtifact):
            raise TypeError("registry holds SourceStateArtifact")
        if art.disease != self.disease:
            raise ValueError(f"artifact disease {art.disease} != registry disease {self.disease}")
        art.verify_integrity()
        if art.source_state_ref in self._by_ref:
            raise ValueError("duplicate source_state_ref in registry")
        self._by_ref[art.source_state_ref] = art
        return self

    @property
    def refs(self):
        return tuple(sorted(self._by_ref))

    def resolve(self, batch: DeploymentBatch) -> SourceStateArtifact:
        art = self._by_ref.get(batch.source_state_ref)
        if art is None:
            raise ValueError(f"unregistered source_state_ref for batch (disease {batch.disease})")
        return art

    def execute(self, batch: DeploymentBatch):
        return self.resolve(batch).execute(batch)


# ============================================================================================== deployment (no y_te)
def load_deployment_batches(path, *, disease, dataset_id, source_state_ref):
    """Build DeploymentBatch list reading ONLY the four deployment fields. y_te is never opened here."""
    arr, _have = _read_arrays(path, REQUIRED_DEPLOY)        # y_te is NOT in the read set
    z = _z2d(arr["z_te"], "z_te")
    sub = _str_list(arr["subject_id_te"], "subject_id_te")
    rec = _str_list(arr["recording_id_te"], "recording_id_te")
    win = _int_window(arr["window_index_te"], "window_index_te")
    n = z.shape[0]
    if not (len(sub) == len(rec) == len(win) == n):
        raise ValueError("z_te/subject/recording/window length mismatch")
    rows = [(sub[i], rec[i], win[i], z[i]) for i in range(n)]
    return build_deployment_batches(dataset_id, disease, rows, source_state_ref)


# ====================================================================================== single execution + outputs
@dataclass(frozen=True, slots=True)
class ActionOutputsRecord:
    """Frozen deployment forward pass for ONE batch: identity p0 + each non-identity pa. Hash binds the batch digest,
    action vocab, prob schema, and the raw probability bytes."""
    deployment_batch_digest: str
    action_vocab: tuple
    prob_schema: str
    p0: np.ndarray
    pa_by_action: tuple
    action_outputs_sha256: str = field(default="", init=False)

    def __post_init__(self):
        if not _is_hex64(self.deployment_batch_digest):
            raise ValueError("deployment_batch_digest must be a full lowercase hex SHA-256")
        if tuple(self.action_vocab) != ACTION_VOCAB:
            raise ValueError("action_vocab must equal the frozen ACTION_VOCAB")
        if self.prob_schema != PROB_SCHEMA:
            raise ValueError("prob_schema mismatch")
        p0 = np.ascontiguousarray(np.asarray(self.p0, float)); n = p0.shape[0]
        _validate_proba(p0, n, N_CLS, "p0")
        items = tuple((a, np.ascontiguousarray(np.asarray(pa, float))) for a, pa in self.pa_by_action)
        if tuple(a for a, _ in items) != NON_IDENTITY:
            raise ValueError(f"pa_by_action must be in canonical order {NON_IDENTITY}")
        for a, pa in items:
            _validate_proba(pa, n, N_CLS, f"pa[{a}]")
        object.__setattr__(self, "p0", frozen_array(p0))
        object.__setattr__(self, "pa_by_action", tuple((a, frozen_array(pa)) for a, pa in items))
        object.__setattr__(self, "action_vocab", tuple(self.action_vocab))
        object.__setattr__(self, "action_outputs_sha256", self._hash())

    def _hash(self):
        items = [LOADER_SCHEMA.encode(), self.deployment_batch_digest.encode(), self.prob_schema.encode()]
        items += [a.encode() for a in self.action_vocab]
        items.append(np.ascontiguousarray(self.p0, "<f8").tobytes())
        for a, pa in self.pa_by_action:
            items.append(a.encode()); items.append(np.ascontiguousarray(pa, "<f8").tobytes())
        return _digest(b"AOUT/1", items)

    def verify_integrity(self):
        if self._hash() != self.action_outputs_sha256:
            raise ValueError("ActionOutputsRecord integrity failure")


@dataclass(frozen=True, slots=True)
class BatchActionExecutionRecord:
    """The SINGLE execution of identity + the 3 actions for a batch. Both the WindowActionSets (features) and the
    LabeledRiskRecord (ΔR) are derived from THIS record, so they cannot come from different adapter passes."""
    source_state_ref: str
    disease: str
    deployment_batch_digest: str
    canonical_row_digest: str
    window_keys: tuple
    z0: np.ndarray
    p0: np.ndarray
    per_action: tuple                    # ((action, za_or_None, pa), ...) canonical NON_IDENTITY order
    action_outputs_sha256: str = field(default="", init=False)
    execution_sha256: str = field(default="", init=False)

    def __post_init__(self):
        for h in (self.source_state_ref, self.deployment_batch_digest, self.canonical_row_digest):
            if not _is_hex64(h):
                raise ValueError("execution record hashes must be full lowercase hex SHA-256")
        if self.disease not in ("PD", "SCZ"):
            raise ValueError("disease must be PD or SCZ")
        if tuple(a for a, _, _ in self.per_action) != NON_IDENTITY:
            raise ValueError(f"per_action must be in canonical order {NON_IDENTITY}")
        z0 = np.ascontiguousarray(np.asarray(self.z0, float)); p0 = np.ascontiguousarray(np.asarray(self.p0, float))
        n = z0.shape[0]
        _validate_proba(p0, n, N_CLS, "p0")
        if len(self.window_keys) != n:
            raise ValueError("window_keys length != n")
        per = []
        for a, za, pa in self.per_action:
            pa = np.ascontiguousarray(np.asarray(pa, float)); _validate_proba(pa, n, N_CLS, f"pa[{a}]")
            if (za is not None) != ACTION_GEOMETRY[a]:
                raise ValueError(f"geometry drift for {a}")
            za = None if za is None else np.ascontiguousarray(np.asarray(za, float))
            if za is not None and za.shape != z0.shape:
                raise ValueError("za shape mismatch")
            per.append((a, za, pa))
        object.__setattr__(self, "z0", frozen_array(z0)); object.__setattr__(self, "p0", frozen_array(p0))
        object.__setattr__(self, "per_action",
                           tuple((a, None if za is None else frozen_array(za), frozen_array(pa)) for a, za, pa in per))
        object.__setattr__(self, "window_keys", tuple(self.window_keys))
        ao = self.action_outputs_record()
        object.__setattr__(self, "action_outputs_sha256", ao.action_outputs_sha256)
        object.__setattr__(self, "execution_sha256", self._exec_hash())

    def action_outputs_record(self) -> ActionOutputsRecord:
        return ActionOutputsRecord(self.deployment_batch_digest, ACTION_VOCAB, PROB_SCHEMA, self.p0,
                                   tuple((a, pa) for a, _za, pa in self.per_action))

    def _exec_hash(self):
        items = [LOADER_SCHEMA.encode(), self.source_state_ref.encode(), self.disease.encode(),
                 self.deployment_batch_digest.encode(), self.canonical_row_digest.encode(),
                 self.action_outputs_sha256.encode(), np.ascontiguousarray(self.z0, "<f8").tobytes()]
        for a, za, _pa in self.per_action:
            items.append(a.encode()); items.append(b"NONE" if za is None else np.ascontiguousarray(za, "<f8").tobytes())
        return _digest(b"EXEC/1", items)

    def verify_integrity(self):
        if self.action_outputs_record().action_outputs_sha256 != self.action_outputs_sha256 \
                or self._exec_hash() != self.execution_sha256:
            raise ValueError("BatchActionExecutionRecord integrity failure")

    def window_action_sets(self, source_artifact: SourceStateArtifact) -> dict:
        """Build the 3 WindowActionSets from the CAPTURED outputs (no re-execution)."""
        if source_artifact.source_state_ref != self.source_state_ref:
            raise ValueError("execution record does not belong to this source artifact")
        source_artifact.verify_integrity()
        state = source_artifact._ephemeral_state()
        out = {}
        for a, za, pa in self.per_action:
            out[a] = _build_was(state, np.asarray(self.z0, float), self.window_keys,
                                a, np.asarray(self.p0, float), np.asarray(self.z0, float),
                                np.asarray(pa, float), None if za is None else np.asarray(za, float))
        return out

    def labeled_risk_record(self, labels_by_window) -> LabeledRiskRecord:
        """ΔR_a = NLL(p_a) - NLL(p_0) of the true class, labels gathered BY WindowKey in canonical row order. Binds
        deployment_batch_digest + action_outputs_sha256 (both fixed by THIS execution)."""
        self.verify_integrity()
        try:
            y = np.array([labels_by_window[wk] for wk in self.window_keys], int)
        except KeyError as e:
            raise ValueError(f"missing label for window {e.args[0]}") from None
        r0 = _nll(np.asarray(self.p0, float), y)
        drs = tuple((a, _nll(np.asarray(pa, float), y) - r0) for a, _za, pa in self.per_action)
        return LabeledRiskRecord(self.deployment_batch_digest, drs, self.action_outputs_sha256)

    def deployment_feature_record(self, source_artifact, batch, labels_by_window):
        """Bound (features, ΔR) for a batch from ONE execution. Verifies the record belongs to the batch first."""
        from .training import DeploymentFeatureRecord
        if deployment_batch_digest(batch) != self.deployment_batch_digest \
                or canonical_row_digest(batch) != self.canonical_row_digest:
            raise ValueError("execution record does not belong to this batch (digest/row mismatch)")
        sets = self.window_action_sets(source_artifact)
        dr = dict(self.labeled_risk_record(labels_by_window).delta_r_by_action)
        return DeploymentFeatureRecord(batch.disease, batch.subject, self.deployment_batch_digest,
                                       tuple((a, sets[a], dr[a]) for a in NON_IDENTITY),
                                       execution_sha256=self.execution_sha256,
                                       action_outputs_sha256=self.action_outputs_sha256)


def _nll(p, y):
    pe = np.clip(p[np.arange(len(y)), np.asarray(y, int)], 1e-12, 1.0)
    return float(-np.log(pe).mean())


# ======================================================================================= labels (the ONLY y_te reader)
def load_labels_by_window(path, *, dataset_id) -> dict:
    arr, _have = _read_arrays(path, ("y_te",) + REQUIRED_DEPLOY[1:])
    y = _int_labels(arr["y_te"], "y_te")
    sub = _str_list(arr["subject_id_te"], "subject_id_te")
    rec = _str_list(arr["recording_id_te"], "recording_id_te")
    win = _int_window(arr["window_index_te"], "window_index_te")
    n = len(y)
    if not (len(sub) == len(rec) == len(win) == n):
        raise ValueError("label field length mismatch")
    out = {}
    for i in range(n):
        wk = WindowKey(dataset_id, sub[i], rec[i], win[i])
        if wk in out:
            raise ValueError("duplicate window key among labels")
        out[wk] = int(y[i])
    return out


# =================================================================================================== manifest (S9)
@dataclass(frozen=True, slots=True)
class LoadedDumpManifest:
    dataset_id: str
    disease: str
    full_dump_sha256: str          # whole file — AUDIT ONLY, never propagated to identity
    source_fit_sha256: str
    deployment_input_sha256: str
    label_sha256: str
    subject_list_sha256: str
    n_subjects: int
    n_recordings: int
    n_windows: int
    embedding_dim: int
    schema_version: str = LOADER_SCHEMA

    def __post_init__(self):
        if self.disease not in ("PD", "SCZ"):
            raise ValueError("disease must be PD or SCZ")
        for h in (self.full_dump_sha256, self.source_fit_sha256, self.deployment_input_sha256,
                  self.label_sha256, self.subject_list_sha256):
            if not _is_hex64(h):
                raise ValueError("manifest hashes must be full lowercase hex SHA-256")
        for n in (self.n_subjects, self.n_recordings, self.n_windows, self.embedding_dim):
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise ValueError("manifest counts must be positive ints")


def build_manifest(path, *, dataset_id, disease, source_artifact, batches, labels_by_window) -> LoadedDumpManifest:
    subs = {b.subject for b in batches}
    recs = {(b.recording.subject_id, b.recording.recording_id) for b in batches}
    return LoadedDumpManifest(dataset_id, disease, file_sha256(path), source_artifact.source_fit_sha256,
                              hash_deployment_input(batches), hash_labels(labels_by_window), hash_subject_list(subs),
                              len(subs), len(recs), sum(len(b.window_keys) for b in batches),
                              int(batches[0].z.shape[1]))


# =================================================================================== immutable per-cohort binding input
@dataclass(frozen=True, eq=False)
class CohortInput:
    """One DEV cohort as a single immutable object binding dataset_id ↔ LoadedDumpManifest ↔ SourceStateArtifact ↔
    batches ↔ labels. __post_init__ verifies they are mutually consistent (so two cohorts' source states cannot be
    swapped undetected): manifest dataset/disease match; every batch's disease/dataset_id/source_state_ref match;
    recomputed field hashes == manifest; counts == manifest; batch digests unique; labels cover all WindowKeys exactly.
    `labels` is stored as an IMMUTABLE mapping (post-construction mutation that would desync the label hash is blocked).
    `raw_pipeline_sha256` / `dataset_version` carry the upstream raw→feature provenance (None at the synthetic stage)."""
    dataset_id: str
    disease: str
    manifest: LoadedDumpManifest
    source_artifact: SourceStateArtifact
    batches: tuple
    labels: dict
    full_dump_path: str
    raw_pipeline_sha256: str = None
    dataset_version: str = None

    def __post_init__(self):
        m, sa, batches = self.manifest, self.source_artifact, self.batches
        if self.raw_pipeline_sha256 is not None and not _is_hex64(self.raw_pipeline_sha256):
            raise ValueError("raw_pipeline_sha256 must be a full lowercase hex SHA-256 (or None)")
        if not (isinstance(self.dataset_id, str) and self.dataset_id):
            raise ValueError("dataset_id must be a non-empty str")
        if self.disease not in ("PD", "SCZ"):
            raise ValueError("disease must be PD or SCZ")
        if m.dataset_id != self.dataset_id or m.disease != self.disease:
            raise ValueError("manifest dataset/disease mismatch")
        sa.verify_integrity()
        if sa.disease != self.disease:
            raise ValueError("source artifact disease mismatch")
        if not batches:
            raise ValueError("empty cohort")
        for b in batches:
            if b.disease != self.disease or b.subject.dataset_id != self.dataset_id:
                raise ValueError("batch disease/dataset_id mismatch")
            if b.source_state_ref != sa.source_state_ref:
                raise ValueError("batch source_state_ref != cohort source artifact ref")
        if hash_deployment_input(batches) != m.deployment_input_sha256 or hash_labels(self.labels) != m.label_sha256:
            raise ValueError("manifest deployment_input/label hash mismatch")
        subs = {b.subject for b in batches}
        if hash_subject_list(subs) != m.subject_list_sha256:
            raise ValueError("manifest subject_list hash mismatch")
        recs = {(b.recording.subject_id, b.recording.recording_id) for b in batches}
        if (m.n_subjects, m.n_recordings, m.n_windows, m.embedding_dim) != \
                (len(subs), len(recs), sum(len(b.window_keys) for b in batches), int(batches[0].z.shape[1])):
            raise ValueError("manifest counts mismatch")
        digs = [deployment_batch_digest(b) for b in batches]
        if len(set(digs)) != len(digs):
            raise ValueError("duplicate batch digest within cohort")
        all_wk = {wk for b in batches for wk in b.window_keys}
        if set(self.labels) != all_wk:
            raise ValueError("labels do not cover the cohort WindowKeys exactly (missing/extra)")
        object.__setattr__(self, "batches", tuple(batches))
        from types import MappingProxyType
        object.__setattr__(self, "labels", MappingProxyType(dict(self.labels)))   # immutable; cannot desync label hash


def build_cohort_input(path, *, disease, dataset_id, env=None, raw_pipeline_sha256=None,
                       dataset_version=None) -> CohortInput:
    """Load one cohort dump into a fully-validated CohortInput (DEV substrate; reads y_ev for f_0 and y_te for ΔR, never
    a lockbox endpoint)."""
    sa = load_source_artifact_from_dump(path, disease=disease, env=env)
    batches = load_deployment_batches(path, disease=disease, dataset_id=dataset_id, source_state_ref=sa.source_state_ref)
    labels = load_labels_by_window(path, dataset_id=dataset_id)
    man = build_manifest(path, dataset_id=dataset_id, disease=disease, source_artifact=sa, batches=batches,
                         labels_by_window=labels)
    return CohortInput(dataset_id, disease, man, sa, tuple(batches), labels, path, raw_pipeline_sha256, dataset_version)


# ===================================================================================================== prediction gate
def predict_batch(artifact, source_artifact: SourceStateArtifact, batch: DeploymentBatch):
    """Disease- AND state-gated deployment prediction. Both gates run BEFORE any forward pass. Returns
    {action: CandidatePrediction}, or None for a fallback/forced-identity batch."""
    artifact.assert_disease(batch.disease)                 # predictor disease gate
    source_artifact.assert_compatible(batch)               # source-state ref/disease/dim gate
    if batch.fallback:
        return None
    exe = source_artifact.execute(batch)                   # ONE execution
    sets = exe.window_action_sets(source_artifact)
    return {a: artifact.predict(sets[a]) for a in NON_IDENTITY}
