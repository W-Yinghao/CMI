"""ACAR v3 structural real loader (DEV-engineering, NON-BINDING). SYNTHETIC FIXTURES ONLY at this stage — NO real DEV
cohort values are read until `acar-v3-dev-design-v1` is tagged. This module mirrors the real erm_0 dump schema
(`z_ev,y_ev,z_te,y_te,subject_id_te,recording_id_te,window_index_te`) so the same code path serves both, but the
LABEL FIREWALL is structural:

  * `load_deployment_batches()` reads ONLY z_te / subject_id_te / recording_id_te / window_index_te. It never opens
    `y_te`. Strict dtypes: window index must be a true np.integer (bool/float/str rejected — no `int()` coercion);
    ids must be string arrays (numeric rejected — no coercion).
  * `compute_action_outputs()` runs the frozen source readout (identity + each action) to produce an immutable
    `ActionOutputsRecord` (deployment forward pass; reads z only).
  * `load_labels_by_window()` / `labeled_risk_record()` are the ONLY functions that read `y_te`; ΔR is aligned to the
    batch by WindowKey (NOT row order) and the resulting `LabeledRiskRecord` binds BOTH the deployment_batch_digest
    AND the action_outputs_sha256, so a ΔR can never be silently paired with different action outputs.
  * `predict_batch()` calls `artifact.assert_disease(batch.disease)` BEFORE any forward pass, so a PD artifact on an
    SCZ batch fails closed before touching the data.

All SHA-256 are full lowercase 64-hex. The v3 canonical source-state ref folds the readout/moment hash together with
the source-dump file hash, the schema, the action vocab, the probability schema, and the library versions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import struct
import numpy as np

from cmi.eval.source_state import fit_source_state, source_state_hash
from acar.config import N_CLS, RHO
from acar.actions import apply_action
from .set_features import (WindowKey, ACTION_VOCAB, NON_IDENTITY, build_action_sets, _validate_proba)
from .data import (DeploymentBatch, LabeledRiskRecord, deployment_batch_digest, build_deployment_batches, _is_hex64)
from .predictors import SCHEMA_VERSION, env_versions

LOADER_SCHEMA = "acar-v3-loader/1"
PROB_SCHEMA = f"n_cls={N_CLS};class_order=ascending;rowsum=1;dtype=<f8"
REQUIRED_DEPLOY = ("z_te", "subject_id_te", "recording_id_te", "window_index_te")


# ----------------------------------------------------------------------------------------------- strict field readers
def _str_list(a, name):
    """1-D array of NON-EMPTY strings. Numeric arrays are rejected (no coercion); bytes are decoded; object arrays must
    hold only str."""
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
    """1-D array of true integers. bool/float/str are rejected (NOT coerced via int())."""
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if arr.dtype.kind == "b" or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must be an integer array (got dtype {arr.dtype}; bool/float/str not coerced)")
    return [int(x) for x in arr.tolist()]


def _int_labels(a, name):
    """Class labels: true integers in [0, N_CLS). bool/float rejected."""
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
    arr = np.asarray(a, float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty 2-D [n, d] array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} has non-finite entries")
    return arr


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _open(path):
    o = np.load(path, allow_pickle=False)        # allow_pickle=False: a poisoned object array cannot execute code
    have = set(o.files)
    return o, have


# ------------------------------------------------------------------------------------------- source state + canonical ref
def v3_source_state_ref(state, source_dump_sha256, env) -> str:
    """Lowercase 64-hex over an INJECTIVE (length-prefixed) encoding of: schema, action vocab, probability schema,
    the readout/moment hash, the source-dump file hash, and the library versions."""
    if not _is_hex64(source_dump_sha256):
        raise ValueError("source_dump_sha256 must be a full lowercase hex SHA-256")
    ssh = source_state_hash(state)
    if not _is_hex64(ssh):
        raise ValueError("source_state_hash is not 64-hex")
    h = hashlib.sha256()

    def lp(b):
        h.update(struct.pack(">Q", len(b))); h.update(b)
    h.update(b"V3SS")
    lp(SCHEMA_VERSION.encode())
    for a in ACTION_VOCAB:
        lp(a.encode())
    lp(PROB_SCHEMA.encode())
    lp(ssh.encode())
    lp(source_dump_sha256.encode())
    for k, v in sorted(env.items()):
        lp(str(k).encode()); lp(str(v).encode())
    return h.hexdigest()


def load_source_state(path):
    """Fit the deployed source readout from a dump's SOURCE split (z_ev,y_ev — the f_0 training data, never the target
    deployment labels). Returns (state, source_dump_sha256). Reading y_ev is part of building f_0; it is NOT the
    deployment-time label firewall (that protects y_te)."""
    o, have = _open(path)
    for k in ("z_ev", "y_ev"):
        if k not in have:
            raise ValueError(f"source dump missing {k}")
    zev = _z2d(o["z_ev"], "z_ev"); yev = _int_labels(o["y_ev"], "y_ev")
    if len(yev) != zev.shape[0]:
        raise ValueError("z_ev/y_ev length mismatch")
    state = fit_source_state(zev, np.asarray(yev, int), N_CLS, rho=RHO)
    return state, file_sha256(path)


# ------------------------------------------------------------------------------------------------- deployment (no y_te)
def load_deployment_batches(path, *, disease, dataset_id, source_state_ref):
    """Build DeploymentBatch list reading ONLY the four deployment fields. y_te is never opened here."""
    o, have = _open(path)
    for k in REQUIRED_DEPLOY:
        if k not in have:
            raise ValueError(f"deployment dump missing {k}")
    z = _z2d(o["z_te"], "z_te")
    sub = _str_list(o["subject_id_te"], "subject_id_te")
    rec = _str_list(o["recording_id_te"], "recording_id_te")
    win = _int_window(o["window_index_te"], "window_index_te")
    n = z.shape[0]
    if not (len(sub) == len(rec) == len(win) == n):
        raise ValueError("z_te/subject/recording/window length mismatch")
    rows = [(sub[i], rec[i], win[i], z[i]) for i in range(n)]
    return build_deployment_batches(dataset_id, disease, rows, source_state_ref)


# ------------------------------------------------------------------------------------------------ immutable action outputs
@dataclass(frozen=True, slots=True)
class ActionOutputsRecord:
    """Frozen deployment forward pass for ONE batch: identity p0 + each non-identity pa. Hash binds the batch digest,
    the action vocabulary, the probability schema, and the raw probability bytes — so a ΔR computed against it cannot be
    silently re-paired with different outputs."""
    deployment_batch_digest: str
    action_vocab: tuple
    prob_schema: str
    p0: np.ndarray
    pa_by_action: tuple                          # ((action, [n,n_cls]), ...) in canonical NON_IDENTITY order
    action_outputs_sha256: str = field(default="", init=False)

    def __post_init__(self):
        from ._util import frozen_array
        if not _is_hex64(self.deployment_batch_digest):
            raise ValueError("deployment_batch_digest must be a full lowercase hex SHA-256")
        if tuple(self.action_vocab) != ACTION_VOCAB:
            raise ValueError("action_vocab must equal the frozen ACTION_VOCAB")
        if self.prob_schema != PROB_SCHEMA:
            raise ValueError("prob_schema mismatch")
        p0 = np.ascontiguousarray(np.asarray(self.p0, float))
        n = p0.shape[0]
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
        h = hashlib.sha256()

        def lp(b):
            h.update(struct.pack(">Q", len(b))); h.update(b)
        h.update(b"AOUT")
        lp(LOADER_SCHEMA.encode()); lp(self.deployment_batch_digest.encode()); lp(self.prob_schema.encode())
        for a in self.action_vocab:
            lp(a.encode())
        lp(np.ascontiguousarray(self.p0, dtype="<f8").tobytes())
        for a, pa in self.pa_by_action:
            lp(a.encode()); lp(np.ascontiguousarray(pa, dtype="<f8").tobytes())
        return h.hexdigest()

    def verify_integrity(self):
        if self._hash() != self.action_outputs_sha256:
            raise ValueError("ActionOutputsRecord integrity failure")


def compute_action_outputs(state, batch: DeploymentBatch) -> ActionOutputsRecord:
    """Deployment forward pass (reads z only). Fallback batches are forced identity upstream and have no action set."""
    if batch.fallback:
        raise ValueError("fallback batch is forced to identity; it has no action outputs")
    z = np.asarray(batch.z, float)
    p0, _z0 = apply_action("identity", state, z)
    p0 = np.asarray(p0, float)
    _validate_proba(p0, z.shape[0], N_CLS, "p0")
    pa_items = []
    for a in NON_IDENTITY:
        pa, _za = apply_action(a, state, z)
        pa = np.asarray(pa, float)
        _validate_proba(pa, z.shape[0], N_CLS, f"pa[{a}]")
        pa_items.append((a, pa))
    return ActionOutputsRecord(deployment_batch_digest(batch), ACTION_VOCAB, PROB_SCHEMA, p0, tuple(pa_items))


# ------------------------------------------------------------------------------------------------------- labels (y_te ONLY)
def load_labels_by_window(path, *, dataset_id) -> dict:
    """The ONLY function that reads y_te. Returns {WindowKey: int}; aligned later by key, never by row order."""
    o, have = _open(path)
    for k in ("y_te",) + REQUIRED_DEPLOY[1:]:
        if k not in have:
            raise ValueError(f"label dump missing {k}")
    y = _int_labels(o["y_te"], "y_te")
    sub = _str_list(o["subject_id_te"], "subject_id_te")
    rec = _str_list(o["recording_id_te"], "recording_id_te")
    win = _int_window(o["window_index_te"], "window_index_te")
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


def _nll(p, y):
    pe = np.clip(p[np.arange(len(y)), np.asarray(y, int)], 1e-12, 1.0)
    return float(-np.log(pe).mean())


def labeled_risk_record(batch: DeploymentBatch, ao: ActionOutputsRecord, labels_by_window) -> LabeledRiskRecord:
    """ΔR_a = R(p_a) - R(p_0) (mean NLL of the true class), labels gathered BY WindowKey in the batch's window order
    (the same order as ao.p0/pa rows). Binds digest + action_outputs_sha256."""
    if deployment_batch_digest(batch) != ao.deployment_batch_digest:
        raise ValueError("action outputs do not belong to this batch")
    ao.verify_integrity()
    try:
        y = np.array([labels_by_window[wk] for wk in batch.window_keys], int)
    except KeyError as e:
        raise ValueError(f"missing label for window {e.args[0]}") from None
    r0 = _nll(ao.p0, y)
    pad = dict(ao.pa_by_action)
    drs = tuple((a, _nll(pad[a], y) - r0) for a in NON_IDENTITY)
    return LabeledRiskRecord(ao.deployment_batch_digest, drs, ao.action_outputs_sha256)


# ----------------------------------------------------------------------------------------------------- prediction gate
def predict_batch(artifact, state, batch: DeploymentBatch):
    """Disease-gated deployment prediction. assert_disease runs BEFORE any forward pass (PD artifact on SCZ batch fails
    closed here). Returns {action: CandidatePrediction}, or None for a fallback/forced-identity batch."""
    artifact.assert_disease(batch.disease)
    if batch.fallback:
        return None
    sets = build_action_sets(state, np.asarray(batch.z, float), batch.window_keys)
    if not isinstance(sets, dict):                # FallbackBatchRecord (defensive; fallback already handled above)
        return None
    return {a: artifact.predict(sets[a]) for a in NON_IDENTITY}
