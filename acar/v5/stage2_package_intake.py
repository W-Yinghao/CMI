"""ACAR V5 Stage-2A package INTAKE (read-only; fail-closed; numpy imported LAZILY inside the header reader).

This is the ONLY Stage-2 module that touches the real Stage-1B package. It ADMITS a run via the Stage-1B downstream gate
(`admit_run`: registry.json + FINALIZED.json present, marker status FINALIZED, n_refs==30, marker.registry_sha256 ==
sha256(registry.json bytes)) and then RE-VALIDATES the package for Stage-2 selection:

  * the registry is EXACTLY the 30 canonical fold refs (`CANONICAL_FOLD_REFS`);
  * NO external / provisional / excluded / held-out site token, and no other-disease cohort, appears anywhere in a ref or its
    registered metadata;
  * the 10 canonical SELECTION refs (seed 20260711) are all present and are exactly `plan.selection_refs()`;
  * the other 20 refs carry S1-robustness seeds (20260712/20260713) ONLY — they are partitioned OUT of selection.

Optionally it reads the 10 selection feature-dump HEADERS (schema/provenance only — the `embedding` array is NEVER
materialized) and asserts the pinned V5 schema and a label-free field set. It performs NO candidate selection, reads NO label,
computes NO score, and fits NO threshold.
"""
from __future__ import annotations
import json
import os
from acar.v5 import protocol as P
from acar.v5.substrate import plan as PLAN
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1b_registry_io as RIO
from acar.v5.substrate import stage1b_output_layout as LO
from acar.v5.substrate import feature_dump_schema as FS
from acar.v5.substrate import build_manifest_schema as SCH


class Stage2IntakeError(RuntimeError):
    """Raised when the real Stage-1B package is not admissible for Stage-2 selection (fail-closed)."""


FORBIDDEN_SITE_TOKENS = tuple(SCH.FORBIDDEN_SITE_TOKENS)                       # zenodo14808296, ds007526, zenodo14178398, ds007020
CANONICAL_SELECTION_REFS = frozenset(r["ref"] for r in PLAN.selection_refs())  # the 10 (seed 20260711)
# header fields carrying a JSON-string provenance map — scanned for NESTED label-like keys (mirrors feature_dump_schema.validate_loaded)
_JSON_MAP_HEADER_FIELDS = ("montage_completion_by_subject", "brainvision_read_repair_by_recording",
                           "channel_name_repair_by_recording", "channel_name_repair_subtype_by_recording")
_OTHER_DISEASE_COHORTS = {d: tuple(c for dd, cs in P.DEV_COHORTS.items() if dd != d for c in cs) for d in P.DEV_COHORTS}


class Stage2PackageView:
    """Immutable, admitted view of a Stage-1B package for Stage-2 selection. Partitions the 30 registered refs into the 10
    canonical SELECTION refs (seed 20260711) and the 20 S1-robustness-only refs (seeds 20260712/20260713). Carries NO labels."""

    def __init__(self, registry, output_root, run_id, selection_refs, robustness_only_refs):
        self._registry = registry
        self._output_root = output_root
        self._run_id = run_id
        self._selection_refs = tuple(sorted(selection_refs))
        self._robustness_only_refs = tuple(sorted(robustness_only_refs))

    @property
    def output_root(self):
        return self._output_root

    @property
    def run_id(self):
        return self._run_id

    @property
    def registry(self):
        return self._registry

    @property
    def all_refs(self):
        return tuple(sorted(self._registry._entries))

    @property
    def selection_refs(self):
        return self._selection_refs

    @property
    def robustness_only_refs(self):
        return self._robustness_only_refs

    def is_selection_ref(self, ref):
        return ref in self._selection_refs

    def assert_selection_ref(self, ref):
        if ref not in self._selection_refs:
            raise Stage2IntakeError(
                f"{ref!r} is not one of the 10 canonical Stage-2 selection refs (seed {P.SELECTION_SEED})")
        return True


def _seed_of(ref):
    try:
        return int(str(ref).rsplit("seed", 1)[1])
    except (IndexError, ValueError) as e:
        raise Stage2IntakeError(f"cannot parse seed from ref {ref!r}: {e}")


def _assert_no_forbidden_tokens(ref, entry):
    """Property: no external/held-out/excluded site token, and no other-disease DEV cohort, appears in a registered ref or its
    metadata. Scans the ref string plus the canonical JSON of the entry (hashes are hex, so only meta can carry a cohort name)."""
    disease = str(ref).split("/")[0]
    blob = str(ref) + " " + json.dumps(entry, sort_keys=True)
    hit = [tok for tok in FORBIDDEN_SITE_TOKENS if tok in blob]
    if hit:
        raise Stage2IntakeError(f"{ref}: package references forbidden external/held-out/excluded site token(s) {hit}")
    foreign = [c for c in _OTHER_DISEASE_COHORTS.get(disease, ()) if c in blob]
    if foreign:
        raise Stage2IntakeError(f"{ref}: package references other-disease DEV cohort(s) {foreign}")


def admit_and_validate_registry(output_root, run_id):
    """Admit the Stage-1B run and re-validate it for Stage-2 selection. Read-only; fail-closed (Stage2IntakeError). Returns a
    Stage2PackageView. Performs NO selection, reads NO label, computes NO score."""
    try:
        registry = RIO.admit_run(output_root, run_id)                 # PROP 1: admit_run must succeed
    except RIO.RegistryIoError as e:
        raise Stage2IntakeError(f"Stage-1B package not admissible: {e}")

    refs = set(registry._entries)
    if len(refs) != 30:                                               # PROP 2 (belt-and-suspenders; admit_run also checks)
        raise Stage2IntakeError(f"registry has {len(refs)} refs != 30")
    canonical = set(SA.CANONICAL_FOLD_REFS)
    if refs != canonical:
        raise Stage2IntakeError(f"registry refs are not exactly the 30 canonical fold refs; diff {sorted(refs ^ canonical)}")

    for ref in sorted(refs):                                          # PROP 7: no forbidden/foreign token anywhere
        _assert_no_forbidden_tokens(ref, registry._entries[ref])

    selection = sorted(r for r in refs if str(r).endswith(f"seed{P.SELECTION_SEED}"))   # PROP 3
    if set(selection) != set(CANONICAL_SELECTION_REFS):
        raise Stage2IntakeError(
            f"selection refs {selection} != the 10 canonical selection refs {sorted(CANONICAL_SELECTION_REFS)}")

    robustness_only = sorted(refs - set(selection))                   # PROP 4: seeds 12/13 partitioned OUT of selection
    if len(robustness_only) != 20:
        raise Stage2IntakeError(f"expected 20 S1-robustness-only refs, got {len(robustness_only)}")
    for ref in robustness_only:
        seed = _seed_of(ref)
        if seed == P.SELECTION_SEED:
            raise Stage2IntakeError(f"{ref}: non-selection partition contains the selection seed {P.SELECTION_SEED}")
        PLAN.assert_seed_role(seed, PLAN.S1_ROLE)                     # must be a valid pinned S1 seed

    return Stage2PackageView(registry, output_root, run_id, selection, robustness_only)


def read_feature_dump_header(npz_path):
    """Read ONLY the scalar header/provenance of a feat_dump.npz — the heavy `embedding` array is NEVER indexed. Fail-closed:
    a forbidden label-like field name, a missing header field, or a non-V5 schema_version raises. Returns a header dict."""
    import numpy as np                                                # lazy — module import stays heavy-free
    if not os.path.isfile(npz_path):
        raise Stage2IntakeError(f"feat_dump not found: {npz_path}")
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            files = set(z.files)
            forbidden = sorted(files & set(FS.FORBIDDEN_FIELDS))
            if forbidden:
                raise Stage2IntakeError(f"{npz_path}: feat_dump carries forbidden label-like field(s) {forbidden}")
            missing = [k for k in FS.HEADER_FIELDS if k not in files]
            if missing:
                raise Stage2IntakeError(f"{npz_path}: feat_dump missing header field(s) {missing}")
            # closed-schema ALLOWLIST (mirror validate_loaded): reject ANY out-of-schema member name — this closes the
            # denylist gap where a case-variant/synonym label field ('Diagnosis', 'dx', 'class', ...) would otherwise pass.
            extra = sorted(files - set(FS.HEADER_FIELDS) - set(FS.RECORD_ARRAYS))
            if extra:
                raise Stage2IntakeError(f"{npz_path}: feat_dump carries unexpected out-of-schema field(s) {extra}")

            def _scalar(k):
                v = z[k]                                              # scalar header field only — 'embedding' never indexed
                return v.item() if getattr(v, "shape", None) == () else v

            schema = str(_scalar("schema_version"))
            if schema != FS.SCHEMA_VERSION:
                raise Stage2IntakeError(f"{npz_path}: schema_version {schema!r} != {FS.SCHEMA_VERSION}")
            header = {k: _scalar(k) for k in FS.HEADER_FIELDS}
            # scan the JSON-string provenance maps for NESTED label-like keys (embedding never touched) — mirror validate_loaded
            for field in _JSON_MAP_HEADER_FIELDS:
                try:
                    parsed = json.loads(str(header[field]))
                except ValueError as e:
                    raise Stage2IntakeError(f"{npz_path}: {field} is not valid JSON: {e}")
                if set(FS._flatten_keys(parsed)) & set(FS.FORBIDDEN_FIELDS):
                    raise Stage2IntakeError(f"{npz_path}: {field} carries a nested label-like field")
            return header
    except Stage2IntakeError:
        raise
    except Exception as e:  # noqa: BLE001 — any numpy/IO failure becomes a fail-closed intake error
        raise Stage2IntakeError(f"{npz_path}: cannot read feat_dump header: {e}")


def validate_selection_feature_dumps(view):
    """Read the HEADER of each of the 10 selection refs' feat_dump.npz (headers only — no embeddings, no labels) and assert the
    pinned V5 schema plus ref/disease/seed agreement. Fail-closed. Returns {ref: header}."""
    out = {}
    for ref in view.selection_refs:
        path = os.path.join(LO.ref_output_dir(view.output_root, view.run_id, ref), "feat_dump.npz")
        header = read_feature_dump_header(path)
        if str(header.get("ref")) != ref:
            raise Stage2IntakeError(f"{path}: header ref {header.get('ref')!r} != {ref!r}")
        if str(header.get("disease")) != ref.split("/")[0]:
            raise Stage2IntakeError(f"{path}: header disease {header.get('disease')!r} != {ref.split('/')[0]!r}")
        if int(header.get("seed")) != P.SELECTION_SEED:
            raise Stage2IntakeError(f"{path}: header seed {header.get('seed')} != selection seed {P.SELECTION_SEED}")
        out[ref] = header
    return out
