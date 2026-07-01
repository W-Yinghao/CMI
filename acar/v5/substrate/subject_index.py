"""ACAR V5 Stage-1B canonical subject index (pure/stdlib). Turns per-cohort RAW subject ids into explicit, collision-proof
canonical SubjectKeys `"{disease}/{cohort}/{raw}"` so two cohorts returning the same raw id (e.g. sub-001) never collapse, and
every subject's cohort is always recoverable. Duplicates are fail-closed.
"""
from __future__ import annotations
from acar.v5 import protocol as P


class SubjectIndexError(RuntimeError):
    """Raised on malformed / duplicate / cohort-mismatched subject inputs."""


def canonical_subject_key(disease, cohort, raw_subject_id):
    return f"{disease}/{cohort}/{raw_subject_id}"


class SubjectIndex:
    """Immutable disease subject index: canonical key -> (cohort, raw). Exposes sorted subject_keys + cohort/raw lookup."""

    def __init__(self, disease, triples):
        self._disease = disease
        self._by_key = {}
        for cohort, raw, key in triples:
            if key in self._by_key:
                raise SubjectIndexError(f"duplicate subject_key {key}")
            self._by_key[key] = (cohort, raw)
        self.subject_keys = tuple(sorted(self._by_key))

    @property
    def disease(self):
        return self._disease

    def __contains__(self, key):
        return key in self._by_key

    def __len__(self):
        return len(self._by_key)

    def cohort_of(self, key):
        if key not in self._by_key:
            raise SubjectIndexError(f"unknown subject_key {key}")
        return self._by_key[key][0]

    def raw_of(self, key):
        if key not in self._by_key:
            raise SubjectIndexError(f"unknown subject_key {key}")
        return self._by_key[key][1]


def build_subject_index(disease, per_cohort_raw):
    """Build a SubjectIndex from {cohort: [raw ids]}. Keys must be EXACTLY the disease's frozen DEV cohorts; a duplicate
    (cohort, raw) or a duplicate canonical key fails closed."""
    if disease not in P.DEV_COHORTS:
        raise SubjectIndexError(f"unknown disease {disease!r}")
    if set(per_cohort_raw) != set(P.DEV_COHORTS[disease]):
        raise SubjectIndexError(f"{disease}: per_cohort_raw keys must equal {sorted(P.DEV_COHORTS[disease])}")
    seen_cohort_raw, triples = set(), []
    for cohort in sorted(per_cohort_raw):
        for raw in per_cohort_raw[cohort]:
            raw = str(raw)
            if not raw:
                raise SubjectIndexError(f"{disease}/{cohort}: empty raw subject id")
            if "/" in raw:                                     # readers must return RAW ids ("sub-001"), NOT namespaced ones
                raise SubjectIndexError(f"{disease}/{cohort}: raw subject id {raw!r} must not be namespaced (no '/'); "
                                        f"the reader returns raw ids and the orchestrator builds the canonical key")
            cr = (cohort, raw)
            if cr in seen_cohort_raw:
                raise SubjectIndexError(f"{disease}: duplicate (cohort, raw) {cr}")
            seen_cohort_raw.add(cr)
            triples.append((cohort, raw, canonical_subject_key(disease, cohort, raw)))
    if not triples:
        raise SubjectIndexError(f"{disease}: no subjects")
    return SubjectIndex(disease, triples)
