"""ACAR v4 — REAL adapter: derive V4OOFRecords for the seven OLD DEV cohorts from the v3 single-execution substrate.

NON-BINDING / POST-V3 DEV_STOP / OLD-SEVEN DEV ONLY / NO EXTERNAL ARM / NO LOCKBOX. This module REUSES (imports, never
modifies) the frozen v3 machinery to compute, per (cohort, batch, fold, split): ΔR_a(B) and the bit-for-bit v2 11-D
feature vector per non-identity action, plus the v3 S5 subject-disjoint outer-fold + FIT/CAL/EVAL split. It then runs the
EXPLORATORY V4 orchestrator (acar.v4.develop.run_dev_exploration, real_mode=True) and writes
results/acar_v4_dev_exploration_001/. It produces NO SELECT/DEV_STOP/binding verdict and consumes no lockbox / external
cohort. Pre-registration: notes/ACAR_V4_DEV_EXPLORATION_RUN_PLAN.md.

The record-emission core (_fold_roles / _emit_records) is PURE and synthetic-tested; the v3-coupled glue (build_cohort
inputs, disease_exec_cache, cv_assignment, run_c0) is the same path the v3 DEV run used and is exercised only by the real
run (which fails closed via exact OOF coverage).
"""
from __future__ import annotations
import argparse
import os
from typing import Dict, List, Tuple

import numpy as np

from acar.v4.develop import (V4OOFRecord, ACTIONS, A, N_FEAT, V4DevConfig, SCORE_FAMILY_REGISTRY,
                             run_dev_exploration, write_dev_exploration_result)


# ----------------------------------------------------------------------------- pure record-emission core (synthetic-tested)

def _fold_roles(assignment_canon):
    """assignment_canon: list of {fold:int, eval:set[str], cal:set[str], fit:set[str]} (canonical subject keys, disjoint
    per fold). Returns [(fold, {canon: 'EVAL'|'CAL'|'FIT'})] with EVAL taking precedence, then CAL, then FIT; subjects
    in none of the three for a fold are omitted."""
    out = []
    for fa in assignment_canon:
        roles = {}
        for cc in fa.get("eval", ()):
            roles[cc] = "EVAL"
        for cc in fa.get("cal", ()):
            roles.setdefault(cc, "CAL")
        for cc in fa.get("fit", ()):
            roles.setdefault(cc, "FIT")
        out.append((int(fa["fold"]), roles))
    return out


def _emit_records(disease, fold_roles, cells):
    """Emit V4OOFRecords. cells: {canon: {dataset, subject, eligible:[(batch_id, dr[A], feats[A,F])], fallback:[batch_id]}}.
    Eligible batches carry real ΔR + v2 features; fallback batches realize identity (zeros, fallback=True)."""
    records = []
    for fold, roles in fold_roles:
        for cc, role in roles.items():
            cell = cells[cc]
            ds, sid = cell["dataset"], cell["subject"]
            for bid, dr, feats in cell["eligible"]:
                records.append(V4OOFRecord(disease, sid, ds, bid, fold, role, False,
                                           np.asarray(dr, float), np.asarray(feats, float), ACTIONS))
            for bid in cell["fallback"]:
                records.append(V4OOFRecord(disease, sid, ds, bid, fold, role, True,
                                           np.zeros(A), np.zeros((A, N_FEAT)), ACTIONS))
    return records


# ----------------------------------------------------------------------------- v3-coupled derivation (real data)

def build_cohort_inputs(feat_dir=None, env=None):
    """Build the seven v3 CohortInputs (fits each cohort's source state, loads label-free batches + labels)."""
    from acar.config import DISEASE, feat_dump_dir
    from acar.v3.loader import build_cohort_input
    feat_dir = feat_dir or feat_dump_dir()
    cis = []
    for disease, datasets in DISEASE.items():
        for ds in datasets:
            path = os.path.join(feat_dir, f"audit_{disease}_{ds}_erm_0.npz")
            cis.append(build_cohort_input(path, disease=disease, dataset_id=ds, env=env))
    return cis


def derive(cohort_inputs, alpha=0.10, delta=0.0):
    """Return (records, v2_replay_red_by_disease). Per disease: execute each batch ONCE (v3 disease_exec_cache) for ΔR +
    v2 features, compute the v2-replay comparator (v3 run_c0.red_router), and emit V4OOFRecords over the v3 S5 split."""
    from acar.v3 import develop as V3D
    from acar.v3.data import deployment_batch_digest, canon_subject
    from acar.v3.splits import cv_assignment
    grouped = V3D._group_cohorts(cohort_inputs)        # fail-closed: exactly the seven cohorts, both diseases
    records: List[V4OOFRecord] = []
    v2_replay: Dict[str, float] = {}
    for disease, (registry, batches, labels, _cis) in grouped.items():
        idx = V3D._subject_batches(batches)
        eligible = V3D._eligible_subjects(idx)
        all_subjects = [v["key"] for v in idx.values()]
        elig_canon = {canon_subject(s) for s in eligible}
        assignment, _ = cv_assignment(all_subjects, eligible=elig_canon)   # same call/seeds as v3 run_oof
        cache = V3D.disease_exec_cache(registry, batches, labels)
        v2_replay[disease] = float(V3D.run_c0(disease, registry, batches, labels, alpha, delta, cache=cache).red_router)
        cells = {}
        for cc, slot in idx.items():
            key = slot["key"]
            elig = []
            for b in slot["eligible"]:
                c = cache[deployment_batch_digest(b)]
                dr = np.array([float(c["dr"][a]) for a in ACTIONS], float)
                feats = np.stack([np.asarray(c["c0feat"][a], float) for a in ACTIONS])
                elig.append((deployment_batch_digest(b), dr, feats))
            fb = [deployment_batch_digest(b) for b in slot["fallback"]]
            cells[cc] = {"dataset": key.dataset_id, "subject": key.subject_id, "eligible": elig, "fallback": fb}
        assignment_canon = [{"fold": fa["fold"], "eval": {canon_subject(s) for s in fa["eval"]},
                             "cal": {canon_subject(s) for s in fa["cal"]},
                             "fit": {canon_subject(s) for s in fa["fit"]}} for fa in assignment]
        records += _emit_records(disease, _fold_roles(assignment_canon), cells)
    return records, v2_replay


# ----------------------------------------------------------------------------- runner

def run(output, feat_dir=None, env=None, apply_env_lock=True):
    """Full real exploratory run → write `output`. real_mode=True forces exact OOF coverage; G3 comparator = v2_replay
    (best_fixed reported descriptively). Returns the V4DevExplorationResult."""
    if apply_env_lock:
        from acar.v3.envlock import apply_runtime          # determinism: torch deterministic + 1-thread + threadpool
        apply_runtime()
    cohort_inputs = build_cohort_inputs(feat_dir=feat_dir, env=env)
    records, v2_replay = derive(cohort_inputs)
    cfg = V4DevConfig(g3_comparator="v2_replay")
    result = run_dev_exploration(records, config=cfg, score_families=sorted(SCORE_FAMILY_REGISTRY), real_mode=True,
                                 v2_replay_red_by_disease=v2_replay)
    write_dev_exploration_result(result, output)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="ACAR v4 real DEV exploratory run (old-seven; NON-BINDING)")
    ap.add_argument("--output", required=True, help="output dir (must not exist), e.g. results/acar_v4_dev_exploration_001")
    ap.add_argument("--feat-dir", default=None, help="feat_dump_v4 dir (default: acar.config.feat_dump_dir / $ACAR_FEAT_DUMP)")
    args = ap.parse_args(argv)
    result = run(args.output, feat_dir=args.feat_dir)
    print(f"V4_DEV run_status={result.run_status} verdict={result.verdict} "
          f"manifest_sha256={result.manifest_sha256} output={args.output}")
    return result


if __name__ == "__main__":
    main()
