"""C86L authorized full-field build — real data, Semantics-B, fail-closed.

Runs ONLY under a direct '授权 C86L' (see c86l_production.execute). Builds the
development-only query field on three separate physical roots from the verified
real inputs:

  predictions  : oaci-c84-full-field-target-replay-v2/.../complete_target_unlabeled_v2/*.npz (1944 units)
  construction : oaci-c84s-analysis-v3/stage_a_labels/target_construction_label_view/labels.csv (4773 rows)

Topology (verified): context = (dataset, subject, panel, seed, level) = 944;
candidate = (regime, trajectory_order) = 1 ERM + 40 OACI + 40 SRC = 81;
join key = target_trial_id. One physical construction label -> its 8 contexts.

Every arithmetic/identity invariant is asserted; any mismatch aborts with no
partial publication (the output root is only finalized on full success).
"""
from __future__ import annotations

import collections
import csv
import glob
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .contribution import compute_contribution

# --- verified real input roots + identities (bound; fail-closed on mismatch) ---
PRED_ROOT = ("/projects/EEG-foundation-model/yinghao/oaci-c84-full-field-target-replay-v2/"
             "lock_f0c369ee273352b47e36/complete_target_unlabeled_v2")
LABEL_VIEW = ("/projects/EEG-foundation-model/yinghao/oaci-c84s-analysis-v3/"
              "stage_a_labels/target_construction_label_view")
EVAL_VIEW = ("/projects/EEG-foundation-model/yinghao/oaci-c84s-analysis-v3/"
             "stage_a_labels/target_evaluation_label_view")
CONSTRUCTION_LABELS_SHA = "fdf36052d36ad9546cda06cbc567f68cdcced7ad08fd1311ab949471218b3134"

N_UNITS = 1944
N_CONTEXTS = 944
N_CANDIDATES = 81
N_CONSTRUCTION = 4773
N_CONTEXT_TRIALS = 38_184           # 4773 * 8
N_CONTRIBUTIONS = 3_092_904         # 38_184 * 81
REGIME_RANK = {"ERM": 0, "OACI": 1, "SRC": 2}


class C86LBuildError(RuntimeError):
    pass


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise C86LBuildError(msg)


def load_construction_labels() -> dict:
    """(dataset, subject, trial_id) -> int label. Verifies SHA, count, split purity."""
    csv_path = os.path.join(LABEL_VIEW, "labels.csv")
    _require(_sha256_file(csv_path) == CONSTRUCTION_LABELS_SHA,
             "construction labels.csv SHA mismatch — fail closed")
    labels = {}
    rows = list(csv.DictReader(open(csv_path)))
    _require(len(rows) == N_CONSTRUCTION, f"construction rows {len(rows)} != {N_CONSTRUCTION}")
    for r in rows:
        _require(r["split_identity"] == "construction", "non-construction row in construction view")
        key = (r["dataset"], int(r["target_subject_id"]), r["target_trial_id"])
        _require(key not in labels, f"duplicate construction trial {key}")
        labels[key] = int(r["canonical_class_label"])
    return labels


def load_eval_trial_ids() -> set:
    """Evaluation trial ids, for construction ⟂ evaluation disjointness."""
    csv_path = os.path.join(EVAL_VIEW, "labels.csv")
    return {(r["dataset"], int(r["target_subject_id"]), r["target_trial_id"])
            for r in csv.DictReader(open(csv_path))}


def scan_units() -> dict:
    """zoo (dataset,panel,seed,level) -> canonically ordered list of (path, regime, traj)."""
    paths = sorted(glob.glob(os.path.join(PRED_ROOT, "*.npz")))
    _require(len(paths) == N_UNITS, f"found {len(paths)} npz != {N_UNITS}")
    zoo = collections.defaultdict(list)
    for p in paths:
        z = np.load(p, allow_pickle=True)
        key = (str(z["dataset"]), str(z["panel"]), int(z["training_seed"]), int(z["level"]))
        zoo[key].append((p, str(z["regime"]), int(z["trajectory_order"])))
    _require(len(zoo) == 24, f"{len(zoo)} zoo-contexts != 24")
    for key, units in zoo.items():
        _require(len(units) == N_CANDIDATES, f"zoo {key} has {len(units)} candidates != 81")
        comp = collections.Counter(r for _, r, _ in units)
        _require(comp == {"ERM": 1, "OACI": 40, "SRC": 40}, f"zoo {key} composition {dict(comp)}")
        units.sort(key=lambda u: (REGIME_RANK[u[1]], u[2]))     # canonical order
    return zoo


def build(output_root: str) -> dict:
    t0 = time.time()
    labels = load_construction_labels()
    eval_ids = load_eval_trial_ids()
    _require(not (set(labels) & eval_ids), "construction ⟂ evaluation overlap is non-zero")
    zoo = scan_units()

    # subjects per dataset (only those with construction trials)
    subjects = collections.defaultdict(set)
    for (ds, subj, _tid) in labels:
        subjects[ds].add(subj)

    staging = Path(output_root + ".staging")
    pool_root = staging / "acquisition_unlabeled_pool"
    contrib_root = staging / "query_contribution_store"
    oracle_root = staging / "acquisition_label_oracle"
    for r in (pool_root, contrib_root, oracle_root):
        r.mkdir(parents=True, exist_ok=True)

    total_contexts = 0
    total_context_trials = 0
    total_contributions = 0
    context_index = []

    for (ds, panel, seed, level), units in sorted(zoo.items()):
        # load the 81 candidates' (subject, trial_id, probs) once for this zoo
        loaded = []
        cand_order = []
        for p, regime, traj in units:
            z = np.load(p, allow_pickle=True)
            loaded.append((np.asarray(z["target_subject_id"]),
                           np.asarray(z["target_trial_id"]),
                           np.asarray(z["probabilities"], dtype=np.float64)))
            cand_order.append(f"{regime}:{traj}")

        for subj in sorted(subjects[ds]):
            cons = sorted(t for (d, s, t) in labels if d == ds and s == subj)
            _require(len(cons) > 0, f"no construction trials for {ds} subj {subj}")
            n = len(cons)
            probs = np.empty((n, N_CANDIDATES, 2), dtype=np.float64)
            for ci, (usubj, utid, uprob) in enumerate(loaded):
                m = usubj == subj
                tid2row = {t: uprob[i] for i, t in zip(np.nonzero(m)[0], utid[m])}
                for j, t in enumerate(cons):
                    row = tid2row.get(t)
                    _require(row is not None, f"missing prediction {ds}/{subj}/{t} cand {ci}")
                    probs[j, ci] = row
            y = np.array([labels[(ds, subj, t)] for t in cons], dtype=np.int64)

            # per-trial 81-candidate contribution (Semantics-B: this context's rows)
            nll = np.empty((n, N_CANDIDATES)); correct = np.empty((n, N_CANDIDATES), dtype=np.int64)
            conf = np.empty((n, N_CANDIDATES)); cbin = np.empty((n, N_CANDIDATES), dtype=np.int64)
            scal = np.empty((n, N_CANDIDATES))
            for j in range(n):
                cr = compute_contribution(cons[j], int(y[j]), probs[j])
                nll[j] = cr.nll; correct[j] = cr.correct; conf[j] = cr.confidence
                cbin[j] = cr.conf_bin; scal[j] = cr.signed_calibration

            ctx_id = hashlib.sha256(f"{ds}|subj={subj}|panel={panel}|seed={seed}|level={level}"
                                    .encode()).hexdigest()[:24]
            meta = dict(dataset=ds, subject=subj, panel=panel, seed=seed, level=level,
                        n_trials=n, candidate_order=cand_order)
            # client-visible pool: probs only, NO labels
            np.savez(pool_root / f"{ctx_id}.npz", trial_ids=np.array(cons),
                     probabilities=probs.astype(np.float32),
                     candidate_order=np.array(cand_order), meta=json.dumps(meta))
            # server-private contribution store: label-derived
            np.savez(contrib_root / f"{ctx_id}.npz", trial_ids=np.array(cons), true_label=y,
                     nll=nll.astype(np.float32), correct=correct,
                     confidence=conf.astype(np.float32), conf_bin=cbin,
                     signed_calibration=scal.astype(np.float32),
                     candidate_order=np.array(cand_order), meta=json.dumps(meta))
            context_index.append({"context_id": ctx_id, **{k: meta[k] for k in
                                  ("dataset", "subject", "panel", "seed", "level", "n_trials")}})
            total_contexts += 1
            total_context_trials += n
            total_contributions += n * N_CANDIDATES

    # seal the label oracle (construction labels), separate from the client pool
    import shutil
    shutil.copyfile(os.path.join(LABEL_VIEW, "labels.csv"), oracle_root / "labels.csv")

    # fail-closed arithmetic invariants
    _require(total_contexts == N_CONTEXTS, f"contexts {total_contexts} != {N_CONTEXTS}")
    _require(total_context_trials == N_CONTEXT_TRIALS, f"context-trials {total_context_trials} != {N_CONTEXT_TRIALS}")
    _require(total_contributions == N_CONTRIBUTIONS, f"contributions {total_contributions} != {N_CONTRIBUTIONS}")

    manifest = {
        "gate": "C86L_C84_CONSTRUCTION_POOL_TRIAL_CONTRIBUTION_FIELD_FROZEN_C86D_PROTOCOL_REVIEW_REQUIRED",
        "authorization": "授权 C86L",
        "n_contexts": total_contexts,
        "n_candidates_per_context": N_CANDIDATES,
        "n_construction_trials": len(labels),
        "n_context_trials": total_context_trials,
        "n_contributions": total_contributions,
        "n_binary_probability_scalars": total_contributions * 2,
        "construction_labels_sha256": CONSTRUCTION_LABELS_SHA,
        "construction_evaluation_overlap": 0,
        "isolation": {"pool": "acquisition_unlabeled_pool (no labels)",
                      "oracle": "acquisition_label_oracle (sealed)",
                      "contribution": "query_contribution_store (label-derived)",
                      "held_c85u_outcome": "identity-bound only, not opened"},
        "endpoint_definition": "target_near_opt_prob = P(target 8-context mean regret <= eps)",
        "build_seconds": round(time.time() - t0, 1),
        "context_index_count": len(context_index),
    }
    (staging / "C86L_RESULT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    (staging / "C86L_CONTEXT_INDEX.json").write_text(json.dumps(context_index))
    # atomic finalize: only rename to the real root on full success
    os.replace(staging, output_root)
    manifest["output_root"] = output_root
    return manifest


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--authorization", required=True)
    a = ap.parse_args()
    if a.authorization != "授权 C86L":
        raise SystemExit("C86L build requires --authorization '授权 C86L'")
    m = build(a.output_root)
    print(json.dumps(m, indent=2))
