"""C39 selected-vs-better point atom decomposition."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np

from ..selector_trace_recovery import artifact_loader as c37_loader
from . import artifact_loader as al
from . import atom_replay, schema


def _unit_key(job):
    return (job.seed, job.target, job.level, job.regime)


def _jobs_by_unit(jobs):
    out = defaultdict(list)
    for job in jobs:
        out[_unit_key(job)].append(job)
    return dict(out)


def _replay_unit(job_dicts):
    jobs = [al.CandidateReplayJob(**j) for j in job_dicts]
    first = jobs[0]
    trace = c37_loader.load_c10_trace([first.regime])
    local = {"trace": trace, "ctx_cache": c37_loader.ContextCache(trace)}
    rows = []
    atom_rows = []
    for job in jobs:
        ctx = al.unit_context(local, job)
        support_graph, fold_plan = al.support_and_fold(ctx, split=job.split)
        feat_art = al.feature_by_hash(ctx, job.model_hash, split=job.split)
        replay = atom_replay.replay_point_atoms(
            feat_art.features, support_graph, fold_plan, ctx.fold.execution_config.critic,
            expected_point=job.expected_point)
        bootstrap_hash = ctx.bootstrap_plan.plan_hash if job.split == "selection" else ""
        row = {
            "job_key": job.job_key,
            "seed": job.seed,
            "target": job.target,
            "level": job.level,
            "regime": job.regime,
            "pair_key": job.pair_key,
            "pair_id": job.pair_id,
            "candidate_role": job.candidate_role,
            "candidate_order": job.candidate_order,
            "candidate_id": job.candidate_id,
            "split": job.split,
            "expected_point": job.expected_point,
            "recomputed_point": replay["aggregate_point"],
            "point_abs_diff": replay["point_abs_diff"],
            "selected_capacity": replay["selected_capacity"],
            "atom_sum": replay["atom_sum"],
            "additive_abs_diff": replay["additive_abs_diff"],
            "max_class_mass_abs_diff": replay["max_class_mass_abs_diff"],
            "identity_pass": replay["identity_pass"],
            "support_graph_hash": support_graph.support_hash(),
            "fold_plan_hash": fold_plan.plan_hash,
            "bootstrap_plan_hash": bootstrap_hash,
            "population_hash": fold_plan.population_hash,
            "feature_population_hash_matches": int(feat_art.population_hash == fold_plan.population_hash),
            "target_labels_loaded_for_replay": replay["target_labels_loaded_for_replay"],
            "n_atoms": len(replay["atoms"]),
        }
        rows.append(row)
        for a in replay["atoms"]:
            ar = dict(a)
            ar.update({
                "job_key": job.job_key,
                "seed": job.seed,
                "target": job.target,
                "level": job.level,
                "regime": job.regime,
                "pair_key": job.pair_key,
                "pair_id": job.pair_id,
                "candidate_role": job.candidate_role,
                "candidate_order": job.candidate_order,
                "candidate_id": job.candidate_id,
                "split": job.split,
                "selected_capacity": replay["selected_capacity"],
            })
            atom_rows.append(ar)
    return {"identity_rows": rows, "atom_rows": atom_rows}


def replay_all(ctx, *, n_jobs=1):
    jobs = al.candidate_jobs(ctx, split="selection") + al.candidate_jobs(ctx, split="source_audit")
    units = list(_jobs_by_unit(jobs).values())
    identity_rows = []
    atom_rows = []
    if int(n_jobs) <= 1:
        for unit in units:
            part = _replay_unit([asdict(j) for j in unit])
            identity_rows.extend(part["identity_rows"])
            atom_rows.extend(part["atom_rows"])
    else:
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
            futs = [ex.submit(_replay_unit, [asdict(j) for j in unit]) for unit in units]
            for fut in as_completed(futs):
                part = fut.result()
                identity_rows.extend(part["identity_rows"])
                atom_rows.extend(part["atom_rows"])
    identity_rows.sort(key=lambda r: (r["split"], int(r["seed"]), int(r["target"]), int(r["level"]),
                                      r["candidate_role"], int(r["candidate_order"])))
    atom_rows.sort(key=lambda r: (r["split"], int(r["seed"]), int(r["target"]), int(r["level"]),
                                  r["candidate_role"], int(r["candidate_order"]),
                                  int(r["class_id"]), int(r["domain_id"])))
    return {"identity_rows": identity_rows, "atom_rows": atom_rows}


def _atom_index(atom_rows, split):
    out = {}
    for r in atom_rows:
        if r["split"] != split:
            continue
        out[(r["job_key"], r["atom_id"])] = r
    return out


def selected_vs_better_atoms(ctx, atom_rows, *, split="selection"):
    idx = _atom_index(atom_rows, split)
    rows = []
    for pair in ctx["pairs"]:
        selected_key = "|".join([pair["seed"], pair["target"], pair["level"],
                                 pair["selected_order"], "selected", split])
        better_key = "|".join([pair["seed"], pair["target"], pair["level"],
                               pair["better_order"], "better", split])
        atom_ids = sorted({aid for (jk, aid) in idx if jk in (selected_key, better_key)})
        for atom_id in atom_ids:
            s = idx.get((selected_key, atom_id))
            b = idx.get((better_key, atom_id))
            if s is None or b is None:
                continue
            delta = float(b["atom_value"]) - float(s["atom_value"])
            rows.append({
                "pair_id": pair["pair_id"],
                "pair_key": pair["pair_key"],
                "seed": pair["seed"],
                "target": pair["target"],
                "level": pair["level"],
                "regime": pair["regime"],
                "selected_order": pair["selected_order"],
                "better_order": pair["better_order"],
                "split": split,
                "atom_id": atom_id,
                "class_id": s["class_id"],
                "class_name": s["class_name"],
                "domain_id": s["domain_id"],
                "domain_name": s["domain_name"],
                "selected_atom": float(s["atom_value"]),
                "better_atom": float(b["atom_value"]),
                "atom_delta_better_minus_selected": delta,
                "positive_selected_advantage": max(delta, 0.0),
                "selected_advantage_sign": (
                    "selected" if delta > schema.ATOM_DELTA_EPS else
                    "better" if delta < -schema.ATOM_DELTA_EPS else "flat"),
                "selected_point": al.as_float(pair["selected_point"]),
                "better_point": al.as_float(pair["better_point"]),
                "point_delta_better_minus_selected": al.as_float(pair["point_delta_better_minus_selected"]),
                "atom_fraction_of_point_delta": (
                    delta / al.as_float(pair["point_delta_better_minus_selected"])
                    if abs(al.as_float(pair["point_delta_better_minus_selected"])) > 0 else np.nan),
                "support_count": s["support_count"],
                "support_m": s["support_m"],
                "cell_mass": s["cell_mass"],
                "class_overlap_mass": s["class_overlap_mass"],
                "p_ref_y": s["p_ref_y"],
                "p_d_given_y": s["p_d_given_y"],
                "eligible": s["eligible"],
                "present": s["present"],
                "support_edge": s["support_edge"],
                "selected_oof_mass": s["oof_mass"],
                "better_oof_mass": b["oof_mass"],
                "selected_capacity": s["selected_capacity"],
                "better_capacity": b["selected_capacity"],
            })
    rows.sort(key=lambda r: (r["pair_id"], int(r["class_id"]), int(r["domain_id"])))
    totals = defaultdict(float)
    for r in rows:
        totals[r["pair_id"]] += float(r["positive_selected_advantage"])
    for r in rows:
        denom = totals[r["pair_id"]]
        r["positive_advantage_share"] = (
            float(r["positive_selected_advantage"]) / denom if denom > 0 else 0.0)
    return rows


def concentration_summary(point_atom_rows):
    by_pair = defaultdict(list)
    for r in point_atom_rows:
        by_pair[r["pair_id"]].append(r)
    rows = []
    for pair_id, rs in sorted(by_pair.items()):
        positives = sorted([float(r["positive_selected_advantage"]) for r in rs if
                            float(r["positive_selected_advantage"]) > schema.ATOM_DELTA_EPS], reverse=True)
        pos_sum = float(sum(positives))
        shares = [v / pos_sum for v in positives] if pos_sum > 0 else []
        top1 = shares[0] if shares else 0.0
        top3 = float(sum(shares[:3])) if shares else 0.0
        top5 = float(sum(shares[:5])) if shares else 0.0
        hhi = float(sum(s * s for s in shares)) if shares else 0.0
        concentrated = int(top3 >= schema.CONCENTRATED_TOP3_SHARE_GATE or
                           hhi >= schema.CONCENTRATED_HHI_GATE)
        broad = int(top3 < schema.BROAD_TOP3_SHARE_GATE and
                    len(positives) >= schema.BROAD_MIN_POSITIVE_ATOMS and
                    hhi < schema.BROAD_HHI_GATE)
        first = rs[0]
        rows.append({
            "pair_id": pair_id,
            "pair_key": first["pair_key"],
            "seed": first["seed"],
            "target": first["target"],
            "level": first["level"],
            "regime": first["regime"],
            "selected_order": first["selected_order"],
            "better_order": first["better_order"],
            "n_atoms": len(rs),
            "n_positive_atoms": len(positives),
            "positive_advantage_sum": pos_sum,
            "top1_positive_share": top1,
            "top3_positive_share": top3,
            "top5_positive_share": top5,
            "positive_hhi": hhi,
            "concentrated_flag": concentrated,
            "broad_flag": broad,
            "concentration_class": "cell_concentrated" if concentrated else
                                   "broad" if broad else "mixed",
        })
    return rows


def class_domain_contributions(point_atom_rows):
    scopes = [("class", "class_id", "class_name"), ("domain", "domain_id", "domain_name"),
              ("cell", "atom_id", None)]
    rows = []
    total_positive = float(sum(float(r["positive_selected_advantage"]) for r in point_atom_rows))
    for scope, key, label_key in scopes:
        grouped = defaultdict(list)
        for r in point_atom_rows:
            grouped[str(r[key])].append(r)
        for k, rs in grouped.items():
            positive = float(sum(float(r["positive_selected_advantage"]) for r in rs))
            signed = float(sum(float(r["atom_delta_better_minus_selected"]) for r in rs))
            label = rs[0][label_key] if label_key else k
            rows.append({
                "scope": scope,
                "atom_key": k,
                "label": label,
                "n_rows": len(rs),
                "positive_selected_advantage_sum": positive,
                "signed_delta_sum": signed,
                "positive_selected_advantage_share": (
                    positive / total_positive if total_positive > 0 else 0.0),
                "mean_atom_delta_better_minus_selected": signed / len(rs) if rs else 0.0,
            })
    rows.sort(key=lambda r: (r["scope"], -float(r["positive_selected_advantage_share"]), r["atom_key"]))
    return rows


def summaries(identity_rows, point_atom_rows):
    selection_identity = [r for r in identity_rows if r["split"] == "selection"]
    audit_identity = [r for r in identity_rows if r["split"] == "source_audit"]
    conc = concentration_summary(point_atom_rows)
    return {
        "identity": {
            "n_selection_candidates": len(selection_identity),
            "n_selection_identity_pass": sum(int(r["identity_pass"]) for r in selection_identity),
            "selection_identity_pass": bool(selection_identity and
                                            all(int(r["identity_pass"]) for r in selection_identity)),
            "n_source_audit_candidates": len(audit_identity),
            "n_source_audit_additive_pass": sum(float(r["additive_abs_diff"]) <= schema.ATOM_ADDITIVE_TOL
                                                for r in audit_identity),
            "source_audit_additive_pass": bool(audit_identity and
                                               all(float(r["additive_abs_diff"]) <= schema.ATOM_ADDITIVE_TOL
                                                   for r in audit_identity)),
            "max_selection_point_abs_diff": max([float(r["point_abs_diff"]) for r in selection_identity
                                                 if al.finite(r["point_abs_diff"])], default=None),
            "max_selection_additive_abs_diff": max([float(r["additive_abs_diff"])
                                                    for r in selection_identity], default=None),
            "max_audit_additive_abs_diff": max([float(r["additive_abs_diff"])
                                                for r in audit_identity], default=None),
        },
        "concentration": {
            "n_pairs": len(conc),
            "concentrated_pair_count": sum(int(r["concentrated_flag"]) for r in conc),
            "broad_pair_count": sum(int(r["broad_flag"]) for r in conc),
            "concentrated_pair_fraction": (
                sum(int(r["concentrated_flag"]) for r in conc) / len(conc) if conc else None),
            "broad_pair_fraction": (
                sum(int(r["broad_flag"]) for r in conc) / len(conc) if conc else None),
            "mean_top3_positive_share": al.finite_mean([r["top3_positive_share"] for r in conc]),
            "mean_positive_hhi": al.finite_mean([r["positive_hhi"] for r in conc]),
        },
    }
