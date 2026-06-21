"""Aggregate DualPC paper-profile JSONs into auditable method/comparison tables.

This is the cross-run companion to ``analysis/dualpc_readiness.py``. Readiness is
row-oriented and flags each JSON. This script groups runner and source-selector
outputs across seeds/files so the paper table can be checked for the DualPC claim:
accuracy parity plus lower GLS P(z) leakage and no raised JS P(Y|Z) leakage.

Example:
  python analysis/dualpc_paper_summary.py 'results/dualpc_protocol/*.json' \
    --out-json results/dualpc_protocol/dualpc_paper_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any


SCPS_DATASETS = {"ADFTD", "ADFTD_bin", "MUMTAZ", "TUAB"}


def _stem(path: str) -> str:
    return os.path.basename(path).replace(".json", "")


def _get(d: dict[str, Any], *keys: str, default=None):
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return default


def _num(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _mean(vals):
    vals = [_num(v) for v in vals]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _std(vals):
    vals = [_num(v) for v in vals]
    vals = [v for v in vals if v is not None]
    if len(vals) <= 1:
        return 0.0 if vals else None
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))


def _fmt(x):
    if x is None:
        return ""
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, (int, float)):
        return f"{float(x):.6g}"
    return str(x)


def _method_family(method: str) -> str:
    family = str(method).split(":", 1)[0]
    return "dualpc" if family == "dualpc_hinge" else family


def _is_runner_summary(obj: dict[str, Any]) -> bool:
    summary = obj.get("summary")
    if not isinstance(summary, dict):
        return False
    # Runner summaries are method -> metrics. Some runner metric dicts contain
    # nested diagnostic maps (e.g. SCPS per_cohort), so do not classify by
    # "contains a nested dict". Synthetic summaries use DGP names at top level.
    families = {_method_family(str(k)) for k in summary}
    runner_families = {
        "erm", "lpc_prior", "dualc", "dualpc", "dualpc_hinge", "dualpc_marginal",
        "dual", "marginal", "chain", "lpc_uniform",
    }
    return bool(families & runner_families) and any(isinstance(v, dict) for v in summary.values())


def _task_label(path: str, obj: dict[str, Any]) -> str:
    cfg = obj.get("config", {}) if isinstance(obj.get("config"), dict) else {}
    if cfg.get("dataset"):
        return f"loso:{cfg['dataset']}"
    condition = cfg.get("condition") or obj.get("condition")
    if condition:
        domain = cfg.get("domain") or obj.get("domain")
        dec_domain = cfg.get("dec_domain") or obj.get("dec_domain")
        if domain or dec_domain:
            return f"scps:{condition}:D={domain or '?'}:decD={dec_domain or domain or '?'}"
        return f"scps:{condition}"
    return _stem(path)


def _paper_acc(vals: dict[str, Any], obj: dict[str, Any]):
    cfg = obj.get("config", {}) if isinstance(obj.get("config"), dict) else {}
    dataset = str(cfg.get("dataset", ""))
    if cfg.get("condition") or obj.get("condition"):
        return _num(_get(vals, "per_target_balanced_acc_mean", "balanced_acc_mean",
                         "subject_balanced_acc", "pooled_balanced_acc"))
    if dataset in SCPS_DATASETS:
        return _num(_get(vals, "subject_balanced_acc", "pooled_balanced_acc",
                         "per_target_balanced_acc_mean", "balanced_acc_mean"))
    return _num(_get(vals, "per_target_balanced_acc_mean", "balanced_acc_mean",
                     "pooled_balanced_acc", "subject_balanced_acc"))


def _metrics(vals: dict[str, Any], obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "acc": _paper_acc(vals, obj),
        "cond_kl_rw": _num(_get(vals, "leakage_kl_rw", "cond_kl_rw")),
        "pz_kl_rw": _num(_get(vals, "marginal_leakage_kl_rw", "pz_kl_rw")),
        "py_js_rw": _num(_get(vals, "decoder_js_res_rw_valid_mean",
                              "decoder_js_res_valid_mean", "decoder_js_res_rw",
                              "decoder_js_res", "py_js_rw_valid_mean", "py_js_rw")),
        "decoder_valid_n": _num(_get(vals, "decoder_valid_n")),
        "n_folds": _num(_get(vals, "n_folds")),
    }


def _delta(a, b):
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _comparison_status(row: dict[str, Any], acc_floor: float, probe_tol: float):
    missing = [k for k in ("delta_acc", "delta_cond_kl_rw", "delta_pz_kl_rw", "delta_py_js_rw")
               if row.get(k) is None]
    if missing:
        return "WARN", "missing " + ",".join(missing)
    if row["delta_acc"] < acc_floor:
        return "FAIL", "accuracy below floor"
    raised = []
    if row["delta_cond_kl_rw"] > probe_tol:
        raised.append("cond")
    if row["delta_pz_kl_rw"] > probe_tol:
        raised.append("pz")
    if row["delta_py_js_rw"] > probe_tol:
        raised.append("py_js")
    if raised:
        return "WARN", "probe(s) raised: " + ",".join(raised)
    return "PASS", "accuracy/probe deltas ok"


def _method_rows(path: str, obj: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    task = _task_label(path, obj)
    cfg = obj.get("config", {}) if isinstance(obj.get("config"), dict) else {}
    for method, vals in obj.get("summary", {}).items():
        if not isinstance(vals, dict):
            continue
        row = {
            "file": _stem(path),
            "task": task,
            "seed": cfg.get("seed"),
            "method": str(method),
            "family": _method_family(str(method)),
        }
        row.update(_metrics(vals, obj))
        rows.append(row)
    return rows


def _runner_comparisons(path: str, obj: dict[str, Any], acc_floor: float, probe_tol: float):
    summary = obj.get("summary", {})
    baselines = []
    for method, vals in summary.items():
        fam = _method_family(str(method))
        if fam in {"erm", "lpc_prior"} and isinstance(vals, dict):
            baselines.append((str(method), vals))
    rows = []
    task = _task_label(path, obj)
    cfg = obj.get("config", {}) if isinstance(obj.get("config"), dict) else {}
    for method, vals in summary.items():
        if _method_family(str(method)) != "dualpc" or not isinstance(vals, dict):
            continue
        m = _metrics(vals, obj)
        for baseline, base_vals in baselines:
            b = _metrics(base_vals, obj)
            row = {
                "file": _stem(path),
                "task": task,
                "seed": cfg.get("seed"),
                "method": str(method),
                "baseline": baseline,
                "acc": m["acc"],
                "baseline_acc": b["acc"],
                "delta_acc": _delta(m["acc"], b["acc"]),
                "delta_cond_kl_rw": _delta(m["cond_kl_rw"], b["cond_kl_rw"]),
                "delta_pz_kl_rw": _delta(m["pz_kl_rw"], b["pz_kl_rw"]),
                "delta_py_js_rw": _delta(m["py_js_rw"], b["py_js_rw"]),
            }
            row["status"], row["note"] = _comparison_status(row, acc_floor, probe_tol)
            rows.append(row)
    return rows


def _selector_records(path: str, obj: dict[str, Any]) -> list[dict[str, Any]]:
    task = _task_label(path, obj)
    cfg = obj.get("config", {}) if isinstance(obj.get("config"), dict) else {}
    rows = []
    for rec in obj.get("selection_records", []):
        if not isinstance(rec, dict):
            continue
        selected = rec.get("selected")
        candidates = rec.get("candidates", [])
        selected_candidate = next((c for c in candidates if c.get("config") == selected), {})
        has_final = "final_selected_probe_valid" in rec
        prefix = "final_selected" if has_final else "select"
        valid = rec.get("final_selected_probe_valid") if has_final \
            else selected_candidate.get("selection_probe_valid")
        row = {
            "file": _stem(path),
            "task": task,
            "seed": cfg.get("seed"),
            "target": rec.get("target"),
            "selected": selected,
            "target_bacc": _num(rec.get("target_bacc")),
            "target_erm_bacc": _num(rec.get("target_erm_bacc")),
            "delta_target_bacc_vs_erm": _delta(_num(rec.get("target_bacc")),
                                                _num(rec.get("target_erm_bacc"))),
            "probe_valid": bool(valid),
            "selector_penalty": _num(rec.get(f"{prefix}_probe_penalty")
                                     if has_final else selected_candidate.get("selector_penalty")),
            "cond_kl_rw": _num(rec.get(f"{prefix}_cond_kl_rw")
                               if has_final else selected_candidate.get("select_cond_kl_rw")),
            "pz_kl_rw": _num(rec.get(f"{prefix}_pz_kl_rw")
                             if has_final else selected_candidate.get("select_pz_kl_rw")),
            "py_js_rw": _num(rec.get(f"{prefix}_py_js_rw")
                             if has_final else selected_candidate.get("select_py_js_rw")),
            "final": bool(has_final),
        }
        rows.append(row)
    return rows


def _aggregate_methods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["method"])].append(row)
    out = []
    for (task, method), vals in sorted(groups.items()):
        out.append({
            "task": task,
            "method": method,
            "family": _method_family(method),
            "n": len(vals),
            "files": len({v["file"] for v in vals}),
            "acc_mean": _mean(v.get("acc") for v in vals),
            "acc_std": _std(v.get("acc") for v in vals),
            "cond_kl_rw_mean": _mean(v.get("cond_kl_rw") for v in vals),
            "pz_kl_rw_mean": _mean(v.get("pz_kl_rw") for v in vals),
            "py_js_rw_mean": _mean(v.get("py_js_rw") for v in vals),
            "decoder_valid_n_mean": _mean(v.get("decoder_valid_n") for v in vals),
        })
    return out


def _aggregate_comparisons(rows: list[dict[str, Any]], acc_floor: float, probe_tol: float):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["method"], row["baseline"])].append(row)
    out = []
    for (task, method, baseline), vals in sorted(groups.items()):
        row = {
            "task": task,
            "method": method,
            "baseline": baseline,
            "n": len(vals),
            "files": len({v["file"] for v in vals}),
            "pass_rate": sum(v["status"] == "PASS" for v in vals) / len(vals),
            "acc_mean": _mean(v.get("acc") for v in vals),
            "baseline_acc_mean": _mean(v.get("baseline_acc") for v in vals),
            "delta_acc_mean": _mean(v.get("delta_acc") for v in vals),
            "delta_acc_std": _std(v.get("delta_acc") for v in vals),
            "delta_cond_kl_rw_mean": _mean(v.get("delta_cond_kl_rw") for v in vals),
            "delta_pz_kl_rw_mean": _mean(v.get("delta_pz_kl_rw") for v in vals),
            "delta_py_js_rw_mean": _mean(v.get("delta_py_js_rw") for v in vals),
        }
        status_input = {
            "delta_acc": row["delta_acc_mean"],
            "delta_cond_kl_rw": row["delta_cond_kl_rw_mean"],
            "delta_pz_kl_rw": row["delta_pz_kl_rw_mean"],
            "delta_py_js_rw": row["delta_py_js_rw_mean"],
        }
        row["status"], row["note"] = _comparison_status(status_input, acc_floor, probe_tol)
        if any(v["status"] == "FAIL" for v in vals):
            row["status"] = "FAIL"
            row["note"] = "one or more runs fail"
        elif any(v["status"] == "WARN" for v in vals) and row["status"] == "PASS":
            row["status"] = "WARN"
            row["note"] = "one or more runs warn"
        out.append(row)
    return out


def _aggregate_selectors(rows: list[dict[str, Any]], acc_floor: float):
    groups = defaultdict(list)
    for row in rows:
        groups[row["task"]].append(row)
    out = []
    for task, vals in sorted(groups.items()):
        final_vals = [v for v in vals if v.get("final")]
        final_record_frac = len(final_vals) / len(vals)
        final_probe_valid_frac = sum(v.get("probe_valid") for v in final_vals) / len(vals)
        selected_hist = Counter(str(v.get("selected")) for v in vals)
        row = {
            "task": task,
            "n": len(vals),
            "files": len({v["file"] for v in vals}),
            "selected_hist": dict(sorted(selected_hist.items())),
            "target_bacc_mean": _mean(v.get("target_bacc") for v in final_vals),
            "target_bacc_std": _std(v.get("target_bacc") for v in final_vals),
            "delta_target_bacc_vs_erm_mean": _mean(v.get("delta_target_bacc_vs_erm") for v in final_vals),
            "final_record_frac": final_record_frac,
            "final_probe_valid_frac": final_probe_valid_frac,
            "selector_penalty_mean": _mean(v.get("selector_penalty") for v in final_vals),
            "cond_kl_rw_mean": _mean(v.get("cond_kl_rw") for v in final_vals),
            "pz_kl_rw_mean": _mean(v.get("pz_kl_rw") for v in final_vals),
            "py_js_rw_mean": _mean(v.get("py_js_rw") for v in final_vals),
        }
        missing = [k for k in ("cond_kl_rw_mean", "pz_kl_rw_mean", "py_js_rw_mean")
                   if row[k] is None]
        if final_record_frac < 1.0:
            row["status"], row["note"] = "WARN", "selector final retrain probes missing"
        elif missing or final_probe_valid_frac < 1.0:
            row["status"], row["note"] = "WARN", "selector final probes incomplete"
        elif row["delta_target_bacc_vs_erm_mean"] is not None \
                and row["delta_target_bacc_vs_erm_mean"] < acc_floor:
            row["status"], row["note"] = "FAIL", "selected final accuracy below ERM floor"
        else:
            row["status"], row["note"] = "PASS", "selector final probes complete"
        out.append(row)
    return out


def summarize_paths(paths: list[str], acc_floor: float = -0.05, probe_tol: float = 0.005):
    method_rows = []
    comparison_rows = []
    selector_records = []
    parse_errors = []
    for path in paths:
        try:
            obj = json.load(open(path))
        except Exception as exc:
            parse_errors.append({"file": _stem(path), "status": "FAIL", "note": f"parse error: {exc}"})
            continue
        if _is_runner_summary(obj):
            method_rows.extend(_method_rows(path, obj))
            comparison_rows.extend(_runner_comparisons(path, obj, acc_floor, probe_tol))
        if isinstance(obj.get("selection_records"), list):
            selector_records.extend(_selector_records(path, obj))

    method_summary = _aggregate_methods(method_rows)
    comparison_summary = _aggregate_comparisons(comparison_rows, acc_floor, probe_tol)
    selector_summary = _aggregate_selectors(selector_records, acc_floor)
    status_rows = comparison_summary + selector_summary + parse_errors
    counts = {k: sum(1 for r in status_rows if r.get("status") == k) for k in ("PASS", "WARN", "FAIL")}
    return {
        "counts": counts,
        "method_rows": method_rows,
        "method_summary": method_summary,
        "comparison_rows": comparison_rows,
        "comparison_summary": comparison_summary,
        "selector_records": selector_records,
        "selector_summary": selector_summary,
        "parse_errors": parse_errors,
    }


def _print_section(title: str, cols: list[str], rows: list[dict[str, Any]]):
    print(f"\n# {title}")
    print("\t".join(cols))
    for row in rows:
        print("\t".join(_fmt(row.get(c)) for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="DualPC runner/selector JSONs. Default: results/dualpc_protocol/*.json")
    ap.add_argument("--out-json", default="", help="Optional aggregate JSON path")
    ap.add_argument("--only-problems", action="store_true", help="Only print WARN/FAIL comparison/selector rows")
    ap.add_argument("--acc-floor", type=float, default=-0.05,
                    help="Allowed mean DualPC accuracy delta vs baseline/ERM before FAIL")
    ap.add_argument("--probe-tol", type=float, default=0.005,
                    help="Allowed positive mean probe delta before WARN")
    args = ap.parse_args()

    raw_files = args.files or ["results/dualpc_protocol/*.json"]
    paths = []
    for item in raw_files:
        matches = sorted(glob.glob(item))
        paths.extend(matches if matches else [item])

    out = summarize_paths(paths, acc_floor=args.acc_floor, probe_tol=args.probe_tol)
    method_rows = out["method_summary"]
    comparison_rows = out["comparison_summary"]
    selector_rows = out["selector_summary"]
    if args.only_problems:
        method_rows = []
        comparison_rows = [r for r in comparison_rows if r.get("status") != "PASS"]
        selector_rows = [r for r in selector_rows if r.get("status") != "PASS"]

    _print_section("method_summary",
                   ["task", "method", "n", "files", "acc_mean", "acc_std",
                    "cond_kl_rw_mean", "pz_kl_rw_mean", "py_js_rw_mean", "decoder_valid_n_mean"],
                   method_rows)
    _print_section("comparison_summary",
                   ["status", "task", "method", "baseline", "n", "files", "pass_rate",
                    "acc_mean", "baseline_acc_mean", "delta_acc_mean", "delta_acc_std",
                    "delta_cond_kl_rw_mean", "delta_pz_kl_rw_mean", "delta_py_js_rw_mean", "note"],
                   comparison_rows)
    _print_section("selector_summary",
                   ["status", "task", "n", "files", "target_bacc_mean", "target_bacc_std",
                    "delta_target_bacc_vs_erm_mean", "final_record_frac", "final_probe_valid_frac",
                    "selector_penalty_mean", "cond_kl_rw_mean", "pz_kl_rw_mean",
                    "py_js_rw_mean", "selected_hist", "note"],
                   selector_rows)
    for err in out["parse_errors"]:
        print(f"\n# parse_error\t{err['file']}\t{err['note']}")
    if args.only_problems:
        status_rows = comparison_rows + selector_rows + out["parse_errors"]
        counts = {k: sum(1 for r in status_rows if r.get("status") == k) for k in ("PASS", "WARN", "FAIL")}
    else:
        counts = out["counts"]
    print(f"\n# counts: PASS={counts['PASS']} WARN={counts['WARN']} FAIL={counts['FAIL']}")
    if args.out_json:
        json.dump(out, open(args.out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
