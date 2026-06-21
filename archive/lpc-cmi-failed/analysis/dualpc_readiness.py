"""Summarize DualPC evidence and source-only selection gates.

This script is intentionally narrow: it reads DualPC synthetic, LOSO/cross-dataset,
and lambda-selection JSON files and emits a compact table of the evidence needed for
the paper claim "control P(z) and P(y|Z) under the GLS reference measure".

Example:
  python analysis/dualpc_readiness.py \
    results/dualpc_gate_js_concept_null_quick.json \
    results/dualpc_condrw_lamsel_valid_smoke.json \
    --out-json results/dualpc_readiness_smoke.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any


def _stem(path: str) -> str:
    return os.path.basename(path).replace(".json", "")


def _get(d: dict[str, Any], *keys: str, default=None):
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return default


def _acc(v: dict[str, Any]):
    return _get(v, "target_bacc", "subject_balanced_acc", "pooled_balanced_acc",
                "per_target_balanced_acc_mean", "balanced_acc_mean")


def _delta(a, b):
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _first_method(summary: dict[str, Any], prefixes: tuple[str, ...]):
    for name, vals in summary.items():
        if isinstance(vals, dict) and any(str(name).startswith(p) for p in prefixes):
            return str(name), vals
    return None, None


def _dualpc_methods(summary: dict[str, Any]):
    out = []
    for name, vals in summary.items():
        if not isinstance(vals, dict):
            continue
        s = str(name)
        if s.startswith("dualpc:") or s == "dualpc" or s.startswith("dualpc_hinge:") or s == "dualpc_hinge":
            out.append((s, vals))
    return out


def _f(x):
    if x is None:
        return ""
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, (int, float)):
        return f"{float(x):.6g}"
    return str(x)


def _status_from_gates(group: str, v: dict[str, Any], require_target=True):
    """Read synthetic gates with null-safety tolerance.

    Positive/concept rows should improve the probe metrics relative to ERM. Null rows should
    not materially raise them; tiny finite-sample increases are not algorithmic failures.
    """
    is_null = "null" in str(group).lower()
    probe_tol = 0.005 if is_null else 0.0
    checks = []
    if require_target:
        if "gate_target_not_hurt_5pts" in v:
            checks.append(bool(v["gate_target_not_hurt_5pts"]))
        elif "delta_target_bacc_vs_erm" in v:
            checks.append(float(v["delta_target_bacc_vs_erm"]) >= -0.05)
    if "delta_cond_kl_rw_vs_erm" in v:
        checks.append(float(v["delta_cond_kl_rw_vs_erm"]) <= probe_tol)
    elif "gate_cond_rw_improved" in v:
        checks.append(bool(v["gate_cond_rw_improved"]))
    elif "gate_cond_improved" in v:
        checks.append(bool(v["gate_cond_improved"]))
    if "delta_pz_kl_rw_vs_erm" in v:
        checks.append(float(v["delta_pz_kl_rw_vs_erm"]) <= probe_tol)
    elif "gate_pz_improved" in v:
        checks.append(bool(v["gate_pz_improved"]))
    if "delta_py_js_rw_vs_erm" in v:
        checks.append(float(v["delta_py_js_rw_vs_erm"]) <= 0.005)
    elif "gate_py_js_not_raised" in v:
        checks.append(bool(v["gate_py_js_not_raised"]))
    elif "gate_py_not_raised" in v:
        checks.append(bool(v["gate_py_not_raised"]))
    if not checks:
        return "WARN", "no baseline gates"
    if all(checks):
        return "PASS", "synthetic gates pass" + (" (null tolerance)" if is_null else "")
    return "FAIL", "one or more synthetic gates fail"


def _row(path: str, suite: str, group: str, method: str, status: str, note: str,
         metrics: dict[str, Any]):
    return {
        "file": _stem(path),
        "suite": suite,
        "group": group,
        "method": method,
        "status": status,
        "note": note,
        "acc": _acc(metrics),
        "cond_kl": _get(metrics, "cond_kl", "leakage_kl"),
        "cond_kl_rw": _get(metrics, "cond_kl_rw", "leakage_kl_rw"),
        "pz_kl_rw": _get(metrics, "pz_kl_rw", "marginal_leakage_kl_rw"),
        "py_js_rw": _get(metrics, "py_js_rw_valid_mean", "py_js_rw",
                         "decoder_js_res_rw_valid_mean", "decoder_js_res_valid_mean",
                         "decoder_js_res_rw", "decoder_js_res"),
        "py_res_rw": _get(metrics, "py_res_rw_valid_mean", "py_res_rw",
                          "decoder_cmi_res_rw_valid_mean", "decoder_cmi_res_valid_mean",
                          "decoder_cmi_res_rw", "decoder_cmi_res"),
        "decoder_valid_n": _get(metrics, "decoder_valid_n"),
        "decoder_valid_frac": _get(metrics, "decoder_valid_frac", "decoder_valid"),
        "selected": _get(metrics, "selected_config", "selected"),
        "selector_penalty": _get(metrics, "selector_penalty"),
        "source_val_bacc": _get(metrics, "source_val_bacc"),
        "baseline": _get(metrics, "baseline"),
        "delta_acc": _get(metrics, "delta_acc"),
        "delta_cond_kl_rw": _get(metrics, "delta_cond_kl_rw"),
        "delta_pz_kl_rw": _get(metrics, "delta_pz_kl_rw"),
        "delta_py_js_rw": _get(metrics, "delta_py_js_rw"),
    }


def _parse_synthetic(path: str, obj: dict[str, Any]):
    rows = []
    summary = obj.get("summary", {})
    for dgp, methods in summary.items():
        if not isinstance(methods, dict):
            continue
        for method, vals in methods.items():
            if not isinstance(vals, dict):
                continue
            if not (method.startswith("dualpc") or method == "erm" or method == "dualc"):
                continue
            status, note = _status_from_gates(str(dgp), vals) if method != "erm" else ("WARN", "baseline")
            if method.endswith("_select") and vals.get("selected_hist"):
                note += f"; selected={vals['selected_hist']}"
            rows.append(_row(path, "synthetic", str(dgp), method, status, note, vals))
    return rows


def _parse_synthetic_selector(path: str, obj: dict[str, Any]):
    rows = []
    summary = obj.get("summary", {})
    for dgp, methods in summary.items():
        if not isinstance(methods, dict):
            continue
        for method, vals in methods.items():
            if not isinstance(vals, dict) or not (method.endswith("_select") or vals.get("selected_hist")):
                continue
            has_terms = all(vals.get(k) is not None for k in
                            ("cond_kl_rw", "pz_kl_rw", "py_js_rw"))
            if vals.get("selected_hist") and has_terms:
                status = "PASS"
                note = f"synthetic source-selector smoke; selected={vals['selected_hist']}"
            else:
                status = "WARN"
                note = "synthetic source-selector smoke missing selected/probe fields"
            rows.append(_row(path, "synthetic_selector", str(dgp), method, status, note, vals))
    return rows


def _parse_runner(path: str, obj: dict[str, Any]):
    rows = []
    summary = obj.get("summary", {})
    for method, vals in summary.items():
        if not isinstance(vals, dict):
            continue
        if "dualpc" not in method and method != "erm:0":
            continue
        valid_n = vals.get("decoder_valid_n")
        has_cond = vals.get("leakage_kl_rw") is not None
        has_pz = vals.get("marginal_leakage_kl_rw") is not None
        has_py = (vals.get("decoder_js_res_rw_valid_mean") is not None
                  or vals.get("decoder_js_res_valid_mean") is not None
                  or vals.get("decoder_js_res_rw") is not None)
        if method == "erm:0":
            status, note = "WARN", "baseline"
        elif valid_n == 0:
            status, note = "WARN", "decoder invalid; P(y|Z) evidence unavailable"
        elif has_cond and has_pz and has_py:
            status, note = "PASS", "fields present"
        else:
            status, note = "WARN", "missing one or more current DualPC probe fields"
        rows.append(_row(path, "runner", "summary", method, status, note, vals))
    rows.extend(_parse_runner_comparisons(path, summary))
    return rows


def _parse_runner_comparisons(path: str, summary: dict[str, Any]):
    rows = []
    baselines = []
    erm_name, erm_vals = _first_method(summary, ("erm:", "erm"))
    if erm_vals is not None:
        baselines.append((erm_name, erm_vals))
    lpc_name, lpc_vals = _first_method(summary, ("lpc_prior:", "lpc_prior"))
    if lpc_vals is not None:
        baselines.append((lpc_name, lpc_vals))

    for method, vals in _dualpc_methods(summary):
        for base_name, base in baselines:
            metrics = dict(vals)
            acc_delta = _delta(_acc(vals), _acc(base))
            cond_delta = _delta(_get(vals, "leakage_kl_rw", "cond_kl_rw"),
                                _get(base, "leakage_kl_rw", "cond_kl_rw"))
            pz_delta = _delta(_get(vals, "marginal_leakage_kl_rw", "pz_kl_rw"),
                              _get(base, "marginal_leakage_kl_rw", "pz_kl_rw"))
            py_delta = _delta(_get(vals, "decoder_js_res_rw_valid_mean", "decoder_js_res_valid_mean",
                                   "decoder_js_res_rw", "decoder_js_res", "py_js_rw_valid_mean",
                                   "py_js_rw"),
                              _get(base, "decoder_js_res_rw_valid_mean", "decoder_js_res_valid_mean",
                                   "decoder_js_res_rw", "decoder_js_res", "py_js_rw_valid_mean",
                                   "py_js_rw"))
            metrics.update(baseline=base_name, delta_acc=acc_delta,
                           delta_cond_kl_rw=cond_delta,
                           delta_pz_kl_rw=pz_delta,
                           delta_py_js_rw=py_delta)
            missing = [k for k, v in {
                "delta_acc": acc_delta,
                "delta_cond_kl_rw": cond_delta,
                "delta_pz_kl_rw": pz_delta,
                "delta_py_js_rw": py_delta,
            }.items() if v is None]
            if missing:
                status = "WARN"
                note = f"comparison missing {','.join(missing)}"
            elif acc_delta < -0.05:
                status = "FAIL"
                note = f"accuracy drops >5pts vs {base_name}"
            else:
                probe_rises = []
                if cond_delta > 0.005:
                    probe_rises.append("cond")
                if pz_delta > 0.005:
                    probe_rises.append("pz")
                if py_delta > 0.005:
                    probe_rises.append("py_js")
                if probe_rises:
                    status = "WARN"
                    note = f"probe(s) raised vs {base_name}: {','.join(probe_rises)}"
                else:
                    status = "PASS"
                    note = f"null-safety/probe comparison ok vs {base_name}"
            rows.append(_row(path, "runner_compare", base_name, method, status, note, metrics))
    return rows


def _parse_selector(path: str, obj: dict[str, Any]):
    rows = []
    for rec in obj.get("selection_records", []):
        target = str(rec.get("target", ""))
        selected = rec.get("selected")
        candidates = rec.get("candidates", [])
        valid_selected = None
        for cand in candidates:
            if cand.get("config") == selected:
                valid_selected = cand
                break
        if valid_selected is None:
            metrics = {"selected": selected}
            rows.append(_row(path, "selector", target, str(selected), "WARN",
                             "selected config missing from candidates", metrics))
            continue
        has_terms = all(valid_selected.get(k) is not None for k in
                        ("select_cond_kl_rw", "select_pz_kl_rw", "select_py_js_rw"))
        if valid_selected.get("selection_probe_valid") and has_terms:
            status, note = "PASS", "selected candidate has valid GLS/JS probes"
        elif not valid_selected.get("selection_probe_valid"):
            status, note = "WARN", "selected candidate probe invalid"
        else:
            status, note = "WARN", "selected candidate missing GLS/JS terms"
        metrics = {
            "selected": selected,
            "selector_penalty": valid_selected.get("selector_penalty"),
            "source_val_bacc": valid_selected.get("source_val_bacc"),
            "cond_kl": valid_selected.get("select_cond_kl"),
            "cond_kl_rw": valid_selected.get("select_cond_kl_rw"),
            "pz_kl_rw": valid_selected.get("select_pz_kl_rw"),
            "py_js_rw": valid_selected.get("select_py_js_rw"),
            "py_res_rw": valid_selected.get("select_py_res_rw"),
            "decoder_valid_frac": valid_selected.get("select_decoder_valid"),
        }
        rows.append(_row(path, "selector", target, str(selected), status, note, metrics))
        if "final_selected_probe_valid" in rec:
            final_metrics = {
                "selected": selected,
                "selector_penalty": rec.get("final_selected_probe_penalty"),
                "source_val_bacc": rec.get("source_val_bacc"),
                "cond_kl": rec.get("final_selected_cond_kl"),
                "cond_kl_rw": rec.get("final_selected_cond_kl_rw"),
                "pz_kl_rw": rec.get("final_selected_pz_kl_rw"),
                "py_js_rw": rec.get("final_selected_py_js_rw"),
                "py_res_rw": rec.get("final_selected_py_res_rw"),
                "decoder_valid_frac": rec.get("final_selected_decoder_valid"),
                "target_bacc": rec.get("target_bacc"),
            }
            has_final_terms = all(final_metrics.get(k) is not None for k in
                                  ("cond_kl_rw", "pz_kl_rw", "py_js_rw"))
            if rec.get("final_selected_probe_valid") and has_final_terms:
                final_status = "PASS"
                final_note = "final retrained selected model has valid GLS/JS probes"
            elif not rec.get("final_selected_probe_valid"):
                final_status = "WARN"
                final_note = "final retrained selected model probe invalid"
            else:
                final_status = "WARN"
                final_note = "final retrained selected model missing GLS/JS terms"
            rows.append(_row(path, "selector_final", target, str(selected),
                             final_status, final_note, final_metrics))
    return rows


def parse_file(path: str):
    obj = json.load(open(path))
    rows = []
    if isinstance(obj.get("summary"), dict):
        # Synthetic summaries are nested as dgp -> method -> metrics.
        nested = any(isinstance(v, dict) and any(isinstance(x, dict) for x in v.values())
                     for v in obj["summary"].values())
        if nested and "rows" in obj:
            if obj.get("config", {}).get("source_select"):
                rows.extend(_parse_synthetic_selector(path, obj))
            else:
                rows.extend(_parse_synthetic(path, obj))
        else:
            rows.extend(_parse_runner(path, obj))
    if "selection_records" in obj:
        rows.extend(_parse_selector(path, obj))
    return rows


def print_table(rows):
    cols = ["status", "suite", "file", "group", "method", "acc", "cond_kl_rw",
            "pz_kl_rw", "py_js_rw", "baseline", "delta_acc", "delta_cond_kl_rw",
            "delta_pz_kl_rw", "delta_py_js_rw", "decoder_valid_n", "selected",
            "selector_penalty", "note"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(_f(r.get(c)) for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="DualPC JSON files. Default: results/dualpc*.json")
    ap.add_argument("--out-json", default="", help="Optional JSON rows output")
    ap.add_argument("--only-problems", action="store_true", help="Only print WARN/FAIL rows")
    args = ap.parse_args()

    raw_files = args.files or ["results/dualpc*.json"]
    files = []
    for item in raw_files:
        matches = sorted(glob.glob(item))
        files.extend(matches if matches else [item])
    rows = []
    for path in files:
        try:
            rows.extend(parse_file(path))
        except Exception as exc:
            rows.append({
                "file": _stem(path), "suite": "parse", "group": "", "method": "",
                "status": "FAIL", "note": f"parse error: {exc}",
            })
    if args.only_problems:
        rows = [r for r in rows if r.get("status") != "PASS"]
    print_table(rows)
    counts = {k: sum(1 for r in rows if r.get("status") == k) for k in ("PASS", "WARN", "FAIL")}
    print(f"\n# counts: PASS={counts['PASS']} WARN={counts['WARN']} FAIL={counts['FAIL']}")
    if args.out_json:
        json.dump({"counts": counts, "rows": rows}, open(args.out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
