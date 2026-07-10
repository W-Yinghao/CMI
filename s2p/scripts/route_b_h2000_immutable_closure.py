#!/usr/bin/env python
"""Fail-closed immutable closure for completed Route-B H2000 checkpoints.

This script performs no training and no downstream evaluation. It validates the
two completed H2000 runs, copies their selected checkpoints to SHA-named,
read-only paths, and emits an auditable closure manifest.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, os.path.expanduser("~/eeg2025/CBraMod"))
from models.cbramod import CBraMod


CELLS = {"H2000_s0": 0, "H2000_s1": 1}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(obj):
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path):
    with Path(path).open() as f:
        return json.load(f)


def read_jsonl(path):
    rows = []
    with Path(path).open() as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
    return rows


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_csv(path, rows):
    path = Path(path)
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def require_jobs_absent(job_ids):
    if not job_ids:
        raise RuntimeError("closure requires explicit parent job ids")
    proc = subprocess.run(
        ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i"],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cannot verify parent jobs are absent: {proc.stderr.strip()}")
    live = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if live:
        raise RuntimeError(f"parent training jobs still live: {live}")


def strict_reload(checkpoint_path):
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "model_state" not in obj:
        raise RuntimeError(f"checkpoint contract missing model_state: {checkpoint_path}")
    model = CBraMod(
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=30,
        n_layer=12,
        nhead=8,
    )
    model.load_state_dict(obj["model_state"], strict=True)
    return obj


def validate_cell(source_root, tag, seed):
    cell_dir = Path(source_root) / tag
    paths = {
        "checkpoint": cell_dir / "best.pth",
        "summary": cell_dir / "run_summary.json",
        "log": cell_dir / "train_log.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"{tag} missing completion artifacts: {missing}")

    summary = read_json(paths["summary"])
    events = read_jsonl(paths["log"])
    epoch_events = [row for row in events if row.get("event") == "epoch"]
    done_events = [row for row in events if row.get("event") == "done"]
    data_events = [row for row in events if row.get("event") == "data"]
    if [row.get("epoch") for row in epoch_events] != list(range(1, 51)):
        raise RuntimeError(f"{tag} epoch log is not exactly 1..50")
    if len(done_events) != 1 or done_events[0] != summary:
        raise RuntimeError(f"{tag} done marker missing or differs from run_summary.json")
    if len(data_events) != 1:
        raise RuntimeError(f"{tag} requires exactly one data event")
    for row in epoch_events:
        if not all(math.isfinite(float(row[key])) for key in ("train_loss", "val_loss")):
            raise RuntimeError(f"{tag} has non-finite loss at epoch {row.get('epoch')}")

    required_summary = {
        "event": "done",
        "route": "B_33ch_cbramod_only",
        "budget_h": 2000.0,
        "subset_seed": seed,
        "init_seed": seed,
        "epochs": 50,
        "checkpoint_strict_reload": True,
        "target_labels_used": False,
        "smoke": False,
    }
    for key, expected in required_summary.items():
        if summary.get(key) != expected:
            raise RuntimeError(f"{tag} summary {key}={summary.get(key)!r}, expected {expected!r}")
    route_manifest = summary.get("cell_manifest") or {}
    if route_manifest.get("route") != "B_33ch_cbramod_only":
        raise RuntimeError(f"{tag} route manifest mismatch")
    if route_manifest.get("train_val_disjoint") is not True:
        raise RuntimeError(f"{tag} train/val disjoint assertion failed")

    source_stat = paths["checkpoint"].stat()
    source_sha = sha256_file(paths["checkpoint"])
    checkpoint = strict_reload(paths["checkpoint"])
    if int(checkpoint.get("epoch", -1)) != int(summary["best_epoch"]):
        raise RuntimeError(f"{tag} checkpoint epoch does not match summary best_epoch")
    if not math.isclose(float(checkpoint.get("val_loss")), float(summary["best_val_loss"]), rel_tol=0, abs_tol=1e-12):
        raise RuntimeError(f"{tag} checkpoint val loss does not match summary best_val_loss")
    config = checkpoint.get("config") or {}
    for key, expected in {"budget_h": 2000.0, "subset_seed": seed, "init_seed": seed, "epochs": 50}.items():
        if config.get(key) != expected:
            raise RuntimeError(f"{tag} checkpoint config {key}={config.get(key)!r}, expected {expected!r}")
    if checkpoint.get("route_b_manifest") != route_manifest:
        raise RuntimeError(f"{tag} checkpoint and summary route manifests differ")

    return {
        "tag": tag,
        "seed": seed,
        "paths": paths,
        "summary": summary,
        "checkpoint": checkpoint,
        "source_stat": source_stat,
        "source_sha256": source_sha,
        "config_hash_sha256": canonical_hash(config),
        "route_manifest_hash_sha256": canonical_hash(route_manifest),
    }


def immutable_copy(validated, immutable_root):
    tag = validated["tag"]
    source = validated["paths"]["checkpoint"]
    source_sha = validated["source_sha256"]
    cell_dir = Path(immutable_root) / tag
    cell_dir.mkdir(parents=True, exist_ok=True)
    payload_name = f"best.{source_sha}.pth"
    payload = cell_dir / payload_name
    link = cell_dir / "best.pth"

    if payload.exists():
        if sha256_file(payload) != source_sha:
            raise RuntimeError(f"immutable payload exists with wrong content: {payload}")
    else:
        tmp = cell_dir / f".{payload_name}.tmp.{os.getpid()}"
        shutil.copy2(source, tmp)
        if sha256_file(tmp) != source_sha:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"copy SHA mismatch for {tag}")
        os.chmod(tmp, 0o444)
        os.replace(tmp, payload)

    if link.is_symlink():
        if os.readlink(link) != payload_name:
            raise RuntimeError(f"immutable best.pth points to a different payload: {link}")
    elif link.exists():
        raise RuntimeError(f"immutable best.pth exists but is not a symlink: {link}")
    else:
        os.symlink(payload_name, link)

    for key, name in (("summary", "run_summary.json"), ("log", "train_log.jsonl")):
        src = validated["paths"][key]
        dst = cell_dir / name
        src_sha = sha256_file(src)
        if dst.exists():
            if sha256_file(dst) != src_sha:
                raise RuntimeError(f"immutable metadata exists with different content: {dst}")
        else:
            shutil.copy2(src, dst)
            os.chmod(dst, 0o444)

    destination_sha = sha256_file(payload)
    strict_reload(payload)
    source_stat_after = source.stat()
    source_sha_after = sha256_file(source)
    before = validated["source_stat"]
    if (source_stat_after.st_size, source_stat_after.st_mtime_ns, source_sha_after) != (
        before.st_size,
        before.st_mtime_ns,
        source_sha,
    ):
        raise RuntimeError(f"{tag} source checkpoint changed during closure")
    os.chmod(payload, 0o444)
    os.chmod(cell_dir, 0o555)

    summary = validated["summary"]
    route_manifest = summary["cell_manifest"]
    return {
        "tag": tag,
        "budget_h": 2000,
        "seed": validated["seed"],
        "source_checkpoint": str(source.resolve()),
        "immutable_checkpoint": str(payload.resolve()),
        "immutable_best_link": str(link),
        "sha256": destination_sha,
        "bytes": payload.stat().st_size,
        "checkpoint_epoch": int(validated["checkpoint"]["epoch"]),
        "best_val_loss": float(summary["best_val_loss"]),
        "training_epochs_completed": int(summary["epochs"]),
        "source_git": str(validated["checkpoint"].get("git")),
        "config_hash_sha256": validated["config_hash_sha256"],
        "route_manifest_hash_sha256": validated["route_manifest_hash_sha256"],
        "selected_subjects_sha": route_manifest.get("selected_subjects_sha"),
        "train_total_windows": route_manifest.get("train_total_windows"),
        "train_val_disjoint": route_manifest.get("train_val_disjoint"),
        "strict_reload_source": True,
        "strict_reload_immutable": True,
        "immutable_mode_octal": oct(payload.stat().st_mode & 0o777),
        "target_labels_used": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source-root",
        default="/home/infres/yinwang/CMI_AAAI_s2p_b1_launch/results/s2p_route_b_33ch_b1",
    )
    ap.add_argument(
        "--immutable-root",
        default="/home/infres/yinwang/CMI_AAAI_s2p_b1_launch/results/s2p_route_b_33ch_b1_immutable",
    )
    ap.add_argument(
        "--manifest-dir",
        default="results/s2p_route_b_h2000_immutable_closure",
    )
    ap.add_argument("--require-job-ids", nargs="+", default=["890151_6", "890151_7"])
    args = ap.parse_args()

    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    go_nogo_path = manifest_dir / "h2000_immutable_closure_go_nogo.json"
    base = {
        "phase": "A_H2000_immutable_closure",
        "source_root": str(Path(args.source_root).resolve()),
        "immutable_root": str(Path(args.immutable_root).resolve()),
        "parent_job_ids": args.require_job_ids,
        "training_launched": False,
        "downstream_launched": False,
        "h4000_launched": False,
        "target_labels_used": False,
    }
    try:
        if Path(args.source_root).resolve() == Path(args.immutable_root).resolve():
            raise RuntimeError("source and immutable roots must differ")
        require_jobs_absent(args.require_job_ids)
        validated = [validate_cell(args.source_root, tag, seed) for tag, seed in CELLS.items()]
        rows = [immutable_copy(item, args.immutable_root) for item in validated]
        os.chmod(Path(args.immutable_root), 0o555)
        write_csv(manifest_dir / "h2000_immutable_checkpoint_manifest.csv", rows)
        write_json(manifest_dir / "h2000_immutable_checkpoint_manifest.json", {"checkpoints": rows})
        verdict = {
            **base,
            "status": "PASS",
            "immutable_checkpoint_count": len(rows),
            "all_sha256_unique": len({row["sha256"] for row in rows}) == len(rows),
            "all_strict_reload_pass": all(row["strict_reload_immutable"] for row in rows),
            "all_read_only": all(row["immutable_mode_octal"] == "0o444" for row in rows),
            "faced_reaudit_unlocked": True,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(go_nogo_path, verdict)
        print(json.dumps(verdict, indent=2, sort_keys=True), flush=True)
    except Exception as exc:
        verdict = {
            **base,
            "status": "NO_GO",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "faced_reaudit_unlocked": False,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(go_nogo_path, verdict)
        print(json.dumps(verdict, indent=2, sort_keys=True), file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
