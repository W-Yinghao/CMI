"""CEDAR_01F Route C source-ERM frozen feature dump generator.

This runner is feature supply only. It trains a baseline source ERM backbone on
pre-registered LOSO folds and writes compliant frozen feature dumps. It does not
run CEDAR mask selection or scientific readout.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cedar_eeg.data.feature_schema import stable_json_hash
from cedar_eeg.data.load_frozen_features import load_frozen_feature_npz, write_feature_manifest


BACKBONE_ALIASES = {
    "EEGNetMini": "EEGNet",
    "EEGConformerMini": "EEGConformer",
}

BACKBONE_DEFAULTS = {
    "EEGNetMini": {"tmin": 0.5, "tmax": 3.5, "resample": 128},
    "EEGConformerMini": {"tmin": 0.0, "tmax": 4.0, "resample": 250},
}


@dataclass(frozen=True)
class FeatureDumpPlanItem:
    dataset: str
    backbone: str
    cmi_backbone: str
    seed: int
    fold_id: str
    target_subject: str
    tmin: float
    tmax: float
    resample: int
    roles: tuple[str, ...]
    output_npz: str
    output_manifest: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _scalar_text(x: object) -> str:
    return str(x).replace("/", "_").replace(" ", "_")


def _recording_groups(meta) -> np.ndarray:
    cols = [c for c in ("subject", "session", "run") if c in meta.columns]
    if not cols:
        raise ValueError("metadata lacks subject/session/run columns for grouped feature dump")
    out = meta[cols[0]].astype(str)
    for col in cols[1:]:
        out = out + "|" + meta[col].astype(str)
    return out.to_numpy()


def _remap(values: np.ndarray) -> np.ndarray:
    uniq = {v: i for i, v in enumerate(sorted(np.unique(values)))}
    return np.array([uniq[v] for v in values], dtype=np.int64)


def _source_train_audit_split(meta, source_mask: np.ndarray, *, seed: int, fold_id: str, frac: float) -> tuple[np.ndarray, np.ndarray]:
    groups = _recording_groups(meta)
    source_groups = np.array(sorted(np.unique(groups[source_mask])), dtype=object)
    if len(source_groups) < 2:
        raise ValueError("source split requires at least two recording groups")
    rng_seed = int(stable_json_hash({"seed": seed, "fold_id": fold_id})[:8], 16)
    rng = np.random.default_rng(rng_seed)
    shuffled = source_groups.copy()
    rng.shuffle(shuffled)
    n_audit = max(1, int(round(len(shuffled) * frac)))
    n_audit = min(n_audit, len(shuffled) - 1)
    audit_groups = set(shuffled[:n_audit])
    audit_mask = source_mask & np.array([g in audit_groups for g in groups])
    train_mask = source_mask & ~audit_mask
    if not train_mask.any() or not audit_mask.any():
        raise ValueError("deterministic source split produced an empty role")
    return train_mask, audit_mask


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    items: list[FeatureDumpPlanItem] = []
    for backbone in args.backbones:
        if backbone not in BACKBONE_ALIASES:
            raise ValueError(f"unsupported CEDAR_01F backbone {backbone}")
        defaults = BACKBONE_DEFAULTS[backbone]
        target_subjects = args.target_subjects or [str(i) for i in range(1, 10)]
        for target in target_subjects:
            fold_id = _scalar_text(target)
            stem = f"{args.dataset}_{backbone}_seed{args.seed}_{fold_id}"
            items.append(
                FeatureDumpPlanItem(
                    dataset=args.dataset,
                    backbone=backbone,
                    cmi_backbone=BACKBONE_ALIASES[backbone],
                    seed=args.seed,
                    fold_id=fold_id,
                    target_subject=str(target),
                    tmin=float(defaults["tmin"]),
                    tmax=float(defaults["tmax"]),
                    resample=int(defaults["resample"]),
                    roles=("source_train", "source_audit", "target_audit"),
                    output_npz=str(Path(args.out_dir) / f"{stem}.npz"),
                    output_manifest=str(Path(args.out_dir) / f"{stem}.manifest.json"),
                )
            )
    plan = {
        "project": "CEDAR-EEG",
        "phase": "CEDAR_01F_feature_supply_route_c",
        "selection_run": False,
        "scientific_readout_run": False,
        "deployable": False,
        "dataset": args.dataset,
        "seed": args.seed,
        "backbones": args.backbones,
        "source_audit_fraction": args.source_audit_fraction,
        "items": [x.to_dict() for x in items],
    }
    plan["plan_hash"] = stable_json_hash(plan)
    return plan


def write_plan(plan: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(plan, f, indent=2, sort_keys=True)


def _load_dataset(dataset: str, *, tmin: float, tmax: float, resample: int):
    from cmi.run_loso import load

    return load(dataset, subjects=None, tmin=tmin, tmax=tmax, resample=resample)


def _train_and_dump_item(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from cmi.data.moabb_data import domain_labels, loso_splits
    from cmi.models.backbones import build_backbone
    from cmi.train.trainer import embed, train_model

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    X, y, meta, classes = _load_dataset(
        item["dataset"],
        tmin=float(item["tmin"]),
        tmax=float(item["tmax"]),
        resample=int(item["resample"]),
    )
    target_subject = str(item["target_subject"])
    selected = None
    for tgt, source_mask, target_mask in loso_splits(meta):
        if str(tgt) == target_subject or _scalar_text(tgt) == target_subject:
            selected = (tgt, source_mask, target_mask)
            break
    if selected is None:
        raise ValueError(f"target subject {target_subject} not found")

    tgt, source_mask, target_mask = selected
    fold_id = _scalar_text(tgt)
    source_train, source_audit = _source_train_audit_split(
        meta, source_mask, seed=int(item["seed"]), fold_id=fold_id, frac=float(args.source_audit_fraction)
    )
    subjects = meta["subject"].astype(str).to_numpy()
    groups = _recording_groups(meta)
    dom_train = _remap(subjects[source_train])

    n_cls = len(classes)
    n_ch, n_t = X.shape[1], X.shape[2]
    backbone = build_backbone(item["cmi_backbone"], n_ch, n_t, n_cls, device=device)
    backbone, _, train_diag = train_model(
        backbone,
        X[source_train],
        y[source_train],
        dom_train,
        n_cls,
        method="erm",
        lam=0.0,
        epochs=args.epochs,
        bs=args.bs,
        warmup=max(1, min(args.warmup, args.epochs)),
        n_inner=1,
        sampler=args.sampler,
        weight_decay=args.weight_decay,
        device=device,
        seed=int(item["seed"]),
    )

    role_parts = [
        ("source_train", source_train),
        ("source_audit", source_audit),
        ("target_audit", target_mask),
    ]
    z_parts = []
    y_parts = []
    domain_parts = []
    group_parts = []
    role_values = []
    subject_parts = []
    session_parts = []
    recording_parts = []
    sample_parts = []
    for role, mask in role_parts:
        z_role = embed(backbone, X[mask], device=device, bs=args.embed_bs).astype("float32")
        idx = np.where(mask)[0]
        z_parts.append(z_role)
        y_parts.append(y[mask].astype("int64"))
        domain_parts.append(subjects[mask].astype(str))
        group_parts.append(groups[mask].astype(str))
        role_values.append(np.array([role] * int(mask.sum())))
        subject_parts.append(subjects[mask].astype(str))
        session_parts.append(meta["session"].astype(str).to_numpy()[mask] if "session" in meta.columns else np.array([""] * int(mask.sum())))
        recording_parts.append(groups[mask].astype(str))
        sample_parts.append(np.array([f"{item['dataset']}|{fold_id}|{int(i)}" for i in idx]))

    out_npz = Path(item["output_npz"])
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        z=np.concatenate(z_parts),
        y=np.concatenate(y_parts),
        domain=np.concatenate(domain_parts),
        groups=np.concatenate(group_parts),
        role=np.concatenate(role_values),
        dataset=np.array(item["dataset"]),
        backbone=np.array(item["backbone"]),
        cmi_backbone=np.array(item["cmi_backbone"]),
        seed=np.array(int(item["seed"])),
        fold_id=np.array(fold_id),
        target_subject=np.array(str(tgt)),
        sample_id=np.concatenate(sample_parts),
        subject_id=np.concatenate(subject_parts),
        session_id=np.concatenate(session_parts),
        recording_id=np.concatenate(recording_parts),
        train_diag=json.dumps(train_diag, sort_keys=True),
        classes=np.asarray(classes, dtype=str),
        deployable=np.array(False),
        cedar_role=np.array("feature_supply_candidate_only"),
    )
    bundle = load_frozen_feature_npz(out_npz)
    manifest = write_feature_manifest(bundle, item["output_manifest"])
    return {
        "fold_id": fold_id,
        "target_subject": str(tgt),
        "npz": str(out_npz),
        "manifest": str(item["output_manifest"]),
        "manifest_hash": manifest["manifest_hash"],
        "file_sha256": manifest["file_sha256"],
        "n_samples": manifest["n_samples"],
        "z_dim": manifest["z_dim"],
        "train_diag": train_diag,
    }


def execute_plan(plan: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    selected_backbones = set(args.backbones)
    selected_targets = set(str(x) for x in args.target_subjects) if args.target_subjects else None
    outputs = []
    for item in plan["items"]:
        if item["backbone"] not in selected_backbones:
            continue
        if selected_targets is not None and str(item["target_subject"]) not in selected_targets:
            continue
        outputs.append(_train_and_dump_item(item, args))
    run_manifest = {
        "project": "CEDAR-EEG",
        "phase": "CEDAR_01F_feature_supply_route_c",
        "plan_hash": plan["plan_hash"],
        "selection_run": False,
        "scientific_readout_run": False,
        "deployable": False,
        "outputs": outputs,
    }
    run_manifest["run_manifest_hash"] = stable_json_hash(run_manifest)
    out = Path(args.out_dir) / "run_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(run_manifest, f, indent=2, sort_keys=True)
    return run_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--backbones", nargs="+", default=["EEGNetMini", "EEGConformerMini"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-subjects", nargs="*", default=[])
    ap.add_argument("--out-dir", default="results/cedar/feature_supply/bnci2014_001_seed0")
    ap.add_argument("--plan-out", default="")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--embed-bs", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--sampler", default="classbal", choices=["classbal", "raw", "domainbal"])
    ap.add_argument("--source-audit-fraction", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    plan = build_plan(args)
    plan_out = args.plan_out or str(Path(args.out_dir) / "feature_dump_plan.json")
    write_plan(plan, plan_out)
    print(json.dumps({"plan_out": plan_out, "plan_hash": plan["plan_hash"], "n_items": len(plan["items"])}, indent=2))
    if not args.plan_only:
        manifest = execute_plan(plan, args)
        print(json.dumps({"run_manifest": str(Path(args.out_dir) / "run_manifest.json"), **manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
