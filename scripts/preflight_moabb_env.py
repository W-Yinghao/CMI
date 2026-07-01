"""MOABB / braindecode environment preflight for the Graph-DualCMI GPU pilot (docs/CIGL_46).

Hard gate BEFORE any GPU run: verify the run environment can (a) build the graph backbone, (b) build the
requested braindecode task baselines (or explain why not), and (c) load one subject of each pilot dataset
offline — so a GPU job never crashes at data load (the BNCI2014001 -> BNCI2014_001 moabb rename class,
Risk-of-record). Non-GPU, no training, no download forced.

    python scripts/preflight_moabb_env.py --datasets BNCI2014_001 BNCI2015_001 \
        --backbones DGCNNGraph EEGNet --subjects 1 --no-download
    # (or: python -m scripts.preflight_moabb_env ...)

Exit code 0 iff all REQUIRED checks pass: DGCNNGraph builds AND every dataset loads. braindecode task
backbones (EEGNet/ShallowConvNet/...) are reported but NON-required (the fallback env may lack braindecode);
a failing one is a WARN so the venue/env can be chosen accordingly.
"""
from __future__ import annotations
import argparse
import importlib
import os
import platform
import sys
import traceback

# make `cmi` importable whether invoked as `python scripts/preflight_moabb_env.py` or `-m scripts.preflight_moabb_env`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GRAPH_BACKBONES = {"DGCNNGraph", "GraphCMI", "DGCNN", "RGNN"}   # non-braindecode; required-capable


def _ver(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "(no __version__)")
    except Exception as e:
        return f"IMPORT FAIL ({type(e).__name__}: {e})"


def check_backbone(name, n_ch, n_times, n_cls):
    from cmi.models.backbones import build_backbone
    try:
        bb = build_backbone(name, n_ch, n_times, n_cls, device="cpu")
        fg = callable(getattr(bb, "forward_graph", None))
        return True, f"z_dim={getattr(bb, 'z_dim', '?')} forward_graph={fg}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_dataset(name, subject):
    from cmi.run_loso import load
    try:
        X, y, meta, classes = load(name, subjects=[subject])
        y_uniq = sorted(int(v) for v in set(y.tolist()))
        return True, dict(subject=subject, n_trials=int(X.shape[0]), n_channels=int(X.shape[1]),
                          n_times=int(X.shape[2]), n_classes=len(classes),
                          class_names=[str(c) for c in classes], y_unique=y_uniq)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="MOABB/braindecode env preflight for the Graph-DualCMI pilot")
    ap.add_argument("--datasets", nargs="+", default=["BNCI2014_001", "BNCI2015_001"])
    ap.add_argument("--backbones", nargs="+", default=["DGCNNGraph", "EEGNet"])
    ap.add_argument("--subjects", type=int, default=1, help="first subject id to probe per dataset")
    ap.add_argument("--n_ch", type=int, default=22)
    ap.add_argument("--n_times", type=int, default=512)
    ap.add_argument("--n_classes", type=int, default=4)
    ap.add_argument("--no-download", action="store_true", help="informational; moabb is configured offline")
    args = ap.parse_args(argv)

    print("=" * 78)
    print("Graph-DualCMI environment preflight")
    print("=" * 78)
    print(f"python          : {platform.python_version()} ({sys.executable})")
    print(f"torch           : {_ver('torch')}")
    print(f"numpy           : {_ver('numpy')}")
    print(f"moabb           : {_ver('moabb')}")
    print(f"braindecode     : {_ver('braindecode')}")
    print(f"MNE_DATA        : {os.environ.get('MNE_DATA', '(unset)')}")
    print(f"MNE_DATASETS_BNCI_PATH: {os.environ.get('MNE_DATASETS_BNCI_PATH', '(unset)')}")
    print(f"download        : {'DISABLED (offline)' if args.no_download else 'as-configured'}")

    required_fail, warn_fail = [], []

    print("\n-- backbones --")
    for name in args.backbones:
        ok, info = check_backbone(name, args.n_ch, args.n_times, args.n_classes)
        required = name in GRAPH_BACKBONES
        tag = "PASS" if ok else ("FAIL(required)" if required else "WARN(non-required)")
        print(f"  {name:14s} {tag:18s} {info}")
        if not ok:
            (required_fail if required else warn_fail).append(f"backbone:{name}")

    print("\n-- datasets (offline load, 1 subject) --")
    for name in args.datasets:
        ok, info = check_dataset(name, args.subjects)
        print(f"  {name:14s} {'PASS' if ok else 'FAIL(required)':18s} {info}")
        if not ok:
            required_fail.append(f"dataset:{name}")

    print("\n" + "=" * 78)
    if required_fail:
        print(f"PREFLIGHT: FAIL — required checks failed: {required_fail}")
        if warn_fail:
            print(f"           (also non-required WARN: {warn_fail})")
        print("Do NOT submit the GPU pilot until required checks pass (fix env / moabb version / dataset cache).")
        return 1
    print("PREFLIGHT: PASS — DGCNNGraph builds and all datasets load offline.")
    if warn_fail:
        print(f"           non-required WARN (braindecode backbones): {warn_fail} — "
              f"run those baselines in a braindecode-capable env, or drop them from the pilot.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
