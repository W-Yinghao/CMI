#!/usr/bin/env python
"""S2P Route B B1 CBraMod 33ch training runner.

Native CBraMod objective/mask/loss; Route-B fixed-group-mix 33ch TUEG loader.
No labels are read. Checkpoint selection is subject-disjoint pretrain-val loss.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

S2P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S2P / "s2p" / "scripts"))
CBRAMOD = Path(os.path.expanduser("~/eeg2025/CBraMod"))
sys.path.insert(0, str(CBRAMOD))

import route_b_33ch_loader as RBL
from models.cbramod import CBraMod
from utils.util import generate_mask


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def iter_batches(rows, batch_size, train, seed, epoch, contract_dir, cap=None, drop_last=True):
    rng = np.random.default_rng(seed + epoch * 1009 + (31 if train else 0))
    order = np.arange(len(rows))
    if train:
        rng.shuffle(order)
    emitted = 0
    carry = []
    for ri in order:
        for x, _subj in RBL.windows_for([rows[int(ri)]], contract_dir=contract_dir):
            if cap is not None and emitted >= cap:
                break
            idx = np.arange(x.shape[0])
            if train:
                rng.shuffle(idx)
            x = x[idx]
            for start in range(0, x.shape[0], batch_size):
                chunk = x[start:start + batch_size]
                if cap is not None:
                    remain = cap - emitted
                    if remain <= 0:
                        break
                    chunk = chunk[:remain]
                if carry:
                    need = batch_size - sum(c.shape[0] for c in carry)
                    carry.append(chunk[:need])
                    chunk = chunk[need:]
                    if sum(c.shape[0] for c in carry) == batch_size:
                        batch = np.concatenate(carry, axis=0)
                        emitted += batch.shape[0]
                        carry = []
                        yield torch.from_numpy(batch)
                if chunk.shape[0] >= batch_size:
                    n_full = chunk.shape[0] // batch_size
                    full = chunk[: n_full * batch_size].reshape(n_full, batch_size, *chunk.shape[1:])
                    for bi in range(n_full):
                        emitted += batch_size
                        yield torch.from_numpy(full[bi])
                    chunk = chunk[n_full * batch_size:]
                if chunk.shape[0] > 0:
                    carry.append(chunk)
        if cap is not None and emitted >= cap:
            break
    if carry and not drop_last:
        batch = np.concatenate(carry, axis=0)
        if batch.shape[0] > 0:
            yield torch.from_numpy(batch)


def run_epoch(model, rows, batch_size, device, mask_ratio, train, seed, epoch, contract_dir,
              opt=None, sched=None, clip=1.0, max_steps=None, cap=None):
    model.train() if train else model.eval()
    crit = torch.nn.MSELoss(reduction="mean")
    losses = []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, x in enumerate(iter_batches(rows, batch_size, train=train, seed=seed, epoch=epoch,
                                           contract_dir=contract_dir, cap=cap, drop_last=train)):
            if max_steps and i >= max_steps:
                break
            x = x.to(device)
            bz, ch, pn, _ = x.shape
            mask = generate_mask(bz, ch, pn, mask_ratio=mask_ratio, device=device)
            if train:
                opt.zero_grad()
            y = model(x, mask=mask)
            loss = crit(y[mask == 1], x[mask == 1])
            if train:
                loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                sched.step()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-h", type=float, required=True)
    ap.add_argument("--subset-seed", type=int, required=True)
    ap.add_argument("--init-seed", type=int, required=True)
    ap.add_argument("--contract-dir", default="results/s2p_route_b_33ch_contract")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-2)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--clip-value", type=float, default=1.0)
    ap.add_argument("--max-train-windows", type=int, default=None)
    ap.add_argument("--max-val-windows", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    setup_seed(args.init_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        git = subprocess.check_output(["git", "-C", str(S2P), "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git = "unknown"

    t0 = time.time()
    cell = RBL.build_route_b_cell(args.budget_h, args.subset_seed, contract_dir=args.contract_dir)
    n_train_windows = int(cell["manifest"]["WT"])
    if args.max_train_windows is not None:
        n_train_windows = min(n_train_windows, args.max_train_windows)
    n_val_windows = int(cell["manifest"]["val_total_windows"])
    if args.max_val_windows is not None:
        n_val_windows = min(n_val_windows, args.max_val_windows)

    log = (out / "train_log.jsonl").open("w")

    def emit(obj):
        log.write(json.dumps(obj) + "\n")
        log.flush()
        print(obj, flush=True)

    emit({
        "event": "data",
        "route": "B_33ch_cbramod_only",
        "git": git,
        "budget_h": args.budget_h,
        "subset_seed": args.subset_seed,
        "init_seed": args.init_seed,
        "n_train_windows": n_train_windows,
        "n_val_windows": n_val_windows,
        "load_s": round(time.time() - t0, 1),
        "cell_manifest": cell["manifest"],
        "target_labels_used": False,
    })

    model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, n_train_windows // args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs * steps_per_epoch), eta_min=1e-5)
    emit({"event": "model", "n_params": int(n_params), "steps_per_epoch": int(steps_per_epoch), "device": str(device)})

    best_val, best_epoch, first_val = float("inf"), -1, None
    for ep in range(args.epochs):
        tl = run_epoch(model, cell["train"], args.batch_size, device, args.mask_ratio, train=True,
                       seed=args.init_seed, epoch=ep, contract_dir=args.contract_dir, opt=opt, sched=sched,
                       clip=args.clip_value, max_steps=args.max_steps, cap=args.max_train_windows)
        vl = run_epoch(model, cell["val"], args.batch_size, device, args.mask_ratio, train=False,
                       seed=args.init_seed, epoch=ep, contract_dir=args.contract_dir, max_steps=args.max_steps,
                       cap=args.max_val_windows)
        if first_val is None:
            first_val = vl
        peak = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
        emit({"event": "epoch", "epoch": ep + 1, "train_loss": tl, "val_loss": vl,
              "lr": opt.param_groups[0]["lr"], "gpu_peak_gb": round(peak, 2)})
        if not (np.isfinite(tl) and np.isfinite(vl)):
            raise RuntimeError(f"NaN/Inf at epoch {ep + 1}")
        ckpt = {
            "epoch": ep + 1,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scheduler_state": sched.state_dict(),
            "train_loss": tl,
            "val_loss": vl,
            "config": vars(args),
            "git": git,
            "n_params": int(n_params),
            "route_b_manifest": cell["manifest"],
        }
        torch.save(ckpt, out / "last.pth")
        if vl < best_val:
            best_val, best_epoch = vl, ep + 1
            torch.save(ckpt, out / "best.pth")

    reloaded = torch.load(out / "best.pth", map_location=device)
    check = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8).to(device)
    check.load_state_dict(reloaded["model_state"])
    summary = {
        "event": "done",
        "route": "B_33ch_cbramod_only",
        "budget_h": args.budget_h,
        "subset_seed": args.subset_seed,
        "init_seed": args.init_seed,
        "epochs": args.epochs,
        "first_val_loss": first_val,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "checkpoint_strict_reload": True,
        "target_labels_used": False,
        "training_wall_s": round(time.time() - t0, 1),
        "smoke": bool(args.smoke),
        "cell_manifest": cell["manifest"],
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    emit(summary)
    log.close()


if __name__ == "__main__":
    main()
