#!/usr/bin/env python
"""S2P P1 — CBraMod fixed-budget subject-vs-depth FRONTIER runner (native objective, thin adapter).

One frontier cell = (n_subjects, subset_seed) at fixed T=200 h. Uses the NATIVE CBraMod model + NATIVE
`generate_mask` + NATIVE masked-patch reconstruction MSE (mask 0.5) + native AdamW/CosineAnnealingLR + best-by-
pretrain-val-loss checkpoint — i.e. `pretrain_trainer.Trainer_valid` semantics REPLICATED faithfully, with two
DOCUMENTED adaptations required for the 19-channel TUEG substrate (9A.5 audit, S2P_06 v4):
  (1) the hardcoded 129-ch HBN `EEGNormalizer.transform` (Trainer_valid loads FMCA_plus/...task2.pth) is
      NEUTRALIZED — our loader already per-PATCH z-scores 19-ch windows; NO /100, NO HBN normalizer.
  (2) validation = our FIXED GLOBAL subject-disjoint pool (identical across all cells) for comparable checkpoint
      selection; T_max = epochs*len(train_loader) (native 40 was tied to the 60-epoch HBN run).
Objective/optimizer/scheduler/mask/loss are otherwise byte-for-byte the native ones. Unsupervised: NO target labels
anywhere (firewall). Deterministic (init_seed drives torch/cuda/np/random; subset_seed drives the subject draw only).
"""
import argparse, json, os, sys, time, random, subprocess
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

S2P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S2P / "s2p" / "scripts"))
CBRAMOD = Path(os.path.expanduser("~/eeg2025/CBraMod"))
sys.path.insert(0, str(CBRAMOD))
import tueg_subject_loader as L
from models.cbramod import CBraMod          # native model
from utils.util import generate_mask        # native bernoulli mask (B,C,P)


def setup_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def load_windows(rows, cap=None):
    """materialize (N,19,30,200) float32 windows (per-patch z-scored by the loader). cap = max windows (smoke)."""
    xs, got = [], 0
    for X, _subj in L.windows_for(rows):          # loader firewall: yields X + subject id only, NO target label
        xs.append(X); got += X.shape[0]
        if cap and got >= cap:
            break
    X = np.concatenate(xs, 0)
    return X[:cap] if cap else X


def iter_window_batches(rows, batch_size, train, seed, epoch, cap=None, drop_last=True):
    """Yield CPU batches without materializing the full 9C-v2 high-budget corpus."""
    rng = np.random.default_rng(seed + epoch * 1009 + (17 if train else 0))
    order = np.arange(len(rows))
    if train:
        rng.shuffle(order)
    emitted = 0
    carry = []
    for ri in order:
        for X, _subj in L.windows_for([rows[int(ri)]]):
            if cap is not None and emitted >= cap:
                break
            idx = np.arange(X.shape[0])
            if train:
                rng.shuffle(idx)
            X = X[idx]
            for start in range(0, X.shape[0], batch_size):
                chunk = X[start:start + batch_size]
                if cap is not None:
                    remain = cap - emitted
                    if remain <= 0:
                        break
                    chunk = chunk[:remain]
                if len(carry):
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
                    full = chunk[:n_full * batch_size].reshape(n_full, batch_size, *chunk.shape[1:])
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


def run_epoch(model, loader, device, mask_ratio, opt=None, sched=None, clip=1.0, max_steps=None):
    train = opt is not None
    model.train() if train else model.eval()
    losses = []
    crit = torch.nn.MSELoss(reduction="mean")
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, (x,) in enumerate(loader):
            if max_steps and i >= max_steps:
                break
            x = x.to(device)                                   # already per-patch z-scored; NO /100, NO HBN normalizer
            bz, ch, pn, ps = x.shape
            mask = generate_mask(bz, ch, pn, mask_ratio=mask_ratio, device=device)   # native (B,19,30) bernoulli
            if train:
                opt.zero_grad()
            y = model(x, mask=mask)
            loss = crit(y[mask == 1], x[mask == 1])            # native masked-patch reconstruction MSE
            if train:
                loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step(); sched.step()
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def run_epoch_streaming(model, rows, batch_size, device, mask_ratio, train, seed, epoch, opt=None, sched=None,
                        clip=1.0, max_steps=None, cap=None):
    model.train() if train else model.eval()
    losses = []
    crit = torch.nn.MSELoss(reduction="mean")
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, x in enumerate(iter_window_batches(rows, batch_size, train=train, seed=seed, epoch=epoch,
                                                  cap=cap, drop_last=train)):
            if max_steps and i >= max_steps:
                break
            x = x.to(device)
            bz, ch, pn, ps = x.shape
            mask = generate_mask(bz, ch, pn, mask_ratio=mask_ratio, device=device)
            if train:
                opt.zero_grad()
            y = model(x, mask=mask)
            loss = crit(y[mask == 1], x[mask == 1])
            if train:
                loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step(); sched.step()
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, required=True)
    ap.add_argument("--subset-seed", type=int, required=True)
    ap.add_argument("--init-seed", type=int, required=True)
    ap.add_argument("--total-hours", type=float, default=200.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-2)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--clip-value", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--allocation-policy", default="frontier", choices=["frontier", "high_coverage"])
    ap.add_argument("--min-exposure-hours", type=float, default=0.25)
    ap.add_argument("--loader-mode", default="materialized", choices=["materialized", "streaming", "auto"])
    ap.add_argument("--max-train-windows", type=int, default=None)   # smoke cap
    ap.add_argument("--max-val-windows", type=int, default=None)     # smoke cap
    ap.add_argument("--max-steps", type=int, default=None)           # smoke cap (steps/epoch)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    setup_seed(a.init_seed)
    log = (out / "train_log.jsonl").open("w")
    def emit(d): log.write(json.dumps(d) + "\n"); log.flush(); print(d, flush=True)
    try:
        sha = subprocess.check_output(["git", "-C", str(S2P), "rev-parse", "HEAD"]).decode().strip()[:12]
    except Exception:
        sha = "unknown"

    # ---- data: our frontier cell (native objective, our subset) ----
    t0 = time.time()
    if a.allocation_policy == "high_coverage":
        cell = L.build_high_coverage_cell(a.total_hours, subset_seed=a.subset_seed,
                                          min_exposure_h=a.min_exposure_hours,
                                          expected_n_subjects=a.n_subjects)
    else:
        cell = L.build_frontier_cell(a.n_subjects, subset_seed=a.subset_seed, total_hours=a.total_hours)
    loader_mode = a.loader_mode
    if loader_mode == "auto":
        loader_mode = "streaming" if cell["manifest"]["WT"] > 60000 else "materialized"
    n_train_windows = int(cell["manifest"]["WT"]) if a.max_train_windows is None else min(int(cell["manifest"]["WT"]), a.max_train_windows)
    n_val_windows = int(cell["manifest"]["n_val"] * cell["manifest"]["val_cap_windows"])
    if a.max_val_windows is not None:
        n_val_windows = min(n_val_windows, a.max_val_windows)
    if loader_mode == "materialized":
        Xtr = load_windows(cell["train"], cap=a.max_train_windows)
        Xva = load_windows(cell["val"], cap=a.max_val_windows)
        n_train_windows, n_val_windows = int(Xtr.shape[0]), int(Xva.shape[0])
        tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr)), batch_size=a.batch_size, shuffle=True,
                               num_workers=a.num_workers, drop_last=True, pin_memory=True)
        va_loader = DataLoader(TensorDataset(torch.from_numpy(Xva)), batch_size=a.batch_size, shuffle=False,
                               num_workers=a.num_workers, pin_memory=True)
    else:
        tr_loader = va_loader = None
    emit({"event": "data", "git": sha, "n_subjects": cell["n_subjects"], "subset_seed": a.subset_seed,
          "init_seed": a.init_seed, "total_hours": a.total_hours, "allocation_policy": a.allocation_policy,
          "loader_mode": loader_mode, "n_train_windows": n_train_windows, "n_val_windows": n_val_windows,
          "cell_manifest": cell["manifest"], "load_s": round(time.time() - t0, 1)})

    # ---- native model + native optimizer/scheduler ----
    model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps_per_epoch = max(1, n_train_windows // a.batch_size)
    T_max = max(1, a.epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max, eta_min=1e-5)
    emit({"event": "model", "n_params": int(n_params), "steps_per_epoch": steps_per_epoch, "T_max": T_max})

    # ---- train: best-by-pretrain-val-loss checkpoint (firewall: val loss is unsupervised recon, no target) ----
    best_val, best_ep, first_val = float("inf"), -1, None
    for ep in range(a.epochs):
        if loader_mode == "materialized":
            tl = run_epoch(model, tr_loader, device, a.mask_ratio, opt=opt, sched=sched,
                           clip=a.clip_value, max_steps=a.max_steps)
            vl = run_epoch(model, va_loader, device, a.mask_ratio, max_steps=a.max_steps)
        else:
            tl = run_epoch_streaming(model, cell["train"], a.batch_size, device, a.mask_ratio, train=True,
                                     seed=a.init_seed, epoch=ep, opt=opt, sched=sched, clip=a.clip_value,
                                     max_steps=a.max_steps, cap=a.max_train_windows)
            vl = run_epoch_streaming(model, cell["val"], a.batch_size, device, a.mask_ratio, train=False,
                                     seed=a.init_seed, epoch=ep, max_steps=a.max_steps, cap=a.max_val_windows)
        if first_val is None:
            first_val = vl
        peak = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
        emit({"event": "epoch", "epoch": ep + 1, "train_loss": tl, "val_loss": vl,
              "lr": opt.param_groups[0]["lr"], "gpu_peak_gb": round(peak, 2)})
        assert np.isfinite(tl) and np.isfinite(vl), f"NaN/Inf at epoch {ep+1}"
        ck = {"epoch": ep + 1, "model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
              "scheduler_state": sched.state_dict(), "train_loss": tl, "val_loss": vl,
              "config": vars(a), "git": sha, "n_params": int(n_params)}
        torch.save(ck, out / "last.pth")
        if vl < best_val:
            best_val, best_ep = vl, ep + 1
            torch.save(ck, out / "best.pth")

    # ---- checkpoint reload verification (param-exact) ----
    ck = torch.load(out / "best.pth", map_location=device)
    m2 = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8).to(device)
    m2.load_state_dict(ck["model_state"])
    exact = all(torch.equal(a_.cpu(), b_.cpu()) for a_, b_ in zip(model.state_dict().values(), m2.state_dict().values())) \
        if best_ep == a.epochs else True   # model==best only if best is last epoch; else just confirm strict load
    rel_drop = (first_val - best_val) / first_val if first_val and first_val > 0 else None
    summary = {"event": "done", "n_subjects": cell["n_subjects"], "subset_seed": a.subset_seed, "init_seed": a.init_seed,
               "total_hours": a.total_hours, "allocation_policy": a.allocation_policy,
               "min_exposure_hours": a.min_exposure_hours, "loader_mode": loader_mode,
               "epochs": a.epochs, "first_val_loss": first_val, "best_val_loss": best_val, "best_epoch": best_ep,
               "val_rel_decrease": rel_drop, "converged_20pct": bool(rel_drop is not None and rel_drop >= 0.20),
               "checkpoint_strict_reload": True, "checkpoint_param_exact_if_last": bool(exact),
               "n_params": int(n_params), "git": sha, "smoke": a.smoke,
               "target_labels_used": False, "hbn_normalizer_neutralized": True, "div_by_100": False,
               "cell_manifest": cell["manifest"]}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    emit(summary); log.close()
    print("VERDICT:", "SMOKE_OK" if a.smoke else "RUN_DONE")


if __name__ == "__main__":
    main()
