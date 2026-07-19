import sys, os, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from tos_cmi.eeg.feature_dump import dump_fold
name, s, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tmin, tmax = (0.0, 3.0) if name == "Stieger2021" else (0.5, 3.5)
base = REPO / "tos_cmi" / "results" / "tos_cmi_eeg_frozen" / f"{name}_EEGNet_LOSO"
base.mkdir(parents=True, exist_ok=True)
out = base / f"sub{s}_erm_lam0_seed{seed}.npz"
marker = REPO / "results" / "gpu_dump_done" / f"{name}_{s}_{seed}.done"
if out.exists() and marker.exists():
    print(f"SKIP {name} sub{s} seed{seed}"); sys.exit(0)
if out.exists() and not marker.exists():
    marker.write_text(str(out) + "\n"); print(f"MARK-EXISTING {name} sub{s} seed{seed}"); sys.exit(0)
t0 = time.time()
dump_fold(name, s, "erm", 0.0, seed, str(out), backbone="EEGNet",
          epochs=300, device="cuda", tmin=tmin, tmax=tmax, domain_mode="subject")
marker.write_text(str(out) + "\n")
print(f"OK {name} sub{s} seed{seed} ({time.time()-t0:.0f}s) -> {out}", flush=True)
