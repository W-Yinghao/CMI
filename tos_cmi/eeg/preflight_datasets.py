"""Preflight: verify each real-EEG dataset loads OFFLINE from the datalake, and report shape/n_cls/n_ch,
so we don't waste GPU on a mass dump that can't load. CPU; run via scripts/tos_eeg_preflight.sbatch."""
import traceback
from cmi.paths import configure_offline_moabb
configure_offline_moabb()
from cmi.data.moabb_data import load
import moabb.datasets as D

DATASETS = ["BNCI2014_004", "Lee2019_MI", "Cho2017", "Schirrmeister2017", "Stieger2021"]
for name in DATASETS:
    try:
        ds = getattr(D, name)()
        subj = ds.subject_list[0]
        res = load(name, subjects=[subj], resample=128)
        X, y, classes = res[0], res[1], res[-1]
        print("[OK]   %-20s subj=%s X=%s n_ch=%d n_times=%d n_cls=%d classes=%s n_subj_total=%d"
              % (name, subj, tuple(X.shape), X.shape[1], X.shape[2], len(classes),
                 list(classes), len(ds.subject_list)), flush=True)
    except Exception as e:
        print("[FAIL] %-20s %r" % (name, e), flush=True)
        traceback.print_exc()
print("PREFLIGHT_DONE")
