"""D3 test (amendment 03, Blocker 1): the DGCNN session backfill maps the SAME MOABB canonical trial order as the
EEGNet dumps and passes element-wise y parity, so the oracle runs the identical session-macro split for DGCNN.
Runs against real artifacts; skips (does not fail) if the DGCNN dumps / backfill sidecars are absent on this node."""
import glob
from pathlib import Path
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
from tos_cmi.eeg.relaxation_ladder import feat_from_audit_npz, feat_from_tos_dump


def _cells():
    return sorted(glob.glob(str(REPO / "results/cmi_trace_p0p1/objective_comparison/*/audit/*erm*seed*.audit.npz")))


@pytest.mark.skipif(not _cells(), reason="no DGCNN audit dumps on this node")
def test_dgcnn_session_mapping_and_feature_parity():
    import re
    checked = 0
    for ap in _cells()[:8]:
        name = Path(ap).name
        m = re.search(r"^(BNCI\d+_\d+)_fold\d+_sub(\w+?)_erm_seed(\d+)\.audit\.npz$", name)
        if not m:
            continue
        ds, subj, seed = m.group(1), m.group(2), m.group(3)
        eeg = REPO / "tos_cmi/results/tos_cmi_eeg_frozen" / f"{ds}_EEGNet_LOSO" / f"sub{subj}_erm_lam0_seed{seed}.npz"
        if not eeg.exists():
            continue
        fd = feat_from_audit_npz(ap); fe = feat_from_tos_dump(str(eeg))
        yd = np.asarray(fd["y_target"]).astype(int); ye = np.asarray(fe["y_target"]).astype(int)
        # (1) mapping identity: EEGNet and DGCNN target blocks are the SAME trials in the SAME order
        assert len(yd) == len(ye) and np.array_equal(yd, ye), f"{name}: y_target mismatch (count-match is not enough)"
        # (2) backfill attaches session_target aligned to y_target (len + membership), if the sidecar exists
        if "session_target" in fd:
            st = np.asarray(fd["session_target"])
            assert len(st) == len(yd)
            assert "session_target" in fe and np.array_equal(st.astype(str), np.asarray(fe["session_target"]).astype(str))
            # each session's trial count is a real block (not a length-only coincidence)
            vals, counts = np.unique(st, return_counts=True)
            assert counts.min() >= 4 and len(vals) >= 2
        checked += 1
    if checked == 0:
        pytest.skip("no matching EEGNet dumps for the available DGCNN cells")


@pytest.mark.skipif(not _cells(), reason="no DGCNN audit dumps on this node")
def test_dgcnn_session_backfill_is_reason_coded_when_absent():
    # a DGCNN cell WITHOUT a sidecar must simply not carry session_target (the runner then reason-codes it),
    # never a silently-wrong session vector.
    for ap in _cells():
        fd = feat_from_audit_npz(ap)
        sc = Path(ap).parent / "session_backfill" / (Path(ap).name.replace(".audit.npz", "") + ".session.npz")
        if not sc.exists():
            assert "session_target" not in fd
            return
    pytest.skip("all cells have sidecars (nothing to check for the absent-sidecar branch)")
