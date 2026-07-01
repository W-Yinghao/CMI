"""CIGL_47 P3-H — central_strip_v1 electrode grouping for the MI datasets (CPU only).

The coarse region taxonomy degenerated on these sensorimotor caps (BNCI2014 -> one 17-node central blob;
BNCI2015 -> index fallback). central_strip_v1 splits the FC/C/CP strip by anterior-central-posterior x
left-mid-right so no group dominates and neither dataset uses an index fallback.
"""
import torch

from cmi.models.backbones import build_backbone
from cmi.models.fb_lgg_dualcmi import FBLGGDualCMIBackbone, central_strip_groups, _CENTRAL_STRIP_V1
from cmi.run_loso import _DATASET_CH_NAMES


def _groups(dataset):
    ch = _DATASET_CH_NAMES[dataset]
    idx, named, warn = central_strip_groups(dataset, ch)
    return ch, idx, named, warn


def test_bnci2014_covers_22_once_no_blob():
    ch, idx, named, warn = _groups("BNCI2014_001")
    assert warn is None and idx is not None
    flat = sorted(c for g in idx for c in g)
    assert flat == list(range(22)), "central_strip_v1 must assign all 22 channels exactly once"
    assert max(len(g) for g in idx) <= 6, "no group may dominate (the 17-node blob is gone)"
    assert max(len(g) for g in idx) == 3 and len(idx) == 9   # 9 FC/C/CP x L/mid/R groups


def test_bnci2015_covers_13_once_not_index_fallback():
    ch, idx, named, warn = _groups("BNCI2015_001")
    assert warn is None and idx is not None
    flat = sorted(c for g in idx for c in g)
    assert flat == list(range(13)) and len(flat) == len(set(flat))
    assert len(idx) == 5 and max(len(g) for g in idx) <= 3     # FC_strip, C_l/mid/r, CP_strip
    # named groups (not index labels) -> provenance is montage-meaningful, not index-fallback
    assert set(named) == {"FC_strip", "C_left", "C_mid", "C_right", "CP_strip"}


def test_central_strip_fail_closed_on_bad_montage():
    # missing electrode -> warning (fail closed), never a silent partial grouping
    idx, named, warn = central_strip_groups("BNCI2014_001", ["Fz", "FC3"])   # only 2 of 22 names
    assert idx is None and named is None and "not in ch_names" in warn
    # unknown dataset -> no preset
    idx2, _, warn2 = central_strip_groups("SomeDataset", ["A", "B"])
    assert idx2 is None and "no central_strip_v1 preset" in warn2


def test_fblgg_forward_graph_with_both_presets():
    for ds, C in (("BNCI2014_001", 22), ("BNCI2015_001", 13)):
        ch, idx, named, warn = _groups(ds)
        torch.manual_seed(0)
        bb = build_backbone("FBLGGGraph", C, 128, 2, device="cpu",
                            ch_names=ch, groups=idx, group_names=named, grouping_scheme="central_strip_v1")
        assert bb.grouping_scheme == "central_strip_v1" and bb.n_groups == len(idx)
        assert bb.group_names == named
        bb.eval()
        out = bb.forward_graph(torch.randn(4, C, 128))
        assert len(out) == 5 and out[0].shape == (4, 2) and out[2].shape == (4, C, bb.node_z_dim)


def test_fblgg_ablations_run_with_presets_eval():
    ch, idx, named, warn = _groups("BNCI2014_001")
    torch.manual_seed(0)
    bb = build_backbone("FBLGGGraph", 22, 128, 2, device="cpu",
                        ch_names=ch, groups=idx, group_names=named, grouping_scheme="central_strip_v1")
    bb.eval()
    x = torch.randn(8, 22, 128)
    full = bb.forward_graph(x)[0]
    for mode in ("zero_graph", "zero_temporal", "permute_nodes"):
        torch.manual_seed(3)
        ab = bb.ablate(x, mode)
        assert ab.shape == (8, 2) and not torch.allclose(full, ab, atol=1e-4)


def test_build_backbone_records_grouping_provenance_attrs():
    # the attributes run_loso reads into the per-fold record (grouping_scheme / channel_groups)
    ch, idx, named, warn = _groups("BNCI2015_001")
    bb = build_backbone("FBLGGGraph", 13, 128, 2, device="cpu",
                        ch_names=ch, groups=idx, group_names=named, grouping_scheme="central_strip_v1")
    assert bb.grouping_scheme == "central_strip_v1"
    assert isinstance(bb.group_names, dict) and "FC_strip" in bb.group_names
    assert sorted(c for g in bb.groups for c in g) == list(range(13))


def test_malformed_explicit_groups_fail_closed():
    # a grouping that does not cover every channel exactly once must raise (backbone-level guard)
    import pytest
    with pytest.raises(ValueError, match="exactly once"):
        FBLGGDualCMIBackbone(4, 128, 2, groups=[[0, 1], [1, 2]])   # 3 missing, 1 duplicated
