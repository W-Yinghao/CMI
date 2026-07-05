"""R3 visualization — topomap grid interpolation (no GPU/display) + optional headless PNG."""
import numpy as np
import pytest
from cmi.eval.topomaps import interpolate_topomap, ring_positions, export_topomap_png


def test_grid_shape_and_disc_mask():
    pos = ring_positions(8); vals = np.arange(8.0)
    Z, extent = interpolate_topomap(vals, pos, grid=32)
    assert Z.shape == (32, 32) and len(extent) == 4
    assert np.isnan(Z[0, 0])                                    # corner is outside the unit disc
    assert np.isfinite(Z[16, 16])                              # center is inside


def test_constant_values_give_constant_disc():
    pos = ring_positions(6); Z, _ = interpolate_topomap(np.full(6, 2.5), pos, grid=24)
    inside = Z[np.isfinite(Z)]
    assert np.allclose(inside, 2.5)


def test_all_nan_values_do_not_crash():
    pos = ring_positions(5); Z, _ = interpolate_topomap(np.full(5, np.nan), pos, grid=16)
    assert np.isnan(Z).all()


def test_ring_positions_shape():
    assert ring_positions(19).shape == (19, 2)


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        interpolate_topomap(np.arange(5.0), ring_positions(6))


def test_png_export_headless(tmp_path):
    mpl = pytest.importorskip("matplotlib")
    p = export_topomap_png(np.arange(8.0), ring_positions(8), str(tmp_path / "t"), title="leak")
    assert p.endswith(".png")
    import os
    assert os.path.getsize(p) > 0
