"""CIGL R3 (visualization) — export a per-node scalar map (e.g. the node leakage map) as a 2-D interpolated
topomap grid, and optionally a PNG, WITHOUT a GPU or a display. VISUALIZATION ONLY: a topomap is a picture of
where leakage concentrates; it is NOT evidence of task reliance (that is leakage_removal.py). Uses a headless
matplotlib Agg backend if a PNG is requested; the grid itself is pure numpy (no matplotlib needed to test).

Channel positions come in as [C,2] 2-D scalp coordinates (already projected). We render on a unit disc and mask
outside it so callers without MNE montage machinery can still produce a figure.
"""
from __future__ import annotations
import numpy as np


def interpolate_topomap(values, pos, grid=64, radius=1.0):
    """Inverse-distance-weighted interpolation of per-node `values` [C] at 2-D positions `pos` [C,2] onto a
    grid x grid image on the unit disc. Returns (Z [grid,grid] with NaN outside the disc, extent). No GPU, no
    scipy. Constant/degenerate inputs are handled (returns a constant disc / all-nan safely)."""
    values = np.asarray(values, dtype=float); pos = np.asarray(pos, dtype=float)
    if pos.shape[0] != values.shape[0] or pos.shape[1] != 2:
        raise ValueError(f"pos {pos.shape} incompatible with values {values.shape}")
    ok = np.isfinite(values)
    if ok.sum() == 0:
        return np.full((grid, grid), np.nan), (-radius, radius, -radius, radius)
    values, pos = values[ok], pos[ok]
    # normalize positions into the disc
    span = np.max(np.abs(pos)) or 1.0
    p = pos / span * radius * 0.9
    lin = np.linspace(-radius, radius, grid)
    gx, gy = np.meshgrid(lin, lin)
    Z = np.full((grid, grid), np.nan)
    disc = gx ** 2 + gy ** 2 <= radius ** 2
    gxf, gyf = gx[disc], gy[disc]
    d2 = (gxf[:, None] - p[None, :, 0]) ** 2 + (gyf[:, None] - p[None, :, 1]) ** 2
    d2 = np.maximum(d2, 1e-9)
    w = 1.0 / d2
    Z[disc] = (w @ values) / w.sum(1)
    return Z, (-radius, radius, -radius, radius)


def export_topomap_png(values, pos, path, grid=64, title="", cmap="viridis"):
    """Render + save a topomap PNG headlessly (matplotlib Agg). Returns the path. matplotlib is imported lazily
    so importing this module (and the grid path) never requires it."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    Z, extent = interpolate_topomap(values, pos, grid=grid)
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    im = ax.imshow(Z, extent=extent, origin="lower", cmap=cmap)
    ax.add_artist(plt.Circle((0, 0), 1.0, fill=False, lw=1.2, color="k"))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    p = str(path) if str(path).endswith(".png") else str(path) + ".png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    return p


def ring_positions(n_channels):
    """Fallback channel layout: evenly spaced on a ring, for callers with no montage. Deterministic."""
    ang = np.linspace(0, 2 * np.pi, int(n_channels), endpoint=False)
    return np.stack([np.cos(ang), np.sin(ang)], 1)
