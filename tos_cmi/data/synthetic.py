"""Controllable feature-space generators with a known (Z_Y, Z_N) structure.

These are the minimal worlds in which the idealized proposition is exactly testable.

`make(overlap)` plants a label signal in a subspace L and a class-conditional domain
signal whose carrier is rotated by `overlap` from a task-orthogonal subspace N toward
the *class-discriminant* subspace M = span(centered class means) inside L:

  overlap = 0 : domain carriers live in N, orthogonal to the label signal.
                => a task-orthogonal nuisance subspace exists; removing it preserves the
                   Bayes risk (Prop. 1) and the leakage is concentrated there.
  overlap -> 1: domain carriers move into the class-discriminant directions, so deleting
                them would cost label information; the deletable subspace shrinks.

`make_collinear()` is the deterministic worst case: the domain shift is exactly along the
single class-discriminant axis, so the ONLY domain-rich direction is also the most
label-rich one. No risk-feasible nuisance subspace exists and the selector must return
identity. This is the clean falsification of "always delete" (global LPC).

A random global rotation makes the planted structure non-axis-aligned, so recovering it
is a subspace problem. Ground-truth `nuisance_basis` / `label_basis` are returned for
principal-angle scoring.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np


@dataclass
class SynthSpec:
    n: int = 4000
    d: int = 24              # ambient feature dim
    n_cls: int = 4
    n_dom: int = 6
    d_label: int = 4         # dims spanning the label signal
    d_nuis: int = 4          # dims spanning the task-orthogonal domain signal
    sep_label: float = 2.0   # class-mean magnitude
    sep_dom: float = 2.5     # domain-mean magnitude (within class)
    noise: float = 1.0
    overlap: float = 0.0     # 0 => task-orthogonal nuisance; 1 => domain along discriminant
    rotate: bool = True      # random global rotation (non-axis-aligned)


def _random_orthonormal(d, k, rng):
    A = rng.standard_normal((d, k))
    Q, _ = np.linalg.qr(A)
    return Q[:, :k]


def _centered_unit(M):
    """Center rows and scale to unit average row RMS, so cos/sin interpolation between two
    carrier sets is a fair (magnitude-preserving) rotation."""
    Mc = M - M.mean(0, keepdims=True)
    rms = math.sqrt((Mc ** 2).sum(1).mean()) + 1e-8
    return Mc / rms


def _basis_of(M, tol=1e-8):
    U, sv, _ = np.linalg.svd(M.T, full_matrices=False)
    return U[:, : int((sv > tol).sum())]


def make(spec: SynthSpec, seed: int = 0, struct_seed: int | None = None):
    """`struct_seed` fixes the planted geometry (axes, class/domain means, rotation); `seed`
    draws the finite sample (labels, domains, noise). Fix `struct_seed` and vary `seed` to
    measure *estimator* stability across draws of the SAME distribution (the termination
    gate); leave it None and the two coincide (each seed is a fresh world)."""
    s = spec
    gen = np.random.default_rng(struct_seed if struct_seed is not None else seed)  # geometry
    smp = np.random.default_rng(seed)                                              # samples

    base = _random_orthonormal(s.d, s.d_label + s.d_nuis, gen)
    L = base[:, : s.d_label]                       # label axes      [d, d_label]
    N = base[:, s.d_label : s.d_label + s.d_nuis]  # nuisance axes    [d, d_nuis]

    # class means in label span; M = their centered span (the discriminant subspace)
    cls_coords = gen.standard_normal((s.n_cls, s.d_label))
    class_means = s.sep_label * (cls_coords @ L.T)            # [n_cls, d]
    CM = class_means - class_means.mean(0, keepdims=True)     # centered, spans M

    # domain carriers: orthogonal part in N, entangled part along the discriminant M
    theta = s.overlap * (math.pi / 2.0)
    dom_n = _centered_unit(gen.standard_normal((s.n_dom, s.d_nuis)) @ N.T)        # in N
    dom_m = _centered_unit(gen.standard_normal((s.n_dom, s.n_cls)) @ CM)          # in M
    carriers = s.sep_dom * (math.cos(theta) * dom_n + math.sin(theta) * dom_m)    # [n_dom, d]
    domain_means = carriers
    Q = _random_orthonormal(s.d, s.d, gen) if s.rotate else None

    y = smp.integers(0, s.n_cls, size=s.n)
    d = smp.integers(0, s.n_dom, size=s.n)
    Z = class_means[y] + domain_means[d] + s.noise * smp.standard_normal((s.n, s.d))

    nuis_basis = _basis_of(carriers - carriers.mean(0, keepdims=True))
    label_basis = _basis_of(CM)

    if s.rotate:
        Z = Z @ Q.T
        nuis_basis = Q @ nuis_basis
        label_basis = Q @ label_basis

    return {"Z": Z.astype(np.float32), "y": y.astype(np.int64), "d": d.astype(np.int64),
            "nuisance_basis": nuis_basis.astype(np.float32),
            "label_basis": label_basis.astype(np.float32), "spec": s}


def make_collinear(n=4000, n_dom=6, d=24, sep_label=2.0, sep_dom=2.5, noise=1.0,
                   rotate=True, seed=0):
    """Deterministic worst case: 2 classes separated along axis u, and the domain shift is
    ALSO along u. The only domain-rich direction is the most label-rich one => no
    risk-feasible nuisance subspace => the selector must return identity."""
    rng = np.random.default_rng(seed)
    u = _random_orthonormal(d, 1, rng)[:, 0]                  # unit class-discriminant axis
    y = rng.integers(0, 2, size=n)
    d_lab = rng.integers(0, n_dom, size=n)
    sign = (2 * y - 1).astype(float)                          # +-1 class offset along u
    t = np.linspace(-1.0, 1.0, n_dom)                         # per-domain offset along u
    Z = (sep_label * sign[:, None] * u[None, :]
         + sep_dom * t[d_lab][:, None] * u[None, :]
         + noise * rng.standard_normal((n, d)))
    nuis_basis = u[:, None].copy()
    if rotate:
        Q = _random_orthonormal(d, d, rng)
        Z = Z @ Q.T
        nuis_basis = Q @ nuis_basis

    spec = SimpleNamespace(n_cls=2, n_dom=n_dom, d=d)
    return {"Z": Z.astype(np.float32), "y": y.astype(np.int64), "d": d_lab.astype(np.int64),
            "nuisance_basis": nuis_basis.astype(np.float32),
            "label_basis": nuis_basis.astype(np.float32), "spec": spec}


def make_covariance_only(n=6000, n_dom=6, d=24, sep_label=2.0, dom_var=2.0, noise=1.0,
                         rotate=True, seed=0):
    """Honest limitation case: the label is a MEAN shift (along u) but the domain leakage is
    a *covariance* shift -- each domain scales the variance along a fixed axis w (orthogonal
    to u), with ZERO domain mean. The between-domain MEAN scatter F_{D|Y} ~ 0, so the
    first-moment selector is BLIND to it and (correctly, given what it measures) returns
    identity -- even though D is decodable from Z|Y by a quadratic/covariance probe. This is
    why the current method is `label-mean-scatter-light`, NOT `task/domain-orthogonal`."""
    rng = np.random.default_rng(seed)
    basis = _random_orthonormal(d, 2, rng)
    u, w = basis[:, 0], basis[:, 1]                           # label-mean axis / domain-var axis
    y = rng.integers(0, 2, size=n)
    d_lab = rng.integers(0, n_dom, size=n)
    sign = (2 * y - 1).astype(float)
    std_w = np.linspace(0.4, dom_var, n_dom)                  # per-domain std along w (mean 0)
    base = noise * rng.standard_normal((n, d))
    base = base - (base @ w)[:, None] * w[None, :]            # strip the isotropic w-component
    w_comp = (std_w[d_lab] * rng.standard_normal(n))[:, None] * w[None, :]   # domain-specific variance
    Z = sep_label * sign[:, None] * u[None, :] + base + w_comp
    nuis_basis = w[:, None].copy()                           # TRUE covariance carrier (invisible to means)
    label_basis = u[:, None].copy()
    if rotate:
        Q = _random_orthonormal(d, d, rng)
        Z = Z @ Q.T
        nuis_basis = Q @ nuis_basis
        label_basis = Q @ label_basis
    spec = SimpleNamespace(n_cls=2, n_dom=n_dom, d=d)
    return {"Z": Z.astype(np.float32), "y": y.astype(np.int64), "d": d_lab.astype(np.int64),
            "nuisance_basis": nuis_basis.astype(np.float32),
            "label_basis": label_basis.astype(np.float32), "spec": spec}
