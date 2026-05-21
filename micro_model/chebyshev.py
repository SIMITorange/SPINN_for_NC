"""Chebyshev-Gauss-Lobatto nodes and tensor-product operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChebyshevOperators:
    """Dense tensor-product spectral operators on normalized coordinates."""

    nodes_x: np.ndarray
    nodes_y: np.ndarray
    nodes_z: np.ndarray
    weights: np.ndarray
    mass: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    dz: np.ndarray
    laplacian: np.ndarray
    boundary_operator: np.ndarray


def cgl_nodes(order: int) -> np.ndarray:
    """Return Chebyshev-Gauss-Lobatto nodes in ascending order."""

    if order < 2:
        raise ValueError("CGL order must be >= 2")
    k = np.arange(order + 1)
    return np.cos(np.pi * k / order)[::-1].astype(np.float64)


def differentiation_matrix(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return first derivative matrix and CGL nodes."""

    x_desc = np.cos(np.pi * np.arange(order + 1) / order)
    c = np.ones(order + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** np.arange(order + 1))
    x_i = x_desc[:, None]
    x_j = x_desc[None, :]
    d_x = x_i - x_j
    d = (c[:, None] / c[None, :]) / (d_x + np.eye(order + 1))
    d = d - np.diag(np.sum(d, axis=1))

    reverse = np.arange(order, -1, -1)
    d_asc = -d[np.ix_(reverse, reverse)]
    x_asc = x_desc[reverse]
    return d_asc.astype(np.float64), x_asc.astype(np.float64)


def clenshaw_curtis_like_weights(nodes: np.ndarray) -> np.ndarray:
    """Simple positive integration weights on CGL nodes."""

    x = np.asarray(nodes, dtype=np.float64)
    weights = np.zeros_like(x)
    weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    return np.abs(weights)


def build_chebyshev_operators(
    order_x: int,
    order_y: int,
    order_z: int,
    length_x: float,
    length_y: float,
    thickness: float,
) -> ChebyshevOperators:
    """Build tensor-product derivative and Laplacian matrices."""

    d_xi, nx = differentiation_matrix(order_x)
    d_eta, ny = differentiation_matrix(order_y)
    d_zeta, nz = differentiation_matrix(order_z)
    sx = 2.0 / length_x
    sy = 2.0 / length_y
    sz = 2.0 / thickness

    ix = np.eye(order_x + 1)
    iy = np.eye(order_y + 1)
    iz = np.eye(order_z + 1)
    dx = np.kron(np.kron(iz, iy), sx * d_xi)
    dy = np.kron(np.kron(iz, sy * d_eta), ix)
    dz = np.kron(np.kron(sz * d_zeta, iy), ix)
    laplacian = dx @ dx + dy @ dy + dz @ dz

    wx = clenshaw_curtis_like_weights(nx) * (length_x / 2.0)
    wy = clenshaw_curtis_like_weights(ny) * (length_y / 2.0)
    wz = clenshaw_curtis_like_weights(nz) * (thickness / 2.0)
    weights = np.kron(np.kron(wz, wy), wx)
    mass = np.diag(weights)

    boundary = np.zeros_like(mass)
    shape = (order_z + 1, order_y + 1, order_x + 1)
    for flat in range(np.prod(shape)):
        iz_i, iy_i, ix_i = np.unravel_index(flat, shape)
        if (
            ix_i in (0, order_x)
            or iy_i in (0, order_y)
            or iz_i in (0, order_z)
        ):
            boundary[flat, flat] = 1.0

    return ChebyshevOperators(
        nodes_x=nx,
        nodes_y=ny,
        nodes_z=nz,
        weights=weights,
        mass=mass,
        dx=dx,
        dy=dy,
        dz=dz,
        laplacian=laplacian,
        boundary_operator=boundary,
    )

