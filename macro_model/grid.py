"""Structured grid helpers for the die-level thermal network."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MacroGridConfig


@dataclass(frozen=True)
class StructuredGrid:
    """Regular 3D grid with flattened node indexing."""

    nx: int
    ny: int
    nz: int
    length_x: float
    length_y: float
    thickness: float
    dx: float
    dy: float
    dz: float
    centers: np.ndarray

    @property
    def num_nodes(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def volume(self) -> float:
        return self.dx * self.dy * self.dz

    def index(self, ix: int, iy: int, iz: int) -> int:
        return (iz * self.ny + iy) * self.nx + ix

    def unravel(self, idx: int) -> tuple[int, int, int]:
        ix = idx % self.nx
        iy = (idx // self.nx) % self.ny
        iz = idx // (self.nx * self.ny)
        return ix, iy, iz

    def top_layer_mask(self) -> np.ndarray:
        mask = np.zeros(self.num_nodes, dtype=bool)
        start = self.index(0, 0, self.nz - 1)
        mask[start : start + self.nx * self.ny] = True
        return mask

    def nearest_top_node(self, x: float, y: float) -> int:
        ix = int(np.clip(np.floor(x / self.dx), 0, self.nx - 1))
        iy = int(np.clip(np.floor(y / self.dy), 0, self.ny - 1))
        return self.index(ix, iy, self.nz - 1)


def create_structured_grid(config: MacroGridConfig) -> StructuredGrid:
    """Create a regular grid and cell-center coordinate array."""

    dx = config.length_x / config.nx
    dy = config.length_y / config.ny
    dz = config.thickness / config.nz
    xs = (np.arange(config.nx) + 0.5) * dx
    ys = (np.arange(config.ny) + 0.5) * dy
    zs = (np.arange(config.nz) + 0.5) * dz
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    centers = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return StructuredGrid(
        nx=config.nx,
        ny=config.ny,
        nz=config.nz,
        length_x=config.length_x,
        length_y=config.length_y,
        thickness=config.thickness,
        dx=dx,
        dy=dy,
        dz=dz,
        centers=centers.astype(np.float64),
    )

