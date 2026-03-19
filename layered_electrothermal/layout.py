from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VoidRegion:
    x0_m: float
    y0_m: float
    width_m: float
    height_m: float

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x >= self.x0_m) & (x < self.x0_m + self.width_m) & (y >= self.y0_m) & (y < self.y0_m + self.height_m)


def grid_size_from_pitch(length_m: float, pitch_m: float) -> int:
    n = int(round(length_m / pitch_m))
    if n <= 0:
        raise ValueError("grid size must be positive")
    return n


def build_active_mask(
    chip_length_m: float,
    chip_width_m: float,
    nx: int,
    ny: int,
    voids: tuple[VoidRegion, ...] | None = None,
) -> np.ndarray:
    if nx <= 0 or ny <= 0:
        raise ValueError("nx/ny must be positive")
    x_centers = (np.arange(nx, dtype=float) + 0.5) * (chip_length_m / float(nx))
    y_centers = (np.arange(ny, dtype=float) + 0.5) * (chip_width_m / float(ny))
    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")

    mask = np.ones((nx, ny), dtype=bool)
    if voids:
        for void in voids:
            mask &= ~void.contains(X, Y)
    return mask


def count_cells(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))
