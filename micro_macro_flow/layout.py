"""Cell placement helpers connecting die-level nodes to cell-level solvers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CellLayout:
    """Cell centers and approximate footprint on the macro die."""

    cell_xy: np.ndarray
    cell_size: tuple[float, float]
    die_size: tuple[float, float]

    @property
    def num_cells(self) -> int:
        return int(self.cell_xy.shape[0])


def make_regular_cell_layout(
    die_size: tuple[float, float],
    cell_size: tuple[float, float],
    margin: tuple[float, float] = (0.0, 0.0),
) -> CellLayout:
    """Generate a regular cell grid over the active die region."""

    die_x, die_y = die_size
    cell_x, cell_y = cell_size
    margin_x, margin_y = margin
    xs = np.arange(margin_x + 0.5 * cell_x, die_x - margin_x, cell_x)
    ys = np.arange(margin_y + 0.5 * cell_y, die_y - margin_y, cell_y)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    cell_xy = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    return CellLayout(cell_xy=cell_xy, cell_size=cell_size, die_size=die_size)

