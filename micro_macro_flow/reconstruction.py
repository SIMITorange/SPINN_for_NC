"""Reconstruct coupled 3D fields after micro/macro convergence."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def reconstruct_3d_fields(
    macro_state,
    macro_grid,
    cell_xy: np.ndarray,
    micro_states: Sequence,
    cell_power: np.ndarray,
) -> dict[str, np.ndarray]:
    """Collect macro and micro fields into one post-processing dictionary."""

    macro_temperature = macro_state.as_field(macro_grid)
    micro_temperature = np.asarray([state.temperature for state in micro_states])
    micro_k = np.asarray([state.k_sic for state in micro_states], dtype=np.float64)
    micro_c = np.asarray([state.c_sic for state in micro_states], dtype=np.float64)
    return {
        "macro_temperature_zyx": macro_temperature,
        "macro_centers_xyz": macro_grid.centers,
        "cell_xy": np.asarray(cell_xy, dtype=np.float64),
        "cell_power": np.asarray(cell_power, dtype=np.float64),
        "micro_temperature": micro_temperature,
        "micro_k_sic": micro_k,
        "micro_c_sic": micro_c,
        "time": np.asarray([macro_state.time], dtype=np.float64),
    }

