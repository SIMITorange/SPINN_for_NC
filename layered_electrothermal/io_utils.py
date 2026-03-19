from __future__ import annotations

import numpy as np


def generate_pulse(
    t_end_s: float,
    num_steps: int,
    vds_on_v: float,
    vgs_on_v: float,
    t_on_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_s = np.linspace(0.0, t_end_s, num_steps, dtype=float)
    vds_v = np.zeros_like(time_s)
    vgs_v = np.zeros_like(time_s)
    vds_v[time_s >= t_on_s] = vds_on_v
    vgs_v[time_s >= t_on_s] = vgs_on_v
    return time_s, vds_v, vgs_v


def load_drive_from_hdf5(hdf5_path: str, group_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("h5py not available; install requirements.txt") from exc

    with h5py.File(hdf5_path, "r") as h5f:
        if group_name not in h5f:
            raise KeyError(f"group not found: {group_name}")
        grp = h5f[group_name]
        data = np.array(grp["data"], dtype=np.float64)
    if data.shape[1] < 4:
        raise ValueError("expected columns [time, Ids, Vds, Vgs]")
    time_s = data[:, 0]
    vds_v = data[:, 2]
    vgs_v = data[:, 3]
    return time_s, vds_v, vgs_v


def macro_to_cell_map(
    macro_map: np.ndarray,
    chip_length_m: float,
    chip_width_m: float,
    cell_pitch_x_m: float,
    cell_pitch_y_m: float,
) -> np.ndarray:
    """Upsample macro temperature map to per-cell grid using nearest-neighbor repeats."""
    if macro_map.ndim == 2:
        macro = macro_map[None, ...]
    elif macro_map.ndim == 3:
        macro = macro_map
    else:
        raise ValueError("macro_map must be 2D or 3D")

    cells_x = int(round(chip_length_m / cell_pitch_x_m))
    cells_y = int(round(chip_width_m / cell_pitch_y_m))
    nx = macro.shape[1]
    ny = macro.shape[2]

    scale_x = int(np.ceil(cells_x / float(nx)))
    scale_y = int(np.ceil(cells_y / float(ny)))

    expanded = np.repeat(np.repeat(macro, scale_x, axis=1), scale_y, axis=2)
    expanded = expanded[:, :cells_x, :cells_y]
    return expanded if macro_map.ndim == 3 else expanded[0]
