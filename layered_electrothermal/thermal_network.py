from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ChipGeometry, ThermalBC, LayerConfig
from .material import ThermalMaterial


@dataclass(frozen=True)
class Layer:
    name: str
    thickness_m: float
    material: ThermalMaterial


class ThermalNetwork:
    def __init__(self, chip: ChipGeometry, layers: list[Layer], bc: ThermalBC) -> None:
        self.chip = chip
        self.layers = layers
        self.bc = bc

        self.nx = chip.macro_nx
        self.ny = chip.macro_ny
        self.dx = chip.macro_dx_m
        self.dy = chip.macro_dy_m
        self.n_layers = len(layers)
        self.node_count = self.n_layers * self.nx * self.ny

    def index(self, layer: int, ix: int, iy: int) -> int:
        return (layer * self.nx * self.ny) + (ix * self.ny) + iy

    def reshape(self, vec: np.ndarray) -> np.ndarray:
        return vec.reshape(self.n_layers, self.nx, self.ny)

    def flatten(self, field: np.ndarray) -> np.ndarray:
        return field.reshape(self.node_count)

    def initial_temperature(self, temp_k: float) -> np.ndarray:
        return np.full((self.n_layers, self.nx, self.ny), temp_k, dtype=float)

    def solve_step(
        self,
        T_prev: np.ndarray,
        power_top_w: np.ndarray,
        dt_s: float,
        T_props: np.ndarray | None = None,
    ) -> np.ndarray:
        """Implicit Euler thermal step with temperature-dependent properties."""
        if T_prev.shape != (self.n_layers, self.nx, self.ny):
            raise ValueError("T_prev shape mismatch")
        if power_top_w.shape != (self.nx, self.ny):
            raise ValueError("power_top_w shape mismatch")

        T_for_props = T_prev if T_props is None else T_props
        if T_for_props.shape != T_prev.shape:
            raise ValueError("T_props shape mismatch")

        k_map = np.zeros_like(T_prev, dtype=float)
        cp_map = np.zeros_like(T_prev, dtype=float)
        for lid, layer in enumerate(self.layers):
            k_map[lid] = layer.material.conductivity_w_mk(T_for_props[lid])
            cp_map[lid] = layer.material.heat_capacity_j_kgk(T_for_props[lid])

        G = np.zeros((self.node_count, self.node_count), dtype=float)
        C = np.zeros(self.node_count, dtype=float)
        b = np.zeros(self.node_count, dtype=float)

        area_x = self.dy
        area_y = self.dx
        area_z = self.dx * self.dy

        for lid, layer in enumerate(self.layers):
            vol = area_z * layer.thickness_m
            rho = layer.material.rho_kg_m3
            for ix in range(self.nx):
                for iy in range(self.ny):
                    idx = self.index(lid, ix, iy)
                    C[idx] = rho * cp_map[lid, ix, iy] * vol

        # Lateral conductance inside each layer.
        for lid, layer in enumerate(self.layers):
            t = layer.thickness_m
            for ix in range(self.nx):
                for iy in range(self.ny):
                    idx = self.index(lid, ix, iy)
                    if ix + 1 < self.nx:
                        jdx = self.index(lid, ix + 1, iy)
                        k_eff = 2.0 / (1.0 / k_map[lid, ix, iy] + 1.0 / k_map[lid, ix + 1, iy])
                        g = k_eff * (area_x * t) / self.dx
                        G[idx, idx] += g
                        G[jdx, jdx] += g
                        G[idx, jdx] -= g
                        G[jdx, idx] -= g
                    if iy + 1 < self.ny:
                        jdx = self.index(lid, ix, iy + 1)
                        k_eff = 2.0 / (1.0 / k_map[lid, ix, iy] + 1.0 / k_map[lid, ix, iy + 1])
                        g = k_eff * (area_y * t) / self.dy
                        G[idx, idx] += g
                        G[jdx, jdx] += g
                        G[idx, jdx] -= g
                        G[jdx, idx] -= g

        # Vertical conductance between layers.
        for lid in range(self.n_layers - 1):
            t1 = self.layers[lid].thickness_m
            t2 = self.layers[lid + 1].thickness_m
            for ix in range(self.nx):
                for iy in range(self.ny):
                    idx = self.index(lid, ix, iy)
                    jdx = self.index(lid + 1, ix, iy)
                    k1 = k_map[lid, ix, iy]
                    k2 = k_map[lid + 1, ix, iy]
                    denom = 0.5 * t1 / k1 + 0.5 * t2 / k2
                    g = area_z / np.clip(denom, 1.0e-12, None)
                    G[idx, idx] += g
                    G[jdx, jdx] += g
                    G[idx, jdx] -= g
                    G[jdx, idx] -= g

        # Convection boundaries.
        for ix in range(self.nx):
            for iy in range(self.ny):
                top = self.index(0, ix, iy)
                bottom = self.index(self.n_layers - 1, ix, iy)

                g_top = self.bc.h_top_w_m2k * area_z
                G[top, top] += g_top
                b[top] += g_top * self.bc.ambient_temp_k

                g_bot = self.bc.h_bottom_w_m2k * area_z
                G[bottom, bottom] += g_bot
                b[bottom] += g_bot * self.bc.ambient_temp_k

        # Power source on top layer nodes.
        power_vec = np.zeros(self.node_count, dtype=float)
        for ix in range(self.nx):
            for iy in range(self.ny):
                idx = self.index(0, ix, iy)
                power_vec[idx] = power_top_w[ix, iy]

        T_prev_vec = self.flatten(T_prev)
        A = np.diag(C / dt_s) + G
        b = b + (C / dt_s) * T_prev_vec + power_vec

        T_next_vec = np.linalg.solve(A, b)
        return self.reshape(T_next_vec)


def build_layers(layer_cfgs: list[LayerConfig], materials: dict[str, ThermalMaterial]) -> list[Layer]:
    layers: list[Layer] = []
    for cfg in layer_cfgs:
        if cfg.material_key not in materials:
            raise KeyError(f"Unknown material_key: {cfg.material_key}")
        layers.append(Layer(name=cfg.name, thickness_m=cfg.thickness_m, material=materials[cfg.material_key]))
    return layers


class SparseThermalNetwork:
    def __init__(
        self,
        chip: ChipGeometry,
        layers: list[Layer],
        bc: ThermalBC,
        active_mask: np.ndarray | None = None,
        solver: str = "cg",
        cg_tol: float = 1.0e-6,
        cg_maxiter: int = 2000,
    ) -> None:
        self.chip = chip
        self.layers = layers
        self.bc = bc
        self.nx = chip.macro_nx
        self.ny = chip.macro_ny
        self.dx = chip.macro_dx_m
        self.dy = chip.macro_dy_m
        self.n_layers = len(layers)

        if active_mask is None:
            active_mask = np.ones((self.nx, self.ny), dtype=bool)
        if active_mask.shape != (self.nx, self.ny):
            raise ValueError("active_mask shape mismatch")

        self.active_mask = active_mask
        self.active_index = -np.ones((self.n_layers, self.nx, self.ny), dtype=int)
        idx = 0
        for lid in range(self.n_layers):
            for ix in range(self.nx):
                for iy in range(self.ny):
                    if active_mask[ix, iy]:
                        self.active_index[lid, ix, iy] = idx
                        idx += 1
        self.node_count = idx

        self.solver = solver
        self.cg_tol = cg_tol
        self.cg_maxiter = cg_maxiter

    def initial_temperature(self, temp_k: float) -> np.ndarray:
        field = np.full((self.n_layers, self.nx, self.ny), temp_k, dtype=float)
        return np.where(self.active_mask[None, :, :], field, self.bc.ambient_temp_k)

    def _iter_active_nodes(self):
        for lid in range(self.n_layers):
            for ix in range(self.nx):
                for iy in range(self.ny):
                    idx = self.active_index[lid, ix, iy]
                    if idx >= 0:
                        yield lid, ix, iy, idx

    def solve_step(
        self,
        T_prev: np.ndarray,
        power_top_w: np.ndarray,
        dt_s: float,
        T_props: np.ndarray | None = None,
    ) -> np.ndarray:
        if T_prev.shape != (self.n_layers, self.nx, self.ny):
            raise ValueError("T_prev shape mismatch")
        if power_top_w.shape != (self.nx, self.ny):
            raise ValueError("power_top_w shape mismatch")
        if dt_s <= 0.0:
            return T_prev.copy()

        try:
            from scipy import sparse
            from scipy.sparse import linalg as splinalg
        except ImportError as exc:
            raise RuntimeError("scipy is required for SparseThermalNetwork") from exc

        T_for_props = T_prev if T_props is None else T_props
        if T_for_props.shape != T_prev.shape:
            raise ValueError("T_props shape mismatch")

        # Replace inactive cells with ambient for property evaluation.
        T_eval = np.where(self.active_mask[None, :, :], T_for_props, self.bc.ambient_temp_k)

        k_map = np.zeros_like(T_prev, dtype=float)
        cp_map = np.zeros_like(T_prev, dtype=float)
        for lid, layer in enumerate(self.layers):
            k_map[lid] = layer.material.conductivity_w_mk(T_eval[lid])
            cp_map[lid] = layer.material.heat_capacity_j_kgk(T_eval[lid])

        area_x = self.dy
        area_y = self.dx
        area_z = self.dx * self.dy

        C = np.zeros(self.node_count, dtype=float)
        b = np.zeros(self.node_count, dtype=float)
        T_prev_vec = np.zeros(self.node_count, dtype=float)

        for lid, ix, iy, idx in self._iter_active_nodes():
            layer = self.layers[lid]
            vol = area_z * layer.thickness_m
            rho = layer.material.rho_kg_m3
            C[idx] = rho * cp_map[lid, ix, iy] * vol
            T_prev_vec[idx] = T_prev[lid, ix, iy]

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        diag = np.zeros(self.node_count, dtype=float)

        def add_conductance(i: int, j: int, g: float) -> None:
            diag[i] += g
            diag[j] += g
            rows.append(i)
            cols.append(j)
            data.append(-g)
            rows.append(j)
            cols.append(i)
            data.append(-g)

        # Lateral conductance in each layer.
        for lid, ix, iy, idx in self._iter_active_nodes():
            if ix + 1 < self.nx and self.active_mask[ix + 1, iy]:
                jdx = self.active_index[lid, ix + 1, iy]
                k_eff = 2.0 / (1.0 / k_map[lid, ix, iy] + 1.0 / k_map[lid, ix + 1, iy])
                g = k_eff * (area_x * self.layers[lid].thickness_m) / self.dx
                add_conductance(idx, jdx, g)
            if iy + 1 < self.ny and self.active_mask[ix, iy + 1]:
                jdx = self.active_index[lid, ix, iy + 1]
                k_eff = 2.0 / (1.0 / k_map[lid, ix, iy] + 1.0 / k_map[lid, ix, iy + 1])
                g = k_eff * (area_y * self.layers[lid].thickness_m) / self.dy
                add_conductance(idx, jdx, g)

        # Vertical conductance between layers.
        for lid in range(self.n_layers - 1):
            t1 = self.layers[lid].thickness_m
            t2 = self.layers[lid + 1].thickness_m
            for ix in range(self.nx):
                for iy in range(self.ny):
                    if not self.active_mask[ix, iy]:
                        continue
                    idx = self.active_index[lid, ix, iy]
                    jdx = self.active_index[lid + 1, ix, iy]
                    k1 = k_map[lid, ix, iy]
                    k2 = k_map[lid + 1, ix, iy]
                    denom = 0.5 * t1 / k1 + 0.5 * t2 / k2
                    g = area_z / np.clip(denom, 1.0e-12, None)
                    add_conductance(idx, jdx, g)

        # Convection boundaries.
        for ix in range(self.nx):
            for iy in range(self.ny):
                if not self.active_mask[ix, iy]:
                    continue
                top = self.active_index[0, ix, iy]
                bottom = self.active_index[self.n_layers - 1, ix, iy]
                g_top = self.bc.h_top_w_m2k * area_z
                diag[top] += g_top
                b[top] += g_top * self.bc.ambient_temp_k
                g_bot = self.bc.h_bottom_w_m2k * area_z
                diag[bottom] += g_bot
                b[bottom] += g_bot * self.bc.ambient_temp_k

        # Power source on top layer.
        for ix in range(self.nx):
            for iy in range(self.ny):
                if not self.active_mask[ix, iy]:
                    continue
                idx = self.active_index[0, ix, iy]
                b[idx] += power_top_w[ix, iy]

        A = sparse.csr_matrix(
            (np.concatenate([data, diag]), (rows + list(range(self.node_count)), cols + list(range(self.node_count)))),
            shape=(self.node_count, self.node_count),
        )
        rhs = b + (C / dt_s) * T_prev_vec
        A = A + sparse.diags(C / dt_s)

        if self.solver == "cg":
            T_next_vec, info = splinalg.cg(A, rhs, atol=0.0, tol=self.cg_tol, maxiter=self.cg_maxiter)
            if info != 0:
                T_next_vec = splinalg.spsolve(A, rhs)
        elif self.solver == "spsolve":
            T_next_vec = splinalg.spsolve(A, rhs)
        else:
            raise ValueError("solver must be 'cg' or 'spsolve'")

        T_next = np.full((self.n_layers, self.nx, self.ny), self.bc.ambient_temp_k, dtype=float)
        for lid, ix, iy, idx in self._iter_active_nodes():
            T_next[lid, ix, iy] = T_next_vec[idx]
        return T_next
