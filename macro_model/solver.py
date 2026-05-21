"""Die-level implicit-Euler thermal network solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

try:
    from scipy import sparse
    from scipy.sparse.linalg import cg
except ImportError:  # pragma: no cover - dense fallback is used
    sparse = None
    cg = None

from .config import BoundaryCondition, MacroSolverConfig, ThermalMaterial
from .grid import StructuredGrid


@dataclass
class MacroThermalState:
    """Current die temperature state."""

    temperature: np.ndarray
    time: float = 0.0

    def as_field(self, grid: StructuredGrid) -> np.ndarray:
        return self.temperature.reshape(grid.nz, grid.ny, grid.nx)


class MacroThermalSolver:
    """Implements the macro solver boxes from the algorithm figure."""

    def __init__(
        self,
        grid: StructuredGrid,
        material: ThermalMaterial,
        boundary: BoundaryCondition,
        config: MacroSolverConfig,
    ) -> None:
        self.grid = grid
        self.material = material
        self.boundary = boundary
        self.config = config

    def initial_state(self) -> MacroThermalState:
        temperature = np.full(
            self.grid.num_nodes, self.config.initial_temperature, dtype=np.float64
        )
        return MacroThermalState(temperature=temperature, time=0.0)

    def assemble_capacity(self, temperature: np.ndarray) -> np.ndarray:
        """Assemble diagonal `C(T)` as volumetric heat capacity per node."""

        cp = self.material.heat_capacity(temperature)
        return self.material.rho * cp * self.grid.volume

    def assemble_conductance(self, temperature: np.ndarray):
        """Assemble `G(T)` for six-neighbor finite-volume conduction."""

        grid = self.grid
        k_node = self.material.thermal_conductivity(temperature)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        diag = np.zeros(grid.num_nodes, dtype=np.float64)

        def add_pair(i: int, j: int, conductance: float) -> None:
            diag[i] += conductance
            diag[j] += conductance
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([-conductance, -conductance])

        for iz in range(grid.nz):
            for iy in range(grid.ny):
                for ix in range(grid.nx):
                    i = grid.index(ix, iy, iz)
                    if ix + 1 < grid.nx:
                        j = grid.index(ix + 1, iy, iz)
                        k = 0.5 * (k_node[i] + k_node[j])
                        add_pair(i, j, k * grid.dy * grid.dz / grid.dx)
                    if iy + 1 < grid.ny:
                        j = grid.index(ix, iy + 1, iz)
                        k = 0.5 * (k_node[i] + k_node[j])
                        add_pair(i, j, k * grid.dx * grid.dz / grid.dy)
                    if iz + 1 < grid.nz:
                        j = grid.index(ix, iy, iz + 1)
                        k = 0.5 * (k_node[i] + k_node[j])
                        add_pair(i, j, k * grid.dx * grid.dy / grid.dz)

        rows.extend(range(grid.num_nodes))
        cols.extend(range(grid.num_nodes))
        data.extend(diag.tolist())
        if sparse is not None:
            return sparse.csr_matrix((data, (rows, cols)), shape=(grid.num_nodes, grid.num_nodes))
        dense = np.zeros((grid.num_nodes, grid.num_nodes), dtype=np.float64)
        np.add.at(dense, (rows, cols), np.asarray(data, dtype=np.float64))
        return dense

    def apply_convection(self, conductance_matrix):
        """Return `G + H` and ambient source vector `b` for convection BCs."""

        grid = self.grid
        h_diag = np.zeros(grid.num_nodes, dtype=np.float64)
        ambient_source = np.zeros(grid.num_nodes, dtype=np.float64)

        for iz in range(grid.nz):
            for iy in range(grid.ny):
                for ix in range(grid.nx):
                    i = grid.index(ix, iy, iz)
                    area_h = 0.0
                    if iz == grid.nz - 1:
                        area_h += self.boundary.h_top * grid.dx * grid.dy
                    if iz == 0:
                        area_h += self.boundary.h_bottom * grid.dx * grid.dy
                    if ix == 0:
                        area_h += self.boundary.h_side * grid.dy * grid.dz
                    if ix == grid.nx - 1:
                        area_h += self.boundary.h_side * grid.dy * grid.dz
                    if iy == 0:
                        area_h += self.boundary.h_side * grid.dx * grid.dz
                    if iy == grid.ny - 1:
                        area_h += self.boundary.h_side * grid.dx * grid.dz
                    h_diag[i] = area_h
                    ambient_source[i] = area_h * self.boundary.ambient_temperature

        if sparse is not None and sparse.issparse(conductance_matrix):
            return conductance_matrix + sparse.diags(h_diag), ambient_source
        return conductance_matrix + np.diag(h_diag), ambient_source

    def map_cell_power_to_top_layer(
        self,
        cell_xy: np.ndarray,
        cell_power: np.ndarray,
    ) -> np.ndarray:
        """Map per-cell power values to nearest top-layer macro nodes."""

        power = np.zeros(self.grid.num_nodes, dtype=np.float64)
        for xy, p in zip(np.asarray(cell_xy), np.asarray(cell_power)):
            node = self.grid.nearest_top_node(float(xy[0]), float(xy[1]))
            power[node] += float(p)
        return power

    def uniform_top_power(self, total_power: float) -> np.ndarray:
        """Distribute a scalar total power evenly over the die top layer."""

        power = np.zeros(self.grid.num_nodes, dtype=np.float64)
        mask = self.grid.top_layer_mask()
        power[mask] = float(total_power) / max(int(mask.sum()), 1)
        return power

    def implicit_euler_step(
        self,
        state: MacroThermalState,
        power_vector: np.ndarray,
        dt: Optional[float] = None,
    ) -> MacroThermalState:
        """Advance one macro step with `(C/dt + G)T = C*T_old/dt + b + P`."""

        dt = float(dt or self.config.dt)
        capacity = self.assemble_capacity(state.temperature)
        conductance = self.assemble_conductance(state.temperature)
        conductance_bc, ambient_source = self.apply_convection(conductance)

        if sparse is not None and sparse.issparse(conductance_bc):
            lhs = sparse.diags(capacity / dt) + conductance_bc
            rhs = capacity * state.temperature / dt + ambient_source + power_vector
            try:
                new_temperature, info = cg(
                    lhs,
                    rhs,
                    x0=state.temperature,
                    rtol=self.config.cg_tol,
                    maxiter=self.config.cg_maxiter,
                )
            except TypeError:
                new_temperature, info = cg(
                    lhs,
                    rhs,
                    x0=state.temperature,
                    tol=self.config.cg_tol,
                    maxiter=self.config.cg_maxiter,
                )
            if info != 0:
                raise RuntimeError(f"Macro CG did not converge, scipy info={info}")
        else:
            lhs = np.diag(capacity / dt) + conductance_bc
            rhs = capacity * state.temperature / dt + ambient_source + power_vector
            new_temperature = np.linalg.solve(lhs, rhs)

        return MacroThermalState(temperature=np.asarray(new_temperature), time=state.time + dt)

    def run(
        self,
        initial_state: Optional[MacroThermalState],
        power_schedule: Iterable[np.ndarray],
        dt: Optional[float] = None,
    ) -> list[MacroThermalState]:
        """Run a provided power schedule and return all macro states."""

        state = initial_state or self.initial_state()
        states = [state]
        for power_vector in power_schedule:
            state = self.implicit_euler_step(state, power_vector=np.asarray(power_vector), dt=dt)
            states.append(state)
        return states

    def sample_temperature_at_cells(
        self,
        state: MacroThermalState,
        cell_xy: Sequence[Sequence[float]],
        top_layer: bool = True,
    ) -> np.ndarray:
        """Sample macro temperature at cell centers for micro-model feedback."""

        iz = self.grid.nz - 1 if top_layer else self.grid.nz // 2
        values = []
        for x, y in cell_xy:
            ix = int(np.clip(np.floor(float(x) / self.grid.dx), 0, self.grid.nx - 1))
            iy = int(np.clip(np.floor(float(y) / self.grid.dy), 0, self.grid.ny - 1))
            values.append(state.temperature[self.grid.index(ix, iy, iz)])
        return np.asarray(values, dtype=np.float64)
