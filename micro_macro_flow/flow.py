"""Fixed-point micro/macro electrothermal coupling flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .config import CouplingConfig
from .reconstruction import reconstruct_3d_fields


@dataclass
class CellState:
    """Operating state and optional graph data for one cell."""

    cell_id: int
    xy: np.ndarray
    vds: float
    vgs: float
    ids: float
    temperature: float
    power_loss: float = 0.0
    graph_pos: Optional[np.ndarray] = None
    graph_edge_index: Optional[np.ndarray] = None
    graph_doping: Optional[np.ndarray] = None
    current_density_xy: Optional[np.ndarray] = None
    active_mask: Optional[np.ndarray] = None


@dataclass
class MicroMacroResult:
    """Coupled iteration result."""

    converged: bool
    iterations: int
    macro_state: object
    micro_states: list
    cell_states: list[CellState]
    reconstructed_fields: dict[str, np.ndarray]
    history: list[dict[str, float]]


class MicroMacroFlow:
    """Implements steps A-E in the requested micro/macro algorithm."""

    def __init__(
        self,
        macro_solver,
        micro_solver,
        coupling_config: CouplingConfig,
        e_field_surrogate=None,
        current_density_factory: Optional[Callable[[CellState, int], np.ndarray]] = None,
    ) -> None:
        self.macro_solver = macro_solver
        self.micro_solver = micro_solver
        self.config = coupling_config
        self.e_field_surrogate = e_field_surrogate
        self.current_density_factory = current_density_factory

    def initialize_cells(
        self,
        cell_xy: np.ndarray,
        vds: float,
        vgs: float,
        ids_per_cell: float,
    ) -> list[CellState]:
        """Step A helper: initialize each cell at `T_env`."""

        cells = []
        for idx, xy in enumerate(np.asarray(cell_xy, dtype=np.float64)):
            cells.append(
                CellState(
                    cell_id=idx,
                    xy=xy,
                    vds=float(vds),
                    vgs=float(vgs),
                    ids=float(ids_per_cell),
                    temperature=self.config.initial_temperature,
                )
            )
        return cells

    def run(self, cells: Sequence[CellState]) -> MicroMacroResult:
        """Run the fixed-point loop from micro loss to macro temperature."""

        cell_states = [cell for cell in cells]
        macro_state = self.macro_solver.initial_state()
        previous_temperature = np.asarray([cell.temperature for cell in cell_states])
        previous_power = np.asarray([cell.power_loss for cell in cell_states])
        stable_count = 0
        history: list[dict[str, float]] = []
        micro_states = [self.micro_solver.initial_state(cell.temperature) for cell in cell_states]

        for iteration in range(1, self.config.max_iterations + 1):
            raw_power, micro_states = self._micro_step(cell_states, iteration)
            relaxed_power = self._relax_power(previous_power, raw_power)
            for cell, p_loss in zip(cell_states, relaxed_power):
                cell.power_loss = float(p_loss)

            power_vector = self.macro_solver.map_cell_power_to_top_layer(
                cell_xy=np.asarray([cell.xy for cell in cell_states]),
                cell_power=relaxed_power,
            )
            macro_state = self.macro_solver.implicit_euler_step(
                macro_state,
                power_vector=power_vector,
                dt=self.config.macro_dt,
            )
            new_temperature = self.macro_solver.sample_temperature_at_cells(
                macro_state,
                cell_xy=[cell.xy for cell in cell_states],
                top_layer=True,
            )
            for cell, temp in zip(cell_states, new_temperature):
                cell.temperature = float(temp)

            temp_delta = float(np.max(np.abs(new_temperature - previous_temperature)))
            power_delta = float(np.max(np.abs(relaxed_power - previous_power)))
            history.append(
                {
                    "iteration": float(iteration),
                    "max_temperature_delta": temp_delta,
                    "max_power_delta": power_delta,
                    "total_power": float(np.sum(relaxed_power)),
                    "max_temperature": float(np.max(new_temperature)),
                }
            )

            converged_now = (
                temp_delta <= self.config.temperature_tolerance
                and power_delta <= self.config.power_tolerance
            )
            stable_count = stable_count + 1 if converged_now else 0
            if stable_count >= self.config.stable_iterations:
                fields = reconstruct_3d_fields(
                    macro_state=macro_state,
                    macro_grid=self.macro_solver.grid,
                    cell_xy=np.asarray([cell.xy for cell in cell_states]),
                    micro_states=micro_states,
                    cell_power=relaxed_power,
                )
                return MicroMacroResult(
                    converged=True,
                    iterations=iteration,
                    macro_state=macro_state,
                    micro_states=micro_states,
                    cell_states=cell_states,
                    reconstructed_fields=fields,
                    history=history,
                )

            previous_temperature = new_temperature
            previous_power = relaxed_power

        fields = reconstruct_3d_fields(
            macro_state=macro_state,
            macro_grid=self.macro_solver.grid,
            cell_xy=np.asarray([cell.xy for cell in cell_states]),
            micro_states=micro_states,
            cell_power=previous_power,
        )
        return MicroMacroResult(
            converged=False,
            iterations=self.config.max_iterations,
            macro_state=macro_state,
            micro_states=micro_states,
            cell_states=cell_states,
            reconstructed_fields=fields,
            history=history,
        )

    def _micro_step(
        self,
        cells: Sequence[CellState],
        iteration: int,
    ) -> tuple[np.ndarray, list]:
        """Step B: calculate per-cell loss and update micro thermal state."""

        powers = []
        micro_states = []
        for cell in cells:
            p_loss, power_density = self._calculate_cell_power_loss(cell, iteration)
            micro_state = self.micro_solver.solve(
                initial_temperature=cell.temperature,
                power_density=power_density,
                macro_temperature=cell.temperature,
                t_max=self.config.micro_tmax,
            )
            micro_state.power_loss = p_loss
            powers.append(p_loss)
            micro_states.append(micro_state)
        return np.asarray(powers, dtype=np.float64), micro_states

    def _calculate_cell_power_loss(
        self,
        cell: CellState,
        iteration: int,
    ) -> tuple[float, np.ndarray]:
        """Calculate `P_loss` from GNN Ex/Ey when graph data is available."""

        if (
            self.e_field_surrogate is not None
            and cell.graph_pos is not None
            and cell.graph_edge_index is not None
            and cell.graph_doping is not None
        ):
            exey = self.e_field_surrogate.predict_arrays(
                pos=cell.graph_pos,
                edge_index=cell.graph_edge_index,
                doping=cell.graph_doping,
                vds=cell.vds,
                vgs=cell.vgs,
                temperature=cell.temperature,
                die_xy=cell.xy,
            )
            current_density = self._current_density(cell, iteration, exey.shape[0])
            p_loss, density = self.micro_solver.calculate_power_loss(
                electric_field_xy=exey,
                current_density_xy=current_density,
                active_mask=cell.active_mask,
            )
            return p_loss, density

        p_loss = max(cell.vds * cell.ids, 0.0)
        density = self.micro_solver.distribute_scalar_power(p_loss, cell.active_mask)
        return p_loss, density

    def _current_density(
        self,
        cell: CellState,
        iteration: int,
        num_nodes: int,
    ) -> np.ndarray:
        if cell.current_density_xy is not None:
            return np.asarray(cell.current_density_xy, dtype=np.float64)
        if self.current_density_factory is not None:
            return np.asarray(self.current_density_factory(cell, iteration), dtype=np.float64)
        area = self.micro_solver.domain.length_y * self.micro_solver.domain.thickness
        jx = cell.ids / max(area, 1.0e-30)
        return np.column_stack([np.full(num_nodes, jx), np.zeros(num_nodes)])

    def _relax_power(self, previous_power: np.ndarray, raw_power: np.ndarray) -> np.ndarray:
        alpha = float(np.clip(self.config.power_relaxation, 0.0, 1.0))
        if previous_power.shape != raw_power.shape or not np.any(previous_power):
            return raw_power
        return alpha * raw_power + (1.0 - alpha) * previous_power

