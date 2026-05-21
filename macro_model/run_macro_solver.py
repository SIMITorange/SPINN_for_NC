"""Example CLI scaffold for the die-level macro thermal solver."""

from __future__ import annotations

import argparse

import numpy as np

from config import BoundaryCondition, MacroGridConfig, MacroSolverConfig, ThermalMaterial
from grid import create_structured_grid
from solver import MacroThermalSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Macro thermal solver scaffold.")
    parser.add_argument("--total-power", type=float, default=1.0, help="Total top-layer power W.")
    parser.add_argument("--steps", type=int, default=1, help="Number of implicit Euler steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = create_structured_grid(MacroGridConfig())
    solver = MacroThermalSolver(
        grid=grid,
        material=ThermalMaterial(),
        boundary=BoundaryCondition(),
        config=MacroSolverConfig(),
    )
    state = solver.initial_state()
    power = solver.uniform_top_power(args.total_power)
    for _ in range(args.steps):
        state = solver.implicit_euler_step(state, power)
    np.savez(
        "macro_temperature_field.npz",
        temperature=state.as_field(grid),
        centers=grid.centers,
        time=state.time,
    )


if __name__ == "__main__":
    main()

