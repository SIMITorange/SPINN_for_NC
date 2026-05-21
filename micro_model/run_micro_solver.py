"""Example CLI scaffold for the cell-level Chebyshev micro solver."""

from __future__ import annotations

import argparse

import numpy as np

from config import MicroDomainConfig, MicroMaterial, MicroSolverConfig
from solver import MicroChebyshevSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Micro Chebyshev solver scaffold.")
    parser.add_argument("--power-loss", type=float, default=1.0e-3, help="Scalar cell loss W.")
    parser.add_argument("--macro-temperature", type=float, default=300.15, help="Local macro Tnode K.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    solver = MicroChebyshevSolver(
        domain=MicroDomainConfig(),
        material=MicroMaterial(),
        config=MicroSolverConfig(),
    )
    power_density = solver.distribute_scalar_power(args.power_loss)
    state = solver.solve(
        initial_temperature=args.macro_temperature,
        power_density=power_density,
        macro_temperature=args.macro_temperature,
    )
    np.savez(
        "micro_temperature_field.npz",
        temperature=state.temperature,
        nodes=solver.physical_nodes,
        time=state.time,
        k_sic=state.k_sic,
        c_sic=state.c_sic,
    )


if __name__ == "__main__":
    main()

