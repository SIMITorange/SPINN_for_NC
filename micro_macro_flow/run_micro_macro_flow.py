"""Example CLI scaffold for the coupled micro-macro flow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from macro_model import (  # noqa: E402
    BoundaryCondition,
    MacroGridConfig,
    MacroSolverConfig,
    MacroThermalSolver,
    ThermalMaterial,
    create_structured_grid,
)
from micro_model import MicroChebyshevSolver, MicroDomainConfig, MicroMaterial, MicroSolverConfig  # noqa: E402
from micro_macro_flow.config import CouplingConfig  # noqa: E402
from micro_macro_flow.flow import MicroMacroFlow  # noqa: E402
from micro_macro_flow.layout import make_regular_cell_layout  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Micro-macro coupling scaffold.")
    parser.add_argument("--vds", type=float, default=600.0)
    parser.add_argument("--vgs", type=float, default=15.0)
    parser.add_argument("--ids-per-cell", type=float, default=1.0e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    macro_grid_cfg = MacroGridConfig()
    macro_grid = create_structured_grid(macro_grid_cfg)
    macro_solver = MacroThermalSolver(
        grid=macro_grid,
        material=ThermalMaterial(),
        boundary=BoundaryCondition(),
        config=MacroSolverConfig(),
    )
    micro_solver = MicroChebyshevSolver(
        domain=MicroDomainConfig(),
        material=MicroMaterial(),
        config=MicroSolverConfig(),
    )
    layout = make_regular_cell_layout(
        die_size=(macro_grid_cfg.length_x, macro_grid_cfg.length_y),
        cell_size=(MicroDomainConfig().length_x, MicroDomainConfig().length_y),
        margin=(0.2e-3, 0.2e-3),
    )
    flow = MicroMacroFlow(
        macro_solver=macro_solver,
        micro_solver=micro_solver,
        coupling_config=CouplingConfig(),
        e_field_surrogate=None,
    )
    cells = flow.initialize_cells(
        cell_xy=layout.cell_xy,
        vds=args.vds,
        vgs=args.vgs,
        ids_per_cell=args.ids_per_cell,
    )
    result = flow.run(cells)
    np.savez("micro_macro_reconstruction.npz", **result.reconstructed_fields)


if __name__ == "__main__":
    main()

