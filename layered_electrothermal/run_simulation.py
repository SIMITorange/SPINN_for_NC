from __future__ import annotations

import argparse

import numpy as np

from .config import CellGeometry, ChipGeometry, CouplingConfig, LayerConfig, SimulationConfig, TimeConfig
from .coupling import CoupledElectroThermalSimulator
from .io_utils import generate_pulse, load_drive_from_hdf5, macro_to_cell_map
from .layout import VoidRegion, build_active_mask, count_cells, grid_size_from_pitch
from .material import default_materials
from .micro_cell import CellElectricalModel, ElectricalParams
from .thermal_network import SparseThermalNetwork, ThermalNetwork, build_layers


def build_default_simulation(
    grid_nx: int,
    grid_ny: int,
    power_mask: np.ndarray,
    coupling_cfg: CouplingConfig,
    time_cfg: TimeConfig,
    current_area_mode: str = "chip",
    current_area_m2: float | None = None,
    solver: str = "cg",
    use_sparse: bool = True,
    cg_tol: float = 1.0e-6,
    cg_maxiter: int = 2000,
) -> CoupledElectroThermalSimulator:
    chip = ChipGeometry(macro_nx=grid_nx, macro_ny=grid_ny)
    cfg = SimulationConfig(
        chip=chip,
        coupling=coupling_cfg,
        time=time_cfg,
    )

    materials = default_materials()
    layer_cfgs = [
        LayerConfig(name="epi", thickness_m=1.0e-5, material_key="sic"),
        LayerConfig(name="drift", thickness_m=8.0e-5, material_key="sic"),
        LayerConfig(name="substrate", thickness_m=3.0e-4, material_key="sic"),
    ]
    layers = build_layers(layer_cfgs, materials)
    if use_sparse:
        thermal = SparseThermalNetwork(
            cfg.chip,
            layers,
            cfg.bc,
            active_mask=None,
            solver=solver,
            cg_tol=cg_tol,
            cg_maxiter=cg_maxiter,
        )
    else:
        thermal = ThermalNetwork(cfg.chip, layers, cfg.bc)

    active_fraction = float(np.count_nonzero(power_mask)) / float(power_mask.size)
    chip_active_area_m2 = cfg.chip.chip_area_m2 * cfg.cell.active_fill * active_fraction

    if current_area_m2 is not None:
        area = current_area_m2
    elif current_area_mode == "cell":
        area = cfg.cell.active_area_m2
    elif current_area_mode == "chip":
        area = chip_active_area_m2
    else:
        raise ValueError("current_area_mode must be 'chip' or 'cell'")
    if area <= 0.0:
        raise ValueError("current_area_m2 must be positive")

    electrical = CellElectricalModel(ElectricalParams(), cfg.cell, current_area_m2=area)
    return CoupledElectroThermalSimulator(cfg, electrical, thermal, power_mask=power_mask)


def main() -> None:
    parser = argparse.ArgumentParser(description="Layered electro-thermal transient simulator")
    parser.add_argument("--hdf5", type=str, default="", help="Optional HDF5 input path")
    parser.add_argument("--group", type=str, default="", help="Group name in HDF5")
    parser.add_argument("--t-end", type=float, default=2.0e-6, help="End time in seconds")
    parser.add_argument("--steps", type=int, default=400, help="Number of time steps")
    parser.add_argument("--vds", type=float, default=800.0, help="Vds step in volts")
    parser.add_argument("--vgs", type=float, default=18.0, help="Vgs step in volts")
    parser.add_argument("--t-on", type=float, default=2.0e-7, help="Pulse turn-on time in seconds")
    parser.add_argument("--out", type=str, default="layered_electrothermal_output.npz", help="Output npz path")
    parser.add_argument("--export-cell-map", action="store_true", help="Export per-cell top-layer map")
    parser.add_argument(
        "--grid-mode",
        type=str,
        choices=["macro", "cell"],
        default="cell",
        help="Grid resolution: macro (coarse) or cell (10um pitch)",
    )
    parser.add_argument("--grid-nx", type=int, default=35, help="Macro grid Nx (used when grid-mode=macro)")
    parser.add_argument("--grid-ny", type=int, default=35, help="Macro grid Ny (used when grid-mode=macro)")
    parser.add_argument("--void-size-um", type=float, default=530.0, help="Square void size in um")
    parser.add_argument("--void-x0-um", type=float, default=None, help="Void x0 (um), default 0")
    parser.add_argument("--void-y0-um", type=float, default=None, help="Void y0 (um), default centered")
    parser.add_argument("--no-void", action="store_true", help="Disable the 530um void region")
    parser.add_argument("--no-sparse", action="store_true", help="Disable sparse thermal solver")
    parser.add_argument(
        "--solver",
        type=str,
        choices=["cg", "spsolve"],
        default="cg",
        help="Sparse solver backend (cg recommended for large grids)",
    )
    parser.add_argument("--cg-tol", type=float, default=1.0e-6, help="CG solver tolerance")
    parser.add_argument("--cg-maxiter", type=int, default=2000, help="CG solver max iterations")
    parser.add_argument("--coupling-max-iters", type=int, default=40, help="Max micro-macro iterations")
    parser.add_argument("--coupling-tol", type=float, default=5.0e-4, help="Coupling temperature tolerance (K)")
    parser.add_argument("--relax-init", type=float, default=0.5, help="Coupling initial relaxation factor")
    parser.add_argument("--relax-min", type=float, default=0.1, help="Coupling minimum relaxation factor")
    parser.add_argument("--relax-max", type=float, default=0.9, help="Coupling maximum relaxation factor")
    parser.add_argument("--max-dt", type=float, default=2.0e-9, help="Thermal max dt before sub-stepping")
    parser.add_argument("--max-substeps", type=int, default=200, help="Thermal max substeps per time step")
    parser.add_argument(
        "--current-area-mode",
        type=str,
        choices=["chip", "cell"],
        default="chip",
        help="Interpret Ids as chip or cell current",
    )
    parser.add_argument(
        "--current-area-m2",
        type=float,
        default=None,
        help="Override current reference area in m^2",
    )
    args = parser.parse_args()

    chip_geo = ChipGeometry()
    cell_geo = CellGeometry()
    chip_length_um = chip_geo.chip_length_m * 1.0e6
    chip_width_um = chip_geo.chip_width_m * 1.0e6
    cell_pitch_x_um = cell_geo.cell_pitch_x_m * 1.0e6
    cell_pitch_y_um = cell_geo.cell_pitch_y_m * 1.0e6

    if args.grid_mode == "cell":
        grid_nx = grid_size_from_pitch(chip_length_um * 1.0e-6, cell_pitch_x_um * 1.0e-6)
        grid_ny = grid_size_from_pitch(chip_width_um * 1.0e-6, cell_pitch_y_um * 1.0e-6)
    else:
        grid_nx = args.grid_nx
        grid_ny = args.grid_ny

    voids = ()
    if not args.no_void:
        void_size_m = args.void_size_um * 1.0e-6
        if args.void_x0_um is None:
            void_x0_m = 0.0
        else:
            void_x0_m = args.void_x0_um * 1.0e-6
        if args.void_y0_um is None:
            void_y0_m = (chip_width_um * 1.0e-6 - void_size_m) * 0.5
        else:
            void_y0_m = args.void_y0_um * 1.0e-6
        voids = (VoidRegion(x0_m=void_x0_m, y0_m=void_y0_m, width_m=void_size_m, height_m=void_size_m),)

    power_mask = build_active_mask(chip_length_um * 1.0e-6, chip_width_um * 1.0e-6, grid_nx, grid_ny, voids)
    power_cells = count_cells(power_mask)
    total_cells = int(power_mask.size)
    print(f"Grid: {grid_nx} x {grid_ny} (power cells: {power_cells}/{total_cells})")

    coupling_cfg = CouplingConfig(
        max_iters=args.coupling_max_iters,
        temp_tol_k=args.coupling_tol,
        relax_init=args.relax_init,
        relax_min=args.relax_min,
        relax_max=args.relax_max,
    )
    time_cfg = TimeConfig(max_dt_s=args.max_dt, max_substeps=args.max_substeps)
    use_sparse = True if args.grid_mode == "cell" else not args.no_sparse
    sim = build_default_simulation(
        grid_nx,
        grid_ny,
        power_mask,
        coupling_cfg,
        time_cfg,
        current_area_mode=args.current_area_mode,
        current_area_m2=args.current_area_m2,
        solver=args.solver,
        use_sparse=use_sparse,
        cg_tol=args.cg_tol,
        cg_maxiter=args.cg_maxiter,
    )

    if args.hdf5:
        if not args.group:
            raise SystemExit("--group is required when --hdf5 is provided")
        time_s, vds_v, vgs_v = load_drive_from_hdf5(args.hdf5, args.group)
    else:
        time_s, vds_v, vgs_v = generate_pulse(args.t_end, args.steps, args.vds, args.vgs, args.t_on)

    result = sim.simulate(time_s, vds_v, vgs_v, initial_temp_k=sim.cfg.bc.ambient_temp_k)

    output = {
        "time_s": time_s,
        "temperatures_k": result.temperatures_k,
        "power_top_w": result.power_top_w,
        "current_density_a_m2": result.current_density_a_m2,
        "vds_v": vds_v,
        "vgs_v": vgs_v,
    }
    top_temp_k = result.temperatures_k[:, 0, :, :]
    output["top_layer_temperature_k"] = top_temp_k
    if args.export_cell_map:
        if args.grid_mode == "cell":
            output["top_layer_cell_temperature_k"] = top_temp_k
        else:
            output["top_layer_cell_temperature_k"] = macro_to_cell_map(
                top_temp_k,
                sim.cfg.chip.chip_length_m,
                sim.cfg.chip.chip_width_m,
                sim.cfg.cell.cell_pitch_x_m,
                sim.cfg.cell.cell_pitch_y_m,
            )

    np.savez(args.out, **output)

    t_max = float(np.max(result.temperatures_k))
    print(f"Done. Max temperature: {t_max:.2f} K. Output: {args.out}")


if __name__ == "__main__":
    main()
