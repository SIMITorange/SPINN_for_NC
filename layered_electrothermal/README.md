# Layered Electro-Thermal Transient Simulator

This folder contains a structured, multi-file implementation of a layered electro-thermal
transient simulator based on the physics and thermal relations in
`PINN_short_circuit_no_temperature.py`.

## Model Structure
- Micro scale (cell proxy): electrical model evaluates Ids(T, Vds, Vgs) for one cell.
- Macro scale (thermal network): layered RC network (Nx x Ny x layers) with
  temperature-dependent conductivity and heat capacity.
- Coupling loop: micro power feeds macro temperature, and macro temperature
  feeds micro current density, iterated per time step.

## Convergence Techniques Used
- Fixed-point micro-macro iteration with under-relaxation.
- Adaptive relaxation (reduces when residual grows).
- Implicit Euler for the thermal step (stable for stiff RC networks).
- Sub-stepping when dt is larger than the configured stability budget.
- Temperature clamping for material property evaluation.

These are common practices to stabilize multi-physics coupling: robust time
integration, residual-based iteration, and cautious damping.

## Files
- `config.py`: geometry, BC, time, coupling configs.
- `material.py`: temperature-dependent k(T), Cp(T) from the existing script.
- `micro_cell.py`: electrical equation and current-density conversion.
- `thermal_network.py`: layered thermal RC network solver.
- `coupling.py`: micro-macro coupling loop and transient simulation driver.
- `io_utils.py`: waveform generator and optional HDF5 loader.
- `run_simulation.py`: CLI entry for running a transient case.

## Usage
From repo root:
```bash
python -m layered_electrothermal.run_simulation
```

With data from HDF5 (group name required):
```bash
python -m layered_electrothermal.run_simulation --hdf5 combined_training_data.h5 --group <GROUP>
```

Output is an `.npz` file with time, temperatures, power, and current density.
It also includes `top_layer_temperature_k` for convenient 2D visualization.
Add `--export-cell-map` to write a per-cell top-layer temperature map (upsampled
from the macro grid).

Quick plotting:
```bash
python -m layered_electrothermal.plot_results --input layered_electrothermal_output.npz
```
This writes PNGs to `layered_electrothermal_plots/`.

### Grid Resolution
`run_simulation` defaults to a cell grid using the 10um pitch (350 x 350 for a
3.5mm chip). Use `--grid-mode macro` to run a coarser grid (faster), or set
`--grid-nx/--grid-ny`.

The 530um square void is enabled by default (x0=0, y0 centered). It only affects
power injection (no electrical loss there), but the region still participates in
thermal conduction and heat capacity. Disable it with `--no-void` or override its
position using `--void-x0-um/--void-y0-um`.

### Speed Tips
- Use a coarser grid: `--grid-mode macro --grid-nx 70 --grid-ny 70`.
- Reduce coupling iterations: `--coupling-max-iters 8 --coupling-tol 1e-3`.
- Relax CG accuracy: `--cg-tol 1e-4 --cg-maxiter 600`.
- Reduce time steps: `--steps 150` or allow larger `--max-dt`.

## Notes
- The electrical equation is copied from the existing PINN script and applied
  to the measured Ids; current density defaults to dividing by the chip active
  area to avoid over-scaling when Ids is chip-level.
- You can switch to cell-area scaling via `--current-area-mode cell`, or set
  a custom area using `--current-area-m2`.
- Power is applied on the top layer using the macro-cell active volume.
- Layer thickness and grid resolution can be edited in `run_simulation.py`.
