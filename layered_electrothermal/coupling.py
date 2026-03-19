from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CouplingConfig, SimulationConfig
from .micro_cell import CellElectricalModel
from .thermal_network import ThermalNetwork


@dataclass(frozen=True)
class CouplingResult:
    temperatures_k: np.ndarray
    power_top_w: np.ndarray
    current_density_a_m2: np.ndarray


class CoupledElectroThermalSimulator:
    def __init__(
        self,
        config: SimulationConfig,
        electrical: CellElectricalModel,
        thermal: ThermalNetwork,
        power_mask: np.ndarray | None = None,
    ) -> None:
        self.cfg = config
        self.electrical = electrical
        self.thermal = thermal

        if power_mask is None:
            power_mask = np.ones((self.thermal.nx, self.thermal.ny), dtype=bool)
        if power_mask.shape != (self.thermal.nx, self.thermal.ny):
            raise ValueError("power_mask shape mismatch")
        self.power_mask = power_mask

        self.macro_active_area = self.cfg.cell.active_area_per_macro(self.cfg.chip.macro_area_m2)
        self.macro_active_volume = self.macro_active_area * self.cfg.cell.active_thickness_m
        self.active_volume_map = self.macro_active_volume * self.power_mask.astype(float)

    def step(
        self,
        T_prev: np.ndarray,
        vds_v: float,
        vgs_v: float,
        dt_s: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if dt_s <= 0.0:
            return T_prev.copy(), np.zeros((self.thermal.nx, self.thermal.ny)), np.zeros((self.thermal.nx, self.thermal.ny))

        cfg = self.cfg.coupling
        T_guess = T_prev.copy()
        relax = cfg.relax_init
        prev_resid = None

        power_top = np.zeros((self.thermal.nx, self.thermal.ny), dtype=float)
        current_density = np.zeros((self.thermal.nx, self.thermal.ny), dtype=float)

        for _ in range(cfg.max_iters):
            T_top = T_guess[0]
            T_top_eval = np.where(self.power_mask, T_top, self.cfg.bc.ambient_temp_k)
            current_density = self.electrical.current_density_a_m2(vds_v, vgs_v, T_top_eval)
            power_density = self.electrical.power_density_w_m3(vds_v, vgs_v, T_top_eval)
            power_top = power_density * self.active_volume_map
            current_density = np.where(self.power_mask, current_density, 0.0)

            T_solved = self.thermal.solve_step(T_prev, power_top, dt_s, T_props=T_guess)
            T_new = relax * T_solved + (1.0 - relax) * T_guess

            resid = float(np.max(np.abs(T_new - T_guess)))
            if resid <= cfg.temp_tol_k:
                return T_new, power_top, current_density

            if prev_resid is not None and resid > prev_resid:
                relax = max(cfg.relax_min, 0.5 * relax)
            else:
                relax = min(cfg.relax_max, relax * 1.1)

            prev_resid = resid
            T_guess = T_new

        return T_guess, power_top, current_density

    def simulate(
        self,
        time_s: np.ndarray,
        vds_v: np.ndarray,
        vgs_v: np.ndarray,
        initial_temp_k: float,
    ) -> CouplingResult:
        if len(time_s) != len(vds_v) or len(time_s) != len(vgs_v):
            raise ValueError("time_s, vds_v, vgs_v lengths must match")

        T_prev = self.thermal.initial_temperature(initial_temp_k)
        temps = []
        power_hist = []
        current_hist = []

        for i in range(len(time_s)):
            if i == 0:
                temps.append(T_prev.copy())
                power_hist.append(np.zeros((self.thermal.nx, self.thermal.ny)))
                current_hist.append(np.zeros((self.thermal.nx, self.thermal.ny)))
                continue

            dt_s = float(time_s[i] - time_s[i - 1])
            if dt_s <= 0.0:
                temps.append(T_prev.copy())
                power_hist.append(np.zeros((self.thermal.nx, self.thermal.ny)))
                current_hist.append(np.zeros((self.thermal.nx, self.thermal.ny)))
                continue

            if dt_s > self.cfg.time.max_dt_s:
                n_sub = min(self.cfg.time.max_substeps, int(np.ceil(dt_s / self.cfg.time.max_dt_s)))
                sub_dt = dt_s / float(n_sub)
                for _ in range(n_sub):
                    T_prev, power_top, current_density = self.step(
                        T_prev, float(vds_v[i]), float(vgs_v[i]), sub_dt
                    )
                temps.append(T_prev.copy())
                power_hist.append(power_top.copy())
                current_hist.append(current_density.copy())
                continue

            T_prev, power_top, current_density = self.step(T_prev, float(vds_v[i]), float(vgs_v[i]), dt_s)
            temps.append(T_prev.copy())
            power_hist.append(power_top.copy())
            current_hist.append(current_density.copy())

        return CouplingResult(
            temperatures_k=np.stack(temps),
            power_top_w=np.stack(power_hist),
            current_density_a_m2=np.stack(current_hist),
        )
