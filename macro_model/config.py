"""Configuration objects for the die-level thermal solver."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class MacroGridConfig:
    """Structured die grid and physical dimensions."""

    nx: int = 60
    ny: int = 60
    nz: int = 8
    length_x: float = 6.0e-3
    length_y: float = 6.0e-3
    thickness: float = 3.5e-4


@dataclass(frozen=True)
class ThermalMaterial:
    """Temperature-dependent thermal coefficients for the macro network."""

    rho: float = 3210.0
    cp_ref: float = 690.0
    k_ref: float = 370.0
    t_ref: float = 300.15
    cp_temp_coeff: float = 2.0e-4
    k_temp_coeff: float = -1.2e-3
    k_min: float = 50.0

    def heat_capacity(self, temperature):
        return self.cp_ref * (1.0 + self.cp_temp_coeff * (temperature - self.t_ref))

    def thermal_conductivity(self, temperature):
        k = self.k_ref * (1.0 + self.k_temp_coeff * (temperature - self.t_ref))
        return k.clip(min=self.k_min) if hasattr(k, "clip") else max(k, self.k_min)


@dataclass(frozen=True)
class BoundaryCondition:
    """Convection plus ambient thermal boundary condition."""

    h_top: float = 2.0e4
    h_bottom: float = 5.0e3
    h_side: float = 2.0e3
    ambient_temperature: float = 300.15


@dataclass(frozen=True)
class MacroSolverConfig:
    """Time stepping and linear solver options."""

    dt: float = 1.0e-7
    t_end: float = 1.0e-5
    cg_tol: float = 1.0e-8
    cg_maxiter: int = 1000
    initial_temperature: float = 300.15
    top_power_spread: int = 1

