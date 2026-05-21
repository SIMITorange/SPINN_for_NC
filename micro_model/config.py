"""Configuration objects for the cell-level Chebyshev micro solver."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MicroDomainConfig:
    """Approximate cell domain dimensions."""

    length_x: float = 80.0e-6
    length_y: float = 40.0e-6
    thickness: float = 12.0e-6
    origin_x: float = 0.0
    origin_y: float = 0.0
    origin_z: float = 0.0
    order_x: int = 12
    order_y: int = 10
    order_z: int = 8


@dataclass(frozen=True)
class MicroMaterial:
    """Temperature-dependent SiC-like material coefficients."""

    rho: float = 3210.0
    c_ref: float = 690.0
    k_ref: float = 370.0
    t_ref: float = 300.15
    c_temp_coeff: float = 2.0e-4
    k_temp_coeff: float = -1.2e-3
    k_min: float = 50.0

    def heat_capacity(self, temperature):
        return self.c_ref * (1.0 + self.c_temp_coeff * (temperature - self.t_ref))

    def thermal_conductivity(self, temperature):
        value = self.k_ref * (1.0 + self.k_temp_coeff * (temperature - self.t_ref))
        return value.clip(min=self.k_min) if hasattr(value, "clip") else max(value, self.k_min)


@dataclass(frozen=True)
class MicroSolverConfig:
    """RK4 and boundary coupling settings."""

    dt: float = 2.0e-9
    t_max: float = 1.0e-6
    ambient_temperature: float = 300.15
    macro_relaxation_time: float = 2.0e-7
    coefficient_update_interval: int = 10
    max_temperature_delta_per_step: float = 200.0
    min_power_density: float = 0.0

