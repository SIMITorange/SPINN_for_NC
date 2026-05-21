"""Cell-level Chebyshev PDE discretization and RK4 solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .chebyshev import ChebyshevOperators, build_chebyshev_operators
from .config import MicroDomainConfig, MicroMaterial, MicroSolverConfig


@dataclass(frozen=True)
class RegionalNormalizer:
    """Map between physical cell coordinates and normalized CGL coordinates."""

    origin: np.ndarray
    span: np.ndarray

    @classmethod
    def from_domain(cls, domain: MicroDomainConfig) -> "RegionalNormalizer":
        origin = np.array([domain.origin_x, domain.origin_y, domain.origin_z], dtype=np.float64)
        span = np.array([domain.length_x, domain.length_y, domain.thickness], dtype=np.float64)
        return cls(origin=origin, span=span)

    def normalize(self, xyz: np.ndarray) -> np.ndarray:
        return 2.0 * (np.asarray(xyz) - self.origin) / self.span - 1.0

    def denormalize(self, xyz_bar: np.ndarray) -> np.ndarray:
        return self.origin + 0.5 * (np.asarray(xyz_bar) + 1.0) * self.span


@dataclass
class MicroThermalState:
    """Cell-level temperature field and restored coefficients."""

    temperature: np.ndarray
    time: float
    k_sic: float
    c_sic: float
    power_loss: float


class MicroChebyshevSolver:
    """Implements the micro solver blocks from the algorithm figure."""

    def __init__(
        self,
        domain: MicroDomainConfig,
        material: MicroMaterial,
        config: MicroSolverConfig,
    ) -> None:
        self.domain = domain
        self.material = material
        self.config = config
        self.normalizer = RegionalNormalizer.from_domain(domain)
        self.operators = self.construct_matrices()
        self.physical_nodes = self._build_physical_nodes()

    def set_cgl_nodes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ops = self.operators
        return ops.nodes_x, ops.nodes_y, ops.nodes_z

    def construct_matrices(self) -> ChebyshevOperators:
        """Construct `M`, derivative matrices, stiffness core `K`, and `F`."""

        return build_chebyshev_operators(
            order_x=self.domain.order_x,
            order_y=self.domain.order_y,
            order_z=self.domain.order_z,
            length_x=self.domain.length_x,
            length_y=self.domain.length_y,
            thickness=self.domain.thickness,
        )

    def initial_state(self, temperature: Optional[float] = None) -> MicroThermalState:
        t0 = self.config.ambient_temperature if temperature is None else float(temperature)
        field = np.full(self.operators.weights.shape[0], t0, dtype=np.float64)
        k_sic, c_sic = self.restore_temperature_coefficients(field)
        return MicroThermalState(
            temperature=field,
            time=0.0,
            k_sic=k_sic,
            c_sic=c_sic,
            power_loss=0.0,
        )

    def discretize_pde(
        self,
        temperature: np.ndarray,
        power_density: np.ndarray,
        macro_temperature: float,
        k_sic: float,
        c_sic: float,
    ) -> np.ndarray:
        """Return `dT/dt` for the spectral heat equation."""

        rho_c = self.material.rho * c_sic
        diffusion = (k_sic / rho_c) * (self.operators.laplacian @ temperature)
        source = np.maximum(power_density, self.config.min_power_density) / rho_c
        boundary_feedback = (
            self.operators.boundary_operator
            @ (np.full_like(temperature, macro_temperature) - temperature)
        ) / self.config.macro_relaxation_time
        return diffusion + source + boundary_feedback

    def rk4_step(
        self,
        state: MicroThermalState,
        power_density: np.ndarray,
        macro_temperature: float,
        dt: Optional[float] = None,
    ) -> MicroThermalState:
        """Advance the cell PDE by one RK4 step."""

        dt = float(dt or self.config.dt)
        t = state.temperature
        rhs = lambda y: self.discretize_pde(
            temperature=y,
            power_density=power_density,
            macro_temperature=macro_temperature,
            k_sic=state.k_sic,
            c_sic=state.c_sic,
        )
        k1 = rhs(t)
        k2 = rhs(t + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt * k2)
        k4 = rhs(t + dt * k3)
        new_t = t + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        delta = np.clip(
            new_t - t,
            -self.config.max_temperature_delta_per_step,
            self.config.max_temperature_delta_per_step,
        )
        new_t = t + delta
        k_sic, c_sic = self.restore_temperature_coefficients(new_t)
        return MicroThermalState(
            temperature=new_t,
            time=state.time + dt,
            k_sic=k_sic,
            c_sic=c_sic,
            power_loss=state.power_loss,
        )

    def solve(
        self,
        initial_temperature: float,
        power_density: np.ndarray,
        macro_temperature: float,
        t_max: Optional[float] = None,
    ) -> MicroThermalState:
        """Run RK4 iterations until `t_max` and restore coefficients."""

        state = self.initial_state(initial_temperature)
        final_time = self.config.t_max if t_max is None else float(t_max)
        step = 0
        while state.time < final_time:
            state = self.rk4_step(state, power_density, macro_temperature)
            step += 1
            if step % max(self.config.coefficient_update_interval, 1) == 0:
                k_sic, c_sic = self.restore_temperature_coefficients(state.temperature)
                state.k_sic = k_sic
                state.c_sic = c_sic
        return state

    def restore_temperature_coefficients(self, temperature: np.ndarray) -> tuple[float, float]:
        """Restore temperature-dependent `k_SiC` and `C_SiC` from the field."""

        t_mean = float(np.mean(temperature))
        k_sic = float(self.material.thermal_conductivity(t_mean))
        c_sic = float(self.material.heat_capacity(t_mean))
        return k_sic, c_sic

    def denormalize_temperature_coefficients(
        self,
        coeff_temperature: np.ndarray,
        basis: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Restore a nodal temperature field from modal coefficients."""

        if basis is None:
            return np.asarray(coeff_temperature, dtype=np.float64)
        return np.asarray(basis, dtype=np.float64) @ np.asarray(coeff_temperature, dtype=np.float64)

    def calculate_power_loss(
        self,
        electric_field_xy: np.ndarray,
        current_density_xy: np.ndarray,
        active_mask: Optional[np.ndarray] = None,
    ) -> tuple[float, np.ndarray]:
        """Compute cell power loss from `J dot E` and return nodal density."""

        e = np.asarray(electric_field_xy, dtype=np.float64)
        j = np.asarray(current_density_xy, dtype=np.float64)
        if e.shape != j.shape or e.shape[-1] != 2:
            raise ValueError("electric_field_xy and current_density_xy must both be [N, 2]")
        joule = np.sum(j * e, axis=-1)
        joule = np.maximum(joule, 0.0)
        if active_mask is not None:
            joule = joule * np.asarray(active_mask, dtype=np.float64)
        weights = self.operators.weights
        if joule.shape[0] != weights.shape[0]:
            joule = self.interpolate_to_cgl(joule)
        power_loss = float(np.sum(joule * weights))
        return power_loss, joule

    def distribute_scalar_power(self, power_loss: float, active_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert a scalar cell loss into a normalized nodal power density."""

        weights = self.operators.weights
        mask = np.ones_like(weights) if active_mask is None else np.asarray(active_mask, dtype=np.float64)
        effective_volume = float(np.sum(weights * mask))
        if effective_volume <= 0.0:
            raise ValueError("active volume must be positive")
        return mask * float(power_loss) / effective_volume

    def interpolate_to_cgl(self, values: Sequence[float]) -> np.ndarray:
        """Placeholder interpolation hook for external meshes to CGL nodes."""

        values = np.asarray(values, dtype=np.float64).reshape(-1)
        target = self.operators.weights.shape[0]
        if values.size == target:
            return values
        src_x = np.linspace(0.0, 1.0, values.size)
        dst_x = np.linspace(0.0, 1.0, target)
        return np.interp(dst_x, src_x, values)

    def _build_physical_nodes(self) -> np.ndarray:
        nx, ny, nz = self.set_cgl_nodes()
        zz, yy, xx = np.meshgrid(nz, ny, nx, indexing="ij")
        normalized = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        return self.normalizer.denormalize(normalized)

