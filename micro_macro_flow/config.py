"""Configuration for the micro-macro fixed-point coupling loop."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CouplingConfig:
    """Iteration controls for the micro/macro electrothermal loop."""

    max_iterations: int = 30
    temperature_tolerance: float = 1.0e-3
    power_tolerance: float = 1.0e-6
    power_relaxation: float = 0.5
    initial_temperature: float = 300.15
    macro_dt: float = 1.0e-7
    micro_tmax: float = 1.0e-6
    stable_iterations: int = 2

