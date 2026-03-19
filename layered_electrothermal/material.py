from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _safe_temperature(T: np.ndarray | float) -> np.ndarray:
    T_arr = np.asarray(T, dtype=float)
    return np.maximum(T_arr, 1.0)


@dataclass(frozen=True)
class ThermalMaterial:
    name: str
    rho_kg_m3: float

    def conductivity_w_mk(self, T: np.ndarray | float) -> np.ndarray:
        """SiC conductivity from PINN_short_circuit_no_temperature.py."""
        T_safe = _safe_temperature(T)
        denom = -0.0003 + 1.05e-5 * T_safe
        return 1.0 / np.clip(denom, 1.0e-12, None)

    def heat_capacity_j_kgk(self, T: np.ndarray | float) -> np.ndarray:
        """SiC heat capacity from PINN_short_circuit_no_temperature.py."""
        T_safe = _safe_temperature(T)
        cp = 300.0 * (5.13 - 1001.0 / T_safe + 3.23e4 / (T_safe**2))
        return np.clip(cp, 1.0e-12, None)


def default_materials() -> dict[str, ThermalMaterial]:
    return {
        "sic": ThermalMaterial(name="SiC", rho_kg_m3=3200.0),
    }
