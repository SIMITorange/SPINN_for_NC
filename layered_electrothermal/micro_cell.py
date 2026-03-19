from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CellGeometry


def _softplus(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x_arr))) + np.maximum(x_arr, 0.0)


@dataclass(frozen=True)
class ElectricalParams:
    param1: float = 75.132
    param2: float = 2.062
    param3: float = 6.11
    param4: float = 3.5389
    param5: float = -0.102
    param6: float = 10.138
    param7: float = 1.222
    param8: float = 5.0
    param9: float = -0.5
    vds_coef_slope: float = 0.0
    vds_coef_intercept: float = 0.0
    min_temperature_k: float = 1.0
    use_vds_coef: bool = False


class CellElectricalModel:
    def __init__(self, params: ElectricalParams, cell: CellGeometry, current_area_m2: float | None = None) -> None:
        self.params = params
        self.cell = cell
        self.current_area_m2 = current_area_m2 if current_area_m2 is not None else cell.active_area_m2

    def ids_a(self, vds_v: np.ndarray | float, vgs_v: np.ndarray | float, T_k: np.ndarray | float) -> np.ndarray:
        p = self.params
        vds = np.asarray(vds_v, dtype=float)
        vgs = np.asarray(vgs_v, dtype=float)
        T = np.maximum(np.asarray(T_k, dtype=float), p.min_temperature_k)

        coef = 1.0 / (1.0 + np.exp(-(p.vds_coef_slope * T + p.vds_coef_intercept)))
        vds_eff = vds * coef if p.use_vds_coef else vds

        net5 = p.param3 / (p.param1 * (T / 300.0) ** (-p.param7) + p.param6 * (T / 300.0) ** p.param2)
        net3 = -0.004263 * T + 3.422579
        net2 = -0.005 * vgs + 0.165
        net1 = -0.1717 * vgs + 3.5755

        exp_arg = np.clip(p.param5 * vds_eff, -50.0, 50.0)
        exp_term = np.exp(exp_arg)
        term3 = _softplus(vgs - net3) ** 2 - _softplus(
            vgs - net3 - (net2 * vds_eff * ((1.0 + exp_term) ** net1))
        ) ** 2
        term1 = net5 * (vgs - net3)
        term2 = 1.0 + 0.0005 * vds_eff
        return term2 * term1 * term3

    def current_density_a_m2(
        self, vds_v: np.ndarray | float, vgs_v: np.ndarray | float, T_k: np.ndarray | float
    ) -> np.ndarray:
        ids = self.ids_a(vds_v, vgs_v, T_k)
        return ids / self.current_area_m2

    def power_density_w_m3(
        self, vds_v: np.ndarray | float, vgs_v: np.ndarray | float, T_k: np.ndarray | float
    ) -> np.ndarray:
        ids = self.ids_a(vds_v, vgs_v, T_k)
        vol = self.current_area_m2 * self.cell.active_thickness_m
        return (ids * np.asarray(vds_v, dtype=float)) / vol
