from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChipGeometry:
    chip_length_m: float = 3.5e-3
    chip_width_m: float = 3.5e-3
    macro_nx: int = 7
    macro_ny: int = 7

    @property
    def macro_dx_m(self) -> float:
        return self.chip_length_m / float(self.macro_nx)

    @property
    def macro_dy_m(self) -> float:
        return self.chip_width_m / float(self.macro_ny)

    @property
    def macro_area_m2(self) -> float:
        return self.macro_dx_m * self.macro_dy_m

    @property
    def chip_area_m2(self) -> float:
        return self.chip_length_m * self.chip_width_m


@dataclass(frozen=True)
class CellGeometry:
    cell_pitch_x_m: float = 10e-6
    cell_pitch_y_m: float = 10e-6
    active_fill: float = 0.5
    active_thickness_m: float = 1.0e-5

    @property
    def pitch_area_m2(self) -> float:
        return self.cell_pitch_x_m * self.cell_pitch_y_m

    @property
    def active_area_m2(self) -> float:
        return self.pitch_area_m2 * self.active_fill

    @property
    def cells_per_m2(self) -> float:
        return 1.0 / self.pitch_area_m2

    def active_area_per_macro(self, macro_area_m2: float) -> float:
        return macro_area_m2 * self.active_fill


@dataclass(frozen=True)
class ThermalBC:
    ambient_temp_k: float = 300.0
    h_top_w_m2k: float = 10.0
    h_bottom_w_m2k: float = 5.0e4


@dataclass(frozen=True)
class LayerConfig:
    name: str
    thickness_m: float
    material_key: str


@dataclass(frozen=True)
class TimeConfig:
    max_dt_s: float = 2.0e-9
    min_dt_s: float = 1.0e-12
    max_substeps: int = 200


@dataclass(frozen=True)
class CouplingConfig:
    max_iters: int = 30
    temp_tol_k: float = 1.0e-3
    relax_init: float = 0.5
    relax_min: float = 0.1
    relax_max: float = 0.9


@dataclass(frozen=True)
class ElectricalDrive:
    vds_v: float
    vgs_v: float


@dataclass(frozen=True)
class SimulationConfig:
    chip: ChipGeometry = ChipGeometry()
    cell: CellGeometry = CellGeometry()
    bc: ThermalBC = ThermalBC()
    time: TimeConfig = TimeConfig()
    coupling: CouplingConfig = CouplingConfig()
