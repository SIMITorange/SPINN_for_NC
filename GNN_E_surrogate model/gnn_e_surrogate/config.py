"""Configuration objects for the cell-level electric-field surrogate."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import torch


DEFAULT_FIELD_TO_INDEX: Dict[str, int] = {
    "ElectrostaticPotential": 0,
    "eDensity": 1,
    "hDensity": 2,
    "SpaceCharge": 3,
    "ElectricField_x": 4,
    "ElectricField_y": 5,
    "DopingConcentration": 6,
}


@dataclass(frozen=True)
class CellGraphConfig:
    """Filesystem and schema settings for cell graph HDF5 data."""

    project_root: Path = Path(__file__).resolve().parents[1]
    hdf5_path: Path = project_root / "data" / "cell_e_graphs.h5"
    output_dir: Path = project_root / "outputs"
    checkpoint_dir: Path = output_dir / "checkpoints"
    normalizer_dir: Path = output_dir / "normalizers"
    prediction_dir: Path = output_dir / "predictions"
    log_dir: Path = output_dir / "logs"
    field_to_index: Dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_FIELD_TO_INDEX)
    )
    output_fields: Tuple[str, str] = ("ElectricField_x", "ElectricField_y")
    feature_names: Tuple[str, ...] = (
        "x",
        "y",
        "z",
        "doping_asinh",
        "vds",
        "vgs",
        "temperature",
        "die_x",
        "die_y",
    )
    boundary_percentile: float = 90.0

    def ensure_dirs(self) -> None:
        for path in (
            self.output_dir,
            self.checkpoint_dir,
            self.normalizer_dir,
            self.prediction_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ModelConfig:
    """MeshGraphNet architecture settings."""

    hidden_dim: int = 128
    num_message_passing_steps: int = 5
    activation: str = "gelu"
    dropout: float = 0.05
    use_grad_checkpoint: bool = True
    position_dim: int = 3
    output_dim: int = 2


@dataclass(frozen=True)
class TrainingConfig:
    """Default training settings; scripts may override these via CLI."""

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_seed: int = 42
    train_fraction: float = 0.85
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = True
    epochs: int = 2000
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    validate_every: int = 5
    checkpoint_every: int = 100
    print_every: int = 10
    use_mixed_precision: bool = True
    amp_dtype: torch.dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    node_loss_weight: float = 1.0
    grad_loss_weight: float = 0.5
    boundary_loss_weight: float = 0.25
    magnitude_loss_weight: float = 0.1


def latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Return the latest checkpoint path by epoch number/name sorting."""

    candidates = sorted(checkpoint_dir.glob("e_field_meshgraphnet_epoch_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No electric-field checkpoints found in {checkpoint_dir}")
    return candidates[-1]

