"""HDF5 graph dataset utilities for cell-level electric-field prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data
except ImportError:  # pragma: no cover - fallback for static use without PyG
    Data = None

from .normalization import EFieldNormalizer, as_3d_position


@dataclass(frozen=True)
class SampleSpec:
    """One graph sample in the cell graph HDF5 file."""

    group: str
    sheet: int = 0


class SimpleGraphData:
    """Minimal PyG-like graph object used when torch_geometric is unavailable."""

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)

    def to(self, device: torch.device):
        for key, value in list(self.__dict__.items()):
            if torch.is_tensor(value):
                setattr(self, key, value.to(device))
        return self


def enumerate_samples(h5_path: str) -> list[SampleSpec]:
    """Enumerate all group/sheet combinations in a graph HDF5 file."""

    samples: list[SampleSpec] = []
    with h5py.File(h5_path, "r") as h5:
        for group_name in h5.keys():
            fields = h5[group_name]["fields"]
            num_sheets = int(fields.shape[0]) if fields.ndim == 3 else 1
            for sheet in range(num_sheets):
                samples.append(SampleSpec(group=group_name, sheet=sheet))
    return samples


def split_samples(
    samples: Sequence[SampleSpec],
    train_fraction: float,
    seed: int,
) -> Tuple[list[SampleSpec], list[SampleSpec]]:
    """Shuffle and split graph samples into train/validation subsets."""

    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    split = int(len(indices) * train_fraction)
    return [samples[i] for i in indices[:split]], [samples[i] for i in indices[split:]]


class CellGraphEFieldDataset(Dataset):
    """Dataset producing graph objects with normalized features and Ex/Ey targets."""

    def __init__(
        self,
        h5_path: str,
        samples: Sequence[SampleSpec],
        field_to_index: dict[str, int],
        normalizer: Optional[EFieldNormalizer],
        boundary_percentile: float = 90.0,
        override_temperature: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.h5_path = h5_path
        self.samples = list(samples)
        self.field_to_index = field_to_index
        self.normalizer = normalizer
        self.boundary_percentile = boundary_percentile
        self.override_temperature = override_temperature

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        spec = self.samples[idx]
        with h5py.File(self.h5_path, "r") as h5:
            grp = h5[spec.group]
            pos_raw = grp["pos"][:].astype(np.float32)
            pos = as_3d_position(pos_raw)
            edge_index = grp["edge_index"][:].astype(np.int64)
            fields = _read_fields(grp, spec.sheet)
            metadata = _read_metadata(grp)

        if self.override_temperature is not None:
            metadata["temperature"] = float(self.override_temperature)

        doping = fields[:, self.field_to_index["DopingConcentration"]]
        target = np.column_stack(
            [
                fields[:, self.field_to_index["ElectricField_x"]],
                fields[:, self.field_to_index["ElectricField_y"]],
            ]
        ).astype(np.float32)

        if self.normalizer is None:
            x_np = np.column_stack(
                [
                    pos,
                    doping,
                    np.full(pos.shape[0], metadata["vds"], dtype=np.float32),
                    np.full(pos.shape[0], metadata["vgs"], dtype=np.float32),
                    np.full(pos.shape[0], metadata["temperature"], dtype=np.float32),
                    np.full(pos.shape[0], metadata["die_x"], dtype=np.float32),
                    np.full(pos.shape[0], metadata["die_y"], dtype=np.float32),
                ]
            ).astype(np.float32)
            y_np = target
        else:
            x_np = self.normalizer.transform_inputs(
                pos=pos,
                doping=doping,
                vds=metadata["vds"],
                vgs=metadata["vgs"],
                temperature=metadata["temperature"],
                die_xy=(metadata["die_x"], metadata["die_y"]),
            )
            y_np = self.normalizer.transform_targets(target)

        edge_index_tensor = torch.from_numpy(edge_index).long()
        boundary_mask = self._boundary_mask(
            torch.from_numpy(doping).float(),
            edge_index_tensor,
            self.boundary_percentile,
        )
        kwargs = dict(
            x=torch.from_numpy(x_np).float(),
            y=torch.from_numpy(y_np).float(),
            edge_index=edge_index_tensor,
            pos=torch.from_numpy(pos).float(),
            boundary_mask=boundary_mask,
            raw_temperature=torch.tensor([metadata["temperature"]], dtype=torch.float32),
            vds=torch.tensor([metadata["vds"]], dtype=torch.float32),
            vgs=torch.tensor([metadata["vgs"]], dtype=torch.float32),
            group=spec.group,
            sheet_idx=torch.tensor([spec.sheet], dtype=torch.long),
        )
        if Data is not None:
            return Data(**kwargs)
        return SimpleGraphData(**kwargs)

    @staticmethod
    def _boundary_mask(
        doping: torch.Tensor,
        edge_index: torch.Tensor,
        percentile: float,
    ) -> torch.Tensor:
        src, dst = edge_index
        diff = torch.abs(doping[src] - doping[dst])
        num_nodes = doping.numel()
        accum = torch.zeros(num_nodes, dtype=torch.float32)
        counts = torch.zeros(num_nodes, dtype=torch.float32)
        accum.scatter_add_(0, src, diff)
        accum.scatter_add_(0, dst, diff)
        counts.scatter_add_(0, src, torch.ones_like(diff))
        counts.scatter_add_(0, dst, torch.ones_like(diff))
        score = accum / counts.clamp_min(1.0)
        threshold = torch.quantile(score, percentile / 100.0)
        return (score >= threshold).float()


def single_graph_collate(batch):
    """Fallback collate for batch_size=1 without PyG batching."""

    if len(batch) != 1:
        raise RuntimeError("Install torch_geometric for graph batching or use batch_size=1")
    return batch[0]


def _read_fields(group, sheet: int) -> np.ndarray:
    fields = group["fields"]
    if fields.ndim == 3:
        return fields[sheet].astype(np.float32)
    return fields[:].astype(np.float32)


def _read_metadata(group) -> dict[str, float]:
    attrs = group.attrs
    return {
        "vds": float(attrs.get("vds", attrs.get("Vds", 0.0))),
        "vgs": float(attrs.get("vgs", attrs.get("Vgs", 0.0))),
        "temperature": float(attrs.get("temperature", attrs.get("Tnode", 300.15))),
        "die_x": float(attrs.get("die_x", 0.0)),
        "die_y": float(attrs.get("die_y", 0.0)),
    }

