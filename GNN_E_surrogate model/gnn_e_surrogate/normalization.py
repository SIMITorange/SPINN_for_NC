"""Normalizers for conditioned cell graph inputs and Ex/Ey targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np


def as_3d_position(pos: np.ndarray) -> np.ndarray:
    """Return coordinates with three columns, padding z=0 when needed."""

    pos = np.asarray(pos, dtype=np.float32)
    if pos.ndim != 2:
        raise ValueError(f"pos must be a 2D array, got shape {pos.shape}")
    if pos.shape[1] == 3:
        return pos
    if pos.shape[1] == 2:
        z = np.zeros((pos.shape[0], 1), dtype=np.float32)
        return np.concatenate([pos, z], axis=1)
    if pos.shape[1] > 3:
        return pos[:, :3]
    raise ValueError("pos must have at least x/y coordinates")


def conditioned_features(
    pos: np.ndarray,
    doping: np.ndarray,
    vds: float,
    vgs: float,
    temperature: float,
    die_xy: Sequence[float] = (0.0, 0.0),
    doping_scale: float = 1.0,
    eps: float = 1.0e-8,
) -> np.ndarray:
    """Build raw model features before standardization."""

    pos3 = as_3d_position(pos)
    doping_asinh = np.arcsinh(np.asarray(doping, dtype=np.float32) / max(doping_scale, eps))
    die_x, die_y = float(die_xy[0]), float(die_xy[1])
    cond = np.column_stack(
        [
            np.full(pos3.shape[0], float(vds), dtype=np.float32),
            np.full(pos3.shape[0], float(vgs), dtype=np.float32),
            np.full(pos3.shape[0], float(temperature), dtype=np.float32),
            np.full(pos3.shape[0], die_x, dtype=np.float32),
            np.full(pos3.shape[0], die_y, dtype=np.float32),
        ]
    )
    return np.column_stack([pos3, doping_asinh[:, None], cond]).astype(np.float32)


@dataclass
class EFieldNormalizer:
    """Standardize model inputs and asinh-scaled vector electric-field targets."""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    doping_scale: float
    efield_scale: np.ndarray
    eps: float = 1.0e-8

    @classmethod
    def empty(cls, input_dim: int = 9, output_dim: int = 2) -> "EFieldNormalizer":
        return cls(
            feature_mean=np.zeros(input_dim, dtype=np.float32),
            feature_std=np.ones(input_dim, dtype=np.float32),
            target_mean=np.zeros(output_dim, dtype=np.float32),
            target_std=np.ones(output_dim, dtype=np.float32),
            doping_scale=1.0,
            efield_scale=np.ones(output_dim, dtype=np.float32),
        )

    @classmethod
    def fit(
        cls,
        feature_blocks: Iterable[np.ndarray],
        target_blocks: Iterable[np.ndarray],
        eps: float = 1.0e-8,
    ) -> "EFieldNormalizer":
        """Fit statistics from arrays yielded by the dataset loader."""

        features = [np.asarray(block, dtype=np.float32) for block in feature_blocks]
        targets = [np.asarray(block, dtype=np.float32) for block in target_blocks]
        if not features or not targets:
            raise ValueError("Cannot fit normalizer without feature and target blocks")
        feature_all = np.concatenate(features, axis=0)
        target_all = np.concatenate(targets, axis=0)
        efield_scale = np.maximum(np.median(np.abs(target_all), axis=0), eps)
        target_prepared = np.arcsinh(target_all / efield_scale)
        return cls(
            feature_mean=feature_all.mean(axis=0),
            feature_std=np.maximum(feature_all.std(axis=0), eps),
            target_mean=target_prepared.mean(axis=0),
            target_std=np.maximum(target_prepared.std(axis=0), eps),
            doping_scale=1.0,
            efield_scale=efield_scale.astype(np.float32),
            eps=eps,
        )

    @classmethod
    def fit_from_hdf5(
        cls,
        h5_path: Path,
        samples: Sequence,
        field_to_index: dict[str, int],
        eps: float = 1.0e-8,
    ) -> "EFieldNormalizer":
        """Fit normalizer directly from the expected graph HDF5 schema."""

        doping_pool: list[np.ndarray] = []
        with h5py.File(h5_path, "r") as h5:
            for spec in samples:
                grp = h5[spec.group]
                fields = _read_fields(grp, spec.sheet)
                doping_pool.append(np.abs(fields[:, field_to_index["DopingConcentration"]]))
        doping_scale = float(np.maximum(np.median(np.concatenate(doping_pool)), eps))

        feature_blocks: list[np.ndarray] = []
        target_blocks: list[np.ndarray] = []
        with h5py.File(h5_path, "r") as h5:
            for spec in samples:
                grp = h5[spec.group]
                pos = grp["pos"][:]
                fields = _read_fields(grp, spec.sheet)
                metadata = _read_metadata(grp)
                features = conditioned_features(
                    pos=pos,
                    doping=fields[:, field_to_index["DopingConcentration"]],
                    vds=metadata["vds"],
                    vgs=metadata["vgs"],
                    temperature=metadata["temperature"],
                    die_xy=(metadata["die_x"], metadata["die_y"]),
                    doping_scale=doping_scale,
                    eps=eps,
                )
                target = np.column_stack(
                    [
                        fields[:, field_to_index["ElectricField_x"]],
                        fields[:, field_to_index["ElectricField_y"]],
                    ]
                )
                feature_blocks.append(features)
                target_blocks.append(target.astype(np.float32))

        norm = cls.fit(feature_blocks, target_blocks, eps=eps)
        norm.doping_scale = doping_scale
        return norm

    def transform_inputs(
        self,
        pos: np.ndarray,
        doping: np.ndarray,
        vds: float,
        vgs: float,
        temperature: float,
        die_xy: Sequence[float] = (0.0, 0.0),
    ) -> np.ndarray:
        features = conditioned_features(
            pos=pos,
            doping=doping,
            vds=vds,
            vgs=vgs,
            temperature=temperature,
            die_xy=die_xy,
            doping_scale=self.doping_scale,
            eps=self.eps,
        )
        return ((features - self.feature_mean) / self.feature_std).astype(np.float32)

    def transform_targets(self, efield_xy: np.ndarray) -> np.ndarray:
        prepared = np.arcsinh(np.asarray(efield_xy, dtype=np.float32) / self.efield_scale)
        return ((prepared - self.target_mean) / self.target_std).astype(np.float32)

    def inverse_targets(self, normalized_efield_xy: np.ndarray) -> np.ndarray:
        arr = np.asarray(normalized_efield_xy, dtype=np.float32)
        unnorm = arr * self.target_std + self.target_mean
        return (np.sinh(unnorm) * self.efield_scale).astype(np.float32)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
            doping_scale=self.doping_scale,
            efield_scale=self.efield_scale,
            eps=self.eps,
        )

    @classmethod
    def load(cls, path: Path) -> "EFieldNormalizer":
        data = np.load(path, allow_pickle=False)
        return cls(
            feature_mean=data["feature_mean"].astype(np.float32),
            feature_std=data["feature_std"].astype(np.float32),
            target_mean=data["target_mean"].astype(np.float32),
            target_std=data["target_std"].astype(np.float32),
            doping_scale=float(data["doping_scale"]),
            efield_scale=data["efield_scale"].astype(np.float32),
            eps=float(data["eps"]),
        )


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

