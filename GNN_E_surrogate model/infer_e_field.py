"""Inference entry point for the adapted cell-level Ex/Ey MeshGraphNet branch."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from gnn_e_surrogate.config import CellGraphConfig, ModelConfig
from gnn_e_surrogate.data import CellGraphEFieldDataset, SampleSpec, enumerate_samples
from gnn_e_surrogate.inference import ElectricFieldSurrogate
from gnn_e_surrogate.normalization import EFieldNormalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer physical-unit Ex/Ey on cell graphs.")
    parser.add_argument("--h5", type=Path, default=None, help="Cell graph HDF5 path.")
    parser.add_argument("--group", type=str, default=None, help="HDF5 group name.")
    parser.add_argument("--sheet", type=int, default=0, help="Sheet index inside the group.")
    parser.add_argument("--all", action="store_true", help="Run all groups/sheets.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path.")
    parser.add_argument("--normalizer", type=Path, default=None, help="Normalizer .npz path.")
    parser.add_argument("--temperature", type=float, default=None, help="Override macro Tnode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_cfg = CellGraphConfig(hdf5_path=args.h5) if args.h5 else CellGraphConfig()
    graph_cfg.ensure_dirs()
    normalizer_path = args.normalizer or graph_cfg.normalizer_dir / "e_field_normalizer.npz"
    normalizer = EFieldNormalizer.load(normalizer_path)
    surrogate = ElectricFieldSurrogate(
        checkpoint_path=args.checkpoint,
        normalizer_path=normalizer_path,
        graph_config=graph_cfg,
        model_config=ModelConfig(),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    if args.all:
        samples = enumerate_samples(str(graph_cfg.hdf5_path))
    else:
        if args.group is None:
            raise ValueError("--group is required unless --all is set")
        samples = [SampleSpec(group=args.group, sheet=args.sheet)]

    dataset = CellGraphEFieldDataset(
        h5_path=str(graph_cfg.hdf5_path),
        samples=samples,
        field_to_index=graph_cfg.field_to_index,
        normalizer=normalizer,
        boundary_percentile=graph_cfg.boundary_percentile,
        override_temperature=args.temperature,
    )
    for sample, graph in zip(samples, dataset):
        pred_exey = surrogate.predict_graph(graph)
        save_path = graph_cfg.prediction_dir / f"{sample.group}_s{sample.sheet}_exey.npz"
        np.savez(
            save_path,
            ex=pred_exey[:, 0],
            ey=pred_exey[:, 1],
            exey=pred_exey,
            pos=graph.pos.detach().cpu().numpy(),
            edge_index=graph.edge_index.detach().cpu().numpy(),
            group=sample.group,
            sheet=sample.sheet,
        )


if __name__ == "__main__":
    main()

