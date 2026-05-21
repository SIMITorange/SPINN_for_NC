"""Training entry point for the adapted cell-level Ex/Ey MeshGraphNet branch."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import amp

try:
    from torch_geometric.loader import DataLoader
    from gnn_e_surrogate.data import single_graph_collate

    COLLATE_FN = None
except ImportError:
    from torch.utils.data import DataLoader
    from gnn_e_surrogate.data import single_graph_collate

    COLLATE_FN = single_graph_collate

from gnn_e_surrogate.config import CellGraphConfig, ModelConfig, TrainingConfig
from gnn_e_surrogate.data import CellGraphEFieldDataset, enumerate_samples, split_samples
from gnn_e_surrogate.losses import compute_e_field_loss
from gnn_e_surrogate.model import ElectricFieldMeshGraphNet
from gnn_e_surrogate.normalization import EFieldNormalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train cell-level Ex/Ey GNN surrogate.")
    parser.add_argument("--h5", type=Path, default=None, help="Cell graph HDF5 path.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--resume", type=Path, default=None, help="Optional checkpoint to resume.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset, cfg: TrainingConfig, shuffle: bool):
    kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    if COLLATE_FN is not None:
        kwargs["collate_fn"] = COLLATE_FN
    return DataLoader(dataset, **kwargs)


def run_epoch(
    model: ElectricFieldMeshGraphNet,
    loader,
    optimizer: torch.optim.Optimizer,
    cfg: TrainingConfig,
    train: bool,
) -> Dict[str, float]:
    model.train(train)
    totals = {"total": 0.0, "node": 0.0, "grad": 0.0, "boundary": 0.0, "magnitude": 0.0}
    count = 0
    scaler = amp.GradScaler(enabled=train and cfg.use_mixed_precision and cfg.device.type == "cuda")

    for batch in loader:
        batch = batch.to(cfg.device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with amp.autocast(
                device_type=cfg.device.type,
                dtype=cfg.amp_dtype if cfg.device.type == "cuda" else None,
                enabled=cfg.use_mixed_precision and cfg.device.type == "cuda",
            ):
                pred = model(batch)
                loss, comps = compute_e_field_loss(
                    pred=pred,
                    target=batch.y,
                    edge_index=batch.edge_index,
                    boundary_mask=getattr(batch, "boundary_mask", None),
                    node_weight=cfg.node_loss_weight,
                    grad_weight=cfg.grad_loss_weight,
                    boundary_weight=cfg.boundary_loss_weight,
                    magnitude_weight=cfg.magnitude_loss_weight,
                )
        if train:
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
        for key in totals:
            totals[key] += comps[key]
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    graph_cfg = CellGraphConfig(hdf5_path=args.h5) if args.h5 else CellGraphConfig()
    train_cfg = TrainingConfig(
        epochs=args.epochs or TrainingConfig().epochs,
        batch_size=args.batch_size or TrainingConfig().batch_size,
    )
    model_cfg = ModelConfig()
    graph_cfg.ensure_dirs()
    set_seed(train_cfg.random_seed)

    samples = enumerate_samples(str(graph_cfg.hdf5_path))
    train_samples, val_samples = split_samples(
        samples, train_fraction=train_cfg.train_fraction, seed=train_cfg.random_seed
    )

    norm_path = graph_cfg.normalizer_dir / "e_field_normalizer.npz"
    if norm_path.exists():
        normalizer = EFieldNormalizer.load(norm_path)
    else:
        normalizer = EFieldNormalizer.fit_from_hdf5(
            graph_cfg.hdf5_path, train_samples, graph_cfg.field_to_index
        )
        normalizer.save(norm_path)

    train_dataset = CellGraphEFieldDataset(
        h5_path=str(graph_cfg.hdf5_path),
        samples=train_samples,
        field_to_index=graph_cfg.field_to_index,
        normalizer=normalizer,
        boundary_percentile=graph_cfg.boundary_percentile,
    )
    val_dataset = CellGraphEFieldDataset(
        h5_path=str(graph_cfg.hdf5_path),
        samples=val_samples,
        field_to_index=graph_cfg.field_to_index,
        normalizer=normalizer,
        boundary_percentile=graph_cfg.boundary_percentile,
    )
    train_loader = make_loader(train_dataset, train_cfg, shuffle=True)
    val_loader = make_loader(val_dataset, train_cfg, shuffle=False)

    input_dim = len(graph_cfg.feature_names)
    model = ElectricFieldMeshGraphNet(
        input_dim=input_dim,
        hidden_dim=model_cfg.hidden_dim,
        num_message_passing_steps=model_cfg.num_message_passing_steps,
        position_dim=model_cfg.position_dim,
        output_dim=model_cfg.output_dim,
        activation=model_cfg.activation,
        dropout=model_cfg.dropout,
        use_grad_checkpoint=model_cfg.use_grad_checkpoint,
    ).to(train_cfg.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
    )
    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=train_cfg.device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1

    history_path = graph_cfg.log_dir / "e_field_training_history.csv"
    with history_path.open("a", encoding="utf-8") as log_file:
        if history_path.stat().st_size == 0:
            log_file.write("epoch,train_total,val_total,train_node,val_node,train_grad,val_grad\n")
        for epoch in range(start_epoch, train_cfg.epochs + 1):
            train_metrics = run_epoch(model, train_loader, optimizer, train_cfg, train=True)
            val_metrics = {"total": float("nan"), "node": float("nan"), "grad": float("nan")}
            if epoch % train_cfg.validate_every == 0:
                val_metrics = run_epoch(model, val_loader, optimizer, train_cfg, train=False)

            log_file.write(
                f"{epoch},{train_metrics['total']},{val_metrics['total']},"
                f"{train_metrics['node']},{val_metrics['node']},"
                f"{train_metrics['grad']},{val_metrics['grad']}\n"
            )
            log_file.flush()

            if epoch % train_cfg.checkpoint_every == 0 or epoch == train_cfg.epochs:
                ckpt_path = graph_cfg.checkpoint_dir / f"e_field_meshgraphnet_epoch_{epoch}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "input_dim": input_dim,
                        "output_fields": graph_cfg.output_fields,
                    },
                    ckpt_path,
                )


if __name__ == "__main__":
    main()

