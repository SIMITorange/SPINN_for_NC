"""PINN training with HDF5 groups, using a simple temperature integrator (no Chebyshev).

Aside from the temperature integration, the surrounding pipeline matches
PINN_short_circuit_no_temperature.py: same HDF5 loading, per-group training,
logging, checkpointing, and TXT outputs for downstream plotting scripts.
"""
from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

try:
    import h5py  # type: ignore
except ImportError:  # pragma: no cover
    h5py = None


@dataclass(frozen=True)
class TrainConfig:
    # Data
    hdf5_path: str = "combined_training_data.h5"
    output_dir: str = "results_no_cheb"
    checkpoint_dir: str = "checkpoints_no_cheb"
    # Training
    num_epochs: int = 500
    # batch_size: int = 300 # Not used; full sequence batch
    alpha: float = 0.9
    init_w_data: float = 0.5
    init_w_physics: float = 0.5
    lr_net: float = 1e-3
    lr_params: float = 1e-1
    weight_decay: float = 1e-4
    clip_grad_norm: float = 1.0
    scheduler_patience: int = 20
    scheduler_factor: float = 0.5
    use_tensorboard: bool = True
    tb_log_dir: str = "runs_no_cheb"
    # Thermal (simple Euler model)
    rho: float = 3200.0          # density proxy
    c_th: float = 600.0          # specific heat proxy
    ids_area: float = 20e-6      # area scaling used in legacy formula
    min_temperature: float = 1.0


CFG = TrainConfig()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_ ." else "_" for ch in name)


def _save_txt(path: str, array_2d: np.ndarray, header: str) -> None:
    _ensure_dir(os.path.dirname(path))
    np.savetxt(path, array_2d, fmt="%.10e", header=header)


def _append_csv_row(csv_path: str, header: List[str], row: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(csv_path))
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_hdf5_groups(hdf5_path: str) -> List[Tuple[str, np.ndarray, Dict[str, Any]]]:
    """Return [(group_name, data_np, labels_dict)] from HDF5."""
    if h5py is None:
        raise RuntimeError("缺少依赖 h5py：请先安装 `pip install h5py` 后再运行。")

    items: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []
    with h5py.File(hdf5_path, "r") as h5f:
        for group_name in h5f.keys():
            grp = h5f[group_name]
            if "data" not in grp:
                continue
            data = np.array(grp["data"], dtype=np.float32)
            labels: Dict[str, Any] = {
                "group": group_name,
                "source": grp.attrs.get("source", ""),
                "vds_max": float(grp.attrs.get("vds_max", np.nan)),
                "vgs_max": float(grp.attrs.get("vgs_max", np.nan)),
                "time_max": float(grp.attrs.get("time_max", np.nan)),
                "columns": list(grp.attrs.get("columns", [])),
            }
            items.append((group_name, data, labels))
    return items


class PINN(nn.Module):
    def __init__(self, dropout_rate: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.param1 = nn.Parameter(torch.tensor(75.132, dtype=torch.float32))
        self.param2 = nn.Parameter(torch.tensor(2.062, dtype=torch.float32))
        self.param3 = nn.Parameter(torch.tensor(6.11, dtype=torch.float32))
        self.param4 = nn.Parameter(torch.tensor(3.5389, dtype=torch.float32))
        self.param5 = nn.Parameter(torch.tensor(-0.102, dtype=torch.float32))
        self.param6 = nn.Parameter(torch.tensor(10.138, dtype=torch.float32))
        self.param7 = nn.Parameter(torch.tensor(1.222, dtype=torch.float32))
        self.param8 = nn.Parameter(torch.tensor(5.0, dtype=torch.float32))
        self.param9 = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))

        # Thermal parameters (match sample script)
        self.epi_thickness = nn.Parameter(torch.tensor(1.0e-5, dtype=torch.float32))
        self.T_initial = nn.Parameter(torch.tensor(350.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def physics_model(
    Vds: torch.Tensor | float,
    Vgs: torch.Tensor | float,
    T: torch.Tensor | float,
    param1: torch.Tensor,
    param2: torch.Tensor,
    param3: torch.Tensor,
    param4: torch.Tensor,
    param5: torch.Tensor,
    param6: torch.Tensor,
    param7: torch.Tensor,
    param8: torch.Tensor,
    param9: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(Vds, torch.Tensor):
        Vds = torch.tensor(Vds, dtype=torch.float32)
    if not isinstance(Vgs, torch.Tensor):
        Vgs = torch.tensor(Vgs, dtype=torch.float32)
    if not isinstance(T, torch.Tensor):
        T = torch.tensor(T, dtype=torch.float32)

    net5 = param3 * 1.0 / (param1 * (T / 300.0) ** -param7 + param6 * (T / 300.0) ** param2)
    net3 = -0.004263 * T + 3.422579
    p9 = param5
    net2 = -0.005 * Vgs + 0.165
    net1 = -0.1717 * Vgs + 3.5755
    term3 = (torch.log(1 + torch.exp(Vgs - net3))) ** 2 - (
        torch.log(1 + torch.exp(Vgs - net3 - (net2 * Vds * ((1 + torch.exp(p9 * Vds)) ** net1))))
    ) ** 2
    term1 = net5 * (Vgs - net3)
    term2 = 1 + 0.0005 * Vds
    return term2 * term1 * term3


def integrate_temperature(
    time_raw: torch.Tensor,
    vds_raw: torch.Tensor,
    vgs_raw: torch.Tensor,
    model: PINN,
    cfg: TrainConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simple Euler temperature update (no spectral solver)."""
    if not isinstance(time_raw, torch.Tensor):
        time_raw = torch.tensor(time_raw, dtype=torch.float32)
    if not isinstance(vds_raw, torch.Tensor):
        vds_raw = torch.tensor(vds_raw, dtype=torch.float32)
    if not isinstance(vgs_raw, torch.Tensor):
        vgs_raw = torch.tensor(vgs_raw, dtype=torch.float32)

    device = time_raw.device
    dtype = time_raw.dtype

    delta_time = torch.zeros_like(time_raw, dtype=dtype, device=device)
    delta_time[1:] = time_raw[1:] - time_raw[:-1]

    rho_cp = cfg.rho * cfg.c_th
    ids_area = cfg.ids_area

    T_list: List[torch.Tensor] = []
    physics_list: List[torch.Tensor] = []

    prev_T = model.T_initial
    for i in range(len(time_raw)):
        if i > 0:
            ids_prev = physics_model(
                vds_raw[i],
                vgs_raw[i],
                prev_T.detach(),
                model.param1,
                model.param2,
                model.param3,
                model.param4,
                model.param5,
                model.param6,
                model.param7,
                model.param8,
                model.param9,
            )
            delta_T = (2.0 * vds_raw[i] / 1e-5) * (ids_prev / ids_area) * delta_time[i] / rho_cp
            current_T = prev_T + delta_T
        else:
            current_T = model.T_initial

        T_list.append(current_T)
        physics_list.append(
            physics_model(
                vds_raw[i],
                vgs_raw[i],
                current_T,
                model.param1,
                model.param2,
                model.param3,
                model.param4,
                model.param5,
                model.param6,
                model.param7,
                model.param8,
                model.param9,
            )
        )
        prev_T = current_T

    T_seq = torch.stack(T_list)
    physics_preds = torch.stack(physics_list)
    return T_seq, physics_preds


def train_one_group(
    group_name: str,
    data_np: np.ndarray,
    labels: Dict[str, Any],
    cfg: TrainConfig,
) -> Dict[str, Any]:
    # data: [time(s), Ids, Vds, Vgs]
    time_col = data_np[:, 0:1]
    ids_col = data_np[:, 1:2]
    vds_col = data_np[:, 2:3]
    vgs_col = data_np[:, 3:4]

    inputs_data = np.concatenate([time_col, vds_col, vgs_col], axis=1)
    targets_data = ids_col

    input_scaler = StandardScaler()
    target_scaler = StandardScaler()
    X_scaled = input_scaler.fit_transform(inputs_data)
    y_scaled = target_scaler.fit_transform(targets_data)

    input_mean = torch.tensor(input_scaler.mean_, dtype=torch.float32)
    input_std = torch.tensor(input_scaler.scale_, dtype=torch.float32)
    target_mean = torch.tensor(float(target_scaler.mean_[0]), dtype=torch.float32)
    target_std = torch.tensor(float(target_scaler.scale_[0]), dtype=torch.float32)

    dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y_scaled, dtype=torch.float32))
    # Use full sequence in one batch for temperature feedback consistency
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    writer = None
    if cfg.use_tensorboard:
        log_run_name = _safe_filename(f"{group_name}_Vds{labels.get('vds_max','')}_Vgs{labels.get('vgs_max','')}")
        writer = SummaryWriter(log_dir=os.path.join(cfg.tb_log_dir, log_run_name))

    model = PINN()
    optimizer = optim.AdamW(
        [
            {"params": model.net.parameters(), "lr": cfg.lr_net},
            {
                "params": [
                    model.param1,
                    model.param2,
                    model.param3,
                    model.param4,
                    model.param5,
                    model.param6,
                    model.param7,
                    model.param8,
                    model.param9,
                    model.epi_thickness,
                    model.T_initial,
                ],
                "lr": cfg.lr_params,
            },
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        patience=cfg.scheduler_patience,
        factor=cfg.scheduler_factor,
    )
    mse_loss = nn.MSELoss()

    w_data = cfg.init_w_data
    w_physics = cfg.init_w_physics
    losses: List[float] = []
    data_metrics: List[float] = []
    physics_metrics: List[float] = []
    weight_metrics: List[Tuple[float, float]] = []
    lr_metrics: List[float] = []

    for epoch in range(cfg.num_epochs):
        model.train()
        for X_batch, y_batch in dataloader:
            pred = model(X_batch)
            loss_data = mse_loss(pred, y_batch)

            time_raw = X_batch[:, 0] * input_std[0] + input_mean[0]
            vds_raw = X_batch[:, 1] * input_std[1] + input_mean[1]
            vgs_raw = X_batch[:, 2] * input_std[2] + input_mean[2]

            T_mean, physics_preds = integrate_temperature(time_raw, vds_raw, vgs_raw, model, cfg)

            physics_pred_scaled = (physics_preds - target_mean) / target_std
            loss_physics = mse_loss(pred.squeeze(), physics_pred_scaled)

            current_data = float(loss_data.detach().cpu().item())
            current_physics = float(loss_physics.detach().cpu().item())
            total = current_data + current_physics + 1e-8
            new_wd = current_data / total
            new_wp = current_physics / total
            w_data = cfg.alpha * w_data + (1 - cfg.alpha) * new_wd
            w_physics = cfg.alpha * w_physics + (1 - cfg.alpha) * new_wp

            total_loss = w_data * loss_data + w_physics * loss_physics

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            optimizer.step()

        scheduler.step(float(total_loss.detach().cpu().item()))
        losses.append(float(total_loss.detach().cpu().item()))
        data_metrics.append(float(loss_data.detach().cpu().item()))
        physics_metrics.append(float(loss_physics.detach().cpu().item()))
        weight_metrics.append((w_data, w_physics))
        lr_metrics.append(float(optimizer.param_groups[0]["lr"]))

        if writer is not None:
            writer.add_scalars(
                "Loss",
                {
                    "total": losses[-1],
                    "data": data_metrics[-1],
                    "physics": physics_metrics[-1],
                },
                epoch,
            )
            writer.add_scalar("Learning_Rate", lr_metrics[-1], epoch)
            writer.add_scalars("Weights", {"w_data": w_data, "w_physics": w_physics}, epoch)

        if (epoch + 1) % 50 == 0:
            print(
                f"[{group_name}] Epoch {epoch + 1:04d} | Loss: {losses[-1]:.3e} | "
                f"Data: {data_metrics[-1]:.2e} | Physics: {physics_metrics[-1]:.2e} | "
                f"LR: {lr_metrics[-1]:.2e}"
            )

    # Full evaluation
    model.eval()
    with torch.no_grad():
        X_full = torch.tensor(X_scaled, dtype=torch.float32)
        pred_scaled = model(X_full).squeeze()
        ids_pred = pred_scaled * target_std + target_mean

        time_raw = X_full[:, 0] * input_std[0] + input_mean[0]
        vds_raw = X_full[:, 1] * input_std[1] + input_mean[1]
        vgs_raw = X_full[:, 2] * input_std[2] + input_mean[2]
        T_mean, ids_physics = integrate_temperature(time_raw, vds_raw, vgs_raw, model, cfg)

    if writer is not None:
        param_keys = [f"param{i}" for i in range(1, 10)] + [
            "epi_thickness",
            "T_initial",
        ]
        for pk in param_keys:
            if hasattr(model, pk):
                val = getattr(model, pk)
                writer.add_scalar(f"Params/{pk}", float(val.detach().cpu().item()), cfg.num_epochs)
        writer.flush()
        writer.close()

    # Save txt outputs
    group_dir = os.path.join(cfg.output_dir, _safe_filename(group_name))
    _ensure_dir(group_dir)

    raw_out = np.concatenate([time_col, ids_col, vds_col, vgs_col], axis=1)
    _save_txt(
        os.path.join(group_dir, "raw_time_ids_vds_vgs.txt"),
        raw_out,
        header="time_s Ids_A Vds_V Vgs_V",
    )

    fit_out = np.column_stack(
        [
            time_raw.cpu().numpy(),
            ids_col.squeeze(),
            ids_pred.cpu().numpy(),
            ids_physics.cpu().numpy(),
            T_mean.cpu().numpy(),
        ]
    )
    _save_txt(
        os.path.join(group_dir, "fit_time_ids_true_pred_physics_Tmean.txt"),
        fit_out,
        header="time_s Ids_true_A Ids_pred_A Ids_physics_A T_mean_K",
    )

    loss_out = np.column_stack(
        [
            np.arange(1, cfg.num_epochs + 1),
            np.array(losses),
            np.array(data_metrics),
            np.array(physics_metrics),
            np.array([w[0] for w in weight_metrics]),
            np.array([w[1] for w in weight_metrics]),
            np.array(lr_metrics),
        ]
    )
    _save_txt(
        os.path.join(group_dir, "loss_curves.txt"),
        loss_out,
        header="epoch total_loss data_loss physics_loss w_data w_physics lr",
    )

    # Save checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name = _safe_filename(
        f"{group_name}_Vds{labels.get('vds_max','')}_Vgs{labels.get('vgs_max','')}_{timestamp}.pt"
    )
    ckpt_path = os.path.join(cfg.checkpoint_dir, ckpt_name)
    _ensure_dir(cfg.checkpoint_dir)

    checkpoint = {
        "group_name": group_name,
        "labels": labels,
        "config": asdict(cfg),
        "model_state": model.state_dict(),
        "input_scaler": {"mean": input_scaler.mean_.tolist(), "scale": input_scaler.scale_.tolist()},
        "target_scaler": {"mean": float(target_scaler.mean_[0]), "scale": float(target_scaler.scale_[0])},
        "loss": {
            "total": losses,
            "data": data_metrics,
            "physics": physics_metrics,
            "w_data": [w[0] for w in weight_metrics],
            "w_physics": [w[1] for w in weight_metrics],
            "lr": lr_metrics,
        },
    }
    torch.save(checkpoint, ckpt_path)

    learned = {
        **labels,
        "checkpoint": ckpt_path,
        "param1": float(model.param1.detach().cpu().item()),
        "param2": float(model.param2.detach().cpu().item()),
        "param3": float(model.param3.detach().cpu().item()),
        "param4": float(model.param4.detach().cpu().item()),
        "param5": float(model.param5.detach().cpu().item()),
        "param6": float(model.param6.detach().cpu().item()),
        "param7": float(model.param7.detach().cpu().item()),
        "param8": float(model.param8.detach().cpu().item()),
        "param9": float(model.param9.detach().cpu().item()),
        "epi_thickness": float(model.epi_thickness.detach().cpu().item()),
        "T_initial": float(model.T_initial.detach().cpu().item()),
    }
    return learned


def main() -> None:
    _ensure_dir(CFG.output_dir)
    _ensure_dir(CFG.checkpoint_dir)

    groups = load_hdf5_groups(CFG.hdf5_path)
    if not groups:
        raise RuntimeError(f"未在 {CFG.hdf5_path} 中找到可训练的组数据")

    summary_csv = os.path.join(CFG.output_dir, "learned_params_all_groups.csv")
    header = [
        "group",
        "source",
        "vds_max",
        "vgs_max",
        "time_max",
        "checkpoint",
        "param1",
        "param2",
        "param3",
        "param4",
        "param5",
        "param6",
        "param7",
        "param8",
        "param9",
        "epi_thickness",
        "T_initial",
    ]

    for group_name, data_np, labels in groups:
        print(f"\n=== Training group: {group_name} ===")
        learned = train_one_group(group_name, data_np, labels, CFG)
        row = {k: learned.get(k, "") for k in header}
        _append_csv_row(summary_csv, header, row)

    print(f"\nAll groups finished. Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
