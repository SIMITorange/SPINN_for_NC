"""Train a PINN without Chebyshev temperature solver.

This script mirrors the provided reference code but cleans the flow and keeps
all temperature updates in a lightweight loop (no spectral solver). It reads
Excel sheets, trains a small PINN with a physics-informed loss, then plots
currents and temperature.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    excel_path: str = "processed_data_only_400_600_new.xlsx"
    batch_size: int = 300
    num_epochs: int = 150
    alpha: float = 0.9  # EWMA for loss weights
    lr_net: float = 1e-3
    lr_params: float = 1e-1
    weight_decay: float = 1e-4
    clip_grad_norm: float = 1.0
    scheduler_patience: int = 20
    scheduler_factor: float = 0.5
    output_dir: str = "results_no_cheb"
    plots_dir: str = "results_no_cheb/plots"


CFG = Config()


# -------------------------
# Utilities
# -------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_excel_data(path: str) -> np.ndarray:
    """Load all sheets and standardize column names/order to [Ids, Time, Vds, Vgs]."""
    excel_file = pd.ExcelFile(path)
    dfs: List[pd.DataFrame] = []
    for sheet in excel_file.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet)
        df.columns = df.columns.str.strip().str.lower()
        df = df.rename(
            columns={
                "time(s)": "Time",
                "vds": "Vds",
                "vgs": "Vgs",
                "ids": "Ids",
            }
        )[["Ids", "Time", "Vds", "Vgs"]]
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    return combined.to_numpy()


# -------------------------
# Model
# -------------------------
class PINN(nn.Module):
    def __init__(self, dropout_rate: float = 0.0) -> None:
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
        # Trainable physics parameters
        self.param1 = nn.Parameter(torch.tensor(110.0, dtype=torch.float32))
        self.param2 = nn.Parameter(torch.tensor(4.0, dtype=torch.float32))
        self.param3 = nn.Parameter(torch.tensor(10.6, dtype=torch.float32))
        self.param4 = nn.Parameter(torch.tensor(0.0011, dtype=torch.float32))
        self.param5 = nn.Parameter(torch.tensor(-0.102, dtype=torch.float32))
        self.param6 = nn.Parameter(torch.tensor(10.0, dtype=torch.float32))
        self.param7 = nn.Parameter(torch.tensor(-0.1, dtype=torch.float32))
        self.param8 = nn.Parameter(torch.tensor(5.0, dtype=torch.float32))
        self.param9 = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))
        # Thermal parameters
        self.epi_thickness = nn.Parameter(torch.tensor(1.0e-5, dtype=torch.float32))
        self.T_initial = nn.Parameter(torch.tensor(350.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [N, 3] (Time, Vds, Vgs)
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
    """Original physics formula with temperature as input."""
    if not isinstance(Vds, torch.Tensor):
        Vds = torch.tensor(Vds, dtype=torch.float32)
    if not isinstance(Vgs, torch.Tensor):
        Vgs = torch.tensor(Vgs, dtype=torch.float32)
    if not isinstance(T, torch.Tensor):
        T = torch.tensor(T, dtype=torch.float32)

    net5 = param3 * 1.0 / (param1 * (T / 300.0) ** -0.01 + param6 * (T / 300.0) ** param2)
    net3 = -0.004263 * T + 3.422579
    p9 = param5
    net2 = -0.005 * Vgs + 0.165
    net1 = -0.1717 * Vgs + 3.5755
    term3 = (torch.log1p(torch.exp(Vgs - net3))) ** 2 - (
        torch.log1p(torch.exp(Vgs - net3 - (net2 * Vds * ((1 + torch.exp(p9 * Vds)) ** net1))))
    ) ** 2
    term1 = net5 * (Vgs - net3)
    term2 = 1 + 0.0005 * Vds
    return term2 * term1 * term3


# -------------------------
# Temperature integrator (simple, no Chebyshev)
# -------------------------
def integrate_temperature(
    time_raw: torch.Tensor,
    vds_raw: torch.Tensor,
    vgs_raw: torch.Tensor,
    model: PINN,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward Euler temperature update and physics predictions.

    T[0] = T_initial
    T[i] = T[i-1] + dT, where dT depends on Vds, Ids (from physics), and dt.
    """
    T = torch.full_like(time_raw, model.T_initial)
    physics_preds = torch.zeros_like(vds_raw)
    delta_time = torch.zeros_like(time_raw)
    delta_time[1:] = time_raw[1:] - time_raw[:-1]

    epi_thickness = torch.nn.functional.softplus(model.epi_thickness)

    for i in range(len(time_raw)):
        if i > 0:
            ids_prev = physics_model(
                vds_raw[i],
                vgs_raw[i],
                T[i - 1].detach(),
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
            delta_T = 2 * vds_raw[i] / epi_thickness * (ids_prev / 20e-6) * delta_time[i] / (3200 * 600)
            current_T = T[i - 1] + delta_T
        else:
            current_T = model.T_initial

        T[i] = current_T
        physics_preds[i] = physics_model(
            vds_raw[i],
            vgs_raw[i],
            T[i],
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

    return T, physics_preds


# -------------------------
# Training
# -------------------------
def train() -> None:
    raw_data = _load_excel_data(CFG.excel_path)
    inputs_data = raw_data[:, 1:].copy()  # [Time, Vds, Vgs]
    targets_data = raw_data[:, 0].reshape(-1, 1)

    input_scaler = StandardScaler()
    target_scaler = StandardScaler()
    X_scaled = input_scaler.fit_transform(inputs_data)
    y_scaled = target_scaler.fit_transform(targets_data)

    input_mean = torch.tensor(input_scaler.mean_, dtype=torch.float32)
    input_std = torch.tensor(input_scaler.scale_, dtype=torch.float32)
    target_mean = torch.tensor(target_scaler.mean_[0], dtype=torch.float32)
    target_std = torch.tensor(target_scaler.scale_[0], dtype=torch.float32)

    dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y_scaled, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=min(CFG.batch_size, len(dataset)), shuffle=False)

    model = PINN()
    optimizer = optim.AdamW(
        [
            {"params": model.net.parameters(), "lr": CFG.lr_net},
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
                "lr": CFG.lr_params,
            },
        ],
        weight_decay=CFG.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        patience=CFG.scheduler_patience,
        factor=CFG.scheduler_factor,
    )
    mse_loss = nn.MSELoss()

    alpha = CFG.alpha
    w_data, w_physics = 0.5, 0.5
    losses: List[float] = []
    data_metrics: List[float] = []
    physics_metrics: List[float] = []

    for epoch in range(CFG.num_epochs):
        model.train()
        for X_batch, y_batch in dataloader:
            pred = model(X_batch)
            loss_data = mse_loss(pred, y_batch)

            time_raw = X_batch[:, 0] * input_std[0] + input_mean[0]
            vds_raw = X_batch[:, 1] * input_std[1] + input_mean[1]
            vgs_raw = X_batch[:, 2] * input_std[2] + input_mean[2]

            T_seq, physics_preds = integrate_temperature(time_raw, vds_raw, vgs_raw, model)
            physics_pred_scaled = (physics_preds - target_mean) / target_std
            loss_physics = mse_loss(pred.squeeze(), physics_pred_scaled)

            current_data = float(loss_data.detach().cpu())
            current_physics = float(loss_physics.detach().cpu())
            total = current_data + current_physics + 1e-8
            new_wd = current_data / total
            new_wp = current_physics / total
            w_data = alpha * w_data + (1 - alpha) * new_wd
            w_physics = alpha * w_physics + (1 - alpha) * new_wp

            total_loss = w_data * loss_data + w_physics * loss_physics

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.clip_grad_norm)
            optimizer.step()

        scheduler.step(float(total_loss.detach().cpu()))
        losses.append(float(total_loss.detach().cpu()))
        data_metrics.append(float(loss_data.detach().cpu()))
        physics_metrics.append(float(loss_physics.detach().cpu()))

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:04d} | Loss: {losses[-1]:.3e} | "
                f"Data: {data_metrics[-1]:.2e} | Physics: {physics_metrics[-1]:.2e} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"params: {model.param1.item():.3f}, {model.param2.item():.3f}, "
                f"{model.param3.item():.3f}, {model.param4.item():.5f}, {model.param5.item():.3f}, "
                f"{model.param6.item():.3f}, {model.param7.item():.3f}"
            )

    # Evaluation and plots
    _ensure_dir(CFG.output_dir)
    _ensure_dir(CFG.plots_dir)

    def neural(time: float, Vds: float, Vgs: float) -> float:
        inp = np.array([[time, Vds, Vgs]])
        scaled = input_scaler.transform(inp)
        with torch.no_grad():
            scaled_out = model(torch.tensor(scaled, dtype=torch.float32))
        return target_scaler.inverse_transform(scaled_out.numpy())[0][0]

    true_values = targets_data.flatten()
    predicted_values = [neural(t, vds, vgs) for t, vds, vgs in inputs_data]

    with torch.no_grad():
        full_inputs = torch.tensor(X_scaled, dtype=torch.float32)
        full_time = full_inputs[:, 0] * input_std[0] + input_mean[0]
        full_vds = full_inputs[:, 1] * input_std[1] + input_mean[1]
        full_vgs = full_inputs[:, 2] * input_std[2] + input_mean[2]
        T_sequence, physics_full = integrate_temperature(full_time, full_vds, full_vgs, model)
        physics_np = physics_full.detach().cpu().numpy()

    # Plots
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label="True", marker="o", markersize=3)
    plt.plot(predicted_values, label="Pred", marker="x", markersize=3)
    plt.plot(physics_np, label="Physics", marker=".", markersize=3)
    plt.xlabel("Sample Index")
    plt.ylabel("Ids")
    plt.title("True vs Pred vs Physics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.plots_dir, "ids_comparison.png"), dpi=300)

    plt.figure(figsize=(10, 6))
    plt.plot(full_time.detach().cpu().numpy(), T_sequence.detach().cpu().numpy(), label="Temperature", color="r")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.title("Device Temperature Evolution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.plots_dir, "temperature.png"), dpi=300)

    plt.figure()
    plt.plot(np.log10(losses), label="Total")
    plt.plot(np.log10(data_metrics), label="Data")
    plt.plot(np.log10(physics_metrics), label="Physics")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.plots_dir, "loss_curves.png"), dpi=300)

    print(
        f"关键参数: Epi_thickness={model.epi_thickness.item():.4e} m, "
        f"T_initial={model.T_initial.item():.1f} K"
    )
    print(
        f"其他参数: param1={model.param1.item():.3f}, param2={model.param2.item():.3f}, "
        f"param3={model.param3.item():.3f}, param4={model.param4.item():.5f}, "
        f"param5={model.param5.item():.3f}, param6={model.param6.item():.3f}, "
        f"param7={model.param7.item():.3f}, param8={model.param8.item():.3f}, "
        f"param9={model.param9.item():.3f}"
    )


if __name__ == "__main__":
    train()
