from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import h5py  # type: ignore

@dataclass(frozen=True)
class TrainConfig:
    # 数据
    hdf5_path: str = "combined_training_data.h5"
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    # 训练
    num_epochs: int = 150
    batch_size: int = 300
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
    tb_log_dir: str = "runs"
    # 热传导（Chebyshev）
    a: float = 1.5e-3
    b: float = 1e-3
    c: float = 10e-5
    rho: float = 3200.0
    Nx: int = 10
    Ny: int = 10
    Nz: int = 10
    h_z0: float = 5e4
    h_zc: float = 0.0
    T_inf: float = 300.0
    # 数值稳定
    max_substep_dt: float = 2e-9
    min_temperature: float = 1.0


CFG = TrainConfig()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


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
    """返回 [(group_name, data_np, labels_dict)]

    data_np 形状: (N, 4) 列顺序通常是 [time(s), Ids, Vds, Vgs]。
    labels_dict 来自 group.attrs。
    """
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

        self.param1 = nn.Parameter(torch.tensor(110.0, dtype=torch.float32))
        self.param2 = nn.Parameter(torch.tensor(4.0, dtype=torch.float32))
        self.param3 = nn.Parameter(torch.tensor(10.6, dtype=torch.float32))
        self.param4 = nn.Parameter(torch.tensor(0.0011, dtype=torch.float32))
        self.param5 = nn.Parameter(torch.tensor(-0.102, dtype=torch.float32))
        self.param6 = nn.Parameter(torch.tensor(10.0, dtype=torch.float32))
        self.param7 = nn.Parameter(torch.tensor(-0.1, dtype=torch.float32))
        self.param8 = nn.Parameter(torch.tensor(5.0, dtype=torch.float32))
        self.param9 = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))

        # 热相关参数（epi_thickness 固定为常数，不参与训练）
        self.register_buffer("epi_thickness", torch.tensor(1.0e-5, dtype=torch.float32))
        # self.epi_thickness = nn.Parameter(torch.tensor(1.0e-5, dtype=torch.float32))
        self.T_initial = nn.Parameter(torch.tensor(350.0, dtype=torch.float32))

        # 修改4：Vds 系数（一次函数经 sigmoid 约束到 0~1）
        self.vds_coef_slope = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.vds_coef_intercept = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

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
    vds_coef_slope: torch.Tensor,
    vds_coef_intercept: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(Vds, torch.Tensor):
        Vds = torch.tensor(Vds, dtype=torch.float32)
    if not isinstance(Vgs, torch.Tensor):
        Vgs = torch.tensor(Vgs, dtype=torch.float32)
    if not isinstance(T, torch.Tensor):
        T = torch.tensor(T, dtype=torch.float32)

    # Vds 有效系数改为温度的一次函数后取 sigmoid
    coef = torch.sigmoid(vds_coef_slope * T + vds_coef_intercept)
    Vds_eff = Vds * coef

    T_safe = torch.clamp(T, min=CFG.min_temperature)

    NET5_value = param3 * 1.0 / (param1 * (T_safe / 300.0) ** -0.01 + param6 * (T_safe / 300.0) ** param2)
    NET3_value = -0.004263 * T_safe + 3.422579
    p9 = param5
    NET2_value = -0.005 * Vgs + 0.165
    NET1_value = -0.1717 * Vgs + 3.5755
    term3 = (torch.log(1 + torch.exp(Vgs - NET3_value))) ** 2 - (
        torch.log(
            1
            + torch.exp(
                Vgs
                - NET3_value
                - (NET2_value * Vds_eff * ((1 + torch.exp(p9 * Vds_eff)) ** NET1_value))
            )
        )
        ** 2
    )
    term1 = NET5_value * (Vgs - NET3_value)
    term2 = 1 + 0.0005 * Vds_eff
    return term2 * term1 * term3


_CHEB_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}


def _chebyshev_diff_matrix_torch(N: int, order: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """torch 版本的 Chebyshev 一/二阶微分矩阵（与 chebeshev.py 一致的构造）。"""
    key = (N, order)
    cache_key = (key[0] * 10 + key[1], id(device))
    # 简单 device 级别缓存（避免反复构建）
    if cache_key in _CHEB_CACHE:
        return _CHEB_CACHE[cache_key].to(device=device, dtype=dtype)

    if order == 1:
        x = torch.cos(torch.pi * torch.arange(N, device=device, dtype=dtype) / (N - 1))
        c = torch.ones(N, device=device, dtype=dtype)
        c[0] = 2.0
        c[-1] = 2.0

        ii = torch.arange(N, device=device)
        I, J = torch.meshgrid(ii, ii, indexing="ij")
        D = torch.zeros((N, N), device=device, dtype=dtype)
        mask = I != J

        sign = torch.where(((I + J) % 2) == 0, 1.0, -1.0).to(dtype)
        D[mask] = (c[I[mask]] / c[J[mask]]) * sign[mask] / (x[I[mask]] - x[J[mask]])
        D.fill_diagonal_(0.0)
        D.diagonal().copy_(-D.sum(dim=1))

        _CHEB_CACHE[cache_key] = D.detach().cpu()
        return D

    if order == 2:
        D1 = _chebyshev_diff_matrix_torch(N, 1, device=device, dtype=dtype)
        D2 = D1 @ D1
        _CHEB_CACHE[cache_key] = D2.detach().cpu()
        return D2

    raise ValueError("只支持一阶和二阶导数")


def integrate_temperature(time_raw, vds_raw, vgs_raw, model, physics_model):
    """温度积分计算函数（保持原始接口不变）。

    替换为 chebeshev.py 的热传导逻辑（Chebyshev 伪谱 + 显式步进）。
    这里返回的 T 为“温度分布的空间平均值序列”，从而保持与原脚本兼容。
    """
    device = time_raw.device if isinstance(time_raw, torch.Tensor) else torch.device("cpu")
    dtype = time_raw.dtype if isinstance(time_raw, torch.Tensor) else torch.float32

    if not isinstance(time_raw, torch.Tensor):
        time_raw = torch.tensor(time_raw, dtype=torch.float32, device=device)
    if not isinstance(vds_raw, torch.Tensor):
        vds_raw = torch.tensor(vds_raw, dtype=torch.float32, device=device)
    if not isinstance(vgs_raw, torch.Tensor):
        vgs_raw = torch.tensor(vgs_raw, dtype=torch.float32, device=device)

    # 输出：每个时间点的空间平均温度（避免对需要梯度的张量做原地写入）
    # 这里用 list 累积，再 stack；可避免 autograd 的 inplace version mismatch。
    t_mean_list: List[torch.Tensor] = []
    physics_pred_list: List[torch.Tensor] = []

    delta_time = torch.zeros_like(time_raw, dtype=torch.float32)
    delta_time[1:] = time_raw[1:] - time_raw[:-1]

    # 几何/网格
    a, b, c = CFG.a, CFG.b, CFG.c
    Nx, Ny, Nz = CFG.Nx, CFG.Ny, CFG.Nz

    x = -torch.cos(torch.pi * torch.arange(Nx, device=device, dtype=dtype) / (Nx - 1))
    y = -torch.cos(torch.pi * torch.arange(Ny, device=device, dtype=dtype) / (Ny - 1))
    z = -torch.cos(torch.pi * torch.arange(Nz, device=device, dtype=dtype) / (Nz - 1))
    x_mapped = a * (x + 1) / 2
    y_mapped = b * (y + 1) / 2
    z_mapped = c * (z + 1) / 2
    dz0 = (z_mapped[1] - z_mapped[0]).abs()
    dzc = (z_mapped[-1] - z_mapped[-2]).abs()

    Dx = (2.0 / a) * _chebyshev_diff_matrix_torch(Nx, 1, device=device, dtype=dtype)
    Dy = (2.0 / b) * _chebyshev_diff_matrix_torch(Ny, 1, device=device, dtype=dtype)
    Dz = (2.0 / c) * _chebyshev_diff_matrix_torch(Nz, 1, device=device, dtype=dtype)
    Dxx = (2.0 / a) ** 2 * _chebyshev_diff_matrix_torch(Nx, 2, device=device, dtype=dtype)
    Dyy = (2.0 / b) ** 2 * _chebyshev_diff_matrix_torch(Ny, 2, device=device, dtype=dtype)
    Dzz = (2.0 / c) ** 2 * _chebyshev_diff_matrix_torch(Nz, 2, device=device, dtype=dtype)

    # 初始温度场（保持对 T_initial 的梯度）
    T_field = torch.ones((Nx, Ny, Nz), device=device, dtype=torch.float32) * model.T_initial

    epi_thickness = model.epi_thickness
    h_z0 = CFG.h_z0
    h_zc = CFG.h_zc
    T_inf = CFG.T_inf
    rho = CFG.rho

    X, Y, Z = torch.meshgrid(x_mapped, y_mapped, z_mapped, indexing="ij")
    region_mask = ~((X > 0.8 * a) & (Y > 3.0 / 8.0 * b) & (Y < 5.0 / 8.0 * b))

    # 只反馈 z=0 表面的平均温度（该平面温度最高）
    prev_T_surface: Optional[torch.Tensor] = None
    for i in range(len(time_raw)):
        T_surface_prev = T_field[:, :, 0].mean() if prev_T_surface is None else prev_T_surface

        ids_prev = physics_model(
            vds_raw[i],
            vgs_raw[i],
            T_surface_prev,
            model.param1,
            model.param2,
            model.param3,
            model.param4,
            model.param5,
            model.param6,
            model.param7,
            model.param8,
            model.param9,
            model.vds_coef_slope,
            model.vds_coef_intercept,
        )

        dt_i = float(delta_time[i].detach().cpu().item()) if i > 0 else 0.0
        if dt_i > 0:
            n_sub = max(1, int(np.ceil(dt_i / CFG.max_substep_dt)))
            sub_dt = dt_i / n_sub

            # z 分布：z<=epi_thickness 时线性衰减
            z_prof = torch.where(Z <= epi_thickness, 1.0 - Z / torch.clamp(epi_thickness, min=1e-12), 0.0)
            base_value = ids_prev * (2.0 * vds_raw[i]) / (a * b * c)
            P = base_value * z_prof * region_mask.to(z_prof.dtype)

            for _ in range(n_sub):
                T_safe = torch.clamp(T_field, min=CFG.min_temperature)

                denom_k = torch.clamp(-0.0003 + 1.05e-5 * T_safe, min=1e-12)
                k = 1.0 / denom_k
                C = 300.0 * (5.13 - 1001.0 / T_safe + 3.23e4 / (T_safe**2))
                C = torch.clamp(C, min=1e-12)

                d2T_dx2 = torch.einsum("im,mjk->ijk", Dxx, T_field)
                d2T_dy2 = torch.einsum("jm,imk->ijk", Dyy, T_field)
                d2T_dz2 = torch.einsum("km,ijm->ijk", Dzz, T_field)
                laplacian_T = d2T_dx2 + d2T_dy2 + d2T_dz2

                dk_dx = torch.einsum("im,mjk->ijk", Dx, k)
                dT_dx = torch.einsum("im,mjk->ijk", Dx, T_field)
                dk_dy = torch.einsum("jm,imk->ijk", Dy, k)
                dT_dy = torch.einsum("jm,imk->ijk", Dy, T_field)
                dk_dz = torch.einsum("km,ijm->ijk", Dz, k)
                dT_dz = torch.einsum("km,ijm->ijk", Dz, T_field)
                grad_k_dot_grad_T = dk_dx * dT_dx + dk_dy * dT_dy + dk_dz * dT_dz

                rhs = k * laplacian_T + grad_k_dot_grad_T + P
                T_new = T_field + (sub_dt / (rho * C)) * rhs

                # 边界条件：x/y 绝热
                T_new[0, :, :] = T_new[1, :, :]
                T_new[-1, :, :] = T_new[-2, :, :]
                T_new[:, 0, :] = T_new[:, 1, :]
                T_new[:, -1, :] = T_new[:, -2, :]

                # z=0
                if h_z0 > 0:
                    k_surf = torch.clamp(k[:, :, 0], min=1e-12)
                    beta = (h_z0 * dz0) / k_surf
                    T_new[:, :, 0] = (T_new[:, :, 1] + beta * T_inf) / (1.0 + beta)
                else:
                    T_new[:, :, 0] = T_new[:, :, 1]

                # z=c
                if h_zc > 0:
                    k_surf = torch.clamp(k[:, :, -1], min=1e-12)
                    beta = (h_zc * dzc) / k_surf
                    T_new[:, :, -1] = (T_new[:, :, -2] + beta * T_inf) / (1.0 + beta)
                else:
                    T_new[:, :, -1] = T_new[:, :, -2]

                T_field = T_new

        T_mean = T_field[:, :, 0].mean()
        ids_now = physics_model(
            vds_raw[i],
            vgs_raw[i],
            T_mean,
            model.param1,
            model.param2,
            model.param3,
            model.param4,
            model.param5,
            model.param6,
            model.param7,
            model.param8,
            model.param9,
            model.vds_coef_slope,
            model.vds_coef_intercept,
        )

        t_mean_list.append(T_mean)
        physics_pred_list.append(ids_now)
        prev_T_surface = T_mean

        if torch.isnan(ids_now):
            print(f"!!! NaN detected at step {i} !!!")
            print(f"T_mean={T_mean.item()}, Vds={float(vds_raw[i])}, Vgs={float(vgs_raw[i])}")
            break

    # 保持与原接口一致：返回形状与 time_raw 相同的一维序列
    if len(t_mean_list) == 0:
        # 极端兜底：避免空 stack
        T_mean_seq = torch.ones_like(time_raw, dtype=torch.float32) * model.T_initial
        physics_preds = torch.zeros_like(vds_raw, dtype=torch.float32)
        return T_mean_seq, physics_preds

    T_mean_seq = torch.stack(t_mean_list).to(dtype=torch.float32)
    physics_preds = torch.stack(physics_pred_list).to(dtype=torch.float32)

    # 若输入不是 1D（理论上不该发生），做 reshape 以匹配输入
    if T_mean_seq.shape != time_raw.shape:
        T_mean_seq = T_mean_seq.reshape(time_raw.shape)
    if physics_preds.shape != vds_raw.shape:
        physics_preds = physics_preds.reshape(vds_raw.shape)

    return T_mean_seq, physics_preds


def train_one_group(
    group_name: str,
    data_np: np.ndarray,
    labels: Dict[str, Any],
    cfg: TrainConfig,
) -> Dict[str, Any]:
    """单组训练：返回可学习参数汇总 dict（用于最终CSV）。"""
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
    dataloader = DataLoader(dataset, batch_size=min(cfg.batch_size, len(dataset)), shuffle=False)

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
                    model.T_initial,
                    model.vds_coef_slope,
                    model.vds_coef_intercept,
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

            T_mean, physics_preds = integrate_temperature(time_raw, vds_raw, vgs_raw, model, physics_model)

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

        if (epoch + 1) % 10 == 0:
            print(
                f"[{group_name}] Epoch {epoch + 1:04d} | Loss: {losses[-1]:.3e} | "
                f"Data: {data_metrics[-1]:.2e} | Physics: {physics_metrics[-1]:.2e} | "
                f"LR: {lr_metrics[-1]:.2e}"
            )

    # 训练后：生成全量序列用于保存（保持原始数据）
    model.eval()
    with torch.no_grad():
        X_full = torch.tensor(X_scaled, dtype=torch.float32)
        pred_scaled = model(X_full).squeeze()
        ids_pred = pred_scaled * target_std + target_mean

        time_raw = X_full[:, 0] * input_std[0] + input_mean[0]
        vds_raw = X_full[:, 1] * input_std[1] + input_mean[1]
        vgs_raw = X_full[:, 2] * input_std[2] + input_mean[2]
        T_mean, ids_physics = integrate_temperature(time_raw, vds_raw, vgs_raw, model, physics_model)

    if writer is not None:
        param_keys = [f"param{i}" for i in range(1, 10)] + [
            "epi_thickness",
            "T_initial",
            "vds_coef_slope",
            "vds_coef_intercept",
        ]
        for pk in param_keys:
            if hasattr(model, pk):
                val = getattr(model, pk)
                writer.add_scalar(f"Params/{pk}", float(val.detach().cpu().item()), cfg.num_epochs)
        writer.flush()
        writer.close()

    # 保存 txt（所有绘图数据保持原始单位）
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

    # 保存 checkpoint
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
        "vds_coef_slope": float(model.vds_coef_slope.detach().cpu().item()),
        "vds_coef_intercept": float(model.vds_coef_intercept.detach().cpu().item()),
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
        "vds_coef_slope",
        "vds_coef_intercept",
    ]

    for group_name, data_np, labels in groups:
        print(f"\n=== Training group: {group_name} ===")
        learned = train_one_group(group_name, data_np, labels, CFG)
        row = {k: learned.get(k, "") for k in header}
        _append_csv_row(summary_csv, header, row)

    print(f"\nAll groups finished. Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()

