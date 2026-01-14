"""从训练 checkpoint 推理：输出平均结温/电流随时间。

用法示例：
  python scripts/infer_from_checkpoint.py --checkpoint checkpoints/xxx.pt
  python scripts/infer_from_checkpoint.py --all --checkpoint-dir checkpoints
输出：
  inference/<ckpt_stem>/
    raw_time_ids_vds_vgs.txt
    infer_time_ids_true_pred_physics_Tmean.txt

说明：
- 该脚本不会画图，只负责推理与落盘 txt。
- 绘图请用 scripts/plot_paper_figures.py。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch

try:
    import h5py  # type: ignore
except ImportError:  # pragma: no cover
    h5py = None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


def _save_txt(path: str, array_2d: np.ndarray, header: str) -> None:
    _ensure_dir(os.path.dirname(path))
    np.savetxt(path, array_2d, fmt="%.10e", header=header)


# -------------------- 从训练脚本复制的模型/物理/温度逻辑 --------------------


@dataclass(frozen=True)
class InferConfig:
    # 下面默认值仅兜底，实际会从 checkpoint["config"] 覆盖
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
    max_substep_dt: float = 2e-9
    min_temperature: float = 1.0


class PINN(torch.nn.Module):
    def __init__(self, dropout_rate: float = 0.0):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 128),
            torch.nn.Tanh(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 1),
        )

        self.param1 = torch.nn.Parameter(torch.tensor(110.0, dtype=torch.float32))
        self.param2 = torch.nn.Parameter(torch.tensor(4.0, dtype=torch.float32))
        self.param3 = torch.nn.Parameter(torch.tensor(10.6, dtype=torch.float32))
        self.param4 = torch.nn.Parameter(torch.tensor(0.0011, dtype=torch.float32))
        self.param5 = torch.nn.Parameter(torch.tensor(-0.102, dtype=torch.float32))
        self.param6 = torch.nn.Parameter(torch.tensor(10.0, dtype=torch.float32))
        self.param7 = torch.nn.Parameter(torch.tensor(-0.1, dtype=torch.float32))
        self.param8 = torch.nn.Parameter(torch.tensor(5.0, dtype=torch.float32))
        self.param9 = torch.nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))

        # epi_thickness 固定常数，保持与训练脚本一致
        self.register_buffer("epi_thickness", torch.tensor(1.0e-5, dtype=torch.float32))
        self.T_initial = torch.nn.Parameter(torch.tensor(350.0, dtype=torch.float32))

        self.vds_coef_slope = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.vds_coef_intercept = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

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
    *,
    min_temperature: float,
) -> torch.Tensor:
    if not isinstance(Vds, torch.Tensor):
        Vds = torch.tensor(Vds, dtype=torch.float32)
    if not isinstance(Vgs, torch.Tensor):
        Vgs = torch.tensor(Vgs, dtype=torch.float32)
    if not isinstance(T, torch.Tensor):
        T = torch.tensor(T, dtype=torch.float32)

    coef = torch.sigmoid(vds_coef_slope * T + vds_coef_intercept)
    # 与当前训练脚本保持一致，暂不缩放 Vds
    # Vds_eff = Vds * coef
    Vds_eff = Vds * 1

    T_safe = torch.clamp(T, min=min_temperature)

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


_CHEB_CACHE: Dict[Tuple[int, int, str], torch.Tensor] = {}


def _chebyshev_diff_matrix_torch(N: int, order: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (N, order, str(device))
    if key in _CHEB_CACHE:
        return _CHEB_CACHE[key].to(device=device, dtype=dtype)

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

        _CHEB_CACHE[key] = D.detach().cpu()
        return D

    if order == 2:
        D1 = _chebyshev_diff_matrix_torch(N, 1, device=device, dtype=dtype)
        D2 = D1 @ D1
        _CHEB_CACHE[key] = D2.detach().cpu()
        return D2

    raise ValueError("只支持一阶和二阶导数")


def integrate_temperature(time_raw, vds_raw, vgs_raw, model, cfg: InferConfig):
    """保持与训练脚本一致的返回：每步芯片最高温度序列与物理电流。"""
    device = time_raw.device
    dtype = time_raw.dtype

    a, b, c = cfg.a, cfg.b, cfg.c
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz

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

    T_field = torch.ones((Nx, Ny, Nz), device=device, dtype=torch.float32) * model.T_initial
    T_mean_seq = torch.ones_like(time_raw, dtype=torch.float32) * model.T_initial
    ids_physics = torch.zeros_like(vds_raw)

    delta_time = torch.zeros_like(time_raw)
    delta_time[1:] = time_raw[1:] - time_raw[:-1]

    epi_thickness = model.epi_thickness

    X, Y, Z = torch.meshgrid(x_mapped, y_mapped, z_mapped, indexing="ij")
    region_mask = ~((X > 0.8 * a) & (Y > 3.0 / 8.0 * b) & (Y < 5.0 / 8.0 * b))

    prev_T_max: torch.Tensor | None = None
    for i in range(len(time_raw)):
        T_prev = T_field.max() if prev_T_max is None else prev_T_max
        ids_prev = physics_model(
            vds_raw[i],
            vgs_raw[i],
            T_prev,
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
            min_temperature=cfg.min_temperature,
        )

        dt_i = float(delta_time[i].detach().cpu().item()) if i > 0 else 0.0
        if dt_i > 0:
            n_sub = max(1, int(np.ceil(dt_i / cfg.max_substep_dt)))
            sub_dt = dt_i / n_sub

            z_prof = torch.where(Z <= epi_thickness, 1.0 - Z / torch.clamp(epi_thickness, min=1e-12), 0.0)
            base_value = ids_prev * (2.0 * vds_raw[i]) / (a * b * c)
            P = base_value * z_prof * region_mask.to(z_prof.dtype)

            for _ in range(n_sub):
                T_safe = torch.clamp(T_field, min=cfg.min_temperature)
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
                T_new = T_field + (sub_dt / (cfg.rho * C)) * rhs

                T_new[0, :, :] = T_new[1, :, :]
                T_new[-1, :, :] = T_new[-2, :, :]
                T_new[:, 0, :] = T_new[:, 1, :]
                T_new[:, -1, :] = T_new[:, -2, :]

                if cfg.h_z0 > 0:
                    k_surf = torch.clamp(k[:, :, 0], min=1e-12)
                    beta = (cfg.h_z0 * dz0) / k_surf
                    T_new[:, :, 0] = (T_new[:, :, 1] + beta * cfg.T_inf) / (1.0 + beta)
                else:
                    T_new[:, :, 0] = T_new[:, :, 1]

                if cfg.h_zc > 0:
                    k_surf = torch.clamp(k[:, :, -1], min=1e-12)
                    beta = (cfg.h_zc * dzc) / k_surf
                    T_new[:, :, -1] = (T_new[:, :, -2] + beta * cfg.T_inf) / (1.0 + beta)
                else:
                    T_new[:, :, -1] = T_new[:, :, -2]

                T_field = T_new

        T_mean = T_field.max()
        T_mean_seq[i] = T_mean
        ids_physics[i] = physics_model(
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
            min_temperature=cfg.min_temperature,
        )

        prev_T_max = T_mean

    return T_mean_seq, ids_physics


# -------------------- HDF5 读取与主流程 --------------------


def _load_group_from_hdf5(hdf5_path: str, group_name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    if h5py is None:
        raise RuntimeError("缺少依赖 h5py：请先安装 `pip install h5py` 后再运行。")

    with h5py.File(hdf5_path, "r") as h5f:
        grp = h5f[group_name]
        data = np.array(grp["data"], dtype=np.float32)
        labels: Dict[str, Any] = {
            "group": group_name,
            "source": grp.attrs.get("source", ""),
            "vds_max": float(grp.attrs.get("vds_max", np.nan)),
            "vgs_max": float(grp.attrs.get("vgs_max", np.nan)),
            "time_max": float(grp.attrs.get("time_max", np.nan)),
            "columns": list(grp.attrs.get("columns", [])),
        }
    return data, labels


def _run_inference_for_checkpoint(ckpt_path: str, args) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    group_name = args.group or ckpt.get("group_name")
    if not group_name:
        raise RuntimeError(f"无法确定 group_name：请通过 --group 指定或在 checkpoint 中保存 group_name。({ckpt_path})")

    cfg_dict = ckpt.get("config", {})
    cfg = InferConfig(
        a=float(cfg_dict.get("a", InferConfig.a)),
        b=float(cfg_dict.get("b", InferConfig.b)),
        c=float(cfg_dict.get("c", InferConfig.c)),
        rho=float(cfg_dict.get("rho", InferConfig.rho)),
        Nx=int(cfg_dict.get("Nx", InferConfig.Nx)),
        Ny=int(cfg_dict.get("Ny", InferConfig.Ny)),
        Nz=int(cfg_dict.get("Nz", InferConfig.Nz)),
        h_z0=float(cfg_dict.get("h_z0", InferConfig.h_z0)),
        h_zc=float(cfg_dict.get("h_zc", InferConfig.h_zc)),
        T_inf=float(cfg_dict.get("T_inf", InferConfig.T_inf)),
        max_substep_dt=float(cfg_dict.get("max_substep_dt", InferConfig.max_substep_dt)),
        min_temperature=float(cfg_dict.get("min_temperature", InferConfig.min_temperature)),
    )

    hdf5_path = args.hdf5 or cfg_dict.get("hdf5_path") or "combined_training_data.h5"

    data_np, labels = _load_group_from_hdf5(hdf5_path, group_name)

    time_col = data_np[:, 0:1]
    ids_true = data_np[:, 1:2]
    vds_col = data_np[:, 2:3]
    vgs_col = data_np[:, 3:4]

    in_mean = np.array(ckpt["input_scaler"]["mean"], dtype=np.float32)
    in_scale = np.array(ckpt["input_scaler"]["scale"], dtype=np.float32)
    tgt_mean = float(ckpt["target_scaler"]["mean"])
    tgt_scale = float(ckpt["target_scaler"]["scale"])

    X = np.concatenate([time_col, vds_col, vgs_col], axis=1)
    X_scaled = (X - in_mean) / in_scale

    model = PINN()
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    with torch.no_grad():
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        ids_pred = model(X_t).squeeze() * tgt_scale + tgt_mean

        time_raw = torch.tensor(time_col.squeeze(), dtype=torch.float32)
        vds_raw = torch.tensor(vds_col.squeeze(), dtype=torch.float32)
        vgs_raw = torch.tensor(vgs_col.squeeze(), dtype=torch.float32)

        T_mean, ids_physics = integrate_temperature(time_raw, vds_raw, vgs_raw, model, cfg)

    ckpt_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    out_dir = os.path.join(args.outdir, _safe_filename(ckpt_stem))
    _ensure_dir(out_dir)

    raw_out = np.concatenate([time_col, ids_true, vds_col, vgs_col], axis=1)
    _save_txt(os.path.join(out_dir, "raw_time_ids_vds_vgs.txt"), raw_out, "time_s Ids_A Vds_V Vgs_V")

    fit_out = np.column_stack(
        [
            time_col.squeeze(),
            ids_true.squeeze(),
            ids_pred.cpu().numpy(),
            ids_physics.cpu().numpy(),
            T_mean.cpu().numpy(),
        ]
    )
    _save_txt(
        os.path.join(out_dir, "infer_time_ids_true_pred_physics_Tmean.txt"),
        fit_out,
        "time_s Ids_true_A Ids_pred_A Ids_physics_A T_mean_K",
    )

    meta = {
        **labels,
        "checkpoint": ckpt_path,
        "hdf5": hdf5_path,
        "out_dir": out_dir,
    }
    print("Inference done:")
    for k, v in meta.items():
        print(f"  {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="单个 checkpoint .pt；若使用 --all 可忽略此项")
    ap.add_argument("--checkpoint-dir", default="checkpoints", help="批量推理的目录，需包含 .pt 文件")
    ap.add_argument("--all", action="store_true", help="对 checkpoint-dir 下所有 .pt 执行推理")
    ap.add_argument("--hdf5", default=None, help="combined_training_data.h5 路径（默认读取 checkpoint 中的 config）")
    ap.add_argument("--group", default=None, help="HDF5 group name（默认读取 checkpoint 内保存的 group_name）")
    ap.add_argument("--outdir", default="inference", help="输出目录")
    args = ap.parse_args()

    if args.all:
        if not os.path.isdir(args.checkpoint_dir):
            raise RuntimeError(f"找不到 checkpoint 目录: {args.checkpoint_dir}")
        ckpt_files = sorted(
            [os.path.join(args.checkpoint_dir, f) for f in os.listdir(args.checkpoint_dir) if f.endswith('.pt')]
        )
        if not ckpt_files:
            raise RuntimeError(f"目录 {args.checkpoint_dir} 下未找到 .pt 文件")
    else:
        if not args.checkpoint:
            raise RuntimeError("请提供 --checkpoint 或使用 --all")
        ckpt_files = [args.checkpoint]

    for ckpt_path in ckpt_files:
        _run_inference_for_checkpoint(ckpt_path, args)


if __name__ == "__main__":
    main()
