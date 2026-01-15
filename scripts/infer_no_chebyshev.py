"""Inference for no-Chebyshev PINN checkpoints.

Example usage:
    # single checkpoint
    python scripts/infer_no_chebyshev.py --checkpoint checkpoints_no_cheb/xxx.pt

    # batch all .pt under a directory
    python scripts/infer_no_chebyshev.py --all --checkpoint-dir checkpoints_no_cheb

    # specify custom HDF5 or group name
    python scripts/infer_no_chebyshev.py --checkpoint checkpoints_no_cheb/xxx.pt --hdf5 combined_training_data.h5 --group 400V__foo

Outputs the same TXT structure as infer_from_checkpoint.py so plotting scripts
remain compatible.
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
    return "".join(ch if ch.isalnum() or ch in "-_ ." else "_" for ch in name)


def _save_txt(path: str, array_2d: np.ndarray, header: str) -> None:
    _ensure_dir(os.path.dirname(path))
    np.savetxt(path, array_2d, fmt="%.10e", header=header)


@dataclass(frozen=True)
class InferConfig:
    rho: float = 3200.0
    c_th: float = 600.0
    ids_area: float = 20e-6
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
    Vds_eff = Vds * 1  # kept unchanged for consistency

    T_safe = torch.clamp(T, min=min_temperature)

    net5 = param3 * 1.0 / (param1 * (T_safe / 300.0) ** -0.01 + param6 * (T_safe / 300.0) ** param2)
    net3 = -0.004263 * T_safe + 3.422579
    p9 = param5
    net2 = -0.005 * Vgs + 0.165
    net1 = -0.1717 * Vgs + 3.5755
    term3 = (torch.log1p(torch.exp(Vgs - net3))) ** 2 - (
        torch.log1p(torch.exp(Vgs - net3 - (net2 * Vds_eff * ((1 + torch.exp(p9 * Vds_eff)) ** net1))))
    ) ** 2
    term1 = net5 * (Vgs - net3)
    term2 = 1 + 0.0005 * Vds_eff
    return term2 * term1 * term3


def integrate_temperature(
    time_raw: torch.Tensor,
    vds_raw: torch.Tensor,
    vgs_raw: torch.Tensor,
    model: PINN,
    cfg: InferConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(time_raw, torch.Tensor):
        time_raw = torch.tensor(time_raw, dtype=torch.float32)
    if not isinstance(vds_raw, torch.Tensor):
        vds_raw = torch.tensor(vds_raw, dtype=torch.float32)
    if not isinstance(vgs_raw, torch.Tensor):
        vgs_raw = torch.tensor(vgs_raw, dtype=torch.float32)

    device = time_raw.device
    dtype = time_raw.dtype

    T = torch.ones_like(time_raw, dtype=dtype, device=device) * model.T_initial
    physics_preds = torch.zeros_like(vds_raw, dtype=dtype, device=device)

    delta_time = torch.zeros_like(time_raw, dtype=dtype, device=device)
    delta_time[1:] = time_raw[1:] - time_raw[:-1]

    epi = torch.clamp(model.epi_thickness, min=1e-9)
    rho_cp = cfg.rho * cfg.c_th
    ids_area = cfg.ids_area

    for i in range(len(time_raw)):
        if i > 0:
            ids_prev = physics_model(
                vds_raw[i],
                vgs_raw[i],
                T[i - 1],
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
            delta_T = (2.0 * vds_raw[i] / epi) * (ids_prev / ids_area) * delta_time[i] / rho_cp
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
            model.vds_coef_slope,
            model.vds_coef_intercept,
            min_temperature=cfg.min_temperature,
        )

    return torch.clamp(T, min=cfg.min_temperature), physics_preds


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
        rho=float(cfg_dict.get("rho", InferConfig.rho)),
        c_th=float(cfg_dict.get("c_th", InferConfig.c_th)),
        ids_area=float(cfg_dict.get("ids_area", InferConfig.ids_area)),
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
    ap.add_argument("--checkpoint-dir", default="checkpoints_no_cheb", help="批量推理目录，需包含 .pt 文件")
    ap.add_argument("--all", action="store_true", help="对 checkpoint-dir 下所有 .pt 执行推理")
    ap.add_argument("--hdf5", default=None, help="combined_training_data.h5 路径（默认读取 checkpoint 中的 config）")
    ap.add_argument("--group", default=None, help="HDF5 group name（默认读取 checkpoint 内保存的 group_name）")
    ap.add_argument("--outdir", default="inference_no_cheb", help="输出目录")
    args = ap.parse_args()

    if args.all:
        if not os.path.isdir(args.checkpoint_dir):
            raise RuntimeError(f"找不到 checkpoint 目录: {args.checkpoint_dir}")
        ckpt_files = sorted(
            [os.path.join(args.checkpoint_dir, f) for f in os.listdir(args.checkpoint_dir) if f.endswith(".pt")]
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
