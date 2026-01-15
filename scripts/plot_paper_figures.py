"""论文级绘图脚本（独立）。

读取训练/推理落盘的 txt 数据，输出：
1) 电流拟合效果图（Ids vs time）
2) 训练 loss 曲线图（total/data/physics + 权重/学习率）

用法示例：
  python scripts/plot_paper_figures.py --input results/<group_name>
  python scripts/plot_paper_figures.py --input inference/<ckpt_stem>
  python scripts/plot_paper_figures.py --all --root results
  python scripts/plot_paper_figures.py --all --root inference
要求对应：
- 绘图参数都集中在本文件顶部 CONFIG 区
- 绘图数据本身不在这里生成：由训练/推理脚本以 txt 形式落盘
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# =========================
# CONFIG（绘图参数配置区）
# =========================

@dataclass(frozen=True)
class PlotConfig:
    # 字体与线条
    font_family: str = "Times New Roman"
    font_size: int = 14
    title_size: int = 16
    label_size: int = 14
    tick_size: int = 12
    line_width: float = 2.0

    # 图尺寸
    fig_w: float = 6.5
    fig_h: float = 4.0

    # 颜色
    color_true: str = "black"
    color_pred: str = "tab:blue"
    color_physics: str = "tab:orange"

    # 坐标轴范围（None 表示自动）
    xlim_time: Optional[Tuple[float, float]] = None
    ylim_ids: Optional[float] = None

    # 标题/标签
    title_ids: str = "Drain Current Fitting"
    xlabel_time: str = "Time (s)"
    ylabel_ids: str = "Ids (A)"

    title_loss: str = "Training Loss Curves"
    xlabel_epoch: str = "Epoch"
    ylabel_loss: str = "Loss (log10)"

    # 输出
    save_vector: bool = False  # True->pdf, False->png
    dpi: int = 300


CFG = PlotConfig()


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": CFG.font_family,
            "font.size": CFG.font_size,
            "axes.titlesize": CFG.title_size,
            "axes.labelsize": CFG.label_size,
            "xtick.labelsize": CFG.tick_size,
            "ytick.labelsize": CFG.tick_size,
            "lines.linewidth": CFG.line_width,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "legend.frameon": False,
        }
    )


def _load_txt(path: str) -> np.ndarray:
    return np.loadtxt(path, comments="#")


def plot_ids(input_dir: str) -> str:
    fit_path = os.path.join(input_dir, "fit_time_ids_true_pred_physics_Tmean.txt")
    if not os.path.exists(fit_path):
        fit_path = os.path.join(input_dir, "infer_time_ids_true_pred_physics_Tmean.txt")

    data = _load_txt(fit_path)
    time_s = data[:, 0]
    ids_true = data[:, 1]
    ids_pred = data[:, 2]
    ids_physics = data[:, 3]

    fig, ax = plt.subplots(figsize=(CFG.fig_w, CFG.fig_h))
    ax.plot(time_s, ids_true, label="Measured", color=CFG.color_true)
    ax.plot(time_s, ids_pred, label="NN Pred", color=CFG.color_pred)
    ax.plot(time_s, ids_physics, label="Physics", color=CFG.color_physics)

    ax.set_title(CFG.title_ids)
    ax.set_xlabel(CFG.xlabel_time)
    ax.set_ylabel(CFG.ylabel_ids)
    if CFG.xlim_time is not None:
        ax.set_xlim(*CFG.xlim_time)
    if CFG.ylim_ids is not None:
        ax.set_ylim(*CFG.ylim_ids)
    ax.legend(loc="best")
    fig.tight_layout()

    ext = "pdf" if CFG.save_vector else "png"
    out_path = os.path.join(input_dir, f"fig_ids_fit.{ext}")
    if CFG.save_vector:
        fig.savefig(out_path)
    else:
        fig.savefig(out_path, dpi=CFG.dpi)
    plt.close(fig)
    return out_path


def plot_loss(input_dir: str) -> Optional[str]:
    loss_path = os.path.join(input_dir, "loss_curves.txt")
    if not os.path.exists(loss_path):
        return None

    data = _load_txt(loss_path)
    epoch = data[:, 0]
    total = data[:, 1]
    data_loss = data[:, 2]
    physics_loss = data[:, 3]
    w_data = data[:, 4]
    w_physics = data[:, 5]
    lr = data[:, 6]

    fig, ax = plt.subplots(figsize=(CFG.fig_w, CFG.fig_h))
    ax.plot(epoch, np.log10(np.maximum(total, 1e-30)), label="Total")
    ax.plot(epoch, np.log10(np.maximum(data_loss, 1e-30)), label="Data")
    ax.plot(epoch, np.log10(np.maximum(physics_loss, 1e-30)), label="Physics")
    ax.set_title(CFG.title_loss)
    ax.set_xlabel(CFG.xlabel_epoch)
    ax.set_ylabel(CFG.ylabel_loss)
    ax.legend(loc="best")

    ax2 = ax.twinx()
    ax2.plot(epoch, w_data, "--", label="w_data", color="tab:green")
    ax2.plot(epoch, w_physics, "--", label="w_physics", color="tab:red")
    ax2.plot(epoch, lr / np.max(lr), ":", label="lr(norm)", color="tab:purple")
    ax2.set_ylabel("Weights / LR (norm)")

    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    ext = "pdf" if CFG.save_vector else "png"
    out_path = os.path.join(input_dir, f"fig_loss_curves.{ext}")
    if CFG.save_vector:
        fig.savefig(out_path)
    else:
        fig.savefig(out_path, dpi=CFG.dpi)
    plt.close(fig)
    return out_path


def _has_fit_file(directory: str) -> bool:
    fit_path = os.path.join(directory, "fit_time_ids_true_pred_physics_Tmean.txt")
    infer_path = os.path.join(directory, "infer_time_ids_true_pred_physics_Tmean.txt")
    return os.path.exists(fit_path) or os.path.exists(infer_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="单个目录：results/<group> 或 inference/<ckpt_stem>")
    ap.add_argument("--all", action="store_true", help="遍历 root 下所有包含 fit/infer txt 的目录")
    ap.add_argument("--root", default="results", help="--all 模式下的根目录（如 results 或 inference）")
    args = ap.parse_args()

    _apply_style()

    targets = []
    if args.all:
        if not os.path.isdir(args.root):
            raise RuntimeError(f"根目录不存在: {args.root}")
        for name in sorted(os.listdir(args.root)):
            candidate = os.path.join(args.root, name)
            if os.path.isdir(candidate) and _has_fit_file(candidate):
                targets.append(candidate)
        if not targets:
            raise RuntimeError(f"在 {args.root} 下未找到包含 fit/infer 数据的子目录")
    else:
        if not args.input:
            raise RuntimeError("请提供 --input 或使用 --all")
        targets = [args.input]

    for tdir in targets:
        ids_fig = plot_ids(tdir)
        loss_fig = plot_loss(tdir)
        print(f"Saved: {ids_fig}")
        if loss_fig:
            print(f"Saved: {loss_fig}")
        else:
            print(f"No loss_curves.txt found in {tdir}; skipped loss figure.")


if __name__ == "__main__":
    main()
