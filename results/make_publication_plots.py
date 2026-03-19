from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import win32com.client as win32
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D


# =========================
# Publication plot config
# =========================
CONFIG = {
    "root_dir": Path(__file__).resolve().parent,
    "output_dir": "publication_figures",
    "fit_txt_name": "fit_time_ids_true_pred_physics_Tmean.txt",
    "raw_txt_name": "raw_time_ids_vds_vgs.txt",
    "figure_dpi": 1500,
    "png_facecolor": "white",
    "font_family": "Arial",
    "font_weight": "bold",
    "legend_fontsize": 12,
    "section_title_fontsize": 15,
    "panel_title_fontsize": 10,
    "tick_fontsize": 10,
    "global_label_fontsize": 14,
    "boxplot_title_fontsize": 11.5,
    "boxplot_label_fontsize": 12,
    "line_width_true": 2.0,
    "line_width_pred": 1.9,
    "line_width_physics": 1.85,
    "color_true": "#111111",
    "color_pred": "#0072B2",
    "color_physics": "#D55E00",
    "color_grid": "#d9d9d9",
    "color_spine": "#2d2d2d",
    "grid_alpha": 0.18,
    "panel_facecolor": "#ffffff",
    "title_box_facecolor": "#fbfbfb",
    "title_box_alpha": 0.96,
    "panel_width": 2.7,
    "panel_height": 1.95,
    "figure_top_margin": 0.88,
    "figure_bottom_margin": 0.12,
    "figure_left_margin": 0.06,
    "figure_right_margin": 0.985,
    "section_hspace": 0.08,
    "panel_wspace": 0.12,
    "panel_hspace": 0.18,
    "y_floor": 0.0,
    "y_padding_ratio": 0.04,
    "y_tick_step": 100.0,
    "ytick_step_major": 150.0,
    "xticks_per_panel": 4,
    "xtick_label_decimals": 1,
    "time_scale": 1e6,
    "time_unit_label": "Time (us)",
    "current_label": "IDS (A)",
    "error_metric_name": "Normalized MAE to peak IDS (%)",
    "jitter_seed": 7,
    "box_width": 0.22,
    "box_scatter_alpha": 0.14,
    "box_scatter_size": 8,
    "boxplot_ylim": (0.9, 4.0),
    "spine_linewidth": 1.15,
    "save_individual_prediction_panels": False,
}


@dataclass
class CaseData:
    folder_name: str
    folder_path: Path
    vds_group: int
    panel_title: str
    sort_key: tuple
    fit_data: np.ndarray
    raw_data: np.ndarray | None

    @property
    def time_us(self) -> np.ndarray:
        return self.fit_data[:, 0] * CONFIG["time_scale"]

    @property
    def ids_true(self) -> np.ndarray:
        return self.fit_data[:, 1]

    @property
    def ids_pred(self) -> np.ndarray:
        return self.fit_data[:, 2]

    @property
    def ids_physics(self) -> np.ndarray:
        return self.fit_data[:, 3]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": CONFIG["font_family"],
            "font.weight": CONFIG["font_weight"],
            "axes.labelweight": CONFIG["font_weight"],
            "axes.titleweight": CONFIG["font_weight"],
            "axes.linewidth": CONFIG["spine_linewidth"],
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": CONFIG["png_facecolor"],
            "figure.facecolor": CONFIG["png_facecolor"],
            "axes.facecolor": CONFIG["panel_facecolor"],
        }
    )


def parse_folder_metadata(folder_name: str) -> tuple[int, str, tuple]:
    vds_match = re.match(r"(?P<vds>\d+)V__", folder_name, flags=re.IGNORECASE)
    if not vds_match:
        raise ValueError(f"Cannot parse VDS group from folder name: {folder_name}")
    vds_group = int(vds_match.group("vds"))
    lower_name = folder_name.lower()

    time_match = re.search(r"-(\d+(?:_\d+)?)us", lower_name)
    if time_match:
        pulse_us = float(time_match.group(1).replace("_", "."))
        return vds_group, f"{pulse_us:g} us", (0, pulse_us)

    vgs_match = re.search(r"vgs(\d+(?:_\d+)?)", lower_name)
    if vgs_match:
        vgs_value = float(vgs_match.group(1).replace("_", "."))
        suffix = " NG" if "-ng" in lower_name else ""
        return vds_group, f"VGS {vgs_value:g} V{suffix}", (1, -vgs_value, suffix)

    return vds_group, folder_name, (9, folder_name.lower())


def discover_cases(root_dir: Path) -> list[CaseData]:
    cases: list[CaseData] = []
    for folder in sorted(root_dir.iterdir()):
        if not folder.is_dir():
            continue

        fit_path = folder / CONFIG["fit_txt_name"]
        if not fit_path.exists():
            continue

        raw_path = folder / CONFIG["raw_txt_name"]
        fit_data = np.loadtxt(fit_path, comments="#")
        raw_data = np.loadtxt(raw_path, comments="#") if raw_path.exists() else None
        vds_group, panel_title, sort_key = parse_folder_metadata(folder.name)
        cases.append(
            CaseData(
                folder_name=folder.name,
                folder_path=folder,
                vds_group=vds_group,
                panel_title=panel_title,
                sort_key=sort_key,
                fit_data=fit_data,
                raw_data=raw_data,
            )
        )
    return sorted(cases, key=lambda item: (item.vds_group, item.sort_key, item.folder_name.lower()))


def choose_section_ncols(num_panels: int) -> int:
    if num_panels >= 10:
        return 5
    if num_panels >= 8:
        return 4
    if num_panels >= 6:
        return 3
    if num_panels == 4:
        return 4
    if num_panels == 3:
        return 3
    return max(1, min(2, num_panels))


def compute_global_ylim(cases: Iterable[CaseData]) -> tuple[float, float]:
    maxima = [float(np.max(case.fit_data[:, 1:4])) for case in cases]
    ymax = max(maxima)
    padded = ymax * (1.0 + CONFIG["y_padding_ratio"])
    ytop = math.ceil(padded / CONFIG["y_tick_step"]) * CONFIG["y_tick_step"]
    return CONFIG["y_floor"], ytop


def compute_group_ylim(vds_group: int, cases: Iterable[CaseData]) -> tuple[float, float]:
    if vds_group == 800:
        return CONFIG["y_floor"], 450.0
    return compute_global_ylim(cases)


def build_xticks(xmin: float, xmax: float) -> tuple[np.ndarray, list[str]]:
    ticks = np.linspace(xmin, xmax, CONFIG["xticks_per_panel"])
    decimals = CONFIG["xtick_label_decimals"]
    labels = [f"{tick:.{decimals}f}" for tick in ticks]
    return ticks, labels


def build_yticks(ymin: float, ymax: float) -> np.ndarray:
    step = CONFIG["ytick_step_major"]
    ticks = np.arange(ymin, ymax + step * 0.5, step)
    return ticks


def export_svg_to_emf(svg_path: Path, emf_path: Path) -> None:
    app = None
    pres = None
    try:
        app = win32.Dispatch("PowerPoint.Application")
        app.Visible = 1
        pres = app.Presentations.Add()
        slide = pres.Slides.Add(1, 12)
        slide.Shapes.AddPicture(str(svg_path), 0, -1, 0, 0, -1, -1)
        slide.Export(str(emf_path), "EMF")
    finally:
        if pres is not None:
            pres.Close()
        if app is not None:
            app.Quit()


def save_figure_outputs(fig: plt.Figure, stem_path: Path) -> None:
    png_path = stem_path.with_suffix(".png")
    pdf_path = stem_path.with_suffix(".pdf")
    svg_path = stem_path.with_suffix(".svg")
    emf_path = stem_path.with_suffix(".emf")

    fig.savefig(png_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight", pad_inches=0.02)
    try:
        fig.savefig(pdf_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight", pad_inches=0.02)
    except PermissionError:
        print(f"Skipped locked PDF: {pdf_path}")
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.02)
    try:
        export_svg_to_emf(svg_path, emf_path)
    except Exception as exc:
        print(f"Skipped EMF export for {svg_path.name}: {exc}")


def set_bold_ticks(ax: plt.Axes) -> None:
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight(CONFIG["font_weight"])
        tick_label.set_fontsize(CONFIG["tick_fontsize"])
        tick_label.set_fontfamily(CONFIG["font_family"])


def save_individual_panel(case: CaseData, output_dir: Path, y_limits: tuple[float, float]) -> None:
    panel_dir = output_dir / "individual_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.2, 2.25))
    ax.plot(case.time_us, case.ids_true, color=CONFIG["color_true"], lw=CONFIG["line_width_true"])
    ax.plot(case.time_us, case.ids_pred, color=CONFIG["color_pred"], lw=CONFIG["line_width_pred"])
    ax.plot(case.time_us, case.ids_physics, color=CONFIG["color_physics"], lw=CONFIG["line_width_physics"])
    ax.set_ylim(*y_limits)
    ax.set_xlabel(CONFIG["time_unit_label"], fontsize=10, fontweight=CONFIG["font_weight"])
    ax.set_ylabel(CONFIG["current_label"], fontsize=10, fontweight=CONFIG["font_weight"])
    ax.set_title(case.panel_title, fontsize=11, fontweight=CONFIG["font_weight"])
    ax.grid(True, color=CONFIG["color_grid"], alpha=CONFIG["grid_alpha"], linewidth=0.6)
    set_bold_ticks(ax)
    fig.savefig(panel_dir / f"{case.folder_name}.png", dpi=CONFIG["figure_dpi"], bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def create_vds_prediction_figure(vds_group: int, group_cases: list[CaseData], output_dir: Path) -> Path:
    ncols = choose_section_ncols(len(group_cases))
    nrows = math.ceil(len(group_cases) / ncols)
    fig_width = ncols * CONFIG["panel_width"] + 0.55
    fig_height = nrows * CONFIG["panel_height"] + 1.05

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        top=0.84,
        bottom=0.16,
        left=0.075,
        right=0.99,
        wspace=CONFIG["panel_wspace"],
        hspace=CONFIG["panel_hspace"],
    )

    y_limits = compute_group_ylim(vds_group, group_cases)
    y_ticks = build_yticks(*y_limits)
    flat_axes = list(axes.flat)
    legend_handles = [
        Line2D([], [], color=CONFIG["color_true"], lw=CONFIG["line_width_true"], label="Measured"),
        Line2D([], [], color=CONFIG["color_pred"], lw=CONFIG["line_width_pred"], label="NN Pred"),
        Line2D([], [], color=CONFIG["color_physics"], lw=CONFIG["line_width_physics"], label="Physics"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.945),
        ncol=3,
        frameon=False,
        fontsize=CONFIG["legend_fontsize"],
        handlelength=2.0,
        columnspacing=1.2,
    )

    for idx, case in enumerate(group_cases):
        ax = flat_axes[idx]
        row_idx = idx // ncols
        col_idx = idx % ncols

        ax.plot(case.time_us, case.ids_true, color=CONFIG["color_true"], lw=CONFIG["line_width_true"], solid_capstyle="round")
        ax.plot(case.time_us, case.ids_pred, color=CONFIG["color_pred"], lw=CONFIG["line_width_pred"], solid_capstyle="round")
        ax.plot(case.time_us, case.ids_physics, color=CONFIG["color_physics"], lw=CONFIG["line_width_physics"], solid_capstyle="round")

        x_min = 0.0
        x_max = float(case.time_us.max())
        ax.set_ylim(*y_limits)
        ax.set_xlim(x_min, x_max)
        x_ticks, x_labels = build_xticks(x_min, x_max)
        ax.set_xticks(x_ticks, x_labels)
        ax.set_yticks(y_ticks)
        ax.grid(True, axis="y", color=CONFIG["color_grid"], alpha=0.34, linewidth=0.7)
        ax.grid(True, axis="x", color=CONFIG["color_grid"], alpha=0.22, linewidth=0.55)
        ax.tick_params(width=1.0, length=4.0, pad=2.6, direction="in")
        set_bold_ticks(ax)

        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(CONFIG["color_spine"])
            ax.spines[side].set_linewidth(CONFIG["spine_linewidth"])

        if col_idx != 0:
            ax.tick_params(labelleft=False)

        ax.text(
            0.96,
            0.93,
            case.panel_title,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=CONFIG["panel_title_fontsize"],
            fontweight=CONFIG["font_weight"],
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": CONFIG["title_box_facecolor"],
                "edgecolor": "#d0d0d0",
                "linewidth": 0.5,
                "alpha": CONFIG["title_box_alpha"],
            },
        )

        if CONFIG["save_individual_prediction_panels"]:
            save_individual_panel(case, output_dir, y_limits)

    for ax in flat_axes[len(group_cases) :]:
        ax.axis("off")

    stem = output_dir / f"transient_predictions_{vds_group}V_publication"
    save_figure_outputs(fig, stem)
    plt.close(fig)
    return stem.with_suffix(".png")


def create_all_vds_prediction_figures(cases: list[CaseData], output_dir: Path) -> list[Path]:
    groups: dict[int, list[CaseData]] = {}
    for case in cases:
        groups.setdefault(case.vds_group, []).append(case)

    output_paths: list[Path] = []
    for vds_group, group_cases in sorted(groups.items(), key=lambda item: item[0]):
        output_paths.append(create_vds_prediction_figure(vds_group, group_cases, output_dir))
    return output_paths


def branch_metrics(y_true: np.ndarray, y_model: np.ndarray) -> dict[str, float]:
    peak_scale = max(float(np.max(np.abs(y_true))), 1e-12)
    mae = float(np.mean(np.abs(y_model - y_true)))
    rmse = float(np.sqrt(np.mean((y_model - y_true) ** 2)))
    return {
        "mae_A": mae,
        "rmse_A": rmse,
        "nmae_pct": mae / peak_scale * 100.0,
        "nrmse_pct": rmse / peak_scale * 100.0,
    }


def compute_branch_error_table(cases: list[CaseData]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case in cases:
        model_series = {
            "NN branch": case.ids_pred,
            "Physics branch": case.ids_physics,
        }
        for model_name, series in model_series.items():
            rows.append(
                {
                    "case_name": case.folder_name,
                    "vds_group_V": case.vds_group,
                    "panel_title": case.panel_title,
                    "model": model_name,
                    "branch": model_name,
                    "n_points": len(case.ids_true),
                    "peak_ids_A": float(np.max(case.ids_true)),
                    **branch_metrics(case.ids_true, series),
                }
            )
    return pd.DataFrame(rows)


def create_error_boxplot(error_df: pd.DataFrame, output_dir: Path) -> Path:
    rng = np.random.default_rng(CONFIG["jitter_seed"])
    fig, ax = plt.subplots(figsize=(5.8, 4.0))

    vds_order = sorted(error_df["vds_group_V"].unique())
    model_order = ["NN branch", "Physics branch"]
    offsets = {
        "NN branch": -CONFIG["box_width"] * 0.62,
        "Physics branch": CONFIG["box_width"] * 0.62,
    }
    colors = {
        "NN branch": CONFIG["color_pred"],
        "Physics branch": CONFIG["color_physics"],
    }
    base_positions = np.arange(1, len(vds_order) + 1, dtype=float)

    for group_idx, vds_value in enumerate(vds_order):
        base_x = base_positions[group_idx]
        for model in model_order:
            values = (
                error_df.loc[
                    (error_df["vds_group_V"] == vds_value) & (error_df["model"] == model),
                    "nmae_pct",
                ]
                .dropna()
                .to_numpy()
            )
            if values.size == 0:
                continue

            position = base_x + offsets[model]
            boxplot = ax.boxplot(
                values,
                positions=[position],
                widths=CONFIG["box_width"],
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#111111", "linewidth": 1.45},
                whiskerprops={"color": colors[model], "linewidth": 1.0},
                capprops={"color": colors[model], "linewidth": 1.0},
                boxprops={"edgecolor": colors[model], "linewidth": 1.25},
            )
            for patch in boxplot["boxes"]:
                patch.set_facecolor(to_rgba(colors[model], 0.16))

            jitter = rng.normal(loc=0.0, scale=CONFIG["box_width"] * 0.10, size=values.size)
            ax.scatter(
                np.full(values.size, position) + jitter,
                values,
                s=CONFIG["box_scatter_size"],
                color=colors[model],
                alpha=CONFIG["box_scatter_alpha"],
                linewidths=0,
                zorder=3,
            )

    ax.set_xlim(0.45, len(vds_order) + 0.55)
    ax.set_xticks(base_positions, [f"{vds_value} V" for vds_value in vds_order])
    ax.set_ylim(*CONFIG["boxplot_ylim"])
    ax.grid(True, axis="y", color=CONFIG["color_grid"], alpha=0.28, linewidth=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CONFIG["color_spine"])
    ax.spines["bottom"].set_color(CONFIG["color_spine"])
    ax.tick_params(width=0.8, length=3.2, direction="in")
    set_bold_ticks(ax)

    legend_handles = [
        Line2D([], [], color=CONFIG["color_pred"], lw=2.0, label="NN branch"),
        Line2D([], [], color=CONFIG["color_physics"], lw=2.0, label="Physics branch"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper right",
        fontsize=9.8,
        ncol=2,
        bbox_to_anchor=(0.995, 1.02),
    )

    stem = output_dir / "all_transient_branch_error_boxplot"
    save_figure_outputs(fig, stem)
    plt.close(fig)
    return stem.with_suffix(".png")


def write_error_summary(error_df: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "branch_error_summary.csv"
    error_df.sort_values(["vds_group_V", "case_name", "model"]).to_csv(output_path, index=False)
    return output_path


def write_case_summary(cases: list[CaseData], output_dir: Path) -> Path:
    rows = []
    for case in cases:
        peak_idx = int(np.argmax(case.ids_true))
        rows.append(
            {
                "case_name": case.folder_name,
                "vds_group_V": case.vds_group,
                "panel_title": case.panel_title,
                "num_points": len(case.ids_true),
                "peak_time_us": case.time_us[peak_idx],
                "peak_ids_A": case.ids_true[peak_idx],
            }
        )

    output_path = output_dir / "case_summary.csv"
    pd.DataFrame(rows).sort_values(["vds_group_V", "case_name"]).to_csv(output_path, index=False)
    return output_path


def main() -> None:
    configure_matplotlib()
    root_dir = Path(CONFIG["root_dir"]).resolve()
    output_dir = root_dir / CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(root_dir)
    if not cases:
        raise RuntimeError(f"No valid result folders with {CONFIG['fit_txt_name']} were found under {root_dir}")

    prediction_paths = create_all_vds_prediction_figures(cases, output_dir)
    error_df = compute_branch_error_table(cases)
    boxplot_path = create_error_boxplot(error_df, output_dir)
    error_csv_path = write_error_summary(error_df, output_dir)
    case_csv_path = write_case_summary(cases, output_dir)

    for prediction_path in prediction_paths:
        print(f"Saved prediction figure: {prediction_path}")
    print(f"Saved branch error boxplot: {boxplot_path}")
    print(f"Saved error summary: {error_csv_path}")
    print(f"Saved case summary: {case_csv_path}")


if __name__ == "__main__":
    main()
