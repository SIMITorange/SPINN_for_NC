from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _save_heatmap(field: np.ndarray, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=140)
    im = ax.imshow(field, origin="lower", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Macro X")
    ax.set_ylabel("Macro Y")
    fig.colorbar(im, ax=ax, label="Temperature (K)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_series(time_s: np.ndarray, series: np.ndarray, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=140)
    ax.plot(time_s, series)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (K)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot layered electro-thermal outputs")
    parser.add_argument("--input", type=str, default="layered_electrothermal_output.npz", help="NPZ file path")
    parser.add_argument("--out-dir", type=str, default="layered_electrothermal_plots", help="Output folder")
    parser.add_argument("--time-index", type=int, default=-1, help="Index into time axis (default: last)")
    parser.add_argument("--show", action="store_true", help="Show figures in a GUI window")
    args = parser.parse_args()

    data = np.load(args.input)
    time_s = data["time_s"]
    temps = data["temperatures_k"]
    top = data.get("top_layer_temperature_k", temps[:, 0, :, :])

    idx = args.time_index if args.time_index >= 0 else len(time_s) - 1
    idx = int(np.clip(idx, 0, len(time_s) - 1))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_heatmap(top[idx], f"Top Layer Temperature (t={time_s[idx]:.3e}s)", out_dir / "top_layer.png")

    if "top_layer_cell_temperature_k" in data:
        cell_map = data["top_layer_cell_temperature_k"]
        _save_heatmap(
            cell_map[idx],
            f"Top Layer Cell Temperature (t={time_s[idx]:.3e}s)",
            out_dir / "top_layer_cell.png",
        )

    max_t = temps.reshape(len(time_s), -1).max(axis=1)
    _save_series(time_s, max_t, "Max Temperature (All Layers)", out_dir / "max_temperature.png")

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
