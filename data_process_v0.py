import csv
import os
import tkinter as tk
from tkinter import filedialog
from typing import Dict, Iterable, List, Tuple

import pandas as pd

try:
    import h5py  # type: ignore
except ImportError:  # pragma: no cover
    h5py = None

SAMPLE_STEP = 30
REQUIRED_COLUMNS = 4


def is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def find_data_start(filepath: str) -> int:
    """Locate the first row that can be parsed as numeric data."""
    with open(filepath, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        skiprows = 0
        for row in reader:
            if len(row) >= 4:
                try:
                    [float(x) for x in row[:4]]
                    return skiprows
                except (ValueError, TypeError):
                    skiprows += 1
            else:
                skiprows += 1
        return 0


def load_and_clean_csv(filepath: str) -> pd.DataFrame:
    """Read a CSV similar to the original logic and return cleaned numeric data."""
    skip_rows = find_data_start(filepath)

    df = pd.read_csv(
        filepath,
        skiprows=skip_rows,
        header=None,
        usecols=[0, 1, 2, 3],
        engine="python",
        on_bad_lines="skip",
        encoding="utf-8-sig",
    )

    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        raise ValueError("No usable data after cleaning.")

    df = df[df[3] > 10]
    df = df.iloc[::SAMPLE_STEP].reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable data after cleaning.")

    df.columns = ["time(s)", "Vds", "Vgs", "Ids"]
    return df[["time(s)", "Ids", "Vds", "Vgs"]]


def next_available_path(base_dir: str, base_name: str) -> str:
    output_path = os.path.join(base_dir, base_name)
    counter = 1
    while os.path.exists(output_path):
        stem, ext = os.path.splitext(base_name)
        output_path = os.path.join(base_dir, f"{stem}_{counter}{ext}")
        counter += 1
    return output_path


def process_single_csv(filepath: str) -> Tuple[pd.DataFrame, Dict[str, float], str]:
    df = load_and_clean_csv(filepath)

    vds_max = float(df["Vds"].max())
    vgs_max = float(df["Vgs"].max())
    time_max = float(df["time(s)"].max())

    output_dir = os.path.join(os.path.dirname(filepath), "output_files")
    os.makedirs(output_dir, exist_ok=True)

    rounded_vds = int(vds_max) // 100 * 100
    base_name = f"Ids-data_Vds={rounded_vds}_Vgs={int(vgs_max)}.csv"
    output_path = next_available_path(output_dir, base_name)

    df.to_csv(output_path, index=False)

    labels = {"vds_max": vds_max, "vgs_max": vgs_max, "time_max": time_max}
    return df, labels, output_path


def iter_csv_files(root_folder: str) -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(root_folder):
        dirnames[:] = [d for d in dirnames if d != "output_files"]
        for name in filenames:
            if name.lower().endswith(".csv"):
                yield os.path.join(dirpath, name)


def safe_group_name(rel_path: str) -> str:
    name = rel_path.replace(os.sep, "__")
    return name.replace(" ", "_").replace(".", "_")


def unique_group_name(h5file, base_name: str) -> str:
    name = base_name
    counter = 1
    while name in h5file:
        name = f"{base_name}_{counter}"
        counter += 1
    return name


def save_hdf5(root_folder: str, items: List[Tuple[str, pd.DataFrame, Dict[str, float]]]) -> None:
    if not items:
        return

    if h5py is None:
        print("Skipping HDF5 export because h5py is not installed.")
        return

    hdf5_path = os.path.join(root_folder, "combined_training_data.h5")
    with h5py.File(hdf5_path, "w") as h5f:
        for rel_path, df, labels in items:
            base_name = safe_group_name(rel_path)
            group_name = unique_group_name(h5f, base_name)
            grp = h5f.create_group(group_name)
            grp.create_dataset("data", data=df.to_numpy(dtype="float32"), compression="gzip")
            grp.attrs["columns"] = df.columns.tolist()
            grp.attrs["source"] = rel_path
            grp.attrs["vds_max"] = labels["vds_max"]
            grp.attrs["vgs_max"] = labels["vgs_max"]
            grp.attrs["time_max"] = labels["time_max"]

    print(f"Aggregated HDF5 saved to {hdf5_path}")


def process_files() -> None:
    folder = filedialog.askdirectory()
    if not folder:
        return

    aggregated: List[Tuple[str, pd.DataFrame, Dict[str, float]]] = []

    for filepath in iter_csv_files(folder):
        try:
            df, labels, output_path = process_single_csv(filepath)
            rel_path = os.path.relpath(filepath, folder)
            aggregated.append((rel_path, df, labels))
            print(f"Processed {rel_path} -> {output_path}")
        except Exception as exc:  # pragma: no cover
            print(f"Skipping {filepath}: {exc}")

    save_hdf5(folder, aggregated)
    print("Processing complete.")


root = tk.Tk()
root.title("CSV Processor")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

btn_process = tk.Button(
    frame,
    text="Select folder and process CSV files",
    command=process_files,
    padx=10,
    pady=5,
)
btn_process.pack()

root.mainloop()
