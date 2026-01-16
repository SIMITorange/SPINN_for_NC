# SPINN for NC / 短路PINN工程

## English Guide

### Overview
This project builds a Physics-Informed Neural Network (PINN) for short-circuit behavior. It fuses an electrical physics model with temperature feedback (Chebyshev heat conduction or a simplified thermal model) and exports TXT files for paper-quality plots.

Core pipeline:
- Data: clean/decimate CSV → aggregate into grouped HDF5.
- Training: inputs (time, Vds, Vgs) → target Ids, plus physics loss with temperature feedback.
- Inference & plotting: load checkpoints/TXT and generate figures.

### Quick Start
1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Data preprocessing (CSV → HDF5)
```bash
python data_process_v0.py
```
Select a folder containing CSV files; it generates combined_training_data.h5.

3) Train (with Chebyshev thermal solver)
```bash
python PINN_short_circuit_no_temperature.py
```

4) Train (no Chebyshev, simplified thermal model)
```bash
python scripts/pinn_no_chebyshev_h5.py
```

5) Inference from checkpoint
```bash
python scripts/infer_from_checkpoint.py --checkpoint checkpoints/xxx.pt
```
Batch mode:
```bash
python scripts/infer_from_checkpoint.py --all --checkpoint-dir checkpoints
```

6) Plot paper-style figures
```bash
python scripts/plot_paper_figures.py --input results/<group_name>
```
Batch mode:
```bash
python scripts/plot_paper_figures.py --all --root results
```

### Script Guide
- PINN_short_circuit_no_temperature.py: main training (HDF5 groups + Chebyshev thermal solver).
- data_process_v0.py: CSV cleaning/decimation and HDF5 aggregation.
- chebeshev.py: standalone Chebyshev heat-conduction demo.
- scripts/pinn_no_chebyshev_h5.py: HDF5 training with simplified thermal model.
- scripts/pinn_no_chebyshev.py: Excel-based legacy/single-file training flow.
- scripts/infer_from_checkpoint.py: inference for Chebyshev checkpoints.
- scripts/infer_no_chebyshev.py: inference for no-Chebyshev checkpoints.
- scripts/plot_paper_figures.py: reads TXT outputs to generate figures.

### Folder Guide
- checkpoints/: checkpoints from Chebyshev training.
- checkpoints_no_cheb/: checkpoints from no-Chebyshev training.
- results/: training outputs (TXT/loss curves) with Chebyshev.
- results_no_cheb/: outputs from no-Chebyshev runs.
- runs/: TensorBoard logs (Chebyshev).
- runs_no_cheb/: TensorBoard logs (no Chebyshev).
- scripts/: inference and plotting utilities.

## 中文说明

### 项目简介
本工程用于短路工况下的物理约束神经网络（PINN）建模，结合电学物理模型与温度反馈，训练得到参数化模型，并输出可用于论文绘图的中间结果。

核心逻辑概览：
- 数据管线：CSV 数据清洗与降采样 → 合并为 HDF5 分组数据。
- 训练：以时间、Vds、Vgs 为输入，Ids 为目标；同时引入物理模型与温度反馈（含 Chebyshev 热传导或简化温度模型）。
- 推理与绘图：读取训练生成的 checkpoint 和 txt 文件，生成可视化图像。

### 快速使用
1) 安装依赖
```bash
pip install -r requirements.txt
```

2) 数据预处理（CSV → HDF5）
```bash
python data_process_v0.py
```
按界面选择含 CSV 的目录，生成 combined_training_data.h5。

3) 训练（含 Chebyshev 热传导）
```bash
python PINN_short_circuit_no_temperature.py
```

4) 训练（无 Chebyshev，简化温度模型）
```bash
python scripts/pinn_no_chebyshev_h5.py
```

5) 推理（从 checkpoint 输出 txt）
```bash
python scripts/infer_from_checkpoint.py --checkpoint checkpoints/xxx.pt
```
或批量：
```bash
python scripts/infer_from_checkpoint.py --all --checkpoint-dir checkpoints
```

6) 绘图（论文级图）
```bash
python scripts/plot_paper_figures.py --input results/<group_name>
```
或批量：
```bash
python scripts/plot_paper_figures.py --all --root results
```

### 脚本说明
- PINN_short_circuit_no_temperature.py：主训练脚本（HDF5 分组训练 + Chebyshev 热传导）。
- data_process_v0.py：CSV 清洗、降采样并合并为 HDF5。
- chebeshev.py：Chebyshev 伪谱热传导求解器示例（独立演示）。
- scripts/pinn_no_chebyshev_h5.py：HDF5 分组训练，但温度使用简化模型。
- scripts/pinn_no_chebyshev.py：从 Excel 训练的旧版/单文件流程示例。
- scripts/infer_from_checkpoint.py：从含 Chebyshev 的 checkpoint 推理并落盘 txt。
- scripts/infer_no_chebyshev.py：从无 Chebyshev 的 checkpoint 推理并落盘 txt。
- scripts/plot_paper_figures.py：读取 txt 生成论文风格图像。

### 目录说明
- checkpoints/：含 Chebyshev 训练保存的模型权重。
- checkpoints_no_cheb/：无 Chebyshev 训练保存的模型权重。
- results/：含 Chebyshev 的训练输出（txt 数据、loss 曲线）。
- results_no_cheb/：无 Chebyshev 的训练输出。
- runs/：TensorBoard 日志（含 Chebyshev）。
- runs_no_cheb/：TensorBoard 日志（无 Chebyshev）。
- scripts/：推理与绘图等辅助脚本。

---

