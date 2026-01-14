import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# 1. 数据准备（修改输入列为Time）
# file_path = "./processed_data.xlsx"
file_path = "processed_data_only_400_600_new.xlsx"
excel_file = pd.ExcelFile(file_path)
data_frames = []
for sheet in excel_file.sheet_names:
    df = pd.read_excel(excel_file, sheet_name=sheet)
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={
        'time(s)': 'Time',
        'vds': 'Vds',
        'vgs': 'Vgs',
        'ids': 'Ids'
    })[['Ids', 'Time', 'Vds', 'Vgs']]  # 调整列顺序
    data_frames.append(df)

combined_df = pd.concat(data_frames, ignore_index=True)
# 按时间排序数据
#  combined_df.sort_values(by='Time', inplace=True)
raw_data = combined_df.to_numpy()

print(f"Raw data: {raw_data}")

# 输入列为Time, Vds, Vgs
inputs_data = raw_data[:, 1:].copy()  # [Time, Vds, Vgs]
targets_data = raw_data[:, 0].reshape(-1, 1)

# 数据标准化
input_scaler = StandardScaler()
target_scaler = StandardScaler()
X_scaled = input_scaler.fit_transform(inputs_data)
y_scaled = target_scaler.fit_transform(targets_data)

# 保存标准化参数
input_mean = torch.tensor(input_scaler.mean_, dtype=torch.float32)
input_std = torch.tensor(input_scaler.scale_, dtype=torch.float32)
target_mean = torch.tensor(target_scaler.mean_[0], dtype=torch.float32)
target_std = torch.tensor(target_scaler.scale_[0], dtype=torch.float32)

# 创建数据集（关闭shuffle保证时间序列）
dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32),
                        torch.tensor(y_scaled, dtype=torch.float32))
# dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)  # 全量数据作为单个batch
dataloader = DataLoader(dataset, batch_size=300, shuffle=False)
# 2. 定义改进的神经网络模型（新增Epi_thickness和T_initial参数）
class PINN(nn.Module):
    def __init__(self, dropout_rate=0):
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
            nn.Linear(128, 1)
        )
        # 可学习参数
        self.param1 = nn.Parameter(torch.tensor(110.0, dtype=torch.float32))
        self.param2 = nn.Parameter(torch.tensor(4, dtype=torch.float32))
        self.param3 = nn.Parameter(torch.tensor(10.6, dtype=torch.float32))
        self.param4 = nn.Parameter(torch.tensor(0.0011, dtype=torch.float32))
        self.param5 = nn.Parameter(torch.tensor(-0.102, dtype=torch.float32))
        self.param6 = nn.Parameter(torch.tensor(10.0, dtype=torch.float32))
        self.param7 = nn.Parameter(torch.tensor(-0.1, dtype=torch.float32))
        self.param8 = nn.Parameter(torch.tensor(5.0, dtype=torch.float32))
        self.param9 = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))
        # 新增参数
        self.epi_thickness = nn.Parameter(torch.tensor(1.0e-5, dtype=torch.float32))
        self.T_initial = nn.Parameter(torch.tensor(350.0, dtype=torch.float32))

    def forward(self, x):
        return self.net(x)

model = PINN()

# 3. 物理模型定义（保持原结构，T由外部传入）
def physics_model(Vds, Vgs, T, param1, param2, param3, param4, param5, param6, param7, param8, param9):
    # 转换为张量（如果输入是标量或numpy数组）
    if not isinstance(Vds, torch.Tensor):
        Vds = torch.tensor(Vds, dtype=torch.float32)
    if not isinstance(Vgs, torch.Tensor):
        Vgs = torch.tensor(Vgs, dtype=torch.float32)
    if not isinstance(T, torch.Tensor):
        T = torch.tensor(T, dtype=torch.float32)

    # 物理公式计算
    NET5_value = param3 * 1 / (param1*(T/300)**-0.01 + param6*(T/300)**param2)
    NET3_value = -0.004263*T + 3.422579
    p9 = param5
    NET2_value = -0.005 * Vgs + 0.165
    NET1_value = -0.1717 * Vgs + 3.5755
    term3 = (torch.log(1 + torch.exp(Vgs - NET3_value)))**2 - (torch.log(1 + torch.exp(Vgs - NET3_value - (NET2_value * Vds * ((1 + torch.exp(p9 * Vds))**NET1_value)))))**2
    term1 = NET5_value * (Vgs - NET3_value)
    term2 = 1 + 0.0005 * Vds
    return term2 * term1 * term3

# 4. 训练配置
optimizer = optim.AdamW([
    {'params': model.net.parameters(), 'lr': 0.001},
    {'params': [model.param1, model.param2, model.param3, model.param4, model.param5,
                model.param6, model.param7, model.param8, model.param9,
                model.epi_thickness, model.T_initial], 'lr': 0.1}
], weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
mse_loss = nn.MSELoss()

# 5. 训练循环（全量数据处理）
num_epochs = 150
alpha = 0.9
w_data, w_physics = 0.5, 0.5
losses = []
data_metrics = []
physics_metrics = []

def integrate_temperature(time_raw, vds_raw, vgs_raw, model, physics_model):
    """
    温度积分计算函数（支持梯度传播）
    返回：温度序列T，物理预测序列physics_preds
    """
    T = torch.full_like(time_raw, model.T_initial.item())

    physics_preds = torch.zeros_like(vds_raw)
    delta_time = torch.zeros_like(time_raw)
    delta_time[1:] = time_raw[1:] - time_raw[:-1]

    # 确保epi_thickness为正
    epi_thickness = torch.nn.functional.softplus(model.epi_thickness)

    for i in range(len(time_raw)):
        if i > 0:
            with torch.enable_grad():
                # 打印前一步温度状态
                # print(f"\nStep {i}: Prev T={T[i-1].item():.2f}K, Vds={vds_raw[i].item():.2f}V, Vgs={vgs_raw[i].item():.2f}V")

                ids_prev = physics_model(vds_raw[i], vgs_raw[i], T[i-1].detach(),
                                      model.param1, model.param2, model.param3,
                                      model.param4, model.param5, model.param6,
                                      model.param7, model.param8, model.param9)

                # # 打印中间计算结果
                # print(f"  ids_prev raw: {ids_prev.item():.3e} A")
                # ids_prev = torch.clamp(ids_prev, min=1e-12, max=1e3)
                # print(f"  ids_prev clamped: {ids_prev.item():.3e} A")

            delta_T = 2 * vds_raw[i] / 1e-5  * (ids_prev / 20e-6) * delta_time[i] /(3200*600)
            current_T = T[i-1] + delta_T

            # # 打印温度变化
            # print(f"  delta_T: {delta_T.item():.3e} K")
            # print(f"  current_T: {current_T.item():.2f} K")
        else:
            current_T = model.T_initial
            # print(f"\nStep 0: Init T={current_T.item():.2f} K")

        T[i] = current_T
        physics_preds[i] = physics_model(vds_raw[i], vgs_raw[i], T[i],
                                      model.param1, model.param2, model.param3,
                                      model.param4, model.param5, model.param6,
                                      model.param7, model.param8, model.param9)

        # 检查NaN
        if torch.isnan(physics_preds[i]):
            print(f"!!! NaN detected at step {i} !!!")
            print(f"T[i] = {T[i].item()}, Vds={vds_raw[i].item()}, Vgs={vgs_raw[i].item()}")
            break

    return T, physics_preds


for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in dataloader:  # 单次迭代（全量数据）
        # 数据前向传播
        pred = model(X_batch)
        loss_data = mse_loss(pred, y_batch)

        # 在训练循环中使用：
        # 计算物理损失
        time_raw = X_batch[:, 0] * input_std[0] + input_mean[0]
        vds_raw = X_batch[:, 1] * input_std[1] + input_mean[1]
        vgs_raw = X_batch[:, 2] * input_std[2] + input_mean[2]

        # 执行温度积分
        T, physics_preds = integrate_temperature(time_raw, vds_raw, vgs_raw, model, physics_model)

        # 标准化处理
        physics_pred_scaled = (physics_preds - target_mean) / target_std

        # 计算损失（增加数值稳定性）
        loss_physics = mse_loss(pred.squeeze(), physics_pred_scaled)

        # 动态权重调整
        current_data = loss_data.item()
        current_physics = loss_physics.item()
        total = current_data + current_physics + 1e-8
        new_wd = current_data / total
        new_wp = current_physics / total
        w_data = alpha * w_data + (1 - alpha) * new_wd
        w_physics = alpha * w_physics + (1 - alpha) * new_wp

        # 总损失
        total_loss = w_data * loss_data + w_physics * loss_physics

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

    # 更新学习率和记录指标
    scheduler.step(total_loss.item())
    losses.append(total_loss.item())
    data_metrics.append(loss_data.item())
    physics_metrics.append(loss_physics.item())

    # 打印训练信息
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1:04d} | Loss: {total_loss.item():.3e} | "
              f"Data: {loss_data.item():.2e} | Physics: {loss_physics.item():.2e} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"params: {model.param1.item():.3f}, {model.param2.item():.3f}, "
              f"{model.param3.item():.3f}, {model.param4.item():.3f}, {model.param5.item():.3f}, {model.param6.item():.3f}, {model.param7.item():.3f}")

# 6. 模型预测与可视化
def neural(time, Vds, Vgs):
    input_arr = np.array([[time, Vds, Vgs]])
    scaled_input = input_scaler.transform(input_arr)
    with torch.no_grad():
        scaled_output = model(torch.tensor(scaled_input, dtype=torch.float32))
    return target_scaler.inverse_transform(scaled_output.numpy())[0][0]

def plot_results(true_values, predicted_values, physic_values):
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label="True Values", marker='o', markersize=3)
    plt.plot(predicted_values, label="Predicted Values", marker='x', markersize=3)
    plt.plot(physic_values, label="Physics Values", marker='.', markersize=3)
    plt.xlabel("Sample Index")
    plt.ylabel("Ids")
    plt.title("Comparison of True and Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.show()

true_values = targets_data.flatten()[:900]
predicted_values = [neural(Vds, Vgs, T) for Vds, Vgs, T in inputs_data[:900]]
physic_values = [physics_model(Vds, Vgs, T, model.param1, model.param2, model.param3, model.param4, model.param5, model.param6, model.param7, model.param8, model.param9)
                 for Vds, Vgs, T in inputs_data[:900]]
physic_values = [tensor.detach().numpy() for tensor in physic_values]

plot_results(true_values, predicted_values, physic_values)

# 绘制损失曲线
plt.figure()
plt.plot(np.log10(losses), label='Total Loss')
plt.plot(np.log10(data_metrics), label='Data Loss')
plt.plot(np.log10(physics_metrics), label='Physics Loss')
plt.xlabel('Epoch'), plt.ylabel('Log Loss'), plt.legend(), plt.grid()
plt.show()

# 参数输出
print(f"关键参数: Epi_thickness={model.epi_thickness.item():.4f} μm, T_initial={model.T_initial.item():.1f} K")
print(f"其他参数: param1={model.param1.item():.3f}, param2={model.param2.item():.3f}")

# 在模型预测与可视化部分添加以下代码（放在plot_results之后）

# 7. 温度变化可视化
def plot_temperature(time, temperature):
    plt.figure(figsize=(10, 6))
    plt.plot(time, temperature, label="Temperature", color='r', linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.title("Device Temperature Evolution")
    plt.legend()
    plt.grid(True)
    plt.show()

# 获取完整温度序列（使用全量数据）
with torch.no_grad():
    # 准备完整数据
    full_inputs = torch.tensor(X_scaled, dtype=torch.float32)
    full_time = full_inputs[:, 0] * input_std[0] + input_mean[0]
    full_vds = full_inputs[:, 1] * input_std[1] + input_mean[1]
    full_vgs = full_inputs[:, 2] * input_std[2] + input_mean[2]

    # 计算温度序列
    T_sequence, _ = integrate_temperature(full_time, full_vds, full_vgs, model, physics_model)

    # 绘制温度变化
    plot_temperature(full_time.numpy(), T_sequence.numpy())

# # 7. 温度变化可视化
# def plot_temperature(time, temperature):
# plt.figure(figsize=(10, 6))
# plt.plot(time, temperature, label="Temperature", color='r', linewidth=2)
# plt.xlabel("Time (s)")
# plt.ylabel("Temperature")
# plt.title("Device Temperature Evolution")
# plt.legend()
# plt.grid(True)
# plt.show()

# # 获取完整温度序列（使用全量数据）
# with torch.no_grad():
# # 准备完整数据
# full_inputs = torch.tensor(X_scaled, dtype=torch.float32)
# full_time = full_inputs[:, 0] * input_std[0] + input_mean[0]
# full_vds = full_inputs[:, 1] * input_std[1] + input_mean[1]
# full_vgs = full_inputs[:, 2] * input_std[2] + input_mean[2]

# # 计算温度序列
# T_sequence, _ = integrate_temperature(full_time, full_vds, full_vgs, model, physics_model)

# # 绘制温度变化
# plot_temperature(full_time.numpy(), T_sequence.numpy())

# # 对比结果可视化（使用前300个样本）
# def plot_comparison(time, true, pred, physics):
# plt.figure(figsize=(10, 6))
# plt.plot(time, true, label="True Values", marker='o', markersize=3)
# plt.plot(time, pred, label="Predicted Values", marker='x', markersize=3)
# plt.plot(time, physics, label="Physics Values", marker='.', markersize=3)
# plt.xlabel("Time (s)")
# plt.ylabel("Drain Current (A)")
# plt.title("Current Comparison Over Time")
# plt.legend()
# plt.grid(True)
# plt.show()

# # 主可视化流程
# # 修复AttributeError的核心修改部分
# with torch.no_grad():
# # 2. 准备对比数据（前300个样本）
# # 正确获取时间序列（全numpy运算）
# input_std_np = input_std.numpy()
# input_mean_np = input_mean.numpy()
# partial_time = X_scaled[:, 0] * input_std_np[0] + input_mean_np[0]

# # 修复神经网络预测值的转换
# predicted_values = [neural(Vds, Vgs, T) for Vds, Vgs, T in inputs_data[:]] # 原始输出类型

# # 正确转换物理模型输出
# # 修复部分：批量计算物理模型的预测值 确保physic_values为numpy类型
# physic_values = physics_model(full_vds, full_vgs, T_sequence,
# model.param1, model.param2, model.param3,
# model.param4, model.param5, model.param6,
# model.param7, model.param8, model.param9).detach().numpy().flatten()

# # 确保所有数据为numpy类型
# true_values = targets_data.flatten()[:]
# predicted_values = np.array(predicted_values)
# plot_comparison(partial_time, true_values, predicted_values, physic_values)

# # 其他可视化保持不变
# # 损失曲线
# plt.figure()
# plt.plot(np.log10(losses), label='Total Loss')
# plt.plot(np.log10(data_metrics), label='Data Loss')
# plt.plot(np.log10(physics_metrics), label='Physics Loss')
# plt.xlabel('Epoch'), plt.ylabel('Log Loss'), plt.legend(), plt.grid()
# plt.show()

# # 参数输出
# print(f"关键参数:\n"
# f"T_initial={model.T_initial.item():.1f} K\n"
# f"param1={model.param1.item():.3f}\n"
# f"param2={model.param2.item():.3f}\n"
# f"param3={model.param3.item():.3f}\n"
# f"param4={model.param4.item():.5f}\n"
# f"param5={model.param5.item():.3f}\n"
# f"param6={model.param6.item():.3f}\n"
# f"param7={model.param7.item():.3f}"
# )
