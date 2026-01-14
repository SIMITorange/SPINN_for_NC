import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time

class HeatConductionSolver:
    # [Keeping the __init__ and chebyshev_diff_matrix methods unchanged]
    def __init__(self, a, b, c, Nx, Ny, Nz, k_func, C_func, P_func, T_init, dt, total_time, rho=1.0,
                 h_z0=0.0, h_zc=0.0, T_inf=300.0):
        # [Previous __init__ code remains the same]
        self.a = a
        self.b = b
        self.c = c
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.k_func = k_func
        self.C_func = C_func
        self.P_func = P_func
        self.dt = dt
        self.total_time = total_time
        self.rho = rho
        self.h_z0 = h_z0  # 对 z=0 面的对流换热系数，0 表示绝热
        self.h_zc = h_zc  # 对 z=c 面的对流换热系数，0 表示绝热
        self.T_inf = T_inf  # 环境温度

        self.x = -np.cos(np.pi * np.arange(Nx) / (Nx - 1))
        self.y = -np.cos(np.pi * np.arange(Ny) / (Ny - 1))
        self.z = -np.cos(np.pi * np.arange(Nz) / (Nz - 1))

        self.x_mapped = a * (self.x + 1) / 2
        self.y_mapped = b * (self.y + 1) / 2
        self.z_mapped = c * (self.z + 1) / 2
        # 相邻节点间距，用于近似一阶边界导数
        self.dz0 = self.z_mapped[1] - self.z_mapped[0]
        self.dzc = self.z_mapped[-1] - self.z_mapped[-2]

        self.wx = np.ones(Nx)
        self.wx[0] = self.wx[-1] = 0.5
        self.wx = self.wx * np.pi / (Nx - 1)

        self.wy = np.ones(Ny)
        self.wy[0] = self.wy[-1] = 0.5
        self.wy = self.wy * np.pi / (Ny - 1)

        self.wz = np.ones(Nz)
        self.wz[0] = self.wz[-1] = 0.5
        self.wz = self.wz * np.pi / (Nz - 1)

        if callable(T_init):
            self.T = np.zeros((Nx, Ny, Nz))
            for i in range(Nx):
                for j in range(Ny):
                    for k in range(Nz):
                        self.T[i, j, k] = T_init(self.x_mapped[i], self.y_mapped[j], self.z_mapped[k])
        else:
            self.T = np.full((Nx, Ny, Nz), T_init)

        self.Dx = self.chebyshev_diff_matrix(Nx, 1)
        self.Dy = self.chebyshev_diff_matrix(Ny, 1)
        self.Dz = self.chebyshev_diff_matrix(Nz, 1)

        self.Dxx = self.chebyshev_diff_matrix(Nx, 2)
        self.Dyy = self.chebyshev_diff_matrix(Ny, 2)
        self.Dzz = self.chebyshev_diff_matrix(Nz, 2)

        self.Dx = 2/a * self.Dx
        self.Dy = 2/b * self.Dy
        self.Dz = 2/c * self.Dz

        self.Dxx = (2/a)**2 * self.Dxx
        self.Dyy = (2/b)**2 * self.Dyy
        self.Dzz = (2/c)**2 * self.Dzz

    def chebyshev_diff_matrix(self, N, order=1):
        # [Previous chebyshev_diff_matrix code remains the same]
        if order == 1:
            D = np.zeros((N, N))
            x = np.cos(np.pi * np.arange(N) / (N - 1))
            c = np.ones(N)
            c[0] = c[-1] = 2

            for i in range(N):
                for j in range(N):
                    if i != j:
                        D[i, j] = c[i] / c[j] * (-1)**(i+j) / (x[i] - x[j])

            for i in range(N):
                D[i, i] = -np.sum(D[i, :])

            return D

        elif order == 2:
            D1 = self.chebyshev_diff_matrix(N, 1)
            D2 = np.matmul(D1, D1)
            return D2

        else:
            raise ValueError("只支持一阶和二阶导数")

    def solve(self):
        t = 0
        step = 0
        T_history = []
        t_history = []

        while t < self.total_time:
            if step % 10 == 0:
                T_history.append(self.T.copy())
                t_history.append(t)
                print(f"时间 t = {t*1e6:.2f}, 最大温度 = {np.max(self.T):.4f}")

            k = np.zeros_like(self.T)
            C = np.zeros_like(self.T)
            P = np.zeros_like(self.T)

            # Modified P_func calculation to include spatial coordinates
            for i in range(self.Nx):
                for j in range(self.Ny):
                    for m in range(self.Nz):
                        T_val = self.T[i, j, m]
                        k[i, j, m] = self.k_func(T_val)
                        C[i, j, m] = self.C_func(T_val)
                        # Pass all required parameters to P_func
                        P[i, j, m] = self.P_func(T_val,
                                               self.x_mapped[i],
                                               self.y_mapped[j],
                                               self.z_mapped[m])

            T_new = self.T.copy()

            for i in range(1, self.Nx-1):
                for j in range(1, self.Ny-1):
                    for k_index in range(1, self.Nz-1):
                        laplacian_T = 0

                        d2T_dx2 = 0
                        for m in range(self.Nx):
                            d2T_dx2 += self.Dxx[i, m] * self.T[m, j, k_index]

                        d2T_dy2 = 0
                        for m in range(self.Ny):
                            d2T_dy2 += self.Dyy[j, m] * self.T[i, m, k_index]

                        d2T_dz2 = 0
                        for m in range(self.Nz):
                            d2T_dz2 += self.Dzz[k_index, m] * self.T[i, j, m]

                        laplacian_T = d2T_dx2 + d2T_dy2 + d2T_dz2

                        grad_k_dot_grad_T = 0

                        dk_dx = 0
                        dT_dx = 0
                        for m in range(self.Nx):
                            dk_dx += self.Dx[i, m] * k[m, j, k_index]
                            dT_dx += self.Dx[i, m] * self.T[m, j, k_index]

                        dk_dy = 0
                        dT_dy = 0
                        for m in range(self.Ny):
                            dk_dy += self.Dy[j, m] * k[i, m, k_index]
                            dT_dy += self.Dy[j, m] * self.T[i, m, k_index]

                        dk_dz = 0
                        dT_dz = 0
                        for m in range(self.Nz):
                            dk_dz += self.Dz[k_index, m] * k[i, j, m]
                            dT_dz += self.Dz[k_index, m] * self.T[i, j, m]

                        grad_k_dot_grad_T = dk_dx * dT_dx + dk_dy * dT_dy + dk_dz * dT_dz

                        rhs = k[i, j, k_index] * laplacian_T + grad_k_dot_grad_T + P[i, j, k_index]

                        T_new[i, j, k_index] = self.T[i, j, k_index] + self.dt / (self.rho * C[i, j, k_index]) * rhs

            # 边界条件：x/y 方向保持绝热，z 方向可选对流（Robin）或绝热
            for j in range(self.Ny):
                for k_index in range(self.Nz):
                    T_new[0, j, k_index] = T_new[1, j, k_index]
                    T_new[-1, j, k_index] = T_new[-2, j, k_index]

            for i in range(self.Nx):
                for k_index in range(self.Nz):
                    T_new[i, 0, k_index] = T_new[i, 1, k_index]
                    T_new[i, -1, k_index] = T_new[i, -2, k_index]

            # z=0 面：可选对流边界 -k*dT/dz = h*(T-T_inf)
            if self.h_z0 > 0:
                for i in range(self.Nx):
                    for j in range(self.Ny):
                        k_surf = max(k[i, j, 0], 1e-12)
                        beta = self.h_z0 * self.dz0 / k_surf
                        T_new[i, j, 0] = (T_new[i, j, 1] + beta * self.T_inf) / (1 + beta)
            else:
                for i in range(self.Nx):
                    for j in range(self.Ny):
                        T_new[i, j, 0] = T_new[i, j, 1]

            # z=c 面：可选对流边界
            if self.h_zc > 0:
                for i in range(self.Nx):
                    for j in range(self.Ny):
                        k_surf = max(k[i, j, -1], 1e-12)
                        beta = self.h_zc * self.dzc / k_surf
                        T_new[i, j, -1] = (T_new[i, j, -2] + beta * self.T_inf) / (1 + beta)
            else:
                for i in range(self.Nx):
                    for j in range(self.Ny):
                        T_new[i, j, -1] = T_new[i, j, -2]

            self.T = T_new
            t += self.dt
            step += 1

        return np.array(T_history), np.array(t_history)

    # [Keeping plot_slice, plot_3d, and _compute_isosurface methods unchanged]
    def plot_slice(self, axis='z', index=None, time_index=-1, T_history=None):
        # [Previous plot_slice code remains the same]
        if T_history is not None:
            T_plot = T_history[time_index]
        else:
            T_plot = self.T

        if index is None:
            if axis == 'x':
                index = self.Nx // 2
            elif axis == 'y':
                index = self.Ny // 2
            elif axis == 'z':
                index = self.Nz // 2

        plt.figure(figsize=(10, 8))

        if axis == 'x':
            plt.pcolormesh(self.y_mapped, self.z_mapped, T_plot[index, :, :].T, cmap='hot', shading='auto')
            plt.xlabel('y')
            plt.ylabel('z')
            plt.title(f'Temperature at x = {self.x_mapped[index]:.2f}')
        elif axis == 'y':
            plt.pcolormesh(self.x_mapped, self.z_mapped, T_plot[:, index, :].T, cmap='hot', shading='auto')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title(f'Temperature at y = {self.y_mapped[index]:.2f}')
        elif axis == 'z':
            plt.pcolormesh(self.x_mapped, self.y_mapped, T_plot[:, :, index], cmap='hot', shading='auto')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Temperature at z = {self.z_mapped[index]:.2f}')

        plt.colorbar(label='Temperature')
        plt.tight_layout()
        plt.show()

    def plot_3d(self, time_index=-1, T_history=None):
        # [Previous plot_3d code remains the same]
        if T_history is not None:
            T_plot = T_history[time_index]
        else:
            T_plot = self.T

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        X, Y, Z = np.meshgrid(self.x_mapped, self.y_mapped, self.z_mapped, indexing='ij')

        levels = np.linspace(np.min(T_plot), np.max(T_plot), 5)
        for level in levels:
            vertices, triangles = self._compute_isosurface(X, Y, Z, T_plot, level)
            if len(vertices) > 0:
                ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                triangles=triangles, color='r', alpha=0.3)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Temperature Field')
        plt.tight_layout()
        plt.show()

    def _compute_isosurface(self, X, Y, Z, T, level):
        # [Previous _compute_isosurface code remains the same]
        vertices = []
        triangles = []
        return np.array(vertices), np.array(triangles)

    def plot_3d_surface_over_time(self, T_history, t_history, z_index=None, time_indices=None):
        """
        绘制芯片表面或指定 z 平面温度随时间变化的三维图。

        参数：
        - T_history: 温度历史记录，形状为 (时间步数, Nx, Ny, Nz)。
        - t_history: 时间历史记录，形状为 (时间步数, )。
        - z_index: 要绘制的 z 平面索引。如果为 None，则默认为芯片表面（z = c）。
        - time_indices: 要绘制的时间点索引列表。如果为 None，则自动选择几个代表性时间点。
        """
        if z_index is None:
            z_index = -1  # 默认选择芯片表面（z = c）

        if time_indices is None:
            time_indices = np.linspace(0, len(t_history) - 1, 5, dtype=int)  # 默认选择 5 个时间点

        fig = plt.figure(figsize=(18, 6))

        for i, time_idx in enumerate(time_indices):
            ax = fig.add_subplot(1, len(time_indices), i + 1, projection='3d')

            T_plane = T_history[time_idx][:, :, z_index]  # 取指定 z 平面的温度分布
            X, Y = np.meshgrid(self.x_mapped, self.y_mapped, indexing='ij')

            surf = ax.plot_surface(X, Y, T_plane, cmap='hot', edgecolor='none')
            ax.set_title(f"t = {t_history[time_idx] * 1e6:.2f} µs, z = {self.z_mapped[z_index]:.2e} m")
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('Temperature (K)')

        fig.colorbar(surf, ax=fig.axes, shrink=0.5, aspect=10, label='Temperature (K)')
        plt.tight_layout()
        plt.show()

# Modified example usage with your new P_func
if __name__ == "__main__":
    a, b, c = 1.5e-3, 1e-3, 10e-5

    def k_func(T):
        return 1 / (-0.0003 + 1.05e-5 * T)

    def C_func(T):
        return 300 * (5.13 - 1001 / T + 3.23e4 / T ** 2)

    def physics_model(Vds, Vgs, T, param1, param3, param6):
        # 数值保护，防止 T 过小或分母为 0
        T_safe = np.maximum(T, 1e-6)
        denom = param1 * (T_safe / 300) ** -1.222 + param6 * (T_safe / 300) ** 2.062
        # 若 denom 为 0，则 NET5_value 设为 0
        NET5_value = param3 * np.where(denom != 0, 1 / denom, 0.0)
        NET3_value = -0.004263 * T_safe + 3.53892
        #term3 = np.log(1 + np.exp(Vgs - NET3_value)) ** 2
        term3 = (Vgs - NET3_value) ** 2
        term1 = NET5_value * (Vgs - NET3_value)
        term2 = 1 + 0.0005 * Vds
        return term2 * term1 * term3

    def z_function(z, c1):
        if 0 <= z <= c1:
            return 1 - z / c1
        else:
            return 0

    def P_func(T, x, y, z):
        Vds = 600.0
        Vgs = 18.0
        param1 = 75.098
        param3 = 6.170
        param6 = 10.079
        c1 = 1e-5
        base_value = physics_model(Vds, Vgs, T, param1, param3, param6)/(a*b)*2*Vds/c*z_function(z,c1)
        if x > 0.8*a and (3/8 * b < y < 5/8 * b):
            return base_value * 0
        return base_value

    def T_init(x, y, z):
        return 300.0

    Nx, Ny, Nz = 20, 20, 20
    dt = 0.01e-6
    total_time = 1e-6

    solver = HeatConductionSolver(
        a, b, c, Nx, Ny, Nz,
        k_func, C_func, P_func, T_init,
        dt, total_time, rho=3200.0,
        h_z0=5e4,   # 顶面对流换热系数，可根据封装/风冷条件调整
        h_zc=0.0,   # 底面若贴散热器可设 >0，否则设 0 绝热
        T_inf=300.0
    )

    start_time = time.time()
    T_history, t_history = solver.solve()
    end_time = time.time()

    print(f"求解耗时: {end_time - start_time:.2f} 秒")

    # 自定义颜色映射
    cmap = plt.get_cmap('viridis')  # 选择更美观的颜色映射，如 'viridis'

    # 绘制切片图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # z 切片
    im1 = axes[0].pcolormesh(solver.x_mapped, solver.y_mapped, T_history[-1][:, :, 0], cmap=cmap, shading='auto')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title(f'Temperature at z = {solver.z_mapped[0]:.2e}')

    # x 切片
    im2 = axes[1].pcolormesh(solver.y_mapped, solver.z_mapped, T_history[-1][Nx // 2, :, :].T, cmap=cmap, shading='auto')
    axes[1].set_xlabel('y')
    axes[1].set_ylabel('z')
    axes[1].set_title(f'Temperature at x = {solver.x_mapped[Nx // 2]:.2e}')

    # y 切片
    im3 = axes[2].pcolormesh(solver.x_mapped, solver.z_mapped, T_history[-1][:, Ny // 2, :].T, cmap=cmap, shading='auto')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('z')
    axes[2].set_title(f'Temperature at y = {solver.y_mapped[Ny // 2]:.2e}')

    # 共享颜色条
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im1, cax=cbar_ax, label='Temperature (K)')

    plt.show()

    # 绘制温度变化曲线
    plt.figure(figsize=(10, 6))
    center_temp = [T[Nx // 2, Ny // 2, Nz // 2] for T in T_history]
    plt.plot(t_history, center_temp)
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature at center point over time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# %%
    # 绘制芯片表面温度随时间变化的三维图
    solver.plot_3d_surface_over_time(T_history, t_history, z_index=0)
