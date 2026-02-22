import torch
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt


class TrueUShapedTimeStepSampler:
    """真正的U形时间步采样器 (两端高，中间低)"""

    def __init__(self, a: float = 4.0, device: str = 'cuda',
                 num_points: int = 10000, use_inverse_transform: bool = True):
        """
        初始化真正的U形时间步采样器

        参数:
            a: 控制U形尖锐程度的参数
            device: 计算设备
            num_points: 用于数值积分的点数
            use_inverse_transform: 是否使用逆变换采样
        """
        self.a = a
        self.device = device
        self.num_points = num_points
        self.use_inverse_transform = use_inverse_transform

        if use_inverse_transform:
            # 预计算CDF和查找表
            self.u_grid, self.cdf_grid = self._precompute_cdf_lookup()

    def _pdf(self, t: torch.Tensor) -> torch.Tensor:
        """
        真正的U形概率密度函数
        在t=0和t=1附近有高概率，中间低
        """
        # 方法1：使用两个对称的指数函数
        return torch.exp(-self.a * t) + torch.exp(-self.a * (1 - t))

    def _pdf_variant(self, t: torch.Tensor) -> torch.Tensor:
        """
        另一种U形PDF定义：beta-like分布
        """
        # Beta分布的PDF形状，当alpha=beta<1时呈U形
        alpha = 0.3  # 控制U形尖锐程度
        pdf = t ** (alpha - 1) * (1 - t) ** (alpha - 1)
        return pdf

    def _precompute_cdf_lookup(self) -> tuple:
        """预计算CDF查找表用于逆变换采样"""
        # 创建均匀网格
        u_grid = torch.linspace(0, 1, self.num_points, device=self.device)

        # 计算PDF值
        pdf_values = self._pdf(u_grid)

        # 使用梯形法则计算累积分布函数(CDF)
        # 计算每个小梯形的面积
        area = (pdf_values[:-1] + pdf_values[1:]) / 2 * (1.0 / (self.num_points - 1))
        cdf_values = torch.cumsum(area, dim=0)
        cdf_values = torch.cat([torch.tensor([0.0], device=self.device), cdf_values])

        # 归一化CDF
        cdf_values = cdf_values / cdf_values[-1]

        return u_grid, cdf_values

    def sample_inverse_transform(self, batch_size: int) -> torch.Tensor:
        """使用逆变换采样"""
        # 生成均匀随机数
        u = torch.rand(batch_size, device=self.device)

        # 使用查找表找到对应的t值
        indices = torch.searchsorted(self.cdf_grid, u)
        indices = torch.clamp(indices, 0, len(self.u_grid) - 1)

        t_samples = self.u_grid[indices]

        # 如果需要更精确的插值
        idx_lower = torch.clamp(indices - 1, 0, len(self.u_grid) - 1)
        idx_upper = torch.clamp(indices, 0, len(self.u_grid) - 1)

        # 避免除零
        cdf_diff = self.cdf_grid[idx_upper] - self.cdf_grid[idx_lower]
        cdf_diff = torch.where(cdf_diff == 0, torch.ones_like(cdf_diff), cdf_diff)

        weight = (u - self.cdf_grid[idx_lower]) / cdf_diff
        t_samples = self.u_grid[idx_lower] * (1 - weight) + self.u_grid[idx_upper] * weight

        return t_samples

    def sample_beta_distribution(self, batch_size: int, alpha: float = 0.3) -> torch.Tensor:
        """使用Beta分布实现真正的U形采样"""
        # Beta(α, β) 当 α=β<1 时呈现U形
        t = torch.distributions.Beta(alpha, alpha).sample((batch_size,))
        return t.to(self.device)

    def sample(self, batch_size: int, method: str = 'beta') -> torch.Tensor:
        """采样时间步"""
        if method == 'inverse_transform':
            return self.sample_inverse_transform(batch_size)
        elif method == 'beta':
            return self.sample_beta_distribution(batch_size)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def visualize_distribution(self, num_samples: int = 100000, method: str = 'beta'):
        """可视化采样分布"""
        # 采样
        samples = self.sample(num_samples, method=method).cpu().numpy()

        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 1. 直方图
        axes[0].hist(samples, bins=50, density=True, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Time step t')
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'True U-shaped Distribution (α={0.3})')
        axes[0].axvline(x=0.5, color='r', linestyle='--', alpha=0.5)

        # 理论PDF曲线
        t_vals = np.linspace(0.001, 0.999, 1000)

        # Beta分布的PDF
        alpha = 0.3
        from scipy.special import beta as beta_func
        pdf_vals = t_vals ** (alpha - 1) * (1 - t_vals) ** (alpha - 1) / beta_func(alpha, alpha)

        axes[0].plot(t_vals, pdf_vals, 'r-', linewidth=2, label='Beta PDF (α=0.3)')
        axes[0].legend()

        # 2. CDF图
        axes[1].hist(samples, bins=50, density=True, cumulative=True,
                     alpha=0.7, edgecolor='black', label='Empirical CDF')

        # 理论CDF (Beta分布的CDF)
        from scipy.special import betainc
        cdf_vals = betainc(alpha, alpha, t_vals)
        axes[1].plot(t_vals, cdf_vals, 'r-', linewidth=2, label='Theoretical CDF')
        axes[1].set_xlabel('Time step t')
        axes[1].set_ylabel('Cumulative Probability')
        axes[1].set_title('Cumulative Distribution Function')
        axes[1].legend()
        axes[1].axvline(x=0.5, color='r', linestyle='--', alpha=0.5)

        # 3. 分位数图
        sorted_samples = np.sort(samples)
        expected_quantiles = np.linspace(0, 1, len(sorted_samples))
        axes[2].plot(expected_quantiles, sorted_samples, 'b.', alpha=0.5, markersize=1)
        axes[2].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[2].set_xlabel('Expected Quantile')
        axes[2].set_ylabel('Sample Quantile')
        axes[2].set_title('Q-Q Plot')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 打印统计信息
        print(f"Sample statistics (n={num_samples}, method={method}):")
        print(f"  Mean: {np.mean(samples):.4f}")
        print(f"  Median: {np.median(samples):.4f}")
        print(f"  Std: {np.std(samples):.4f}")
        print(f"  Min: {np.min(samples):.4f}")
        print(f"  Max: {np.max(samples):.4f}")
        print(f"  Proportion t<0.1: {np.mean(samples < 0.1):.4f}")
        print(f"  Proportion t>0.9: {np.mean(samples > 0.9):.4f}")
        print(f"  Proportion 0.4<t<0.6: {np.mean((samples >= 0.4) & (samples <= 0.6)):.4f}")
        print(f"  Proportion 0.2<t<0.8: {np.mean((samples >= 0.2) & (samples <= 0.8)):.4f}")


# 比较不同U形尖锐程度
def compare_beta_shapes():
    """比较不同α值的Beta分布形状"""
    t_vals = np.linspace(0.001, 0.999, 1000)

    plt.figure(figsize=(10, 6))

    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    from scipy.special import beta as beta_func

    for alpha in alpha_values:
        pdf_vals = t_vals ** (alpha - 1) * (1 - t_vals) ** (alpha - 1) / beta_func(alpha, alpha)
        plt.plot(t_vals, pdf_vals, label=f'Beta({alpha},{alpha})', linewidth=2)

    plt.xlabel('Time step t')
    plt.ylabel('Probability Density')
    plt.title('U-shaped Beta Distributions with Different α Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 比较不同Beta分布形状
    compare_beta_shapes()

    # 切换为逆变换采样
    sampler_inverse = TrueUShapedTimeStepSampler(a=4.0, device='cpu', use_inverse_transform=True)

    # 采样一批时间步
    batch_size = 64
    t_samples = sampler_inverse.sample(batch_size, method='inverse_transform')
    print(t_samples)

