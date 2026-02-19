"""
ImprovedSeisEncoder - 改进的地震数据编码器

设计原则（基于设计文档）：
1. Conv1d处理时间维度（kernel_size=11），捕捉波形的时间依赖
2. Conv2d处理空间维度（时间×接收器的2D特征图）
3. 震源感知特征聚合（1x1卷积融合，替代简单拼接）
4. 可学习时间池化（替代简单平均池化）
5. 空间注意力机制（自适应选择重要接收器位置）
6. 可选的低频FFT特征（仅前10%频率，默认禁用）

输入: (B, 5, 1000, 70) - 5个震源, 1000时间步, 70接收器
输出: (B, 64, 64, 64) - 64通道, 64x64空间分辨率

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 9.2, 9.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableTimePooling(nn.Module):
    """
    可学习时间池化模块
    
    替代简单平均池化，让模型自动学习哪些时刻重要
    
    Requirements: 1.2
    """
    def __init__(self, in_channels: int, target_size: int = 64):
        super().__init__()
        self.target_size = target_size
        
        # 使用1D卷积进行可学习的时间压缩
        # 先用stride=4的卷积压缩，再用自适应池化微调到目标尺寸
        self.conv_pool = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=7, stride=4, padding=3),
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.SiLU(),
        )
        
        # 最终自适应池化到目标尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_size)
        
        # 可学习的注意力权重（时间维度）
        self.time_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4),
            nn.SiLU(),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) - 时间序列特征
        
        Returns:
            (B, C, target_size) - 压缩后的时间特征
        """
        # 计算时间注意力权重
        attn = self.time_attention(x)  # (B, C)
        attn = attn.unsqueeze(-1)  # (B, C, 1)
        
        # 应用注意力加权
        x = x * attn
        
        # 可学习的卷积池化
        x = self.conv_pool(x)
        
        # 自适应池化到目标尺寸
        x = self.adaptive_pool(x)
        
        return x


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    
    自适应选择重要接收器位置，不同接收器位置的重要性不同
    
    Requirements: 1.3
    """
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        
        # 通道注意力（SE-like）
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.SiLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        
        # 空间注意力（CBAM-like）
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) - 2D特征图
        
        Returns:
            (B, C, H, W) - 注意力加权后的特征图
        """
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(spatial_input)
        x = x * sa
        
        return x


class LowFreqFFTExtractor(nn.Module):
    """
    低频FFT特征提取模块
    
    仅保留前10%频率，避免高频噪声导致过拟合
    默认禁用，作为实验选项
    
    Requirements: 1.5, 9.3
    """
    def __init__(self, in_sources: int = 5, out_channels: int = 10, freq_ratio: float = 0.1):
        super().__init__()
        self.in_sources = in_sources
        self.out_channels = out_channels
        self.freq_ratio = freq_ratio  # 保留前10%频率
        
        # 融合层：将低频特征融合到指定通道数
        # 确保num_groups能整除out_channels
        num_groups = min(4, out_channels)
        while out_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        
        self.fusion = nn.Sequential(
            nn.Conv2d(in_sources, out_channels, kernel_size=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
        )
    
    def forward(self, seis: torch.Tensor) -> torch.Tensor:
        """
        提取低频FFT特征（仅前10%频率）
        
        Args:
            seis: (B, 5, 1000, 70) - 原始地震数据
        
        Returns:
            (B, out_channels, 64, 64) - 低频特征图
        """
        B, S, T, R = seis.shape
        
        # FFT沿时间维度
        fft = torch.fft.rfft(seis, dim=2)  # (B, 5, 501, 70)
        
        # 只保留前10%频率
        n_freq = fft.shape[2]
        low_freq_idx = max(1, int(n_freq * self.freq_ratio))  # 至少保留1个频率
        low_freq = fft[:, :, :low_freq_idx, :]  # (B, 5, ~50, 70)
        
        # 取幅度
        low_freq_amp = torch.abs(low_freq)  # (B, 5, ~50, 70)
        
        # 插值到64x64
        low_freq_amp = F.interpolate(
            low_freq_amp,
            size=(70, 70),
            mode='bilinear',
            align_corners=False
        )  # (B, 5, 70, 70)
        
        # 融合到目标通道数
        low_freq_features = self.fusion(low_freq_amp)  # (B, out_channels, 70, 70)
        
        return low_freq_features



class ImprovedSeisEncoder(nn.Module):
    """
    改进的地震数据编码器
    
    关键改进:
    1. 可学习时间池化（替代简单平均池化）
    2. 空间注意力机制（自适应选择重要接收器）
    3. 震源感知特征聚合（替代简单拼接）
    4. 可选的低频FFT特征（仅前10%频率）
    
    卷积使用策略:
    - Stage 1 (时间处理): Conv1d 沿时间轴，kernel_size=11
    - Stage 2 (空间处理): Conv2d 处理(T', R)的2D特征图
    - Stage 3 (特征融合): Conv2d 处理最终的64x64特征图
    
    输入: (B, 5, 1000, 70) - 5个震源, 1000时间步, 70接收器
    输出: (B, 64, 70, 70) - 64通道, 70x70空间分辨率
    
    Requirements: 1.1, 1.2, 1.3, 1.4, 9.2
    """
    
    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 64,
        use_low_freq_fft: bool = False,  # 谨慎使用，仅提取低频，默认禁用
        conv1d_kernel_size: int = 11,    # 7-15范围内，用于时间维度 (Requirements 9.2)
        time_pool_learnable: bool = True,
        use_spatial_attention: bool = True,
        fft_channels: int = 10,  # 低频FFT特征通道数
    ):
        super().__init__()
        
        # 验证conv1d_kernel_size在合理范围内 (Requirements 9.2)
        assert 7 <= conv1d_kernel_size <= 15, f"conv1d_kernel_size must be in [7, 15], got {conv1d_kernel_size}"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_low_freq_fft = use_low_freq_fft
        self.conv1d_kernel_size = conv1d_kernel_size
        self.time_pool_learnable = time_pool_learnable
        self.use_spatial_attention = use_spatial_attention
        
        # ========================================
        # Stage 1: 时间维度处理 (Conv1d)
        # 输入: (B*5*70, 1, 1000) -> 输出: (B*5*70, 32, T')
        # Requirements: 1.1, 1.2, 9.2
        # 🔥 优化：第一层加 stride=2，显存占用减半
        # ========================================
        '''self.time_conv = nn.Sequential(
            # 🔥 第一层就降采样：1000 -> 500
            nn.Conv1d(1, 16, kernel_size=conv1d_kernel_size, padding=conv1d_kernel_size // 2, stride=2),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            # 第二层继续降采样：500 -> 250
            nn.Conv1d(16, 32, kernel_size=conv1d_kernel_size, padding=conv1d_kernel_size // 2, stride=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )'''
        self.time_conv = nn.Sequential(
            # 🔥 第一层就降采样：1000 -> 500
            nn.Conv1d(1, 16, kernel_size=conv1d_kernel_size, padding=conv1d_kernel_size // 2, stride=2),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            # 第二层继续降采样：500 -> 250
            nn.Conv1d(16, 32, kernel_size=conv1d_kernel_size, padding=conv1d_kernel_size // 2, stride=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )

        # 可学习时间池化（替代简单平均池化）(Requirements 1.2)
        if time_pool_learnable:
            self.time_pool = LearnableTimePooling(in_channels=32, target_size=70)
        else:
            self.time_pool = nn.AdaptiveAvgPool1d(70)
        
        # ========================================
        # Stage 2: 空间维度处理 (Conv2d)
        # 输入: (B*5, 32, 70, 70) -> 输出: (B*5, 64, 70, 70)
        # Requirements: 1.1
        # ========================================
        self.spatial_conv = nn.Sequential(
            # Conv2d 处理 (时间', 接收器) 的2D特征图
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            # 接收器维度 70 -> 70 (通过卷积+池化)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        

        
        # 空间注意力（自适应选择重要接收器位置）(Requirements 1.3)
        if use_spatial_attention:
            self.spatial_attention = SpatialAttention(in_channels=64, reduction=4)
        else:
            self.spatial_attention = None
        
        # ========================================
        # Stage 3: 震源融合 (Conv2d)
        # 输入: (B, 5*64, 70, 70) -> 输出: (B, 64, 70, 70)
        # Requirements: 1.4
        # ========================================
        self.source_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 64, 128, kernel_size=1),  # 1x1卷积融合震源
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )
        
        # 可选: 低频FFT特征 (Requirements 1.5, 9.3)
        self.use_low_freq_fft = use_low_freq_fft
        if use_low_freq_fft:
            self.fft_extractor = LowFreqFFTExtractor(
                in_sources=in_channels,
                out_channels=fft_channels,
                freq_ratio=0.1  # 仅保留前10%频率
            )
            self.fft_fusion = nn.Conv2d(out_channels + fft_channels, out_channels, kernel_size=1)
    
    def forward(self, seis: torch.Tensor) -> torch.Tensor:
        """
        处理流程:
        1. Conv1d处理时间维度（kernel_size=11）
        2. 可学习时间池化（保留关键时刻）
        3. Conv2d处理空间维度
        4. 空间注意力（自适应选择接收器）
        5. 震源感知聚合（1x1卷积融合）
        6. 可选: 低频FFT特征融合（前10%频率）
        
        Args:
            seis: (B, 5, 1000, 70) - 5个震源, 1000时间步, 70接收器
        
        Returns:
            features: (B, 64, 70, 70) - 64通道, 70x70空间分辨率
        """
        B, S, T, R = seis.shape  # B, 5, 1000, 70
        
        # 数据预处理：软限幅，保留动态范围
        seis = torch.tanh(seis / 2.0) * 2.0
        
        # ========================================
        # Stage 1: 时间处理 (Conv1d)
        # ========================================
        # Reshape: (B, 5, 1000, 70) -> (B*5*70, 1, 1000)
        x = seis.permute(0, 1, 3, 2).reshape(B * S * R, 1, T)
        x = self.time_conv(x)  # (B*5*70, 32, T')

        # 可学习时间池化
        x = self.time_pool(x)  # (B*5*70, 32, 70)
        
        # ========================================
        # Stage 2: 空间处理 (Conv2d)
        # ========================================
        # Reshape: (B*5*70, 32, 70) -> (B*5, 32, 70, 70)
        x = x.reshape(B * S, R, 32, 70).permute(0, 2, 3, 1)  # (B*5, 32, 70, 70)
        
        # 空间卷积
        x = self.spatial_conv(x)  # (B*5, 70, H, W)

        
        # 空间注意力
        if self.spatial_attention is not None:
            x = self.spatial_attention(x)
        
        # ========================================
        # Stage 3: 震源融合
        # ========================================
        # Reshape: (B*5, 64, 70, 70) -> (B, 5*64, 70, 70)
        x = x.reshape(B, S * 64, 70, 70)
        x = self.source_fusion(x)  # (B, 64, 70, 70)
        
        # ========================================
        # 可选: 低频FFT特征融合
        # ========================================
        if self.use_low_freq_fft:
            fft_features = self.fft_extractor(seis)  # (B, fft_channels, 70, 70)
            x = torch.cat([x, fft_features], dim=1)
            x = self.fft_fusion(x)  # (B, 64, 70, 70)
        
        return x


# 便捷函数：创建默认配置的ImprovedSeisEncoder
def create_improved_seis_encoder(
    out_channels: int = 64,
    use_low_freq_fft: bool = False,
    conv1d_kernel_size: int = 11,
) -> ImprovedSeisEncoder:
    """
    创建改进的地震编码器
    
    Args:
        out_channels: 输出通道数，默认64
        use_low_freq_fft: 是否使用低频FFT特征，默认False（谨慎使用）
        conv1d_kernel_size: Conv1d核大小，必须在[7, 15]范围内，默认11
    
    Returns:
        ImprovedSeisEncoder实例
    """
    return ImprovedSeisEncoder(
        in_channels=5,
        out_channels=out_channels,
        use_low_freq_fft=use_low_freq_fft,
        conv1d_kernel_size=conv1d_kernel_size,
        time_pool_learnable=True,
        use_spatial_attention=True,
    )

"""
物理约束的数据增强模块

实现符合地震物理规律的数据增强方法，避免破坏物理一致性。
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import random


class PhysicsAwareAugmentation:
    """
    物理约束的数据增强
    
    允许的增强:
    - 水平翻转: 同步翻转速度模型和地震数据接收器维度
    - Mixup: 仅在相似地质结构样本间混合（alpha=0.2）
    - 噪声注入: 高斯噪声std=0.01-0.02
    - 时间扰动: 随机shift时间轴±5%
    
    禁止的增强:
    - 旋转（破坏重力方向）
    - 垂直翻转（破坏地层顺序）
    - 弹性形变（破坏波传播规律）
    """
    
    def __init__(
        self,
        enable_horizontal_flip: bool = True,
        enable_mixup: bool = True,
        mixup_alpha: float = 0.2,
        noise_std_range: Tuple[float, float] = (0.01, 0.02),
        time_shift_range: float = 0.05,  # ±5%
        similarity_threshold: float = 0.7  # Mixup相似度阈值
    ):
        """
        初始化物理约束数据增强器
        
        Args:
            enable_horizontal_flip: 是否启用水平翻转
            enable_mixup: 是否启用Mixup
            mixup_alpha: Mixup的alpha参数
            noise_std_range: 噪声标准差范围 (min, max)
            time_shift_range: 时间扰动范围（相对于时间长度的比例）
            similarity_threshold: Mixup相似度阈值
        """
        self.enable_horizontal_flip = enable_horizontal_flip
        self.enable_mixup = enable_mixup
        self.mixup_alpha = mixup_alpha
        self.noise_std_range = noise_std_range
        self.time_shift_range = time_shift_range
        self.similarity_threshold = similarity_threshold
    
    def horizontal_flip(
        self, 
        velocity: torch.Tensor, 
        seismic: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        同步水平翻转速度模型和地震数据
        
        物理约束: 水平翻转保持地层结构的物理一致性，
        速度模型和地震数据必须同步翻转以保持对应关系。
        
        Args:
            velocity: (B, 1, H, W) 速度模型
            seismic: (B, 5, T, R) 地震数据
        
        Returns:
            flipped_velocity: (B, 1, H, W) 翻转后的速度模型
            flipped_seismic: (B, 5, T, R) 翻转后的地震数据
        """
        # 速度模型沿宽度维度翻转 (dim=-1)
        flipped_velocity = torch.flip(velocity, dims=[-1])
        
        # 地震数据沿接收器维度翻转 (dim=-1)
        flipped_seismic = torch.flip(seismic, dims=[-1])
        
        return flipped_velocity, flipped_seismic
    
    def mixup(
        self,
        velocity1: torch.Tensor,
        seismic1: torch.Tensor,
        velocity2: torch.Tensor,
        seismic2: torch.Tensor,
        similarity: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        条件Mixup: 仅在相似样本间混合
        
        物理约束: 只有地质结构相似的样本才能混合，
        避免混合差异过大的样本导致非物理的结果。
        
        Args:
            velocity1: (B, 1, H, W) 第一个速度模型
            seismic1: (B, 5, T, R) 第一个地震数据
            velocity2: (B, 1, H, W) 第二个速度模型
            seismic2: (B, 5, T, R) 第二个地震数据
            similarity: 两个样本的相似度 (0-1)，如果为None则计算
        
        Returns:
            mixed_velocity: (B, 1, H, W) 混合后的速度模型
            mixed_seismic: (B, 5, T, R) 混合后的地震数据
        """
        # 如果未提供相似度，计算速度模型的相似度
        if similarity is None:
            similarity = self._compute_similarity(velocity1, velocity2)
        
        # 只有相似度超过阈值才执行Mixup
        if similarity < self.similarity_threshold:
            return velocity1, seismic1
        
        # 从Beta分布采样lambda
        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
        
        # 混合速度模型和地震数据
        mixed_velocity = lam * velocity1 + (1 - lam) * velocity2
        mixed_seismic = lam * seismic1 + (1 - lam) * seismic2
        
        return mixed_velocity, mixed_seismic
    
    def add_noise(self, seismic: torch.Tensor) -> torch.Tensor:
        """
        注入小幅度高斯噪声
        
        物理约束: 模拟真实地震数据采集中的噪声，
        噪声标准差控制在0.01-0.02范围内。
        
        Args:
            seismic: (B, 5, T, R) 地震数据
        
        Returns:
            noisy_seismic: (B, 5, T, R) 注入噪声后的地震数据
        """
        # 随机选择噪声标准差
        std = random.uniform(self.noise_std_range[0], self.noise_std_range[1])
        
        # 生成高斯噪声
        noise = torch.randn_like(seismic) * std
        
        # 注入噪声
        noisy_seismic = seismic + noise
        
        return noisy_seismic
    
    def time_shift(self, seismic: torch.Tensor) -> torch.Tensor:
        """
        随机时间轴偏移
        
        物理约束: 模拟地震波到达时间的微小变化，
        偏移量控制在±5%范围内以保持物理合理性。
        
        Args:
            seismic: (B, 5, T, R) 地震数据
        
        Returns:
            shifted_seismic: (B, 5, T, R) 时间偏移后的地震数据
        """
        B, S, T, R = seismic.shape
        
        # 计算最大偏移量（±5%）
        max_shift = int(T * self.time_shift_range)
        
        # 随机选择偏移量
        shift = random.randint(-max_shift, max_shift)
        
        if shift == 0:
            return seismic
        
        # 执行时间偏移
        if shift > 0:
            # 向右偏移（延迟）
            shifted_seismic = torch.cat([
                torch.zeros(B, S, shift, R, device=seismic.device, dtype=seismic.dtype),
                seismic[:, :, :-shift, :]
            ], dim=2)
        else:
            # 向左偏移（提前）
            shifted_seismic = torch.cat([
                seismic[:, :, -shift:, :],
                torch.zeros(B, S, -shift, R, device=seismic.device, dtype=seismic.dtype)
            ], dim=2)
        
        return shifted_seismic
    
    def _compute_similarity(
        self, 
        velocity1: torch.Tensor, 
        velocity2: torch.Tensor
    ) -> float:
        """
        计算两个速度模型的相似度
        
        使用归一化的L2距离作为相似度度量
        
        Args:
            velocity1: (B, 1, H, W) 第一个速度模型
            velocity2: (B, 1, H, W) 第二个速度模型
        
        Returns:
            similarity: 相似度 (0-1)，1表示完全相同
        """
        # 计算L2距离
        diff = velocity1 - velocity2
        l2_dist = torch.sqrt(torch.sum(diff ** 2))
        
        # 归一化到[0, 1]范围
        # 使用速度模型的范数作为归一化因子
        norm1 = torch.sqrt(torch.sum(velocity1 ** 2))
        norm2 = torch.sqrt(torch.sum(velocity2 ** 2))
        max_dist = norm1 + norm2
        
        if max_dist > 0:
            similarity = 1.0 - (l2_dist / max_dist).item()
        else:
            similarity = 1.0
        
        return max(0.0, min(1.0, similarity))
    
    def apply(
        self,
        velocity: torch.Tensor,
        seismic: torch.Tensor,
        velocity2: Optional[torch.Tensor] = None,
        seismic2: Optional[torch.Tensor] = None,
        apply_horizontal_flip: bool = True,
        apply_mixup: bool = True,
        apply_noise: bool = True,
        apply_time_shift: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用数据增强
        
        Args:
            velocity: (B, 1, H, W) 速度模型
            seismic: (B, 5, T, R) 地震数据
            velocity2: (B, 1, H, W) 第二个速度模型（用于Mixup）
            seismic2: (B, 5, T, R) 第二个地震数据（用于Mixup）
            apply_horizontal_flip: 是否应用水平翻转
            apply_mixup: 是否应用Mixup
            apply_noise: 是否应用噪声注入
            apply_time_shift: 是否应用时间扰动
        
        Returns:
            augmented_velocity: (B, 1, H, W) 增强后的速度模型
            augmented_seismic: (B, 5, T, R) 增强后的地震数据
        """
        aug_velocity = velocity
        aug_seismic = seismic
        
        # 水平翻转（50%概率）
        if apply_horizontal_flip and self.enable_horizontal_flip and random.random() < 0.5:
            aug_velocity, aug_seismic = self.horizontal_flip(aug_velocity, aug_seismic)
        
        # Mixup（需要提供第二个样本）
        if apply_mixup and self.enable_mixup and velocity2 is not None and seismic2 is not None:
            aug_velocity, aug_seismic = self.mixup(
                aug_velocity, aug_seismic, velocity2, seismic2
            )
        
        # 噪声注入（50%概率）
        if apply_noise and random.random() < 0.5:
            aug_seismic = self.add_noise(aug_seismic)
        
        # 时间扰动（50%概率）
        if apply_time_shift and random.random() < 0.5:
            aug_seismic = self.time_shift(aug_seismic)
        
        return aug_velocity, aug_seismic


"""
渐进式正则化控制器

根据训练阶段和SSIM指标动态调整正则化强度。

核心逻辑:
- Train SSIM < 0.85: 禁用所有正则化（专注学习）
- Train SSIM 0.85-0.90: 轻度正则化（Dropout=0.1）
- Train SSIM > 0.90: 中度正则化（Dropout=0.15, weight_decay=2e-2）
- Test gap > 0.1: 动态增强（Dropout最高0.25）
- Test gap < 0.05: 停止增强

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RegularizationConfig:
    """
    正则化配置数据类
    
    包含所有正则化相关的超参数配置。
    
    Requirements: 3.1, 3.2, 3.3
    """
    dropout_prob: float
    weight_decay: float
    use_mixup: bool
    use_horizontal_flip: bool
    use_noise_injection: bool = False
    use_time_shift: bool = False
    
    def __repr__(self) -> str:
        return (
            f"RegularizationConfig("
            f"dropout={self.dropout_prob:.3f}, "
            f"weight_decay={self.weight_decay:.2e}, "
            f"mixup={self.use_mixup}, "
            f"h_flip={self.use_horizontal_flip}, "
            f"noise={self.use_noise_injection}, "
            f"time_shift={self.use_time_shift})"
        )


class ProgressiveRegularizer:
    """
    渐进式正则化控制器
    
    根据训练集SSIM和测试集SSIM动态调整正则化强度，实现三阶段训练策略：
    
    阶段1（Train SSIM < 0.85）：禁用所有正则化
    - 目标：让模型充分学习训练数据
    - 策略：dropout=0, weight_decay=0, 禁用所有数据增强
    
    阶段2（Train SSIM 0.85-0.90）：轻度正则化
    - 目标：开始缩小泛化gap
    - 策略：dropout=0.1, 启用水平翻转
    
    阶段3（Train SSIM > 0.90）：中度正则化
    - 目标：进一步缩小gap至<0.10
    - 策略：dropout=0.15, weight_decay=2e-2, 启用Mixup
    
    动态调整：
    - 当gap > 0.1时，逐步增强正则化（dropout最高0.25）
    - 当gap < 0.05时，停止增强（避免过度正则化）
    
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
    """
    
    def __init__(
        self,
        ssim_threshold_low: float = 0.85,
        ssim_threshold_high: float = 0.90,
        gap_threshold_high: float = 0.10,
        gap_threshold_low: float = 0.05,
        dropout_min: float = 0.0,
        dropout_max: float = 0.25,
        dropout_step: float = 0.05,
        weight_decay_base: float = 1e-2,
        weight_decay_enhanced: float = 2e-2
    ):
        """
        初始化渐进式正则化控制器
        
        Args:
            ssim_threshold_low: 训练集SSIM低阈值（默认0.85）
            ssim_threshold_high: 训练集SSIM高阈值（默认0.90）
            gap_threshold_high: 泛化gap高阈值（默认0.10）
            gap_threshold_low: 泛化gap低阈值（默认0.05）
            dropout_min: 最小dropout概率（默认0.0）
            dropout_max: 最大dropout概率（默认0.25）
            dropout_step: dropout增强步长（默认0.05）
            weight_decay_base: 基础weight decay（默认1e-2）
            weight_decay_enhanced: 增强weight decay（默认2e-2）
        """
        self.ssim_threshold_low = ssim_threshold_low
        self.ssim_threshold_high = ssim_threshold_high
        self.gap_threshold_high = gap_threshold_high
        self.gap_threshold_low = gap_threshold_low
        self.dropout_min = dropout_min
        self.dropout_max = dropout_max
        self.dropout_step = dropout_step
        self.weight_decay_base = weight_decay_base
        self.weight_decay_enhanced = weight_decay_enhanced
        
        # 当前dropout增强级别（用于动态调整）
        self.current_dropout_boost = 0.0
    
    def get_regularization_config(
        self, 
        train_ssim: float, 
        test_ssim: Optional[float] = None
    ) -> RegularizationConfig:
        """
        根据当前训练状态返回正则化配置
        
        核心逻辑：
        1. 根据train_ssim确定基础正则化级别（阶段1/2/3）
        2. 如果提供test_ssim，根据gap动态调整dropout
        3. 返回完整的正则化配置
        
        Args:
            train_ssim: 训练集SSIM（必需）
            test_ssim: 测试集SSIM（可选，用于gap计算）
        
        Returns:
            RegularizationConfig: 包含所有正则化参数的配置对象
        
        Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
        """
        # ========================================
        # 阶段1：Train SSIM < 0.85 - 禁用所有正则化
        # Requirements: 3.1
        # ========================================
        if train_ssim < self.ssim_threshold_low:
            return RegularizationConfig(
                dropout_prob=0.0,
                weight_decay=0.0,
                use_mixup=False,
                use_horizontal_flip=False,
                use_noise_injection=False,
                use_time_shift=False
            )
        
        # ========================================
        # 阶段2：Train SSIM 0.85-0.90 - 轻度正则化
        # Requirements: 3.2
        # ========================================
        elif train_ssim < self.ssim_threshold_high:
            base_dropout = 0.1
            
            # 如果提供test_ssim，根据gap动态调整
            if test_ssim is not None:
                gap = train_ssim - test_ssim
                dropout_prob = self._adjust_dropout_by_gap(base_dropout, gap)
            else:
                dropout_prob = base_dropout
            
            return RegularizationConfig(
                dropout_prob=dropout_prob,
                weight_decay=self.weight_decay_base,
                use_mixup=False,  # 阶段2还不启用Mixup
                use_horizontal_flip=True,
                use_noise_injection=False,
                use_time_shift=False
            )
        
        # ========================================
        # 阶段3：Train SSIM >= 0.90 - 中度正则化
        # Requirements: 3.3
        # ========================================
        else:
            base_dropout = 0.15
            
            # 如果提供test_ssim，根据gap动态调整
            if test_ssim is not None:
                gap = train_ssim - test_ssim
                dropout_prob = self._adjust_dropout_by_gap(base_dropout, gap)
            else:
                dropout_prob = base_dropout
            
            return RegularizationConfig(
                dropout_prob=dropout_prob,
                weight_decay=self.weight_decay_enhanced,
                use_mixup=True,  # 阶段3启用Mixup
                use_horizontal_flip=True,
                use_noise_injection=True,  # 可选启用噪声注入
                use_time_shift=True  # 可选启用时间扰动
            )
    
    def _adjust_dropout_by_gap(self, base_dropout: float, gap: float) -> float:
        """
        根据泛化gap动态调整dropout
        
        逻辑：
        - gap > 0.1: 增强dropout（每次+0.05，最高0.25）
        - gap < 0.05: 停止增强，使用基础dropout
        - 0.05 <= gap <= 0.1: 保持当前dropout
        
        Args:
            base_dropout: 基础dropout概率
            gap: 泛化gap（train_ssim - test_ssim）
        
        Returns:
            float: 调整后的dropout概率
        
        Requirements: 3.4, 3.5
        """
        # ========================================
        # Requirements 3.4: gap > 0.1时动态增强
        # ========================================
        if gap > self.gap_threshold_high:
            # 增强dropout（但不超过最大值）
            self.current_dropout_boost = min(
                self.current_dropout_boost + self.dropout_step,
                self.dropout_max - base_dropout
            )
        
        # ========================================
        # Requirements 3.5: gap < 0.05时停止增强
        # ========================================
        elif gap < self.gap_threshold_low:
            # 停止增强，回到基础dropout
            self.current_dropout_boost = 0.0
        
        # 0.05 <= gap <= 0.1: 保持当前dropout boost
        
        # 计算最终dropout（不超过最大值）
        final_dropout = min(
            base_dropout + self.current_dropout_boost,
            self.dropout_max
        )
        
        return final_dropout
    
    def reset_dropout_boost(self) -> None:
        """
        重置dropout增强级别
        
        用于实验或调试时手动重置动态调整状态。
        """
        self.current_dropout_boost = 0.0
    
    def get_current_phase(self, train_ssim: float) -> str:
        """
        获取当前训练阶段描述
        
        Args:
            train_ssim: 训练集SSIM
        
        Returns:
            str: 阶段描述（"Phase 1", "Phase 2", "Phase 3"）
        """
        if train_ssim < self.ssim_threshold_low:
            return "Phase 1: No Regularization (Train SSIM < 0.85)"
        elif train_ssim < self.ssim_threshold_high:
            return "Phase 2: Light Regularization (Train SSIM 0.85-0.90)"
        else:
            return "Phase 3: Moderate Regularization (Train SSIM >= 0.90)"


# ========================================
# 使用示例
# ========================================
if __name__ == "__main__":
    # 创建控制器
    regularizer = ProgressiveRegularizer()
    
    # 模拟训练过程
    print("=== 模拟训练过程 ===\n")
    
    # 阶段1：欠拟合阶段
    print("阶段1：Train SSIM = 0.78, Test SSIM = 0.65")
    config = regularizer.get_regularization_config(train_ssim=0.78, test_ssim=0.65)
    print(f"配置: {config}")
    print(f"阶段: {regularizer.get_current_phase(0.78)}\n")
    
    # 阶段2：轻度正则化
    print("阶段2：Train SSIM = 0.87, Test SSIM = 0.75")
    config = regularizer.get_regularization_config(train_ssim=0.87, test_ssim=0.75)
    print(f"配置: {config}")
    print(f"阶段: {regularizer.get_current_phase(0.87)}\n")
    
    # 阶段3：中度正则化
    print("阶段3：Train SSIM = 0.92, Test SSIM = 0.80")
    config = regularizer.get_regularization_config(train_ssim=0.92, test_ssim=0.80)
    print(f"配置: {config}")
    print(f"阶段: {regularizer.get_current_phase(0.92)}\n")
    
    # 动态调整：gap过大
    print("动态调整：Train SSIM = 0.92, Test SSIM = 0.78 (gap=0.14 > 0.1)")
    config = regularizer.get_regularization_config(train_ssim=0.92, test_ssim=0.78)
    print(f"配置: {config}")
    print(f"Dropout增强: {regularizer.current_dropout_boost:.3f}\n")
    
    # 继续增强
    print("继续增强：Train SSIM = 0.93, Test SSIM = 0.79 (gap=0.14 > 0.1)")
    config = regularizer.get_regularization_config(train_ssim=0.93, test_ssim=0.79)
    print(f"配置: {config}")
    print(f"Dropout增强: {regularizer.current_dropout_boost:.3f}\n")
    
    # gap缩小，停止增强
    print("Gap缩小：Train SSIM = 0.93, Test SSIM = 0.90 (gap=0.03 < 0.05)")
    config = regularizer.get_regularization_config(train_ssim=0.93, test_ssim=0.90)
    print(f"配置: {config}")
    print(f"Dropout增强: {regularizer.current_dropout_boost:.3f}\n")


"""
学习率控制器

实现优化的学习率调度策略，避免过早衰减导致训练集欠拟合。

核心功能:
- 初始学习率: 3e-4（比原来2e-4更激进）
- Warmup: 10k步（比原来5k更平滑）
- Cosine Annealing周期: 80k步（避免过早衰减）
- 动态调整: 停滞时重启，下降时衰减

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import math
from typing import Optional


class LearningRateController:
    """
    学习率控制器
    
    基础配置:
    - 初始学习率: 3e-4（比原来2e-4更激进，加速收敛）
    - Warmup: 10k步（比原来5k更平滑，避免早期震荡）
    - Cosine Annealing周期: 80k步（避免过早衰减到20%）
    - 最小学习率比例: 0.2（最低衰减到初始值的20%）
    
    动态调整:
    - Train SSIM停滞: 重启学习率至初始值50%（给模型"第二次机会"）
    - Test SSIM下降: 降低学习率至当前值50%（减缓过拟合速度）
    
    Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5
    """
    
    def __init__(
        self,
        initial_lr: float = 3e-4,
        warmup_steps: int = 10000,
        cosine_period: int = 80000,
        min_lr_ratio: float = 0.2,
        restart_ratio: float = 0.5,
        decay_ratio: float = 0.5
    ):
        """
        初始化学习率控制器
        
        Args:
            initial_lr: 初始学习率，默认3e-4
            warmup_steps: Warmup步数，默认10k
            cosine_period: Cosine Annealing周期，默认80k
            min_lr_ratio: 最小学习率比例，默认0.2
            restart_ratio: 重启时的学习率比例，默认0.5
            decay_ratio: 衰减时的学习率比例，默认0.5
        """
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.cosine_period = cosine_period
        self.min_lr_ratio = min_lr_ratio
        self.restart_ratio = restart_ratio
        self.decay_ratio = decay_ratio
        
        # 当前有效的初始学习率（可能因重启或衰减而改变）
        self.current_base_lr = initial_lr
        
        # 记录重启和衰减的步数
        self.restart_step: Optional[int] = None
        self.decay_step: Optional[int] = None
        
        # 记录调整历史
        self.adjustment_history = []
    
    def get_lr(self, step: int) -> float:
        """
        获取当前步数的学习率
        
        学习率调度策略:
        1. Warmup阶段（0 - warmup_steps）: 线性增长从0到current_base_lr
        2. Cosine Annealing阶段（warmup_steps - cosine_period）: 余弦衰减
        3. 最小学习率阶段（> cosine_period）: 保持在min_lr
        
        Args:
            step: 当前训练步数
        
        Returns:
            当前学习率
        """
        # 阶段1: Warmup
        if step < self.warmup_steps:
            # 线性增长: lr = current_base_lr * (step / warmup_steps)
            return self.current_base_lr * (step / self.warmup_steps)
        
        # 阶段2: Cosine Annealing
        elif step < self.cosine_period:
            # 余弦衰减
            progress = (step - self.warmup_steps) / (self.cosine_period - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            
            # lr = min_lr + (current_base_lr - min_lr) * cosine_decay
            min_lr = self.current_base_lr * self.min_lr_ratio
            return min_lr + (self.current_base_lr - min_lr) * cosine_decay
        
        # 阶段3: 最小学习率
        else:
            return self.current_base_lr * self.min_lr_ratio
    
    def restart_lr(self, current_step: int) -> float:
        """
        重启学习率（用于突破训练停滞）
        
        当训练集SSIM连续5k步无提升时触发，将学习率重启至初始值的50%，
        给模型"第二次机会"突破局部最优。
        
        Args:
            current_step: 当前训练步数
        
        Returns:
            重启后的学习率
        
        Validates: Requirements 8.4
        """
        # 重启学习率至初始值的50%
        self.current_base_lr = self.initial_lr * self.restart_ratio
        self.restart_step = current_step
        
        # 记录调整历史
        self.adjustment_history.append({
            'step': current_step,
            'type': 'restart',
            'new_base_lr': self.current_base_lr,
            'reason': 'Train SSIM停滞，重启学习率突破瓶颈'
        })
        
        # 返回当前学习率
        return self.get_lr(current_step)
    
    def decay_lr(self, current_step: int) -> float:
        """
        衰减学习率（用于减缓过拟合）
        
        当测试集SSIM下降时触发，将当前学习率降低至50%，
        减缓过拟合速度，给模型更多时间泛化。
        
        Args:
            current_step: 当前训练步数
        
        Returns:
            衰减后的学习率
        
        Validates: Requirements 8.5
        """
        # 衰减当前基础学习率至50%
        self.current_base_lr = self.current_base_lr * self.decay_ratio
        self.decay_step = current_step
        
        # 记录调整历史
        self.adjustment_history.append({
            'step': current_step,
            'type': 'decay',
            'new_base_lr': self.current_base_lr,
            'reason': 'Test SSIM下降，降低学习率减缓过拟合'
        })
        
        # 返回当前学习率
        return self.get_lr(current_step)
    
    def get_adjustment_history(self):
        """
        获取学习率调整历史
        
        Returns:
            调整历史列表，每个元素包含step, type, new_base_lr, reason
        """
        return self.adjustment_history
    
    def reset(self):
        """
        重置控制器到初始状态
        
        用于重新开始训练或实验
        """
        self.current_base_lr = self.initial_lr
        self.restart_step = None
        self.decay_step = None
        self.adjustment_history = []
    
    def __repr__(self):
        return (
            f"LearningRateController("
            f"initial_lr={self.initial_lr}, "
            f"current_base_lr={self.current_base_lr}, "
            f"warmup_steps={self.warmup_steps}, "
            f"cosine_period={self.cosine_period})"
        )

"""
训练监控器模块 - TrainingMonitor

实现Requirements 10.1-10.5:
- 每400步记录训练集SSIM
- 每1k步在10个测试样本上评估
- 记录每个测试样本的SSIM曲线
- 停滞检测（连续5k步无提升）
- 过拟合检测（连续5k步gap扩大）
- Early Stopping（train_ssim ≥ 0.90且gap < 0.05）
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class AdjustmentSuggestion:
    """调整建议数据类"""
    type: str  # 'lr_restart', 'lr_decay', 'regularization_increase', 'early_stop'
    reason: str
    suggested_value: Optional[float] = None


@dataclass
class TrainingMonitorConfig:
    """训练监控器配置"""
    train_log_interval: int = 400  # 训练集SSIM记录间隔
    test_eval_interval: int = 1000  # 测试集评估间隔
    stagnation_window: int = 5000  # 停滞检测窗口（步数）
    gap_expansion_window: int = 5000  # gap扩大检测窗口（步数）
    target_train_ssim: float = 0.90  # 目标训练集SSIM
    target_gap: float = 0.05  # 目标泛化gap
    test_samples: int = 10  # 测试样本数量


class TrainingMonitor:
    """
    训练监控与控制器
    
    核心功能:
    1. 记录训练集SSIM（每400步）
    2. 评估测试集SSIM（每1k步，10个样本）
    3. 记录每个测试样本的SSIM曲线
    4. 停滞检测（连续5k步train_ssim无提升）
    5. 过拟合检测（连续5k步gap扩大）
    6. Early Stopping（train_ssim ≥ 0.90且gap < 0.05）
    
    Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 6.2, 6.3
    """
    
    def __init__(self, config: Optional[TrainingMonitorConfig] = None):
        """
        初始化训练监控器
        
        Args:
            config: 监控器配置，默认使用TrainingMonitorConfig()
        """
        self.config = config or TrainingMonitorConfig()
        
        # 训练集SSIM历史: [(step, ssim), ...]
        self.train_ssim_history: List[Tuple[int, float]] = []
        
        # 测试集SSIM历史: [(step, ssim), ...]
        self.test_ssim_history: List[Tuple[int, float]] = []
        
        # 每个测试样本的SSIM曲线: {sample_id: [(step, ssim), ...]}
        self.per_sample_ssim: Dict[str, List[Tuple[int, float]]] = {}
        
        # 泛化gap历史: [(step, gap), ...]
        self.gap_history: List[Tuple[int, float]] = []
        
        # 最佳指标追踪
        self.best_train_ssim: float = 0.0
        self.best_train_ssim_step: int = 0
        self.best_test_ssim: float = 0.0
        self.best_test_ssim_step: int = 0
        self.best_gap: float = float('inf')
        self.best_gap_step: int = 0
        
        # 停滞检测状态
        self.last_improvement_step: int = 0
        self.stagnation_warning_issued: bool = False
        
        # 过拟合检测状态
        self.gap_expansion_warning_issued: bool = False
    
    def log_train_ssim(self, step: int, ssim: float) -> None:
        """
        记录训练集SSIM
        
        Args:
            step: 当前训练步数
            ssim: 训练集SSIM值
            
        Requirements: 10.1
        """
        self.train_ssim_history.append((step, ssim))
        
        # 更新最佳训练集SSIM
        if ssim > self.best_train_ssim:
            self.best_train_ssim = ssim
            self.best_train_ssim_step = step
            self.last_improvement_step = step
            self.stagnation_warning_issued = False  # 重置警告状态
    
    def log_test_ssim(
        self, 
        step: int, 
        ssim: float,
        per_sample_ssim: Optional[Dict[str, float]] = None
    ) -> None:
        """
        记录测试集SSIM
        
        Args:
            step: 当前训练步数
            ssim: 测试集平均SSIM值
            per_sample_ssim: 每个测试样本的SSIM值 {sample_id: ssim}
            
        Requirements: 10.2, 6.2, 6.3
        """
        self.test_ssim_history.append((step, ssim))
        
        # 更新最佳测试集SSIM
        if ssim > self.best_test_ssim:
            self.best_test_ssim = ssim
            self.best_test_ssim_step = step
        
        # 记录每个测试样本的SSIM曲线（Requirements 6.3）
        if per_sample_ssim is not None:
            for sample_id, sample_ssim in per_sample_ssim.items():
                if sample_id not in self.per_sample_ssim:
                    self.per_sample_ssim[sample_id] = []
                self.per_sample_ssim[sample_id].append((step, sample_ssim))
        
        # 计算并记录泛化gap
        if self.train_ssim_history:
            # 找到最近的训练集SSIM
            latest_train_ssim = self.train_ssim_history[-1][1]
            gap = abs(latest_train_ssim - ssim)
            self.gap_history.append((step, gap))
            
            # 更新最佳gap
            if gap < self.best_gap:
                self.best_gap = gap
                self.best_gap_step = step
                self.gap_expansion_warning_issued = False  # 重置警告状态
    
    def check_stagnation(self, current_step: int) -> Optional[str]:
        """
        检查训练集SSIM是否停滞
        
        停滞定义: 连续stagnation_window步无提升
        
        Args:
            current_step: 当前训练步数
            
        Returns:
            警告信息（如果停滞），否则返回None
            
        Requirements: 10.3
        """
        if not self.train_ssim_history:
            return None
        
        # 计算自上次提升以来的步数
        steps_since_improvement = current_step - self.last_improvement_step
        
        # 检查是否超过停滞窗口
        if steps_since_improvement >= self.config.stagnation_window:
            if not self.stagnation_warning_issued:
                self.stagnation_warning_issued = True
                warning = (
                    f"⚠️ 训练停滞检测: 连续 {steps_since_improvement} 步 "
                    f"Train SSIM 无提升（最佳: {self.best_train_ssim:.4f} @ step {self.best_train_ssim_step}）"
                )
                return warning
        
        return None
    
    def check_gap_expansion(self, current_step: int) -> bool:
        """
        检查泛化gap是否连续扩大
        
        gap扩大定义: 连续gap_expansion_window步内gap持续增大
        
        Args:
            current_step: 当前训练步数
            
        Returns:
            是否检测到gap扩大
            
        Requirements: 10.4
        """
        if len(self.gap_history) < 2:
            return False
        
        # 获取窗口内的gap历史
        window_gaps = [
            (step, gap) for step, gap in self.gap_history
            if current_step - step <= self.config.gap_expansion_window
        ]
        
        if len(window_gaps) < 2:
            return False
        
        # 检查gap是否持续增大（使用线性回归斜率）
        steps = np.array([step for step, _ in window_gaps])
        gaps = np.array([gap for _, gap in window_gaps])
        
        # 简单线性回归：slope = cov(x,y) / var(x)
        if len(steps) > 1:
            slope = np.cov(steps, gaps)[0, 1] / np.var(steps)
            
            # 如果斜率显著为正（gap增大），触发警告
            if slope > 0.001 and not self.gap_expansion_warning_issued:
                self.gap_expansion_warning_issued = True
                return True
        
        return False
    
    def should_early_stop(self) -> bool:
        """
        检查是否应该Early Stopping
        
        Early Stopping条件: train_ssim ≥ 0.90 且 gap < 0.05
        
        Returns:
            是否应该停止训练
            
        Requirements: 10.5
        """
        if not self.train_ssim_history or not self.gap_history:
            return False
        
        latest_train_ssim = self.train_ssim_history[-1][1]
        latest_gap = self.gap_history[-1][1]
        
        return (
            latest_train_ssim >= self.config.target_train_ssim and
            latest_gap < self.config.target_gap
        )
    
    def get_adjustment_suggestion(
        self, 
        current_step: int
    ) -> Optional[AdjustmentSuggestion]:
        """
        获取调整建议
        
        根据当前训练状态，返回调整建议：
        - 停滞 → 学习率重启
        - gap扩大 → 正则化增强
        - 达到目标 → Early Stopping
        
        Args:
            current_step: 当前训练步数
            
        Returns:
            AdjustmentSuggestion对象（如果有建议），否则返回None
        """
        # 优先级1: Early Stopping
        if self.should_early_stop():
            return AdjustmentSuggestion(
                type='early_stop',
                reason=f'达到目标: Train SSIM ≥ {self.config.target_train_ssim}, gap < {self.config.target_gap}'
            )
        
        # 优先级2: 停滞检测 → 学习率重启
        stagnation_warning = self.check_stagnation(current_step)
        if stagnation_warning is not None:
            return AdjustmentSuggestion(
                type='lr_restart',
                reason=stagnation_warning,
                suggested_value=0.5  # 重启至初始学习率的50%
            )
        
        # 优先级3: gap扩大 → 正则化增强
        if self.check_gap_expansion(current_step):
            return AdjustmentSuggestion(
                type='regularization_increase',
                reason=f'过拟合检测: 连续 {self.config.gap_expansion_window} 步 gap 扩大',
                suggested_value=0.05  # Dropout增加0.05
            )
        
        return None
    
    def get_summary(self) -> Dict[str, any]:
        """
        获取训练监控摘要
        
        Returns:
            包含所有关键指标的字典
        """
        summary = {
            'best_train_ssim': self.best_train_ssim,
            'best_train_ssim_step': self.best_train_ssim_step,
            'best_test_ssim': self.best_test_ssim,
            'best_test_ssim_step': self.best_test_ssim_step,
            'best_gap': self.best_gap,
            'best_gap_step': self.best_gap_step,
            'total_train_logs': len(self.train_ssim_history),
            'total_test_logs': len(self.test_ssim_history),
            'tracked_samples': len(self.per_sample_ssim),
        }
        
        # 添加最新指标
        if self.train_ssim_history:
            summary['latest_train_ssim'] = self.train_ssim_history[-1][1]
            summary['latest_train_step'] = self.train_ssim_history[-1][0]
        
        if self.test_ssim_history:
            summary['latest_test_ssim'] = self.test_ssim_history[-1][1]
            summary['latest_test_step'] = self.test_ssim_history[-1][0]
        
        if self.gap_history:
            summary['latest_gap'] = self.gap_history[-1][1]
        
        return summary
    
    def print_summary(self) -> None:
        """打印训练监控摘要到控制台"""
        summary = self.get_summary()
        
        print(f"\n{'='*60}")
        print("训练监控摘要")
        print(f"{'='*60}")
        print(f"最佳 Train SSIM: {summary['best_train_ssim']:.4f} @ step {summary['best_train_ssim_step']}")
        print(f"最佳 Test SSIM:  {summary['best_test_ssim']:.4f} @ step {summary['best_test_ssim_step']}")
        print(f"最佳 Gap:        {summary['best_gap']:.4f} @ step {summary['best_gap_step']}")
        
        if 'latest_train_ssim' in summary:
            print(f"\n最新 Train SSIM: {summary['latest_train_ssim']:.4f} @ step {summary['latest_train_step']}")
        if 'latest_test_ssim' in summary:
            print(f"最新 Test SSIM:  {summary['latest_test_ssim']:.4f} @ step {summary['latest_test_step']}")
        if 'latest_gap' in summary:
            print(f"最新 Gap:        {summary['latest_gap']:.4f}")
        
        print(f"\n记录统计:")
        print(f"  训练集记录数: {summary['total_train_logs']}")
        print(f"  测试集记录数: {summary['total_test_logs']}")
        print(f"  追踪样本数:   {summary['tracked_samples']}")
        print(f"{'='*60}\n")
    
    def get_per_sample_summary(self) -> Dict[str, Dict[str, float]]:
        """
        获取每个测试样本的SSIM统计
        
        Returns:
            {sample_id: {'best': float, 'latest': float, 'avg': float}}
            
        Requirements: 6.3
        """
        summary = {}
        
        for sample_id, history in self.per_sample_ssim.items():
            if not history:
                continue
            
            ssim_values = [ssim for _, ssim in history]
            summary[sample_id] = {
                'best': max(ssim_values),
                'latest': ssim_values[-1],
                'avg': np.mean(ssim_values),
                'std': np.std(ssim_values),
            }
        
        return summary
    
    def print_per_sample_summary(self) -> None:
        """打印每个测试样本的SSIM统计"""
        summary = self.get_per_sample_summary()
        
        if not summary:
            print("暂无测试样本数据")
            return
        
        print(f"\n{'='*60}")
        print("测试样本SSIM统计")
        print(f"{'='*60}")
        print(f"{'样本ID':<20} {'最佳':<10} {'最新':<10} {'平均':<10}")
        print(f"{'-'*60}")
        
        for sample_id, stats in sorted(summary.items()):
            print(f"{sample_id:<20} {stats['best']:<10.4f} {stats['latest']:<10.4f} {stats['avg']:<10.4f}")
        
        print(f"{'='*60}\n")
"""
数据错误处理器

验证输入数据的有效性：
- 地震数据：形状、数值范围、方差检查
- 速度模型：形状、物理合理性检查
"""

import torch
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """数据验证结果"""
    is_valid: bool
    error_message: Optional[str] = None
    warning_message: Optional[str] = None


class DataErrorHandler:
    """
    数据错误处理器
    
    功能：
    1. 验证地震数据有效性：形状、数值范围、方差
    2. 验证速度模型有效性：形状、物理合理的速度值
    """
    
    def __init__(
        self,
        # 地震数据配置
        expected_seis_shape: Tuple[int, int, int] = (5, 1000, 70),  # (震源, 时间, 接收器)
        seis_min_variance: float = 1e-6,  # 最小方差（避免全零输入）
        
        # 速度模型配置
        expected_vel_shapes: Tuple[Tuple[int, int, int], ...] = ((1, 70, 70), (1, 64, 64)),
        vel_min_value: float = 1000.0,  # 最小速度值 (m/s)
        vel_max_value: float = 6000.0,  # 最大速度值 (m/s)
        vel_min_variance: float = 1e-6,  # 最小方差（避免全常数）
    ):
        """
        初始化数据错误处理器
        
        Args:
            expected_seis_shape: 期望的地震数据形状（不含batch维度）
            seis_min_variance: 地震数据最小方差阈值
            expected_vel_shapes: 期望的速度模型形状列表（不含batch维度）
            vel_min_value: 速度模型最小物理值
            vel_max_value: 速度模型最大物理值
            vel_min_variance: 速度模型最小方差阈值
        """
        self.expected_seis_shape = expected_seis_shape
        self.seis_min_variance = seis_min_variance
        
        self.expected_vel_shapes = expected_vel_shapes
        self.vel_min_value = vel_min_value
        self.vel_max_value = vel_max_value
        self.vel_min_variance = vel_min_variance
        
        # 统计信息
        self.validation_count = 0
        self.error_count = 0
    
    def validate_seismic_data(self, seis: torch.Tensor) -> ValidationResult:
        """
        验证地震数据有效性
        
        检查项：
        1. 形状：(B, 5, 1000, 70)
        2. 数值范围：无NaN/Inf
        3. 方差：非零（避免全零输入）
        
        Args:
            seis: 地震数据张量，形状 (B, 5, 1000, 70)
        
        Returns:
            ValidationResult: 验证结果
        """
        self.validation_count += 1
        
        # 检查形状
        if len(seis.shape) != 4:
            self.error_count += 1
            return ValidationResult(
                is_valid=False,
                error_message=f"地震数据维度错误：期望4维，实际{len(seis.shape)}维"
            )
        
        batch_size = seis.shape[0]
        actual_shape = seis.shape[1:]
        
        if actual_shape != self.expected_seis_shape:
            self.error_count += 1
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"地震数据形状错误：\n"
                    f"  期望: (B, {self.expected_seis_shape[0]}, "
                    f"{self.expected_seis_shape[1]}, {self.expected_seis_shape[2]})\n"
                    f"  实际: ({batch_size}, {actual_shape[0]}, {actual_shape[1]}, {actual_shape[2]})"
                )
            )
        
        # 检查NaN/Inf
        if torch.isnan(seis).any():
            self.error_count += 1
            nan_count = torch.isnan(seis).sum().item()
            return ValidationResult(
                is_valid=False,
                error_message=f"地震数据包含{nan_count}个NaN值"
            )
        
        if torch.isinf(seis).any():
            self.error_count += 1
            inf_count = torch.isinf(seis).sum().item()
            return ValidationResult(
                is_valid=False,
                error_message=f"地震数据包含{inf_count}个Inf值"
            )
        
        # 检查方差（避免全零或全常数输入）
        variance = torch.var(seis).item()
        if variance < self.seis_min_variance:
            self.error_count += 1
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"地震数据方差过低：{variance:.2e} < {self.seis_min_variance:.2e}\n"
                    f"  可能是全零或全常数输入"
                )
            )
        
        # 检查数值范围（警告级别）
        min_val = seis.min().item()
        max_val = seis.max().item()
        mean_val = seis.mean().item()
        std_val = seis.std().item()
        
        warning_msg = None
        if abs(mean_val) > 100:
            warning_msg = (
                f"地震数据均值较大：{mean_val:.2f}\n"
                f"  建议检查是否需要归一化"
            )
        
        logger.debug(
            f"地震数据统计：\n"
            f"  形状: {seis.shape}\n"
            f"  范围: [{min_val:.4f}, {max_val:.4f}]\n"
            f"  均值: {mean_val:.4f}\n"
            f"  标准差: {std_val:.4f}\n"
            f"  方差: {variance:.4e}"
        )
        
        return ValidationResult(
            is_valid=True,
            warning_message=warning_msg
        )
    
    def validate_velocity_model(self, vel: torch.Tensor) -> ValidationResult:
        """
        验证速度模型有效性
        
        检查项：
        1. 形状：(B, 1, 70, 70) 或 (B, 1, 64, 64)
        2. 数值范围：无NaN/Inf
        3. 物理合理性：速度值在[1000, 6000] m/s范围内
        4. 方差：非零（避免全常数）
        
        Args:
            vel: 速度模型张量，形状 (B, 1, H, W)
        
        Returns:
            ValidationResult: 验证结果
        """
        self.validation_count += 1
        
        # 检查形状
        if len(vel.shape) != 4:
            self.error_count += 1
            return ValidationResult(
                is_valid=False,
                error_message=f"速度模型维度错误：期望4维，实际{len(vel.shape)}维"
            )
        
        batch_size = vel.shape[0]
        actual_shape = vel.shape[1:]
        
        # 检查是否匹配任一期望形状
        shape_valid = False
        for expected_shape in self.expected_vel_shapes:
            if actual_shape == expected_shape:
                shape_valid = True
                break
        
        if not shape_valid:
            self.error_count += 1
            expected_shapes_str = " 或 ".join([
                f"(B, {s[0]}, {s[1]}, {s[2]})" 
                for s in self.expected_vel_shapes
            ])
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"速度模型形状错误：\n"
                    f"  期望: {expected_shapes_str}\n"
                    f"  实际: ({batch_size}, {actual_shape[0]}, {actual_shape[1]}, {actual_shape[2]})"
                )
            )
        
        # 检查通道数
        if vel.shape[1] != 1:
            self.error_count += 1
            return ValidationResult(
                is_valid=False,
                error_message=f"速度模型通道数错误：期望1，实际{vel.shape[1]}"
            )
        
        # 检查NaN/Inf
        if torch.isnan(vel).any():
            self.error_count += 1
            nan_count = torch.isnan(vel).sum().item()
            return ValidationResult(
                is_valid=False,
                error_message=f"速度模型包含{nan_count}个NaN值"
            )
        
        if torch.isinf(vel).any():
            self.error_count += 1
            inf_count = torch.isinf(vel).sum().item()
            return ValidationResult(
                is_valid=False,
                error_message=f"速度模型包含{inf_count}个Inf值"
            )
        
        # 检查物理合理性
        min_val = vel.min().item()
        max_val = vel.max().item()
        
        if min_val < self.vel_min_value:
            self.error_count += 1
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"速度模型最小值过低：{min_val:.2f} < {self.vel_min_value:.2f} m/s\n"
                    f"  物理上不合理"
                )
            )
        
        if max_val > self.vel_max_value:
            self.error_count += 1
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"速度模型最大值过高：{max_val:.2f} > {self.vel_max_value:.2f} m/s\n"
                    f"  物理上不合理"
                )
            )
        
        # 检查方差（避免全常数）
        variance = torch.var(vel).item()
        if variance < self.vel_min_variance:
            self.error_count += 1
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"速度模型方差过低：{variance:.2e} < {self.vel_min_variance:.2e}\n"
                    f"  可能是全常数输入"
                )
            )
        
        # 统计信息
        mean_val = vel.mean().item()
        std_val = vel.std().item()
        
        logger.debug(
            f"速度模型统计：\n"
            f"  形状: {vel.shape}\n"
            f"  范围: [{min_val:.2f}, {max_val:.2f}] m/s\n"
            f"  均值: {mean_val:.2f} m/s\n"
            f"  标准差: {std_val:.2f} m/s\n"
            f"  方差: {variance:.4e}"
        )
        
        return ValidationResult(is_valid=True)
    
    def validate_batch(
        self, 
        seis: torch.Tensor, 
        vel: torch.Tensor
    ) -> Tuple[ValidationResult, ValidationResult]:
        """
        验证一个batch的地震数据和速度模型
        
        Args:
            seis: 地震数据张量
            vel: 速度模型张量
        
        Returns:
            (seis_result, vel_result): 两个验证结果
        """
        seis_result = self.validate_seismic_data(seis)
        vel_result = self.validate_velocity_model(vel)
        
        # 检查batch size是否匹配
        if seis.shape[0] != vel.shape[0]:
            logger.warning(
                f"Batch size不匹配：地震数据{seis.shape[0]}，速度模型{vel.shape[0]}"
            )
        
        return seis_result, vel_result
    
    def get_statistics(self) -> str:
        """
        获取验证统计信息
        
        Returns:
            统计信息的格式化字符串
        """
        if self.validation_count == 0:
            return "无验证记录"
        
        error_rate = self.error_count / self.validation_count * 100
        
        return (
            f"数据验证统计：\n"
            f"  总验证次数: {self.validation_count}\n"
            f"  错误次数: {self.error_count}\n"
            f"  错误率: {error_rate:.2f}%"
        )
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.validation_count = 0
        self.error_count = 0
        logger.info("数据验证统计已重置")


def validate_numpy_data(
    seis_path: str,
    vel_path: str,
    handler: Optional[DataErrorHandler] = None
) -> Tuple[ValidationResult, ValidationResult]:
    """
    验证从文件加载的numpy数据
    
    Args:
        seis_path: 地震数据文件路径
        vel_path: 速度模型文件路径
        handler: 数据错误处理器（可选，默认创建新实例）
    
    Returns:
        (seis_result, vel_result): 两个验证结果
    """
    if handler is None:
        handler = DataErrorHandler()
    
    try:
        # 加载数据
        seis_np = np.load(seis_path)
        vel_np = np.load(vel_path)
        
        # 转换为torch张量并添加batch维度
        seis = torch.from_numpy(seis_np).float().unsqueeze(0)
        vel = torch.from_numpy(vel_np).float().unsqueeze(0)
        
        # 验证
        return handler.validate_batch(seis, vel)
        
    except Exception as e:
        error_msg = f"加载数据失败：{str(e)}"
        logger.error(error_msg)
        return (
            ValidationResult(is_valid=False, error_message=error_msg),
            ValidationResult(is_valid=False, error_message=error_msg)
        )
"""
训练错误处理器

处理训练过程中的各种异常情况：
- NaN损失：回滚checkpoint，降低学习率
- SSIM崩溃：回滚checkpoint，禁用数据增强
- 显存溢出：减小batch size，启用梯度累积
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """错误类型枚举"""
    NAN_LOSS = "nan_loss"
    SSIM_COLLAPSE = "ssim_collapse"
    MEMORY_OVERFLOW = "memory_overflow"
    UNKNOWN = "unknown"


@dataclass
class ErrorAction:
    """错误处理动作"""
    action_type: str  # 'rollback', 'adjust_lr', 'disable_augmentation', 'reduce_batch_size'
    reason: str
    rollback_to_step: Optional[int] = None
    new_lr_ratio: Optional[float] = None  # 相对于当前学习率的比例
    disable_augmentation: bool = False
    new_batch_size: Optional[int] = None
    enable_gradient_accumulation: bool = False
    accumulation_steps: int = 1


class TrainingErrorHandler:
    """
    训练错误处理器
    
    功能：
    1. NaN损失处理：回滚到最近checkpoint，降低学习率50%
    2. SSIM崩溃处理：回滚到最近checkpoint，禁用所有数据增强
    3. 显存溢出处理：减小batch_size，启用梯度累积
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        ssim_collapse_threshold: float = 0.5,  # SSIM低于此值视为崩溃
        lr_decay_ratio: float = 0.5,  # NaN时学习率衰减比例
        batch_size_reduction_ratio: float = 0.5,  # 显存溢出时batch size缩减比例
    ):
        """
        初始化训练错误处理器
        
        Args:
            checkpoint_dir: checkpoint保存目录
            ssim_collapse_threshold: SSIM崩溃阈值（低于此值触发处理）
            lr_decay_ratio: NaN损失时学习率衰减比例
            batch_size_reduction_ratio: 显存溢出时batch size缩减比例
        """
        self.checkpoint_dir = checkpoint_dir
        self.ssim_collapse_threshold = ssim_collapse_threshold
        self.lr_decay_ratio = lr_decay_ratio
        self.batch_size_reduction_ratio = batch_size_reduction_ratio
        
        # 记录最近的有效checkpoint
        self.last_valid_checkpoint: Optional[Dict[str, Any]] = None
        self.last_valid_step: Optional[int] = None
        
        # 错误历史记录
        self.error_history: list[tuple[int, ErrorType, str]] = []
    
    def handle_nan_loss(self, step: int, current_lr: float) -> ErrorAction:
        """
        处理NaN损失
        
        策略：
        1. 回滚到最近的有效checkpoint
        2. 降低学习率至当前值的50%
        3. 记录错误日志
        
        Args:
            step: 当前训练步数
            current_lr: 当前学习率
        
        Returns:
            ErrorAction: 包含回滚步数和新学习率的动作
        """
        logger.error(f"[步数 {step}] 检测到NaN损失！")
        
        # 记录错误
        self.error_history.append((step, ErrorType.NAN_LOSS, "Loss变为NaN"))
        
        # 计算新学习率
        new_lr_ratio = self.lr_decay_ratio
        
        # 确定回滚目标
        rollback_step = self.last_valid_step if self.last_valid_step is not None else max(0, step - 1000)
        
        logger.warning(
            f"执行NaN损失处理：\n"
            f"  - 回滚到步数: {rollback_step}\n"
            f"  - 学习率衰减至: {new_lr_ratio * 100:.0f}%\n"
            f"  - 建议检查数据是否存在异常值"
        )
        
        return ErrorAction(
            action_type="rollback_and_adjust_lr",
            reason=f"NaN损失检测（步数{step}）",
            rollback_to_step=rollback_step,
            new_lr_ratio=new_lr_ratio,
            disable_augmentation=False
        )
    
    def handle_ssim_collapse(
        self, 
        step: int, 
        current_ssim: float,
        previous_ssim: Optional[float] = None
    ) -> ErrorAction:
        """
        处理SSIM崩溃
        
        策略：
        1. 回滚到最近的有效checkpoint
        2. 禁用所有数据增强（可能是增强过强导致）
        3. 记录错误日志
        
        Args:
            step: 当前训练步数
            current_ssim: 当前SSIM值
            previous_ssim: 之前的SSIM值（用于判断是否突然下降）
        
        Returns:
            ErrorAction: 包含回滚步数和禁用增强标志的动作
        """
        logger.error(
            f"[步数 {step}] 检测到SSIM崩溃！\n"
            f"  - 当前SSIM: {current_ssim:.4f}\n"
            f"  - 崩溃阈值: {self.ssim_collapse_threshold:.4f}"
        )
        
        if previous_ssim is not None:
            logger.error(f"  - 之前SSIM: {previous_ssim:.4f}")
            logger.error(f"  - 下降幅度: {previous_ssim - current_ssim:.4f}")
        
        # 记录错误
        reason = f"SSIM崩溃至{current_ssim:.4f}"
        if previous_ssim is not None:
            reason += f"（从{previous_ssim:.4f}下降）"
        self.error_history.append((step, ErrorType.SSIM_COLLAPSE, reason))
        
        # 确定回滚目标
        rollback_step = self.last_valid_step if self.last_valid_step is not None else max(0, step - 1000)
        
        logger.warning(
            f"执行SSIM崩溃处理：\n"
            f"  - 回滚到步数: {rollback_step}\n"
            f"  - 禁用所有数据增强\n"
            f"  - 建议检查数据增强配置是否过强"
        )
        
        return ErrorAction(
            action_type="rollback_and_disable_augmentation",
            reason=reason,
            rollback_to_step=rollback_step,
            new_lr_ratio=None,
            disable_augmentation=True
        )
    
    def handle_memory_overflow(
        self, 
        step: int, 
        current_batch_size: int
    ) -> ErrorAction:
        """
        处理显存溢出
        
        策略：
        1. 减小batch_size至当前值的50%
        2. 启用梯度累积以保持等效batch size
        3. 记录错误日志
        
        Args:
            step: 当前训练步数
            current_batch_size: 当前batch size
        
        Returns:
            ErrorAction: 包含新batch size和梯度累积配置的动作
        """
        logger.error(f"[步数 {step}] 检测到显存溢出！")
        
        # 记录错误
        self.error_history.append((step, ErrorType.MEMORY_OVERFLOW, "CUDA out of memory"))
        
        # 计算新batch size
        new_batch_size = max(1, int(current_batch_size * self.batch_size_reduction_ratio))
        
        # 计算梯度累积步数以保持等效batch size
        accumulation_steps = max(1, current_batch_size // new_batch_size)
        
        logger.warning(
            f"执行显存溢出处理：\n"
            f"  - 原batch size: {current_batch_size}\n"
            f"  - 新batch size: {new_batch_size}\n"
            f"  - 梯度累积步数: {accumulation_steps}\n"
            f"  - 等效batch size: {new_batch_size * accumulation_steps}"
        )
        
        return ErrorAction(
            action_type="reduce_batch_size",
            reason=f"显存溢出（步数{step}）",
            rollback_to_step=None,
            new_lr_ratio=None,
            disable_augmentation=False,
            new_batch_size=new_batch_size,
            enable_gradient_accumulation=True,
            accumulation_steps=accumulation_steps
        )
    
    def save_checkpoint(
        self, 
        step: int, 
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        ssim: float,
        **kwargs
    ) -> None:
        """
        保存checkpoint作为有效恢复点
        
        Args:
            step: 当前步数
            model: 模型
            optimizer: 优化器
            ssim: 当前SSIM值
            **kwargs: 其他需要保存的状态
        """
        self.last_valid_checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ssim': ssim,
            **kwargs
        }
        self.last_valid_step = step
        
        logger.info(f"保存有效checkpoint（步数{step}, SSIM={ssim:.4f}）")
    
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> Optional[Dict[str, Any]]:
        """
        加载最近的有效checkpoint
        
        Args:
            model: 模型
            optimizer: 优化器
        
        Returns:
            checkpoint字典，如果没有有效checkpoint则返回None
        """
        if self.last_valid_checkpoint is None:
            logger.warning("没有可用的有效checkpoint")
            return None
        
        model.load_state_dict(self.last_valid_checkpoint['model_state_dict'])
        optimizer.load_state_dict(self.last_valid_checkpoint['optimizer_state_dict'])
        
        logger.info(
            f"加载checkpoint成功（步数{self.last_valid_checkpoint['step']}, "
            f"SSIM={self.last_valid_checkpoint['ssim']:.4f}）"
        )
        
        return self.last_valid_checkpoint
    
    def check_loss_validity(self, loss: torch.Tensor) -> bool:
        """
        检查损失值是否有效（非NaN/Inf）
        
        Args:
            loss: 损失张量
        
        Returns:
            True如果有效，False如果是NaN或Inf
        """
        if torch.isnan(loss).any():
            return False
        if torch.isinf(loss).any():
            return False
        return True
    
    def check_ssim_collapse(
        self, 
        current_ssim: float,
        previous_ssim: Optional[float] = None,
        sudden_drop_threshold: float = 0.2
    ) -> bool:
        """
        检查SSIM是否崩溃
        
        判断标准：
        1. SSIM低于崩溃阈值（默认0.5）
        2. 或者SSIM突然下降超过阈值（默认0.2）
        
        Args:
            current_ssim: 当前SSIM值
            previous_ssim: 之前的SSIM值
            sudden_drop_threshold: 突然下降阈值
        
        Returns:
            True如果检测到崩溃，False否则
        """
        # 检查绝对值
        if current_ssim < self.ssim_collapse_threshold:
            return True
        
        # 检查突然下降
        if previous_ssim is not None:
            if previous_ssim - current_ssim > sudden_drop_threshold:
                return True
        
        return False
    
    def get_error_summary(self) -> str:
        """
        获取错误历史摘要
        
        Returns:
            错误历史的格式化字符串
        """
        if not self.error_history:
            return "无错误记录"
        
        summary = "错误历史记录：\n"
        for step, error_type, reason in self.error_history:
            summary += f"  - 步数{step}: {error_type.value} - {reason}\n"
        
        return summary


# ============================================================
# 自适应损失调度器（Requirements 5.1-5.5）
# ============================================================

@dataclass
class LossWeights:
    """
    损失权重配置
    
    Attributes:
        l1_weight: L1损失权重
        ssim_weight: SSIM损失权重
    """
    l1_weight: float
    ssim_weight: float


class AdaptiveLossScheduler:
    """
    自适应损失函数调度器
    
    阶段性调整:
    - 步数 < 30k: 纯L1 Loss（快速学习像素对应）
    - 步数 30k-60k: L1 + SSIM（权重0.3→0.7线性增长）
    - 步数 > 60k: L1 + 0.7*SSIM
    
    动态调整:
    - Train SSIM停滞: 临时提升SSIM权重至0.9
    - Test SSIM下降: 降低SSIM权重至0.5
    """
    
    def __init__(
        self,
        phase1_end: int = 30000,
        phase2_end: int = 60000,
        ssim_weight_min: float = 0.3,
        ssim_weight_max: float = 0.7,
        ssim_weight_boost: float = 0.9,
        ssim_weight_reduce: float = 0.5,
        stagnation_threshold: int = 5000  # 连续5k步无提升视为停滞
    ):
        """
        初始化自适应损失调度器
        
        Args:
            phase1_end: 阶段1结束步数（纯L1阶段）
            phase2_end: 阶段2结束步数（SSIM权重增长阶段）
            ssim_weight_min: SSIM权重最小值（阶段2开始）
            ssim_weight_max: SSIM权重最大值（阶段2结束及之后）
            ssim_weight_boost: SSIM权重提升值（停滞时）
            ssim_weight_reduce: SSIM权重降低值（测试集下降时）
            stagnation_threshold: 停滞检测窗口（步数）
        """
        self.phase1_end = phase1_end
        self.phase2_end = phase2_end
        self.ssim_weight_min = ssim_weight_min
        self.ssim_weight_max = ssim_weight_max
        self.ssim_weight_boost = ssim_weight_boost
        self.ssim_weight_reduce = ssim_weight_reduce
        self.stagnation_threshold = stagnation_threshold
        
        # 状态追踪
        self.is_boosted = False  # 是否处于boost状态
        self.is_reduced = False  # 是否处于reduce状态
    
    def get_loss_weights(
        self, 
        step: int, 
        train_ssim_history: Optional[List[float]] = None,
        test_ssim_history: Optional[List[float]] = None
    ) -> LossWeights:
        """
        返回当前步数应使用的损失权重
        
        Args:
            step: 当前训练步数
            train_ssim_history: 训练集SSIM历史（用于停滞检测）
            test_ssim_history: 测试集SSIM历史（用于下降检测）
        
        Returns:
            LossWeights: 包含l1_weight和ssim_weight的配置
        """
        # 基础权重计算（根据阶段）
        if step < self.phase1_end:
            # 阶段1: 纯L1
            base_ssim_weight = 0.0
        elif step < self.phase2_end:
            # 阶段2: SSIM权重线性增长
            progress = (step - self.phase1_end) / (self.phase2_end - self.phase1_end)
            base_ssim_weight = self.ssim_weight_min + progress * (self.ssim_weight_max - self.ssim_weight_min)
        else:
            # 阶段3: 固定权重
            base_ssim_weight = self.ssim_weight_max
        
        # 动态调整检测
        ssim_weight = base_ssim_weight
        
        # L1权重始终为1.0
        l1_weight = 1.0
        
        return LossWeights(l1_weight=l1_weight, ssim_weight=ssim_weight)
    
    def _detect_stagnation(self, ssim_history: List[float], window: int = 5000) -> bool:
        """
        检测SSIM是否停滞（连续window步无提升）
        
        Args:
            ssim_history: SSIM历史记录（按步数顺序）
            window: 检测窗口大小
        
        Returns:
            bool: True表示停滞，False表示正常
        """
        if len(ssim_history) < 2:
            return False
        
        # 取最近的记录
        recent_history = ssim_history[-min(len(ssim_history), window):]
        
        if len(recent_history) < 2:
            return False
        
        # 计算窗口内的最大值和当前值
        max_ssim = max(recent_history)
        current_ssim = recent_history[-1]
        
        # 停滞定义：当前值与窗口内最大值的差距小于阈值
        improvement = max_ssim - current_ssim
        
        # 如果当前值接近窗口内最大值（差距<0.001），检查是否真的在提升
        if improvement < 0.001:
            # 进一步检查：比较窗口前半段和后半段的平均值
            mid_point = len(recent_history) // 2
            first_half_avg = sum(recent_history[:mid_point]) / mid_point if mid_point > 0 else 0
            second_half_avg = sum(recent_history[mid_point:]) / (len(recent_history) - mid_point)
            
            # 如果后半段平均值没有明显高于前半段，视为停滞
            return second_half_avg - first_half_avg < 0.001
        
        # 如果当前值明显低于窗口内最大值，也视为停滞（可能在下降）
        return improvement > 0.01
    
    def _detect_decline(self, ssim_history: List[float], window: int = 1000) -> bool:
        """
        检测SSIM是否下降
        
        Args:
            ssim_history: SSIM历史记录（按步数顺序）
            window: 检测窗口大小
        
        Returns:
            bool: True表示下降，False表示正常
        """
        if len(ssim_history) < 2:
            return False
        
        # 取最近的记录
        recent_history = ssim_history[-min(len(ssim_history), window):]
        
        if len(recent_history) < 2:
            return False
        
        # 比较窗口开始和结束的SSIM
        start_ssim = recent_history[0]
        end_ssim = recent_history[-1]
        
        # 下降定义：结束值比开始值低超过0.02
        decline = start_ssim - end_ssim
        
        return decline > 0.02
