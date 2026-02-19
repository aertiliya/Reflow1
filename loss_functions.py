"""
高级损失函数模块
包含：Edge Loss (基于Pidinet的边界检测)、AdaGN条件注入等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 方案三：Edge Loss - 基于Pidinet的边界锐化损失
# ============================================================================

class PDC(nn.Module):
    """
    Pixel Difference Convolution (像素差分卷积)
    来自《基于像素差分神经网络的断层识别方法》
    用于提取边界特征
    
    改进：支持 dilation（膨胀卷积）实现多尺度边缘检测
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # 差分卷积核：计算相邻像素的差异
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False)
        
        # 初始化为差分算子（Sobel-like）
        self._init_pdc_kernel(in_channels, out_channels, kernel_size)
    
    def _init_pdc_kernel(self, in_channels, out_channels, kernel_size):
        """初始化为像素差分算子（仅支持 kernel_size=3）"""
        with torch.no_grad():
            # 创建Sobel-like差分核
            if kernel_size == 3:
                # 水平差分
                kernel_h = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
                # 竖直差分
                kernel_v = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
                
                # 分配给输出通道
                # weight shape: (out_channels, in_channels, kernel_size, kernel_size)
                for i in range(out_channels):
                    kernel = kernel_h if i % 2 == 0 else kernel_v
                    # 为每个输入通道复制相同的核
                    for j in range(in_channels):
                        self.conv.weight[i, j, :, :] = kernel / (in_channels ** 0.5)
    
    def forward(self, x):
        return self.conv(x)
# PDC1D - 1D版本的像素差分卷积
# ============================================================================

class PDC1D(nn.Module):
    """
    1D版本的 Pixel Difference Convolution
    用于1D信号的边缘检测（断层边界识别）
    
    原理：
    - 使用1D差分算子（类似Sobel的1D版本）
    - 支持 dilation 实现多尺度边缘检测
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        padding = dilation * (kernel_size - 1) // 2
        
        # 1D差分卷积
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=padding, dilation=dilation, bias=False)
        
        # 初始化为1D差分算子
        self._init_pdc_kernel(in_channels, out_channels, kernel_size)
    
    def _init_pdc_kernel(self, in_channels, out_channels, kernel_size):
        """初始化为1D差分算子"""
        with torch.no_grad():
            if kernel_size == 3:
                # 1D差分核：[-1, 0, 1] 或 [1, -2, 1]（二阶导数）
                # 使用一阶导数（梯度）
                kernel = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
                
                # 分配给所有输出通道
                for i in range(out_channels):
                    for j in range(in_channels):
                        self.conv.weight[i, j, :] = kernel / (in_channels ** 0.5)
    
    def forward(self, x):
        """
        x: (B, C, L) - 1D信号
        返回: (B, out_channels, L) - 边缘特征
        """
        return torch.abs(self.conv(x))  # 取绝对值，获得边缘强度
class PDC1D(nn.Module):
    """
    1D版本的 Pixel Difference Convolution
    用于1D信号的边缘检测（断层边界识别）
    
    原理：
    - 使用1D差分算子（类似Sobel的1D版本）
    - 支持 dilation 实现多尺度边缘检测
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        padding = dilation * (kernel_size - 1) // 2
        
        # 1D差分卷积
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=padding, dilation=dilation, bias=False)
        
        # 初始化为1D差分算子
        self._init_pdc_kernel(in_channels, out_channels, kernel_size)
    
    def _init_pdc_kernel(self, in_channels, out_channels, kernel_size):
        """初始化为1D差分算子"""
        with torch.no_grad():
            if kernel_size == 3:
                # 1D差分核：[-1, 0, 1] 或 [1, -2, 1]（二阶导数）
                # 使用一阶导数（梯度）
                kernel = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
                
                # 分配给所有输出通道
                for i in range(out_channels):
                    for j in range(in_channels):
                        self.conv.weight[i, j, :] = kernel / (in_channels ** 0.5)
    
    def forward(self, x):
        """
        x: (B, C, L) - 1D信号
        返回: (B, out_channels, L) - 边缘特征
        """
        return torch.abs(self.conv(x))  # 取绝对值，获得边缘强度

class EdgeDetector(nn.Module):
    """
    边界检测器：使用多尺度PDC提取边界特征
    
    改进：使用不同的 dilation 实现真正的多尺度
    - dilation=1: 捕捉细粒度边缘（3x3感受野）
    - dilation=2: 捕捉中等粗细边缘（5x5感受野）
    - dilation=3: 捕捉粗粒度边缘（7x7感受野）
    """
    def __init__(self, in_channels=1, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        
        # 多尺度边界检测：使用不同的 dilation
        self.pdc_layers = nn.ModuleList([
            PDC(in_channels, 2, kernel_size=3, padding=1+i, dilation=1+i) 
            for i in range(num_scales)
        ])
        
        # 融合层：输入是 num_scales 个通道（每个尺度一个梯度幅度）
        self.fusion = nn.Conv2d(num_scales, 1, kernel_size=1)
    
    def forward(self, x):
        """
        输入: (B, C, H, W)
        输出: (B, 1, H, W) - 边界图
        """
        edges = []
        for pdc in self.pdc_layers:
            edge = pdc(x)  # (B, 2, H, W) - 水平和竖直方向
            # 计算梯度幅度：sqrt(Gx^2 + Gy^2)
            edge_magnitude = torch.sqrt(edge[:, 0:1] ** 2 + edge[:, 1:2] ** 2 + 1e-8)  # (B, 1, H, W)
            edges.append(edge_magnitude)
        
        # 融合多尺度边界：(B, num_scales, H, W)
        edges = torch.cat(edges, dim=1)
        edge_map = self.fusion(edges)  # (B, 1, H, W)
        edge_map = torch.sigmoid(edge_map)  # 归一化到[0,1]
        
        return edge_map


class EdgeLoss(nn.Module):
    """
    🔥 纯净版 Edge Loss：只计算边缘差异，不搞锐度惩罚
    
    原理：
    1. 用多尺度 PDC 提取预测和真值的边界
    2. 计算边界图之间的 L1 距离
    3. 冻结边缘检测器参数（只用作"尺子"）
    """
    def __init__(self, num_scales=3):
        super().__init__()
        self.edge_detector = EdgeDetector(in_channels=1, num_scales=num_scales)
        
        # 🔥 冻结边缘检测器的参数：我们不需要训练它，只需要它作为尺子
        for param in self.edge_detector.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        """
        pred: (B, 1, H, W) - 预测的速度模型
        target: (B, 1, H, W) - 真值速度模型
        
        返回: edge_loss (标量)
        """
        # 提取边缘
        pred_edges = self.edge_detector(pred)
        target_edges = self.edge_detector(target)
        
        # 计算边缘图之间的 L1 距离（简单有效）
        return F.l1_loss(pred_edges, target_edges)



# ============================================================================
# 组合损失函数
# ============================================================================

class TVLoss(nn.Module):
    """
    🔥 全变分损失 (Total Variation Loss)
    
    功能：平滑图像，消除高频噪点（雪花/条纹）
    原理：惩罚相邻像素之间的差异，强迫模型生成平滑的块状结构
    
    物理意义：地质分层应该是连续的，不应该有剧烈的高频振荡
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        返回: TV loss (标量)
        """
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        
        # 计算有效像素数
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        
        # 🔥 兵圣修复第二步：改用 .mean() 替代 .sum()，防止像素累加爆炸
        # 原来：.sum() 对 64x64=4096 个像素求和，稍微大一点就炸
        # 现在：.mean() 自动归一化，数值稳定
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).mean()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).mean()
        
        return self.weight * (h_tv + w_tv)
    
    def _tensor_size(self, t):
        return t.size(1) * t.size(2) * t.size(3)


class CombinedLoss(nn.Module):
    """
    组合损失：L1 + SSIM + Edge Loss
    """
    def __init__(self, l1_weight=1.0, ssim_weight=0.1, edge_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        
        self.l1_loss = nn.L1Loss()
        self.edge_loss = EdgeLoss(weight=edge_weight, num_scales=3)
    
    def forward(self, pred, target):
        """
        pred: (B, 1, H, W)
        target: (B, 1, H, W)
        """
        # L1损失
        l1 = self.l1_loss(pred, target)
        
        # SSIM损失（需要外部计算或使用pytorch-msssim）
        # 这里简化为L2损失
        l2 = F.mse_loss(pred, target)
        
        # 边界损失
        edge = self.edge_loss(pred, target)
        
        total_loss = self.l1_weight * l1 + self.ssim_weight * l2 + edge
        
        return total_loss, {
            'l1': l1.item(),
            'l2': l2.item(),
            'edge': edge.item()
        }


class SSIMLoss(nn.Module):
    """
    SSIM Loss：直接优化结构相似性
    
    原理：SSIM越大越好（范围[0,1]），Loss = 1 - SSIM
    """
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
    
    def _gaussian(self, window_size, sigma=1.5):
        import numpy as np
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2 / float(2*sigma**2)) 
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        """
        img1, img2: (B, C, H, W) 范围[-1, 1]或[0, 1]
        返回: SSIM Loss (标量)
        """
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        
        # 计算SSIM
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            ssim_val = ssim_map.mean()
        else:
            ssim_val = ssim_map.mean(1).mean(1).mean(1)
        
        # 返回Loss（1 - SSIM）
        return 1 - ssim_val
