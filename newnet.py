import math
from functools import partial
import torch
from einops import rearrange, reduce, repeat
from torch import nn, einsum
import torch.nn.functional as F


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# ==================== 基础模块（优化版） ====================

class WeightStandardizedConv2d(nn.Conv2d):
    """权重标准化卷积 - 训练更稳定"""

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    """通道层归一化"""

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ==================== 核心改进：自适应地震特征融合 ====================

class AdaptiveSeisFusion(nn.Module):
    """
    自适应地震特征融合模块 - 替代简单的 WithoutAttention

    特点：
    1. 空间自适应调制 (Spatial Feature Modulation)
    2. 通道注意力校准
    3. 门控机制控制融合强度
    """

    def __init__(self, dim, seis_dim, reduction=8):
        super().__init__()
        self.dim = dim
        self.seis_dim = seis_dim

        # 地震特征投影
        self.seis_proj = nn.Sequential(
            nn.Conv2d(seis_dim, dim, 1),
            LayerNorm(dim),
            nn.SiLU()
        )

        # 空间调制：生成空间注意力图
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim // reduction, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim // reduction, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # 通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim // reduction, 1),
            nn.SiLU(),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1),
            LayerNorm(dim),
            nn.SiLU()
        )

        # 可学习的融合比例
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, seis):
        """
        Args:
            x: (B, dim, H, W) - 主特征
            seis: (B, seis_dim, H, W) - 地震特征（已对齐空间尺寸）
        Returns:
            (B, dim, H, W) - 融合后的特征
        """
        # 投影地震特征
        seis_proj = self.seis_proj(seis)

        # 拼接生成注意力图
        concat_feat = torch.cat([x, seis_proj], dim=1)

        # 空间注意力
        spatial_weight = self.spatial_gate(concat_feat)  # (B, 1, H, W)

        # 通道注意力
        channel_weight = self.channel_attn(concat_feat)  # (B, dim, 1, 1)

        # 调制地震特征
        modulated_seis = seis_proj * spatial_weight * channel_weight

        # 门控融合
        gate = torch.sigmoid(self.alpha)
        fused = torch.cat([x * (1 - gate), modulated_seis * gate], dim=1)

        return self.fusion(fused)


class CrossScaleSeisInjector(nn.Module):
    """
    跨尺度地震特征注入器 - 替代 ConvSeisAligner

    使用特征金字塔思想，在不同层级注入不同语义的地震信息
    """

    def __init__(self, base_seis_dim=64, target_dims=[64, 128, 256, 512]):
        super().__init__()
        self.base_seis_dim = base_seis_dim
        self.target_dims = target_dims

        # 多尺度投影头
        self.proj_heads = nn.ModuleList()
        for target_dim in target_dims:
            self.proj_heads.append(nn.Sequential(
                nn.Conv2d(base_seis_dim, target_dim, 3, padding=1),
                LayerNorm(target_dim),
                nn.SiLU(),
                # 下采样到对应分辨率的操作在外部通过 interpolate 完成
                nn.Conv2d(target_dim, target_dim, 1),
            ))

        # 可学习的层级权重
        self.level_weights = nn.Parameter(torch.ones(len(target_dims)))

    def forward(self, seis_feat, target_size, level_idx):
        """
        Args:
            seis_feat: (B, base_seis_dim, H0, W0) 原始地震特征
            target_size: (H, W) 目标分辨率
            level_idx: 当前层级索引
        """
        # 调整空间尺寸
        if seis_feat.shape[-2:] != target_size:
            seis_feat = F.interpolate(
                seis_feat, size=target_size,
                mode='bilinear', align_corners=False
            )

        # 应用对应层级的投影
        proj = self.proj_heads[level_idx](seis_feat)

        # 应用层级权重
        weight = F.softmax(self.level_weights, dim=0)[level_idx]
        return proj * weight


# ==================== ResNet 模块（优化版） ====================

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, classes_emb_dim=None, groups=8):
        super().__init__()
        cond_dim = 0
        if exists(time_emb_dim):
            cond_dim += int(time_emb_dim)
        if exists(classes_emb_dim):
            cond_dim += int(classes_emb_dim)

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim_out * 2)
        ) if cond_dim > 0 else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, class_emb=None):
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1 1')
            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


# ==================== 改进的 U-Net 架构 ====================

class ImprovedUnet(nn.Module):
    def __init__(
            self,
            dim=64,
            num_classes=None,
            cond_drop_prob=0.1,  # 降低 CFG dropout，小数据集不宜过高
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=1,
            seis_channels=64,
            resnet_block_groups=8,
            learned_variance=False,
            use_fusion_gate=True,  # 是否使用门控融合
    ):
        super().__init__()

        self.cond_drop_prob = cond_drop_prob
        self.channels = channels
        self.dim = dim

        init_dim = default(init_dim, dim)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # ========== 输入处理 ==========
        # 使用重叠卷积减少边界信息损失：70x70 -> 64x64
        self.input_resize = nn.Sequential(
            nn.Conv2d(channels, init_dim, kernel_size=7, stride=1, padding=3),
            LayerNorm(init_dim),
            nn.SiLU(),
            nn.Conv2d(init_dim, init_dim, kernel_size=3, stride=1, padding=0),  # 70->68
            nn.Conv2d(init_dim, init_dim, kernel_size=3, stride=1, padding=0),  # 68->66
            nn.Conv2d(init_dim, init_dim, kernel_size=3, stride=1, padding=0),  # 66->64
        )

        # ========== 时间/条件嵌入 ==========
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # 类别嵌入（简化版）
        self.classes_emb = None
        self.classes_mlp = None
        if num_classes is not None and num_classes > 0:
            self.classes_emb = nn.Embedding(num_classes, dim)
            self.null_classes_emb = nn.Parameter(torch.randn(dim))
            self.classes_mlp = nn.Sequential(
                nn.Linear(dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim)
            )

        # ========== 地震特征处理（关键改进） ==========
        # 假设 ImprovedSeisEncoder 输出 (B, 64, 70, 70)
        # 先对齐到 64x64，然后使用跨尺度注入器
        self.seis_input_proj = nn.Sequential(
            nn.Conv2d(seis_channels, seis_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(seis_channels, seis_channels, 3, padding=0),  # 70->68
            nn.Conv2d(seis_channels, seis_channels, 3, padding=0),  # 68->66
            nn.Conv2d(seis_channels, seis_channels, 3, padding=0),  # 66->64
        )

        # 跨尺度注入器
        self.seis_injector = CrossScaleSeisInjector(
            base_seis_dim=seis_channels,
            target_dims=dims[:-1]  # 每个编码器层的维度
        )

        # 自适应融合模块（每个层级一个）
        self.fusion_modules = nn.ModuleList([
            AdaptiveSeisFusion(dim_in, dim_in)
            for dim_in, _ in in_out
        ])

        # 解码器融合模块
        self.dec_fusion_modules = nn.ModuleList([
            AdaptiveSeisFusion(dim_out, dim_out)
            for _, dim_out in reversed(in_out)
        ])

        # ========== U-Net 主体 ==========
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # 编码器
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx == len(in_out) - 1
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                AdaptiveSeisFusion(dim_in, dim_in),  # 地震融合
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        # Bottleneck（使用注意力）
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = nn.MultiheadAttention(mid_dim, num_heads=8, batch_first=True)
        self.mid_norm = LayerNorm(mid_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        # 解码器
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = idx == len(in_out) - 1
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                AdaptiveSeisFusion(dim_out, dim_out),  # 地震融合
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        # 输出
        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, default(out_dim, channels), 1)
        )

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, t, y=None, cond_drop_prob=None, seis=None):
        batch, device = x.shape[0], x.device
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # 条件嵌入
        t_emb = self.time_mlp(t)
        c_emb = None
        if exists(self.classes_emb) and exists(y):
            c_emb = self.classes_emb(y)
            if cond_drop_prob > 0:
                keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
                null_emb = repeat(self.null_classes_emb, 'd -> b d', b=batch)
                c_emb = torch.where(rearrange(keep_mask, 'b -> b 1'), c_emb, null_emb)
            c_emb = self.classes_mlp(c_emb)

        # 处理输入
        x = self.input_resize(x)  # (B, init_dim, 64, 64)
        r = x.clone()
        h = []  # 跳跃连接缓存

        # 处理地震特征
        if exists(seis):
            seis = self.seis_input_proj(seis)  # (B, 64, 64, 64)
            # CFG：随机丢弃地震条件
            if cond_drop_prob > 0:
                keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
                seis = seis * keep_mask.view(-1, 1, 1, 1)
        else:
            seis = torch.zeros(batch, self.seis_channels, 64, 64, device=device)

        # 编码器
        for idx, (block1, block2, fusion, downsample) in enumerate(self.downs):
            x = block1(x, t_emb, c_emb)
            h.append(x)

            # 注入地震特征
            seis_injected = self.seis_injector(seis, x.shape[-2:], idx)
            x = fusion(x, seis_injected)

            x = block2(x, t_emb, c_emb)
            h.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.mid_block1(x, t_emb, c_emb)

        # 使用注意力处理 bottleneck（序列化空间维度）
        b, c, h_b, w_b = x.shape
        x_attn = rearrange(x, 'b c h w -> b (h w) c')
        x_attn, _ = self.mid_attn(x_attn, x_attn, x_attn)
        x_attn = rearrange(x_attn, 'b (h w) c -> b c h w', h=h_b, w=w_b)
        x = self.mid_norm(x + x_attn)  # 残差连接

        x = self.mid_block2(x, t_emb, c_emb)

        # 解码器
        for idx, (block1, block2, fusion, upsample) in enumerate(self.ups):
            x = torch.cat([x, h.pop()], dim=1)
            x = block1(x, t_emb, c_emb)

            # 注入地震特征（使用解码器层级索引）
            dec_idx = len(self.downs) - 1 - idx
            seis_injected = self.seis_injector(seis, x.shape[-2:], dec_idx)
            x = fusion(x, seis_injected)

            x = torch.cat([x, h.pop()], dim=1)
            x = block2(x, t_emb, c_emb)
            x = upsample(x)

        # 输出
        x = torch.cat([x, r], dim=1)
        x = self.final_res_block(x, t_emb, c_emb)
        return self.final_conv(x)

    def forward_with_cfg(self, x, t, y, cfg_scale, seis=None):
        """Classifier-Free Guidance 推理"""
        # 并行计算条件和无条件预测（更高效）
        x_double = torch.cat([x, x], dim=0)
        t_double = torch.cat([t, t], dim=0)
        y_double = torch.cat([y, y], dim=0) if exists(y) else None
        seis_double = torch.cat([seis, seis], dim=0) if exists(seis) else None

        # 前向传播：前半有条件，后半无条件（通过 cond_drop_prob 控制）
        logits = self.forward(
            x_double, t_double, y_double,
            cond_drop_prob=0.,  # 前半会应用，后半通过 mask 实现无条件
            seis=seis_double
        )

        # 手动构造无条件：后半部分置零地震特征
        logits_cond, logits_uncond = logits.chunk(2, dim=0)

        if cfg_scale == 1:
            return logits_cond

        # 重新计算无条件（确保地震特征为零）
        with torch.no_grad():
            logits_uncond = self.forward(x, t, y, cond_drop_prob=1., seis=seis)

        return logits_uncond + (logits_cond - logits_uncond) * cfg_scale


# ==================== 辅助函数 ====================

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


# ==================== 测试 ====================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模拟输入
    x = torch.randn(2, 1, 70, 70).to(device)
    seis = torch.randn(2, 64, 70, 70).to(device)  # 假设编码器输出 64 通道
    t = torch.randint(0, 1000, (2,)).to(device)
    y = torch.tensor([0, 1]).to(device)

    model = ImprovedUnet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        seis_channels=64,
        num_classes=2,
        cond_drop_prob=0.1
    ).to(device)

    # 测试前向传播
    with torch.cuda.amp.autocast():
        out = model(x, t, y, seis=seis)
        print(f"Output shape: {out.shape}")  # 期望: (2, 1, 64, 64)

        # 测试 CFG
        out_cfg = model.forward_with_cfg(x, t, y, cfg_scale=2.0, seis=seis)
        print(f"CFG Output shape: {out_cfg.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")