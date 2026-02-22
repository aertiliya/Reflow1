import math
from functools import partial
import torch
from einops import rearrange, reduce, repeat
from torch import nn, einsum
import torch.nn.functional as F
from ssim_improments import ImprovedSeisEncoder  # 🚀 从统一模块导入改进的地震编码器


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


# U-Net 基础模块（保持不变）
def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, seis_cond):
        x = self.norm(x)
        return self.fn(x, seis_cond)


class FeedForward(nn.Module):
    """
    改进的 MLP 模块：增加深度和残差连接
    用于增强 UnetConcat 的特征表达能力
    """

    def __init__(self, dim, mult=4):
        super().__init__()
        hidden_dim = dim * mult
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),  # 深度可分离卷积
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )
        self.shortcut = nn.Identity()

    def forward(self, x):
        return self.net(x) + self.shortcut(x)


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


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


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
    """改进的 ResNet 块 - 增强条件注入和特征提取"""

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

        # 增强的残差连接
        self.res_conv = nn.Sequential(
            nn.Conv2d(dim, dim_out, 1),
            nn.GroupNorm(groups, dim_out)
        ) if dim != dim_out else nn.Identity()

        # 添加 squeeze-excitation 注意力
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim_out, dim_out // 4, 1),
            nn.SiLU(),
            nn.Conv2d(dim_out // 4, dim_out, 1),
            nn.Sigmoid()
        )

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

        # SE 注意力
        se_weight = self.se(h)
        h = h * se_weight

        return h + self.res_conv(x)


class ConvSeisAligner(nn.Module):
    """专用卷积对齐模块 - 仅使用卷积，不用插值"""

    def __init__(self, base_channels=64, dim=64):
        super().__init__()
        self.base_channels = base_channels
        self.dim = dim

        # ==================== 编码器投影层 ====================
        # 第0层: 64 -> dim, 64x64 -> 64x64
        self.enc_proj_0 = nn.Sequential(
            nn.Conv2d(base_channels, dim, kernel_size=1, stride=1, padding=0, bias=False),  # 下采样
            nn.GroupNorm(32, dim),
            nn.SiLU()
        )
        # 第1层: 64 -> dim, 64x64 -> 32x32
        self.enc_proj_1 = nn.Sequential(
            nn.Conv2d(base_channels, dim, kernel_size=3, stride=2, padding=1, bias=False),  # 下采样
            nn.GroupNorm(32, dim),
            nn.SiLU()
        )

        # 第2层: 64 -> dim*2, 64x64 -> 16x16
        self.enc_proj_2 = nn.Sequential(
            nn.Conv2d(base_channels, dim, kernel_size=3, stride=2, padding=1, bias=False),  # 64->32
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, bias=False),  # 32->16
            nn.GroupNorm(32, dim * 2),
            nn.SiLU()
        )

        # 第3层: 64 -> dim*4, 64x64 -> 8x8
        self.enc_proj_3 = nn.Sequential(
            nn.Conv2d(base_channels, dim, kernel_size=3, stride=2, padding=1, bias=False),  # 64->32
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False),  # 32->16
            nn.Conv2d(dim, dim * 4, kernel_size=3, stride=2, padding=1, bias=False),  # 16->8
            nn.GroupNorm(32, dim * 4),
            nn.SiLU()
        )

        # ==================== 解码器投影层 ====================
        # 第0解码层: 64 -> dim*8, 64x64 -> 8x8 (与编码器第3层相同)
        self.dec_proj_0 = nn.Sequential(
            nn.Conv2d(base_channels, dim, kernel_size=3, stride=2, padding=1, bias=False),  # 64->32
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False),  # 32->16
            nn.Conv2d(dim, dim * 8, kernel_size=3, stride=2, padding=1, bias=False),  # 16->8
            nn.GroupNorm(32, dim * 8),
            nn.SiLU()
        )

        # 第1解码层: 64 -> dim*4, 64x64 -> 16x16 (与编码器第2层相同)
        self.dec_proj_1 = nn.Sequential(
            nn.Conv2d(base_channels, dim, kernel_size=3, stride=2, padding=1, bias=False),  # 64->32
            nn.Conv2d(dim, dim * 4, kernel_size=3, stride=2, padding=1, bias=False),  # 32->16
            nn.GroupNorm(32, dim * 4),
            nn.SiLU()
        )

        # 第2解码层: 64 -> dim*2, 64x64 -> 32x32 (与编码器第1层相同)
        self.dec_proj_2 = nn.Sequential(
            nn.Conv2d(base_channels, dim * 2, kernel_size=3, stride=2, padding=1, bias=False),  # 64->32
            nn.GroupNorm(32, dim * 2),
            nn.SiLU()
        )

        # 第3解码层: 64 -> dim, 64x64 -> 64x64 (初始投影)
        self.dec_proj_3 = self.enc_proj_0

        # ==================== 特殊层 ====================
        # 初始层投影
        self.init_proj = nn.Sequential(
            nn.Conv2d(base_channels, 192, kernel_size=1, bias=False),
            nn.GroupNorm(32, 192),
            nn.SiLU()
        )

        # 瓶颈层投影 (使用编码器第3层)
        self.bottleneck_proj = self.dec_proj_0

    def forward(self, seis_features, mode='encoder', level_idx=0):
        if mode == 'init':
            # 初始层: dim, 64x64
            return self.init_proj(seis_features)

        elif mode == 'encoder':
            if level_idx == 0:
                # 编码器第1层: dim, 64x64
                return self.enc_proj_0(seis_features)
            elif level_idx == 1:
                # 编码器第2层: dim*1, 32x32
                return self.enc_proj_1(seis_features)
            elif level_idx == 2:
                # 编码器第3层: dim*2, 16x16
                return self.enc_proj_2(seis_features)
            elif level_idx == 3:
                # 编码器第3层: dim*4, 8x8
                return self.enc_proj_3(seis_features)
            else:
                # 默认返回原始特征
                return seis_features

        elif mode == 'bottleneck':
            # 瓶颈层: dim*8, 8x8
            return self.bottleneck_proj(seis_features)

        elif mode == 'decoder':
            if level_idx == 0:
                # 解码器第0层: dim*8, 8x8
                return self.dec_proj_0(seis_features)
            elif level_idx == 1:
                # 解码器第1层: dim*4, 16x16
                return self.dec_proj_1(seis_features)
            elif level_idx == 2:
                # 解码器第2层: dim*2, 32x32
                return self.dec_proj_2(seis_features)
            elif level_idx == 3:
                # 解码器第3层: dim, 64x64
                return self.dec_proj_3(seis_features)
            else:
                # 默认返回原始特征
                return seis_features

        else:
            return seis_features


# UnetConcat - Input Concatenation 架构（替代 AdaGN）

class UnetConcat(nn.Module):

    def __init__(
            self,
            dim=64,
            num_classes=None,
            cond_drop_prob=0.5,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=1,  # 速度模型通道数
            seis_channels=64,  # 地震特征通道数
            resnet_block_groups=8,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            use_fft_wavelet=False,  # 实验开关：是否使用 FFT/Wavelet 特征
    ):
        super().__init__()

        self.cond_drop_prob = cond_drop_prob
        self.channels = channels
        self.seis_channels = seis_channels

        init_dim = default(init_dim, dim)

        self.seis_aligner = ConvSeisAligner(seis_channels, init_dim)
        self.init_conv = nn.Sequential(
            nn.Conv2d(channels, init_dim, kernel_size=7, stride=1, padding=3),
            LayerNorm(init_dim),
            nn.SiLU(),
            nn.Conv2d(init_dim, init_dim, kernel_size=3, stride=1, padding=0),  # 70->68
            nn.Conv2d(init_dim, init_dim, kernel_size=3, stride=1, padding=0),  # 68->66
            nn.Conv2d(init_dim, init_dim, kernel_size=3, stride=1, padding=0),  # 66->64
        )

        # 逐层注入编码对齐
        self.seis_layer_conv = nn.Sequential(
            nn.Conv2d(seis_channels, seis_channels, kernel_size=7, stride=1, padding=3),
            nn.SiLU(),
            nn.Conv2d(seis_channels, seis_channels, 3, padding=0),  # 70->68
            nn.Conv2d(seis_channels, seis_channels, 3, padding=0),  # 68->66
            nn.Conv2d(seis_channels, seis_channels, 3, padding=0),  # 66->64
        )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # class embeddings (可选)
        self.classes_emb = nn.Embedding(num_classes, dim) if num_classes is not None and num_classes > 0 else None
        self.null_classes_emb = nn.Parameter(torch.randn(dim)) if num_classes is not None and num_classes > 0 else None

        classes_dim = dim * 4 if (num_classes is not None and num_classes > 0) else None
        self.classes_dim = classes_dim
        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        ) if classes_dim is not None else None

        # 🚀 地震特征提取器：使用ImprovedSeisEncoder
        self.feature_net = ImprovedSeisEncoder(
            in_channels=5,
            out_channels=seis_channels,
            use_low_freq_fft=use_fft_wavelet,  # 与实验开关保持一致
            conv1d_kernel_size=11,  # 使用默认的11（在7-15范围内）
            time_pool_learnable=True,  # 启用可学习时间池化
            use_spatial_attention=True,  # 启用空间注意力
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                ImprovedFeatureFusion(dim_in),
                ImprovedFeatureFusion(dim_in),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.mid_attn = ImprovedFeatureFusion(mid_dim)
        self.mid_mlp1 = FeedForward(mid_dim, mult=4)  # 增强瓶颈层的表达能力
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.mid_mlp2 = FeedForward(mid_dim, mult=4)  # 再加一层 MLP

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                ImprovedFeatureFusion(dim_out),
                ImprovedFeatureFusion(dim_out),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        # 改进的最终输出层
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.final_mlp = FeedForward(dim, mult=2)  # 添加特征增强
        self.final_shape_conv = nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=7, stride=1,
                                                   padding=0, output_padding=0)
        self.final_refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1)
        )
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, t, y, cond_drop_prob=None, seis=None):
        """
        前向传播

        Args:
            x: (B, 1, 64, 64) - 噪声图 x_t
            t: (B,) - 时间步
            y: (B,) - 类别标签（可选）
            cond_drop_prob: CFG 条件丢弃概率
            seis: (B, 5, 1000, 70) - 地震数据

        Returns:
            (B, 1, 64, 64) - 预测的速度场
        """
        batch, device = x.shape[0], x.device
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # 类别嵌入（可选）
        if exists(self.classes_emb):
            classes_emb = self.classes_emb(y)
            if cond_drop_prob > 0:
                keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
                null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b=batch)
                classes_emb = torch.where(
                    rearrange(keep_mask, 'b -> b 1'),
                    classes_emb,
                    null_classes_emb
                )
            c = self.classes_mlp(classes_emb)
        else:
            c = None

        # 提取地震特征 (B, 64, 70, 70)
        seis_features = self.feature_net(seis)

        # CFG 支持：随机将地震特征置零（实现无条件生成）
        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
            # 将 mask 扩展到特征维度 (B,) -> (B, 1, 1, 1)
            keep_mask = keep_mask.view(-1, 1, 1, 1).float()
            seis_features = seis_features * keep_mask

        # (B, 64, 70, 70) -> (B, 64, 64, 64)
        seis_layer_features = self.seis_layer_conv(seis_features)

        # 标准 U-Net 流程
        x = self.init_conv(x)  # (B, 1, 70, 70) -> (B, 64, 64, 64)
        r = x.clone()
        t = self.time_mlp(t)
        h = []

        # Encoder
        for idx, (block1, block2, attn1, attn2, downsample) in enumerate(self.downs):
            x = block1(x, t, c)
            h.append(x)
            aligned_seis = self.seis_aligner(seis_layer_features, mode='encoder', level_idx=idx)
            x = block2(x, t, c)
            x = attn2(x, aligned_seis)
            h.append(x)
            x = downsample(x)

        # Bottleneck - 增强全局特征提取
        x = self.mid_block1(x, t, c)
        mid_seis = self.seis_aligner(seis_layer_features, mode='bottleneck')
        x = self.mid_attn(x, mid_seis)
        x = x + self.mid_mlp1(x)  # 残差连接
        x = self.mid_block2(x, t, c)
        x = x + self.mid_mlp2(x)  # 再次增强

        # Decoder
        for idx, (block1, block2, attn1, attn2, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, c)
            aligned_seis = self.seis_aligner(seis_layer_features, mode='decoder', level_idx=idx)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, c)
            x = attn2(x, aligned_seis)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t, c)
        x = x + self.final_mlp(x)  # 特征增强
        x = self.final_shape_conv(x)
        x = x + self.final_refine(x)  # 最终细化
        return self.final_conv(x)

    def forward_with_cfg(self, x, t, y, cfg_scale, is_train_student=False, seis=None):
        """
        Classifier-Free Guidance 推理

        Args:
            x: (B, 1, 64, 64) - 噪声图
            t: 时间步
            y: 类别标签
            cfg_scale: CFG 缩放因子
            is_train_student: 是否为学生模型训练
            seis: (B, 5, 1000, 70) - 地震数据

        Returns:
            CFG 加权后的输出
        """
        if is_train_student:
            t = t.repeat(2)
        else:
            t = t.repeat(x.shape[0])

        # 有条件预测（地震特征正常）
        logits = self.forward(x, t, y, cond_drop_prob=0., seis=seis)

        if cfg_scale == 1:
            return logits

        # 无条件预测（地震特征置零）
        null_logits = self.forward(x, t, y, cond_drop_prob=1., seis=seis)

        # CFG 公式：output = null + cfg_scale * (cond - null)
        return null_logits + (logits - null_logits) * cfg_scale


class ImprovedFeatureFusion(nn.Module):
    """
    AdaIN-based 特征融合模块
    使用seis的统计量（均值/方差）对x进行风格调制，保留x的内容结构
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim

        # 可选：学习一个混合权重
        self.alpha = nn.Parameter(torch.zeros(1))

        # 投影层，将调制后的特征映射回原始维度
        self.post_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True)
        )
    def adain(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        AdaIN: 将content的特征统计量适配到style
        Args:
            content: [B, C, H, W] - 内容特征（x）
            style: [B, C, H, W] - 风格特征（seis）
        """
        # 计算均值和方差 [B, C, 1, 1]
        content_mean = content.mean(dim=[2, 3], keepdim=True)
        content_std = content.std(dim=[2, 3], keepdim=True) + self.eps

        style_mean = style.mean(dim=[2, 3], keepdim=True)
        style_std = style.std(dim=[2, 3], keepdim=True) + self.eps

        # 标准化content，然后用style的统计量进行缩放和平移
        normalized = (content - content_mean) / content_std
        stylized = normalized * style_std + style_mean

        return stylized

    def forward(self, x: torch.Tensor, seis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, dim, H, W] - 主模态特征
            seis: [B, dim, H, W] - 地震模态特征（作为风格）
        Returns:
            [B, dim, H, W] - 融合后的特征
        """
        # AdaIN调制：用seis的风格统计量调制x
        fused = self.adain(x, seis)

        # 可学习的残差混合
        output = self.alpha * self.post_proj(fused) + (1 - self.alpha) * x

        return output


if __name__ == "__main__":
    x = torch.randn(1, 1, 70, 70)
    seis = torch.randn(1, 5, 1000, 70)
    ts = torch.randint(low=1, high=1000, size=(x.shape[0],))
    y = torch.tensor(0, dtype=torch.int32)

    unet = UnetConcat(
        channels=1,
        dim=64,  # 从160提升至192（增强表达能力，解决欠拟合）
        dim_mults=(1, 2, 4, 8),
        num_classes=1,
        seis_channels=64,  # 地震特征通道数
        cond_drop_prob=0.2,  # CFG 条件丢弃概率（与 training_cfg_rate 一致）
        use_fft_wavelet=False,  # 🔬 传递实验开关
    )

    out = unet.forward(x=x, t=ts, y=y, seis=seis)
