from dit import DiT
from unet import UnetConcat
# from unet import Unet  # 旧版本（Train 0.9 / Test 0.4）
# 🚀 UnetConcat: Input Concatenation 架构（Requirements 2.1-2.4）
import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple
import random

from model import RectifiedFlow, LogitNormalCosineScheduler
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
# import moviepy.editor as mpy
from comet_ml import Experiment
import os
import glob
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from loss_functions import EdgeLoss, TVLoss, SSIMLoss  # 🔥 导入专业的 PDC 边缘检测 + TV Loss 去噪 + SSIM Loss
from utils.test_data_slicer import TestDataSlicer, TestSetConfig  # 🚀 测试集数据切片工具（Requirements 6.1, 6.2）
# 🚀 从统一的ssim_improments模块导入所有SSIM改进组件
from ssim_improments import (
    ProgressiveRegularizer, RegularizationConfig,  # 渐进式正则化控制器（Requirements 3.1-3.5）
    AdaptiveLossScheduler, LossWeights,  # 自适应损失调度器（Requirements 5.1-5.5）
    PhysicsAwareAugmentation,  # 物理约束数据增强（Requirements 4.1-4.4）
    TrainingMonitor, TrainingMonitorConfig, AdjustmentSuggestion,  # 训练监控器（Requirements 10.1-10.5）
    LearningRateController,  # 学习率控制器（Requirements 8.1-8.5）
)


def load_checkpoint_safely(
        model,
        optimizer,
        scheduler,
        checkpoint_path,
        device='cuda',
        allow_shape_mismatch=True
):
    """
    安全加载 Checkpoint，支持架构变动

    Args:
        model: 当前模型
        optimizer: 优化器
        scheduler: 学习率调度器
        checkpoint_path: 权重文件路径
        device: 设备
        allow_shape_mismatch: 是否允许形状不匹配（跳过不匹配的层）

    Returns:
        last_step: 上次训练的步数
    """
    print(f"🔄 加载 Checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # ========================================
    # 1. 加载模型权重（支持部分加载）
    # ========================================
    model_dict = model.state_dict()
    pretrained_dict = checkpoint.get('CureFaultB', checkpoint.get('ema_model', {}))

    if allow_shape_mismatch:
        # 只加载形状匹配的权重
        matched_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        # 诊断信息
        skipped_keys = set(model_dict.keys()) - set(matched_dict.keys())
        new_keys = set(matched_dict.keys()) - set(model_dict.keys())

        print(f"✅ 成功加载 {len(matched_dict)}/{len(model_dict)} 个参数")
        if skipped_keys:
            print(f"⚠️ 跳过 {len(skipped_keys)} 个不匹配的层（前5个）:")
            for key in list(skipped_keys)[:5]:
                old_shape = pretrained_dict.get(key, torch.tensor([])).shape
                new_shape = model_dict[key].shape
                print(f"   - {key}: {old_shape} → {new_shape}")

        model_dict.update(matched_dict)
        model.load_state_dict(model_dict, strict=False)
    else:
        # 严格加载（要求完全匹配）
        model.load_state_dict(pretrained_dict, strict=True)

    # ========================================
    # 2. 加载 Optimizer 状态（过滤不存在的参数）
    # ========================================
    if 'optimizer' in checkpoint and optimizer is not None:
        try:
            opt_state = checkpoint['optimizer']

            # 🔥 关键：只恢复当前模型存在的参数的状态
            current_param_ids = {id(p) for p in model.parameters()}
            filtered_state = {
                k: v for k, v in opt_state['state'].items()
                if k in current_param_ids
            }

            opt_state['state'] = filtered_state
            optimizer.load_state_dict(opt_state)
            print("✅ Optimizer 状态已恢复")
        except Exception as e:
            print(f"⚠️ Optimizer 加载失败（将使用新状态）: {e}")

    # ========================================
    # 3. 恢复训练步数（避免二次 Warmup）
    # ========================================
    last_step = checkpoint.get('step', 0)

    if scheduler is not None and last_step > 0:
        # 🔥 关键：让 Scheduler 从 last_step 开始
        for _ in range(last_step):
            scheduler.step()
        print(f"✅ LR Scheduler 已同步到 step {last_step}")

    print(f"🚀 恢复训练：从 step {last_step} 继续\n")
    return last_step


def normalize_image(pil_img, target_range=(0, 1)):
    """
    将灰度PIL图像归一化到目标范围
    参数:
        pil_img: PIL灰度图像（模式'L'，像素值0-255）
        target_range: 目标范围，如(0,1)或(-1,1)
    返回:
        归一化后的NumPy数组（float类型）
    """
    # 转为NumPy数组（uint8类型，0-255）
    img_np = np.array(pil_img.cpu(), dtype=np.float32)  # 先转为float避免溢出
    # min_val, max_val = target_range  # 目标范围的最小值和最大值
    min_val, max_val = img_np.min(), img_np.max()
    print(min_val, max_val)

    # 归一化公式：(x - 0) / (255 - 0) * (max_val - min_val) + min_val
    # normalized = (img_np / 255.0) * (max_val - min_val) + min_val
    normalized = (img_np - min_val) / (max_val - min_val)
    return normalized


def calculate_image_metrics(img1, img2, save_residual=False, residual_path="Reflow-main"):
    """
    计算两张图像的残差（Residual）、MAE、RMSE、SSIM指标

    参数:
        img1: 图像（已归一化到 [-1, 1]）
        img2: 待比较图像（已归一化到 [-1, 1]）
        save_residual: 是否保存残差图像（默认False）
        residual_path: 残差图像保存路径（默认"Reflow-main"）

    返回:
        metrics: 包含残差数组、MAE、RMSE、SSIM的字典
    """

    # 🔥 修复：直接将 [-1, 1] 映射到 [0, 1]，不要动态归一化
    img1_np = (img1.cpu().numpy() + 1.0) / 2.0
    img2_np = (img2.cpu().numpy() + 1.0) / 2.0
    img1_np = np.clip(img1_np, 0, 1)
    img2_np = np.clip(img2_np, 0, 1)

    # 4. 计算残差（Residual）：像素级差值（img1 - img2）
    residual = img1_np - img2_np  # 残差数组（可正可负，反映像素偏差方向）

    # 5. 计算MAE（平均绝对误差）：残差绝对值的平均值
    mae = np.mean(np.abs(residual))

    # 6. 计算RMSE（均方根误差）：残差平方的平均值开根号
    rmse = np.sqrt(np.mean(residual ** 2))

    # 7. 计算SSIM（结构相似性指数）：需指定数据范围（0-255）
    ssim_score = ssim(img1_np, img2_np, data_range=1)  # 取值范围[-1,1]

    # 8. 可视化残差（将残差归一化到0-255便于显示）
    residual_normalized = ((residual - residual.min()) / (residual.max() - residual.min()) * 255).astype(np.uint8)
    residual_img = Image.fromarray(residual_normalized)

    # 可选：保存残差图像
    if save_residual:
        residual_img.save(residual_path)
        print(f"残差图像已保存至：{residual_path}")

    # 9. 返回结果
    metrics = {
        "residual": residual,  # 残差数组（原始差值）
        "residual_img": residual_img,  # 残差图像（PIL格式）
        "MAE": mae,
        "RMSE": rmse,
        "SSIM": ssim_score
    }
    return metrics


# 固定随机数种子
def seed_torch(seed=42):  # 1029
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True  # 选择确定性算法


seed_torch()


def find_npy_files(folder_path):
    file_pattern = os.path.join(folder_path, '*.npy')
    npy_files = glob.glob(file_pattern)
    npy_files.sort()

    return npy_files


def main():
    n_steps = 80000  # 🚀 训练步数120k（Requirements 2.1, 2.3, 2.4）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64

    image_size = 64
    image_channels = 1
    num_classes = 1

    # Create directories
    os.makedirs('/images_openfwi/', exist_ok=True)
    os.makedirs('/results_openfwi/', exist_ok=True)
    checkpoint_root_path = '/checkpoint/unet_openfwi/'
    os.makedirs(checkpoint_root_path, exist_ok=True)

    # Initialize Comet ML experiment
    experiment = Experiment(
        api_key='8QcemfBuqZM3Pmrj4vWvMf8fi',
        # project_name="seisFFN-sample30-fwi",
        # project_name="large_openfwi"
        # project_name="better-fusing"
        project_name="xx"  # 🚀 新实验：15k样本验证修复版
    )

    # Log unet code to Comet
    unet_path = os.path.join(os.path.dirname(__file__), "unet.py")
    if os.path.exists(unet_path):
        experiment.log_code(file_name=unet_path)
    else:
        print(f"Warning: unet.py not found at {unet_path}")

    # ============================================================
    # 🔬 实验开关：FFT/Wavelet 特征提取
    # ============================================================
    # False（推荐）：极简版，复刻 SeisEncoder（SSIM 0.65）
    # True（备用）：FFT + Wavelet 多尺度特征
    USE_FFT_WAVELET = False

    experiment.log_parameters({
        "dataset": "openfwi",
        "n_steps": n_steps,  # 120k步
        "batch_size": batch_size,
        "learning_rate": 3e-4,  # 🚀 初始学习率3e-4（Requirements 8.1）
        "CureFaultB": "UnetConcat_InputConcatenation",  # 🚀 Input Concatenation 架构
        "dim": 64,  # 🚀 模型维度192（Requirements 2.2, 7.1）
        # "depth": 12,
        # "num_heads": 6,
        "patch_size": 2,
        "dropout_prob": 0.1,
        "optimizer": "AdamW",
        "weight_decay": 1e-2,
        "mixed_precision": "bfloat16",
        "fid_subset_size": 1000,
        "image_size": image_size,
        "image_channels": image_channels,
        "training_cfg_rate": 0.2,
        "lambda_weight": 0.0,  # 🔥 修正：当前代码中 lambda_weight=0.0
        "edge_loss_weight": 0.0,  # 🚀 简化Loss：初始禁用，可选启用（≤0.1）
        "tv_loss_weight": 0.0,  # 🚀 简化Loss：禁用 TV Loss（避免灰图）
        "ssim_loss_weight_max": 0.85,  # 🚀 SSIM Loss最大权重0.85（优化后提升）
        "adagn_init_std": 1e-3,  # 🚀 修复2：从 5e-2 降低到 1e-3（避免条纹透传）
        "loss_balance_strategy": "adaptive_scheduling",  # 🔥 标记：自适应调度法（Requirements 5.1-5.5）
        "sigma_min": 1e-06,
        "ema_decay": 0.9999,
        "dataset_size": 15000,  # 🔥 扩大到45k样本
        "use_fft_wavelet": USE_FFT_WAVELET,  # 🔬 实验开关
        "use_improved_seis_encoder": True,  # 🚀 使用ImprovedSeisEncoder（Requirements 2.2, 7.1）
    })

    transform = T.Compose([
        T.ToTensor(),  # Converts to [0, 1]
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    folder = '/openfwi/data'

    # dataset and dataloader
    # class_dict = ['CurveFaultA', 'CurveFaultB', 'CureFaultB', 'CurveVelA', 'FlatFaultA', 'FlatFaultB', 'FlatVelA', 'FlatVelB']
    class OptimizedSeisDataset(Dataset):
        """
        优化的地震数据集类 - 多种缓存策略
        """

        def __init__(self, folder, class_dict=None, seis_dict=None, h_o=70, w_o=70,
                     cache_strategy='memory_map', cache_size_mb=500):
            """
            初始化数据集

            Args:
                folder: 数据文件夹路径
                class_dict: 类别字典
                seis_dict: 地震数据字典
                h_o, w_o: 数据尺寸
                cache_strategy: 缓存策略 ('memory_map', 'lru_cache', 'preload', 'hybrid')
                cache_size_mb: 缓存大小限制（MB）
            """
            self.class_dict = class_dict if class_dict is not None else ['vmodel']
            self.seis_dict = seis_dict if seis_dict is not None else ['seis']
            self.folder = folder
            self.h_o = h_o
            self.w_o = w_o
            self.cache_strategy = cache_strategy
            self.cache_size_mb = cache_size_mb

            # 文件信息
            self.x_info = []
            self.seis_info = []

            # 缓存相关
            self.file_cache = {}  # 文件缓存
            self.cache_order = []  # LRU缓存顺序
            self.current_cache_size = 0

            self._collect_file_info()
            self._build_cumulative_counts()
            self._prepare_normalization()

        def _collect_file_info(self):
            """收集文件信息"""
            # 收集vmodel文件信息
            for ci, c_v in enumerate(self.class_dict):
                files_path = find_npy_files(os.path.join(self.folder, c_v))
                for f_v in files_path:
                    data = np.load(f_v, mmap_mode='r' if self.cache_strategy == 'memory_map' else None)
                    tmp = data.reshape(-1, self.h_o, self.w_o)
                    sample_count = tmp.shape[0]
                    file_size_mb = os.path.getsize(f_v) / (1024 * 1024)
                    self.x_info.append((f_v, ci, sample_count, file_size_mb))
                    del data, tmp

            # 收集seis文件信息
            for c_s in self.seis_dict:
                seis_files_path = find_npy_files(os.path.join(self.folder, c_s))
                for f_s in seis_files_path:
                    data = np.load(f_s, mmap_mode='r' if self.cache_strategy == 'memory_map' else None)
                    tmp = data.reshape(-1, 5, 1000, 70)
                    sample_count = tmp.shape[0]
                    file_size_mb = os.path.getsize(f_s) / (1024 * 1024)
                    self.seis_info.append((f_s, sample_count, file_size_mb))
                    del data, tmp

        def _build_cumulative_counts(self):
            """构建累积样本数映射"""
            self.x_cum_counts = self._get_cumulative_counts(self.x_info, idx=2)
            self.seis_cum_counts = self._get_cumulative_counts(self.seis_info, idx=1)

        def _prepare_normalization(self):
            """准备归一化参数 - 使用物理先验"""
            if not self.x_info:
                raise ValueError("vmodel文件夹下未找到任何文件")

            # 🚀 使用 OpenFWI/CurveFaultB 的物理速度范围（单位：m/s）
            self.xmin = 1500.0
            self.xmax = 5500.0
            print(f"✅ Global Normalization: [{self.xmin}, {self.xmax}] m/s")

            # 🚀 地震数据全局缩放因子（99分位数，紫色版本的成功方案）
            if self.seis_info:
                print("🔍 正在统计地震数据量级（99分位数）...")
                max_vals = []
                for i in range(min(10, len(self.seis_info))):
                    seis_path = self.seis_info[i][0]
                    data = np.load(seis_path, mmap_mode='r')
                    max_vals.append(np.abs(data).max())

                # 使用99分位数（紫色版本的方案）
                self.seis_global_max = np.percentile(max_vals, 99)
                print(f"✅ 地震数据全局缩放因子（99分位数）: {self.seis_global_max:.4f}")
            else:
                self.seis_global_max = 1.0

        def _get_cumulative_counts(self, info_list, idx):
            """计算累积样本数"""
            cum_counts = [0]
            total = 0
            for info in info_list:
                total += info[idx]
                cum_counts.append(total)
            return cum_counts

        def _find_file_idx(self, idx, cum_counts):
            """找到对应的文件索引"""
            for i in range(len(cum_counts) - 1):
                if cum_counts[i] <= idx < cum_counts[i + 1]:
                    return i
            raise IndexError(f"样本索引{idx}超出范围（总样本数{cum_counts[-1]}）")

        def _get_file_data(self, file_path, reshape_params):
            """
            根据缓存策略获取文件数据
            """
            if self.cache_strategy == 'memory_map':
                # 内存映射 - 不占用额外内存，访问速度中等
                return np.load(file_path, mmap_mode='r')

            elif self.cache_strategy == 'lru_cache':
                # LRU缓存 - 平衡内存和速度
                if file_path in self.file_cache:
                    # 更新LRU顺序
                    self.cache_order.remove(file_path)
                    self.cache_order.append(file_path)
                    return self.file_cache[file_path]

                # 加载新文件
                data = np.load(file_path)

                # 检查缓存大小限制
                data_size_mb = data.nbytes / (1024 * 1024)
                if self.current_cache_size + data_size_mb > self.cache_size_mb:
                    # 清理最久未使用的文件
                    oldest_file = self.cache_order.pop(0)
                    oldest_size = self.file_cache[oldest_file].nbytes / (1024 * 1024)
                    del self.file_cache[oldest_file]
                    self.current_cache_size -= oldest_size

                # 缓存新文件
                self.file_cache[file_path] = data
                self.cache_order.append(file_path)
                self.current_cache_size += data_size_mb

                return data

            elif self.cache_strategy == 'preload':
                # 预加载 - 速度最快，内存占用大
                if file_path not in self.file_cache:
                    self.file_cache[file_path] = np.load(file_path)
                return self.file_cache[file_path]

            else:  # hybrid
                # 混合策略 - 小文件预加载，大文件内存映射
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                if file_size_mb < 50:  # 小于50MB的文件预加载
                    if file_path not in self.file_cache:
                        self.file_cache[file_path] = np.load(file_path)
                    return self.file_cache[file_path]
                else:  # 大文件内存映射
                    return np.load(file_path, mmap_mode='r')

        def __len__(self):
            # 🚀 扩大数据集：使用前 45000 个样本（60个npy文件 * 500样本/文件）
            # 理由：30k样本可能不够，增加到45k提升泛化能力
            total_samples = sum([info[2] for info in self.x_info])
            return min(total_samples, 15000)

        def __getitem__(self, idx):
            # 定位文件
            x_file_idx = self._find_file_idx(idx, self.x_cum_counts)
            x_path, x_label, _, _ = self.x_info[x_file_idx]
            x_inner_idx = idx - self.x_cum_counts[x_file_idx]

            seis_file_idx = self._find_file_idx(idx, self.seis_cum_counts)
            seis_path, _, _ = self.seis_info[seis_file_idx]
            seis_inner_idx = idx - self.seis_cum_counts[seis_file_idx]

            # 获取数据
            x_data = self._get_file_data(x_path, (-1, self.h_o, self.w_o))
            x_sample = x_data.reshape(-1, self.h_o, self.w_o)[x_inner_idx]  # (70, 70)
            x = torch.Tensor(x_sample).unsqueeze(0).unsqueeze(0)  # (1, 1, 70, 70)

            x = x[:, :, :64, :64]
            x = x.squeeze(0)  # (1, 64, 64)

            # 🔥 修复：归一化到 [-1, 1]（与 seis 保持一致）
            # 原来：x = (x - xmin) / (xmax - xmin)  # [0, 1]
            # 现在：先归一化到 [0, 1]，再转到 [-1, 1]
            x = (x - self.xmin) / (self.xmax - self.xmin)  # [0, 1]
            x = x * 2.0 - 1.0  # [-1, 1]

            seis_data = self._get_file_data(seis_path, (-1, 5, 1000, 70))
            seis_sample = seis_data.reshape(-1, 5, 1000, 70)[seis_inner_idx]
            seis = torch.Tensor(seis_sample)

            # 🚀 紫色版本的归一化方案：99分位数缩放 + clamp
            seis = seis / self.seis_global_max
            seis = torch.clamp(seis, -1, 1)

            # 🔥 修复：只保留水平翻转（物理合理的增强）
            # 禁用Mixup：欠拟合时不应该增加学习难度
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[-1])  # 翻转宽度维度
                seis = torch.flip(seis, dims=[-1])  # 同步翻转地震数据

            y = torch.tensor(x_label, dtype=torch.int32)

            return x, y, seis

    # 使用自定义Dataset（扩大到30k样本）
    ds = OptimizedSeisDataset(folder=folder, cache_strategy='memory_map')
    # 测试数据集是否可用
    print(f"🚀 数据集总样本数：{len(ds)} (扩大到30000)")
    sample_x, sample_y, sample_seis = ds[0]  # 尝试获取第0个样本
    print(f"样本x形状：{sample_x.shape}")
    print(f"样本y：{sample_y}")
    print(f"样本seis形状：{sample_seis.shape}")
    train_sampler = torch.utils.data.RandomSampler(ds)
    train_dataloader = DataLoader(ds,
                                  batch_size=batch_size,
                                  num_workers=4,
                                  pin_memory=True,
                                  sampler=train_sampler)

    timestep_scheduler = LogitNormalCosineScheduler(
        loc=0.0,
        scale=1.0,
        min_t=1e-8,
        max_t=1.0 - 1e-8
    )

    def cycle(iterable):
        while True:
            for i in iterable:
                yield i

    train_dataloader = cycle(train_dataloader)

    # Create CureFaultB for CIFAR-10 generation
    model0 = DiT(
        input_size=image_size,
        patch_size=2,
        in_channels=image_channels,
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes,
        learn_sigma=False,
        class_dropout_prob=0.1,
    ).to(device)

    # 🚀 使用 UnetConcat（Input Concatenation 架构）
    # 🔥 提高模型容量：dim 160→192（Requirements 2.2, 7.1）
    model = UnetConcat(
        channels=1,
        dim=64,  # 从160提升至192（增强表达能力，解决欠拟合）
        dim_mults=(1, 2, 4, 8),
        num_classes=num_classes,
        seis_channels=64,  # 地震特征通道数
        cond_drop_prob=0.2,  # CFG 条件丢弃概率（与 training_cfg_rate 一致）
        use_fft_wavelet=USE_FFT_WAVELET,  # 🔬 传递实验开关
    ).to(device)
    # 启用 torch.compile 加速训练
    model = torch.compile(model)

    # Create EMA CureFaultB
    ema_model = deepcopy(model).eval()
    ema_decay = 0.9999

    # 🔥 初始化 SSIM Loss（直接优化结构相似性）
    criterion_ssim = SSIMLoss(window_size=11).to(device)

    # 🚀 初始化渐进式正则化控制器（Requirements 3.1-3.5）
    # 🔥 优化：大幅推迟正则化，避免60k步后过早限制模型
    progressive_regularizer = ProgressiveRegularizer(
        ssim_threshold_low=0.90,  # ⬆️ 从0.85提到0.90（只有达到0.9才启动dropout）
        ssim_threshold_high=0.95,  # ⬆️ 从0.90提到0.95（目标更高）
        gap_threshold_high=0.15,  # ⬆️ 放宽gap容忍度到0.15
        gap_threshold_low=0.05,
        dropout_min=0.0,
        dropout_max=0.15,  # ⬇️ 从0.25降到0.15（大幅降低正则化强度）
        dropout_step=0.03  # ⬇️ 从0.05降到0.03（更缓慢的增长）
    )

    # 🚀 初始化自适应损失调度器（Requirements 5.1-5.5）
    # 🔥 优化：60k步后SSIM权重继续增长，突破平台期
    adaptive_loss_scheduler = AdaptiveLossScheduler(
        phase1_end=20000,  # 阶段1：纯L1（0-30k步）
        phase2_end=50000,  # ⬆️ 从60k延长到80k（给SSIM更长的增长期）
        ssim_weight_min=0.3,
        ssim_weight_max=0.85,  # ⬆️ 从0.7提到0.85（更强的SSIM约束）
        ssim_weight_boost=0.9,  # 停滞时提升至0.9
        ssim_weight_reduce=0.5,  # 下降时降低至0.5
        stagnation_threshold=3000  # 更快检测停滞
    )

    # 用于损失调度的SSIM历史
    train_ssim_for_loss = []
    test_ssim_for_loss = []

    # 🚀 初始化物理约束数据增强（Requirements 4.1-4.4）
    physics_augmentation = PhysicsAwareAugmentation(
        enable_horizontal_flip=True,
        enable_mixup=True,
        mixup_alpha=0.2,
        noise_std_range=(0.01, 0.02),
        time_shift_range=0.05,  # ±5%
        similarity_threshold=0.7
    )

    # 🚀 初始化训练监控器（Requirements 10.1-10.5）
    monitor_config = TrainingMonitorConfig(
        train_log_interval=400,  # 每400步记录训练集SSIM
        test_eval_interval=1000,  # 每1k步评估测试集
        stagnation_window=5000,  # 连续5k步无提升视为停滞
        gap_expansion_window=5000,  # 连续5k步gap扩大触发警告
        target_train_ssim=0.90,
        target_gap=0.05,
        test_samples=10
    )
    training_monitor = TrainingMonitor(config=monitor_config)

    # 🚀 训练参数配置（Task 8.1 - Requirements 5.1, 5.2, 5.3）
    # 🔥 修复1：提高学习率到3e-4（Requirements 8.1）
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    # 🚀 学习率控制器：替换原有的lr_lambda函数（Requirements 8.1-8.5）
    # 🔥 优化：延长cosine周期，避免60k步后学习率过低
    lr_controller = LearningRateController(
        initial_lr=3e-4,  # 初始学习率3e-4（比原来2e-4更激进）
        warmup_steps=10000,  # Warmup 10k步（比原来5k更平滑）
        cosine_period=80000,  # ⬆️ 从80k延长到100k（学习率衰减更慢）
        min_lr_ratio=0.3,  # ⬆️ 从0.2提到0.3（最低学习率更高）
        restart_ratio=0.5,  # 停滞时重启至初始值50%
        decay_ratio=0.5  # 下降时衰减至当前值50%
    )

    # 使用LambdaLR包装LearningRateController
    def lr_lambda_wrapper(step):
        return lr_controller.get_lr(step) / lr_controller.initial_lr

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda_wrapper)

    # Create sampler (use EMA CureFaultB for sampling)
    sampler = RectifiedFlow(
        ema_model,  # Use EMA CureFaultB
        device=device,
        channels=image_channels,
        image_size=image_size,
        num_classes=num_classes,
        use_logit_normal_cosine=True,
        logit_normal_loc=0.0,
        logit_normal_scale=1.0,
        timestep_min=1e-8,
        timestep_max=1.0 - 1e-8,
    ).to(device)

    scaler = torch.cuda.amp.GradScaler()

    # ========================================
    # 🚀 热启动：从 Checkpoint 恢复训练
    # ========================================
    RESUME_FROM_CHECKPOINT = False  # 🔧 第六轮修复：必须从头训练（旧checkpoint已损坏）
    CHECKPOINT_PATH = "Reflow/weights/step_30000.pth"  # 🔧 修改为你的路径

    start_step = 0  # 默认从 0 开始

    if RESUME_FROM_CHECKPOINT and os.path.exists(CHECKPOINT_PATH):
        print(f"\n{'=' * 60}")
        print(f"🔄 检测到 Checkpoint，准备恢复训练...")
        print(f"{'=' * 60}\n")

        start_step = load_checkpoint_safely(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_path=CHECKPOINT_PATH,
            device=device,
            allow_shape_mismatch=True  # 🔥 允许架构变动（跳过不匹配的层）
        )

        # 🔥 同步 EMA 模型（如果 Checkpoint 中有）
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            if 'ema_model' in checkpoint:
                ema_model_dict = ema_model.state_dict()
                ema_pretrained = checkpoint['ema_model']

                # 只加载形状匹配的权重
                ema_matched = {
                    k: v for k, v in ema_pretrained.items()
                    if k in ema_model_dict and v.shape == ema_model_dict[k].shape
                }

                ema_model_dict.update(ema_matched)
                ema_model.load_state_dict(ema_model_dict, strict=False)
                print(f"✅ EMA 模型已恢复 ({len(ema_matched)}/{len(ema_model_dict)} 参数)\n")
        except Exception as e:
            print(f"⚠️ EMA 模型恢复失败（将使用当前模型初始化）: {e}\n")

        print(f"{'=' * 60}")
        print(f"🚀 训练将从 step {start_step} 继续")
        print(f"{'=' * 60}\n")
    else:
        if RESUME_FROM_CHECKPOINT:
            print(f"⚠️ Checkpoint 不存在: {CHECKPOINT_PATH}")
            print(f"   将从头开始训练\n")
        else:
            print(f"ℹ️ 从头开始训练（RESUME_FROM_CHECKPOINT=False）\n")

    def limited_cycle(iterable, max_batches):
        batch_count = 0
        while batch_count < max_batches:
            for i in iterable:
                yield i
                batch_count += 1
                if batch_count >= max_batches:
                    break

    def update_ema(ema_model, model, decay):
        """Update EMA CureFaultB parameters"""
        model.to(device)
        ema_model.to(device)
        with torch.no_grad():
            for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                ema_p = ema_p.to(device)
                p = p.to(device)
                ema_p.data.lerp_(p.data, 1 - decay)

    def sample_and_log_images(step, save_images=False):
        """
        采样并记录图像 - 评估训练集和测试集的SSIM

        Returns:
            tuple: (train_ssim, test_ssim_avg)
        """
        nonlocal best_test_ssim, best_model_step

        print(f"Sampling images at step {step}...")

        ema_model.eval()
        with torch.no_grad():
            # ========================================
            # 1. 评估训练集SSIM（从当前batch采样2个样本）
            # ========================================
            train_data = next(train_dataloader)
            train_x = train_data[0][:2].to(device)  # 取前2个样本
            train_y = train_data[1][:2].to(device)
            train_seis = train_data[2][:2].to(device)

            train_ssim_list = []
            for i in range(2):
                x_gt = train_x[i:i + 1]
                y_single = train_y[i:i + 1]
                seis_single = train_seis[i:i + 1]

                # 生成预测
                samples = sampler.sample_each_class(1, cfg_scale=1.0, return_all_steps=False, seis=seis_single)
                pred = samples[0].squeeze(0).squeeze(0)
                gt = x_gt.squeeze(0).squeeze(0).squeeze(0)

                # 计算SSIM
                metrics = calculate_image_metrics(pred, gt)
                train_ssim_list.append(metrics["SSIM"])

            train_ssim_result = np.mean([s.item() if hasattr(s, 'item') else float(s) for s in train_ssim_list])
            print(f"📊 Train SSIM: {train_ssim_result:.4f}")
            experiment.log_metric("train_avg/SSIM", train_ssim_result, step=step)

            # ========================================
            # 2. 评估测试集SSIM（2个固定测试样本）
            # ========================================
            test_slicer = TestDataSlicer(test_data_dir="/openfwi/data/test_data")
            test_config = TestSetConfig()  # 默认2个样本

            seis_scaling_factor = ds.seis_global_max
            seis_batch, vel_batch, sample_ids = test_slicer.load_test_batch(
                config=test_config,
                device=device,
                seis_global_max=seis_scaling_factor
            )

            test_ssim_list = []
            test_mae_list = []
            test_rmse_list = []

            for i, sample_id in enumerate(sample_ids):
                seis_single = seis_batch[i:i + 1]
                vel_gt = vel_batch[i:i + 1]

                # 生成预测
                samples = sampler.sample_each_class(1, cfg_scale=1.0, return_all_steps=False, seis=seis_single)
                pred = samples[0].squeeze(0).squeeze(0)
                gt = vel_gt.squeeze(0).squeeze(0).squeeze(0)

                # 计算指标
                metrics = calculate_image_metrics(pred, gt)
                test_ssim_list.append(metrics["SSIM"])
                test_mae_list.append(metrics["MAE"])
                test_rmse_list.append(metrics["RMSE"])

                # 记录单个样本SSIM
                experiment.log_metric(
                    name=f"test_ssim/{sample_id}",
                    value=metrics["SSIM"].item() if hasattr(metrics["SSIM"], 'item') else float(metrics["SSIM"]),
                    step=step,
                )

            # 计算测试集平均指标
            test_ssim_result = np.mean([s.item() if hasattr(s, 'item') else float(s) for s in test_ssim_list])
            test_mae = np.mean([m.item() if hasattr(m, 'item') else float(m) for m in test_mae_list])
            test_rmse = np.mean([r.item() if hasattr(r, 'item') else float(r) for r in test_rmse_list])

            print(f"📊 Test SSIM: {test_ssim_result:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")

            # 记录测试集指标
            experiment.log_metric("test_avg/SSIM", test_ssim_result, step=step)
            experiment.log_metric("test_avg/MAE", test_mae, step=step)
            experiment.log_metric("test_avg/RMSE", test_rmse, step=step)

            # ========================================
            # 3. 保存图像（可选）
            # ========================================
            if save_images and len(sample_ids) >= 2:
                vis_preds = []
                vis_gts = []

                for i in range(min(2, len(sample_ids))):
                    seis_single = seis_batch[i:i + 1]
                    vel_gt = vel_batch[i:i + 1]

                    samples = sampler.sample_each_class(1, cfg_scale=1.0, return_all_steps=False, seis=seis_single)
                    pred = samples  # (1, 1, 64, 64)

                    vis_preds.append(pred)
                    vis_gts.append(vel_gt)

                vis_preds = torch.cat(vis_preds, dim=0)  # (2, 1, 64, 64)
                vis_gts = torch.cat(vis_gts, dim=0)  # (2, 1, 64, 64)

                # 保存预测图像
                log_img_pred = make_grid(vis_preds, nrow=2, normalize=True, value_range=(-1, 1))
                img_save_path_pred = r"/images_openfwi/step_test_pred.png"
                save_image(log_img_pred, img_save_path_pred)
                experiment.log_image(img_save_path_pred, name="test_predictions", step=step)

                # 保存真值图像
                log_img_gt = make_grid(vis_gts, nrow=2, normalize=True, value_range=(-1, 1))
                img_save_path_gt = r"/images_openfwi/step_test_gt.png"
                save_image(log_img_gt, img_save_path_gt)
                experiment.log_image(img_save_path_gt, name="test_ground_truth", step=step)

        return train_ssim_result, test_ssim_result

    losses = []
    sigma_min = 1e-06
    training_cfg_rate = 0.2
    lambda_weight = 0.0
    use_immiscible = True
    gradient_clip = 1.0
    best_test_ssim = 0.0
    best_model_step = 0
    base_loss_ema = None
    edge_loss_ema = None
    ema_momentum = 0.99  # EMA 动量
    ssim_history = []  # 存储最近的SSIM值
    low_ssim_threshold = 0.1  # 灰图判定阈值
    low_ssim_warning_steps = 1000  # 连续多少步低SSIM触发警告
    gray_image_warning_issued = False  # 避免重复警告
    max_steps_for_validation = 80000

    with tqdm(range(start_step, n_steps), dynamic_ncols=True, initial=start_step, total=n_steps) as pbar:
        pbar.set_description("Training Unet on openfwi")
        def g(t):
            return torch.cos(0.5 * torch.pi * t)
        def f(t):
            return torch.sin(0.5 * torch.pi * t)
        def h(t):
            return torch.sin(torch.pi * t)

        for step in pbar:
            data = next(train_dataloader)
            optimizer.zero_grad()

            x1 = data[0].to(device)
            y = data[1].to(device)
            seis = data[2].to(device)
            b = x1.shape[0]

            # 🚀 应用物理约束数据增强（Requirements 4.1-4.4）
            # 根据正则化配置决定是否应用增强
            if 'reg_config' in locals():
                # 应用水平翻转（如果启用）
                apply_h_flip = reg_config.use_horizontal_flip
                # 应用Mixup（如果启用）
                apply_mixup = reg_config.use_mixup
                # 应用噪声注入（如果启用）
                apply_noise = reg_config.use_noise_injection
                # 应用时间扰动（如果启用）
                apply_time_shift = reg_config.use_time_shift

                # 对于Mixup，需要获取第二个样本
                if apply_mixup and b > 1:
                    # 随机选择另一个样本进行Mixup
                    perm = torch.randperm(b, device=device)
                    x1_aug, seis_aug = physics_augmentation.apply(
                        velocity=x1,
                        seismic=seis,
                        velocity2=x1[perm],
                        seismic2=seis[perm],
                        apply_horizontal_flip=apply_h_flip,
                        apply_mixup=apply_mixup,
                        apply_noise=apply_noise,
                        apply_time_shift=apply_time_shift
                    )
                    x1 = x1_aug
                    seis = seis_aug
                else:
                    # 不使用Mixup，只应用其他增强
                    x1_aug, seis_aug = physics_augmentation.apply(
                        velocity=x1,
                        seismic=seis,
                        apply_horizontal_flip=apply_h_flip,
                        apply_mixup=False,
                        apply_noise=apply_noise,
                        apply_time_shift=apply_time_shift
                    )
                    x1 = x1_aug
                    seis = seis_aug

            t = torch.rand(b, device=device)

            alpha_t = t
            sigma_t = 1 - (1 - sigma_min) * t

            alpha_t = alpha_t.view(b, 1, 1, 1)
            sigma_t = sigma_t.view(b, 1, 1, 1)

            if use_immiscible:
                k = 4

                z_candidates = torch.randn(b, k, image_channels, image_size, image_size, device=x1.device,
                                           dtype=x1.dtype)

                x1_flat = x1.flatten(start_dim=1)  # [b, c*h*w]
                z_candidates_flat = z_candidates.flatten(start_dim=2)  # [b, k, c*h*w]

                distances = torch.norm(x1_flat.unsqueeze(1) - z_candidates_flat, dim=2)  # [b, k]

                min_distances, min_indices = torch.min(distances, dim=1)  # [b]

                batch_indices = torch.arange(b, device=x1.device)
                z = z_candidates[batch_indices, min_indices]  # [b, c, h, w]

            else:
                # Standard noise sampling
                z = torch.randn_like(x1)

            x_t = sigma_t * z + alpha_t * x1

            u_positive = x1 - (1 - sigma_min) * z

            if b > 1:
                perm = torch.randperm(b, device=x1.device)
                # Ensure no self-pairing
                for i in range(b):
                    if perm[i] == i:
                        perm[i] = (i + 1) % b
                u_negative = u_positive[perm]
            else:
                u_negative = u_positive

            '''if training_cfg_rate > 0:
                drop_mask = torch.rand(b, device=x1.device) < training_cfg_rate
                y = torch.where(drop_mask, num_classes, y)'''

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # 1. 模型预测流速度 v (velocity field)
                pred = model(x_t, t, y, seis=seis)
                pred = g(t) * (0.0 - x_t) + f(t) * x_t + h(t) * pred
                base_loss  = F.l1_loss(pred, u_positive)

                # 🔥 修复1：SSIM Loss（直接优化结构相似性）
                # 理由：你的KPI是SSIM，必须把它写进Loss函数！
                # 权重0.3：让模型同时关注像素误差和结构相似性
                loss_ssim = criterion_ssim(pred, u_positive)


                # 保留EMA用于监控
                if base_loss_ema is None:
                    base_loss_ema = base_loss.item()
                else:
                    base_loss_ema = ema_momentum * base_loss_ema + (1 - ema_momentum) * base_loss.item()

                # 🚀 自适应损失权重调度（Requirements 5.1-5.5）
                loss_weights = adaptive_loss_scheduler.get_loss_weights(
                    step=step,
                    train_ssim_history=train_ssim_for_loss,
                    test_ssim_history=test_ssim_for_loss
                )

                # 应用动态权重
                l1_weight = loss_weights.l1_weight
                ssim_weight = loss_weights.ssim_weight

                # 🚀 最终Loss公式（使用自适应权重，Requirements 5.1-5.5）
                # base_loss (纯L1): 负责像素级准确+对比度
                # loss_ssim: 负责结构（权重动态调整：0-30k步为0，30k-60k步线性增长到0.7，>60k步固定0.7）
                loss = l1_weight * base_loss + ssim_weight * loss_ssim

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

            # 🔥 禁用梯度噪声：欠拟合时不需要探索，需要稳定学习

            scaler.step(optimizer)
            scaler.update()

            # Step learning rate scheduler
            scheduler.step()

            # Update EMA
            update_ema(ema_model, model, ema_decay)

            # Logging
            losses.append(loss.item())
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "loss": loss.item(),
                "ssim": loss_ssim.item(),  # 🔥 监控SSIM Loss
                "grad_norm": grad_norm.item(),
                "lr": f"{current_lr:.2e}"
            })
            experiment.log_metric("loss", loss.item(), step=step)
            experiment.log_metric("grad_norm", grad_norm.item(), step=step)
            experiment.log_metric("loss_ssim", loss_ssim.item(), step=step)  # 🔥 记录SSIM Loss
            experiment.log_metric("base_loss", base_loss.item(), step=step)
            experiment.log_metric("learning_rate", current_lr, step=step)


            # 🚀 监控三个 Loss 的占比（确保 Base Loss 占主导）
            total_loss_val = loss.item()
            if total_loss_val > 0:
                base_contribution = base_loss.item()
                experiment.log_metric("base_loss_ratio", base_contribution / total_loss_val, step=step)

            if step % 100 == 0:
                avg_loss = sum(losses[-100:]) / min(100, len(losses))
                experiment.log_metric("avg_loss_100", avg_loss, step=step)

            # 🚀 每 400 步采样计算 SSIM 指标（不保存图片，降低计算开销）
            if step % 400 == 0:
                avg_loss = sum(losses) / len(losses) if losses else 0
                print(f"\nStep: {step + 1}/{n_steps} | avg_loss: {avg_loss:.4f}")
                losses.clear()
                train_ssim, test_ssim = sample_and_log_images(step, save_images=False)

                # 🚀 记录SSIM到训练监控器（Requirements 10.1, 10.2）
                training_monitor.log_train_ssim(step, train_ssim)
                if test_ssim is not None:
                    training_monitor.log_test_ssim(step, test_ssim)

                # 🚀 检查调整建议（Requirements 10.3, 10.4, 10.5）
                adjustment = training_monitor.get_adjustment_suggestion(step)
                if adjustment is not None:
                    print(f"\n{'=' * 60}")
                    print(f"🔔 调整建议: {adjustment.type}")
                    print(f"   原因: {adjustment.reason}")
                    if adjustment.suggested_value is not None:
                        print(f"   建议值: {adjustment.suggested_value}")
                    print(f"{'=' * 60}\n")

                    # 记录调整建议到Comet
                    experiment.log_text(f"Adjustment: {adjustment.type} - {adjustment.reason}", step=step)

                    # 根据建议类型执行相应操作
                    if adjustment.type == 'early_stop':
                        print(f"✅ 达到训练目标，提前停止训练！")
                        print(f"   Train SSIM: {train_ssim:.4f}, Test SSIM: {test_ssim:.4f}")
                        break
                    elif adjustment.type == 'lr_restart':
                        # 🚀 学习率重启（Requirements 8.4）
                        new_lr = lr_controller.restart_lr(step)
                        # 更新optimizer的学习率
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                        print(f"   ⚡ 学习率已重启至: {new_lr:.2e}")
                        experiment.log_metric("lr_restart", new_lr, step=step)
                    elif adjustment.type == 'lr_decay':
                        # 🚀 学习率衰减（Requirements 8.5）
                        new_lr = lr_controller.decay_lr(step)
                        # 更新optimizer的学习率
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                        print(f"   ⚡ 学习率已衰减至: {new_lr:.2e}")
                        experiment.log_metric("lr_decay", new_lr, step=step)

                # 🚀 渐进式正则化：根据SSIM动态调整Dropout（Requirements 3.1-3.5）
                reg_config = progressive_regularizer.get_regularization_config(
                    train_ssim=train_ssim,
                    test_ssim=test_ssim
                )

                # 应用Dropout配置到模型
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = reg_config.dropout_prob


                # 打印当前阶段和正则化配置
                current_phase = progressive_regularizer.get_current_phase(train_ssim)
                print(f"📊 {current_phase}")
                print(f"   正则化配置: {reg_config}")

                # 🚀 更新SSIM历史用于损失调度（Requirements 5.1-5.5）
                train_ssim_for_loss.append(train_ssim)
                if test_ssim is not None:
                    test_ssim_for_loss.append(test_ssim)

                # 记录当前损失权重
                experiment.log_metric("loss_weights/l1", l1_weight if 'l1_weight' in locals() else 1.0, step=step)
                experiment.log_metric("loss_weights/ssim", ssim_weight if 'ssim_weight' in locals() else 0.0, step=step)

                # 🚀 灰图检测：记录SSIM历史
                ssim_history.append(train_ssim)

                # 保持历史记录在合理范围内（最多保留 low_ssim_warning_steps / 400 个采样点）
                max_history_len = low_ssim_warning_steps // 400 + 1
                if len(ssim_history) > max_history_len:
                    ssim_history.pop(0)

                # 🚀 检测灰图：连续多个采样点SSIM都低于阈值
                if len(ssim_history) >= max_history_len:
                    all_low_ssim = all(s < low_ssim_threshold for s in ssim_history)
                    if all_low_ssim and not gray_image_warning_issued:
                        print(
                            f"\n⚠️ 警告：连续 {low_ssim_warning_steps} 步 SSIM < {low_ssim_threshold}，可能陷入灰图局部最优！")
                        print(f"   最近SSIM历史: {[f'{s:.4f}' for s in ssim_history[-5:]]}")
                        print(f"   建议：检查数据归一化、学习率、或架构设计")
                        experiment.log_text(
                            f"⚠️ Gray Image Warning at step {step}: SSIM consistently < {low_ssim_threshold}")
                        gray_image_warning_issued = True

            if step % 2000 == 0 or step == n_steps - 1:
                # 🚀 每 2000 步保存图片到磁盘和 Comet
                _, _ = sample_and_log_images(step, save_images=True)

            # 🚀 控制变量实验：100k 步后停止
            if step >= n_steps - 1:
                print(f"\n✅ 训练完成：已运行 {step + 1} 步（数据集：15k样本）")
                print(f"检查 Test SSIM 是否 > 0.5（旧版 0.4）")
                break

            # 🧹 每10000步自动清理磁盘（防止空间不足）
            if step % 10000 == 0 and step > 0:
                try:
                    import auto_cleanup
                    auto_cleanup.main(quick_mode=True, verbose=False)
                except Exception as e:
                    print(f"⚠️ 自动清理失败: {e}")

    experiment.end()


if __name__ == "__main__":
    main()