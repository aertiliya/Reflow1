
from inversionNet import InversionNet

import torch

from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from PIL import Image
import random
from torch.utils.data import DataLoader, Dataset
from comet_ml import Experiment
import os
import glob
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from loss_functions import SSIMLoss  # 🔥 导入专业的 PDC 边缘检测 + TV Loss 去噪 + SSIM Loss
from utils.test_data_slicer import TestDataSlicer, TestSetConfig  # 🚀 测试集数据切片工具（Requirements 6.1, 6.2）
# 🚀 从统一的ssim_improments模块导入所有SSIM改进组件
from ssim_improments import (
    ProgressiveRegularizer,  # 渐进式正则化控制器（Requirements 3.1-3.5）
    TrainingMonitor, TrainingMonitorConfig,  # 训练监控器（Requirements 10.1-10.5）
    LearningRateController,  # 学习率控制器（Requirements 8.1-8.5）
)

def calculate_image_metrics(img1, img2, save_residual=False, residual_path="Reflow-main"):

    img1_np = (img1.cpu().numpy() + 1.0) / 2.0
    img2_np = (img2.cpu().numpy() + 1.0) / 2.0
    img1_np = np.clip(img1_np, 0, 1)
    img2_np = np.clip(img2_np, 0, 1)

    residual = img1_np - img2_np

    mae = np.mean(np.abs(residual))

    rmse = np.sqrt(np.mean(residual ** 2))

    ssim_score = ssim(img1_np, img2_np, data_range=1)

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
    n_steps = 120000  # 🚀 训练步数120k（Requirements 2.1, 2.3, 2.4）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64

    image_size = 70
    image_channels = 1
    num_classes = 1

    # Create directories
    os.makedirs('/images_openfwi/', exist_ok=True)
    os.makedirs('/results_openfwi/', exist_ok=True)


    # Initialize Comet ML experiment
    experiment = Experiment(
        api_key='8QcemfBuqZM3Pmrj4vWvMf8fi',
        project_name="cx"  # 🚀 新实验：15k样本验证修复版
    )

    # Log unet code to Comet
    unet_path = os.path.join(os.path.dirname(__file__), "unet.py")
    if os.path.exists(unet_path):
        experiment.log_code(file_name=unet_path)
    else:
        print(f"Warning: unet.py not found at {unet_path}")


    folder = r'C:\Users\MSI\Desktop\openfwi\data'

    class OptimizedSeisDataset(Dataset):
        """
        优化的地震数据集类 - 多种缓存策略
        """

        def __init__(self, folder, class_dict=None, seis_dict=None, h_o=70, w_o=70,
                     cache_strategy='memory_map', cache_size_mb=500):

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
            return min(total_samples, 45000)

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

            x = x.squeeze(0)  # (1, 70, 70)

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
    print(f"🚀 数据集总样本数：{len(ds)} (扩大到45000)")
    sample_x, sample_y, sample_seis = ds[0]  # 尝试获取第0个样本
    print(f"样本x形状：{sample_x.shape}")
    print(f"样本y：{sample_y}")
    print(f"样本seis形状：{sample_seis.shape}")
    train_sampler = torch.utils.data.RandomSampler(ds)
    train_dataloader = DataLoader(ds,
                                  batch_size=batch_size,

                                  pin_memory=True,
                                  sampler=train_sampler)

    def cycle(iterable):
        while True:
            for i in iterable:
                yield i

    train_dataloader = cycle(train_dataloader)

    # 🚀 使用 UnetConcat（Input Concatenation 架构）
    # 🔥 提高模型容量：dim 160→192（Requirements 2.2, 7.1）
    model = InversionNet().to(device)
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
        gap_threshold_high=0.05,  # ⬆️ 放宽gap容忍度到0.15
        gap_threshold_low=0.05,
        dropout_min=0.0,
        dropout_max=0.15,  # ⬇️ 从0.25降到0.15（大幅降低正则化强度）
        dropout_step=0.03  # ⬇️ 从0.05降到0.03（更缓慢的增长）
    )

    # 🚀 初始化自适应损失调度器（Requirements 5.1-5.5）
    # 🔥 优化：60k步后SSIM权重继续增长，突破平台期
    # 用于损失调度的SSIM历史
    train_ssim_for_loss = []
    test_ssim_for_loss = []


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

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    lr_controller = LearningRateController(
        initial_lr=3e-4,  # 初始学习率3e-4（比原来2e-4更激进）
        warmup_steps=10000,  # Warmup 10k步（比原来5k更平滑）
        cosine_period=100000,  # ⬆️ 从80k延长到100k（学习率衰减更慢）
        min_lr_ratio=0.3,  # ⬆️ 从0.2提到0.3（最低学习率更高）
        restart_ratio=0.5,  # 停滞时重启至初始值50%
        decay_ratio=0.5  # 下降时衰减至当前值50%
    )

    # 使用LambdaLR包装LearningRateController
    def lr_lambda_wrapper(step):
        return lr_controller.get_lr(step) / lr_controller.initial_lr

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda_wrapper)

    scaler = torch.cuda.amp.GradScaler()

    start_step = 0  # 默认从 0 开始

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

        print(f"Sampling images at step {step}...")

        ema_model.eval()
        with torch.no_grad():
            # ========================================
            # 1. 评估训练集SSIM（从当前batch采样2个样本）
            # ========================================
            train_data = next(train_dataloader)
            train_x = train_data[0][:2].to(device)  # 取前2个样本

            train_seis = train_data[2][:2].to(device)

            train_ssim_list = []
            # 切换到 eval 模式（BatchNorm 会使用 running statistics）
            model.eval()

            for i in range(2):
                x_gt = train_x[i:i + 1]
                seis_single = train_seis[i:i + 1]

                # 生成预测
                samples = model(x=seis_single)
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
            test_slicer = TestDataSlicer(test_data_dir=r"C:\Users\MSI\Desktop\openfwi\data\test_data")
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
                print(sample_id)

                # 生成预测
                samples = model(x=seis_single)
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

                    samples = model(x=seis_single)
                    pred = samples

                    vis_preds.append(pred)
                    vis_gts.append(vel_gt)

                vis_preds = torch.cat(vis_preds, dim=0)  # (2, 1, 70, 70)
                vis_gts = torch.cat(vis_gts, dim=0)  # (2, 1, 70, 70)

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
    gradient_clip = 1.0
    ema_momentum = 0.99  # EMA 动量
    ssim_history = []  # 存储最近的SSIM值
    base_loss_ema = None

    with tqdm(range(start_step, n_steps), dynamic_ncols=True, initial=start_step, total=n_steps) as pbar:
        pbar.set_description("Training Unet on openfwi")
        for step in pbar:
            data = next(train_dataloader)
            optimizer.zero_grad()

            x1 = data[0].to(device)
            seis = data[2].to(device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred = model.forward(x=seis)

                base_loss = F.l1_loss(pred, x1)
                loss_ssim = criterion_ssim(pred, x1)

                # 保留EMA用于监控
                if base_loss_ema is None:
                    base_loss_ema = base_loss.item()
                else:
                    base_loss_ema = ema_momentum * base_loss_ema + (1 - ema_momentum) * base_loss.item()

                loss = base_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

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

                # 🔥 新增：监控加权后的实际贡献值（方便调参）
                experiment.log_metric("contribution/base", base_contribution, step=step)

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


                # 🚀 灰图检测：记录SSIM历史
                ssim_history.append(train_ssim)


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

