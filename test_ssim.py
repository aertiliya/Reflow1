"""
SSIM测试脚本 - 用于评估FWI模型生成质量
测试生成样本与真实数据的结构相似性
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, TensorDataset
import os
import glob
import scipy.io
import cv2
from tqdm import tqdm

from model import RectifiedFlow
from unet import Unet


def resize_data(data, target_height, target_width, interpolation=cv2.INTER_LINEAR):
    """使用OpenCV对数据进行插值缩放"""
    if len(data.shape) == 2:
        resized_data = cv2.resize(data, (target_width, target_height), interpolation=interpolation)
    else:
        resized_data = cv2.resize(data, (target_width, target_height), interpolation=interpolation)
    return resized_data


def load_test_data(data_folder, class_dict, image_size=256):
    """加载测试数据集及对应的地震数据"""
    print("Loading test data...")
    test_x, test_y, test_seis = [], [], []
    
    for ci, c_v in enumerate(class_dict):
        if c_v == 'CurveVelB':
            # 加载CurveVelB数据（.npy格式）
            folder = data_folder
            vmodel_path = os.path.join(folder, c_v)
            seis_path = os.path.join(folder, 'seis_2CurveFaultB')
            print(f'Loading velocity models from: {vmodel_path}')
            print(f'Loading seismic data from: {seis_path}')
            
            files_path = glob.glob(os.path.join(vmodel_path, '*.npy'))
            
            if not files_path:
                print(f'Warning: No .npy files found in {vmodel_path}')
                continue
            
            print(f'Found {len(files_path)} .npy files, loading all {len(files_path)} files...')
            for f_v in files_path:
                x_tmp = np.load(f_v).reshape(-1, 70, 70)[:, :image_size, :image_size]
                
                if x_tmp.shape[0] > 0:
                    for sample_idx in range(x_tmp.shape[0]):
                        sample = x_tmp[sample_idx]
                        
                        # 调整到目标图像大小
                        if sample.shape != (image_size, image_size):
                            sample = resize_data(sample, image_size, image_size, interpolation=cv2.INTER_LINEAR)
                        
                        sample = sample[np.newaxis, :, :]
                        
                        # 加载对应的地震数据
                        base_name = os.path.splitext(os.path.basename(f_v))[0]
                        seis_file = os.path.join(seis_path, base_name + '.npy')
                        
                        seis_tmp = None
                        if os.path.exists(seis_file):
                            try:
                                seis_tmp = np.load(seis_file).reshape(-1, 5, 1000, 70)
                                if seis_tmp.shape[0] > sample_idx:
                                    seis_tmp = seis_tmp[sample_idx]
                                else:
                                    seis_tmp = seis_tmp[0]
                            except Exception as e:
                                print(f"  Error loading seismic data {seis_file}: {e}")
                                seis_tmp = None
                        
                        if seis_tmp is None:
                            seis_tmp = np.zeros((5, 1000, 70), dtype=np.float32)
                        
                        test_x.append(sample)
                        test_y.append(np.ones(1) * ci)
                        test_seis.append(seis_tmp)
        
        elif c_v == 'SimulateData':
            folder = data_folder
            vmodel_path = os.path.join(folder, c_v, 'vmodel_train')
            seis_path = os.path.join(folder, c_v, 'seis_train')
            print(f'Loading velocity models from: {vmodel_path}')
            print(f'Loading seismic data from: {seis_path}')
            
            files_path = glob.glob(os.path.join(vmodel_path, '*.mat'))
            if not files_path:
                vmodel_path = os.path.join(folder, c_v, c_v, 'vmodel_train')
                seis_path = os.path.join(folder, c_v, c_v, 'seis_train')
                files_path = glob.glob(os.path.join(vmodel_path, '*.mat'))
            
            if not files_path:
                print(f'Warning: No .mat files found in {vmodel_path}')
                continue
            
            print(f'Found {len(files_path)} .mat files, loading all {len(files_path)} files...')
            for f_v in files_path:  # 加载所有样本
                mat_data = scipy.io.loadmat(f_v)
                x_tmp = None
                
                if 'vmodel' in mat_data:
                    x_tmp = mat_data['vmodel']
                else:
                    for key in mat_data.keys():
                        if not key.startswith('__'):
                            potential_data = mat_data[key]
                            if isinstance(potential_data, np.ndarray) and len(potential_data.shape) >= 2:
                                x_tmp = potential_data
                                break
                
                if x_tmp is not None:
                    # 添加数据形状检测
                    print(f"  Data shape: {x_tmp.shape}")
                    
                    # 调整到目标图像大小
                    if x_tmp.shape != (image_size, image_size):
                        x_tmp = resize_data(x_tmp, image_size, image_size, interpolation=cv2.INTER_LINEAR)
                        print(f"  Resized to {image_size}x{image_size}: {x_tmp.shape}")
                    
                    x_tmp = x_tmp[np.newaxis, :, :]
                    print(f"  Final data shape: {x_tmp.shape}")
                    
                    # 加载对应的地震数据
                    base_name = os.path.splitext(os.path.basename(f_v))[0]
                    seis_base_name = base_name.replace('vmodel', 'seismic')
                    seis_file = os.path.join(seis_path, seis_base_name + '.bin')
                    
                    seis_tmp = None
                    if os.path.exists(seis_file):
                        try:
                            seis_tmp = np.fromfile(seis_file, dtype=np.float32)
                            # 尝试标准形状 (5, 1000, 301)
                            expected_size = 5 * 1000 * 301
                            if seis_tmp.size == expected_size:
                                seis_tmp = seis_tmp.reshape((5, 1000, 301))
                                print(f"  Loaded seismic data: {seis_base_name}.bin, shape: {seis_tmp.shape}")
                            else:
                                print(f"  Warning: Seismic data size mismatch, expected {expected_size}, got {seis_tmp.size}")
                                seis_tmp = None
                        except Exception as e:
                            print(f"  Error loading seismic data {seis_file}: {e}")
                            seis_tmp = None
                    else:
                        print(f"  Warning: Seismic file not found: {seis_file}")
                    
                    # 如果没有地震数据，创建零数组
                    if seis_tmp is None:
                        seis_tmp = np.zeros((5, 1000, 301), dtype=np.float32)
                    
                    test_x.append(x_tmp)
                    test_y.append(np.ones(x_tmp.shape[0]) * ci)
                    test_seis.append(seis_tmp)
    
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)
    
    # 处理test_seis的堆叠 - 可能有不同的形状
    if len(test_seis) > 0:
        # 检查所有地震数据的形状
        seis_shapes = [s.shape for s in test_seis]
        print(f"  Seismic data shapes: {set(seis_shapes)}")
        
        # 如果形状不一致，需要统一
        if len(set(seis_shapes)) > 1:
            # 找到最大的形状
            max_shape = tuple(max(s[i] for s in seis_shapes) for i in range(len(seis_shapes[0])))
            print(f"  Unifying seismic data to shape: {max_shape}")
            
            # 将所有数据填充到最大形状
            test_seis_unified = []
            for s in test_seis:
                if s.shape != max_shape:
                    padded = np.zeros(max_shape, dtype=s.dtype)
                    slices = tuple(slice(0, min(s.shape[i], max_shape[i])) for i in range(len(s.shape)))
                    padded[slices] = s[slices]
                    test_seis_unified.append(padded)
                else:
                    test_seis_unified.append(s)
            test_seis = np.stack(test_seis_unified, axis=0)
        else:
            test_seis = np.stack(test_seis, axis=0)
    
    # 添加整体数据形状检测
    print(f"  Concatenated velocity models shape: {test_x.shape}")
    print(f"  Concatenated seismic data shape: {test_seis.shape}")
    
    # 归一化到[0, 1]
    xmin, xmax = test_x.min(), test_x.max()
    test_x = (test_x - xmin) / (xmax - xmin)
    
    # 归一化地震数据到[-1, 1]
    seis_min, seis_max = test_seis.min(), test_seis.max()
    if seis_max > seis_min:
        test_seis = 2 * (test_seis - seis_min) / (seis_max - seis_min) - 1
    
    return test_x, test_y, test_seis, xmin, xmax


def calculate_ssim_batch(real_images, generated_images):
    """批量计算SSIM分数"""
    ssim_scores = []
    
    for i in range(len(real_images)):
        real_img = real_images[i].squeeze()  # 移除通道维度
        gen_img = generated_images[i].squeeze()
        
        # 添加数据范围检查
        real_min, real_max = real_img.min(), real_img.max()
        gen_min, gen_max = gen_img.min(), gen_img.max()
        
        if i == 0:
            print(f"    Real image range: [{real_min:.4f}, {real_max:.4f}]")
            print(f"    Generated image range: [{gen_min:.4f}, {gen_max:.4f}]")
        
        # 检查数据有效性
        if np.isnan(real_img).any() or np.isnan(gen_img).any():
            print(f"    Warning: NaN values found in sample {i}")
            # 将NaN替换为0
            real_img = np.nan_to_num(real_img)
            gen_img = np.nan_to_num(gen_img)
        
        if np.isinf(real_img).any() or np.isinf(gen_img).any():
            print(f"    Warning: Inf values found in sample {i}")
            # 将Inf替换为对应边界值
            real_img = np.nan_to_num(real_img)
            gen_img = np.nan_to_num(gen_img)
        
        # 计算SSIM (data_range=1.0 因为数据已归一化到[0,1])
        score = ssim(real_img, gen_img, data_range=1.0)
        ssim_scores.append(score)
    
    return np.array(ssim_scores)


def main():
    # 配置参数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "Mean-rwkv7/Main/Test_project/Reflow-main/weights/step_100000.pth"
    data_folder = "Mean-rwkv7/Main/Test_project/Reflow-main/data/train_data"
    class_dict = ['SimulateData']
    image_size = 64
    image_channels = 1
    
    # 先加载检查点以获取实际的 num_classes
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'ema_model' in checkpoint:
        # 从 EMA 模型的 classes_emb.weight 推断 num_classes
        num_classes = checkpoint['ema_model']['classes_emb.weight'].shape[0]
    elif 'CurveVelB' in checkpoint:
        num_classes = checkpoint['CurveVelB']['classes_emb.weight'].shape[0]
    else:
        # 备选方案：尝试从任何包含 classes_emb 的键推断
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict) and 'classes_emb.weight' in checkpoint[key]:
                num_classes = checkpoint[key]['classes_emb.weight'].shape[0]
                break
        else:
            num_classes = len(class_dict)  # 降级到默认值
    
    print(f"Detected num_classes from checkpoint: {num_classes}")
    
    # 检查数据形状是否需要调整
    print(f"Target image size: {image_size}x{image_size}")
    
    # 创建输出目录
    output_dir = "Mean-rwkv7/Main/Test_project/Reflow-main/results/ssim_test"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # 加载模型 - 必须与训练时的结构一致
    model = Unet(
        channels=image_channels,
        dim=64,
        dim_mults=(1, 2, 4, 8),  
        num_classes=num_classes,
    ).to(device)
    
    # 加载权重（checkpoint 已在前面加载过）
    if 'ema_model' in checkpoint:
        model.load_state_dict(checkpoint['ema_model'])
        print("Loaded EMA model weights")
    elif 'CurveVelB' in checkpoint:
        model.load_state_dict(checkpoint['CurveVelB'])
        print("Loaded model weights")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded direct state dict")
    
    model.eval()
    
    # 创建采样器
    sampler = RectifiedFlow(
        model,
        device=device,
        channels=image_channels,
        image_size=image_size,
        num_classes=num_classes,
        use_logit_normal_cosine=True,
    )
    
    # 加载测试数据
    test_x, test_y, test_seis, xmin, xmax = load_test_data(data_folder, class_dict, image_size)
    print(f"Loaded {len(test_x)} test samples")
    
    # 添加数据形状检测
    print(f"Test data shape: {test_x.shape}")
    print(f"Test labels shape: {test_y.shape}")
    
    # 生成样本并计算SSIM - 使用全部样本，无条件采样
    num_samples = len(test_x)  # 测试全部样本
    print(f"Will test on all {num_samples} samples (unconditional sampling)")
    
    print(f"\n{'='*60}")
    print(f"Testing with unconditional sampling (no CFG)")
    print(f"{'='*60}")
    
    ssim_scores_all = []
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Generating samples"):
            # 获取真实样本和地震数据
            real_sample = test_x[i:i+1]
            real_seis = torch.from_numpy(test_seis[i:i+1]).to(device).float()
            class_label = int(test_y[i])
            
            # 添加输入数据形状检测
            if i == 0:
                print(f"  Input real sample shape: {real_sample.shape}")
                print(f"  Input seismic data shape: {real_seis.shape}")
            
            # 无条件生成样本 - 不使用CFG
            class_tensor = torch.tensor([class_label], device=device)
            generated = sampler.sample(
                batch_size=None,
                class_labels=class_tensor,
                cfg_scale=1.0,  # CFG=1.0 等同于无条件采样
                sample_steps=5,
                return_all_steps=False,
                seis=real_seis
            )
            
            # 添加模型输出形状检测
            if i == 0:
                print(f"  Generated sample shape: {generated.shape}")
            
            # 转换为numpy
            gen_np = generated.cpu().numpy()
            
            # 计算SSIM
            ssim_score = calculate_ssim_batch(real_sample, gen_np)
            ssim_scores_all.append(ssim_score[0])
    
    ssim_scores_all = np.array(ssim_scores_all)
    
    # 打印统计信息
    print(f"\nSSIM Statistics (unconditional):")
    print(f"  Mean: {ssim_scores_all.mean():.4f}")
    print(f"  Std:  {ssim_scores_all.std():.4f}")
    print(f"  Min:  {ssim_scores_all.min():.4f}")
    print(f"  Max:  {ssim_scores_all.max():.4f}")
    
    # 可视化结果
    plt.figure(figsize=(12, 6))
    
    # SSIM分布直方图
    plt.subplot(1, 2, 1)
    plt.hist(ssim_scores_all, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('SSIM Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('SSIM Score Distribution (Unconditional)', fontsize=14, fontweight='bold')
    plt.axvline(ssim_scores_all.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={ssim_scores_all.mean():.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # SSIM统计箱线图
    plt.subplot(1, 2, 2)
    plt.boxplot([ssim_scores_all], labels=['Unconditional'])
    plt.ylabel('SSIM Score', fontsize=12)
    plt.title('SSIM Score Statistics', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ssim_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved distribution plot to {output_dir}/ssim_distribution.png")
    
    # 保存详细结果
    with open(f'{output_dir}/ssim_results.txt', 'w') as f:
        f.write("SSIM Test Results (Unconditional Sampling)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total samples: {len(ssim_scores_all)}\n")
        f.write(f"Mean SSIM: {ssim_scores_all.mean():.4f}\n")
        f.write(f"Std SSIM:  {ssim_scores_all.std():.4f}\n")
        f.write(f"Min SSIM:  {ssim_scores_all.min():.4f}\n")
        f.write(f"Max SSIM:  {ssim_scores_all.max():.4f}\n")
        f.write(f"Median SSIM: {np.median(ssim_scores_all):.4f}\n")
        
        # 添加百分位数
        f.write(f"\nPercentiles:\n")
        f.write(f"  25th: {np.percentile(ssim_scores_all, 25):.4f}\n")
        f.write(f"  50th: {np.percentile(ssim_scores_all, 50):.4f}\n")
        f.write(f"  75th: {np.percentile(ssim_scores_all, 75):.4f}\n")
        f.write(f"  90th: {np.percentile(ssim_scores_all, 90):.4f}\n")
        f.write(f"  95th: {np.percentile(ssim_scores_all, 95):.4f}\n")
    
    print(f"Saved detailed results to {output_dir}/ssim_results.txt")
    
    # 生成对比可视化
    print("\nGenerating visual comparisons...")
    visualize_comparisons(sampler, test_x, test_y, test_seis, ssim_scores_all, output_dir, num_vis=10)
    
    # 生成带原图的采样对比 - 多个 cfg_scale
    print("\nGenerating samples with original images...")
    for cfg_scale in [1.0, 2.5, 5.0]:
        save_samples_with_originals(sampler, test_x, test_y, test_seis, output_dir, num_samples=20, cfg_scale=cfg_scale)
    
    print("\n" + "="*60)
    print("SSIM测试完成！")
    print("="*60)


def visualize_comparisons(sampler, test_x, test_y, test_seis, ssim_scores_all, output_dir, num_vis=10):
    """为每个cfg_scale生成可视化对比：最好、最坏和随机的样本"""
    cfg_scales = [1.0, 2.5, 5.0]
    
    for cfg_scale in cfg_scales:
        print(f"\nGenerating visualization for cfg_scale={cfg_scale}...")
        
        # 计算该cfg_scale下的SSIM分数
        ssim_scores = []
        with torch.no_grad():
            for i in tqdm(range(len(test_x)), desc=f"Computing SSIM for cfg={cfg_scale}"):
                real_sample = test_x[i]
                class_label = int(test_y[i])
                real_seis = torch.from_numpy(test_seis[i:i+1]).to(sampler.device).float()
                
                class_tensor = torch.tensor([class_label], device=sampler.device)
                generated = sampler.sample(
                    batch_size=None,
                    class_labels=class_tensor,
                    cfg_scale=cfg_scale,
                    sample_steps=5,
                    return_all_steps=False,
                    seis=real_seis
                )
                
                gen_np = generated.cpu().numpy().squeeze()
                real_np = real_sample.squeeze()
                score = ssim(real_np, gen_np, data_range=1.0)
                ssim_scores.append(score)
        
        ssim_scores = np.array(ssim_scores)
        all_samples_mean = ssim_scores.mean()
        all_samples_std = ssim_scores.std()
        
        # 找到最好、最坏的索引
        best_idx = np.argmax(ssim_scores)
        worst_idx = np.argmin(ssim_scores)
        
        # 随机选择8个其他样本
        other_indices = np.random.choice(
            [i for i in range(len(test_x)) if i != best_idx and i != worst_idx],
            size=min(8, len(test_x) - 2),
            replace=False
        )
        
        # 组合索引：最好、最坏、随机8个
        selected_indices = [best_idx, worst_idx] + list(other_indices)
        selected_ssim_scores = ssim_scores[selected_indices]
        selected_mean = selected_ssim_scores.mean()
        selected_std = selected_ssim_scores.std()
        
        # 创建可视化
        fig, axes = plt.subplots(len(selected_indices), 3, figsize=(12, 4*len(selected_indices)))
        if len(selected_indices) == 1:
            axes = axes.reshape(1, -1)
        
        with torch.no_grad():
            for row, idx in enumerate(selected_indices):
                real_sample = test_x[idx]
                class_label = int(test_y[idx])
                real_seis = torch.from_numpy(test_seis[idx:idx+1]).to(sampler.device).float()
                
                class_tensor = torch.tensor([class_label], device=sampler.device)
                generated = sampler.sample(
                    batch_size=None,
                    class_labels=class_tensor,
                    cfg_scale=cfg_scale,
                    sample_steps=5,
                    return_all_steps=False,
                    seis=real_seis
                )
                
                gen_np = generated.cpu().numpy().squeeze()
                real_np = real_sample.squeeze()
                score = ssim(real_np, gen_np, data_range=1.0)
                diff = np.abs(real_np - gen_np)
                
                # 标记最好和最坏
                if idx == best_idx:
                    title_prefix = "🏆 BEST"
                elif idx == worst_idx:
                    title_prefix = "💔 WORST"
                else:
                    title_prefix = "Random"
                
                axes[row, 0].imshow(real_np, cmap='seismic', vmin=0, vmax=1)
                axes[row, 0].set_title(f'Real (Class {class_label})')
                axes[row, 0].axis('off')
                
                axes[row, 1].imshow(gen_np, cmap='seismic', vmin=0, vmax=1)
                axes[row, 1].set_title(f'{title_prefix} - SSIM={score:.4f}')
                axes[row, 1].axis('off')
                
                axes[row, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
                axes[row, 2].set_title('Difference')
                axes[row, 2].axis('off')
        
        plt.suptitle(f'Visualization for cfg_scale={cfg_scale}\nAll Samples: Mean={all_samples_mean:.4f}±{all_samples_std:.4f} | Selected 10: Mean={selected_mean:.4f}±{selected_std:.4f}', 
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/visual_comparison_cfg{cfg_scale}.png', dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_dir}/visual_comparison_cfg{cfg_scale}.png")
        print(f"  Best SSIM: {ssim_scores[best_idx]:.4f} (idx={best_idx})")
        print(f"  Worst SSIM: {ssim_scores[worst_idx]:.4f} (idx={worst_idx})")
        print(f"  All Samples Mean SSIM: {all_samples_mean:.4f} ± {all_samples_std:.4f}")
        print(f"  Selected 10 Samples Mean SSIM: {selected_mean:.4f} ± {selected_std:.4f}")
        plt.close()


def save_samples_with_originals(sampler, test_x, test_y, test_seis, output_dir, num_samples=20, cfg_scale=2.5):
    """保存生成样本，并在下方显示对应的真实原图，同时计算SSIM"""
    print(f"\nGenerating {num_samples} samples with original images (cfg={cfg_scale})...")
    
    generated_samples = []
    real_samples = []
    ssim_scores = []
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc=f"Generating & Computing SSIM (cfg={cfg_scale})"):
            real_sample = test_x[i]
            class_label = int(test_y[i])
            real_seis = torch.from_numpy(test_seis[i:i+1]).to(sampler.device).float()
            
            # 添加输入数据形状检测
            if i == 0:
                print(f"  Input real sample shape: {real_sample.shape}")
                print(f"  Input seismic data shape: {real_seis.shape}")
            
            # 生成样本 - 使用5步采样
            class_tensor = torch.tensor([class_label], device=sampler.device)
            generated = sampler.sample(
                batch_size=None,
                class_labels=class_tensor,
                cfg_scale=cfg_scale,
                sample_steps=5,
                return_all_steps=False,
                seis=real_seis
            )
            
            # 添加输出数据形状检测
            if i == 0:
                print(f"  Generated sample shape: {generated.shape}")
            
            # 立即计算SSIM
            gen_np = generated.cpu().numpy().squeeze()
            real_np = real_sample.squeeze()
            
            # 添加处理后数据形状检测
            if i == 0:
                print(f"  Processed real sample shape: {real_np.shape}")
                print(f"  Processed generated sample shape: {gen_np.shape}")
            
            ssim_score = ssim(real_np, gen_np, data_range=1.0)
            ssim_scores.append(ssim_score)
            
            generated_samples.append(generated)
            # 确保real_sample具有正确的维度，与generated保持一致
            real_tensor = torch.from_numpy(real_sample).unsqueeze(0)
            if real_tensor.dim() != generated.dim():
                # 如果维度不匹配，调整real_tensor的维度
                while real_tensor.dim() < generated.dim():
                    real_tensor = real_tensor.unsqueeze(0)
            real_samples.append(real_tensor)
    
    # 转换为tensor - 确保都在同一设备上
    generated_tensor = torch.cat(generated_samples, dim=0)  # 已经在CUDA上
    real_tensor = torch.cat(real_samples, dim=0).to(sampler.device)  # 移到CUDA上
    ssim_scores = np.array(ssim_scores)
    
    # 打印SSIM统计
    print(f"  SSIM Stats: Mean={ssim_scores.mean():.4f}, Std={ssim_scores.std():.4f}, Min={ssim_scores.min():.4f}, Max={ssim_scores.max():.4f}")
    
    # 创建2行图像：上行是生成的，下行是真实的（现在都在同一设备上）
    # 确保两个张量维度相同后再连接
    if generated_tensor.dim() != real_tensor.dim():
        # 如果仍然维度不一致，则调整较小维度的张量
        while generated_tensor.dim() < real_tensor.dim():
            generated_tensor = generated_tensor.unsqueeze(0)
        while real_tensor.dim() < generated_tensor.dim():
            real_tensor = real_tensor.unsqueeze(0)
    
    combined = torch.cat([generated_tensor, real_tensor], dim=0)
    
    # 创建网格图像 - 3行：生成样本、真实样本、SSIM分数
    nrow = num_samples
    fig = plt.figure(figsize=(num_samples * 2, 6))
    
    # 上行：生成样本
    ax1 = plt.subplot(3, 1, 1)
    gen_grid = make_grid_numpy(generated_tensor, nrow=nrow)
    ax1.imshow(gen_grid, cmap='seismic', vmin=0, vmax=1)
    ax1.set_title(f'Generated Samples (CFG={cfg_scale}, Avg SSIM={ssim_scores.mean():.4f})', 
                     fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 中行：真实样本
    ax2 = plt.subplot(3, 1, 2)
    real_grid = make_grid_numpy(real_tensor, nrow=nrow)
    ax2.imshow(real_grid, cmap='seismic', vmin=0, vmax=1)
    ax2.set_title('Original Real Samples', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 下行：SSIM分数柱状图
    ax3 = plt.subplot(3, 1, 3)
    colors = ['green' if score > ssim_scores.mean() else 'red' for score in ssim_scores]
    ax3.bar(range(len(ssim_scores)), ssim_scores, color=colors, alpha=0.7)
    ax3.axhline(y=ssim_scores.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean={ssim_scores.mean():.4f}')
    ax3.set_xlabel('Sample Index', fontsize=12)
    ax3.set_ylabel('SSIM Score', fontsize=12)
    ax3.set_title('Individual SSIM Scores', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = f'{output_dir}/samples_with_originals_cfg{cfg_scale}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved samples with originals to {save_path}")
    plt.close()
    
    # 同时保存单独的对比图（每个样本一组）- 使用已计算的SSIM
    for i in range(min(20, num_samples)):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        gen_np = generated_tensor[i].cpu().numpy().squeeze()
        real_np = real_tensor[i].cpu().numpy().squeeze()
        diff = np.abs(gen_np - real_np)
        
        # 标记是否高于平均值
        is_above_mean = ssim_scores[i] > ssim_scores.mean()
        color = 'green' if is_above_mean else 'red'
        
        axes[0].imshow(real_np, cmap='seismic', vmin=0, vmax=1)
        axes[0].set_title('Original', fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(gen_np, cmap='seismic', vmin=0, vmax=1)
        axes[1].set_title(f'Generated\nSSIM={ssim_scores[i]:.4f}', fontsize=12, color=color, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[2].set_title('Difference', fontsize=12)
        axes[2].axis('off')
        
        plt.suptitle(f'Sample {i} (cfg={cfg_scale})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sample_cfg{cfg_scale}_id{i}.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {min(20, num_samples)} individual comparison images")
    
    return ssim_scores


def make_grid_numpy(tensor, nrow=8, padding=2):
    """将tensor转换为网格numpy数组用于显示"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(np.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    
    grid = np.ones((height * ymaps + padding, width * xmaps + padding)) * 0.5
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            img = tensor[k].cpu().numpy().squeeze()
            grid[y * height + padding:(y + 1) * height, 
                 x * width + padding:(x + 1) * width] = img
            k += 1
    
    return grid


if __name__ == "__main__":
    main()
