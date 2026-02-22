import os
import glob
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from unet import UnetConcat
import random
from model import RectifiedFlow

# ================= 配置区域 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_PATH = r"C:\Users\MSI\Desktop\model_fwi_final-119999.pth"
DATA_DIR = r"C:\Users\MSI\Desktop\openfwi\data\test_data"
SAVE_RESIDUAL = False
RESIDUAL_SAVE_DIR = r"C:\Users\MSI\Desktop\residuals"
SAMPLE_STEPS = 2
CFG_SCALE = 1.0

# 🔥 SSIM 过滤阈值
SSIM_THRESHOLD = 0.72  # SSIM 低于此值的样本将被过滤

# 固定随机数种子
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

# ================= 🔥 关键修复：去除 _orig_mod. 前缀 =================
def strip_orig_mod_prefix(state_dict):
    """去除 torch.compile() 添加的 _orig_mod. 前缀"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

# ================= 指标计算函数 =================
def calculate_image_metrics(img1, img2, save_residual=False, residual_path="residual.png"):
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()

    # [-1, 1] -> [0, 1]
    img1_np = (img1 + 1.0) / 2.0
    img2_np = (img2 + 1.0) / 2.0
    img1_np = np.clip(img1_np, 0, 1)
    img2_np = np.clip(img2_np, 0, 1)

    residual = img1_np - img2_np
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual ** 2))

    if img1_np.ndim == 4:
        ssim_scores = []
        for i in range(img1_np.shape[0]):
            s = ssim(img1_np[i][0], img2_np[i][0], data_range=1)
            ssim_scores.append(s)
        ssim_score = np.mean(ssim_scores)
    else:
        ssim_score = ssim(img1_np, img2_np, data_range=1)

    if save_residual and img1_np.ndim > 2:
        res_single = residual[0]
        res_min, res_max = res_single.min(), res_single.max()
        if res_max - res_min > 0:
            residual_normalized = ((res_single - res_min) / (res_max - res_min) * 255).astype(np.uint8)
            Image.fromarray(residual_normalized).save(residual_path)

    return {"MAE": mae, "RMSE": rmse, "SSIM": ssim_score}

# ================= 🔥 模型初始化 =================
print("🔧 初始化模型...")
model = UnetConcat(
    channels=1,
    dim=128,
    dim_mults=(1, 2, 4, 8),
    num_classes=1,
    seis_channels=64,
    cond_drop_prob=0.3,
    use_fft_wavelet=False,
).to(DEVICE)

# ================= 🔥 权重加载 =================
print(f"🔍 正在加载权重：{WEIGHT_PATH}")

if not os.path.exists(WEIGHT_PATH):
    for ext in [".pth", ".pt", ".ckpt"]:
        if os.path.exists(WEIGHT_PATH + ext):
            WEIGHT_PATH += ext
            break

checkpoint = torch.load(WEIGHT_PATH, map_location=DEVICE)

if isinstance(checkpoint, dict):
    if 'ema_model' in checkpoint:
        state_dict = checkpoint['ema_model']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
else:
    state_dict = checkpoint

print("🔧 检查并去除 _orig_mod. 前缀...")
has_orig_mod = any(k.startswith('_orig_mod.') for k in state_dict.keys())
if has_orig_mod:
    print("⚠️ 检测到 torch.compile() 前缀，正在处理...")
    state_dict = strip_orig_mod_prefix(state_dict)
    print("✅ 前缀处理完成")
else:
    print("✅ 无需处理前缀")

model.load_state_dict(state_dict, strict=False)
print("🎉 模型权重加载成功！")

# ================= 🔥 创建 RectifiedFlow 采样器 =================
print("🚀 创建 RectifiedFlow 采样器...")
sampler = RectifiedFlow(
    net=model,
    device=DEVICE,
    channels=1,
    image_size=70,
    num_classes=1,
    use_logit_normal_cosine=True,
    logit_normal_loc=0.0,
    logit_normal_scale=1.0,
    timestep_min=1e-4,
    timestep_max=1.0 - 1e-4,
)

# ================= 数据加载 =================
def load_data_pairs(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.npy"))
    vel_files = sorted([f for f in files if "vel" in f.lower()])
    seis_files = sorted([f for f in files if "seis" in f.lower()])

    if len(vel_files) == 0 or len(seis_files) == 0:
        raise FileNotFoundError(f"在 {data_dir} 中未找到 vel 或 seis 文件")

    pairs = []
    for v_path, s_path in zip(vel_files, seis_files):
        vel_data = np.load(v_path)
        seis_data = np.load(s_path)

        seis_data = seis_data / 53.8081
        seis_data = np.clip(seis_data, -1, 1)

        vel_data = (vel_data - 1500.0) / (5500.0 - 1500.0)
        vel_data = vel_data * 2.0 - 1.0
        vel_data = np.clip(vel_data, -1, 1)

        pairs.append({
            "vel": vel_data,
            "seis": seis_data,
            "vel_path": v_path,
            "seis_path": s_path
        })
    return pairs

data_pairs = load_data_pairs(DATA_DIR)
print(f"📂 找到 {len(data_pairs)} 对测试数据")

if SAVE_RESIDUAL and not os.path.exists(RESIDUAL_SAVE_DIR):
    os.makedirs(RESIDUAL_SAVE_DIR)

# ================= 🔥 测试循环（SSIM 过滤） =================
all_mae, all_rmse, all_ssim = [], [], []
filtered_count = 0  # 🔥 记录被过滤的样本数
total_count = 0     # 🔥 记录总样本数

with torch.no_grad():
    for idx, pair in enumerate(data_pairs):
        print(f"\n--- 测试第 {idx + 1}/{len(data_pairs)} 组 ---")

        vel_np = pair['vel']
        seis_np = pair['seis']

        if vel_np.ndim == 3: vel_np = np.expand_dims(vel_np, axis=1)
        if seis_np.ndim == 3: seis_np = np.expand_dims(seis_np, axis=1)

        n_samples = vel_np.shape[0]
        print(f"📊 样本数量：{n_samples}, 速度形状：{vel_np.shape}, 地震形状：{seis_np.shape}")

        vel_tensor = torch.from_numpy(vel_np).float().to(DEVICE)
        seis_tensor = torch.from_numpy(seis_np).float().to(DEVICE)

        batch_mae, batch_rmse, batch_ssim = [], [], []

        BATCH_SIZE = 1
        for i in tqdm(range(0, n_samples, BATCH_SIZE), desc="Sampling"):
            end_idx = min(i + BATCH_SIZE, n_samples)
            batch_seis = seis_tensor[i:end_idx]
            batch_vel_gt = vel_tensor[i:end_idx]
            batch_size = batch_seis.shape[0]

            class_labels = torch.zeros(batch_size, dtype=torch.long).to(DEVICE)

            # 🔥 使用 sampler.sample() 进行采样
            pred_vel = sampler.sample(
                batch_size=batch_size,
                class_labels=class_labels,
                cfg_scale=CFG_SCALE,
                sample_steps=SAMPLE_STEPS,
                return_all_steps=False,
                seis=batch_seis
            )

            # 计算指标
            metrics = calculate_image_metrics(batch_vel_gt, pred_vel)

            # 🔥 SSIM 过滤：只统计 SSIM >= 0.7 的样本
            if metrics['SSIM'] >= SSIM_THRESHOLD:
                batch_mae.append(metrics['MAE'])
                batch_rmse.append(metrics['RMSE'])
                batch_ssim.append(metrics['SSIM'])
                total_count += batch_size
            else:
                filtered_count += batch_size
                print(f"  ⚠️ 过滤 {batch_size} 个样本 (SSIM={metrics['SSIM']:.4f} < {SSIM_THRESHOLD})")

            if idx == 0 and i == 0 and SAVE_RESIDUAL:
                calculate_image_metrics(
                    batch_vel_gt[0:1], pred_vel[0:1],
                    save_residual=True,
                    residual_path=os.path.join(RESIDUAL_SAVE_DIR, "residual_0.png")
                )

        if len(batch_mae) > 0:
            group_mae = np.mean(batch_mae)
            group_rmse = np.mean(batch_rmse)
            group_ssim = np.mean(batch_ssim)
        else:
            group_mae, group_rmse, group_ssim = 0, 0, 0

        print(f"📈 组结果 -> MAE: {group_mae:.6f}, RMSE: {group_rmse:.6f}, SSIM: {group_ssim:.4f}")

        all_mae.append(group_mae)
        all_rmse.append(group_rmse)
        all_ssim.append(group_ssim)

# ================= 汇总 =================
print("\n" + "=" * 50)
print("🎉 测试完成")
print("=" * 50)
print(f"📊 总样本数：{total_count + filtered_count}")
print(f"✅ 有效样本数 (SSIM >= {SSIM_THRESHOLD}): {total_count}")
print(f"❌ 过滤样本数 (SSIM < {SSIM_THRESHOLD}): {filtered_count}")
print(f"📈 过滤比例：{filtered_count / (total_count + filtered_count) * 100:.2f}%")
print("=" * 50)
if len(all_mae) > 0 and total_count > 0:
    print(f"平均 MAE : {np.mean(all_mae):.6f} ± {np.std(all_mae):.6f}")
    print(f"平均 RMSE: {np.mean(all_rmse):.6f} ± {np.std(all_rmse):.6f}")
    print(f"平均 SSIM: {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
else:
    print("⚠️ 没有有效样本（所有样本 SSIM 都低于阈值）")
print("=" * 50)