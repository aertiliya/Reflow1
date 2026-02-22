import os
from datetime import datetime  # 新增：用于 TXT 文件时间戳
import glob
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from unet import UnetConcat
import random
from model import RectifiedFlow
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # 新增：用于多图布局

# ================= 配置区域 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🔥 同时测量两个权重（contrast=0 和 contrast=0.5）
WEIGHT_PATHS = {
    "contrast_0":   r"C:\Users\aertlia\Desktop\zako\AI_study\Reflow1\test\weights\model_fwi_final-119999.pth",      # contrast损失权重=0
    "contrast_0.5": r"C:\Users\aertlia\Desktop\zako\AI_study\Reflow1\test\weights\model_fwi_final-119999_0.5.pth",  # contrast损失权重=0.5
}

DATA_DIR    = r"C:\Users\aertlia\Desktop\zako\AI_study\Reflow1\test\test_data"
SAVE_DIR    = r"C:\Users\aertlia\Desktop\zako\AI_study\Reflow1\test\residuals"
SAMPLE_STEPS = 5
CFG_SCALE    = 1.0
BATCH_SIZE      = 8      # 批量推理大小
SSIM_THRESHOLD  = 0.78   # 🔥 SSIM 最低标准：contrast=0.5 必须 >= 此值才参与最大差值筛选


# ================= 固定随机种子 =================
def seed_torch(seed=41):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()


# ================= 去除 _orig_mod. 前缀 =================
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


# ================= 指标计算 =================
def calculate_image_metrics(img1, img2):
    """计算 MAE、RMSE、SSIM，支持 Tensor 或 ndarray 输入"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()

    # 反归一化到 [0, 1]
    img1_np = np.clip((img1 + 1.0) / 2.0, 0, 1)
    img2_np = np.clip((img2 + 1.0) / 2.0, 0, 1)

    residual = img1_np - img2_np
    mae  = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual ** 2))

    # 按批计算 SSIM
    if img1_np.ndim == 4:
        ssim_scores = [ssim(img1_np[i][0], img2_np[i][0], data_range=1)
                       for i in range(img1_np.shape[0])]
        ssim_score = np.mean(ssim_scores)
    else:
        ssim_score = ssim(img1_np, img2_np, data_range=1)

    return {"MAE": mae, "RMSE": rmse, "SSIM": ssim_score}


# ================= 模型构建 & 权重加载工具 =================
def build_model(weight_path: str) -> RectifiedFlow:
    """构建 UNet + RectifiedFlow，加载指定权重，返回采样器"""
    print(f"  🔧 构建 UNet...")
    model = UnetConcat(
        channels=1,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        num_classes=1,
        seis_channels=64,
        cond_drop_prob=0.3,
        use_fft_wavelet=False,
    ).to(DEVICE)

    print(f"  🔍 加载权重：{os.path.basename(weight_path)}")
    checkpoint = torch.load(weight_path, map_location=DEVICE)

    # 自动提取 state_dict
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

    # 去除 _orig_mod. 前缀（torch.compile 遗留）
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("  ⚠️ 检测到 torch.compile() 前缀，正在处理...")
        state_dict = strip_orig_mod_prefix(state_dict)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

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
    print(f"  ✅ 权重加载完成")
    return sampler


# ================= 数据加载 =================
def load_data_pairs(data_dir: str) -> list:
    """读取 data_dir 下所有 npy 文件组成 vel/seis 配对列表"""
    files     = glob.glob(os.path.join(data_dir, "*.npy"))
    vel_files  = sorted([f for f in files if "vel"  in os.path.basename(f).lower()])
    seis_files = sorted([f for f in files if "seis" in os.path.basename(f).lower()])

    if not vel_files or not seis_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到 vel 或 seis 文件")

    pairs = []
    for v_path, s_path in zip(vel_files, seis_files):
        vel_data  = np.load(v_path)
        seis_data = np.load(s_path)

        # 归一化
        seis_data = np.clip(seis_data / 53.8081, -1, 1)
        vel_data  = np.clip((vel_data - 1500.0) / (5500.0 - 1500.0) * 2.0 - 1.0, -1, 1)

        pairs.append({
            "vel":       vel_data,
            "seis":      seis_data,
            "vel_path":  v_path,
            "seis_path": s_path,
        })
    return pairs


# =====================================
# 🔥 批量对比测试类
# =====================================
class DualWeightBatchTester:
    """
    批量测试 contrast=0 和 contrast=0.5 两个权重的 SSIM，
    并找出 contrast=0.5 相对 contrast=0 SSIM 增益最高的样本。
    """

    def __init__(self, samplers: dict, data_pairs: list, save_dir: str,
                 sample_steps: int = 5, cfg_scale: float = 1.0, batch_size: int = 8):
        self.samplers     = samplers      # {"contrast_0": sampler, "contrast_0.5": sampler}
        self.data_pairs   = data_pairs    # 数据配对列表
        self.save_dir     = save_dir
        self.sample_steps = sample_steps
        self.cfg_scale    = cfg_scale
        self.batch_size   = batch_size
        os.makedirs(save_dir, exist_ok=True)

        # 存储每个样本对每个权重的 SSIM 列表，格式：{key: [ssim0, ssim1, ...]}
        self.all_ssim = {k: [] for k in samplers.keys()}
        # 存储原始 (vel_gt, pred_vel, global_idx) 供可视化
        # 格式: {key: [(vel_gt_np, pred_vel_np, global_idx, ssim_val), ...]}
        self.sample_records = {k: [] for k in samplers.keys()}

    # -------- 单模型全量推理 --------
    def _infer_one_model(self, key: str):
        """对一个权重模型跑完所有数据对，记录逐样本 SSIM"""
        sampler = self.samplers[key]
        ssim_list   = []
        record_list = []

        global_idx = 0  # 跨文件的全局样本编号

        for pair_i, pair in enumerate(self.data_pairs):
            vel_np  = pair["vel"]
            seis_np = pair["seis"]

            if vel_np.ndim  == 3: vel_np  = np.expand_dims(vel_np,  axis=1)
            if seis_np.ndim == 3: seis_np = np.expand_dims(seis_np, axis=1)

            n_samples = vel_np.shape[0]
            vel_tensor  = torch.from_numpy(vel_np).float().to(DEVICE)
            seis_tensor = torch.from_numpy(seis_np).float().to(DEVICE)

            print(f"  [{key}] 文件 {pair_i+1}/{len(self.data_pairs)}，共 {n_samples} 个样本")

            with torch.no_grad():
                for i in tqdm(range(0, n_samples, self.batch_size),
                               desc=f"    Sampling [{key}]"):
                    end_idx = min(i + self.batch_size, n_samples)
                    b_seis  = seis_tensor[i:end_idx]
                    b_vel   = vel_tensor[i:end_idx]
                    bs      = b_seis.shape[0]

                    class_labels = torch.zeros(bs, dtype=torch.long).to(DEVICE)

                    pred_vel = sampler.sample(
                        batch_size=bs,
                        class_labels=class_labels,
                        cfg_scale=self.cfg_scale,
                        sample_steps=self.sample_steps,
                        return_all_steps=False,
                        seis=b_seis,
                    )

                    # 逐样本记录 SSIM
                    vel_np_b  = b_vel.cpu().numpy()  # shape (bs, 1, H, W)
                    pred_np_b = pred_vel.cpu().numpy()

                    for j in range(bs):
                        metrics = calculate_image_metrics(
                            vel_np_b[j:j+1], pred_np_b[j:j+1]
                        )
                        s = metrics["SSIM"]
                        ssim_list.append(s)
                        record_list.append((
                            vel_np_b[j:j+1],   # GT
                            pred_np_b[j:j+1],  # Pred
                            global_idx,         # 全局样本编号
                            s                   # SSIM 值
                        ))
                        global_idx += 1

        self.all_ssim[key]       = ssim_list
        self.sample_records[key] = record_list

    # -------- 主入口 --------
    def run(self):
        """顺序推理两个模型，防止GPU显存不足"""
        for key in self.samplers.keys():
            print(f"\n{'='*55}")
            print(f"🚀 开始推理：{key}")
            print(f"{'='*55}")
            self._infer_one_model(key)

        # 打印平均 SSIM
        print("\n" + "="*55)
        print("📊 批量测试平均 SSIM 对比")
        print("="*55)
        for key, ssim_vals in self.all_ssim.items():
            if ssim_vals:
                mean_s = np.mean(ssim_vals)
                std_s  = np.std(ssim_vals)
                print(f"  [{key:12s}]  平均 SSIM = {mean_s:.4f} ± {std_s:.4f}  (共 {len(ssim_vals)} 个样本)")
        print("="*55)

        # 🔥 把所有采样 SSIM 值写入 TXT，便于后续分析
        self._save_ssim_txt()

        # 在达标样本中找 ΔSSIM 最大的并可视化
        self._find_and_visualize_best_gain()

    # -------- 保存所有 SSIM 到 TXT --------
    def _save_ssim_txt(self):
        """
        将两个权重的逐样本 SSIM 值写入 TXT 文件，
        同时附上汇总统计（均值 / 标准差 / 达标率）。
        """
        txt_path = os.path.join(self.save_dir, "ssim_results.txt")  # 输出路径
        lines = []
        lines.append(f"# SSIM 采样结果  |  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"# SSIM 达标阈值 = {SSIM_THRESHOLD}")
        lines.append("="*65)

        for key, ssim_vals in self.all_ssim.items():
            if not ssim_vals:
                continue
            arr = np.array(ssim_vals)
            above = int(np.sum(arr >= SSIM_THRESHOLD))  # 达标样本数
            lines.append(f"\n[{key}]")
            lines.append(f"  样本总数   : {len(arr)}")
            lines.append(f"  平均 SSIM  : {arr.mean():.4f} ± {arr.std():.4f}")
            lines.append(f"  最大 SSIM  : {arr.max():.4f}")
            lines.append(f"  最小 SSIM  : {arr.min():.4f}")
            lines.append(f"  >= {SSIM_THRESHOLD} 达标数: {above} / {len(arr)}  ({above/len(arr)*100:.1f}%)")
            lines.append("-"*40)
            lines.append("  idx   SSIM")
            for i, s in enumerate(ssim_vals):  # 逐样本列出
                flag = " ✓" if s >= SSIM_THRESHOLD else ""  # 标注达标样本
                lines.append(f"  {i:5d}  {s:.4f}{flag}")

        lines.append("\n" + "="*65)

        # 如果两个权重都有数据，额外写出 ΔSSIM 列表
        if self.all_ssim["contrast_0"] and self.all_ssim["contrast_0.5"]:
            b = np.array(self.all_ssim["contrast_0"])
            c = np.array(self.all_ssim["contrast_0.5"])
            n = min(len(b), len(c))
            gains = c[:n] - b[:n]
            lines.append("\n[ΔSSIM = contrast_0.5 - contrast_0  (逐样本)]")
            lines.append("-"*40)
            lines.append("  idx   SSIM_0    SSIM_0.5  ΔSSIM")
            for i in range(n):
                lines.append(
                    f"  {i:5d}  {b[i]:.4f}   {c[i]:.4f}   {gains[i]:+.4f}"
                )

        with open(txt_path, "w", encoding="utf-8") as f:  # 写文件
            f.write("\n".join(lines))
        print(f"\n📄 所有 SSIM 采样值已写入：{txt_path}")

    # -------- 找达标样本中 ΔSSIM 最大的 --------
    def _find_and_visualize_best_gain(self):
        """
        在 contrast=0.5 SSIM >= SSIM_THRESHOLD 的样本中，
        找 ΔSSIM = SSIM(0.5) - SSIM(0) 最大的那一个并可视化。
        """
        ssim_base = np.array(self.all_ssim["contrast_0"])
        ssim_cont = np.array(self.all_ssim["contrast_0.5"])

        if len(ssim_base) == 0 or len(ssim_cont) == 0:
            print("⚠️ 缺少推理结果，无法计算增益")
            return

        n = min(len(ssim_base), len(ssim_cont))
        gains = ssim_cont[:n] - ssim_base[:n]

        # 🔥 先筛选 contrast=0.5 SSIM 达到阈值的样本
        qualified = np.where(ssim_cont[:n] >= SSIM_THRESHOLD)[0]
        print(f"\n📌 contrast=0.5 中 SSIM >= {SSIM_THRESHOLD} 的样本数：{len(qualified)} / {n}")

        if len(qualified) == 0:
            print(f"⚠️ 没有样本达到 SSIM >= {SSIM_THRESHOLD} 的标准，退出可视化。")
            return

        # 在达标样本里找 ΔSSIM 最大的
        best_q_idx = qualified[int(np.argmax(gains[qualified]))]  # 全局样本索引
        best_gain      = gains[best_q_idx]
        ssim_val_base  = ssim_base[best_q_idx]
        ssim_val_cont  = ssim_cont[best_q_idx]

        print(f"\n🏆 达标样本中 ΔSSIM 最大的（全局索引 {best_q_idx}）：")
        print(f"   contrast=0   SSIM = {ssim_val_base:.4f}")
        print(f"   contrast=0.5 SSIM = {ssim_val_cont:.4f}  (已达标 >= {SSIM_THRESHOLD})")
        print(f"   增益 ΔSSIM        = {best_gain:+.4f}")

        # 取出 GT / Pred
        rec_base = self.sample_records["contrast_0"][best_q_idx]
        rec_cont = self.sample_records["contrast_0.5"][best_q_idx]
        vel_gt_np    = rec_cont[0]
        pred_base_np = rec_base[1]
        pred_cont_np = rec_cont[1]

        self._visualize_best_sample(
            vel_gt_np, pred_base_np, pred_cont_np,
            global_idx=best_q_idx,
            ssim_base=ssim_val_base,
            ssim_cont=ssim_val_cont,
            gain=best_gain,
        )

    # -------- 三列可视化 --------
    def _visualize_best_sample(self, vel_gt, pred_base, pred_cont,
                                global_idx, ssim_base, ssim_cont, gain):
        """
        可视化三列：Ground Truth | Pred(contrast=0) | Pred(contrast=0.5)
        并将图像保存到 save_dir。
        """
        def to_display(arr):
            return np.clip((arr[0, 0] + 1.0) / 2.0, 0, 1)

        gt_img   = to_display(vel_gt)
        pred0_img = to_display(pred_base)
        pred05_img = to_display(pred_cont)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"Best SSIM Gain Sample (Global Index = {global_idx})\n"
            f"ΔSSIM = {gain:+.4f}  "
            f"(contrast=0: {ssim_base:.4f}  →  contrast=0.5: {ssim_cont:.4f})",
            fontsize=14, fontweight='bold', y=1.03
        )

        titles = [
            "Ground Truth",
            f"Pred  [contrast=0]\nSSIM={ssim_base:.4f}",
            f"Pred  [contrast=0.5]\nSSIM={ssim_cont:.4f}  (ΔSSIM={gain:+.4f})",
        ]
        imgs = [gt_img, pred0_img, pred05_img]

        for ax, img, title in zip(axes, imgs, titles):
            im = ax.imshow(img, cmap='viridis', vmin=0, vmax=1)
            ax.set_title(title, fontsize=12, pad=8)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Velocity')

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"best_gain_sample_{global_idx}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f"\n✅ 可视化图像已保存：{save_path}")


# =====================================
# 🔥 主流程
# =====================================
if __name__ == "__main__":
    # 1. 加载数据
    print("📂 加载数据对...")
    data_pairs = load_data_pairs(DATA_DIR)
    print(f"   找到 {len(data_pairs)} 对文件")

    # 2. 依次构建两个模型（顺序加载，避免同时占显存）
    samplers = {}
    for name, wpath in WEIGHT_PATHS.items():
        print(f"\n{'='*55}")
        print(f"⚙️  初始化模型 [{name}]")
        print(f"{'='*55}")
        samplers[name] = build_model(wpath)

    # 3. 创建批量测试器并运行
    tester = DualWeightBatchTester(
        samplers=samplers,
        data_pairs=data_pairs,
        save_dir=SAVE_DIR,
        sample_steps=SAMPLE_STEPS,
        cfg_scale=CFG_SCALE,
        batch_size=BATCH_SIZE,
    )
    tester.run()