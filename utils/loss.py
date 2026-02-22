import torch
import torch.nn.functional as F
# 需要安装：pip install lpips
import lpips


class LPIPSHuberLoss:
    """
    LPIPS-Huber-1/t 损失函数的PyTorch实现
    论文：Improving the Training of Rectified Flows (2405.20320v2)
    """

    def __init__(self, data_dim, lpips_net='alex', device='cuda'):
        """
        初始化损失函数

        参数：
        - data_dim: 数据维度（例如CIFAR-10: 3*32*32=3072）
        - lpips_net: LPIPS网络类型，可选'alex', 'vgg', 'squeeze'
        - device: 计算设备
        """
        self.c = 0.00054 * data_dim  # 根据论文公式
        self.device = device

        # 初始化LPIPS模型
        self.lpips_model = lpips.LPIPS(net=lpips_net, verbose=False)
        self.lpips_model = self.lpips_model.to(device)
        self.lpips_model.eval()

        # 设置LPIPS不需要梯度
        for param in self.lpips_model.parameters():
            param.requires_grad = False

    def huber_loss(self, pred, target):
        """
        计算Pseudo-Huber损失

        参数：
        - pred: 预测的速度向量 (B, ...)
        - target: 真实的速度向量 (B, ...)

        返回：
        - huber损失值
        """
        diff = pred - target
        diff_norm_sq = torch.sum(diff ** 2, dim=list(range(1, len(diff.shape))))

        # Pseudo-Huber公式: sqrt(||u-v||^2 + c^2) - c
        huber = torch.sqrt(diff_norm_sq + self.c ** 2) - self.c

        return huber.mean()

    def compute_lpips(self, img1, img2):
        """
        计算两批图像之间的LPIPS距离

        参数：
        - img1, img2: 图像张量，形状为(B, C, H, W)，值在[0, 1]范围内

        返回：
        - LPIPS距离
        """
        # 确保图像在[0, 1]范围内
        img1 = torch.clamp(img1, 0, 1)
        img2 = torch.clamp(img2, 0, 1)

        # 计算LPIPS
        with torch.no_grad():
            lpips_dist = self.lpips_model(img1, img2)

        return lpips_dist.mean()

    def __call__(self, x, z, v_pred, x_t, t):
        """
        计算LPIPS-Huber-1/t损失

        参数：
        - x: 真实数据 (B, C, H, W)
        - z: 噪声数据 (B, C, H, W)
        - v_pred: 模型预测的速度 (B, C, H, W)
        - x_t: 线性插值点 (B, C, H, W)，x_t = (1-t)*x + t*z
        - t: 时间步 (B,) 或标量，范围[0, 1]

        返回：
        - 损失值
        """
        batch_size = x.shape[0]

        # 确保t是正确形状
        if isinstance(t, (int, float)):
            t = torch.full((batch_size,), t, device=self.device)
        t = t.view(batch_size, 1, 1, 1)  # 调整为(B, 1, 1, 1)

        # 计算真实速度：z - x
        v_true = z - x

        # 1. 计算Huber损失部分
        huber_part = self.huber_loss(v_pred, v_true)  # 形状: 标量

        # 2. 计算LPIPS部分
        # 计算重建的图像：x_0_hat = x_t - t * v_pred
        x_0_hat = x_t - t * v_pred

        # 将图像归一化到[0, 1]（假设输入在[-1, 1]，根据实际调整）
        # 如果是CIFAR-10/ImageNet，通常需要从[-1, 1]转换到[0, 1]
        x_normalized = (x + 1) / 2  # 如果原始数据在[-1, 1]
        x_0_hat_normalized = (x_0_hat + 1) / 2

        lpips_part = self.compute_lpips(x_normalized, x_0_hat_normalized)

        # 3. 应用时间相关的权重
        # 使用安全的t值，避免除零
        t_safe = torch.clamp(t, 1e-8, 1.0)

        # 计算最终损失
        loss = (1 - t_safe).mean() * huber_part + (1 / t_safe).mean() * lpips_part

        return loss

    def forward(self, x, z, v_pred, x_t, t):
        """前向传播别名"""
        return self.__call__(x, z, v_pred, x_t, t)


# ==================== 使用示例 ====================

def train_step(model, optimizer, loss_fn, batch_data, device='cuda'):
    """
    一个训练步骤的示例

    参数：
    - model: 整流流模型
    - optimizer: 优化器
    - loss_fn: LPIPS-Huber损失函数
    - batch_data: 批数据 (真实图像)
    - device: 计算设备
    """
    model.train()

    # 1. 准备数据
    x = batch_data.to(device)  # 真实数据
    batch_size = x.shape[0]

    # 2. 生成噪声（标准高斯分布）
    z = torch.randn_like(x).to(device)

    # 3. 随机采样时间步t（使用U形分布，如论文建议）
    # 论文建议的U形分布：p_t(u) ∝ exp(au) + exp(-au)，其中a=4
    # 这里简化为在[0, 1]内随机采样
    t = torch.rand(batch_size, device=device)

    # 4. 计算线性插值点 x_t = (1-t)*x + t*z
    t_expanded = t.view(batch_size, 1, 1, 1)
    x_t = (1 - t_expanded) * x + t_expanded * z

    # 5. 模型前向传播
    v_pred = model(x_t, t)

    # 6. 计算损失
    loss = loss_fn(x, z, v_pred, x_t, t)

    # 7. 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ==================== 训练循环示例 ====================

def train_rectified_flow(model, train_loader, num_epochs, device='cuda'):
    """
    训练整流流模型的完整示例
    """
    # 计算数据维度（例如CIFAR-10: 3*32*32=3072）
    data_dim = 3 * 32 * 32  # 根据实际数据集调整

    # 初始化损失函数
    loss_fn = LPIPSHuberLoss(data_dim=data_dim, device=device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 训练循环
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, batch_data in enumerate(train_loader):
            if isinstance(batch_data, list):
                batch_data = batch_data[0]  # 假设数据在第一个元素

            # 一个训练步骤
            loss = train_step(model, optimizer, loss_fn, batch_data, device)
            epoch_loss += loss

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}')

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')

    return model


# ==================== 辅助函数 ====================

def generate_samples(model, num_samples, img_size, device='cuda'):
    """
    使用训练好的模型生成样本

    参数：
    - model: 训练好的整流流模型
    - num_samples: 要生成的样本数量
    - img_size: 图像大小 (C, H, W)
    - device: 计算设备
    """
    model.eval()

    with torch.no_grad():
        # 从标准高斯分布采样噪声
        z = torch.randn(num_samples, *img_size, device=device)

        # 时间从1到0（生成过程）
        t = torch.ones(num_samples, device=device)

        # 使用欧拉方法求解ODE（论文中使用了更高级的求解器）
        # 简单的一步生成（如果训练良好）
        x_t = z  # 初始化为噪声
        v = model(x_t, t)
        x_0 = x_t - t.view(-1, 1, 1, 1) * v  # 一步生成

        # 如果需要多步，可以使用循环
        # steps = 10
        # dt = 1.0 / steps
        # for i in range(steps):
        #     t_current = 1.0 - i * dt
        #     v = model(x_t, t_current)
        #     x_t = x_t - dt * v

    return x_0


# ==================== 初始化LPIPS ====================
# 注意：首次运行会自动下载预训练权重

# 安装lpips库：
# pip install lpips

# 或者在代码中检查并安装：
try:
    import lpips

    print("LPIPS库已安装")
except ImportError:
    print("安装LPIPS库...")
    import subprocess

    subprocess.check_call(["pip", "install", "lpips"])
    import lpips