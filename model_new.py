import torch
from torch import nn
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform
import math

# from dit import DiT
from unet import UnetConcat as Unet


class LogitNormalCosineScheduler:
    """
    Combined Logit-Normal timestep sampling + Cosine interpolation scheduling.
    This is the optimal approach used in Stable Diffusion 3.
    """

    def __init__(self, loc: float = 0.0, scale: float = 1.0, min_t: float = 1e-4, max_t: float = 1.0 - 1e-4):
        """
        Args:
            loc: Location parameter (mu) for logit-normal - 0.0 for symmetric around t=0.5
            scale: Scale parameter (sigma) for logit-normal - 1.0 is SD3 default
            min_t: Minimum timestep to avoid singularities
            max_t: Maximum timestep to avoid singularities
        """
        self.loc = loc
        self.scale = scale
        self.min_t = min_t
        self.max_t = max_t
        # Create LogitNormal using TransformedDistribution
        base_normal = Normal(loc, scale)
        self.logit_normal = TransformedDistribution(base_normal, SigmoidTransform())

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps from logit-normal distribution."""
        # Step 1: Sample from logit-normal distribution
        t = self.logit_normal.sample((batch_size,)).to(device)

        # Step 2: Clamp to avoid singularities at 0 and 1
        t = torch.clamp(t, self.min_t, self.max_t)

        return t

    def get_cosine_schedule_params(self, t: torch.Tensor, sigma_min: float = 1e-6) -> tuple:
        """
        Convert logit-normal sampled timesteps to cosine-scheduled interpolation parameters.

        Args:
            t: Logit-normal sampled timesteps [batch_size]
            sigma_min: Minimum noise level

        Returns:
            alpha_t, sigma_t: Cosine-scheduled interpolation parameters
        """
        # Apply cosine scheduling transformation
        t_cos = 0.5 * (1 - torch.cos(math.pi * t))

        # Cosine interpolation parameters
        alpha_t = t_cos
        # sigma_t = 1 - t_cos + sigma_min * t_cos
        sigma_t = 1 - t_cos * (1 - sigma_min)

        return alpha_t, sigma_t

    def get_velocity_target(self, x1: torch.Tensor, z: torch.Tensor, sigma_min: float = 1e-6) -> torch.Tensor:
        """
        Compute velocity target for cosine-scheduled flow matching.

        Args:
            x1: Clean data
            z: Noise
            t: Logit-normal sampled timesteps
            sigma_min: Minimum noise level

        Returns:
            u: Velocity target
        """
        # For cosine scheduling, the velocity target is:
        u = x1 - (1 - sigma_min) * z
        return u

    def create_cosine_schedule(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """Create cosine-scheduled timestep sequence for inference."""
        t_span = torch.linspace(0, 1, num_steps + 1, device=device)
        # Apply cosine transformation for smoother scheduling
        t_span = 0.5 * (1 - torch.cos(math.pi * t_span))
        return t_span


def normalize_to_neg1_1(x):
    return x * 2 - 1


def unnormalize_to_0_1(x):
    return (x + 1) * 0.5


class RectifiedFlow(nn.Module):
    def __init__(
            self,
            net: Unet,
            device="cuda",
            channels=3,
            image_size=32,
            num_classes=10,
            logit_normal_sampling_t=True,  # Kept for backward compatibility
            use_logit_normal_cosine=True,
            # Scheduler parameters
            logit_normal_loc=0.0,
            logit_normal_scale=1.0,
            timestep_min=1e-4,
            timestep_max=1.0 - 1e-4,
    ):
        super().__init__()
        self.net = net
        self.device = device
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_cond = num_classes is not None
        self.logit_normal_sampling_t = logit_normal_sampling_t
        self.use_logit_normal_cosine = use_logit_normal_cosine

        # FIXED: Initialize scheduler properly
        if self.use_logit_normal_cosine:
            self.scheduler = LogitNormalCosineScheduler(
                loc=logit_normal_loc,
                scale=logit_normal_scale,
                min_t=timestep_min,
                max_t=timestep_max
            )
        else:
            self.scheduler = None

    def forward(self, x, c=None):
        """I used forward directly instead of via sampler"""
        pass

    def get_timestep_schedule(self, sample_steps: int):
        """Get timestep schedule based on configuration."""
        if self.use_logit_normal_cosine and self.scheduler is not None:
            return self.scheduler.create_cosine_schedule(sample_steps, self.device)
        else:
            # Fallback to linear schedule
            return torch.linspace(0, 1, sample_steps + 1, device=self.device)

    @torch.no_grad()
    def sample(self, batch_size=None, class_labels=None, cfg_scale=5.0, sample_steps=10, return_all_steps=False, seis=None):
        """
        Sample images using configured scheduling.

        Args:
            batch_size: Number of samples to generate (required if class_labels is None)
            class_labels: Tensor of class labels to condition on (optional)
            cfg_scale: Classifier-free guidance scale
            sample_steps: Number of sampling steps
            return_all_steps: Whether to return all intermediate steps
            seis: Seismic data (B, 5, 1000, 70) for conditioning (Task 8.2)

        Returns:
            Generated samples in [0, 1] range
        """
        # Determine batch size and conditioning
        if class_labels is not None:
            # Use provided class labels
            batch_size = class_labels.shape[0]
            c = class_labels.to(self.device)
        elif self.use_cond and batch_size is not None:
            # Generate random class labels
            c = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        elif batch_size is not None:
            # No conditioning
            c = None
        else:
            raise ValueError("Either batch_size or class_labels must be provided")

        print('RectifiedFlow class labels shape: ', c.shape)

        z = torch.randn((batch_size, self.channels, self.image_size, self.image_size), device=self.device)

        images = []
        if return_all_steps:
            images.append(z.clone())

        t_span = self.get_timestep_schedule(sample_steps)

        t = t_span[0]
        dt = t_span[1] - t_span[0]

        for step in range(1, len(t_span)):
            if self.use_cond and c is not None:
                print(f"sample using cfg_scale: {cfg_scale}")
                # 🚀 Task 8.2: 传递 seis 参数支持 UnetConcat
                v_t = self.net.forward_with_cfg(z, t, c, cfg_scale, seis=seis)
            else:
                v_t = self.net(z, t, seis=seis)

            z = z + dt * v_t
            t = t + dt

            # Store intermediate result
            if return_all_steps:
                images.append(z.clone())

            # Update dt for next step
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        # z_final = unnormalize_to_0_1(z.clip(-1, 1))
        z_final = z.clip(-1, 1)
        # z_final = z

        if return_all_steps:
            # Return both final image and full trajectory
            return z_final, torch.stack(images)
        return z_final

    @torch.no_grad()
    def sample_each_class(self, n_per_class, cfg_scale=5.0, sample_steps=10, return_all_steps=False, seis=None):
        """Sample n_per_class images for each class."""
        if not self.use_cond:
            raise ValueError("Cannot sample each class when num_classes is None")

        c = torch.arange(self.num_classes, device=self.device).repeat(n_per_class)
        print('RectifiedFlow sample_each_class c shape: ', c.shape)
        z = torch.randn(self.num_classes * n_per_class, self.channels, self.image_size, self.image_size,
                        device=self.device)
        z1 = z.clone()

        # FIXED: Consistent trajectory tracking
        images = [z.clone()] if return_all_steps else []
        t_span = self.get_timestep_schedule(sample_steps)

        t = t_span[0]
        print('RectifiedFlow sample_each_class t shape:', t.shape)
        t_next = t_span[1]
        dt = t_span[1] - t_span[0]

        for step in range(1, len(t_span)):
            if self.use_cond:
                print(f"Using cfg_scale: {cfg_scale}")
                v_t = self.net.forward_with_cfg(z, t, c, cfg_scale, is_train_student=False, seis=seis)
            else:
                v_t = self.net(z, t)

            # 预测数据点
            x_hat = z - v_t * t

            # z = z + dt * v_t
            z = (1 - t_next) * x_hat + t_next * z1
            t = t + dt

            # Store intermediate result
            if return_all_steps:
                images.append(z.clone())

            # Update dt for next step
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            t_next = t_next + dt

        # z_final = unnormalize_to_0_1(z.clip(-1, 1))
        z_final = z.clip(-1, 1)
        # z_final = z

        if return_all_steps:
            # Return both final image and full trajectory
            return z_final, torch.stack(images)  # Keep trajectory in [-1, 1] for GIF creation
        return z_final


class DDPM(nn.Module):
    def __init__(self, net: Unet,
            device="cuda",
            channels=3,
            image_size=70,
            num_classes=10
            ):
        super().__init__()
        self.net = net
        self.device = device
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_cond = num_classes is not None

    @torch.no_grad()
    def sample(self, batch_size=None, class_labels=None, cfg_scale=5.0, timesteps=1000, seis=None):
        # Determine batch size and conditioning
        if class_labels is not None:
            # Use provided class labels
            batch_size = class_labels.shape[0]
            c = class_labels.to(self.device)
        elif self.use_cond and batch_size is not None:
            # Generate random class labels
            c = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        elif batch_size is not None:
            # No conditioning
            c = None
        else:
            raise ValueError("Either batch_size or class_labels must be provided")

        x = torch.randn((batch_size, self.channels, self.image_size, self.image_size), device=self.device)

        # 1. 预计算beta调度（线性调度）
        scale = 1000 / timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)

        # 2. 计算alpha相关参数
        alphas = 1 - betas
        alpha_cumulative = torch.cumprod(alphas, dim=0)  # ᾱ_t

        for t in (range(1, timesteps)):
            # 创建时间步张量
            ts = torch.full((1,), t, dtype=torch.long, device=self.device)

            # 噪声采样（最后一步不加噪声）
            z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)

            # 预测噪声
            predicted_noise = self.net.forward_with_cfg(x, ts, c, cfg_scale, is_train_student=False, seis=seis)

            # 获取当前时间步的参数
            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_cumulative_t = alpha_cumulative[t]

            # 计算当前时间步的系数
            one_by_sqrt_alpha_t = 1 / torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_cumulative_t = torch.sqrt(1 - alpha_cumulative_t)

            # 调整维度以便广播
            for _ in range(len(x.shape) - len(beta_t.shape)):
                beta_t = beta_t.unsqueeze(-1)
                one_by_sqrt_alpha_t = one_by_sqrt_alpha_t.unsqueeze(-1)
                sqrt_one_minus_alpha_cumulative_t = sqrt_one_minus_alpha_cumulative_t.unsqueeze(-1)

            # DDPM采样公式
            x = (
                    one_by_sqrt_alpha_t
                    * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
                    + torch.sqrt(beta_t) * z
            )

            # 5. 返回生成的图像
        return x



    @torch.no_grad()
    def sample_each_class(self, n_per_class, cfg_scale=5.0, timesteps=1000, seis=None):
        """Sample n_per_class images for each class."""
        if not self.use_cond:
            raise ValueError("Cannot sample each class when num_classes is None")

        c = torch.arange(self.num_classes, device=self.device).repeat(n_per_class)
        print('RectifiedFlow sample_each_class c shape: ', c.shape)
        x = torch.randn(self.num_classes * n_per_class, self.channels, self.image_size, self.image_size,
                        device=self.device)



        # 1. 预计算beta调度（线性调度）
        scale = 1000 / timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)

        # 2. 计算alpha相关参数
        alphas = 1 - betas
        alpha_cumulative = torch.cumprod(alphas, dim=0)  # ᾱ_t

        for t in reversed(range(1, timesteps)):
            # 创建时间步张量
            ts = torch.full((1,), t, dtype=torch.long, device=self.device)

            # 噪声采样（最后一步不加噪声）
            z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)

            # 预测噪声
            predicted_noise = self.net.forward_with_cfg(x, ts, c, cfg_scale, is_train_student=False, seis=seis)

            # 获取当前时间步的参数
            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_cumulative_t = alpha_cumulative[t]

            # 计算当前时间步的系数
            one_by_sqrt_alpha_t = 1 / torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_cumulative_t = torch.sqrt(1 - alpha_cumulative_t)

            # 调整维度以便广播
            for _ in range(len(x.shape) - len(beta_t.shape)):
                beta_t = beta_t.unsqueeze(-1)
                one_by_sqrt_alpha_t = one_by_sqrt_alpha_t.unsqueeze(-1)
                sqrt_one_minus_alpha_cumulative_t = sqrt_one_minus_alpha_cumulative_t.unsqueeze(-1)

            # DDPM采样公式
            x = (
                    one_by_sqrt_alpha_t
                    * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
                    + torch.sqrt(beta_t) * z
            )

            # 5. 返回生成的图像
        return x

