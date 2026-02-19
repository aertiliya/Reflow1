"""
FWI模型评估模块 - Evaluation Class

实现Requirements 6.1-6.4:
- 同时计算Train和Test样本的SSIM/MAE/RMSE
- 过拟合检测（Train SSIM > 0.8 且 Test SSIM < 0.5）
- 良好泛化检测（Train/Test SSIM差距 < 0.15）
- 保存真值对比图
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from skimage.metrics import structural_similarity as ssim
import os


@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    ssim: float
    mae: float
    rmse: float
    residual: Optional[np.ndarray] = None


@dataclass
class EvaluationResult:
    """完整评估结果"""
    train_metrics: EvaluationMetrics
    test_metrics: EvaluationMetrics
    is_overfitting: bool
    is_good_generalization: bool
    ssim_gap: float
    status: str  # "overfitting", "good_generalization", "normal"


class Evaluation:
    """
    FWI模型评估类
    
    功能:
    - 计算Train和Test样本的SSIM/MAE/RMSE指标
    - 检测过拟合（Train SSIM > 0.8 且 Test SSIM < 0.5）
    - 检测良好泛化（Train/Test SSIM差距 < 0.15）
    - 保存真值对比图
    
    Requirements: 6.1, 6.2, 6.3, 6.4
    """
    
    # 过拟合检测阈值
    OVERFIT_TRAIN_SSIM_THRESHOLD = 0.8
    OVERFIT_TEST_SSIM_THRESHOLD = 0.5
    
    # 良好泛化检测阈值
    GOOD_GENERALIZATION_GAP_THRESHOLD = 0.15
    
    def __init__(self, output_dir: str = "./evaluation_results"):
        """
        初始化评估器
        
        Args:
            output_dir: 评估结果输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    @staticmethod
    def normalize_image(img: torch.Tensor) -> np.ndarray:
        """
        将图像张量归一化到[0, 1]范围
        
        Args:
            img: 输入图像张量
            
        Returns:
            归一化后的numpy数组
        """
        if isinstance(img, torch.Tensor):
            img_np = img.detach().cpu().numpy().astype(np.float32)
        else:
            img_np = np.array(img, dtype=np.float32)
        
        # 处理多维张量
        if img_np.ndim > 2:
            img_np = img_np.squeeze()
        
        # 归一化到[0, 1]
        min_val, max_val = img_np.min(), img_np.max()
        if max_val - min_val > 1e-8:
            img_np = (img_np - min_val) / (max_val - min_val)
        else:
            img_np = np.zeros_like(img_np)
        
        return img_np
    
    def compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> EvaluationMetrics:
        """
        计算单个样本的评估指标
        
        Args:
            pred: 预测图像
            target: 真值图像
            
        Returns:
            EvaluationMetrics对象
            
        Requirements: 6.1
        """
        pred_np = self.normalize_image(pred)
        target_np = self.normalize_image(target)
        
        # 计算残差
        residual = pred_np - target_np
        
        # 计算MAE
        mae = float(np.mean(np.abs(residual)))
        
        # 计算RMSE
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        
        # 计算SSIM
        ssim_score = float(ssim(pred_np, target_np, data_range=1.0))
        
        return EvaluationMetrics(
            ssim=ssim_score,
            mae=mae,
            rmse=rmse,
            residual=residual
        )
    
    def compute_batch_metrics(
        self, 
        preds: List[torch.Tensor], 
        targets: List[torch.Tensor]
    ) -> EvaluationMetrics:
        """
        计算批量样本的平均评估指标
        
        Args:
            preds: 预测图像列表
            targets: 真值图像列表
            
        Returns:
            平均EvaluationMetrics对象
        """
        ssim_scores = []
        mae_scores = []
        rmse_scores = []
        
        for pred, target in zip(preds, targets):
            metrics = self.compute_metrics(pred, target)
            ssim_scores.append(metrics.ssim)
            mae_scores.append(metrics.mae)
            rmse_scores.append(metrics.rmse)
        
        return EvaluationMetrics(
            ssim=float(np.mean(ssim_scores)),
            mae=float(np.mean(mae_scores)),
            rmse=float(np.mean(rmse_scores))
        )
    
    def detect_overfitting(
        self, 
        train_ssim: float, 
        test_ssim: float
    ) -> bool:
        """
        检测过拟合
        
        过拟合条件: Train SSIM > 0.8 且 Test SSIM < 0.5
        
        Args:
            train_ssim: 训练集SSIM
            test_ssim: 测试集SSIM
            
        Returns:
            是否过拟合
            
        Requirements: 6.2
        """
        return (train_ssim > self.OVERFIT_TRAIN_SSIM_THRESHOLD and 
                test_ssim < self.OVERFIT_TEST_SSIM_THRESHOLD)
    
    def detect_good_generalization(
        self, 
        train_ssim: float, 
        test_ssim: float
    ) -> bool:
        """
        检测良好泛化
        
        良好泛化条件: Train/Test SSIM差距 < 0.15
        
        Args:
            train_ssim: 训练集SSIM
            test_ssim: 测试集SSIM
            
        Returns:
            是否良好泛化
            
        Requirements: 6.3
        """
        ssim_gap = abs(train_ssim - test_ssim)
        return ssim_gap < self.GOOD_GENERALIZATION_GAP_THRESHOLD
    
    def evaluate(
        self,
        train_preds: List[torch.Tensor],
        train_targets: List[torch.Tensor],
        test_preds: List[torch.Tensor],
        test_targets: List[torch.Tensor]
    ) -> EvaluationResult:
        """
        完整评估流程
        
        同时计算Train和Test样本的指标，并进行过拟合/泛化检测
        
        Args:
            train_preds: 训练集预测
            train_targets: 训练集真值
            test_preds: 测试集预测
            test_targets: 测试集真值
            
        Returns:
            EvaluationResult对象
            
        Requirements: 6.1, 6.2, 6.3
        """
        # 计算Train和Test指标
        train_metrics = self.compute_batch_metrics(train_preds, train_targets)
        test_metrics = self.compute_batch_metrics(test_preds, test_targets)
        
        # 检测过拟合和良好泛化
        is_overfitting = self.detect_overfitting(train_metrics.ssim, test_metrics.ssim)
        is_good_generalization = self.detect_good_generalization(
            train_metrics.ssim, test_metrics.ssim
        )
        
        # 计算SSIM差距
        ssim_gap = abs(train_metrics.ssim - test_metrics.ssim)
        
        # 确定状态
        if is_overfitting:
            status = "overfitting"
        elif is_good_generalization:
            status = "good_generalization"
        else:
            status = "normal"
        
        return EvaluationResult(
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            is_overfitting=is_overfitting,
            is_good_generalization=is_good_generalization,
            ssim_gap=ssim_gap,
            status=status
        )
    
    def save_comparison_figure(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        filename: str,
        title: str = "Prediction vs Ground Truth",
        metrics: Optional[EvaluationMetrics] = None
    ) -> str:
        """
        保存真值对比图
        
        Args:
            pred: 预测图像
            target: 真值图像
            filename: 保存文件名
            title: 图像标题
            metrics: 可选的评估指标（用于显示）
            
        Returns:
            保存的文件路径
            
        Requirements: 6.4
        """
        pred_np = self.normalize_image(pred)
        target_np = self.normalize_image(target)
        diff = np.abs(pred_np - target_np)
        
        # 如果没有提供metrics，计算一下
        if metrics is None:
            metrics = self.compute_metrics(pred, target)
        
        # 创建3列对比图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 真值
        im0 = axes[0].imshow(target_np, cmap='seismic', vmin=0, vmax=1)
        axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # 预测
        im1 = axes[1].imshow(pred_np, cmap='seismic', vmin=0, vmax=1)
        axes[1].set_title(f'Prediction\nSSIM={metrics.ssim:.4f}', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 差异图
        im2 = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[2].set_title(f'Difference\nMAE={metrics.mae:.4f}, RMSE={metrics.rmse:.4f}', 
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def save_evaluation_report(
        self,
        result: EvaluationResult,
        step: int,
        filename: str = "evaluation_report.txt"
    ) -> str:
        """
        保存评估报告
        
        Args:
            result: 评估结果
            step: 训练步数
            filename: 报告文件名
            
        Returns:
            保存的文件路径
        """
        save_path = os.path.join(self.output_dir, filename)
        
        with open(save_path, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Evaluation Report - Step {step}\n")
            f.write(f"{'='*60}\n\n")
            
            f.write("Train Metrics:\n")
            f.write(f"  SSIM: {result.train_metrics.ssim:.4f}\n")
            f.write(f"  MAE:  {result.train_metrics.mae:.4f}\n")
            f.write(f"  RMSE: {result.train_metrics.rmse:.4f}\n\n")
            
            f.write("Test Metrics:\n")
            f.write(f"  SSIM: {result.test_metrics.ssim:.4f}\n")
            f.write(f"  MAE:  {result.test_metrics.mae:.4f}\n")
            f.write(f"  RMSE: {result.test_metrics.rmse:.4f}\n\n")
            
            f.write(f"SSIM Gap: {result.ssim_gap:.4f}\n")
            f.write(f"Status: {result.status}\n")
            
            if result.is_overfitting:
                f.write("\n⚠️ WARNING: Overfitting detected!\n")
                f.write(f"  Train SSIM ({result.train_metrics.ssim:.4f}) > 0.8\n")
                f.write(f"  Test SSIM ({result.test_metrics.ssim:.4f}) < 0.5\n")
            
            if result.is_good_generalization:
                f.write("\n✅ Good generalization achieved!\n")
                f.write(f"  SSIM gap ({result.ssim_gap:.4f}) < 0.15\n")
        
        return save_path
    
    def print_summary(self, result: EvaluationResult, step: int):
        """
        打印评估摘要到控制台
        
        Args:
            result: 评估结果
            step: 训练步数
        """
        print(f"\n{'='*50}")
        print(f"Evaluation Summary - Step {step}")
        print(f"{'='*50}")
        print(f"Train SSIM: {result.train_metrics.ssim:.4f} | "
              f"Test SSIM: {result.test_metrics.ssim:.4f}")
        print(f"Train MAE:  {result.train_metrics.mae:.4f} | "
              f"Test MAE:  {result.test_metrics.mae:.4f}")
        print(f"Train RMSE: {result.train_metrics.rmse:.4f} | "
              f"Test RMSE: {result.test_metrics.rmse:.4f}")
        print(f"SSIM Gap: {result.ssim_gap:.4f}")
        print(f"Status: {result.status}")
        
        if result.is_overfitting:
            print("\n⚠️  OVERFITTING DETECTED!")
        if result.is_good_generalization:
            print("\n✅ GOOD GENERALIZATION!")
        print(f"{'='*50}\n")
