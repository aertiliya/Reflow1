"""
TestDataSlicer - 测试集数据切片工具

从test_data/seis{6,7,8}_1_35.npy和vel{6,7,8}_1_35.npy加载测试数据
支持按索引切片和直接加载两种模式

Requirements: 6.1, 6.2
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import random

@dataclass
class TestSetConfig:
    """测试集配置"""
    num_samples: int = 2  # 简化为2个样本，加快评估速度
    # 只从seis6选2个样本
    sample_indices: Dict[str, List[int]] = field(default_factory=lambda: {
        'seis6': [100, 200] # 只用2个样本
    })
    eval_interval: int = 1000  # 每1k步评估


class TestDataSlicer:
    """
    从大npy文件中切片测试样本

    数据源: test_data/seis{6,7,8}_1_35.npy 和 vel{6,7,8}_1_35.npy

    支持两种模式:
    1. slice_samples: 预切片并保存到磁盘
    2. load_test_batch: 直接从大npy文件加载（无需预切片）

    Requirements: 6.1, 6.2
    """

    def __init__(self, test_data_dir: str = "test_data"):
        """
        初始化TestDataSlicer

        Args:
            test_data_dir: 测试数据目录路径
        """
        self.test_data_dir = test_data_dir

        # 地震数据文件映射
        self.seis_files = {
            'seis6': 'seis6_1_35.npy',
            'seis7': 'seis7_1_35.npy',
            'seis8': 'seis8_1_35.npy',
        }

        # 速度模型文件映射
        self.vel_files = {
            'seis6': 'vel6_1_35.npy',  # 🔥 修复：对应seis6的速度模型
            'seis7': 'vel7_1_35.npy',  # 对应seis7的速度模型
            'seis8': 'vel8_1_35.npy',  # 对应seis8的速度模型
        }

        # 物理归一化范围（与训练时一致）
        self.xmin_global = 1500.0
        self.xmax_global = 5500.0

        # 缓存已加载的数据
        self._seis_cache: Dict[str, np.ndarray] = {}
        self._vel_cache: Dict[str, np.ndarray] = {}

    def _get_seis_data(self, class_name: str) -> np.ndarray:
        """
        获取地震数据（带缓存）

        Args:
            class_name: 类别名称 ('seis6', 'seis7', 'seis8')

        Returns:
            地震数据数组 (500, 5, 1000, 70)
        """
        if class_name not in self._seis_cache:
            file_path = os.path.join(self.test_data_dir, self.seis_files[class_name])
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"地震数据文件不存在: {file_path}")
            self._seis_cache[class_name] = np.load(file_path)
        return self._seis_cache[class_name]

    def _get_vel_data(self, class_name: str) -> np.ndarray:
        """
        获取速度模型数据（带缓存）

        Args:
            class_name: 类别名称 ('seis6', 'seis7', 'seis8')

        Returns:
            速度模型数组 (500, 1, 70, 70)
        """
        if class_name not in self._vel_cache:
            file_path = os.path.join(self.test_data_dir, self.vel_files[class_name])
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"速度模型文件不存在: {file_path}")
            self._vel_cache[class_name] = np.load(file_path)
        return self._vel_cache[class_name]

    def slice_samples(
        self,
        config: Optional[TestSetConfig] = None,
        output_dir: str = "test_data/sliced"
    ) -> List[Tuple[str, str]]:
        """
        从大npy文件中切片指定索引的样本并保存到磁盘

        Args:
            config: 测试集配置，默认使用TestSetConfig()
            output_dir: 切片输出目录

        Returns:
            List of (seis_path, vel_path) tuples
        """
        if config is None:
            config = TestSetConfig()

        os.makedirs(output_dir, exist_ok=True)
        sliced_paths = []

        for class_name, indices in config.sample_indices.items():
            seis_data = self._get_seis_data(class_name)
            vel_data = self._get_vel_data(class_name)

            for idx in indices:
                # 验证索引有效性
                if idx >= seis_data.shape[0]:
                    print(f"⚠️ 索引 {idx} 超出 {class_name} 数据范围 (max: {seis_data.shape[0]-1})，跳过")
                    continue

                # 切片单个样本
                seis_sample = seis_data[idx]  # (5, 1000, 70)
                vel_sample = vel_data[idx]    # (1, 70, 70)

                # 保存切片
                seis_path = os.path.join(output_dir, f"{class_name}_{idx}_seis.npy")
                vel_path = os.path.join(output_dir, f"{class_name}_{idx}_vel.npy")
                np.save(seis_path, seis_sample)
                np.save(vel_path, vel_sample)

                sliced_paths.append((seis_path, vel_path))
                print(f"✅ 切片保存: {class_name}[{idx}] -> {seis_path}, {vel_path}")

        print(f"\n📦 共切片 {len(sliced_paths)} 个测试样本到 {output_dir}")
        return sliced_paths

    def load_test_batch(
        self,
        config: Optional[TestSetConfig] = None,
        device: str = "cuda",
        seis_global_max: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        直接从大npy文件加载测试batch（无需预切片）

        Args:
            config: 测试集配置，默认使用TestSetConfig()
            device: 目标设备 ('cuda' 或 'cpu')
            seis_global_max: 地震数据全局缩放因子（用于归一化）

        Returns:
            seis: (N, 5, 1000, 70) 归一化后的地震数据
            vel: (N, 1, 64, 64) 归一化后的速度模型（已插值到64x64）
            sample_ids: 样本标识列表 ['seis6_0', 'seis6_100', ...]
        """
        if config is None:
            config = TestSetConfig()

        seis_list = []
        vel_list = []
        sample_ids = []

        for class_name, indices in config.sample_indices.items():
            seis_data = self._get_seis_data(class_name)
            vel_data = self._get_vel_data(class_name)

            for idx in indices:
                # 验证索引有效性
                if idx >= seis_data.shape[0]:
                    print(f"⚠️ 索引 {idx} 超出 {class_name} 数据范围，跳过")
                    continue

                seis_list.append(seis_data[idx])  # (5, 1000, 70)
                vel_list.append(vel_data[idx])    # (1, 70, 70)
                sample_ids.append(f"{class_name}_{idx}")

        # 转换为Tensor
        seis = torch.tensor(np.stack(seis_list), dtype=torch.float32)  # (N, 5, 1000, 70)
        vel = torch.tensor(np.stack(vel_list), dtype=torch.float32)    # (N, 1, 70, 70)

        # 地震数据归一化（与训练时一致） # 53.8081
        if seis_global_max is not None:
            seis = seis / seis_global_max
            seis = torch.clamp(seis, -1, 1)

        # [1500.0, 5500.0]
        vel = (vel - self.xmin_global) / (self.xmax_global - self.xmin_global)  # [0, 1]
        vel = vel * 2.0 - 1.0  # [-1, 1]

        # 移动到目标设备
        seis = seis.to(device)
        vel = vel.to(device)

        print(f"✅ 加载 {len(sample_ids)} 个测试样本")
        print(f"   seis shape: {seis.shape}, range: [{seis.min():.4f}, {seis.max():.4f}]")
        print(f"   vel shape: {vel.shape}, range: [{vel.min():.4f}, {vel.max():.4f}]")

        return seis, vel, sample_ids

    def load_single_sample(
        self,
        class_name: str,
        idx: int,
        device: str = "cuda",
        seis_global_max: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        加载单个测试样本

        Args:
            class_name: 类别名称 ('seis6', 'seis7', 'seis8')
            idx: 样本索引
            device: 目标设备
            seis_global_max: 地震数据全局缩放因子

        Returns:
            seis: (1, 5, 1000, 70) 地震数据
            vel: (1, 1, 64, 64) 速度模型
        """
        seis_data = self._get_seis_data(class_name)
        vel_data = self._get_vel_data(class_name)

        if idx >= seis_data.shape[0]:
            raise IndexError(f"索引 {idx} 超出 {class_name} 数据范围 (max: {seis_data.shape[0]-1})")

        seis = torch.tensor(seis_data[idx:idx+1], dtype=torch.float32)  # (1, 5, 1000, 70)
        vel = torch.tensor(vel_data[idx:idx+1], dtype=torch.float32)    # (1, 1, 70, 70)

        # 地震数据归一化
        if seis_global_max is not None:
            seis = seis / seis_global_max
            seis = torch.clamp(seis, -1, 1)

        # 速度模型处理
        vel = (vel - self.xmin_global) / (self.xmax_global - self.xmin_global)
        vel = vel * 2.0 - 1.0
        
        return seis.to(device), vel.to(device)
    
    def get_available_samples(self) -> Dict[str, int]:
        """
        获取每个类别可用的样本数量
        
        Returns:
            Dict[class_name, sample_count]
        """
        available = {}
        for class_name in self.seis_files.keys():
            try:
                seis_data = self._get_seis_data(class_name)
                available[class_name] = seis_data.shape[0]
            except FileNotFoundError:
                available[class_name] = 0
        return available
    
    def clear_cache(self):
        """清除数据缓存"""
        self._seis_cache.clear()
        self._vel_cache.clear()


# 便捷函数：创建默认配置的TestDataSlicer
def create_test_data_slicer(test_data_dir: str = "test_data") -> TestDataSlicer:
    """
    创建TestDataSlicer实例
    
    Args:
        test_data_dir: 测试数据目录
    
    Returns:
        TestDataSlicer实例
    """
    return TestDataSlicer(test_data_dir=test_data_dir)
