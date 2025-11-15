# scripts/pointnet_dataloader.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class SemanticKITTIDataset(Dataset):
    def __init__(self, sequences, data_dir, num_points=4096, transform=None):
        """
        SemanticKITTI数据集加载器

        Args:
            sequences: 序列列表，如 ['00', '01', '02', '03']
            data_dir: 数据目录
            num_points: 每个样本的点数
            transform: 数据增强
        """
        self.data_dir = Path(data_dir)
        self.num_points = num_points
        self.transform = transform

        # 语义标签映射 (SemanticKITTI官方映射)
        self.learning_map = {
            0: 0,  # 未标注
            1: 0,  # outlier
            10: 1,  # car
            11: 1,  # bicycle
            13: 1,  # bus
            15: 1,  # motorcycle
            16: 1,  # on-rails
            18: 1,  # truck
            20: 1,  # other-vehicle
            30: 1,  # person
            31: 2,  # bicyclist
            32: 2,  # motorcyclist
            252: 1,  # moving-car
            253: 2,  # moving-person
            254: 2,  # moving-bicyclist
            255: 2,  # moving-motorcyclist
            256: 1,  # moving-on-rails
            257: 1,  # moving-bus
            258: 1,  # moving-truck
            259: 1  # moving-other-vehicle
        }

        # 只保留车辆(1)和行人(2)，其他为背景(0)
        self.valid_labels = [1, 2]

        self.point_files = []
        self.label_files = []

        # 收集所有文件
        for seq in sequences:
            velodyne_path = self.data_dir / "dataset" / "sequences" / seq / "velodyne"
            labels_path = self.data_dir / "dataset" / "sequences" / seq / "labels"

            if velodyne_path.exists() and labels_path.exists():
                bin_files = sorted(velodyne_path.glob("*.bin"))
                label_files = sorted(labels_path.glob("*.label"))

                for bin_file, label_file in zip(bin_files, label_files):
                    self.point_files.append(bin_file)
                    self.label_files.append(label_file)

        print(f"加载 {len(self.point_files)} 个样本")

    def __len__(self):
        return len(self.point_files)

    def __getitem__(self, idx):
        # 加载点云
        points = np.fromfile(self.point_files[idx], dtype=np.float32).reshape(-1, 4)
        points = points[:, :3]  # 只取xyz

        # 加载标签
        labels = np.fromfile(self.label_files[idx], dtype=np.uint32)
        labels = labels & 0xFFFF  # 取低16位语义标签

        # 映射标签
        mapped_labels = np.zeros_like(labels, dtype=np.long)
        for original_label, mapped_label in self.learning_map.items():
            mask = labels == original_label
            mapped_labels[mask] = mapped_label

        # 下采样到固定点数
        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
            labels = mapped_labels[indices]
        else:
            # 如果点数不足，重复采样
            indices = np.random.choice(len(points), self.num_points, replace=True)
            points = points[indices]
            labels = mapped_labels[indices]

        # 数据归一化
        points = self.normalize_points(points)

        # 转换为tensor
        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels).long()

        return points, labels

    def normalize_points(self, points):
        """点云归一化"""
        # 移动到原点
        centroid = np.mean(points, axis=0)
        points = points - centroid

        # 缩放以适合单位球
        max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if max_dist > 0:
            points = points / max_dist

        return points


def get_data_loaders(batch_size=16, num_points=4096):
    """获取训练、验证、测试数据加载器"""

    data_dir = project_root / "data" / "raw_dataset"

    # 数据集划分
    train_sequences = ['00', '01', '02', '03']
    val_sequences = ['04']
    test_sequences = ['05']

    train_dataset = SemanticKITTIDataset(train_sequences, data_dir, num_points)
    val_dataset = SemanticKITTIDataset(val_sequences, data_dir, num_points)
    test_dataset = SemanticKITTIDataset(test_sequences, data_dir, num_points)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader