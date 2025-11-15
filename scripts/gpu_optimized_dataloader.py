# scripts/gpu_optimized_dataloader.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class GPUDirectDataset(Dataset):
    """直接将数据预加载到GPU的数据集"""

    def __init__(self, sequences, data_dir, num_points=4096, device='cuda'):
        self.device = device
        self.num_points = num_points

        # 语义标签映射
        self.learning_map = {
            0: 0, 1: 0, 10: 1, 11: 1, 13: 1, 15: 1, 16: 1, 18: 1, 20: 1,
            30: 1, 31: 2, 32: 2, 252: 1, 253: 2, 254: 2, 255: 2, 256: 1,
            257: 1, 258: 1, 259: 1
        }

        self.points_tensors = []
        self.labels_tensors = []

        self._preload_to_gpu(sequences, data_dir)

    def _preload_to_gpu(self, sequences, data_dir):
        """直接将数据预加载到GPU"""
        print("🚀 直接将数据加载到GPU内存...")

        total_files = 0
        for seq in sequences:
            velodyne_path = data_dir / "dataset" / "sequences" / seq / "velodyne"
            labels_path = data_dir / "dataset" / "sequences" / seq / "labels"

            if not velodyne_path.exists() or not labels_path.exists():
                continue

            bin_files = sorted(velodyne_path.glob("*.bin"))
            label_files = sorted(labels_path.glob("*.label"))
            total_files += len(bin_files)

        print(f"📁 找到 {total_files} 个样本，开始GPU预加载...")

        loaded_count = 0
        for seq in sequences:
            velodyne_path = data_dir / "dataset" / "sequences" / seq / "velodyne"
            labels_path = data_dir / "dataset" / "sequences" / seq / "labels"

            if not velodyne_path.exists():
                continue

            bin_files = sorted(velodyne_path.glob("*.bin"))
            label_files = sorted(labels_path.glob("*.label"))

            for bin_file, label_file in tqdm(zip(bin_files, label_files),
                                             desc=f"加载序列 {seq}",
                                             total=len(bin_files)):
                try:
                    # 加载数据
                    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)[:, :3]
                    labels = np.fromfile(label_file, dtype=np.uint32) & 0xFFFF

                    # CPU端最小化预处理
                    points, labels = self._preprocess_on_cpu(points, labels)

                    # 立即送到GPU
                    points_tensor = torch.from_numpy(points).float().to(self.device)
                    labels_tensor = torch.from_numpy(labels).long().to(self.device)

                    self.points_tensors.append(points_tensor)
                    self.labels_tensors.append(labels_tensor)
                    loaded_count += 1

                except Exception as e:
                    print(f"❌ 加载失败 {bin_file}: {e}")
                    continue

        print(f"✅ GPU预加载完成: {loaded_count}/{total_files} 个样本")
        print(f"💾 GPU内存占用: {self._get_gpu_memory_usage()}")

    def _preprocess_on_cpu(self, points, labels):
        """最小化的CPU预处理"""
        n_points = len(points)

        # 下采样
        if n_points >= self.num_points:
            indices = np.random.choice(n_points, self.num_points, replace=False)
        else:
            indices = np.random.choice(n_points, self.num_points, replace=True)

        points = points[indices]
        labels = labels[indices]

        # 标签映射
        mapped_labels = np.zeros_like(labels, dtype=np.long)
        for original_label, mapped_label in self.learning_map.items():
            mask = labels == original_label
            mapped_labels[mask] = mapped_label
        labels = mapped_labels

        # 归一化
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if max_dist > 0:
            points = points / max_dist

        return points, labels

    def _get_gpu_memory_usage(self):
        """获取GPU内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            return f"{allocated:.1f}GB"
        return "N/A"

    def __getitem__(self, idx):
        # 直接从GPU内存返回，零拷贝！
        return self.points_tensors[idx], self.labels_tensors[idx]

    def __len__(self):
        return len(self.points_tensors)


def get_gpu_optimized_loaders(batch_size=64, num_points=4096):
    """GPU优化的数据加载器"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = project_root / "data" / "raw_dataset"

    print(f"🚀 创建GPU优化数据加载器")
    print(f"   - 设备: {device}")
    print(f"   - Batch Size: {batch_size}")
    print(f"   - 数据位置: GPU内存")

    # 数据集划分
    train_sequences = ['00', '01', '02', '03']
    val_sequences = ['04']
    test_sequences = ['05']

    # 训练集使用GPU预加载
    print("📥 加载训练集到GPU...")
    train_dataset = GPUDirectDataset(
        train_sequences, data_dir, num_points, device=device
    )

    # 验证集和测试集可以保持原样或也用GPU加载
    print("📥 加载验证集到GPU...")
    val_dataset = GPUDirectDataset(
        val_sequences, data_dir, num_points, device=device
    )

    print("📥 加载测试集到GPU...")
    test_dataset = GPUDirectDataset(
        test_sequences, data_dir, num_points, device=device
    )

    # 极简数据加载器配置
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 不需要worker
        pin_memory=False,  # 不需要pin_memory
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print("✅ GPU优化数据加载器创建完成")
    return train_loader, val_loader, test_loader


# 测试函数
def test_gpu_loading_speed():
    """测试GPU数据加载速度"""
    print("🧪 测试GPU数据加载速度...")

    import time
    train_loader, _, _ = get_gpu_optimized_loaders(batch_size=64, num_points=4096)

    start_time = time.time()
    batch_count = 0

    for i, (points, labels) in enumerate(train_loader):
        batch_count += 1
        if i == 10:  # 测试10个batch
            break

    total_time = time.time() - start_time
    avg_batch_time = total_time / batch_count

    print(f"📊 GPU加载性能:")
    print(f"   - 平均每个batch: {avg_batch_time:.4f}s")
    print(f"   - 预计训练速度: {64 / avg_batch_time:.1f} 样本/秒")
    print(f"   - CPU-GPU传输: 零拷贝")


if __name__ == "__main__":
    test_gpu_loading_speed()