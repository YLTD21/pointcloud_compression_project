# scripts/gpu_utils.py
import numpy as np
import time
import os

# 尝试导入GPU相关库
try:
    import cupy as cp

    HAS_CUPY = True
    print("✓ 成功导入CuPy，GPU加速可用")
except ImportError:
    HAS_CUPY = False
    print("✗ CuPy不可用，将使用CPU计算")

try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
    print("✓ 成功导入PyTorch，GPU加速可用")
except ImportError:
    HAS_TORCH = False
    print("✗ PyTorch不可用，将使用CPU计算")


class GPUAccelerator:
    def __init__(self, device='auto'):
        self.device = self._setup_device(device)
        self.use_gpu = self.device != 'cpu'

    def _setup_device(self, device):
        if device == 'auto':
            if HAS_TORCH and torch.cuda.is_available():
                return 'cuda'
            elif HAS_CUPY:
                return 'cupy'
            else:
                return 'cpu'
        return device

    def to_gpu(self, array):
        """将numpy数组转移到GPU"""
        if not self.use_gpu or array is None:
            return array

        if self.device == 'cuda' and HAS_TORCH:
            return torch.from_numpy(array).float().to('cuda')
        elif self.device == 'cupy' and HAS_CUPY:
            return cp.asarray(array)
        else:
            return array

    def to_cpu(self, array):
        """将GPU数组转回CPU"""
        if array is None:
            return array

        if isinstance(array, torch.Tensor):
            return array.cpu().numpy()
        elif HAS_CUPY and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        else:
            return array

    def pairwise_distance_gpu(self, points, query_points=None):
        """GPU加速的成对距离计算"""
        if not self.use_gpu or len(points) < 1000:
            # 小数据量使用CPU更高效
            from scipy.spatial.distance import cdist
            if query_points is None:
                query_points = points
            return cdist(points, query_points)

        if self.device == 'cuda' and HAS_TORCH:
            points_tensor = self.to_gpu(points)
            if query_points is None:
                query_points_tensor = points_tensor
            else:
                query_points_tensor = self.to_gpu(query_points)

            # 使用PyTorch计算距离
            dists = torch.cdist(points_tensor, query_points_tensor)
            return self.to_cpu(dists)

        elif self.device == 'cupy' and HAS_CUPY:
            points_gpu = self.to_gpu(points)
            if query_points is None:
                query_points_gpu = points_gpu
            else:
                query_points_gpu = self.to_gpu(query_points)

            # 使用CuPy计算距离
            dists = cp.linalg.norm(points_gpu[:, None] - query_points_gpu[None, :], axis=2)
            return self.to_cpu(dists)

        else:
            from scipy.spatial.distance import cdist
            if query_points is None:
                query_points = points
            return cdist(points, query_points)


def calculate_curvature_gpu(points, k=30, gpu_device='auto'):
    """GPU加速的曲率计算 - 修复版本"""
    if points is None or len(points) == 0:
        return np.array([])

    if len(points) < k * 2:
        # 点太少，使用CPU
        print("  点云点数过少，使用CPU计算曲率")
        from utils import calculate_point_curvature
        return calculate_point_curvature(points, k)

    accelerator = GPUAccelerator(gpu_device)

    try:
        points_gpu = accelerator.to_gpu(points)
        curvatures = np.zeros(len(points))

        if accelerator.device == 'cuda' and HAS_TORCH:
            # PyTorch实现
            from torch import linalg

            for i in range(len(points)):
                try:
                    # 计算距离
                    distances = torch.norm(points_gpu - points_gpu[i], dim=1)
                    _, indices = torch.topk(distances, k, largest=False)

                    # 获取邻居点
                    neighbors = points_gpu[indices]

                    # 计算协方差矩阵
                    centered = neighbors - neighbors.mean(dim=0)
                    cov = centered.T @ centered / (k - 1)

                    # 计算特征值
                    eigenvalues = linalg.eigvalsh(cov)
                    eigenvalues = eigenvalues[eigenvalues > 0]

                    if len(eigenvalues) > 0:
                        curvature = eigenvalues[0] / eigenvalues.sum()
                        curvatures[i] = curvature.item()
                    else:
                        curvatures[i] = 0.0

                except Exception as e:
                    print(f"    点 {i} 曲率计算失败: {e}")
                    curvatures[i] = 0.0

        elif accelerator.device == 'cupy' and HAS_CUPY:
            # CuPy实现
            for i in range(len(points)):
                try:
                    distances = cp.linalg.norm(points_gpu - points_gpu[i], axis=1)
                    indices = cp.argsort(distances)[:k]

                    neighbors = points_gpu[indices]
                    centered = neighbors - neighbors.mean(axis=0)
                    cov = centered.T @ centered / (k - 1)

                    eigenvalues = cp.linalg.eigvalsh(cov)
                    eigenvalues = eigenvalues[eigenvalues > 0]

                    if len(eigenvalues) > 0:
                        curvature = eigenvalues[0] / eigenvalues.sum()
                        curvatures[i] = cp.asnumpy(curvature)
                    else:
                        curvatures[i] = 0.0

                except Exception as e:
                    print(f"    点 {i} 曲率计算失败: {e}")
                    curvatures[i] = 0.0

        else:
            # 回退到CPU
            print("  GPU不可用，回退到CPU计算曲率")
            from utils import calculate_point_curvature
            return calculate_point_curvature(points, k)

        return curvatures

    except Exception as e:
        print(f"  GPU曲率计算失败: {e}，回退到CPU")
        from utils import calculate_point_curvature
        return calculate_point_curvature(points, k)


def fast_voxel_downsample_gpu(points, colors=None, voxel_size=0.1, gpu_device='auto'):
    """GPU加速的体素下采样"""
    accelerator = GPUAccelerator(gpu_device)

    if len(points) < 10000 or not accelerator.use_gpu:
        # 小数据量或GPU不可用时使用Open3D
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        down_pcd = pcd.voxel_down_sample(voxel_size)
        down_points = np.asarray(down_pcd.points)
        down_colors = np.asarray(down_pcd.colors) if down_pcd.has_colors() else None
        return down_points, down_colors

    points_gpu = accelerator.to_gpu(points)

    if accelerator.device == 'cuda' and HAS_TORCH:
        # PyTorch体素下采样
        voxel_indices = (points_gpu / voxel_size).floor().long()
        unique_voxels, inverse_indices = torch.unique(voxel_indices, dim=0, return_inverse=True)

        # 计算每个体素的中心点
        down_points = torch.zeros_like(unique_voxels, dtype=torch.float32)
        if colors is not None:
            colors_gpu = accelerator.to_gpu(colors)
            down_colors = torch.zeros((len(unique_voxels), 3), dtype=torch.float32)

        for i in range(len(unique_voxels)):
            mask = (inverse_indices == i)
            down_points[i] = points_gpu[mask].mean(dim=0)
            if colors is not None:
                down_colors[i] = colors_gpu[mask].mean(dim=0)

        down_points = accelerator.to_cpu(down_points)
        down_colors = accelerator.to_cpu(down_colors) if colors is not None else None

    elif accelerator.device == 'cupy' and HAS_CUPY:
        # CuPy体素下采样
        voxel_indices = (points_gpu / voxel_size).astype(int)
        unique_voxels, inverse_indices = cp.unique(voxel_indices, axis=0, return_inverse=True)

        down_points = cp.zeros_like(unique_voxels, dtype=cp.float32)
        if colors is not None:
            colors_gpu = accelerator.to_gpu(colors)
            down_colors = cp.zeros((len(unique_voxels), 3), dtype=cp.float32)

        for i in range(len(unique_voxels)):
            mask = (inverse_indices == i)
            down_points[i] = points_gpu[mask].mean(axis=0)
            if colors is not None:
                down_colors[i] = colors_gpu[mask].mean(axis=0)

        down_points = accelerator.to_cpu(down_points)
        down_colors = accelerator.to_cpu(down_colors) if colors is not None else None

    else:
        # 回退到Open3D
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        down_pcd = pcd.voxel_down_sample(voxel_size)
        down_points = np.asarray(down_pcd.points)
        down_colors = np.asarray(down_pcd.colors) if down_pcd.has_colors() else None

    return down_points, down_colors