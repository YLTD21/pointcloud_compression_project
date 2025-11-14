import numpy as np
import open3d as o3d
import os
import struct
from pathlib import Path


def get_project_root():
    """获取项目根目录"""
    return Path(__file__).parent.parent


def load_kitti_pointcloud(bin_file_path):
    """
    加载SemanticKITTI的bin文件
    返回: points (N,3), labels (N,)
    """
    points = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3], points[:, 3].astype(np.uint32)


def save_pointcloud(points, filename, labels=None):
    """保存点云到文件"""
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if filename.endswith('.pcd'):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if labels is not None:
            colors = np.zeros((len(points), 3))
            # 根据标签设置颜色
            colors[labels == 253] = [1, 0, 0]  # 车辆-红色
            colors[labels == 252] = [0, 1, 0]  # 行人-绿色
            pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(filename, pcd)
    elif filename.endswith('.npy'):
        np.save(filename, points)


def extract_pedestrian_vehicle_points(points, labels):
    """
    提取行人和车辆点
    252: pedestrian, 253: car
    """
    mask = (labels == 252) | (labels == 253)
    return points[mask], labels[mask]


def calculate_point_curvature(points, k=30):
    """计算点云曲率特征"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 计算法线和曲率
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    pcd.orient_normals_consistent_tangent_plane(k)

    # 使用协方差矩阵特征值计算曲率
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    curvatures = []

    for i in range(len(points)):
        [k, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], k)
        neighbors = points[idx, :]

        # 计算协方差矩阵
        cov = np.cov(neighbors.T)
        eigenvalues = np.linalg.eigvals(cov)
        eigenvalues.sort()

        # 曲率 = 最小特征值 / 特征值和
        if np.sum(eigenvalues) > 0:
            curvature = eigenvalues[0] / np.sum(eigenvalues)
        else:
            curvature = 0
        curvatures.append(curvature)

    return np.array(curvatures)


def find_semantic_kitti_sequences():
    """自动查找SemanticKITTI数据序列"""
    project_root = get_project_root()
    raw_data_dir = project_root / "data" / "raw_dataset"

    sequences = []

    # 检查可能的目录结构
    possible_structures = [
        raw_data_dir / "dataset" / "sequences",  # 您的结构
        raw_data_dir,  # 直接包含序列目录
        raw_data_dir / "sequences"  # 另一种可能的结构
    ]

    for base_dir in possible_structures:
        if base_dir.exists():
            print(f"找到数据目录: {base_dir}")
            # 查找序列目录
            for item in base_dir.iterdir():
                if item.is_dir() and item.name.isdigit():
                    velodyne_dir = item / "velodyne"
                    labels_dir = item / "labels"
                    if velodyne_dir.exists():
                        sequences.append({
                            'seq_id': item.name,
                            'base_path': base_dir,
                            'velodyne_path': velodyne_dir,
                            'labels_path': labels_dir if labels_dir.exists() else None
                        })
                        print(f"  找到序列 {item.name}")

    return sequences