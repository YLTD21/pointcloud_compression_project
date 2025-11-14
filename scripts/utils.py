# scripts/utils.py
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


def load_extracted_pointcloud(pcd_file_path):
    """
    加载提取的点云文件（增强版）
    返回: points (N,3), colors (N,3), labels (N,)
    """
    pcd = o3d.io.read_point_cloud(str(pcd_file_path))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # 从颜色推断标签（基于我们的颜色映射）
    labels = np.zeros(len(points), dtype=np.uint8)  # 0:背景
    if len(colors) > 0:
        # 红色: 车辆 (1), 绿色: 行人 (2)
        red_mask = np.all(np.isclose(colors, [1, 0, 0], atol=0.1), axis=1)
        green_mask = np.all(np.isclose(colors, [0, 1, 0], atol=0.1), axis=1)
        labels[red_mask] = 1  # 车辆
        labels[green_mask] = 2  # 行人

    return points, colors, labels


def save_pointcloud(points, filename, colors=None, labels=None):
    """保存点云到文件"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if filename.endswith('.pcd'):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        elif labels is not None:
            # 根据标签设置颜色
            colors = np.zeros((len(points), 3))
            colors[labels == 1] = [1, 0, 0]  # 车辆-红色
            colors[labels == 2] = [0, 1, 0]  # 行人-绿色
            colors[labels == 0] = [0.5, 0.5, 0.5]  # 背景-灰色
            pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.io.write_point_cloud(filename, pcd)
    elif filename.endswith('.npy'):
        if colors is not None or labels is not None:
            save_data = {'points': points}
            if colors is not None:
                save_data['colors'] = colors
            if labels is not None:
                save_data['labels'] = labels
            np.save(filename, save_data)
        else:
            np.save(filename, points)


def extract_pedestrian_vehicle_points(points, labels):
    """
    提取行人和车辆点（兼容新旧标签系统）
    新标签: 1-车辆, 2-行人, 0-背景
    旧标签: 1,10,51-车辆, 255-行人
    """
    # 新标签系统
    if np.max(labels) <= 2:  # 只有0,1,2
        mask = (labels == 1) | (labels == 2)
    else:
        # 旧标签系统
        vehicle_labels = [1, 10, 51]
        pedestrian_labels = [255]
        mask = np.isin(labels, vehicle_labels) | np.isin(labels, pedestrian_labels)

    return points[mask], labels[mask]


def calculate_point_curvature(points, k=30):
    """计算点云曲率特征（保持不变）"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    pcd.orient_normals_consistent_tangent_plane(k)

    kdtree = o3d.geometry.KDTreeFlann(pcd)
    curvatures = []

    for i in range(len(points)):
        [k, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], k)
        neighbors = points[idx, :]

        cov = np.cov(neighbors.T)
        eigenvalues = np.linalg.eigvals(cov)
        eigenvalues.sort()

        if np.sum(eigenvalues) > 0:
            curvature = eigenvalues[0] / np.sum(eigenvalues)
        else:
            curvature = 0
        curvatures.append(curvature)

    return np.array(curvatures)