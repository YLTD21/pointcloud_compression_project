import numpy as np
import open3d as o3d


def visualize_raw_pointcloud_with_labels(bin_path, label_path):
    # 加载点云（x,y,z,intensity）
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # 加载标签并提取语义ID
    full_labels = np.fromfile(label_path, dtype=np.int32)
    semantic_ids = full_labels >> 16  # 纯语义ID
    # 截断到相同长度
    min_len = min(len(points), len(semantic_ids))
    points = points[:min_len]
    semantic_ids = semantic_ids[:min_len]

    # 为不同ID分配随机颜色（便于区分）
    unique_ids = np.unique(semantic_ids)
    color_map = {id: np.random.rand(3) for id in unique_ids}
    colors = np.array([color_map[id] for id in semantic_ids])

    # 可视化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name=f"原始点云（ID: {unique_ids}）")


# 替换为你的帧文件路径（例如序列00的第0帧）
bin_path = "/media/yml/share/pointcloud_compression_project/data/raw_dataset/dataset/sequences/00/velodyne/000000.bin"
label_path = "/media/yml/share/pointcloud_compression_project/data/raw_dataset/dataset/sequences/00/labels/000000.label"
visualize_raw_pointcloud_with_labels(bin_path, label_path)
