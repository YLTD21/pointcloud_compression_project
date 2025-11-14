# scripts/step2_extract_high_value_features.py
import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils import calculate_point_curvature, save_pointcloud, load_extracted_pointcloud


def extract_high_value_features(points, colors=None, labels=None, curvature_threshold=0.01, min_cluster_size=10):
    """
    提取高价值特征点 - 适配新数据格式
    """
    if len(points) < 10:
        return points, colors, labels

    # 1. 计算曲率特征
    curvatures = calculate_point_curvature(points)

    # 2. 密度聚类去除噪声
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(points)
    labels_dbscan = clustering.labels_

    # 3. 边缘检测 (基于法线变化)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

    normals = np.asarray(pcd.normals)
    edge_scores = np.zeros(len(points))

    for i in range(len(points)):
        distances = np.linalg.norm(points - points[i], axis=1)
        neighbor_indices = np.where(distances < 1.0)[0]

        if len(neighbor_indices) > 1:
            normal_diffs = np.abs(np.dot(normals[neighbor_indices], normals[i]))
            edge_scores[i] = 1 - np.mean(normal_diffs)

    # 4. 综合特征评分
    curvature_normalized = (curvatures - np.min(curvatures)) / (np.max(curvatures) - np.min(curvatures) + 1e-8)
    edge_normalized = (edge_scores - np.min(edge_scores)) / (np.max(edge_scores) - np.min(edge_scores) + 1e-8)

    combined_scores = curvature_normalized + edge_normalized

    # 5. 选择高价值点
    threshold = np.percentile(combined_scores, 80)  # 选择前20%的点
    high_value_mask = combined_scores > threshold

    # 6. 确保每个聚类都有代表点
    cluster_labels = labels_dbscan
    unique_clusters = np.unique(cluster_labels)

    final_mask = high_value_mask.copy()

    for cluster_id in unique_clusters:
        if cluster_id == -1:  # 噪声点
            continue

        cluster_points = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_points) >= min_cluster_size:
            cluster_high_value = high_value_mask[cluster_points]
            if not np.any(cluster_high_value):
                cluster_scores = combined_scores[cluster_points]
                best_point_idx = cluster_points[np.argmax(cluster_scores)]
                final_mask[best_point_idx] = True

    # 返回过滤后的数据
    filtered_points = points[final_mask]
    filtered_colors = colors[final_mask] if colors is not None else None
    filtered_labels = labels[final_mask] if labels is not None else None

    return filtered_points, filtered_colors, filtered_labels


def process_high_value_extraction():
    """批量处理高价值特征点提取 - 适配新数据目录"""

    processed_dir = project_root / "data" / "processed_dataset_final"  # 改为新的数据目录
    high_value_dir = project_root / "data" / "high_value_dataset"

    os.makedirs(high_value_dir, exist_ok=True)

    # 遍历所有序列
    sequences = [d for d in os.listdir(processed_dir)
                 if os.path.isdir(os.path.join(processed_dir, d)) and d.startswith('seq_')]

    for seq in sequences:
        seq_input_dir = os.path.join(processed_dir, seq)
        seq_output_dir = os.path.join(high_value_dir, seq)
        os.makedirs(seq_output_dir, exist_ok=True)

        # 处理每个点云文件
        pcd_files = [f for f in os.listdir(seq_input_dir) if f.endswith('.pcd')]

        for pcd_file in tqdm(pcd_files, desc=f"Extracting high-value features {seq}"):
            try:
                # 加载提取的点云数据
                file_path = os.path.join(seq_input_dir, pcd_file)
                points, colors, labels = load_extracted_pointcloud(file_path)

                # 提取高价值特征点
                high_value_points, high_value_colors, high_value_labels = extract_high_value_features(
                    points, colors, labels
                )

                if len(high_value_points) > 0:
                    # 保存高价值点云
                    output_file = os.path.join(seq_output_dir, pcd_file.replace('.pcd', '_high_value.pcd'))
                    save_pointcloud(high_value_points, output_file, high_value_colors, high_value_labels)

                    # 保存npy格式
                    npy_output = os.path.join(seq_output_dir, pcd_file.replace('.pcd', '_high_value.npy'))
                    np.save(npy_output, {
                        'points': high_value_points,
                        'colors': high_value_colors,
                        'labels': high_value_labels,
                        'original_count': len(points),
                        'compressed_count': len(high_value_points),
                        'compression_ratio': len(high_value_points) / len(points)
                    })

            except Exception as e:
                print(f"Error processing {pcd_file}: {e}")
                continue


if __name__ == "__main__":
    process_high_value_extraction()