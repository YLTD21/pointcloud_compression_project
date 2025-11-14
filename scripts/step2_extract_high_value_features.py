import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from utils import calculate_point_curvature, save_pointcloud
import os
import sys

# 添加当前目录到Python路径，确保模块可以相互导入
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def extract_high_value_features(points, labels, curvature_threshold=0.01, min_cluster_size=10):
    """
    提取高价值特征点
    基于曲率、密度和边缘特征
    """
    if len(points) < 10:
        return points, labels

    # 1. 计算曲率特征
    curvatures = calculate_point_curvature(points)

    # 2. 密度聚类去除噪声
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(points)
    labels_dbscan = clustering.labels_

    # 3. 边缘检测 (基于法线变化)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

    normals = np.asarray(pcd.normals)
    edge_scores = np.zeros(len(points))

    for i in range(len(points)):
        # 查找邻近点
        distances = np.linalg.norm(points - points[i], axis=1)
        neighbor_indices = np.where(distances < 1.0)[0]

        if len(neighbor_indices) > 1:
            # 计算法线变化
            normal_diffs = np.abs(np.dot(normals[neighbor_indices], normals[i]))
            edge_scores[i] = 1 - np.mean(normal_diffs)

    # 4. 综合特征评分
    curvature_normalized = (curvatures - np.min(curvatures)) / (np.max(curvatures) - np.min(curvatures) + 1e-8)
    edge_normalized = (edge_scores - np.min(edge_scores)) / (np.max(edge_scores) - np.min(edge_scores) + 1e-8)

    # 综合评分 = 曲率 + 边缘特征
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
            # 如果聚类中没有高价值点，添加得分最高的点
            cluster_high_value = high_value_mask[cluster_points]
            if not np.any(cluster_high_value):
                cluster_scores = combined_scores[cluster_points]
                best_point_idx = cluster_points[np.argmax(cluster_scores)]
                final_mask[best_point_idx] = True

    return points[final_mask], labels[final_mask] if labels is not None else None


def process_high_value_extraction():
    """批量处理高价值特征点提取"""

    processed_dir = "../data/processed_dataset"
    high_value_dir = "../data/high_value_dataset"

    os.makedirs(high_value_dir, exist_ok=True)

    # 遍历所有序列
    sequences = [d for d in os.listdir(processed_dir)
                 if os.path.isdir(os.path.join(processed_dir, d))]

    for seq in sequences:
        seq_input_dir = os.path.join(processed_dir, seq)
        seq_output_dir = os.path.join(high_value_dir, seq)
        os.makedirs(seq_output_dir, exist_ok=True)

        # 处理每个点云文件
        npy_files = [f for f in os.listdir(seq_input_dir) if f.endswith('.npy')]

        for npy_file in tqdm(npy_files, desc=f"Extracting high-value features {seq}"):
            try:
                data = np.load(os.path.join(seq_input_dir, npy_file), allow_pickle=True).item()
                points = data['points']
                labels = data['labels'] if 'labels' in data else None

                # 提取高价值特征点
                high_value_points, high_value_labels = extract_high_value_features(points, labels)

                if len(high_value_points) > 0:
                    # 保存高价值点云
                    output_file = os.path.join(seq_output_dir, npy_file.replace('.npy', '_high_value.pcd'))
                    save_pointcloud(high_value_points, output_file, high_value_labels)

                    # 保存npy格式
                    npy_output = os.path.join(seq_output_dir, npy_file.replace('.npy', '_high_value.npy'))
                    np.save(npy_output, {
                        'points': high_value_points,
                        'labels': high_value_labels,
                        'original_count': len(points),
                        'compressed_count': len(high_value_points),
                        'compression_ratio': len(high_value_points) / len(points)
                    })

            except Exception as e:
                print(f"Error processing {npy_file}: {e}")
                continue


if __name__ == "__main__":
    process_high_value_extraction()