# scripts/step1_extract_objects_adaptive.py
import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def load_kitti_data(bin_path, label_path):
    """加载KITTI点云和标签数据"""
    points_data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    points = points_data[:, :3]

    if os.path.exists(label_path):
        labels = np.fromfile(label_path, dtype=np.uint32)
        semantic_labels = labels & 0xFFFF  # 取低16位作为语义标签
        return points, semantic_labels
    else:
        return points, np.zeros(len(points), dtype=np.uint32)


def analyze_labels_for_mapping(points, labels, file_name):
    """分析标签数据，自动发现车辆和行人标签"""
    unique_labels = np.unique(labels)

    print(f"\n分析文件: {file_name}")
    print(f"唯一标签: {unique_labels}")

    # 常见的SemanticKITTI移动物体标签
    possible_vehicle_labels = [1, 10, 11, 13, 18]  # car, truck, other-vehicle等
    possible_pedestrian_labels = [30, 31, 32, 252]  # person, rider等

    # 统计每个可能标签的数量
    label_counts = {}
    for label in unique_labels:
        count = np.sum(labels == label)
        label_counts[label] = count
        if count > 0:
            print(f"  标签 {label}: {count} 点")

    # 尝试识别车辆和行人标签
    vehicle_candidates = []
    pedestrian_candidates = []

    for label in unique_labels:
        count = label_counts.get(label, 0)
        if count > 0:
            # 基于经验判断哪些标签可能是车辆或行人
            if label in [1, 10, 11, 13, 18, 252, 253]:
                vehicle_candidates.append((label, count))
            elif label in [30, 31, 32, 254, 255]:
                pedestrian_candidates.append((label, count))
            elif 60 <= label <= 99:  # 可能是静态物体
                pass
            else:
                # 未知标签，也可能是移动物体
                if count < 1000:  # 排除大的静态物体
                    vehicle_candidates.append((label, count))

    return vehicle_candidates, pedestrian_candidates, label_counts


def extract_by_dynamic_clustering(points, labels, background_radius=25.0):
    """
    通过动态聚类方法提取移动物体
    不依赖预定义的标签映射
    """
    # 首先识别可能的移动物体标签
    unique_labels = np.unique(labels)

    # 移动物体通常具有以下特征：
    # 1. 点数相对较少
    # 2. 形成紧凑的聚类
    # 3. 在地面之上

    # 计算每个标签的点数
    label_counts = {}
    for label in unique_labels:
        count = np.sum(labels == label)
        label_counts[label] = count

    # 过滤可能的移动物体标签（排除大的静态物体和地面）
    potential_moving_labels = []
    for label, count in label_counts.items():
        if 10 <= count <= 5000:  # 移动物体通常在这个范围内
            if label not in [40, 44, 48, 49, 60, 72]:  # 排除常见的地面和植被标签
                potential_moving_labels.append(label)

    if not potential_moving_labels:
        return np.array([]), np.array([]), 0, 0

    # 创建移动物体掩码
    moving_mask = np.isin(labels, potential_moving_labels)

    if not np.any(moving_mask):
        return np.array([]), np.array([]), 0, 0

    # 获取移动物体点
    moving_points = points[moving_mask]
    moving_labels = labels[moving_mask]

    # 进一步通过空间聚类筛选
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(moving_points)

    # 使用DBSCAN聚类
    labels_cluster = np.array(pcd.cluster_dbscan(eps=2.0, min_points=10))

    if len(labels_cluster) == 0:
        return np.array([]), np.array([]), 0, 0

    # 找到最大的聚类（最可能是车辆）
    unique_clusters, cluster_counts = np.unique(labels_cluster[labels_cluster >= 0], return_counts=True)

    if len(unique_clusters) == 0:
        return np.array([]), np.array([]), 0, 0

    # 选择点数最多的聚类
    main_cluster = unique_clusters[np.argmax(cluster_counts)]
    main_cluster_mask = (labels_cluster == main_cluster)

    # 获取主要聚类的点
    cluster_points = moving_points[main_cluster_mask]
    cluster_labels = moving_labels[main_cluster_mask]

    # 计算聚类中心
    center = np.mean(cluster_points, axis=0)

    # 提取周围背景
    background_mask = ~moving_mask
    for i in range(3):
        background_mask &= (points[:, i] >= center[i] - background_radius)
        background_mask &= (points[:, i] <= center[i] + background_radius)

    # 合并物体点和背景点
    all_points = np.vstack([cluster_points, points[background_mask]])
    all_labels = np.hstack([cluster_labels, labels[background_mask]])

    # 创建颜色
    colors = create_adaptive_colors(all_labels, cluster_labels)

    return all_points, colors, len(cluster_points), 0


def create_adaptive_colors(all_labels, object_labels):
    """创建自适应颜色"""
    colors = np.zeros((len(all_labels), 3))

    # 对象点用红色
    object_indices = np.isin(all_labels, np.unique(object_labels))
    colors[object_indices] = [1, 0, 0]  # 红色

    # 背景点用灰色
    background_indices = ~object_indices
    colors[background_indices] = [0.3, 0.3, 0.3]  # 灰色

    return colors


def manual_label_mapping(points, labels, vehicle_labels, pedestrian_labels):
    """手动指定标签映射进行提取"""
    vehicle_mask = np.isin(labels, vehicle_labels)
    pedestrian_mask = np.isin(labels, pedestrian_labels)
    object_mask = vehicle_mask | pedestrian_mask

    if not np.any(object_mask):
        return np.array([]), np.array([]), 0, 0

    object_points = points[object_mask]
    object_labels = labels[object_mask]

    # 计算对象中心
    center = np.mean(object_points, axis=0)

    # 提取周围背景
    background_radius = 20.0
    background_mask = ~object_mask
    for i in range(3):
        background_mask &= (points[:, i] >= center[i] - background_radius)
        background_mask &= (points[:, i] <= center[i] + background_radius)

    # 合并点云
    all_points = np.vstack([object_points, points[background_mask]])
    all_labels = np.hstack([object_labels, labels[background_mask]])

    # 创建颜色
    colors = np.zeros((len(all_labels), 3))

    # 车辆用红色，行人用绿色，背景用灰色
    for i, label in enumerate(all_labels):
        if label in vehicle_labels:
            colors[i] = [1, 0, 0]  # 红色
        elif label in pedestrian_labels:
            colors[i] = [0, 1, 0]  # 绿色
        else:
            colors[i] = [0.3, 0.3, 0.3]  # 灰色

    return all_points, colors, np.sum(vehicle_mask), np.sum(pedestrian_mask)


def save_colored_pointcloud(points, colors, filename):
    """保存带颜色的点云"""
    if len(points) == 0:
        return False

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    try:
        o3d.io.write_point_cloud(filename, pcd)
        return True
    except Exception as e:
        print(f"保存点云失败 {filename}: {e}")
        return False


def interactive_label_discovery(sequence_id="00", num_files=5):
    """交互式标签发现"""
    raw_data_dir = project_root / "data" / "raw_dataset"
    seq_path = raw_data_dir / "dataset" / "sequences" / sequence_id / "velodyne"
    label_path = raw_data_dir / "dataset" / "sequences" / sequence_id / "labels"

    bin_files = sorted(seq_path.glob("*.bin"))[:num_files]

    print(f"=== 交互式标签发现 - 序列 {sequence_id} ===")

    all_vehicle_candidates = []
    all_pedestrian_candidates = []

    for bin_file in bin_files:
        label_file = label_path / bin_file.name.replace('.bin', '.label')

        if not label_file.exists():
            continue

        points, labels = load_kitti_data(str(bin_file), str(label_file))
        vehicle_candidates, pedestrian_candidates, label_counts = analyze_labels_for_mapping(
            points, labels, bin_file.name
        )

        all_vehicle_candidates.extend(vehicle_candidates)
        all_pedestrian_candidates.extend(pedestrian_candidates)

    # 统计最常见的候选标签
    from collections import Counter
    vehicle_counter = Counter([c[0] for c in all_vehicle_candidates])
    pedestrian_counter = Counter([c[0] for c in all_pedestrian_candidates])

    print(f"\n=== 标签统计 ===")
    print("车辆候选标签:", vehicle_counter.most_common(10))
    print("行人候选标签:", pedestrian_counter.most_common(10))

    return vehicle_counter, pedestrian_counter


def test_extraction_methods(sequence_id="00", file_index=0):
    """测试不同的提取方法"""
    raw_data_dir = project_root / "data" / "raw_dataset"
    seq_path = raw_data_dir / "dataset" / "sequences" / sequence_id / "velodyne"
    label_path = raw_data_dir / "dataset" / "sequences" / sequence_id / "labels"

    bin_files = sorted(seq_path.glob("*.bin"))

    if file_index >= len(bin_files):
        print(f"文件索引 {file_index} 超出范围")
        return

    bin_file = bin_files[file_index]
    label_file = label_path / bin_file.name.replace('.bin', '.label')

    print(f"测试文件: {bin_file.name}")

    # 加载数据
    points, labels = load_kitti_data(str(bin_file), str(label_file))

    print(f"总点数: {len(points)}")
    print(f"唯一标签: {np.unique(labels)}")

    # 方法1: 动态聚类
    print("\n方法1: 动态聚类提取")
    points1, colors1, vehicle_count1, pedestrian_count1 = extract_by_dynamic_clustering(points, labels)

    if len(points1) > 0:
        print(f"提取点数: {len(points1)}")
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points1)
        pcd1.colors = o3d.utility.Vector3dVector(colors1)

        print("显示动态聚类结果...")
        o3d.visualization.draw_geometries([pcd1],
                                          window_name="动态聚类提取",
                                          width=1000, height=800)
    else:
        print("动态聚类未提取到对象")

    # 方法2: 手动标签映射
    print("\n方法2: 手动标签映射")
    # 基于常见标签尝试
    common_vehicle_labels = [1, 10, 252, 253]
    common_pedestrian_labels = [30, 31, 32, 254, 255]

    points2, colors2, vehicle_count2, pedestrian_count2 = manual_label_mapping(
        points, labels, common_vehicle_labels, common_pedestrian_labels
    )

    if len(points2) > 0:
        print(f"提取点数: {len(points2)}")
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        pcd2.colors = o3d.utility.Vector3dVector(colors2)

        print("显示手动映射结果...")
        o3d.visualization.draw_geometries([pcd2],
                                          window_name="手动映射提取",
                                          width=1000, height=800)
    else:
        print("手动映射未提取到对象")

    # 保存测试结果
    test_dir = project_root / "data" / "test_output"
    test_dir.mkdir(exist_ok=True)

    if len(points1) > 0:
        test_file1 = test_dir / f"dynamic_{sequence_id}_{file_index:06d}.pcd"
        o3d.io.write_point_cloud(str(test_file1), pcd1)
        print(f"动态聚类结果保存: {test_file1}")

    if len(points2) > 0:
        test_file2 = test_dir / f"manual_{sequence_id}_{file_index:06d}.pcd"
        o3d.io.write_point_cloud(str(test_file2), pcd2)
        print(f"手动映射结果保存: {test_file2}")


def visualize_original_with_objects(sequence_id="00", file_index=0):
    """可视化原始点云并标记可能的移动物体"""
    raw_data_dir = project_root / "data" / "raw_dataset"
    seq_path = raw_data_dir / "dataset" / "sequences" / sequence_id / "velodyne"
    label_path = raw_data_dir / "dataset" / "sequences" / sequence_id / "labels"

    bin_files = sorted(seq_path.glob("*.bin"))

    if file_index >= len(bin_files):
        print(f"文件索引 {file_index} 超出范围")
        return

    bin_file = bin_files[file_index]
    label_file = label_path / bin_file.name.replace('.bin', '.label')

    print(f"可视化文件: {bin_file.name}")

    # 加载数据
    points, labels = load_kitti_data(str(bin_file), str(label_file))

    # 创建颜色映射
    colors = np.zeros((len(points), 3))

    # 将可能的移动物体标记为红色，其他为灰色
    possible_moving_labels = [1, 10, 11, 13, 18, 30, 31, 32, 252, 253, 254, 255]

    for i, label in enumerate(labels):
        if label in possible_moving_labels:
            colors[i] = [1, 0, 0]  # 红色标记可能的移动物体
        else:
            colors[i] = [0.5, 0.5, 0.5]  # 灰色标记其他物体

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("显示原始点云（红色点为可能的移动物体）...")
    o3d.visualization.draw_geometries([pcd],
                                      window_name="原始点云 - 移动物体标记",
                                      width=1000, height=800)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='自适应点云对象提取')
    parser.add_argument('--discover', type=str, help='发现标签映射 (例如: 00)')
    parser.add_argument('--test', type=str, help='测试提取方法 (例如: 00)')
    parser.add_argument('--file', type=int, default=0, help='测试文件索引')
    parser.add_argument('--visualize', type=str, help='可视化原始点云 (例如: 00)')

    args = parser.parse_args()

    if args.discover:
        interactive_label_discovery(args.discover)
    elif args.test:
        test_extraction_methods(args.test, args.file)
    elif args.visualize:
        visualize_original_with_objects(args.visualize, args.file)
    else:
        print("请指定操作: --discover <序列ID>, --test <序列ID>, 或 --visualize <序列ID>")