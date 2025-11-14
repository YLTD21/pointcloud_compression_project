# scripts/step1_extract_objects_corrected.py
import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def load_kitti_data_corrected(bin_path, label_path):
    """正确加载KITTI点云和标签数据"""
    # 加载点云
    points_data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    points = points_data[:, :3]

    # 加载标签
    if os.path.exists(label_path):
        labels = np.fromfile(label_path, dtype=np.uint32)
        # SemanticKITTI标签处理：取低16位作为语义标签
        semantic_labels = labels & 0xFFFF
        return points, semantic_labels
    else:
        print(f"警告：标签文件不存在 {label_path}")
        return points, np.zeros(len(points), dtype=np.uint32)


def extract_objects_with_background_corrected(points, labels, background_radius=20.0):
    """
    使用正确的SemanticKITTI标签提取对象和背景
    根据您的调试输出，标签包括：1(car), 10, 40, 44, 48, 50, 51, 52, 60, 70, 71, 72, 80, 81, 99
    """
    # 定义对象标签 - 根据您的调试输出调整
    vehicle_labels = [1]  # car

    # 尝试查找行人标签（可能在您的数据中不存在）
    pedestrian_labels = [6, 7, 8]  # person, bicyclist, motorcyclist

    # 检查哪些标签实际存在于数据中
    unique_labels = np.unique(labels)
    print(f"  发现标签: {unique_labels}")

    vehicle_mask = np.isin(labels, vehicle_labels)
    pedestrian_mask = np.isin(labels, pedestrian_labels)
    object_mask = vehicle_mask | pedestrian_mask

    # 统计对象数量
    vehicle_count = np.sum(vehicle_mask)
    pedestrian_count = np.sum(pedestrian_mask)

    print(f"  车辆点数: {vehicle_count}, 行人数: {pedestrian_count}")

    # 如果没有检测到任何对象，返回空数组
    if not np.any(object_mask):
        return np.array([]), np.array([]), vehicle_count, pedestrian_count

    # 获取所有对象点的坐标
    object_points = points[object_mask]
    object_labels = labels[object_mask]

    # 计算对象点的边界框
    if len(object_points) > 0:
        min_bound = np.min(object_points, axis=0)
        max_bound = np.max(object_points, axis=0)
        center = (min_bound + max_bound) / 2

        # 扩展边界框以包含背景
        expanded_min = center - background_radius
        expanded_max = center + background_radius

        # 创建背景掩码（在扩展边界框内的非对象点）
        background_mask = ~object_mask
        for i in range(3):
            background_mask &= (points[:, i] >= expanded_min[i])
            background_mask &= (points[:, i] <= expanded_max[i])

        # 合并对象点和背景点
        all_points = np.vstack([points[object_mask], points[background_mask]])
        all_labels = np.hstack([object_labels, labels[background_mask]])

        # 为点云创建颜色
        colors = create_colors_based_on_labels_corrected(all_labels)

        return all_points, colors, vehicle_count, pedestrian_count
    else:
        return np.array([]), np.array([]), vehicle_count, pedestrian_count


def create_colors_based_on_labels_corrected(labels):
    """
    根据正确的标签创建颜色
    车辆(1): 红色, 可能的行人(6,7,8): 绿色, 背景: 灰色
    """
    colors = np.zeros((len(labels), 3))

    vehicle_labels = [1]
    pedestrian_labels = [6, 7, 8]

    for i, label in enumerate(labels):
        if label in vehicle_labels:  # 车辆 - 红色
            colors[i] = [1, 0, 0]
        elif label in pedestrian_labels:  # 行人 - 绿色
            colors[i] = [0, 1, 0]
        else:  # 背景 - 灰色
            colors[i] = [0.3, 0.3, 0.3]  # 稍暗的灰色，提高对比度

    return colors


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


def process_semantic_kitti_data_corrected():
    """使用正确标签处理SemanticKITTI数据集"""

    raw_data_dir = project_root / "data" / "raw_dataset"
    processed_dir = project_root / "data" / "processed_dataset_corrected"

    processed_dir.mkdir(parents=True, exist_ok=True)

    sequences = [d for d in os.listdir(raw_data_dir / "dataset" / "sequences")
                 if os.path.isdir(raw_data_dir / "dataset" / "sequences" / d) and d.isdigit()]

    sequences.sort()

    total_frames_processed = 0
    total_frames_with_objects = 0
    total_vehicle_points = 0
    total_pedestrian_points = 0

    for seq in sequences:
        seq_path = raw_data_dir / "dataset" / "sequences" / seq / "velodyne"
        label_path = raw_data_dir / "dataset" / "sequences" / seq / "labels"

        if not seq_path.exists():
            print(f"序列 {seq} 的点云路径不存在: {seq_path}")
            continue

        seq_output_dir = processed_dir / f"seq_{seq}"
        seq_output_dir.mkdir(exist_ok=True)

        bin_files = [f for f in os.listdir(seq_path) if f.endswith('.bin')]
        bin_files.sort()

        seq_frames_with_objects = 0
        seq_vehicle_points = 0
        seq_pedestrian_points = 0

        print(f"\n处理序列 {seq}:")

        for bin_file in tqdm(bin_files, desc=f"序列 {seq}"):
            try:
                bin_file_path = seq_path / bin_file
                label_file = bin_file.replace('.bin', '.label')
                label_file_path = label_path / label_file

                # 使用正确的加载函数
                points, labels = load_kitti_data_corrected(str(bin_file_path), str(label_file_path))

                # 使用正确的提取函数
                extracted_points, colors, vehicle_count, pedestrian_count = extract_objects_with_background_corrected(
                    points, labels, background_radius=25.0  # 增加背景半径
                )

                seq_vehicle_points += vehicle_count
                seq_pedestrian_points += pedestrian_count

                if len(extracted_points) > 0:
                    output_file = seq_output_dir / bin_file.replace('.bin', '.pcd')
                    if save_colored_pointcloud(extracted_points, colors, str(output_file)):
                        seq_frames_with_objects += 1
                        total_frames_with_objects += 1

                    # 保存统计信息
                    npy_file = seq_output_dir / bin_file.replace('.bin', '.npy')
                    np.save(str(npy_file), {
                        'points': extracted_points,
                        'colors': colors,
                        'labels': labels,
                        'vehicle_count': vehicle_count,
                        'pedestrian_count': pedestrian_count,
                        'source_file': str(bin_file_path)
                    })

            except Exception as e:
                print(f"处理文件 {bin_file} 时出错: {e}")
                continue

        total_frames_processed += len(bin_files)
        total_vehicle_points += seq_vehicle_points
        total_pedestrian_points += seq_pedestrian_points

        print(f"序列 {seq}: 处理了 {len(bin_files)} 帧，其中 {seq_frames_with_objects} 帧包含对象")
        print(f"  车辆总点数: {seq_vehicle_points}, 行人总点数: {seq_pedestrian_points}")

    print(f"\n=== 处理完成 ===")
    print(f"总共处理了 {total_frames_processed} 帧")
    print(f"其中 {total_frames_with_objects} 帧包含对象")
    print(f"车辆总点数: {total_vehicle_points}")
    print(f"行人总点数: {total_pedestrian_points}")
    print(f"输出目录: {processed_dir}")


def test_single_file_corrected(sequence_id="00", file_index=0):
    """测试单个文件的处理"""
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
    points, labels = load_kitti_data_corrected(str(bin_file), str(label_file))

    # 统计信息
    unique_labels = np.unique(labels)
    print(f"唯一标签: {unique_labels}")

    vehicle_labels = [1]
    pedestrian_labels = [6, 7, 8]

    vehicle_count = np.sum(np.isin(labels, vehicle_labels))
    pedestrian_count = np.sum(np.isin(labels, pedestrian_labels))

    print(f"总点数: {len(points)}")
    print(f"车辆数: {vehicle_count}")
    print(f"行人数: {pedestrian_count}")

    # 提取对象
    extracted_points, colors, vehicle_count, pedestrian_count = extract_objects_with_background_corrected(
        points, labels, background_radius=25.0
    )

    if len(extracted_points) > 0:
        print(f"提取点数: {len(extracted_points)}")

        # 可视化
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(extracted_points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        print("显示点云...")
        o3d.visualization.draw_geometries([pcd],
                                          window_name=f"测试 - {bin_file.name}",
                                          width=1000,
                                          height=800)

        # 保存测试文件
        test_dir = project_root / "data" / "test_output"
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / f"test_{sequence_id}_{file_index:06d}.pcd"
        o3d.io.write_point_cloud(str(test_file), pcd)
        print(f"测试点云已保存: {test_file}")
    else:
        print("没有提取到对象")


def analyze_sequence_content(sequence_id="00", num_files=10):
    """分析序列内容，了解对象分布"""
    raw_data_dir = project_root / "data" / "raw_dataset"
    seq_path = raw_data_dir / "dataset" / "sequences" / sequence_id / "velodyne"
    label_path = raw_data_dir / "dataset" / "sequences" / sequence_id / "labels"

    bin_files = sorted(seq_path.glob("*.bin"))[:num_files]

    print(f"=== 序列 {sequence_id} 内容分析 ===")

    vehicle_frames = 0
    pedestrian_frames = 0
    total_vehicle_points = 0
    total_pedestrian_points = 0

    for i, bin_file in enumerate(bin_files):
        label_file = label_path / bin_file.name.replace('.bin', '.label')

        if not label_file.exists():
            continue

        points, labels = load_kitti_data_corrected(str(bin_file), str(label_file))

        vehicle_labels = [1]
        pedestrian_labels = [6, 7, 8]

        vehicle_count = np.sum(np.isin(labels, vehicle_labels))
        pedestrian_count = np.sum(np.isin(labels, pedestrian_labels))

        if vehicle_count > 0:
            vehicle_frames += 1
            total_vehicle_points += vehicle_count

        if pedestrian_count > 0:
            pedestrian_frames += 1
            total_pedestrian_points += pedestrian_count

        if i < 5:  # 只显示前5个文件的详细信息
            print(f"{bin_file.name}: 车辆={vehicle_count}, 行人={pedestrian_count}")

    print(f"\n统计信息:")
    print(f"  包含车辆的帧: {vehicle_frames}/{len(bin_files)}")
    print(f"  包含行人的帧: {pedestrian_frames}/{len(bin_files)}")
    print(f"  平均每帧车辆点数: {total_vehicle_points / max(1, vehicle_frames):.1f}")
    print(f"  平均每帧行人数: {total_pedestrian_points / max(1, pedestrian_frames):.1f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='正确标签版点云对象提取')
    parser.add_argument('--process', action='store_true', help='处理所有数据')
    parser.add_argument('--test', type=str, help='测试单个序列 (例如: 00)')
    parser.add_argument('--file', type=int, default=0, help='测试文件索引')
    parser.add_argument('--analyze', type=str, help='分析序列内容 (例如: 00)')

    args = parser.parse_args()

    if args.analyze:
        analyze_sequence_content(args.analyze)
    elif args.test:
        test_single_file_corrected(args.test, args.file)
    elif args.process:
        print("开始正确标签版点云对象提取...")
        process_semantic_kitti_data_corrected()
    else:
        print("请指定操作: --process, --test <序列ID>, 或 --analyze <序列ID>")